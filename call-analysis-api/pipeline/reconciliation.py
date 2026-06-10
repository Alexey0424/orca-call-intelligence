"""Combine ASR word timestamps with diarization turns into utterances.

The flow is:

1. For each ASR word, pick the diarization turn with the largest temporal
   overlap and tag the word with that speaker. Words with zero overlap
   against every turn (e.g. Whisper zero-width words, or boundary words
   that fall in a diarization gap) fall back to the turn containing
   their midpoint, or the nearest turn by distance — instead of being
   tagged ``UNKNOWN`` and split out as their own utterance.
2. (OFF by default — MIN_INTRA_TURN_RUN_WORDS=1.) Collapse short
   same-speaker runs that disagree with both neighbours, IF the word just
   before the run did not end a sentence (``.?!``). Absorbs ~100-300 ms
   word-timestamp jitter, but the 2026-06-02 A/B showed it also absorbs
   REAL short client turns (Alicia opening client recall 0.50 → 0.00),
   so the production default is off; the LLM opening relabel covers the
   role errors instead.
3. Walk the tagged words and group consecutive words into utterances.
   An utterance ends when the next word switches speaker, the gap
   exceeds ``MAX_INTRA_SPEAKER_GAP_SEC``, or a sentence-ending
   punctuation is followed by at least ``SENTENCE_END_GAP_SEC``.

(The word silence-split pre-pass is likewise OFF by default —
WORD_SILENCE_SPLIT_ENABLED=0; marginal benefit, superseded by the relabel.)

The output is a list of utterance dicts shaped like the rows we'll
persist into the ``utterances`` table.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import numpy as np

import config


_SENTENCE_END = re.compile(config.SENTENCE_END_PATTERN)


def _silence_split_word(word: Dict[str, Any],
                        y: np.ndarray, sr: int) -> Dict[str, Any]:
    """If a Whisper word is suspiciously long AND contains an internal
    silence gap, cap its END at the first such gap.

    Motivation: Whisper occasionally emits a single word that spans far
    beyond its actual phoneme energy. Concrete failure mode: on the
    reference call the opening "Hello." spans 10.99-11.95 s (960 ms),
    but the actual client phoneme energy ends around 11.22 s and the
    rest of the window contains the closer's pre-speech breath at
    11.80-11.95 s. The default overlap math then attributes "Hello." to
    the closer by a 22 ms margin against the closer's phantom turn.

    Capping the word END at the first ``WORD_SILENCE_GAP_SEC`` (200 ms)
    of sub-threshold RMS pulls the word boundary back to its real
    phoneme region. Only applied when the word is longer than
    ``WORD_SILENCE_MIN_SUSPICIOUS_DUR_SEC`` (700 ms) — typical Whisper
    words are tight and unaffected.

    Returns the word dict unchanged if any guard trips.
    """
    if not getattr(config, "WORD_SILENCE_SPLIT_ENABLED", True):
        return word

    dur = word["end"] - word["start"]
    min_dur = float(getattr(config, "WORD_SILENCE_MIN_SUSPICIOUS_DUR_SEC", 0.7))
    if dur < min_dur:
        return word

    frame_ms = float(getattr(config, "WORD_SILENCE_FRAME_MS", 10))
    gap_min_sec = float(getattr(config, "WORD_SILENCE_GAP_SEC", 0.20))
    peak_threshold = float(getattr(config, "WORD_SILENCE_PEAK_THRESHOLD", 0.15))

    s0 = int(word["start"] * sr)
    s1 = int(word["end"] * sr)
    if s0 < 0 or s1 > y.size or s1 <= s0:
        return word

    clip = y[s0:s1]
    frame_samples = int(frame_ms / 1000.0 * sr)
    if frame_samples <= 0:
        return word
    n_frames = clip.size // frame_samples
    if n_frames < 3:
        return word

    rms = np.empty(n_frames, dtype=np.float32)
    for i in range(n_frames):
        chunk = clip[i * frame_samples:(i + 1) * frame_samples]
        rms[i] = float(np.sqrt(np.mean(chunk * chunk) + 1e-12))

    peak = float(rms.max())
    if peak <= 1e-6:
        return word

    high = rms > peak * peak_threshold
    if not high.any():
        return word

    first_high = int(np.argmax(high))
    gap_min_frames = max(1, int(gap_min_sec * 1000.0 / frame_ms))
    end_frame = n_frames
    low_run = 0
    last_high_before_gap = first_high
    for i in range(first_high, n_frames):
        if high[i]:
            last_high_before_gap = i
            low_run = 0
        else:
            low_run += 1
            if low_run >= gap_min_frames:
                end_frame = last_high_before_gap + 1
                break

    if end_frame >= n_frames:
        return word

    tight_end = word["start"] + end_frame * frame_ms / 1000.0
    # Sanity guard: keep at least 50 ms of word
    if tight_end - word["start"] < 0.05:
        return word

    out = dict(word)
    out["end"] = tight_end
    return out


def _silence_split_words(words: List[Dict[str, Any]],
                         y: Optional[np.ndarray],
                         sr: Optional[int]) -> List[Dict[str, Any]]:
    if y is None or sr is None or not getattr(config,
                                                "WORD_SILENCE_SPLIT_ENABLED",
                                                True):
        return words
    return [_silence_split_word(w, y, sr) for w in words]


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _nearest_turn_speaker(w: Dict[str, Any],
                          turns: List[Dict[str, Any]]) -> str:
    """Fallback for words with zero overlap against every turn.

    Pick the turn whose interval contains the word's midpoint; if none
    contains it (the word sits in a diarization gap), pick the turn with
    the smallest distance from the midpoint to either boundary.
    """
    mid = 0.5 * (float(w["start"]) + float(w["end"]))
    best, dmin = "UNKNOWN", float("inf")
    for t in turns:
        if t["start"] <= mid <= t["end"]:
            return t["speaker"]
        d = min(abs(mid - t["start"]), abs(mid - t["end"]))
        if d < dmin:
            dmin, best = d, t["speaker"]
    return best


def _tag_speakers(words: List[Dict[str, Any]],
                  turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not turns:
        for w in words:
            w["speaker"] = "SPEAKER_0"
        return words

    for w in words:
        best_speaker, best_overlap = "UNKNOWN", 0.0
        for t in turns:
            ov = _overlap(w["start"], w["end"], t["start"], t["end"])
            if ov > best_overlap:
                best_overlap = ov
                best_speaker = t["speaker"]
        if best_speaker == "UNKNOWN":
            # Either a zero-width Whisper word (start == end produces zero
            # overlap with every turn) or a boundary word that fell in a
            # diarization gap. Fall back to midpoint-containing or
            # nearest-boundary turn instead of stranding it as UNKNOWN.
            best_speaker = _nearest_turn_speaker(w, turns)
        w["speaker"] = best_speaker
    return words


def _collapse_short_runs(words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Smooth out short same-speaker runs that look like timestamp jitter.

    A "run" is a maximal contiguous span of words sharing the same
    speaker label. A short run (``< MIN_INTRA_TURN_RUN_WORDS``) is
    relabelled to its neighbours' speaker when:
      * both immediate neighbours agree on a speaker, AND
      * the word *just before* the short run does NOT end with
        sentence-ending punctuation (so we never absorb a real turn
        change that the speaker just finished a sentence into).

    The combined effect: a single mid-monologue word that leaked into
    the other speaker's turn (because Whisper's word timestamp is
    ~100-300 ms off vs the frame-level diarizer) gets reabsorbed,
    instead of artificially splitting the monologue into three
    utterances.
    """
    min_run = int(getattr(config, "MIN_INTRA_TURN_RUN_WORDS", 1))
    if min_run <= 1 or len(words) < 3:
        return words

    speakers = [w["speaker"] for w in words]
    n = len(speakers)

    def _rebuild_runs() -> List[tuple]:
        runs: List[tuple] = []
        i = 0
        while i < n:
            j = i
            while j < n and speakers[j] == speakers[i]:
                j += 1
            runs.append((i, j, speakers[i]))
            i = j
        return runs

    changed = True
    while changed:
        changed = False
        runs = _rebuild_runs()
        for ri in range(1, len(runs) - 1):
            lo, hi, _sp = runs[ri]
            if (hi - lo) >= min_run:
                continue
            prev_sp = runs[ri - 1][2]
            next_sp = runs[ri + 1][2]
            if prev_sp != next_sp:
                continue
            prev_word_text = words[lo - 1]["text"].strip()
            if _SENTENCE_END.search(prev_word_text):
                continue
            for idx in range(lo, hi):
                speakers[idx] = prev_sp
            changed = True
            break  # rebuild runs and restart

    for w, sp in zip(words, speakers):
        w["speaker"] = sp
    return words


def _utt_record(buf_text: List[str], buf_conf: List[float],
                s0: float, e0: float, spk: str) -> Dict[str, Any]:
    """Build one utterance dict from a buffer of words.

    ``asr_confidence`` is the simple mean of per-word confidences in the
    utterance, when ASR surfaced them; ``None`` if no words in the
    buffer carried a confidence (e.g. Whisper build without word-confidence
    support).
    """
    if buf_conf:
        mean_conf = round(sum(buf_conf) / len(buf_conf), 4)
    else:
        mean_conf = None
    return {
        "start": round(s0, 3), "end": round(e0, 3),
        "text": " ".join(buf_text), "speaker": spk, "language": "en",
        "asr_confidence": mean_conf,
    }


def _group_into_utterances(words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not words:
        return []

    utterances: List[Dict[str, Any]] = []
    buf_text: List[str] = [words[0]["text"]]
    buf_conf: List[float] = []
    if words[0].get("confidence") is not None:
        buf_conf.append(float(words[0]["confidence"]))
    s0  = words[0]["start"]
    e0  = words[0]["end"]
    spk = words[0]["speaker"]

    for w in words[1:]:
        gap = w["start"] - e0
        boundary = (
            w["speaker"] != spk
            or gap > config.MAX_INTRA_SPEAKER_GAP_SEC
            or (gap > config.SENTENCE_END_GAP_SEC
                and bool(_SENTENCE_END.search(buf_text[-1].strip())))
        )
        if boundary:
            utterances.append(_utt_record(buf_text, buf_conf, s0, e0, spk))
            buf_text = [w["text"]]
            buf_conf = ([float(w["confidence"])]
                        if w.get("confidence") is not None else [])
            s0, e0, spk = w["start"], w["end"], w["speaker"]
        else:
            buf_text.append(w["text"])
            if w.get("confidence") is not None:
                buf_conf.append(float(w["confidence"]))
            e0 = w["end"]

    utterances.append(_utt_record(buf_text, buf_conf, s0, e0, spk))
    return utterances


def merge(words: List[Dict[str, Any]],
          turns: List[Dict[str, Any]],
          y: Optional[np.ndarray] = None,
          sr: Optional[int] = None,
          ) -> List[Dict[str, Any]]:
    """Tag words with speakers, smooth jitter, then group into utterances.

    If ``y`` and ``sr`` are provided, silence-split each suspiciously-long
    Whisper word BEFORE speaker tagging — pulls word boundaries back to
    their actual phoneme region and prevents trailing-silence overlap
    with adjacent-speaker turns (fixes the opening-"Hello" mis-tag).
    Called from ``app.py`` which already has the waveform in scope.
    """
    words = _silence_split_words(words, y, sr)
    return _group_into_utterances(
        _collapse_short_runs(_tag_speakers(words, turns))
    )

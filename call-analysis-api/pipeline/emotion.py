"""Per-utterance speech-emotion recognition (emotion2vec+) + VAD projection.

Each utterance gets:
  emotion_label   — argmax of 9 emotion2vec+ classes
  emotion_probs   — full 9-class probability vector (for future re-derivation)
  valence, arousal, dominance — derived dimensional axes in [-1, +1]

The VAD projection is a fixed, deterministic mapping from class probs to
the three psychometric axes. It's not a learned head — but it is monotonic
with affect, stable across SER model versions, and good enough for sales
coaching dashboards. If we later train a proper learned VAD head on our
own data, the new values just overwrite these columns; the categorical
label and the 9 raw probs in JSONB stay untouched.
"""
from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np

import config

logger = logging.getLogger(__name__)


# ── VAD projection ───────────────────────────────────────────────────────────
def derive_vad(probs: Dict[str, float]) -> Tuple[float, float, float]:
    """Project the 9 emotion2vec+ class probs onto (valence, arousal, dominance).

    Each axis is in [-1, +1] after clipping. "other" and "unknown" carry
    no affective signal and contribute zero.
    """
    g = lambda k: float(probs.get(k, 0.0))  # noqa: E731

    valence    = (g("happy") + g("surprised")
                  - g("sad") - g("angry")
                  - g("fearful") - g("disgusted"))
    arousal    = (g("angry") + g("fearful")
                  + g("happy") + g("surprised") + g("disgusted")
                  - g("neutral") - g("sad"))
    dominance  = (g("angry") + g("happy")
                  - g("fearful") - g("sad"))

    clip = lambda v: max(-1.0, min(1.0, v))  # noqa: E731
    return round(clip(valence), 4), round(clip(arousal), 4), round(clip(dominance), 4)


# ── Slice builder ────────────────────────────────────────────────────────────
def _build_slices(utterances: List[Dict[str, Any]], y: np.ndarray, sr: int
                  ) -> List[Tuple[int, np.ndarray]]:
    """Cut each utterance from ``y`` (cap at SER_SEGMENT_CAP_SEC, drop <0.5s)."""
    cap = int(config.SER_SEGMENT_CAP_SEC * sr)
    min_samples = int(sr * config.MIN_SER_SLICE_SEC)
    slices: List[Tuple[int, np.ndarray]] = []
    for i, u in enumerate(utterances):
        a = int(max(0, float(u["start"]) * sr))
        b = int(min(float(u["end"]) * sr, a + cap))
        if (b - a) < min_samples:
            continue
        slices.append((i, y[a:b].astype(np.float32, copy=False)))
    return slices


# ── emotion2vec+ label parsing ───────────────────────────────────────────────
# FunASR returns labels formatted like "生气/angry" or "happy/快乐". We keep
# the English half (the part the rest of the pipeline reasons over).
def _english_label(raw: str) -> str:
    parts = raw.split("/", 1)
    if len(parts) == 2:
        # Pick the side that's pure ASCII (the English token).
        for side in parts:
            s = side.strip()
            if s and s.isascii():
                return s.lower()
    return raw.strip().lower()


def _normalise_probs(raw_labels: List[str], raw_scores: List[float]
                     ) -> Dict[str, float]:
    """Build a {label: prob} dict keyed by canonical English labels."""
    probs: Dict[str, float] = {}
    for lbl, score in zip(raw_labels, raw_scores):
        canon = _english_label(lbl)
        probs[canon] = round(float(score), 4)
    return probs


# ── Inference ────────────────────────────────────────────────────────────────
def _apply_result(utterance: Dict[str, Any], r: Dict[str, Any]) -> None:
    probs = _normalise_probs(r.get("labels", []), r.get("scores", []))
    if not probs:
        return
    valence, arousal, dominance = derive_vad(probs)
    utterance.update({
        "emotion_label":  max(probs, key=probs.get),
        "emotion_probs":  probs,
        "valence":        valence,
        "arousal":        arousal,
        "dominance":      dominance,
    })


def run(model: Any, utterances: List[Dict[str, Any]],
        y: np.ndarray, sr: int) -> None:
    """Annotate each utterance in place with emotion + VAD fields.

    All slices go to FunASR in ONE generate() call. FunASR iterates list items
    individually (no padding), so the results are bit-identical to the old
    per-utterance loop — verified 2026-06-10 on a real call (40 slices,
    max prob diff 0.000000) — while skipping ~0.05 s of per-call overhead per
    utterance (~3x faster on a 130-utterance call). Falls back to the
    per-utterance loop if the list call fails.
    """
    slices = _build_slices(utterances, y, sr)
    if not slices:
        return

    results = None
    try:
        results = model.generate(
            input=[audio for _, audio in slices],
            granularity="utterance",
            extract_embedding=False,
        )
        if not results or len(results) != len(slices):
            logger.warning("ser_batch_shape_mismatch got=%s want=%d — falling "
                           "back to per-utterance",
                           len(results) if results else None, len(slices))
            results = None
    except Exception as exc:  # noqa: BLE001
        logger.warning("ser_batch_failed error=%r — falling back to "
                       "per-utterance", exc)

    if results is not None:
        for (utt_idx, _), r in zip(slices, results):
            _apply_result(utterances[utt_idx], r)
        return

    for utt_idx, audio in slices:   # fallback: the pre-2026-06-10 loop
        try:
            single = model.generate(
                input=audio,
                granularity="utterance",
                extract_embedding=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("ser_inference_failed utt=%d error=%r", utt_idx, exc)
            continue
        if single:
            _apply_result(utterances[utt_idx], single[0])


# ── Per-speaker aggregation ──────────────────────────────────────────────────
def aggregate(utterances: List[Dict[str, Any]], speaker: str) -> Dict[str, Any]:
    """Roll up per-utterance emotion into one speaker-level summary."""
    speaker_utts = [u for u in utterances
                    if u.get("speaker") == speaker and "emotion_label" in u]
    if not speaker_utts:
        return {"ok": False}

    labels   = [u["emotion_label"] for u in speaker_utts]
    valence  = np.array([u.get("valence",   0.0) for u in speaker_utts], dtype=np.float32)
    arousal  = np.array([u.get("arousal",   0.0) for u in speaker_utts], dtype=np.float32)
    dominanc = np.array([u.get("dominance", 0.0) for u in speaker_utts], dtype=np.float32)

    label_counts = Counter(labels)
    total = len(speaker_utts)
    distribution = {k: round(v / total, 4) for k, v in label_counts.items()}

    p = np.array(list(label_counts.values()), dtype=np.float32) / total
    entropy = round(float(-np.sum(p * np.log(p + 1e-9))), 4)

    # Arousal trend across the call: rising / flat / falling
    if len(arousal) >= 3:
        slope = float(np.polyfit(np.arange(len(arousal), dtype=np.float32),
                                  arousal, 1)[0])
    else:
        slope = 0.0
    if   slope >  config.AROUSAL_SLOPE_THRESHOLD: trend = "escalating"
    elif slope < -config.AROUSAL_SLOPE_THRESHOLD: trend = "de-escalating"
    else:                                         trend = "stable"

    n = len(speaker_utts)
    open_end = max(1, n // 5)
    close_at = max(1, (4 * n) // 5)

    def _dominant(slice_):
        return (Counter(u["emotion_label"] for u in slice_).most_common(1)[0][0]
                if slice_ else None)

    return {
        "ok": True,
        "utterance_count":      total,
        "dominant_emotion":     label_counts.most_common(1)[0][0],
        "emotion_distribution": distribution,
        "emotion_entropy":      entropy,
        "valence_mean":         round(float(np.mean(valence)),  4),
        "arousal_mean":         round(float(np.mean(arousal)),  4),
        "dominance_mean":       round(float(np.mean(dominanc)), 4),
        "arousal_std":          round(float(np.std(arousal)),   4),
        "arousal_trend":        trend,
        "phase_emotions": {
            "opening": _dominant(speaker_utts[:open_end]),
            "middle":  _dominant(speaker_utts[open_end:close_at]),
            "closing": _dominant(speaker_utts[close_at:]),
        },
    }

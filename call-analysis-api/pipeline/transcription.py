"""ASR dispatcher — Whisper (faster-whisper), English-only.

Returns a flat list of word records that ``reconciliation.merge`` consumes:

    [{"start": float, "end": float, "text": str,
      "confidence": Optional[float]}, ...]

There is no language detection and no multilingual handling — Whisper is
always invoked with ``language="en"`` so it skips the language-ID step
and goes straight to English decoding. Word-level timestamps come from
Whisper's alignment head; ``confidence`` is propagated from
``word.probability`` when faster-whisper surfaces it.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from faster_whisper import WhisperModel

import config

logger = logging.getLogger(__name__)


def transcribe(model: WhisperModel, wav_path: str) -> List[Dict[str, Any]]:
    """Run Whisper on ``wav_path`` and return word-level records."""
    if model is None:
        raise RuntimeError("Whisper model is not loaded.")

    # English-only deployment (language="en" skips language-ID) with word-level
    # timestamps (reconciliation needs them). Two robustness guards are kept ON
    # because sales/phone audio needs them — standard Whisper practice, not
    # bespoke experiments:
    #   • vad_filter — trims silence / hold / line-noise so Whisper doesn't
    #     hallucinate text over non-speech.
    #   • condition_on_previous_text=False — breaks Whisper's repetition-loop
    #     hallucination on disfluent / overlapping conversational speech.
    # temperature / beam_size / no_speech_threshold stay at faster-whisper
    # defaults.
    # Decoding params restored 2026-06-02 to the pre-rag-api (2026-05-29)
    # behaviour: an explicit beam search + a short temperature-fallback ladder
    # + the default no-speech threshold. (temperature only escalates when a
    # segment fails Whisper's compression/logprob checks.)
    segments, _info = model.transcribe(
        wav_path,
        language=config.ASR_LANGUAGE,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        condition_on_previous_text=False,
        beam_size=5,
        temperature=[0.0, 0.2, 0.4],
        no_speech_threshold=0.6,
    )

    words: List[Dict[str, Any]] = []
    for seg in segments:
        if not seg.words:
            continue
        for w in seg.words:
            text = (w.word or "").strip()
            if not text:
                continue
            conf = getattr(w, "probability", None)
            words.append({
                "start":      float(w.start),
                "end":        float(w.end),
                "text":       text,
                "confidence": float(conf) if conf is not None else None,
            })

    have_conf = sum(1 for w in words if w["confidence"] is not None)
    logger.info("asr_whisper words=%d words_with_confidence=%d",
                len(words), have_conf)
    return words

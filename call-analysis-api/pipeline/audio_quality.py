"""Audio-quality features per call.

Cheap, deterministic measurements computed once over the converted
16 kHz mono WAV. They go on ``calls`` (audio_snr_db, audio_peak_dbfs,
audio_clipping_rate, audio_silence_at_edges_sec) and let downstream code
— dashboards, ML training filters — distinguish a clean recording from a
hissy / clipped / front-cut one without having to re-listen.

None of these require labeled data; they're objective signal measurements.
Phase 2 will use them as filters when building training sets (e.g.
"only train on calls with SNR > 20 dB and clipping_rate < 0.001").
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict

import numpy as np

logger = logging.getLogger(__name__)


# Anything above ~+0.99 abs is essentially clipped in PCM-16 land
# (true clipping samples sit exactly at +/-1.0 after float normalisation).
_CLIP_THRESHOLD = 0.99
# Per-sample amplitude below which we treat a sample as silence when
# computing SNR / silence-at-edges. 1e-3 corresponds to -60 dBFS.
_SILENCE_AMP    = 1e-3
# Window over which "silence at edges" is measured (seconds at front
# and back of the call). Two seconds is enough to catch the common
# "recording started late" pattern without flagging natural pauses.
_EDGE_WINDOW_SEC = 2.0


def _peak_dbfs(y: np.ndarray) -> float:
    """Peak amplitude in dBFS (-inf for pure silence)."""
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak <= 0.0:
        return -120.0
    return round(20.0 * math.log10(peak), 2)


def _clipping_rate(y: np.ndarray) -> float:
    """Fraction of samples whose abs amplitude is >= _CLIP_THRESHOLD."""
    if y.size == 0:
        return 0.0
    clipped = int(np.sum(np.abs(y) >= _CLIP_THRESHOLD))
    return round(clipped / y.size, 6)


def _snr_db(y: np.ndarray) -> float:
    """Rough signal-to-noise estimate.

    Splits samples into "voiced" (|y| > _SILENCE_AMP) and "silent" pools,
    computes RMS for each, returns 20*log10(RMS_voiced / RMS_silent).

    This is intentionally simple — for a real VAD-based SNR we'd need to
    pull pyannote's voice activity. For triage / filtering this is good
    enough: clean studio calls show SNR > 25 dB, phone calls 10-20 dB,
    very noisy or low-volume calls under 10 dB.
    """
    if y.size == 0:
        return 0.0
    abs_y = np.abs(y)
    voiced = y[abs_y > _SILENCE_AMP]
    silent = y[abs_y <= _SILENCE_AMP]
    if voiced.size == 0 or silent.size == 0:
        return 0.0
    rms_voiced = float(np.sqrt(np.mean(np.square(voiced))))
    rms_silent = float(np.sqrt(np.mean(np.square(silent))))
    if rms_silent <= 0.0 or rms_voiced <= 0.0:
        return 0.0
    return round(20.0 * math.log10(rms_voiced / rms_silent), 2)


def _silence_at_edges_sec(y: np.ndarray, sr: int) -> Dict[str, float]:
    """Seconds of near-silence at the front and back of the recording.

    Useful for spotting calls where the recording started a few seconds
    late (front-cut) or kept rolling after the call ended (back-tail).
    Sampled inside an ``_EDGE_WINDOW_SEC`` window.
    """
    if y.size == 0 or sr <= 0:
        return {"front_sec": 0.0, "back_sec": 0.0}
    window = int(min(_EDGE_WINDOW_SEC * sr, y.size))
    front = y[:window]
    back  = y[-window:]
    abs_front = np.abs(front)
    abs_back  = np.abs(back)

    # Trailing-from-start and leading-from-end runs below threshold.
    front_silent = int(np.argmax(abs_front > _SILENCE_AMP)) if np.any(abs_front > _SILENCE_AMP) else front.size
    back_silent  = int(np.argmax(abs_back[::-1] > _SILENCE_AMP)) if np.any(abs_back > _SILENCE_AMP) else back.size

    return {
        "front_sec": round(front_silent / sr, 3),
        "back_sec":  round(back_silent  / sr, 3),
    }


def measure(y: np.ndarray, sr: int) -> Dict[str, Any]:
    """Single call: returns the flat dict the persistence layer writes."""
    try:
        if y is None or sr <= 0 or y.size == 0:
            return {
                "audio_snr_db":              None,
                "audio_peak_dbfs":           None,
                "audio_clipping_rate":       None,
                "audio_silence_front_sec":   None,
                "audio_silence_back_sec":    None,
            }
        edges = _silence_at_edges_sec(y, sr)
        return {
            "audio_snr_db":            _snr_db(y),
            "audio_peak_dbfs":         _peak_dbfs(y),
            "audio_clipping_rate":     _clipping_rate(y),
            "audio_silence_front_sec": edges["front_sec"],
            "audio_silence_back_sec":  edges["back_sec"],
        }
    except Exception as exc:  # noqa: BLE001 — never fail the call over a metric
        logger.warning("audio_quality_measure_failed err=%r", exc)
        return {
            "audio_snr_db":              None,
            "audio_peak_dbfs":           None,
            "audio_clipping_rate":       None,
            "audio_silence_front_sec":   None,
            "audio_silence_back_sec":    None,
        }

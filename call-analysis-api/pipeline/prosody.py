"""Per-speaker prosody features — eGeMAPS-v02 via openSMILE.

The Geneva Minimalistic Acoustic Parameter Set (88 features, functionals
level) is the de-facto standard for paralinguistic analysis — jitter,
shimmer, F0 stats, formant bandwidths, loudness, voicing probability,
spectral slope. It's the input feature set of the Interspeech ComParE
challenges and the right base for sales-call paralinguistics (stress,
hesitation, vocal effort).

We return:
  * one flat ``Dict[str, float]`` of the 88 functionals — stored as a
    JSONB blob on each ``calls`` row,
  * five derived "headline" features promoted to flat columns (so SQL
    dashboards don't have to dig into the JSONB on every query).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

import config

logger = logging.getLogger(__name__)


# ── openSMILE lazy init ──────────────────────────────────────────────────────
_smile = None


def _get_smile():
    global _smile
    if _smile is not None:
        return _smile
    import opensmile
    _smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    return _smile


# ── Speaker slicing ──────────────────────────────────────────────────────────
def slice_speaker(y: np.ndarray, sr: int, utterances: List[Dict[str, Any]],
                  speaker: str, max_seconds: float) -> np.ndarray:
    """Concatenate up to ``max_seconds`` of audio for the given speaker."""
    cap = int(max_seconds * sr)
    used = 0
    out: List[np.ndarray] = []
    for u in utterances:
        if u.get("speaker") != speaker:
            continue
        a = int(max(0, float(u["start"]) * sr))
        b = int(min(len(y), float(u["end"]) * sr))
        take = min(b - a, cap - used)
        if take <= 0:
            break
        out.append(y[a : a + take])
        used += take
        if used >= cap:
            break
    if not out:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(out).astype(np.float32, copy=False)


# ── eGeMAPS extraction ───────────────────────────────────────────────────────
def features(y: np.ndarray, sr: int) -> Dict[str, Any]:
    """Return the 88 eGeMAPS-v02 functionals + 5 derived headline features."""
    if len(y) < sr:
        return {"ok": False}

    smile = _get_smile()
    try:
        df = smile.process_signal(y, sr)
    except Exception as exc:  # noqa: BLE001
        logger.warning("prosody_extract_failed error=%r", exc)
        return {"ok": False, "error": repr(exc)}

    # process_signal returns a single-row pandas frame; flatten to dict
    row = df.iloc[0].to_dict()
    egemaps = {k: _safe_float(v) for k, v in row.items()}

    # Headline features — the ones SQL dashboards routinely use.
    headline = {
        "loudness_mean":   egemaps.get("loudness_sma3_amean"),
        "pitch_hz_mean":   egemaps.get("F0semitoneFrom27.5Hz_sma3nz_amean"),
        "pitch_hz_std":    egemaps.get("F0semitoneFrom27.5Hz_sma3nz_stddevNorm"),
        "jitter_mean":     egemaps.get("jitterLocal_sma3nz_amean"),
        "shimmer_mean":    egemaps.get("shimmerLocaldB_sma3nz_amean"),
    }

    return {
        "ok": True,
        **{k: round(v, 6) for k, v in headline.items() if v is not None},
        "egemaps": egemaps,
    }


def _safe_float(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if np.isnan(f) or np.isinf(f):
        return None
    return f


# ── Public wrapper used by scorecard.py ──────────────────────────────────────
def speaker_features(y: np.ndarray, sr: int,
                     utterances: List[Dict[str, Any]],
                     speaker: str) -> Dict[str, Any]:
    """Slice the speaker's audio (capped) and return their prosody features."""
    clip = slice_speaker(y, sr, utterances, speaker, config.MAX_PROSODY_SECONDS)
    return features(clip, sr)

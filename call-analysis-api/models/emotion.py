"""emotion2vec+ large — multilingual speech emotion recognition.

Replaces the English-only ehcalabres wav2vec2 head-patch. emotion2vec+ is
Apache-2.0, multilingual (works on Spanish out of the box), trained on
real conversational data, and exposes a clean 9-class output:
``angry, disgusted, fearful, happy, neutral, sad, surprised, other, unknown``.

We load via FunASR's ``AutoModel`` (the model's official runtime). The
loader caches the model on disk under ``HF_HOME`` so warm starts are
~1 second.

``load(model_id=None)`` accepts an explicit override so the hot-swap
registry can request a different checkpoint at runtime.
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

import config

logger = logging.getLogger(__name__)


# Fixed label order produced by the emotion2vec+ checkpoint. We pin it
# here so downstream code (VAD projection, persistence) doesn't drift
# if FunASR ever reorders.
EMOTION_LABELS: List[str] = [
    "angry", "disgusted", "fearful", "happy",
    "neutral", "other", "sad", "surprised", "unknown",
]


def load(model_id: Optional[str] = None) -> Tuple[Any, List[str]]:
    """Load emotion2vec+ large and return ``(model, labels)``."""
    from funasr import AutoModel  # local import keeps test mocks easy

    mid = model_id or config.SER_MODEL_ID
    logger.info("loading_ser model=%s", mid)
    model = AutoModel(
        model=mid,
        disable_update=True,    # don't phone home
        device="cuda" if config.USE_GPU else "cpu",
    )
    logger.info("ser_ready labels=%s", EMOTION_LABELS)
    return model, EMOTION_LABELS

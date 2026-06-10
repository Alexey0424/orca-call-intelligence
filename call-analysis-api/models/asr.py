"""ASR loader — faster-whisper (Whisper) for English-only audio.

``base`` is the default Whisper variant. It is ~140 MB, runs in fp16 on
the 4080 Super in well under real-time, and is the lightest cold-start
of the Whisper family. We pull it from Hugging Face on first build via
``faster-whisper``'s download_root cache, which is then baked into the
``hf_cache`` named volume so subsequent recreates don't re-download.

There is no language detection: the dispatcher always passes
``language="en"``. The model still computes word-level timestamps via
the Whisper alignment head.

``load(model_id=None)`` accepts an explicit override so the hot-swap
registry can request a different variant at runtime without ser-api
needing to re-import config (whose constants are frozen at import).
"""
from __future__ import annotations

import logging
from typing import Optional

from faster_whisper import WhisperModel

import config

logger = logging.getLogger(__name__)


def load(model_id: Optional[str] = None) -> WhisperModel:
    mid     = model_id or config.WHISPER_MODEL_ID
    device  = "cuda" if config.USE_GPU else "cpu"
    compute = config.WHISPER_COMPUTE if config.USE_GPU else "float32"
    logger.info("loading_whisper model=%s device=%s compute=%s",
                mid, device, compute)
    return WhisperModel(
        mid,
        device=device,
        compute_type=compute,
        num_workers=4,
        download_root=config.WHISPER_CACHE,
    )

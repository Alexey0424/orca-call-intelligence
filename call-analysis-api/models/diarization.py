"""Pyannote diarization pipeline loader (speaker-diarization-3.1).

``load(model_id=None)`` accepts an explicit override so the hot-swap
registry can request a different pipeline at runtime.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
from pyannote.audio import Pipeline

import config

logger = logging.getLogger(__name__)


def load(model_id: Optional[str] = None) -> Pipeline:
    mid = model_id or config.DIARIZATION_MODEL_ID
    logger.info("loading_diarization model=%s", mid)
    pipe = Pipeline.from_pretrained(mid, token=config.HF_TOKEN)
    if config.USE_GPU:
        pipe.to(torch.device("cuda"))
    # Cap the on-GPU batch sizes so a long call can't transiently allocate the
    # whole card during the segmentation/embedding steps. Batch size is a pure
    # throughput knob — it does not change diarization output. Attributes are
    # set best-effort (the attribute name can drift across pyannote versions).
    for attr, val in (("segmentation_batch_size", config.DIARIZATION_SEG_BATCH),
                      ("embedding_batch_size",    config.DIARIZATION_EMB_BATCH)):
        if hasattr(pipe, attr):
            try:
                setattr(pipe, attr, val)
                logger.info("diarization_batch_cap %s=%d", attr, val)
            except Exception as exc:  # noqa: BLE001
                logger.warning("diarization_batch_cap_failed %s err=%r", attr, exc)
    return pipe

"""Multilingual sentiment classifier (XLM-RoBERTa, 3-class).

``load(model_id=None)`` accepts an explicit override so the hot-swap
registry can request a different classifier at runtime.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

from transformers import AutoTokenizer
from transformers import pipeline as hf_pipeline

import config

logger = logging.getLogger(__name__)


def load(model_id: Optional[str] = None) -> Tuple[object, Dict[int, str]]:
    """Return ``(pipeline, id2label)``. Pipeline is a HF text-classification pipe."""
    mid = model_id or config.SENTIMENT_MODEL_ID
    logger.info("loading_sentiment model=%s", mid)
    tok = AutoTokenizer.from_pretrained(mid)
    pipe = hf_pipeline(
        "sentiment-analysis",
        model=mid, tokenizer=tok,
        device=0 if config.USE_GPU else -1,
        torch_dtype=config.TORCH_DTYPE,
        truncation=True, max_length=512,
        batch_size=config.SENTIMENT_BATCH_SIZE,
    )
    id2label: Dict[int, str] = pipe.model.config.id2label
    return pipe, id2label

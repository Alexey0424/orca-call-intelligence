"""Per-utterance text sentiment (Twitter RoBERTa, English) + aggregation.

Output per utterance (added in place):
    sentiment_label  ∈ {"positive", "neutral", "negative"}
    sentiment_score  ∈ [-1, +1]
    sentiment_probs  full {label: prob} dict

Per-speaker aggregate fields used by dashboards:
    sentiment_mean, sentiment_std,
    positive_ratio, neutral_ratio, negative_ratio,
    sentiment_trajectory ∈ {"improving", "stable", "declining"}
"""
from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict, List

import numpy as np
import torch

import config

logger = logging.getLogger(__name__)


def _normalise_label(raw: str, id2label: Dict[int, str]) -> str:
    """Map LABEL_0 / POSITIVE / etc. to a canonical lowercase name."""
    for k, v in id2label.items():
        if raw == f"LABEL_{k}" or raw.upper() == v.upper():
            return v.lower()
    return raw.lower()


def run(pipe, utterances: List[Dict[str, Any]],
        id2label: Dict[int, str]) -> None:
    """Annotate each utterance in place with sentiment fields."""
    for u in utterances:
        u.update({"sentiment_label": "neutral",
                  "sentiment_score": 0.0,
                  "sentiment_probs": {}})

    indexed = [(i, u["text"].strip()) for i, u in enumerate(utterances)
               if u.get("text", "").strip()]
    if not indexed:
        return

    idxs, texts = zip(*indexed)
    with torch.inference_mode():
        results = pipe(list(texts), top_k=None, truncation=True,
                       batch_size=config.SENTIMENT_BATCH_SIZE)

    for utt_idx, label_scores in zip(idxs, results):
        probs: Dict[str, float] = {}
        for ls in label_scores:
            raw   = ls["label"] if isinstance(ls, dict) else str(ls[0])
            score = float(ls["score"] if isinstance(ls, dict) else ls[1])
            probs[_normalise_label(raw, id2label)] = round(score, 4)

        best = max(probs, key=lambda k: probs[k])
        sign = 1.0 if "positive" in best else (-1.0 if "negative" in best else 0.0)
        utterances[utt_idx].update({
            "sentiment_label": best,
            "sentiment_score": round(sign * probs[best], 4),
            "sentiment_probs": probs,
        })


def aggregate(utterances: List[Dict[str, Any]], speaker: str) -> Dict[str, Any]:
    speaker_utts = [u for u in utterances
                    if u.get("speaker") == speaker and "sentiment_score" in u]
    if not speaker_utts:
        return {"ok": False}

    scores = np.array([u["sentiment_score"] for u in speaker_utts], dtype=np.float32)
    word_counts = np.array(
        [max(1, len(u.get("text", "").split())) for u in speaker_utts],
        dtype=np.float32,
    )
    label_counts = Counter(u.get("sentiment_label", "neutral") for u in speaker_utts)
    total = len(speaker_utts)

    if len(scores) >= 3:
        slope = float(np.polyfit(np.arange(len(scores), dtype=np.float32),
                                  scores, 1)[0])
    else:
        slope = 0.0
    if   slope >  config.SENTIMENT_SLOPE_THRESHOLD: trajectory = "improving"
    elif slope < -config.SENTIMENT_SLOPE_THRESHOLD: trajectory = "declining"
    else:                                           trajectory = "stable"

    return {
        "ok": True,
        "utterance_count":      total,
        "sentiment_mean":       round(float(np.average(scores, weights=word_counts)), 4),
        "sentiment_std":        round(float(np.std(scores)), 4),
        "positive_ratio":       round(label_counts.get("positive", 0) / total, 4),
        "neutral_ratio":        round(label_counts.get("neutral",  0) / total, 4),
        "negative_ratio":       round(label_counts.get("negative", 0) / total, 4),
        "sentiment_trajectory": trajectory,
    }

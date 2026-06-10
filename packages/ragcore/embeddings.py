"""Thin client for Ollama's /api/embeddings with Matryoshka truncation.

Qwen3-Embedding-4B is trained with **Matryoshka Representation Learning**:
the leading dimensions of the native 2560-d output are themselves a
self-contained, well-trained embedding. We truncate to ``EMBEDDING_DIM``
client-side and L2-renormalise — keeping ~95% of retrieval quality at
<26% of the storage cost of the native vector (Qwen3 Technical Report,
arXiv 2505.09388; Alibaba Cloud "Mastering Qwen3 Embedding", June 2025).

This lets us put the best open multilingual embedding model on top of a
1024-d pgvector schema. The schema stays small, HNSW indexes stay
cheap, and dashboards over the column don't blow up.
"""
from __future__ import annotations

import logging
import math
import unicodedata
from typing import List, Optional

import httpx

from ragcore import config

logger = logging.getLogger(__name__)


_TRAILING_PUNCT = ".,;:!?-—–_\"'`"


def _sanitize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).strip()
    text = " ".join(text.split())
    return text.rstrip(_TRAILING_PUNCT).strip()


def _has_nan(vec: List[float]) -> bool:
    return any(math.isnan(v) for v in vec[:64])


def _truncate_and_normalise(vec: List[float]) -> List[float]:
    """Matryoshka truncate to EMBEDDING_DIM and L2-renormalise.

    Native Qwen3-Embedding-4B output is 2560-d; we keep the leading
    EMBEDDING_DIM (default 1024). The native vector is already
    L2-normalised, but truncation changes its norm — we re-normalise so
    cosine similarity stays meaningful.
    """
    truncated = vec[:config.EMBEDDING_DIM]
    norm = math.sqrt(sum(v * v for v in truncated))
    if norm <= 0:
        raise ValueError("Embedding has zero norm after truncation")
    inv = 1.0 / norm
    return [v * inv for v in truncated]


def _embed_options() -> dict:
    """Per-request Ollama options for the embedder.

    ``EMBEDDING_NUM_GPU < 0`` → return ``{}`` so we omit ``num_gpu`` entirely and
    Ollama auto-fits the model to free VRAM (GPU when there's room). ``>= 0`` →
    pin exactly that many layers (0 = CPU). See config.EMBEDDING_NUM_GPU.
    """
    n = config.EMBEDDING_NUM_GPU
    return {} if n < 0 else {"num_gpu": n}


def _post(text: str) -> List[float]:
    """One shot. Raises on non-2xx, schema mismatch, NaN, or wrong shape."""
    payload: dict = {"model": config.embedding_model_id(), "prompt": text}
    opts = _embed_options()
    if opts:
        payload["options"] = opts
    resp = httpx.post(
        f"{config.OLLAMA_HOST}/api/embeddings",
        json=payload,
        timeout=config.EMBEDDING_TIMEOUT_SEC,
    )
    resp.raise_for_status()
    raw = resp.json().get("embedding")
    if not isinstance(raw, list) or not raw:
        raise ValueError(
            f"Ollama returned unexpected embedding shape: {resp.json()!r}"
        )
    if _has_nan(raw):
        raise ValueError("Embedding contains NaN values")

    native_dim = len(raw)
    if native_dim < config.EMBEDDING_DIM:
        raise ValueError(
            f"Native embedding dim={native_dim} is smaller than configured "
            f"EMBEDDING_DIM={config.EMBEDDING_DIM}; cannot truncate up."
        )
    return _truncate_and_normalise(raw)


def embed_one(text: str) -> List[float]:
    """Embed ``text`` and return its MRL-truncated, L2-normalised vector.

    One retry on transient failure (whitespace-trim + trailing-punct strip)
    handles the occasional Ollama NaN that some embedding models emit
    on weirdly-shaped inputs.
    """
    if not text or not text.strip():
        raise ValueError("Cannot embed empty text")
    try:
        return _post(text)
    except (httpx.HTTPStatusError, ValueError) as exc:
        sanitised = _sanitize(text)
        if not sanitised or sanitised == text:
            raise
        logger.warning(
            "embed_failed retrying_sanitised orig_len=%d sani_len=%d error=%r",
            len(text), len(sanitised), exc,
        )
        return _post(sanitised)


def embed_one_safe(text: str) -> Optional[List[float]]:
    """Best-effort embed: returns ``None`` on any failure instead of raising."""
    if not text or not text.strip():
        return None
    try:
        return embed_one(text)
    except Exception as exc:  # noqa: BLE001
        logger.warning("embed_one_safe giving_up len=%d error=%r", len(text), exc)
        return None


# ── Batching (CF-15) ─────────────────────────────────────────────────────────
# Ollama's /api/embed endpoint (note: NOT /api/embeddings) accepts an
# `input: [str, ...]` array and returns `embeddings: [[float,...], ...]`
# in the same order. We chunk to bound per-request memory + timeout: 16
# is a conservative size for 4096-token utterance text (typical Q4_K_M
# context budget on a 14B-class embedding model).
_EMBED_BATCH_CHUNK = 16


def embed_batch_safe(texts: List[str]) -> List[Optional[List[float]]]:
    """Embed many texts in fewer HTTP round-trips.

    Returns a list aligned 1:1 with ``texts``. Slots whose text is empty
    or where the per-row sanitiser/normaliser failed contain ``None``
    (matching ``embed_one_safe``'s contract). On *batch* failure (the
    whole HTTP call errors out) we fall back per-row so a single bad
    input doesn't poison the rest.
    """
    out: List[Optional[List[float]]] = [None] * len(texts)
    if not texts:
        return out

    # Pre-filter blanks so we don't waste batch slots.
    idx_keep: List[int] = []
    payload_keep: List[str] = []
    for i, t in enumerate(texts):
        if t and t.strip():
            idx_keep.append(i)
            payload_keep.append(t)

    for start in range(0, len(payload_keep), _EMBED_BATCH_CHUNK):
        chunk_texts = payload_keep[start:start + _EMBED_BATCH_CHUNK]
        chunk_idx   = idx_keep[start:start + _EMBED_BATCH_CHUNK]
        try:
            vectors = _post_batch(chunk_texts)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "embed_batch_failed chunk_start=%d chunk_len=%d error=%r "
                "— falling back to per-row",
                start, len(chunk_texts), exc,
            )
            for j, txt in zip(chunk_idx, chunk_texts):
                out[j] = embed_one_safe(txt)
            continue
        for j, vec in zip(chunk_idx, vectors):
            out[j] = vec
    return out


def _post_batch(texts: List[str]) -> List[List[float]]:
    """One Ollama batch call. Raises on shape / NaN / non-2xx."""
    payload: dict = {"model": config.embedding_model_id(), "input": texts}
    opts = _embed_options()
    if opts:
        payload["options"] = opts
    resp = httpx.post(
        f"{config.OLLAMA_HOST}/api/embed",
        json=payload,
        timeout=config.EMBEDDING_TIMEOUT_SEC,
    )
    resp.raise_for_status()
    body = resp.json() or {}
    raw_vecs = body.get("embeddings")
    if not isinstance(raw_vecs, list) or len(raw_vecs) != len(texts):
        raise ValueError(
            f"Ollama returned unexpected batch shape: "
            f"got {len(raw_vecs) if isinstance(raw_vecs, list) else type(raw_vecs).__name__} "
            f"vectors for {len(texts)} inputs"
        )
    out: List[List[float]] = []
    for i, raw in enumerate(raw_vecs):
        if not isinstance(raw, list) or not raw:
            raise ValueError(f"Empty/invalid vector at batch index {i}")
        if _has_nan(raw):
            raise ValueError(f"NaN in vector at batch index {i}")
        if len(raw) < config.EMBEDDING_DIM:
            raise ValueError(
                f"Native dim={len(raw)} at batch index {i} smaller than "
                f"EMBEDDING_DIM={config.EMBEDDING_DIM}"
            )
        out.append(_truncate_and_normalise(raw))
    return out

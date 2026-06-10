"""Zero-shot entity extraction via GLiNER.

For each call we extract custom-schema entities (competitors,
neighborhoods, mortgage brokers, price points, property types, agent
names) and persist them into the ``events`` table. The schema lives in
``config.NER_ENTITY_TYPES`` — edit it freely; GLiNER is zero-shot, so
no retraining is required when you add a new type.

Why a separate stage and not just trust the LLM's JSON: GLiNER's
entity-name normalisation is far more stable than an LLM that may
quietly rename "Compass" → "Compass Realty" between runs. Storing the
verbatim text alongside the model's normalisation lets us audit drift
without re-running the LLM.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import config

logger = logging.getLogger(__name__)


_model = None


def _get_model():
    """Lazy-load GLiNER on first call; afterwards keep the weights cached in
    process RAM (~1.2 GB) and only hop CPU↔GPU per call. The from-disk load
    costs ~5-8 s per call in serialized mode; the CPU→GPU move is <1 s."""
    global _model
    if _model is None:
        from gliner import GLiNER  # local import keeps cold-import cheap
        logger.info("loading_ner model=%s", config.NER_MODEL_ID)
        _model = GLiNER.from_pretrained(config.NER_MODEL_ID)
    if config.USE_GPU:
        try:
            _model = _model.to("cuda")
        except Exception as exc:  # noqa: BLE001
            logger.warning("ner_gpu_move_failed err=%r — staying on CPU", exc)
    return _model


def release() -> None:
    """Free GLiNER's VRAM (serialized mode) but keep the weights in CPU RAM.

    Moving to CPU instead of dropping the object preserves the
    VRAM-free-when-idle goal while skipping the ~5-8 s from-disk reload on the
    next call. Caller is responsible for the gc/empty_cache afterwards.
    """
    global _model
    if _model is None:
        return
    try:
        _model = _model.to("cpu")
    except Exception as exc:  # noqa: BLE001 — fall back to a full drop
        logger.warning("ner_cpu_move_failed err=%r — dropping model", exc)
        _model = None


def _utterance_at(utterances: List[Dict[str, Any]], text_offset: int
                  ) -> Optional[Dict[str, Any]]:
    """Map a character offset in the merged transcript back to its utterance."""
    cumulative = 0
    for u in utterances:
        seg_text = (u.get("text") or "").strip()
        if not seg_text:
            continue
        # +1 for the join-newline / space we add between utterances.
        if cumulative + len(seg_text) >= text_offset:
            return u
        cumulative += len(seg_text) + 1
    return None


def extract(utterances: List[Dict[str, Any]],
            roles: Dict[str, str]) -> List[Dict[str, Any]]:
    """Run GLiNER over the merged transcript and return event dicts.

    Each returned dict matches the columns in the ``events`` table:
        kind, text, normalized, confidence, start_sec, speaker_role
    """
    if not utterances:
        return []

    # Join utterances so GLiNER sees full-sentence context, while we still
    # track per-utterance offsets to recover ``start_sec`` and speaker role.
    parts = [(u.get("text") or "").strip() for u in utterances]
    transcript = "\n".join(p for p in parts if p)
    if not transcript:
        return []

    try:
        model = _get_model()
    except Exception as exc:  # noqa: BLE001
        logger.warning("ner_load_failed error=%r — skipping NER stage", exc)
        return []

    try:
        raw = model.predict_entities(
            transcript,
            config.NER_ENTITY_TYPES,
            threshold=config.NER_THRESHOLD,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("ner_predict_failed error=%r", exc)
        return []

    events: List[Dict[str, Any]] = []
    for ent in raw:
        text = (ent.get("text") or "").strip()
        if not text:
            continue
        offset = int(ent.get("start", 0))
        host = _utterance_at(utterances, offset)
        start_sec = float(host.get("start", 0.0)) if host else None
        speaker_role = (roles.get(host.get("speaker", "")) if host else None)
        events.append({
            "kind":         (ent.get("label") or "entity").lower(),
            "text":         text,
            "normalized":   text.strip().lower(),
            "confidence":   round(float(ent.get("score", 0.0)), 4),
            "start_sec":    start_sec,
            "speaker_role": speaker_role,
            "metadata":     None,
        })

    logger.info("ner_extracted entities=%d types=%s",
                len(events), sorted({e["kind"] for e in events}))
    return events

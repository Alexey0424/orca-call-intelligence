"""PII redaction over utterance transcripts using Microsoft Presidio.

For every utterance we keep:
  * ``text``           — the raw ASR transcript (unchanged, for replay)
  * ``text_redacted``  — same content with detected PII replaced by tags
                         like ``<PERSON>`` / ``<PHONE_NUMBER>``

We also build a per-call ``pii_detected`` JSONB summary
({"PERSON": 12, "PHONE_NUMBER": 2, ...}) so the dashboards can show
"this call mentioned 12 names" without re-scanning the transcript.

This module is *additive*: if Presidio fails to load (e.g. spaCy model
missing) the call still completes — the raw ``text`` is duplicated into
``text_redacted`` and ``pii_detected = {}``. The audit-grade right call
is to never block ingestion on a privacy enhancement; the operator sees
the warning in logs.

Phase 2 manual tasks (not done here):
  * Review false positives on real calls (Presidio sometimes flags
    common words like "Brick" as PERSON).
  * Add custom recognizers for company-specific identifiers if needed.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Loaded lazily — Presidio's analyzer is ~150 MB resident with the spaCy
# en_core_web_sm pipeline so we pay it on first use, not on import.
_ANALYZER = None
_ANONYMIZER = None
_LOAD_FAILED = False


# Conservative entity set. Adding more (CREDIT_CARD, IBAN_CODE, NRP, etc.)
# is one config change away; we start narrow so we don't over-redact
# normal sales-call vocabulary.
_DEFAULT_ENTITIES = [
    "PERSON",
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "US_SSN",
    "CREDIT_CARD",
    "LOCATION",
    "IBAN_CODE",
]

# Minimum analyzer confidence to act on. Tweak in phase 2 if the false-
# positive rate is too high — start strict.
_MIN_SCORE = 0.6


def _lazy_load() -> bool:
    """Initialise Presidio + Anonymizer engines. Returns False on failure."""
    global _ANALYZER, _ANONYMIZER, _LOAD_FAILED
    if _ANALYZER is not None and _ANONYMIZER is not None:
        return True
    if _LOAD_FAILED:
        return False
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine

        _ANALYZER = AnalyzerEngine()
        _ANONYMIZER = AnonymizerEngine()
        logger.info("pii_engine_loaded entities=%s min_score=%.2f",
                    _DEFAULT_ENTITIES, _MIN_SCORE)
        return True
    except Exception as exc:  # noqa: BLE001
        _LOAD_FAILED = True
        logger.warning(
            "pii_engine_load_failed err=%r — utterances will be persisted "
            "without redaction; install presidio + spacy en_core_web_sm",
            exc,
        )
        return False


def redact_one(text: str, language: str = "en"
               ) -> Dict[str, Any]:
    """Redact a single utterance.

    Returns ``{"text_redacted": str, "entities": {label: count}}``. On
    engine failure or empty text returns the original text with an empty
    entity map so the caller doesn't have to branch.
    """
    if not text or not text.strip():
        return {"text_redacted": text or "", "entities": {}}

    if not _lazy_load():
        return {"text_redacted": text, "entities": {}}

    try:
        results = _ANALYZER.analyze(
            text=text,
            language=language,
            entities=_DEFAULT_ENTITIES,
            score_threshold=_MIN_SCORE,
        )
        if not results:
            return {"text_redacted": text, "entities": {}}

        anonymised = _ANONYMIZER.anonymize(text=text, analyzer_results=results)

        # Count entities by type (post-threshold).
        counts: Dict[str, int] = {}
        for r in results:
            counts[r.entity_type] = counts.get(r.entity_type, 0) + 1

        return {
            "text_redacted": anonymised.text,
            "entities":      counts,
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("pii_redact_failed text_len=%d err=%r", len(text), exc)
        return {"text_redacted": text, "entities": {}}


def annotate(utterances: List[Dict[str, Any]],
             language: str = "en") -> Dict[str, int]:
    """Run redaction over every utterance in-place. Returns the per-call summary.

    The summary maps entity_type → total count across the call, suitable
    for persistence into ``calls.pii_detected``. Side-effect on each
    utterance dict: adds ``text_redacted`` and ``pii_entities`` keys.
    """
    summary: Dict[str, int] = {}
    for u in utterances:
        out = redact_one(u.get("text") or "", language=language)
        u["text_redacted"] = out["text_redacted"]
        u["pii_entities"]  = out["entities"]
        for k, v in out["entities"].items():
            summary[k] = summary.get(k, 0) + v
    return summary

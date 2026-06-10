"""On-demand call report: render-once-then-cache.

A report is fully reproducible from persisted data: the aggregate scorecard
(``calls.scorecard``), the transcript + per-utterance affect (the ``utterances``
table), and speaker/name info (the ``calls`` row). ``_build_render_payload``
reassembles the ``{metadata, scorecard, utterances}`` envelope that render-api's
exec_view expects (it reads the transcript and the sentiment/emotion timeline
charts from the top-level ``utterances`` list — NOT from ``calls.scorecard``,
which holds aggregates only). We render lazily on first request and cache the
returned URLs into calls.pdf_url/html_url; a later request that finds them set
(and the file still on the shared /data volume) is a cache hit.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

import httpx

from ragcore import config, db

logger = logging.getLogger(__name__)

# Shared report artefact root (render-api writes under {OUTPUT_DIR}=/data/reports).
DATA_DIR = os.getenv("DATA_DIR", "/data")

# Full-report render (transcript + 6 charts) is heavier than the dashboard
# render, so give it more headroom than the generic render timeout.
_RENDER_TIMEOUT_SEC = max(float(config.SER_RENDER_TIMEOUT_SEC), 120.0)


def _url_to_local_path(url: Optional[str]) -> Optional[str]:
    """Map a render-api '/reports/...' URL to its path on the shared volume.

    Returns None if the URL isn't a render-api reports URL we can resolve.
    """
    if not url:
        return None
    marker = "/reports/"
    i = url.find(marker)
    if i == -1:
        return None
    rel = url[i + len(marker):]
    rel = os.path.normpath(rel).lstrip("/\\")
    base = os.path.join(DATA_DIR, "reports")
    candidate = os.path.join(base, rel)
    if not os.path.abspath(candidate).startswith(os.path.abspath(base)):
        return None
    return candidate


def _cached_urls(call_id: str) -> Dict[str, Any]:
    """Cache probe. ``{"_not_found": True}`` if no such call; ``{cached: True,
    pdf_url, html_url}`` if already rendered AND the file is still on disk;
    ``{}`` if a (re)render is needed."""
    conn = db.connect()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT pdf_url, html_url FROM calls WHERE call_id = %s",
                        (call_id,))
            row = cur.fetchone()
    finally:
        conn.close()
    if row is None:
        return {"_not_found": True}
    pdf_url, html_url = row
    local = _url_to_local_path(pdf_url)
    if pdf_url and local and os.path.exists(local):
        return {"pdf_url": pdf_url, "html_url": html_url, "cached": True}
    return {}


def _persist_urls(call_id: str, pdf_url: Optional[str], html_url: Optional[str]) -> None:
    conn = db.connect()
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE calls SET pdf_url = %s, html_url = %s, updated_at = NOW() "
                "WHERE call_id = %s",
                (pdf_url, html_url, call_id),
            )
    finally:
        conn.close()


# Columns pulled from the utterances table to rebuild the transcript + the
# per-utterance data the sentiment/emotion timeline charts consume.
_UTT_SQL = """
    SELECT start_sec, end_sec, speaker, text, text_redacted, pii_entities,
           asr_confidence, is_backchannel, question_types,
           sentiment_label, sentiment_score,
           emotion_label, emotion_probs, valence, arousal, dominance
      FROM utterances WHERE call_id = %s ORDER BY idx
"""


def _closer_speaker_from_roles(roles: Any) -> str:
    """Diarizer label the scorecard tagged as the closer (else SPEAKER_01).

    render-api's exec_view uses ``metadata.closer_speaker`` to assign each
    utterance a closer/client role in the transcript + charts.
    """
    if isinstance(roles, dict):
        for spk, role in roles.items():
            if role == "closer":
                return spk
    return "SPEAKER_01"


def _shape_utterance(d: Dict[str, Any], language: Optional[str]) -> Dict[str, Any]:
    """Map a utterances-table row to the shape render-api's exec_view expects.

    Key rename to watch: the table stores start_sec/end_sec; render reads
    start/end. sentiment_probs isn't persisted per-utterance (the charts use
    label/score), so it's left empty.
    """
    return {
        "start":           d["start_sec"],
        "end":             d["end_sec"],
        "text":            d["text"],
        "speaker":         d["speaker"],
        "language":        language,
        "asr_confidence":  d["asr_confidence"],
        "sentiment_label": d["sentiment_label"],
        "sentiment_score": d["sentiment_score"],
        "sentiment_probs": {},
        "emotion_label":   d["emotion_label"],
        "emotion_probs":   d["emotion_probs"] or {},
        "valence":         d["valence"],
        "arousal":         d["arousal"],
        "dominance":       d["dominance"],
        "is_backchannel":  d["is_backchannel"],
        "question_types":  d["question_types"] or {},
        "text_redacted":   d["text_redacted"],
        "pii_entities":    d["pii_entities"] or {},
    }


def _build_render_payload(call_id: str) -> Optional[Dict[str, Any]]:
    """Reassemble the full ``{metadata, scorecard, utterances}`` envelope.

    ``calls.scorecard`` is aggregates-only; the transcript + per-utterance affect
    live in the ``utterances`` table and speaker/name info in the ``calls`` row.
    Returns None if the call doesn't exist or has no scorecard.
    """
    conn = db.connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT scorecard, closer_name, client_name, "
                "       file_name, file_hash, language, model_versions "
                "FROM calls WHERE call_id = %s",
                (call_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            (scorecard, closer_name, client_name,
             file_name, file_hash, language, model_versions) = row
            if not isinstance(scorecard, dict):
                scorecard = json.loads(scorecard) if scorecard else {}
            if not scorecard:
                return None
            cur.execute(_UTT_SQL, (call_id,))
            ucols = [c[0] for c in cur.description]
            urows = [dict(zip(ucols, r)) for r in cur.fetchall()]
    finally:
        conn.close()

    utterances = [_shape_utterance(u, language) for u in urows]
    metadata = {
        "call_id":        call_id,
        "closer_speaker": _closer_speaker_from_roles(scorecard.get("roles")),
        "closer_name":    closer_name,
        "client_name":    client_name,
        "file_name":      file_name,
        "file_hash":      file_hash,
        "language":       language,
        "model_versions": model_versions,
    }
    return {"metadata": metadata, "scorecard": scorecard, "utterances": utterances}


def get_or_render(call_id: str) -> Dict[str, Any]:
    """Return the report URLs for a call, rendering + caching on first request."""
    cached = _cached_urls(call_id)
    if cached.get("_not_found"):
        return {"ok": False, "error": f"call_id_not_found: {call_id}"}
    if cached.get("cached"):
        return {"ok": True, "cached": True,
                "pdf_url": cached["pdf_url"], "html_url": cached["html_url"]}

    payload = _build_render_payload(call_id)
    if payload is None:
        return {"ok": False, "error": f"no_scorecard_for_call: {call_id}"}

    try:
        r = httpx.post(
            f"{config.SER_RENDER_URL}/render",
            params={"output_format": "both"},
            json=payload,
            timeout=_RENDER_TIMEOUT_SEC,
        )
        r.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        logger.error("render_failed call_id=%s error=%r", call_id, exc)
        return {"ok": False, "error": f"render_api_error: {exc}"}
    files = (r.json() or {}).get("files", {})
    pdf_url, html_url = files.get("pdf_url"), files.get("html_url")
    _persist_urls(call_id, pdf_url, html_url)
    logger.info("report_rendered call_id=%s pdf=%s utterances=%d",
                call_id, bool(pdf_url), len(payload["utterances"]))
    return {"ok": True, "cached": False, "pdf_url": pdf_url, "html_url": html_url}

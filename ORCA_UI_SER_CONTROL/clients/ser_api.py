"""ser-api HTTP client (mostly read).

Model swaps used to require a container restart — we now POST to
``/admin/reload-model`` so ser-api hot-swaps the affected role inside
its own GPU lock, no restart needed.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx

import config

logger = logging.getLogger(__name__)


def _client() -> httpx.Client:
    return httpx.Client(base_url=config.SER_API_URL,
                        timeout=config.SER_API_TIMEOUT_SEC)


def health() -> Dict[str, Any]:
    """Returns ``{"ok": False, "error": "..."}`` on any HTTP failure."""
    try:
        with _client() as c:
            r = c.get("/health")
            r.raise_for_status()
            return r.json()
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": repr(exc)}


def jobs_all(status: Optional[str] = None,
             limit: int = 500,
             offset: int = 0) -> List[Dict[str, Any]]:
    """Pull all jobs from ser-api /jobs/all. Failures return []."""
    params: Dict[str, Any] = {"limit": limit, "offset": offset}
    if status:
        params["status"] = status
    try:
        with _client() as c:
            r = c.get("/jobs/all", params=params)
            r.raise_for_status()
            return r.json().get("jobs") or []
    except Exception as exc:  # noqa: BLE001
        logger.warning("ser_api_jobs_all_failed err=%r", exc)
        return []


def job_one(job_id: str) -> Optional[Dict[str, Any]]:
    try:
        with _client() as c:
            r = c.get(f"/jobs/{job_id}")
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json().get("job")
    except Exception as exc:  # noqa: BLE001
        logger.warning("ser_api_job_one_failed job_id=%s err=%r", job_id, exc)
        return None


def job_current() -> Optional[Dict[str, Any]]:
    try:
        with _client() as c:
            r = c.get("/jobs/current")
            r.raise_for_status()
            return r.json().get("job")
    except Exception as exc:  # noqa: BLE001
        logger.warning("ser_api_job_current_failed err=%r", exc)
        return None


def job_by_run(run_no: int) -> Optional[Dict[str, Any]]:
    try:
        with _client() as c:
            r = c.get(f"/jobs/by-run/{run_no}")
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json().get("job")
    except Exception as exc:  # noqa: BLE001
        logger.warning("ser_api_job_by_run_failed run=%s err=%r", run_no, exc)
        return None


def run_index(limit: int = 500) -> List[Dict[str, Any]]:
    try:
        with _client() as c:
            r = c.get("/jobs/run-index", params={"limit": limit})
            r.raise_for_status()
            return r.json().get("runs") or []
    except Exception as exc:  # noqa: BLE001
        logger.warning("ser_api_run_index_failed err=%r", exc)
        return []


def wipe_database(confirm: str = "DELETE") -> Dict[str, Any]:
    """POST /admin/wipe. The control UI passes the exact confirmation
    string the operator typed; this is just a pass-through."""
    try:
        with httpx.Client(base_url=config.SER_API_URL, timeout=30.0) as c:
            r = c.post("/admin/wipe", json={"confirm": confirm})
            if r.status_code != 200:
                return {"ok": False, "status_code": r.status_code,
                        "detail": r.text}
            return r.json()
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": repr(exc)}


def reload_model(role: str) -> Dict[str, Any]:
    """POST /admin/reload-model. Hot-swap one role without restarting ser-api.

    Generous timeout — a fresh download of large-v3 / emotion2vec_plus_large
    can run for many minutes on the first swap to that variant.
    """
    try:
        with httpx.Client(base_url=config.SER_API_URL, timeout=900.0) as c:
            r = c.post("/admin/reload-model", json={"role": role})
            if r.status_code != 200:
                return {"ok": False, "status_code": r.status_code,
                        "detail": r.text[:500]}
            return r.json()
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": repr(exc)}


def cache_status(asr_ids: list, diar_ids: list,
                 sent_ids: list, ser_ids: list) -> Dict[str, Any]:
    """GET /admin/cache-status. Probe disk-cache state for every dropdown
    option so the UI can show downloaded ✓ / needs-pull next to each id.
    """
    params = {
        "asr_ids":  ",".join(asr_ids),
        "diar_ids": ",".join(diar_ids),
        "sent_ids": ",".join(sent_ids),
        "ser_ids":  ",".join(ser_ids),
    }
    try:
        with httpx.Client(base_url=config.SER_API_URL, timeout=15.0) as c:
            r = c.get("/admin/cache-status", params=params)
            r.raise_for_status()
            return r.json()
    except Exception as exc:  # noqa: BLE001
        logger.warning("cache_status_failed err=%r", exc)
        return {"ok": False, "error": repr(exc),
                "asr": {}, "diarization": {}, "sentiment": {}, "ser": {},
                "loaded": {}}


def run_scorecard(file_bytes: bytes, file_name: str,
                  persist: bool = True,
                  index_utterances: bool = True) -> Dict[str, Any]:
    """Upload an audio file straight to ser-api /scorecard.

    Long timeout (20 min) — the first request after a cold start can be
    slow if Ollama hasn't warmed up. The UI's "Run scorecard" button
    streams the response back so the operator sees the result.
    """
    files  = {"file": (file_name, file_bytes, "application/octet-stream")}
    params = {"persist": str(persist).lower(),
              "index_utterances": str(index_utterances).lower()}
    try:
        with httpx.Client(base_url=config.SER_API_URL, timeout=1200.0) as c:
            r = c.post("/scorecard", params=params, files=files)
            if r.status_code != 200:
                return {"ok": False, "status_code": r.status_code,
                        "detail": r.text[:1000]}
            body = r.json() or {}
            meta = body.get("metadata") or {}
            return {"ok": True,
                    "call_id":     meta.get("call_id"),
                    "duration_sec": (body.get("timings_sec") or {}).get("total"),
                    "language":    meta.get("language")}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": repr(exc)}


def delete_call(call_id: str, purge_files: bool = True) -> Dict[str, Any]:
    """DELETE /admin/call/{call_id} — right-to-be-forgotten. Removes the call
    row (+ cascaded utterances / events), its pipeline_runs rows, and the
    on-disk report / audio files."""
    try:
        with httpx.Client(base_url=config.SER_API_URL, timeout=30.0) as c:
            r = c.delete(f"/admin/call/{call_id}",
                         params={"purge_files": str(purge_files).lower()})
            if r.status_code != 200:
                return {"ok": False, "status_code": r.status_code,
                        "detail": r.text[:500]}
            return r.json()
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": repr(exc)}



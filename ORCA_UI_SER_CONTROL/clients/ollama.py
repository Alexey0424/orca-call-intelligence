"""Ollama HTTP client.

Two flavors:

* ``list_models()`` / ``model_names()`` — cheap, used by the Models page
  to mark which dropdown options are already pulled (``✓ installed``).
* ``pull(name)`` — kicks off ``POST /api/pull`` and streams progress
  back. Ollama's pull endpoint sends a sequence of JSON lines like:

      {"status": "pulling manifest"}
      {"status": "pulling 6a0746a1ec1a...", "digest": "...",
       "total": 4108916992, "completed": 1234567}
      {"status": "verifying sha256 digest"}
      {"status": "writing manifest"}
      {"status": "success"}

  We yield a normalized dict the UI can render: ``{phase, percent,
  total_bytes, completed_bytes, raw}``.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterator, List

import httpx

import config

logger = logging.getLogger(__name__)


def list_models() -> List[Dict[str, Any]]:
    """Returns ``[{"name": "...", "size": int, "modified_at": "..."}, ...]``.

    Empty list on any HTTP failure.
    """
    try:
        with httpx.Client(timeout=config.OLLAMA_TIMEOUT_SEC) as c:
            r = c.get(f"{config.OLLAMA_HOST}/api/tags")
            r.raise_for_status()
            return (r.json() or {}).get("models") or []
    except Exception as exc:  # noqa: BLE001
        logger.warning("ollama_tags_failed err=%r", exc)
        return []


def model_names() -> List[str]:
    """Just the names — convenience for dropdowns."""
    return [m.get("name") for m in list_models() if m.get("name")]


# ── Pull (streaming) ───────────────────────────────────────────────────────
# Ollama's /api/pull is a long-lived response that emits one JSON object
# per status update (the digest-level updates carry total/completed byte
# counters). We surface that as a generator of normalized events the
# route handler can re-emit as Server-Sent Events.

def pull(name: str) -> Iterator[Dict[str, Any]]:
    """Yield normalized ``{phase, percent, total, completed, raw}`` events.

    Long timeouts because a fresh 14B pull can take 10+ minutes on a
    typical home connection. The Ollama server itself doesn't time out
    on the stream.
    """
    if not name:
        yield {"phase": "error", "error": "empty model name", "percent": 0}
        return

    url = f"{config.OLLAMA_HOST}/api/pull"
    # ``stream=True`` keeps the connection open; ``insecure=False`` since
    # we're talking to a sibling container on the docker network.
    timeout = httpx.Timeout(connect=10.0, read=None, write=10.0, pool=10.0)
    try:
        with httpx.stream("POST", url,
                          json={"name": name, "stream": True},
                          timeout=timeout) as r:
            if r.status_code != 200:
                body = r.read().decode("utf-8", errors="replace")[:500]
                yield {"phase": "error",
                        "error": f"HTTP {r.status_code}: {body}",
                        "percent": 0}
                return
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                yield _normalize_pull_event(obj)
    except Exception as exc:  # noqa: BLE001
        yield {"phase": "error", "error": repr(exc), "percent": 0}


def _normalize_pull_event(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Convert one /api/pull JSON line into a UI-friendly event."""
    status = (obj.get("status") or "").strip()
    total = obj.get("total")
    completed = obj.get("completed")
    percent = None
    if isinstance(total, (int, float)) and total > 0 and \
       isinstance(completed, (int, float)):
        percent = max(0.0, min(100.0, completed / total * 100.0))

    phase = "info"
    if status == "success":
        phase = "done"
    elif total is not None:
        phase = "downloading"
    elif "manifest" in status:
        phase = "manifest"
    elif "verif" in status:
        phase = "verifying"

    return {
        "phase":     phase,
        "status":    status,
        "percent":   round(percent, 1) if percent is not None else None,
        "total":     int(total) if isinstance(total, (int, float)) else None,
        "completed": int(completed) if isinstance(completed, (int, float)) else None,
    }

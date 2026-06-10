"""Talk to the Docker Engine over its Unix domain socket.

Used by the control center for three things:
  * Read container state for the ``call-analysis-api`` service (started_at, status,
    Health.Status) → drives the Status card.
  * Restart ``call-analysis-api`` after an audio-model swap.
  * (System page) container list for the rest of the stack.

We avoid the heavyweight ``docker`` SDK to keep the image tiny — the
Docker Engine API is a plain JSON-over-HTTP API and httpx handles UDS
natively.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx

import config

logger = logging.getLogger(__name__)


def _client(timeout: float = 5.0) -> httpx.Client:
    transport = httpx.HTTPTransport(uds=config.DOCKER_SOCKET_PATH)
    # The host part is ignored when uds= is set, but must be non-empty.
    return httpx.Client(transport=transport,
                        base_url="http://docker",
                        timeout=timeout)


def container_state(name: str) -> Dict[str, Any]:
    """Returns ``{"status":..., "started_at":..., "health":..., "raw":...}``.

    On any error returns ``{"ok": False, "error": "..."}``.
    """
    try:
        with _client() as c:
            r = c.get(f"/containers/{name}/json")
            r.raise_for_status()
            body = r.json() or {}
            state = body.get("State") or {}
            health = (state.get("Health") or {}).get("Status")
            return {
                "ok":          True,
                "name":        name,
                "status":      state.get("Status"),
                "running":     bool(state.get("Running")),
                "started_at":  state.get("StartedAt"),
                "finished_at": state.get("FinishedAt"),
                "restart_count": body.get("RestartCount"),
                "health":      health,
                "image":       (body.get("Config") or {}).get("Image"),
                "exit_code":   state.get("ExitCode"),
            }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "name": name, "error": repr(exc)}


def restart(name: str, timeout_sec: int = 60) -> Dict[str, Any]:
    """POST /containers/{name}/restart?t=<seconds>. Returns the action result."""
    try:
        with _client(timeout=timeout_sec + 10) as c:
            r = c.post(f"/containers/{name}/restart",
                       params={"t": timeout_sec})
            # 204 No Content on success.
            if r.status_code in (204, 200):
                return {"ok": True, "name": name}
            return {"ok": False, "name": name,
                    "status_code": r.status_code, "body": r.text}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "name": name, "error": repr(exc)}


def list_containers() -> List[Dict[str, Any]]:
    """Lightweight projection of all containers — for the System page."""
    try:
        with _client() as c:
            r = c.get("/containers/json", params={"all": "true"})
            r.raise_for_status()
            out: List[Dict[str, Any]] = []
            for c_ in r.json() or []:
                names = c_.get("Names") or []
                primary = (names[0] if names else "").lstrip("/")
                out.append({
                    "name":   primary,
                    "image":  c_.get("Image"),
                    "state":  c_.get("State"),
                    "status": c_.get("Status"),
                })
            return out
    except Exception as exc:  # noqa: BLE001
        logger.warning("docker_list_failed err=%r", exc)
        return []

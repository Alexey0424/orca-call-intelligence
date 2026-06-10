"""Trigger the n8n SER pipeline via a webhook URL.

To use:
  1. In the n8n workflow editor, add a "Webhook" trigger node ahead of
     the existing manual-trigger / file-read chain. Give it a path like
     ``orca-trigger-ser``. n8n will give you the webhook URL.
  2. Set N8N_WEBHOOK_URL in this container's environment to that URL
     (typically ``http://n8n:5678/webhook/orca-trigger-ser`` for the
     service-network address).

If N8N_WEBHOOK_URL is unset, the UI's "Run pipeline" button is shown
disabled with a hover tooltip.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)


def webhook_url() -> Optional[str]:
    url = (os.getenv("N8N_WEBHOOK_URL") or "").strip()
    return url or None


def trigger() -> Dict[str, Any]:
    """POST an empty body to the configured webhook URL.

    Returns ``{"ok": bool, ...}``. Callers should not pass user input
    here — the URL is operator-configured at deploy time.
    """
    url = webhook_url()
    if not url:
        return {"ok": False, "error": "N8N_WEBHOOK_URL is not configured"}
    try:
        with httpx.Client(timeout=10.0) as c:
            r = c.post(url, json={})
            return {"ok": r.status_code < 400,
                    "status_code": r.status_code,
                    "body": r.text[:500]}
    except Exception as exc:  # noqa: BLE001
        logger.warning("n8n_trigger_failed url=%s err=%r", url, exc)
        return {"ok": False, "error": repr(exc)}

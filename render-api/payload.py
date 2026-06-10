"""Inbound payload normalisation.

n8n sometimes wraps the SER scorecard inside ``{ "data": ... }``, sometimes
ships it as a single-element list, sometimes as the raw dict. This module
collapses all of those into the canonical ``analysis`` dict that has a
``scorecard`` key.
"""
from __future__ import annotations

from typing import Any, Dict

from fastapi import HTTPException


_WRAPPER_KEYS = ("analysis", "result", "data", "payload", "body")


def unwrap(payload: Any) -> Dict[str, Any]:
    """Return the canonical analysis dict from any of the accepted shapes."""
    if isinstance(payload, list):
        if not payload:
            raise HTTPException(status_code=400, detail="Received an empty list.")
        if isinstance(payload[0], dict):
            return payload[0]
        raise HTTPException(status_code=400,
                            detail="List payload must contain a dict at index 0.")

    if isinstance(payload, dict):
        for key in _WRAPPER_KEYS:
            inner = payload.get(key)
            if isinstance(inner, dict) and inner.get("scorecard"):
                return inner
            if (isinstance(inner, list) and inner
                    and isinstance(inner[0], dict)
                    and inner[0].get("scorecard")):
                return inner[0]
        return payload

    raise HTTPException(status_code=400, detail="Unsupported payload shape.")

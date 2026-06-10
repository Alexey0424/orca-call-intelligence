"""Pydantic models for ``/render``.

The request payload is intentionally permissive (``Dict[str, Any]``)
because it can arrive in several wrapped shapes — see ``payload.unwrap``.
The response shape is locked.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel


class RenderFiles(BaseModel):
    pdf_path:        Optional[str] = None
    html_path:       Optional[str] = None
    input_json_path: Optional[str] = None
    pdf_url:         Optional[str] = None
    html_url:        Optional[str] = None


class RenderResponse(BaseModel):
    ok:                bool = True
    service:           str
    render_id:         str
    call_id:           str
    report_title:      str
    executive_summary: Dict[str, Any]
    files:             RenderFiles

"""Chat-image dashboard renderer.

The ``/dashboard.png`` endpoint pipes through this module:

  1. ``screenshot(name, closer_name, date_range)`` builds the view-model via
     :mod:`dashboard_data` (parameterised SQL against ``calls``).
  2. POSTs that view-model to ``render-api``'s ``/dashboard`` to receive
     the URL of a static, custom-designed HTML page.
  3. Calls Gotenberg's ``/forms/chromium/screenshot/url`` against that HTML
     to produce the final PNG bytes.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import httpx

from ragcore import config

logger = logging.getLogger(__name__)


def _render_dashboard_html(view_model: Dict[str, Any]) -> str:
    """POST a view-model to render-api; return the served HTML URL.

    The returned URL is rewritten to the service-network host
    (``http://render-api:8010/...``) so Gotenberg can fetch it from
    inside the docker network.
    """
    r = httpx.post(
        f"{config.SER_RENDER_URL}/dashboard",
        json=view_model,
        timeout=config.SER_RENDER_TIMEOUT_SEC,
    )
    r.raise_for_status()
    body = r.json()
    html_url: str = body.get("html_url", "")
    if not html_url:
        raise RuntimeError(
            f"render-api returned no html_url (body={body!r})"
        )
    # render-api advertises PUBLIC_BASE_URL (a browser-facing host:port).
    # Swap that base for the service-network URL so Gotenberg can resolve
    # it inside the docker network — works for any PUBLIC_BASE_URL.
    u = urlparse(html_url)
    html_url = (config.SER_RENDER_URL.rstrip("/") + u.path
                + (f"?{u.query}" if u.query else ""))
    return html_url


def screenshot(name: str, closer_name: Optional[str] = None,
               date_range: str = "7d") -> bytes:
    """Render the design dashboard and return PNG bytes.

    ``name='overview'`` selects the team scope; ``name='closer'`` (with
    ``closer_name``) selects a single closer. ``date_range`` is one of
    ``today | yesterday | 7d | 30d | mtd | last_month | all``.
    """
    # Local import: dashboard_data imports ``db``, which we don't want
    # loaded just because the module was imported.
    import dashboard_data

    scope = "closer" if (name == "closer" and closer_name) else "team"
    view_model = dashboard_data.build_view_model(
        scope=scope, date_range=date_range, closer_name=closer_name,
    )
    html_url = _render_dashboard_html(view_model)
    logger.info("dashboard_screenshot url=%s scope=%s range=%s",
                html_url, scope, date_range)

    files = {
        "url":            (None, html_url),
        # Native canvas is 1200px wide; omitting ``height`` lets Gotenberg
        # auto-size to the natural page height (full-page capture) so the
        # PNG isn't padded with empty paper at the bottom. Empirically
        # comes out to ~1200×1900-2100 vs the old fixed 1200×2400.
        "width":          (None, "1200"),
        # Static HTML — no async fetches. A short delay lets Google Fonts
        # settle before capture.
        "waitDelay":      (None, "1s"),
        "format":         (None, "png"),
        "omitBackground": (None, "false"),
    }
    r = httpx.post(
        f"{config.GOTENBERG_URL}/forms/chromium/screenshot/url",
        files=files, timeout=config.GOTENBERG_TIMEOUT_SEC,
    )
    r.raise_for_status()
    return r.content

"""FastAPI entrypoint for ``render-api``.

Exposes ``GET /``, ``GET /health``, ``POST /render``. The render flow is:

    payload  →  payload.unwrap         (normalise envelope shape)
             →  exec_view.build        (executive-view dict)
             →  charts.build_bundle    (six SVG strings)
             →  charts.flush_to_disk   (write to assets/)
             →  rendering.render_html  (Jinja2 + report.html)
             →  rendering.render_pdf   (WeasyPrint over the HTML)
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import charts
import config
import dashboard as dashboard_module
import exec_view
import logging_setup
import payload as payload_module
import rendering
from utils import public_url, slugify

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logging_setup.configure()
    logger.info("starting service=%s output_dir=%s", config.APP_TITLE, config.OUTPUT_DIR)
    yield
    logger.info("shutdown")


app = FastAPI(title=config.APP_TITLE, version=config.APP_VERSION, lifespan=lifespan)
app.mount("/reports", StaticFiles(directory=str(config.OUTPUT_DIR)), name="reports")

# Dashboards live in a sibling directory so /reports listings stay clean.
_DASHBOARDS_DIR = config.OUTPUT_DIR.parent / "dashboards"
_DASHBOARDS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/dashboards", StaticFiles(directory=str(_DASHBOARDS_DIR)), name="dashboards")


def _dashboard_public_url(path: Path) -> str:
    relative = path.relative_to(_DASHBOARDS_DIR).as_posix()
    return f"{config.PUBLIC_BASE_URL}/dashboards/{relative}"


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "ok": True,
        "service":             config.APP_TITLE,
        "version":             config.APP_VERSION,
        "render_endpoint":     "/render",
        "dashboard_endpoint":  "/dashboard",
        "health_endpoint":     "/health",
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "service":         config.APP_TITLE,
        "version":         config.APP_VERSION,
        "output_dir":      str(config.OUTPUT_DIR),
        "public_base_url": config.PUBLIC_BASE_URL,
    }


@app.post("/render")
def render_report(
    payload:         Any  = Body(...),
    output_format:   str  = Query("both", pattern="^(pdf|html|both)$"),
    download:        bool = Query(False),
    save_input_json: bool = Query(True),
) -> Any:
    analysis = payload_module.unwrap(payload)
    if not isinstance(analysis, dict) or not analysis.get("scorecard"):
        raise HTTPException(status_code=400,
                            detail="Body must contain a SER scorecard payload.")

    render_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    render_dir = config.OUTPUT_DIR / render_id
    assets_dir = render_dir / "assets"
    render_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)
    logger.info("render_start render_id=%s call_id=%s",
                render_id, analysis.get("metadata", {}).get("call_id", "?"))

    if save_input_json:
        (render_dir / "input.json").write_text(
            json.dumps(analysis, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    executive   = exec_view.build(analysis)
    chart_svgs  = charts.build_bundle(executive)
    chart_paths = charts.flush_to_disk(chart_svgs, assets_dir)

    slug = slugify(executive["report_title"])
    pdf_path  = render_dir / f"{slug}.pdf"
    html_path = render_dir / f"{slug}.html"

    rendering.render_html(html_path=html_path, executive=executive,
                          chart_paths=chart_paths, render_id=render_id)
    if output_format in {"pdf", "both"}:
        rendering.render_pdf(pdf_path=pdf_path, html_path=html_path)

    response: Dict[str, Any] = {
        "ok":                True,
        "service":           config.APP_TITLE,
        "render_id":         render_id,
        "call_id":           executive["call_id"],
        "report_title":      executive["report_title"],
        "executive_summary": executive["summary_json"],
        "files": {
            "pdf_path":        str(pdf_path)  if pdf_path.exists()  else None,
            "html_path":       str(html_path) if html_path.exists() else None,
            "input_json_path": str(render_dir / "input.json") if save_input_json else None,
            "pdf_url":         public_url(pdf_path)  if pdf_path.exists()  else None,
            "html_url":        public_url(html_path) if html_path.exists() else None,
        },
    }

    logger.info("render_done render_id=%s pdf=%s html=%s",
                render_id, pdf_path.exists(), html_path.exists())

    if download:
        if not pdf_path.exists():
            raise HTTPException(
                status_code=400,
                detail="download=true requires output_format=pdf or both.")
        return FileResponse(str(pdf_path), media_type="application/pdf",
                            filename=pdf_path.name)
    return response


@app.post("/dashboard")
def render_dashboard(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """Render a design dashboard view-model to static HTML.

    Accepts the dashboard data contract (kpis, calls_per_day, trio,
    leaderboard, recent_calls, …) and returns ``{ok, render_id, html_url,
    html_path}``. The HTML is sized for a 1200px Gotenberg screenshot.
    """
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Body must be a JSON object.")

    render_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    render_dir = _DASHBOARDS_DIR / render_id
    render_dir.mkdir(parents=True, exist_ok=True)

    html_path = render_dir / "dashboard.html"
    json_path = render_dir / "input.json"

    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    dashboard_module.render_dashboard_html(
        html_path=html_path, data=payload, render_id=render_id,
    )

    response = {
        "ok":        True,
        "service":   config.APP_TITLE,
        "render_id": render_id,
        "scope":     payload.get("scope_chip_text") or payload.get("scope"),
        "html_url":  _dashboard_public_url(html_path),
        "html_path": str(html_path),
        "input_json_path": str(json_path),
    }
    logger.info("dashboard_render_done render_id=%s scope=%r",
                render_id, response["scope"])
    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010)

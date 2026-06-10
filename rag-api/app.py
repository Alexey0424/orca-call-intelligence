"""rag-api — CPU read/serve path: semantic search, dashboard, on-demand report."""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response

import dashboards
import logging_setup
import report
import retrieval
from ragcore import config

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logging_setup.configure()
    logger.info("starting service=rag-api render_url=%s", config.SER_RENDER_URL)
    yield
    logger.info("shutdown")


app = FastAPI(title="rag-api", version="1.0.0", lifespan=lifespan)


@app.get("/")
def root() -> Dict[str, Any]:
    return {"ok": True, "service": "rag-api",
            "endpoints": ["/search", "/dashboard", "/dashboard.png",
                          "/report/{call_id}", "/health"]}


@app.get("/health")
def health() -> Dict[str, Any]:
    db_ok = False
    try:
        from ragcore import db
        with db.connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                db_ok = cur.fetchone()[0] == 1
    except Exception as exc:  # noqa: BLE001
        logger.warning("health_db_unreachable err=%r", exc)
    return {"ok": True, "service": "rag-api", "db": db_ok}


@app.get("/search")
def search_endpoint(
    q:              str           = Query(..., min_length=1),
    top_k:          int           = Query(5, ge=1, le=50),
    min_similarity: float         = Query(0.0, ge=0.0, le=1.0),
    scope:          str           = Query("calls", pattern="^(calls|utterances)$"),
    role:           Optional[str] = Query(None, pattern="^(closer|client)$"),
) -> Dict[str, Any]:
    if scope == "calls":
        hits = retrieval.search_calls(query=q, top_k=top_k, min_similarity=min_similarity)
    else:
        hits = retrieval.search_utterances(query=q, top_k=top_k,
                                           min_similarity=min_similarity, role=role)
    for h in hits:
        for k, v in list(h.items()):
            if hasattr(v, "isoformat"):
                h[k] = v.isoformat()
            elif v.__class__.__name__ == "Decimal":
                h[k] = float(v)
    return {"ok": True, "scope": scope, "query": q, "results": hits}


def _config_public_base() -> str:
    return os.getenv("RAG_API_PUBLIC_URL",
                     os.getenv("SER_API_PUBLIC_URL", "http://localhost:8002")).rstrip("/")


def _dashboard_image_url(name: str, closer_name: Optional[str], date_range: str) -> str:
    import urllib.parse
    qs = f"?name={name}&date_range={date_range}"
    if closer_name:
        qs += "&closer_name=" + urllib.parse.quote(closer_name)
    # Public URL of THIS service's /dashboard.png. RAG_API_PUBLIC_URL falls back
    # to SER_API_PUBLIC_URL so existing chat embeds keep resolving.
    base = _config_public_base()
    return f"{base}/dashboard.png{qs}"


@app.get("/dashboard")
def dashboard_endpoint(
    name:        str           = Query("overview", pattern="^(overview|closer)$"),
    closer_name: Optional[str] = Query(None),
    date_range:  str           = Query("7d",
                                       pattern="^(today|yesterday|7d|30d|mtd|last_month|all)$"),
) -> Dict[str, Any]:
    import dashboard_data
    scope = "closer" if (name == "closer" and closer_name) else "team"
    vm = dashboard_data.build_view_model(scope=scope, date_range=date_range,
                                         closer_name=closer_name)
    kpis_by_label = {k["label"]: k for k in vm.get("kpis", [])}
    key_metrics = {
        "total_calls":        kpis_by_label.get("Total Calls", {}).get("value"),
        "calls_in_period":    next((k.get("value") for k in vm.get("kpis", [])
                                    if k["label"].startswith("Calls · ")), None),
        "next_step_rate_pct": kpis_by_label.get("Next-Step Rate", {}).get("value"),
        "active_days":        next((s["value"] for s in
                                    vm["calls_per_day"]["side_stats"]
                                    if s["label"] == "Active Days"), None),
        "scope":              vm.get("scope_chip_text"),
    }
    return {"ok": True, "image_url": _dashboard_image_url(name, closer_name, date_range),
            "title": vm.get("doc_title", "ORCA Dashboard"),
            "scope": vm.get("scope_chip_text"), "date_range": date_range,
            "key_metrics": key_metrics}


@app.get("/dashboard.png")
def dashboard_png_endpoint(
    name:        str           = Query("overview", pattern="^(overview|closer)$"),
    closer_name: Optional[str] = Query(None),
    date_range:  str           = Query("7d",
                                       pattern="^(today|yesterday|7d|30d|mtd|last_month|all)$"),
) -> Response:
    try:
        png = dashboards.screenshot(name, closer_name, date_range)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.error("dashboard_png_failed name=%s error=%r", name, exc)
        raise HTTPException(status_code=502, detail=f"screenshot service error: {exc}")
    return Response(content=png, media_type="image/png")


@app.get("/report/{call_id}")
def report_endpoint(call_id: str) -> Dict[str, Any]:
    out = report.get_or_render(call_id)
    if not out.get("ok"):
        err = out.get("error", "render_failed")
        code = 404 if "not_found" in err else 502
        raise HTTPException(status_code=code, detail=err)
    return out

"""ORCA_UI_SER_CONTROL — FastAPI app.

Three concerns:
  * Observability  — render the Status/History page (all jobs, live elapsed
                     for running rows, aggregate metrics card).
  * Steering       — accept POSTs that mutate the runtime-overrides file
                     and optionally restart ser-api.
  * Drill-down     — render a per-call view (transcript, sentiment/emotion
                     timeline, KPI cards, link to PDF / HTML report).

HTMX is used for the live-polling jobs table and the service health card
so the rest of the page never re-renders. Everything else is plain
server-rendered HTML.
"""
from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, Query, Request, UploadFile
from fastapi.responses import (HTMLResponse, JSONResponse, RedirectResponse,
                                 StreamingResponse)
from fastapi.templating import Jinja2Templates

import config
import logging_setup
from clients import (db_browser, db_client, docker_client, n8n as n8n_client,
                     ollama as ollama_client, overrides, ser_api)


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logging_setup.configure()
    logger.info("orca_ui_ser_control_starting ser_api_url=%s db_url_set=%s",
                config.SER_API_URL, bool(config.DATABASE_URL))
    # Ensure the override file's directory exists so first POST doesn't
    # blow up trying to create the temp file.
    try:
        config.RUNTIME_OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("overrides_dir_mkdir_failed err=%r", exc)
    yield
    logger.info("orca_ui_ser_control_stopped")


app = FastAPI(title="ORCA_UI_SER_CONTROL", lifespan=lifespan)
templates = Jinja2Templates(directory="templates")


# ── Jinja filters ───────────────────────────────────────────────────────────
def _fmt_seconds(value: Any) -> str:
    """Compact human-friendly seconds: 0.8s, 12s, 1m 04s, 1h 12m."""
    try:
        s = float(value)
    except (TypeError, ValueError):
        return "—"
    if s < 1:
        return f"{s:.2f}s"
    if s < 60:
        return f"{int(s)}s"
    if s < 3600:
        m, sec = divmod(int(s), 60)
        return f"{m}m {sec:02d}s"
    h, rem = divmod(int(s), 3600)
    m, _ = divmod(rem, 60)
    return f"{h}h {m:02d}m"


def _short_model(name: Any) -> str:
    """`org/repo` → `repo`; `name:tag` → `name:tag`. Keeps tables tidy."""
    if not isinstance(name, str) or not name:
        return "—"
    if "/" in name:
        return name.split("/", 1)[1]
    return name


def _status_class(status: Any) -> str:
    return {
        "running": "bg-amber-100 text-amber-900",
        "done":    "bg-emerald-100 text-emerald-900",
        "failed":  "bg-rose-100 text-rose-900",
    }.get(str(status), "bg-slate-100 text-slate-700")


def _fmt_ts(value: Any) -> str:
    """Format a timestamp as "Sat, 20 May 2026 · 2:31 PM UTC".

    Accepts both ISO-8601 strings (from ser-api JSON) and ``datetime``
    objects (from psycopg2 reads). Anything else → "—".
    """
    if value is None or value == "":
        return "—"
    dt = None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        dt = _parse_iso(value)
    if dt is None:
        return "—"
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    day  = dt.strftime("%d").lstrip("0") or "0"
    hour = dt.strftime("%I").lstrip("0") or "12"
    return dt.strftime(f"%a, {day} %b %Y · {hour}:%M %p UTC")


def _fmt_size(mb: Any) -> str:
    """MB → '~140 MB' / '~1.5 GB'. 0 / None → ''."""
    try:
        n = int(mb or 0)
    except (TypeError, ValueError):
        return ""
    if n <= 0:
        return ""
    if n < 1024:
        return f"~{n} MB"
    return f"~{n/1024:.1f} GB"


def _fmt_uptime(value: Any) -> str:
    """Compact uptime since an ISO timestamp: '4d 12h', '1h 23m', '5m 12s'."""
    if value is None or value == "":
        return "—"
    dt = value if isinstance(value, datetime) else _parse_iso(value)
    if dt is None:
        return "—"
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    delta = (datetime.now(timezone.utc) - dt).total_seconds()
    if delta < 0:
        return "—"
    if delta < 60:
        return f"{int(delta)}s"
    if delta < 3600:
        m, s = divmod(int(delta), 60)
        return f"{m}m {s:02d}s"
    if delta < 86400:
        h, rem = divmod(int(delta), 3600)
        m, _ = divmod(rem, 60)
        return f"{h}h {m:02d}m"
    d, rem = divmod(int(delta), 86400)
    h, _ = divmod(rem, 3600)
    return f"{d}d {h:02d}h"


def _fmt_pct_pair(triple: Any) -> str:
    """Render an ownership metric (VRAM/CPU/RAM) as a 'rel/glob' pair, e.g.
    '41/78' — both % of capacity. rel = ser-api's own share, glob = the whole
    box (all processes); the difference is everything else, mostly Ollama.
    A missing side shows as '·'; an absent triple shows as '—'.
    (GPU compute % is whole-card only and uses _fmt_pct_solo instead.)"""
    if not isinstance(triple, dict):
        return "—"
    rel, glob = triple.get("rel"), triple.get("glob")
    if rel is None and glob is None:
        return "—"
    fmt = lambda v: "·" if v is None else f"{v:.0f}"
    return f"{fmt(rel)}/{fmt(glob)}"


def _fmt_pct_solo(triple: Any) -> str:
    """Render a global-only metric (e.g. GPU compute util) as a single integer
    '%'. Uses glob (falls back to rel); '—' if absent."""
    if not isinstance(triple, dict):
        return "—"
    v = triple.get("glob")
    if v is None:
        v = triple.get("rel")
    return "—" if v is None else f"{v:.0f}"


templates.env.filters["fmt_seconds"]  = _fmt_seconds
templates.env.filters["fmt_ts"]       = _fmt_ts
templates.env.filters["fmt_size"]     = _fmt_size
templates.env.filters["fmt_uptime"]   = _fmt_uptime
templates.env.filters["short_model"]  = _short_model
templates.env.filters["status_class"] = _status_class
templates.env.filters["fmt_pct_pair"] = _fmt_pct_pair
templates.env.filters["fmt_pct_solo"] = _fmt_pct_solo


# ── Helpers ─────────────────────────────────────────────────────────────────
def _display_status(job: Dict[str, Any]) -> str:
    """``running + stage='queued'`` → "pending" badge."""
    if job.get("status") == "running" and (job.get("stage") or "") == "queued":
        return "pending"
    return job.get("status") or "—"


def _display_time(job: Dict[str, Any]) -> float:
    """Pick the right elapsed value per the user's spec.

    running → live_elapsed_sec    (NOW - started_at, from SQL)
    done    → duration_sec        (finalised at jobs.finish)
    failed  → duration_sec        (also finalised at jobs.finish)
    """
    if job.get("status") == "running":
        return job.get("live_elapsed_sec") or 0.0
    return job.get("duration_sec") or 0.0


def _decorate_jobs(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for r in rows:
        r["display_status"] = _display_status(r)
        r["display_time"]   = _display_time(r)
        models = ((r.get("metadata") or {}).get("models") or {})
        r["display_models"] = {
            "asr":       models.get("asr"),
            "llm":       models.get("llm"),
            "embedding": models.get("embedding"),
            "sentiment": models.get("sentiment"),
            "ser":       models.get("ser"),
        }
        r["display_file"] = ((r.get("metadata") or {}).get("file_name") or "—")
    return rows


# ── Pipeline stage definition (drives the control-panel pipeline diagram) ──
# ser-api emits: queued → convert → asr → diarize → scorecard → persist → done.
# We DISPLAY only the 5 work stages — `queued` and `done` are redundant with
# the status badge and carry no duration/peaks. Their timestamps still live in
# metadata.stage_times and anchor the duration math (persist runs up to done_ts).
PIPELINE_STAGES = ["convert", "asr", "diarize", "scorecard", "persist"]


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        # ser-api/jobs.py writes ISO-8601 with millisecond precision + Z.
        # datetime.fromisoformat handles +HH:MM but pre-3.11 chokes on a
        # trailing Z — strip it defensively.
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        return datetime.fromisoformat(ts)
    except Exception:  # noqa: BLE001
        return None


def _max_peaks(peaks_list: List[Optional[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    """Element-wise max of several {vram,cpu,ram:{rel,glob,other}} peak dicts.
    Summarises a stage from its sub-stages when there's no explicit stage peak
    (e.g. asr/diarize/scorecard/persist parent rows). None if no sub carried peaks."""
    present = [p for p in peaks_list if isinstance(p, dict)]
    if not present:
        return None
    out: Dict[str, Any] = {}
    for metric in ("gpu", "vram", "cpu", "ram"):
        rels  = [p[metric]["rel"]  for p in present
                 if isinstance(p.get(metric), dict) and p[metric].get("rel")  is not None]
        globs = [p[metric]["glob"] for p in present
                 if isinstance(p.get(metric), dict) and p[metric].get("glob") is not None]
        r = max(rels)  if rels  else None
        g = max(globs) if globs else None
        o = round(g - r, 1) if (r is not None and g is not None) else None
        out[metric] = {"rel": r, "glob": g, "other": o}
    return out


def _stage_progress(job: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-stage status + duration for the control panel.

    For each stage in PIPELINE_STAGES, returns:
      name        — stage label
      state       — 'done' | 'active' | 'failed' | 'pending'
      duration_sec — seconds spent in this stage (None if never reached)
      live        — True for the currently-running stage whose end has
                    not been observed yet (UI ticks it up live).

    Duration sources, in order of preference:
      1. metadata.stage_times[stage]    → start of this stage
         metadata.stage_times[next]     → start of next stage = end of this
      2. Currently-running last stage: end = NOW (live tick)
      3. Terminal stage on done/failed: end = stage_times[done|failed]
         or fall back to finished_at.
    """
    if not job:
        return {"stages": [{"name": s, "state": "pending",
                             "duration_sec": None, "live": False}
                            for s in PIPELINE_STAGES],
                "current": None}

    md       = job.get("metadata") or {}
    times    = md.get("stage_times") or {}
    smodels  = md.get("stage_models") or {}
    speaks   = md.get("stage_peaks") or {}
    current  = job.get("stage")
    status   = job.get("status")

    # State machine: which stages have started? Anything with a timestamp
    # in stage_times. The CURRENT stage is the one matching `job.stage`.
    try:
        idx = PIPELINE_STAGES.index(current) if current in PIPELINE_STAGES else -1
    except ValueError:
        idx = -1

    # The pipeline also reports intermediate stages that we don't render as
    # their own column (e.g. `turn_relabel`, which sits between diarize and
    # scorecard). When the running job is in one of those, `current` isn't in
    # PIPELINE_STAGES and idx stays -1 — which would mark every started stage
    # "done" and show NOTHING as active. Instead, treat the furthest display
    # stage that has actually started as the live one.
    if idx < 0 and status == "running":
        idx = max((i for i, s in enumerate(PIPELINE_STAGES) if times.get(s)),
                  default=-1)

    # Compute end timestamp for each started stage.
    # For stage[i]: end = stage_times[stage[i+1]] if next stage started,
    # otherwise NOW (running) or finished_at (done/failed).
    now_iso = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    finish_ts = times.get("done") or times.get("failed") or job.get("finished_at")

    stages_out: List[Dict[str, Any]] = []
    for i, s in enumerate(PIPELINE_STAGES):
        start_ts = times.get(s)

        # Classification:
        if idx < 0 and not start_ts:
            state = "pending"
        elif start_ts is None:
            state = "pending"
        elif i < idx:
            state = "done"
        elif i == idx:
            if status == "failed":
                state = "failed"
            elif status == "done":
                state = "done"
            else:
                state = "active"
        else:
            # We have a start_ts but idx says we're past — fall back to done.
            state = "done"

        # If the whole job is done, treat every recorded stage as done.
        if status == "done" and start_ts is not None:
            state = "done"

        # Pick end-of-stage timestamp.
        next_start = None
        for j in range(i + 1, len(PIPELINE_STAGES)):
            cand = times.get(PIPELINE_STAGES[j])
            if cand:
                next_start = cand
                break

        if state == "pending" or start_ts is None:
            duration = None
            live = False
        elif next_start:
            duration = _seconds_between(start_ts, next_start)
            live = False
        elif state == "active":
            duration = _seconds_between(start_ts, now_iso)
            live = True
        else:
            # Terminal stage on done/failed: fall back to finish timestamp.
            end_ts = finish_ts if isinstance(finish_ts, str) else None
            duration = _seconds_between(start_ts, end_ts) if end_ts else None
            live = False

        # Sub-models that ran in this stage (with their individual timings).
        # Decorate each with its approximate on-disk size for display.
        models_in_stage = []
        for entry in (smodels.get(s) or []):
            if entry.get("kind") == "concurrent":
                # A concurrent group: one shared peak (kept on the entry),
                # per-member individual times. Decorate each member with size.
                members = [
                    {**mem, "size_mb": config.model_size_mb(mem.get("model") or "")}
                    for mem in (entry.get("members") or [])
                ]
                models_in_stage.append({**entry, "members": members})
            else:
                mid = entry.get("model") or ""
                models_in_stage.append({
                    **entry,
                    "size_mb": config.model_size_mb(mid),
                })

        # Parent-row peaks: an explicit stage peak (convert) if present,
        # otherwise the element-wise max across the stage's sub-models.
        stage_peaks = speaks.get(s) or _max_peaks(
            [m.get("peaks") for m in models_in_stage])
        stages_out.append({
            "name":         s,
            "state":        state,
            "start_ts":     start_ts,
            "duration_sec": duration,
            "live":         live,
            "models":       models_in_stage,
            "stage_peaks":  stage_peaks,
        })

    return {"stages": stages_out, "current": current}


def _seconds_between(a_iso: Optional[str], b_iso: Optional[str]) -> Optional[float]:
    a = _parse_iso(a_iso)
    b = _parse_iso(b_iso)
    if a is None or b is None:
        return None
    delta = (b - a).total_seconds()
    return max(0.0, delta)


# ── Status / Control Panel (current + history modes) ───────────────────────
@app.get("/", response_class=HTMLResponse)
def status_page(request: Request,
                mode: str = Query("current", pattern="^(current|history)$"),
                run:  Optional[int] = Query(None)):
    # Pick the job to display. Read straight from Postgres so the polling
    # keeps working while ser-api is busy serving a /scorecard.
    job: Optional[Dict[str, Any]] = None
    if mode == "history" and run is not None:
        job = db_client.job_by_run_no(run)
    if job is None:
        job = db_client.job_current()
        if mode == "history" and run is not None and job is None:
            mode = "history-missing"

    job_decorated = _decorate_jobs([job])[0] if job else None
    progress = _stage_progress(job)
    history = db_client.run_index(limit=500)
    container = docker_client.container_state(config.SER_API_CONTAINER)
    health = ser_api.health()
    summary = db_client.jobs_summary()

    return templates.TemplateResponse(
        request,
        "status.html",
        {
            "active":           "status",
            "mode":             mode,
            "selected_run":     run,
            "job":              job_decorated,
            "progress":         progress,
            "history":          history,
            "summary":          summary,
            "container":        container,
            "ser_health":       health,
            "trigger_enabled":  bool(n8n_client.webhook_url()),
        },
    )


# HTMX partial — re-renders the live control panel (job + stage diagram).
@app.get("/_partial/control-panel", response_class=HTMLResponse)
def control_panel_partial(request: Request,
                          mode: str = Query("current"),
                          run:  Optional[int] = Query(None)):
    # Same DB-direct read as the parent page — keeps polling responsive
    # even while ser-api is GPU-bound on a /scorecard.
    job: Optional[Dict[str, Any]] = None
    if mode == "history" and run is not None:
        job = db_client.job_by_run_no(run)
    if job is None:
        job = db_client.job_current()
    job_decorated = _decorate_jobs([job])[0] if job else None
    progress = _stage_progress(job)
    return templates.TemplateResponse(
        request,
        "partials/_control_panel.html",
        {"job": job_decorated, "progress": progress, "mode": mode},
    )


# ── Actions ────────────────────────────────────────────────────────────────
@app.post("/control/trigger")
def trigger_pipeline():
    return n8n_client.trigger()


@app.post("/control/run-scorecard")
async def run_scorecard(file: UploadFile = File(...)):
    """Direct file upload → ser-api /scorecard (no n8n required).

    Returns the resulting call_id + duration so the UI can show a toast
    and link the user to /calls/{id} for the drill-down.
    """
    data = await file.read()
    return ser_api.run_scorecard(data, file.filename or "upload.bin")


@app.get("/control/status")
def control_status():
    """Tiny JSON probe used by the UI's button-state polling. Reads
    directly from Postgres so it stays responsive even while ser-api is
    busy serving a /scorecard."""
    job = db_client.job_current()
    if not job:
        return {"running": False}
    return {
        "running":  job.get("status") == "running",
        "run_no":   job.get("run_no"),
        "stage":    job.get("stage"),
        "elapsed":  job.get("live_elapsed_sec"),
    }


@app.post("/control/eject")
def eject_running():
    """Hard-stop whatever ser-api is processing right now.

    1. Mark any currently-running jobs as ``failed`` with last_error
       'ejected by operator'.
    2. Restart the ser-api container — this kills the in-flight request
       cleanly (otherwise the python process keeps running emotion2vec
       on its utterances even after the HTTP socket closes).

    Returns the number of jobs demoted + the restart result.
    """
    demoted = 0
    try:
        from clients import db_client as _db
        with _db.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE pipeline_runs
                       SET status = 'failed',
                           finished_at = NOW(),
                           last_error = 'ejected by operator',
                           duration_sec = COALESCE(duration_sec,
                                              EXTRACT(EPOCH FROM (NOW() - started_at))::REAL)
                     WHERE status = 'running'
                    """
                )
                demoted = cur.rowcount
            conn.commit()
    except Exception as exc:  # noqa: BLE001
        logger.warning("eject_db_failed err=%r", exc)
    restart_result = docker_client.restart(config.SER_API_CONTAINER)
    return {"ok": True, "demoted": demoted, "restart_result": restart_result}


# HTMX partial — service health card (poll every 2s).
@app.get("/_partial/service", response_class=HTMLResponse)
def service_partial(request: Request):
    container = docker_client.container_state(config.SER_API_CONTAINER)
    health = ser_api.health()
    return templates.TemplateResponse(
        request,
        "partials/_service_card.html",
        {"container": container, "ser_health": health},
    )


# ── Models page ─────────────────────────────────────────────────────────────
# Maps the four audio override keys to the registry roles ser-api exposes
# on /admin/reload-model. Used by the hot-swap path in /models POST.
_AUDIO_KEY_TO_ROLE = {
    "WHISPER_MODEL_ID":     "asr",
    "DIARIZATION_MODEL_ID": "diarization",
    "SENTIMENT_MODEL_ID":   "sentiment",
    "SER_MODEL_ID":         "ser",
}


@app.get("/models", response_class=HTMLResponse)
def models_page(request: Request):
    current_overrides = overrides.read()
    health = ser_api.health()
    ollama_installed = ollama_client.model_names()

    # Probe HF / FunASR disk-cache state once per page load. The endpoint
    # accepts CSV lists so we issue ONE request that covers every dropdown
    # option for every audio role.
    cache = ser_api.cache_status(
        asr_ids=config.ASR_CHOICES,
        diar_ids=config.DIARIZATION_CHOICES,
        sent_ids=config.SENTIMENT_CHOICES,
        ser_ids=config.SER_CHOICES,
    )

    return templates.TemplateResponse(
        request,
        "models.html",
        {
            "active":           "models",
            "ser_health":       health,
            "overrides":        current_overrides,
            "asr_choices":      config.ASR_CHOICES,
            "diar_choices":     config.DIARIZATION_CHOICES,
            "sent_choices":     config.SENTIMENT_CHOICES,
            "ser_choices":      config.SER_CHOICES,
            "llm_choices":      sorted(set(config.LLM_DEFAULT_CHOICES
                                            + ollama_installed)),
            "embed_choices":    sorted(set(config.EMBEDDING_DEFAULT_CHOICES
                                            + ollama_installed)),
            "ollama_installed": ollama_installed,
            "cache":            cache,
        },
    )


@app.post("/models")
def models_apply(
    WHISPER_MODEL_ID:    Optional[str] = Form(None),
    DIARIZATION_MODEL_ID: Optional[str] = Form(None),
    SENTIMENT_MODEL_ID:  Optional[str] = Form(None),
    SER_MODEL_ID:        Optional[str] = Form(None),
    LLM_MODEL_ID:        Optional[str] = Form(None),
    EMBEDDING_MODEL_ID:  Optional[str] = Form(None),
    OPENING_RELABEL_ENABLED: Optional[str] = Form(None),
    OPENING_RELABEL_MODEL:   Optional[str] = Form(None),
):
    """Write the overrides file, then hot-swap every changed audio role.

    Ollama models (LLM / Embedding) take effect on the next /scorecard
    request automatically — config.llm_model_id() / embedding_model_id()
    re-read the override file per call. Audio models need an explicit
    POST to ``/admin/reload-model`` per role so ser-api can evict the
    current weights + load the new ones inside its GPU lock. No restart.
    """
    updates: Dict[str, Optional[str]] = {
        "WHISPER_MODEL_ID":     WHISPER_MODEL_ID,
        "DIARIZATION_MODEL_ID": DIARIZATION_MODEL_ID,
        "SENTIMENT_MODEL_ID":   SENTIMENT_MODEL_ID,
        "SER_MODEL_ID":         SER_MODEL_ID,
        "LLM_MODEL_ID":         LLM_MODEL_ID,
        "EMBEDDING_MODEL_ID":   EMBEDDING_MODEL_ID,
        "OPENING_RELABEL_ENABLED": OPENING_RELABEL_ENABLED,
        "OPENING_RELABEL_MODEL":   OPENING_RELABEL_MODEL,
    }
    result  = overrides.write(updates)
    changed = result.get("changed") or {}

    # Walk every changed audio key and ask ser-api to hot-swap it. We do
    # this serially per role since each holds the GPU lock during the
    # swap — issuing in parallel wouldn't speed things up.
    reload_results: Dict[str, Any] = {}
    for key in changed.keys():
        role = _AUDIO_KEY_TO_ROLE.get(key)
        if role is None:
            continue   # Ollama keys hot-swap themselves; nothing to do.
        reload_results[key] = ser_api.reload_model(role)

    return JSONResponse({
        "ok":             True,
        "changed":        changed,
        "reload_results": reload_results,
        "current":        result.get("current"),
    })


@app.post("/restart-ser-api")
def restart_ser_api():
    return docker_client.restart(config.SER_API_CONTAINER)


# ── Ollama pull (Server-Sent Events stream) ────────────────────────────────
@app.post("/ollama/pull")
def ollama_pull(name: str = Form(...)):
    """Stream progress from ``POST {OLLAMA_HOST}/api/pull`` to the browser.

    Returns a Server-Sent Events stream. Each ``data:`` line is one JSON
    object the Models page renders into the per-row progress bar.
    """
    name = (name or "").strip()
    if not name:
        return JSONResponse(
            {"ok": False, "error": "empty model name"}, status_code=400,
        )

    def gen():
        # SSE preamble — tells the browser to keep the connection open.
        yield "retry: 5000\n\n"
        for evt in ollama_client.pull(name):
            yield "data: " + json.dumps(evt) + "\n\n"
            if evt.get("phase") in ("done", "error"):
                break
        yield "event: end\ndata: {}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


# ── Calls page (browser + drill-down) ───────────────────────────────────────
@app.get("/calls", response_class=HTMLResponse)
def calls_page(request: Request, limit: int = Query(100, ge=1, le=1000)):
    calls = db_client.list_calls(limit=limit)
    return templates.TemplateResponse(
        request,
        "calls.html",
        {"active": "calls", "calls": calls},
    )


@app.get("/calls/{call_id}", response_class=HTMLResponse)
def call_detail(request: Request, call_id: str):
    call = db_client.get_call(call_id)
    if call is None:
        return HTMLResponse(f"<h1>call {call_id} not found</h1>",
                             status_code=404)
    utterances = db_client.utterances_for(call_id)
    events     = db_client.events_for(call_id)

    # Bucket events by kind for the side rail.
    events_by_kind: Dict[str, List[Dict[str, Any]]] = {}
    for e in events:
        events_by_kind.setdefault(e.get("kind") or "other", []).append(e)

    return templates.TemplateResponse(
        request,
        "call_detail.html",
        {
            "active":          "calls",
            "call":            call,
            "utterances":      utterances,
            "events":          events,
            "events_by_kind":  events_by_kind,
            "pdf_url":         _public_url(call.get("pdf_url")),
            "html_url":        _public_url(call.get("html_url")),
        },
    )


@app.post("/calls/{call_id}/delete")
def delete_call_action(call_id: str):
    """Right-to-be-forgotten (F1): delete a call via ser-api, back to /calls.

    The actual deletion (rows + on-disk files) happens in ser-api's
    `DELETE /admin/call/{id}`. We log here + there so the action leaves an
    audit trail in both services' logs.
    """
    result = ser_api.delete_call(call_id)
    logger.warning("rtbf_delete_ui call_id=%s ok=%s db_rows=%s files=%s",
                   call_id, result.get("ok"), result.get("db_rows"),
                   len(result.get("files_removed") or []))
    return RedirectResponse(url="/calls", status_code=303)


def _public_url(url: Optional[str]) -> Optional[str]:
    """Rewrite internal docker-network URLs so the user's browser can fetch
    them. call-analysis-api / render-api publish links like
    ``http://render-api:8010/...`` — the host needs the host-mapped form.
    """
    if not url:
        return None
    return (url
            .replace("http://call-analysis-api:8000", config.SER_API_PUBLIC_URL)
            .replace("http://render-api:8010",         config.SER_RENDER_PUBLIC_URL))


# ── Database page (browser + SQL workbench) ─────────────────────────────────
# Directory containing the curated browse SQL surfaced as Query-tab presets.
# Mounted into the container from ``./shared/queries`` on the host; absent
# during local dev runs without the mount, which is fine — the route falls
# back to an empty preset list.
_DB_PRESETS_DIR = "/data/shared/queries"


def _load_presets() -> List[Dict[str, str]]:
    """Read ``.sql`` files from the preset dir into ``{name, description, body}``.

    Description is the first non-blank ``--`` comment line if present.
    Sorted by filename so the numbered prefix (``01_...``) controls order.
    """
    out: List[Dict[str, str]] = []
    if not os.path.isdir(_DB_PRESETS_DIR):
        return out
    try:
        names = sorted(n for n in os.listdir(_DB_PRESETS_DIR)
                       if n.lower().endswith(".sql"))
    except OSError:
        return out
    for fname in names:
        path = os.path.join(_DB_PRESETS_DIR, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                body = f.read()
        except OSError:
            continue
        description = ""
        for line in body.splitlines():
            stripped = line.strip()
            if stripped.startswith("--"):
                description = stripped.lstrip("- ").strip()
                if description:
                    break
            elif stripped:
                # Hit the first SQL line — no description.
                break
        # Strip the numeric prefix from the visible name for readability.
        nice = fname.rsplit(".", 1)[0]
        out.append({"name": nice, "description": description[:160],
                    "body": body})
    return out


@app.get("/db", response_class=HTMLResponse)
def db_page(request: Request,
            tab:      str           = Query("browser", pattern="^(browser|query)$"),
            database: Optional[str] = Query(None),
            schema:   Optional[str] = Query("public"),
            table:    Optional[str] = Query(None),
            prefill:  Optional[str] = Query(None)):
    """Single template, two tabs. Query string drives state."""
    databases = db_browser.list_databases()

    # Validate ``database`` against the allowlist so a typo'd query string
    # doesn't bubble a ValueError. ``selected_db`` is what the template
    # actually trusts.
    selected_db: Optional[str] = None
    if database and database in db_browser.ALLOWED_DBS:
        selected_db = database

    tables: List[Dict[str, Any]] = []
    table_detail: Optional[Dict[str, Any]] = None
    preview: Optional[Dict[str, Any]] = None

    if tab == "browser" and selected_db:
        try:
            tables = db_browser.list_tables(selected_db)
        except Exception as exc:  # noqa: BLE001
            logger.warning("db_list_tables_failed db=%s err=%r",
                           selected_db, exc)

        if table:
            sch = schema or "public"
            try:
                table_detail = db_browser.table_detail(selected_db, sch, table)
            except Exception as exc:  # noqa: BLE001
                logger.warning("db_table_detail_failed db=%s table=%s err=%r",
                               selected_db, table, exc)
            if table_detail:
                try:
                    preview = db_browser.preview_rows(selected_db, sch, table,
                                                      limit=50)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("db_preview_failed db=%s table=%s err=%r",
                                   selected_db, table, exc)
                    preview = {"columns": [], "rows": [],
                               "row_count": 0, "elapsed_ms": 0,
                               "error": str(exc)[:200]}

    presets = _load_presets() if tab == "query" else []

    return templates.TemplateResponse(
        request,
        "database.html",
        {
            "active":          "database",
            "tab":             tab,
            "databases":       databases,
            "selected_db":     selected_db,
            "selected_schema": schema or "public",
            "selected_table":  table,
            "tables":          tables,
            "table_detail":    table_detail,
            "preview":         preview,
            "presets":         presets,
            "prefill":         prefill or "",
        },
    )


@app.post("/db/query", response_class=HTMLResponse)
def db_query(request: Request,
             database:   str           = Form(...),
             sql:        str           = Form(""),
             write_mode: Optional[str] = Form(None)):
    """Run one SQL statement. Returns the result partial (HTMX target).

    Always renders 200 so HTMX swaps the partial regardless of success —
    errors live inside the rendered card (red border) so the user sees the
    Postgres error message exactly as Postgres produced it.
    """
    read_only = not bool(write_mode)
    if database not in db_browser.ALLOWED_DBS:
        result = {"ok": False, "columns": [], "rows": [], "row_count": 0,
                  "truncated": False, "affected": None, "elapsed_ms": 0,
                  "error": f"database not allowed: {database!r}"}
    else:
        result = db_browser.run_query(database, sql, read_only=read_only)
    return templates.TemplateResponse(
        request, "partials/_db_result.html", {"result": result},
    )


# ── Lightweight liveness probe for this container itself ───────────────────
@app.get("/healthz")
def healthz():
    return {"ok": True, "service": "orca-ui-ser-control"}

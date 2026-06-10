"""Tiny job tracker.

Each /scorecard invocation gets a ``pipeline_runs`` row recording when it
started, which stage it was in when it last checkpointed, and how it
finished. This is the cheapest possible answer to "is anything stuck?" —
no Celery / RQ / Prometheus, just an indexed table the operator can SELECT.

Each stage transition writes its UTC timestamp into
``metadata.stage_times`` (a JSONB dict like ``{"queued": "…", "asr": "…",
…}``) so the control UI can compute per-stage durations. The terminal
event (``done`` or ``failed``) is also recorded there.

Failures are caught one layer up (in app.py) so we can record both the
stage that was running and the error message.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import time
import uuid
from typing import Any, Dict, Optional

import db

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    """UTC timestamp in ISO-8601 with millisecond precision + Z suffix.

    Used as stage_times values. Postgres NOW() would be marginally more
    accurate but reading it back across a separate query introduces a
    second roundtrip, and the audio pipeline runs locally to ser-api so
    Python's clock is plenty good enough for stage timings.
    """
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="milliseconds")


def start(*, kind: str, call_id: Optional[str] = None,
          metadata: Optional[Dict[str, Any]] = None) -> str:
    """Insert a `running` job row and return its job_id.

    The ``run_no`` column (sequential integer) is assigned by the
    Postgres sequence default. Stage timing for "queued" is stamped
    here so the control UI can show how long the job sat in queued
    before convert started.

    Best-effort: a DB hiccup here returns a random job_id and logs a
    warning instead of failing the request.
    """
    job_id = uuid.uuid4().hex
    md: Dict[str, Any] = dict(metadata or {})
    md.setdefault("stage_times", {})["queued"] = _now_iso()
    try:
        with db.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO pipeline_runs (job_id, call_id, kind, status, stage,
                                                started_at, metadata)
                    VALUES (%s, %s, %s, 'running', 'queued', NOW(),
                            %s::jsonb)
                    """,
                    (job_id, call_id, kind, json.dumps(md)),
                )
            conn.commit()
    except Exception as exc:  # noqa: BLE001
        logger.warning("jobs_start_failed job_id=%s error=%r", job_id, exc)
    return job_id


def _model_entry(model: str, kind: str, duration_sec: float,
                 peaks: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build one stage_models entry (pure — unit-tested). ``peaks`` is the
    resource_sampler high-water-mark dict, omitted when None."""
    entry: Dict[str, Any] = {"model": model, "kind": kind,
                             "duration_sec": round(float(duration_sec), 3)}
    if peaks:
        entry["peaks"] = peaks
    return entry


def _concurrent_entry(members: list, duration_sec: float,
                      peaks: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build one ``stage_models`` entry for steps that ran *concurrently*
    (pure — unit-tested).

    Why this exists: when emotion (ser-api GPU), the KPI LLM (gemma4 in the
    Ollama process) and prosody (a CPU thread) run at the same time, **time
    is attributable per step** (each is measured on its own clock) but
    **resource use is NOT** — they share one physical GPU/box, so you can't
    say "emotion used 60%, the LLM used 39%". So every member keeps its own
    ``duration_sec`` while the whole group carries a SINGLE ``peaks``
    high-water mark for the window.
    """
    return {
        "kind": "concurrent",
        "duration_sec": round(float(duration_sec), 3),
        "members": [
            {"model": m["model"], "kind": m["kind"],
             "duration_sec": round(float(m["duration_sec"]), 3)}
            for m in members
        ],
        "peaks": peaks or None,
    }


def _append_stage_model(job_id: str, stage: str, entry: Dict[str, Any]) -> None:
    """Append ``entry`` to ``metadata.stage_models[stage]`` (best-effort).

    jsonb_set only auto-creates the *last* path element. To populate the
    nested ``stage_models.<stage>`` array on a row that has neither key yet,
    we merge instead: build the full stage_models object (preserving existing
    keys + appending to / creating the targeted stage's array) and then ``||``
    it back into metadata.

    Failures here MUST not break the pipeline — the audio work is the actual
    deliverable; per-model timings are reporting sugar.
    """
    try:
        with db.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE pipeline_runs
                       SET metadata = COALESCE(metadata, '{}'::jsonb)
                                       || jsonb_build_object(
                           'stage_models',
                           COALESCE(metadata->'stage_models', '{}'::jsonb)
                             || jsonb_build_object(
                                 %s,
                                 COALESCE(metadata->'stage_models'->%s, '[]'::jsonb)
                                   || jsonb_build_array(%s::jsonb)
                             )
                       )
                     WHERE job_id = %s
                    """,
                    (stage, stage, json.dumps(entry), job_id),
                )
            conn.commit()
    except Exception as exc:  # noqa: BLE001
        logger.warning("jobs_append_stage_model_failed job_id=%s stage=%s err=%r",
                       job_id, stage, exc)


def record_model(job_id: str, stage: str, model: str,
                 kind: str, duration_sec: float,
                 peaks: Optional[Dict[str, Any]] = None) -> None:
    """Append a single model-invocation record to metadata.stage_models[stage].

    ``peaks`` (optional) is the resource_sampler high-water-mark dict. Use this
    for steps that run *alone* (their peak is a clean, uncontended reading).
    For steps that overlap, use ``record_concurrent``. Shape after a typical
    /scorecard run:

        metadata.stage_models = {
            "asr":       [{"model": "whisper-base", "kind": "asr", "duration_sec": 10.8, "peaks": {...}}],
            "diarize":   [{"model": "pyannote/...", "kind": "diarization", "duration_sec": 6.5}],
            "scorecard": [{"model": "...", "kind": "voice_id", ...},
                          {"model": "...", "kind": "sentiment", ...},
                          {"kind": "concurrent", "members": [...], "peaks": {...}}, …],
            "persist":   [{"model": "qwen3-embedding:4b", "kind": "embedding", ...}, …],
        }
    """
    _append_stage_model(job_id, stage,
                        _model_entry(model, kind, duration_sec, peaks))


def record_concurrent(job_id: str, stage: str, members: list,
                      duration_sec: float,
                      peaks: Optional[Dict[str, Any]] = None) -> None:
    """Append a concurrent-group record (per-member times + one shared peak)
    to metadata.stage_models[stage]. Best-effort — see ``_append_stage_model``."""
    _append_stage_model(job_id, stage,
                        _concurrent_entry(members, duration_sec, peaks))


def record_stage_peaks(job_id: str, stage: str,
                       peaks: Dict[str, Any]) -> None:
    """Store stage-level resource peaks under metadata.stage_peaks[stage].
    Used for stages with no per-model sub-rows (convert/persist). Best-effort."""
    if not peaks:
        return
    try:
        with db.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE pipeline_runs
                       SET metadata = COALESCE(metadata, '{}'::jsonb)
                                       || jsonb_build_object(
                           'stage_peaks',
                           COALESCE(metadata->'stage_peaks', '{}'::jsonb)
                             || jsonb_build_object(%s, %s::jsonb)
                       )
                     WHERE job_id = %s
                    """,
                    (stage, json.dumps(peaks), job_id),
                )
            conn.commit()
    except Exception as exc:  # noqa: BLE001
        logger.warning("jobs_record_stage_peaks_failed job_id=%s stage=%s err=%r",
                       job_id, stage, exc)


def update_stage(job_id: str, stage: str) -> None:
    """Mark the stage the job is currently in + record its start timestamp
    in metadata.stage_times for per-stage duration computation.
    """
    ts = _now_iso()
    try:
        with db.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE pipeline_runs
                       SET stage = %s,
                           metadata = jsonb_set(
                               COALESCE(metadata, '{}'::jsonb),
                               '{stage_times}',
                               COALESCE(metadata->'stage_times', '{}'::jsonb)
                                 || jsonb_build_object(%s::text, to_jsonb(%s::text))
                           )
                     WHERE job_id = %s
                    """,
                    (stage, stage, ts, job_id),
                )
            conn.commit()
    except Exception as exc:  # noqa: BLE001
        logger.warning("jobs_stage_failed job_id=%s stage=%s error=%r",
                       job_id, stage, exc)


def finish(job_id: str, *, status: str, call_id: Optional[str] = None,
           started_monotonic: Optional[float] = None,
           error: Optional[str] = None) -> None:
    """Close the row: status='done' or 'failed', record duration + error,
    and stamp the terminal stage timestamp into ``stage_times``."""
    duration = (time.monotonic() - started_monotonic
                if started_monotonic is not None else None)
    terminal_key = "done" if status == "done" else "failed"
    terminal_ts  = _now_iso()
    try:
        with db.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE pipeline_runs
                       SET status       = %s,
                           call_id      = COALESCE(%s, call_id),
                           stage        = CASE WHEN %s = 'done' THEN 'done'
                                                ELSE stage END,
                           finished_at  = NOW(),
                           duration_sec = %s,
                           last_error   = %s,
                           metadata = jsonb_set(
                               COALESCE(metadata, '{}'::jsonb),
                               '{stage_times}',
                               COALESCE(metadata->'stage_times', '{}'::jsonb)
                                 || jsonb_build_object(%s::text, to_jsonb(%s::text))
                           )
                     WHERE job_id = %s
                    """,
                    (status, call_id, status, duration, error,
                     terminal_key, terminal_ts, job_id),
                )
            conn.commit()
    except Exception as exc:  # noqa: BLE001
        logger.warning("jobs_finish_failed job_id=%s status=%s error=%r",
                       job_id, status, exc)


def list_recent(*, status: Optional[str] = None,
                limit: int = 50) -> list[Dict[str, Any]]:
    """Most-recent N rows, optionally filtered by status."""
    sql = """
        SELECT job_id, run_no, call_id, kind, status, stage, attempts,
               started_at, finished_at, duration_sec, last_error,
               metadata,
               EXTRACT(EPOCH FROM (NOW() - started_at))::REAL AS live_elapsed_sec
          FROM pipeline_runs
    """
    params: list = []
    if status:
        sql += " WHERE status = %s"
        params.append(status)
    sql += " ORDER BY started_at DESC LIMIT %s"
    params.append(limit)

    return _fetch_jobs(sql, params)


def list_all(*, status: Optional[str] = None,
             limit: int = 500, offset: int = 0) -> list[Dict[str, Any]]:
    """Every job (paginated). Used by the control UI's jobs table.

    Same shape as ``list_recent`` plus ``live_elapsed_sec`` (NOW - started_at
    in seconds) so the UI can render a live counter for running jobs without
    doing arithmetic in JS.
    """
    sql = """
        SELECT job_id, run_no, call_id, kind, status, stage, attempts,
               started_at, finished_at, duration_sec, last_error,
               metadata,
               EXTRACT(EPOCH FROM (NOW() - started_at))::REAL AS live_elapsed_sec
          FROM pipeline_runs
    """
    params: list = []
    if status:
        sql += " WHERE status = %s"
        params.append(status)
    sql += " ORDER BY started_at DESC LIMIT %s OFFSET %s"
    params.extend([limit, offset])

    return _fetch_jobs(sql, params)


def get_one(job_id: str) -> Optional[Dict[str, Any]]:
    """Single row by job_id, or ``None`` if missing."""
    sql = """
        SELECT job_id, run_no, call_id, kind, status, stage, attempts,
               started_at, finished_at, duration_sec, last_error,
               metadata,
               EXTRACT(EPOCH FROM (NOW() - started_at))::REAL AS live_elapsed_sec
          FROM pipeline_runs
         WHERE job_id = %s
    """
    rows = _fetch_jobs(sql, [job_id])
    return rows[0] if rows else None


def get_by_run_no(run_no: int) -> Optional[Dict[str, Any]]:
    """Single row by sequential run number, or ``None`` if missing."""
    sql = """
        SELECT job_id, run_no, call_id, kind, status, stage, attempts,
               started_at, finished_at, duration_sec, last_error,
               metadata,
               EXTRACT(EPOCH FROM (NOW() - started_at))::REAL AS live_elapsed_sec
          FROM pipeline_runs
         WHERE run_no = %s
    """
    rows = _fetch_jobs(sql, [run_no])
    return rows[0] if rows else None


def current() -> Optional[Dict[str, Any]]:
    """The most relevant job to show in the "Current" panel.

    Picks the most-recently-started *running* job. If nothing is
    running, falls back to the most-recently-started job overall — so
    the control panel always has something to display.
    """
    sql = """
        SELECT job_id, run_no, call_id, kind, status, stage, attempts,
               started_at, finished_at, duration_sec, last_error,
               metadata,
               EXTRACT(EPOCH FROM (NOW() - started_at))::REAL AS live_elapsed_sec
          FROM pipeline_runs
         ORDER BY (status = 'running') DESC, started_at DESC
         LIMIT 1
    """
    rows = _fetch_jobs(sql, [])
    return rows[0] if rows else None


def run_no_index(limit: int = 500) -> list[Dict[str, Any]]:
    """Compact list for the History dropdown: run_no, status, file, started_at.

    Latest first so the most recent runs are at the top of the dropdown.
    """
    sql = """
        SELECT run_no, status, stage, started_at, duration_sec,
               metadata->>'file_name' AS file_name
          FROM pipeline_runs
         ORDER BY run_no DESC
         LIMIT %s
    """
    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, [limit])
            cols = [c[0] for c in cur.description]
            rows = []
            for row in cur.fetchall():
                d = dict(zip(cols, row))
                if d.get("started_at") and hasattr(d["started_at"], "isoformat"):
                    d["started_at"] = d["started_at"].isoformat()
                if d.get("duration_sec") is not None:
                    d["duration_sec"] = float(d["duration_sec"])
                rows.append(d)
            return rows


def _fetch_jobs(sql: str, params: list) -> list[Dict[str, Any]]:
    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            cols = [c[0] for c in cur.description]
            rows = []
            for row in cur.fetchall():
                d = dict(zip(cols, row))
                # Make timestamps JSON-friendly
                for k in ("started_at", "finished_at"):
                    if d.get(k) is not None and hasattr(d[k], "isoformat"):
                        d[k] = d[k].isoformat()
                # Coerce psycopg2 RealDictRow / Decimal into JSON-friendly types
                if d.get("duration_sec") is not None:
                    d["duration_sec"] = float(d["duration_sec"])
                if d.get("live_elapsed_sec") is not None:
                    d["live_elapsed_sec"] = float(d["live_elapsed_sec"])
                rows.append(d)
            return rows


# ── Stuck-job reaper ─────────────────────────────────────────────────────────
# When ser-api gets SIGKILL'd mid-request (classic WSL2 host-OOM signature:
# exit 0, no OOMKilled flag, no shutdown log lines), the jobs row stays in
# status='running' forever and consumes a slot in /jobs visibility. We
# demote any row that hasn't moved off `running` for STUCK_MINUTES so the
# next /jobs?status=failed sweep catches them and an operator can SELECT
# WHERE status='failed' AND last_error LIKE 'stuck%' to see the wreckage.
#
# Async lifespan task — started by app.lifespan, cancelled on shutdown.
import asyncio  # noqa: E402

_STUCK_QUERY = """
    UPDATE pipeline_runs
       SET status      = 'failed',
           finished_at = NOW(),
           last_error  = 'stuck: container restart killed in-flight request'
     WHERE status      = 'running'
       AND started_at  < NOW() - make_interval(mins => %s)
"""


def reap_stuck_sync(stuck_minutes: int) -> int:
    """Single pass. Returns number of rows demoted."""
    try:
        with db.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(_STUCK_QUERY, (stuck_minutes,))
                n = cur.rowcount
            conn.commit()
        return n
    except Exception as exc:  # noqa: BLE001
        logger.warning("jobs_reaper_iteration_failed err=%r", exc)
        return 0


async def reaper_loop(stuck_minutes: int = 30,
                      interval_sec: float = 300.0) -> None:
    """Long-running async loop, started by lifespan."""
    logger.info("jobs_reaper_started stuck_minutes=%d interval_sec=%.0f",
                stuck_minutes, interval_sec)
    # First pass shortly after startup so a restart that left behind
    # orphans gets cleaned promptly.
    await asyncio.sleep(20)
    while True:
        try:
            n = await asyncio.to_thread(reap_stuck_sync, stuck_minutes)
            if n:
                logger.info("jobs_reaper_demoted count=%d", n)
        except asyncio.CancelledError:
            logger.info("jobs_reaper_cancelled")
            raise
        except Exception:  # noqa: BLE001
            logger.exception("jobs_reaper_loop_failed")
        await asyncio.sleep(interval_sec)


def start_reaper(loop: asyncio.AbstractEventLoop,
                 stuck_minutes: int = 30,
                 interval_sec: float = 300.0) -> asyncio.Task:
    return loop.create_task(reaper_loop(stuck_minutes, interval_sec),
                            name="ser_api_jobs_reaper")

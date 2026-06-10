"""Read-mostly Postgres access for the control UI.

We don't reuse ser-api/db.py — that file does schema bootstrap and DDL.
The control UI only needs lightweight read queries against `calls`,
`utterances`, `call_events`, and `pipeline_runs`, plus a few aggregations
for the History/Metrics view.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional

import psycopg2
import psycopg2.extras

import config

logger = logging.getLogger(__name__)


@contextmanager
def connect():
    """Short-lived autocommit connection. Closes on exit."""
    conn = psycopg2.connect(config.DATABASE_URL)
    try:
        yield conn
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass


def _fetchall(sql: str, params: Iterable[Any] = ()) -> List[Dict[str, Any]]:
    with connect() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, list(params))
            return [dict(r) for r in cur.fetchall()]


def _fetchone(sql: str, params: Iterable[Any] = ()) -> Optional[Dict[str, Any]]:
    rows = _fetchall(sql, params)
    return rows[0] if rows else None


# ── Calls ───────────────────────────────────────────────────────────────────
def list_calls(limit: int = 100) -> List[Dict[str, Any]]:
    """Most-recent calls. Light projection — enough for the index table."""
    return _fetchall(
        """
        SELECT call_id, received_at, language, closer_name, client_name,
               duration_sec, coaching_score, predicted_outcome,
               objection_count, buying_signal_count, next_step_secured,
               pdf_url, html_url, model_versions
          FROM calls
         ORDER BY received_at DESC NULLS LAST, call_id DESC
         LIMIT %s
        """,
        (limit,),
    )


def get_call(call_id: str) -> Optional[Dict[str, Any]]:
    return _fetchone("SELECT * FROM calls WHERE call_id = %s", (call_id,))


def utterances_for(call_id: str) -> List[Dict[str, Any]]:
    return _fetchall(
        """
        SELECT idx, start_sec, end_sec, speaker, role, text,
               sentiment_label, sentiment_score,
               emotion_label, valence, arousal, dominance
          FROM utterances
         WHERE call_id = %s
         ORDER BY idx ASC
        """,
        (call_id,),
    )


def events_for(call_id: str) -> List[Dict[str, Any]]:
    return _fetchall(
        """
        SELECT id, kind, text, normalized, confidence, start_sec, speaker_role
          FROM call_events
         WHERE call_id = %s
         ORDER BY start_sec ASC NULLS LAST, id ASC
        """,
        (call_id,),
    )


# ── Metrics / aggregates (History view) ────────────────────────────────────
def jobs_per_model(model_role: str = "asr",
                   days: Optional[int] = None) -> List[Dict[str, Any]]:
    """Aggregate by metadata->'models'->>{role}.

    Returns: model_id, runs, ok_count, fail_count, avg_duration_sec.
    """
    time_filter = ""
    params: List[Any] = [model_role]
    if days is not None:
        time_filter = "AND started_at >= NOW() - make_interval(days => %s)"
        params.append(days)
    sql = f"""
        SELECT
            COALESCE(metadata->'models'->>%s, '(unknown)')      AS model_id,
            COUNT(*)                                             AS runs,
            COUNT(*) FILTER (WHERE status = 'done')              AS ok_count,
            COUNT(*) FILTER (WHERE status = 'failed')            AS fail_count,
            AVG(duration_sec) FILTER (WHERE status = 'done')     AS avg_duration_sec,
            MIN(duration_sec) FILTER (WHERE status = 'done')     AS min_duration_sec,
            MAX(duration_sec) FILTER (WHERE status = 'done')     AS max_duration_sec
          FROM pipeline_runs
         WHERE 1=1 {time_filter}
         GROUP BY model_id
         ORDER BY runs DESC
    """
    return _fetchall(sql, params)


def jobs_summary(days: Optional[int] = None) -> Dict[str, Any]:
    """Top-line counts + averages for the History page header cards."""
    time_filter = ""
    params: List[Any] = []
    if days is not None:
        time_filter = "WHERE started_at >= NOW() - make_interval(days => %s)"
        params.append(days)
    row = _fetchone(
        f"""
        SELECT
            COUNT(*)                                             AS total,
            COUNT(*) FILTER (WHERE status = 'running')           AS running_count,
            COUNT(*) FILTER (WHERE status = 'done')              AS done_count,
            COUNT(*) FILTER (WHERE status = 'failed')            AS failed_count,
            AVG(duration_sec) FILTER (WHERE status = 'done')     AS avg_duration_sec,
            MAX(duration_sec) FILTER (WHERE status = 'done')     AS max_duration_sec
          FROM pipeline_runs
          {time_filter}
        """,
        params,
    )
    return row or {}


def jobs_filter(status: Optional[str] = None,
                model_role: Optional[str] = None,
                model_id: Optional[str] = None,
                days: Optional[int] = None,
                limit: int = 500) -> List[Dict[str, Any]]:
    """Server-side filtering for the jobs table (status / model / time range)."""
    where: List[str] = []
    params: List[Any] = []
    if status:
        where.append("status = %s")
        params.append(status)
    if model_role and model_id:
        where.append("metadata->'models'->>%s = %s")
        params.extend([model_role, model_id])
    if days is not None:
        where.append("started_at >= NOW() - make_interval(days => %s)")
        params.append(days)
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""
    params.append(limit)
    sql = f"""
        SELECT job_id, call_id, kind, status, stage, attempts,
               started_at, finished_at, duration_sec, last_error,
               metadata,
               EXTRACT(EPOCH FROM (NOW() - started_at))::REAL AS live_elapsed_sec
          FROM pipeline_runs
          {where_sql}
         ORDER BY started_at DESC
         LIMIT %s
    """
    rows = _fetchall(sql, params)
    for r in rows:
        for k in ("started_at", "finished_at"):
            if r.get(k) is not None and hasattr(r[k], "isoformat"):
                r[k] = r[k].isoformat()
    return rows


# ── Job reads (bypass ser-api so polling keeps working during /scorecard) ──
# ser-api uses a sync threading.Lock around the GPU pipeline, which blocks
# its async event loop. /jobs/current via HTTP times out while a scorecard
# is in flight, exactly when the operator most wants live polling. We read
# straight from Postgres instead.
_JOB_PROJ = """
    SELECT job_id, run_no, call_id, kind, status, stage, attempts,
           started_at, finished_at, duration_sec, last_error,
           metadata,
           EXTRACT(EPOCH FROM (NOW() - started_at))::REAL AS live_elapsed_sec
      FROM pipeline_runs
"""


def _coerce_job_row(d: Dict[str, Any]) -> Dict[str, Any]:
    for k in ("started_at", "finished_at"):
        if d.get(k) is not None and hasattr(d[k], "isoformat"):
            d[k] = d[k].isoformat()
    if d.get("duration_sec") is not None:
        d["duration_sec"] = float(d["duration_sec"])
    if d.get("live_elapsed_sec") is not None:
        d["live_elapsed_sec"] = float(d["live_elapsed_sec"])
    return d


def job_current() -> Optional[Dict[str, Any]]:
    """Most-recent running job, or — if nothing is running — the most-recent
    job overall. Same heuristic as ser-api/pipeline/jobs.py:current()."""
    rows = _fetchall(
        _JOB_PROJ + " ORDER BY (status = 'running') DESC, started_at DESC LIMIT 1"
    )
    return _coerce_job_row(rows[0]) if rows else None


def job_by_run_no(run_no: int) -> Optional[Dict[str, Any]]:
    rows = _fetchall(_JOB_PROJ + " WHERE run_no = %s", (run_no,))
    return _coerce_job_row(rows[0]) if rows else None


def run_index(limit: int = 500) -> List[Dict[str, Any]]:
    sql = """
        SELECT run_no, status, stage, started_at, duration_sec,
               metadata->>'file_name' AS file_name
          FROM pipeline_runs
         ORDER BY run_no DESC
         LIMIT %s
    """
    rows = _fetchall(sql, (limit,))
    for r in rows:
        if r.get("started_at") and hasattr(r["started_at"], "isoformat"):
            r["started_at"] = r["started_at"].isoformat()
        if r.get("duration_sec") is not None:
            r["duration_sec"] = float(r["duration_sec"])
    return rows


def distinct_models(role: str) -> List[str]:
    """Distinct model ids that show up in metadata->'models'->>role."""
    rows = _fetchall(
        """
        SELECT DISTINCT metadata->'models'->>%s AS m
          FROM pipeline_runs
         WHERE metadata->'models'->>%s IS NOT NULL
         ORDER BY m
        """,
        (role, role),
    )
    return [r["m"] for r in rows if r.get("m")]


# ── Row counts for the System page ──────────────────────────────────────────
def table_counts() -> Dict[str, int]:
    out: Dict[str, int] = {}
    for tbl in ("calls", "utterances", "call_events", "pipeline_runs"):
        try:
            row = _fetchone(f"SELECT COUNT(*) AS n FROM {tbl}")
            out[tbl] = int(row["n"]) if row else 0
        except Exception as exc:  # noqa: BLE001
            logger.warning("count_failed table=%s err=%r", tbl, exc)
            out[tbl] = -1
    return out

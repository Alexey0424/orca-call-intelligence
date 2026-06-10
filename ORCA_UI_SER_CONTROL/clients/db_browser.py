"""Multi-database browser + safe query runner.

Targets the two Postgres databases living in the same container:
``orca`` (our app) and ``n8n`` (workflow runtime). For each database
we expose:

  * ``list_databases()``    — name + on-disk size
  * ``list_tables(db)``     — table list with total / heap / index sizes
  * ``table_detail(...)``   — columns, indexes, size breakdown
  * ``preview_rows(...)``   — first N rows (safe identifier interpolation)
  * ``run_query(...)``      — arbitrary SQL with read-only / write modes,
                              statement timeout, row cap, elapsed timing

Safety:
  * Database names come from the ``ALLOWED_DBS`` allowlist; anything else
    is rejected at the edge.
  * Identifiers (schema, table) are wrapped in ``psycopg2.sql.Identifier``
    so they can never be SQL-injected through the preview path.
  * ``run_query(read_only=True)`` runs the user SQL inside a
    ``READ ONLY`` transaction — Postgres rejects UPDATE/DELETE/DDL itself.
  * ``run_query`` always sets ``statement_timeout`` on the connection
    before running the user statement.
"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, urlunparse

import psycopg2
import psycopg2.extras
import psycopg2.sql as sql

import config

logger = logging.getLogger(__name__)


# ── Allowlist + DSN composition ─────────────────────────────────────────────
# Only these database names are ever passed through to psycopg2. Any
# other name from the request is rejected before we touch the connection.
ALLOWED_DBS = ("orca", "n8n")

# Hard ceilings on the Query tab — protects the UI from accidentally pulling
# a 50 M-row preview into the page.
DEFAULT_STATEMENT_TIMEOUT_MS = 15_000
DEFAULT_MAX_ROWS             = 1_000
PREVIEW_MAX_ROWS             = 200


def _dsn_for(db: str) -> str:
    """Swap the database segment of ``config.DATABASE_URL`` to ``db``.

    DATABASE_URL is shaped ``postgresql://user:pass@host[:port]/<db>`` —
    we keep everything except the path and substitute the database name.
    """
    if db not in ALLOWED_DBS:
        raise ValueError(f"database not allowed: {db!r}")
    parsed = urlparse(config.DATABASE_URL)
    return urlunparse(parsed._replace(path=f"/{db}"))


@contextmanager
def _connect(db: str, *, read_only: bool = True,
             statement_timeout_ms: int = DEFAULT_STATEMENT_TIMEOUT_MS):
    """Open a short-lived connection to ``db`` with sensible safety knobs.

    The connection is configured before yielding:
      * ``statement_timeout`` so a runaway query can't pin a worker.
      * ``default_transaction_read_only`` when ``read_only`` — Postgres
        enforces this server-side, so even a typo'd ``DROP TABLE`` is
        rejected with a clear error instead of silently succeeding.

    The connection is autocommit so the read-only flag actually applies to
    user queries (otherwise the first SELECT opens a default RW txn).
    """
    conn = psycopg2.connect(_dsn_for(db))
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("SET statement_timeout = %s", (statement_timeout_ms,))
            if read_only:
                cur.execute("SET default_transaction_read_only = on")
        yield conn
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass


# ── Database / table listings (system catalog queries) ─────────────────────
def list_databases() -> List[Dict[str, Any]]:
    """Return one row per allowlisted database with size + table count.

    Connects to ``orca`` and uses ``pg_database_size`` to read every DB —
    cheap and avoids opening three separate connections just to list sizes.
    """
    placeholders = sql.SQL(", ").join(sql.Literal(d) for d in ALLOWED_DBS)
    query = sql.SQL(
        """
        SELECT
            d.datname                                            AS name,
            pg_database_size(d.datname)                          AS size_bytes,
            pg_size_pretty(pg_database_size(d.datname))          AS size_pretty
          FROM pg_database d
         WHERE d.datname IN ({names})
         ORDER BY d.datname
        """
    ).format(names=placeholders)

    with _connect("orca") as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query)
            rows = [dict(r) for r in cur.fetchall()]

    # Best-effort table count per DB (one cheap extra query each).
    for row in rows:
        try:
            with _connect(row["name"]) as conn2:
                with conn2.cursor() as cur2:
                    cur2.execute(
                        """
                        SELECT COUNT(*)
                          FROM information_schema.tables
                         WHERE table_schema NOT IN ('pg_catalog','information_schema')
                           AND table_type = 'BASE TABLE'
                        """
                    )
                    row["table_count"] = int(cur2.fetchone()[0])
        except Exception as exc:  # noqa: BLE001
            logger.warning("table_count_failed db=%s err=%r", row["name"], exc)
            row["table_count"] = None

    return rows


def list_tables(db: str) -> List[Dict[str, Any]]:
    """Tables in ``db`` with total / heap / index sizes + row estimate."""
    query = """
        SELECT
            n.nspname                                              AS schema,
            c.relname                                              AS name,
            pg_total_relation_size(c.oid)                          AS size_total_bytes,
            pg_relation_size(c.oid)                                AS size_heap_bytes,
            pg_indexes_size(c.oid)                                 AS size_index_bytes,
            pg_size_pretty(pg_total_relation_size(c.oid))          AS size_total_pretty,
            pg_size_pretty(pg_relation_size(c.oid))                AS size_heap_pretty,
            pg_size_pretty(pg_indexes_size(c.oid))                 AS size_index_pretty,
            COALESCE(s.n_live_tup, 0)                              AS row_estimate
          FROM pg_class c
          JOIN pg_namespace n ON n.oid = c.relnamespace
          LEFT JOIN pg_stat_user_tables s
                 ON s.relid = c.oid
         WHERE c.relkind = 'r'
           AND n.nspname NOT IN ('pg_catalog','information_schema')
         ORDER BY pg_total_relation_size(c.oid) DESC, c.relname ASC
    """
    with _connect(db) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query)
            rows = [dict(r) for r in cur.fetchall()]

    # Coerce psycopg2 numerics → plain ints so Jinja's `|int` and number
    # filters work without further casting.
    for r in rows:
        for k in ("size_total_bytes", "size_heap_bytes",
                  "size_index_bytes", "row_estimate"):
            if r.get(k) is not None:
                r[k] = int(r[k])
    return rows


def table_detail(db: str, schema: str, table: str) -> Optional[Dict[str, Any]]:
    """Return columns + indexes + size breakdown for one table.

    None if the table doesn't exist (catches typos / stale URLs without
    bubbling a 500 to the user).
    """
    with _connect(db) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Columns
            cur.execute(
                """
                SELECT column_name, data_type, udt_name, is_nullable,
                       column_default, character_maximum_length,
                       ordinal_position
                  FROM information_schema.columns
                 WHERE table_schema = %s AND table_name = %s
                 ORDER BY ordinal_position
                """,
                (schema, table),
            )
            cols = [dict(r) for r in cur.fetchall()]
            if not cols:
                return None

            # Indexes
            cur.execute(
                """
                SELECT indexname, indexdef
                  FROM pg_indexes
                 WHERE schemaname = %s AND tablename = %s
                 ORDER BY indexname
                """,
                (schema, table),
            )
            idxs = [dict(r) for r in cur.fetchall()]

            # Sizes + row estimate (relname/relnamespace, scope it down to
            # this exact pair).
            cur.execute(
                """
                SELECT
                    pg_total_relation_size(c.oid)               AS size_total_bytes,
                    pg_relation_size(c.oid)                     AS size_heap_bytes,
                    pg_indexes_size(c.oid)                      AS size_index_bytes,
                    pg_size_pretty(pg_total_relation_size(c.oid)) AS size_total_pretty,
                    pg_size_pretty(pg_relation_size(c.oid))     AS size_heap_pretty,
                    pg_size_pretty(pg_indexes_size(c.oid))      AS size_index_pretty,
                    COALESCE(s.n_live_tup, 0)                   AS row_estimate
                  FROM pg_class c
                  JOIN pg_namespace n ON n.oid = c.relnamespace
                  LEFT JOIN pg_stat_user_tables s ON s.relid = c.oid
                 WHERE n.nspname = %s AND c.relname = %s AND c.relkind = 'r'
                """,
                (schema, table),
            )
            sizes_row = cur.fetchone()
            sizes = dict(sizes_row) if sizes_row else {}

    for k in ("size_total_bytes", "size_heap_bytes",
              "size_index_bytes", "row_estimate"):
        if sizes.get(k) is not None:
            sizes[k] = int(sizes[k])

    return {"schema": schema, "name": table,
            "columns": cols, "indexes": idxs, **sizes}


def preview_rows(db: str, schema: str, table: str,
                 limit: int = 50) -> Dict[str, Any]:
    """Return ``{columns, rows, elapsed_ms}`` for the first N rows.

    Safe: schema + table are passed through ``psycopg2.sql.Identifier`` so
    user-supplied names can't escape the literal slot.
    """
    limit = max(1, min(int(limit or 50), PREVIEW_MAX_ROWS))
    stmt = sql.SQL("SELECT * FROM {sch}.{tbl} LIMIT {lim}").format(
        sch=sql.Identifier(schema),
        tbl=sql.Identifier(table),
        lim=sql.Literal(limit),
    )
    t0 = time.monotonic()
    with _connect(db) as conn:
        with conn.cursor() as cur:
            cur.execute(stmt)
            cols = [c[0] for c in cur.description] if cur.description else []
            rows = cur.fetchall() if cur.description else []
    elapsed_ms = round((time.monotonic() - t0) * 1000, 1)
    return {"columns": cols,
            "rows": [_coerce_row(r) for r in rows],
            "row_count": len(rows),
            "elapsed_ms": elapsed_ms}


# ── Arbitrary SQL runner ────────────────────────────────────────────────────
def run_query(db: str, sql_text: str, *,
              read_only: bool = True,
              max_rows: int = DEFAULT_MAX_ROWS,
              statement_timeout_ms: int = DEFAULT_STATEMENT_TIMEOUT_MS
              ) -> Dict[str, Any]:
    """Execute one SQL statement and return a JSON-friendly result.

    Shape: ``{ok, columns, rows, row_count, truncated, elapsed_ms, error}``.
    On error, ``ok=False`` + ``error`` is set; ``columns``/``rows`` are []
    so the UI can render the response uniformly.

    The read-only enforcement is delegated to Postgres
    (``SET default_transaction_read_only = on``) — much harder to fool than
    pattern-matching the SQL text.
    """
    sql_text = (sql_text or "").strip()
    if not sql_text:
        return _error_result("empty query")
    if db not in ALLOWED_DBS:
        return _error_result(f"database not allowed: {db!r}")

    t0 = time.monotonic()
    try:
        with _connect(db, read_only=read_only,
                       statement_timeout_ms=statement_timeout_ms) as conn:
            with conn.cursor() as cur:
                cur.execute(sql_text)
                # cur.description is None for statements that don't return
                # rows (UPDATE, DELETE, DDL); in that case ``rowcount`` is
                # the count of rows affected.
                if cur.description is None:
                    elapsed_ms = round((time.monotonic() - t0) * 1000, 1)
                    return {
                        "ok": True, "columns": [], "rows": [],
                        "row_count": 0,
                        "affected": int(cur.rowcount) if cur.rowcount is not None else None,
                        "truncated": False,
                        "elapsed_ms": elapsed_ms,
                        "error": None,
                    }
                cols = [c[0] for c in cur.description]
                rows = cur.fetchmany(max_rows + 1)
                truncated = len(rows) > max_rows
                if truncated:
                    rows = rows[:max_rows]
                elapsed_ms = round((time.monotonic() - t0) * 1000, 1)
                return {
                    "ok": True, "columns": cols,
                    "rows": [_coerce_row(r) for r in rows],
                    "row_count": len(rows),
                    "truncated": truncated,
                    "affected": None,
                    "elapsed_ms": elapsed_ms,
                    "error": None,
                }
    except psycopg2.errors.QueryCanceled:
        return _error_result(f"query exceeded statement_timeout "
                              f"({statement_timeout_ms} ms)")
    except psycopg2.errors.ReadOnlySqlTransaction as exc:
        return _error_result(
            f"read-only mode blocked this statement: {str(exc).splitlines()[0]}"
        )
    except Exception as exc:  # noqa: BLE001
        return _error_result(str(exc).splitlines()[0])


def _error_result(msg: str) -> Dict[str, Any]:
    return {"ok": False, "columns": [], "rows": [],
            "row_count": 0, "truncated": False,
            "affected": None, "elapsed_ms": 0.0, "error": msg}


# ── Row coercion (psycopg2 → JSON-friendly) ────────────────────────────────
def _coerce_value(v: Any) -> Any:
    """Make a single column value safe to render in the result table.

    * datetime / date / time → ISO string
    * Decimal → float
    * memoryview / bytes → ``<bytes:N>`` placeholder (no payload)
    * dict / list (JSONB) → kept as-is (Jinja will tojson it)
    * pgvector ``vector`` → string already; truncate if huge
    """
    if v is None:
        return None
    if hasattr(v, "isoformat"):
        return v.isoformat()
    cls = v.__class__.__name__
    if cls == "Decimal":
        return float(v)
    if isinstance(v, (bytes, bytearray, memoryview)):
        try:
            n = len(v)
        except Exception:  # noqa: BLE001
            n = -1
        return f"<bytes:{n}>"
    if isinstance(v, str) and len(v) > 4096:
        # Long strings (JSON dumps, big text columns) — truncate so a
        # single row doesn't blow up the result panel.
        return v[:4096] + f"… [{len(v) - 4096} more chars]"
    return v


def _coerce_row(row: Tuple[Any, ...]) -> List[Any]:
    return [_coerce_value(v) for v in row]

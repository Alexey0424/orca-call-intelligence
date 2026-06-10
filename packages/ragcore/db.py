"""Postgres connection helper, shared by both paths.

Schema bootstrap (``ensure_schema``) lives in call-analysis-api's own db module,
not here — rag-api never creates tables.
"""
from __future__ import annotations

import psycopg2
from psycopg2.extensions import connection as PgConnection

from ragcore import config


def connect() -> PgConnection:
    """Open a fresh psycopg2 connection. Caller closes."""
    return psycopg2.connect(config.DATABASE_URL)

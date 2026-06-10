"""Shared core for the call-analysis read + write paths.

Three modules used by both ``call-analysis-api`` (write path) and ``rag-api``
(read path), so the embedding recipe, DB connection, and core env config live
in exactly one place:

  config      — torch-free env parsing (DB, Ollama, embeddings, render,
                gotenberg, brand). No device detection, no audio-model IDs.
  db          — psycopg2 connection helper.
  embeddings  — Ollama embeddings client + Matryoshka truncation + L2 norm.

Importable as ``from ragcore import config, db, embeddings``. The
``call-analysis-api`` service re-exports these via thin shim modules so its
existing ``import config`` / ``import db`` / ``from pipeline import embeddings``
call sites keep working unchanged.
"""
__all__ = ["config", "db", "embeddings"]

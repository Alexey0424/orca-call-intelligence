"""pytest configuration for rag-api tests.

Mocks psycopg2, httpx, and the ragcore db/embeddings modules so pure-unit
tests (like _vec_literal) can run without a live Postgres or Ollama.
The Docker container has all deps installed; this conftest guards the
host-side ``python -m pytest`` run used to verify the module exists.
"""
import sys
import types


def _make_mock_module(name: str, **attrs):
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    return mod


# ── psycopg2 stub ────────────────────────────────────────────────────────────
if "psycopg2" not in sys.modules:
    ext = _make_mock_module("psycopg2.extensions")
    ext.connection = object  # PgConnection type alias

    psycopg2 = _make_mock_module(
        "psycopg2",
        connect=lambda *a, **kw: None,
        extras=_make_mock_module("psycopg2.extras"),
        extensions=ext,
    )
    sys.modules["psycopg2"] = psycopg2
    sys.modules["psycopg2.extensions"] = ext
    sys.modules["psycopg2.extras"] = psycopg2.extras

# ── httpx stub (for embeddings.embed_one) ────────────────────────────────────
if "httpx" not in sys.modules:
    sys.modules["httpx"] = _make_mock_module("httpx")

# ── ragcore.db stub ──────────────────────────────────────────────────────────
ragcore_db = _make_mock_module("ragcore.db", connect=lambda: None)
sys.modules.setdefault("ragcore.db", ragcore_db)

# ── ragcore.embeddings stub ──────────────────────────────────────────────────
ragcore_embeddings = _make_mock_module(
    "ragcore.embeddings",
    embed_one=lambda q: None,
    embed_one_safe=lambda q: None,
    embed_batch_safe=lambda qs: [None] * len(qs),
)
sys.modules.setdefault("ragcore.embeddings", ragcore_embeddings)

"""Torch-free env config shared by call-analysis-api and rag-api.

Only environment that BOTH the write path and the read path need lives here:
DB URL, Ollama host + embedding model resolver, render-api URL, Gotenberg URL,
brand strings, logging. Deliberately NO ``import torch`` and NO audio-model IDs
— those stay in call-analysis-api's own config so rag-api never pulls torch.
"""
from __future__ import annotations

import json
import logging
import os

_logger = logging.getLogger(__name__)

# ── Runtime overrides (written by ORCA_UI_SER_CONTROL) ──────────────────────
RUNTIME_OVERRIDES_PATH = os.getenv("RUNTIME_OVERRIDES_PATH",
                                   "/data/runtime/overrides.json")


def _load_overrides() -> dict:
    try:
        with open(RUNTIME_OVERRIDES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception as exc:  # noqa: BLE001
        _logger.warning("runtime_overrides_unreadable path=%s err=%r",
                        RUNTIME_OVERRIDES_PATH, exc)
        return {}


def _resolve(name: str, default: str) -> str:
    """Override file > env > default. Read at import."""
    overrides = _load_overrides()
    if name in overrides and overrides[name] is not None:
        return str(overrides[name])
    return os.getenv(name, default)


def _runtime_get(name: str, default: str) -> str:
    """Per-request resolver (re-reads the override file each call)."""
    overrides = _load_overrides()
    if name in overrides and overrides[name] is not None:
        return str(overrides[name])
    return os.getenv(name, default)


# ── Ollama ───────────────────────────────────────────────────────────────────
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")

# ── Embeddings (Ollama + Matryoshka truncation) ──────────────────────────────
EMBEDDING_MODEL_ID    = _resolve("EMBEDDING_MODEL_ID", "qwen3-embedding:4b")
EMBEDDING_NATIVE_DIM  = int(os.getenv("EMBEDDING_NATIVE_DIM", "2560"))
EMBEDDING_DIM         = int(os.getenv("EMBEDDING_DIM", "1024"))
EMBEDDING_TIMEOUT_SEC = float(os.getenv("EMBEDDING_TIMEOUT_SEC", "60"))
# GPU layers Ollama offloads for the embedding model.
#    0  → force CPU. DEFAULT. On this 16 GB card it's the reliable choice: the
#         utterance embedding runs DURING persist, while gemma (10 GB, for
#         coaching) is still resident. Putting the embedder on GPU then leaves
#         gemma+embedder at the very edge of 16 GB, so gemma intermittently
#         spills to CPU and coaching balloons (measured 5 s … 79 s, unreliable).
#         CPU embedding is ~57 s for a long call but keeps coaching steady ~2 s.
#   <0  → omit num_gpu → Ollama auto-fits to free VRAM (GPU when there's room).
#         Faster embedding (~5 s) BUT unreliable coaching on 16 GB — only safe
#         once gemma is freed before the embedding (would need a persist reorder).
#   >0  → force exactly N layers onto the GPU.
EMBEDDING_NUM_GPU     = int(os.getenv("EMBEDDING_NUM_GPU", "0"))


def embedding_model_id() -> str:
    return _runtime_get("EMBEDDING_MODEL_ID", "qwen3-embedding:4b")


# ── render-api (per-call PDF + dashboard HTML) ───────────────────────────────
SER_RENDER_URL         = os.getenv("SER_RENDER_URL", "http://render-api:8010")
SER_RENDER_TIMEOUT_SEC = float(os.getenv("SER_RENDER_TIMEOUT_SEC", "30"))
SER_API_PUBLIC_URL     = os.getenv("SER_API_PUBLIC_URL", "http://localhost:8000")

# ── Gotenberg (dashboard screenshot) ─────────────────────────────────────────
GOTENBERG_URL         = os.getenv("GOTENBERG_URL", "http://gotenberg:3000")
GOTENBERG_TIMEOUT_SEC = float(os.getenv("GOTENBERG_TIMEOUT_SEC", "45"))

# ── Dashboard branding (override per deployment via BRAND_* env vars) ────────
BRAND_MARK         = os.getenv("BRAND_MARK",  "O")
BRAND_NAME         = os.getenv("BRAND_NAME",  "ORCA")
BRAND_SUB          = os.getenv("BRAND_SUB",   "Call Intelligence")
BRAND_FOOTER_LEFT  = os.getenv("BRAND_FOOTER_LEFT",
                               "ORCA · Call Intelligence Report")
BRAND_FOOTER_RIGHT = os.getenv("BRAND_FOOTER_RIGHT", "Powered by ORCA")

# ── Database ─────────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"postgresql://{os.getenv('POSTGRES_USER','root')}:"
    f"{os.getenv('POSTGRES_PASSWORD','password')}@postgres/"
    # Fallback DB is 'orca' (our app DB); never fires in prod (compose injects DATABASE_URL). Was 'n8n' pre-three-DB-split.
    f"{os.getenv('POSTGRES_DB','orca')}",
)
DB_INIT_RETRIES     = int(os.getenv("DB_INIT_RETRIES", "10"))
DB_INIT_BACKOFF_SEC = float(os.getenv("DB_INIT_BACKOFF_SEC", "3"))

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

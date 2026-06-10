"""Central env-var parsing for ORCA_UI_SER_CONTROL.

All modules read settings from here, never directly from os.environ.
"""
from __future__ import annotations

import os
from pathlib import Path

# ── Inward HTTP clients ─────────────────────────────────────────────────────
SER_API_URL          = os.getenv("SER_API_URL",   "http://call-analysis-api:8000")
SER_API_TIMEOUT_SEC  = float(os.getenv("SER_API_TIMEOUT_SEC",  "10"))

OLLAMA_HOST          = os.getenv("OLLAMA_HOST",   "http://ollama:11434")
OLLAMA_TIMEOUT_SEC   = float(os.getenv("OLLAMA_TIMEOUT_SEC",   "10"))

# Docker Engine API — Unix socket inside the container.
DOCKER_SOCKET_PATH   = os.getenv("DOCKER_SOCKET_PATH", "/var/run/docker.sock")
SER_API_CONTAINER    = os.getenv("SER_API_CONTAINER",  "call-analysis-api")

# ── Database (read-mostly: jobs, calls, utterances, events) ────────────────
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"postgresql://{os.getenv('POSTGRES_USER','root')}:"
    f"{os.getenv('POSTGRES_PASSWORD','password')}@postgres/"
    f"{os.getenv('POSTGRES_DB','n8n')}",
)

# ── Runtime overrides file (shared bind-mount with ser-api) ─────────────────
RUNTIME_OVERRIDES_PATH = Path(
    os.getenv("RUNTIME_OVERRIDES_PATH", "/data/runtime/overrides.json")
)

# ── Public URLs surfaced into the UI (so links resolve in the user's browser)
SER_API_PUBLIC_URL    = os.getenv("SER_API_PUBLIC_URL",    "http://localhost:8000")
SER_RENDER_PUBLIC_URL = os.getenv("SER_RENDER_PUBLIC_URL", "http://localhost:8010")

# ── Logging ─────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# ── Model catalog (what shows up in the "swap model" dropdowns) ─────────────
# Hand-curated for now; the System page reconciles this against the
# actually-pulled Ollama models and the actually-cached HF/FunASR models.
#
# Audio models — picking a new one triggers a ser-api restart.
ASR_CHOICES = [
    "tiny", "tiny.en",
    "base", "base.en",
    "small", "small.en",
    "medium", "medium.en",
    "large-v3",
    "distil-large-v3",
]
DIARIZATION_CHOICES = [
    "pyannote/speaker-diarization-community-1",
    "pyannote/speaker-diarization-3.1",
]
SENTIMENT_CHOICES = [
    "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "cardiffnlp/twitter-xlm-roberta-base-sentiment",
    "distilbert-base-uncased-finetuned-sst-2-english",
]
SER_CHOICES = [
    "iic/emotion2vec_plus_large",
    "iic/emotion2vec_plus_base",
    "iic/emotion2vec_base_finetuned",
]

# Ollama models — picking a new one is hot (no restart). The actual list
# of installed Ollama models is also surfaced from /api/tags so you see
# what's *really* pullable on this host.
LLM_DEFAULT_CHOICES = [
    "qwen2.5:14b-instruct-q4_K_M",
    "qwen2.5:7b-instruct-q4_K_M",
    "llama3.1:8b-instruct-q4_K_M",
    "gemma2:9b-instruct-q4_K_M",
    "gemma4:e4b",
    "mistral:7b-instruct",
]
EMBEDDING_DEFAULT_CHOICES = [
    "qwen3-embedding:4b",
    "nomic-embed-text",
    "bge-m3",
    "mxbai-embed-large",
]

# ── Approximate on-disk sizes for display in the control panel ──────────────
# Hand-curated; only used to render "~140 MB" alongside the model id. Misses
# are fine — the UI just omits the size if the model isn't here.
MODEL_SIZES_MB = {
    # Whisper (faster-whisper)
    "whisper-tiny":        75,
    "whisper-tiny.en":     75,
    "whisper-base":       140,
    "whisper-base.en":    140,
    "whisper-small":      470,
    "whisper-small.en":   470,
    "whisper-medium":    1500,
    "whisper-medium.en": 1500,
    "whisper-large-v3":  3000,
    "whisper-distil-large-v3": 1500,
    # Pyannote
    "pyannote/speaker-diarization-community-1": 32,
    "pyannote/speaker-diarization-3.1":         32,
    "pyannote/wespeaker-voxceleb-resnet34-LM":  26,
    # Sentiment
    "cardiffnlp/twitter-roberta-base-sentiment-latest":   500,
    "cardiffnlp/twitter-xlm-roberta-base-sentiment":     1100,
    "distilbert-base-uncased-finetuned-sst-2-english":    270,
    # Speech emotion
    "iic/emotion2vec_plus_large":      1800,
    "iic/emotion2vec_plus_base":        300,
    "iic/emotion2vec_base_finetuned":   300,
    # NER
    "urchade/gliner_multi-v2.1": 750,
    # Ollama (approximate Q4_K_M sizes)
    "gemma4:e4b":                 9600,
    "qwen2.5:14b-instruct-q4_K_M":9500,
    "qwen2.5:7b-instruct-q4_K_M": 4900,
    "llama3.1:8b-instruct-q4_K_M":5300,
    "gemma2:9b-instruct-q4_K_M":  5800,
    "mistral:7b-instruct":        4400,
    "qwen3-embedding:4b":         2700,
    "nomic-embed-text":            274,
    "bge-m3":                     1200,
    "mxbai-embed-large":           670,
}


def model_size_mb(model_id: str) -> int:
    """Approximate on-disk size in MB for display. 0 if unknown."""
    if not model_id:
        return 0
    # Tolerate the "qwen3-embedding:4b@1024" form ser-api emits in model_versions.
    base = model_id.split("@", 1)[0]
    return MODEL_SIZES_MB.get(base, 0)

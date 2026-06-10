# syntax=docker/dockerfile:1.7
# ─────────────────────────────────────────────────────────────────────────────
#  ser-api Dockerfile v7 — Whisper-base + emotion2vec+ + GLiNER + Qwen embeddings
# ─────────────────────────────────────────────────────────────────────────────
#  Normal day-to-day:
#    docker compose --profile gpu-nvidia up -d
#
#  Rebuild image after dependency / Dockerfile changes:
#    docker compose --profile gpu-nvidia up -d --build ser-api
#
#  Pre-bake models into the image at build time (slow, ~6 GB extra in the
#  image; recommended for production deployments where first-boot
#  download time matters):
#    docker build \
#      --build-arg PRELOAD_MODELS=1 \
#      --build-arg HF_TOKEN=hf_xxx \
#      -t ser-api ./ser-api
# ─────────────────────────────────────────────────────────────────────────────

FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ── System dependencies ───────────────────────────────────────────────────────
# ffmpeg          — audio decoding everywhere
# libsndfile1     — soundfile Python package backend
# sox + libsox-* — torchaudio resampling paths (used by pyannote)
# libpq-dev       — psycopg2 build dependency
# build-essential — required by a few transitive C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        sox \
        libsox-fmt-all \
        libpq-dev \
        build-essential \
        python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

# ── Python venv (PEP 668 safe) ────────────────────────────────────────────────
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

# ── Install core CUDA audio stack ─────────────────────────────────────────────
# Keep BEFORE requirements.txt so later installs don't pull CPU-only torch.
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install \
        torchaudio==2.10.0 \
        --index-url https://download.pytorch.org/whl/cu128

# ── Install app dependencies ───────────────────────────────────────────────────
COPY call-analysis-api/requirements.txt ./requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install -r requirements.txt

# Re-pin torchvision to the CUDA build (pyannote pulls a CPU wheel transitively).
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip uninstall -y torchvision && \
    python -m pip install \
        torchvision==0.25.0 \
        --index-url https://download.pytorch.org/whl/cu128

# spaCy model required by Presidio's NER-based PII detection.
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m spacy download en_core_web_sm

# ── HuggingFace cache paths ────────────────────────────────────────────────────
ENV HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface \
    WHISPER_CACHE=/root/.cache/huggingface/whisper

# ── Optional pre-download of ML models at build time ──────────────────────────
ARG PRELOAD_MODELS=0
ARG HF_TOKEN=""
ENV HF_TOKEN=${HF_TOKEN}

RUN python - <<'PYEOF'
import os, sys

if os.getenv("PRELOAD_MODELS", "0") != "1":
    print("Skipping build-time model pre-download (PRELOAD_MODELS=0).")
    sys.exit(0)

tok = os.getenv("HF_TOKEN", "").strip()
kw  = {"token": tok} if tok else {}

print("Build-time model pre-download enabled.")

# ── Whisper-base (ASR, English-only) ──────────────────────────────────────────
print("Downloading Whisper base …")
try:
    from faster_whisper import WhisperModel
    WhisperModel("base", device="cpu", compute_type="float32",
                 download_root="/root/.cache/huggingface/whisper")
    print("  Whisper base OK")
except Exception as e:
    print(f"  Whisper base WARN: {e}")

# ── pyannote community-1 diarizer (A/B alternative to 3.1) ───────────────────
if tok:
    mid = "pyannote/speaker-diarization-community-1"
    print(f"Downloading {mid} …")
    try:
        from pyannote.audio import Pipeline
        Pipeline.from_pretrained(mid, token=tok)
        print(f"  {mid} OK")
    except Exception as e:
        print(f"  {mid} WARN: {e}")
else:
    print("  pyannote models SKIP (no HF_TOKEN provided)")

# ── XLM-RoBERTa multilingual sentiment ────────────────────────────────────────
print("Downloading XLM-R sentiment …")
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    mid = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    AutoTokenizer.from_pretrained(mid, **kw)
    AutoModelForSequenceClassification.from_pretrained(mid, **kw)
    print("  XLM-R sentiment OK")
except Exception as e:
    print(f"  XLM-R sentiment WARN: {e}")

# ── emotion2vec+ large (SER) ──────────────────────────────────────────────────
print("Downloading emotion2vec+ large …")
try:
    from funasr import AutoModel
    AutoModel(model="iic/emotion2vec_plus_large", disable_update=True,
              device="cpu")
    print("  emotion2vec+ OK")
except Exception as e:
    print(f"  emotion2vec+ WARN: {e}")

# ── GLiNER multilingual NER ───────────────────────────────────────────────────
print("Downloading GLiNER multi-v2.1 …")
try:
    from gliner import GLiNER
    GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
    print("  GLiNER OK")
except Exception as e:
    print(f"  GLiNER WARN: {e}")

print("Pre-download complete.")
PYEOF

# ── Application code ──────────────────────────────────────────────────────────
COPY call-analysis-api/app.py             ./app.py
COPY call-analysis-api/config.py          ./config.py
COPY call-analysis-api/db.py              ./db.py
COPY call-analysis-api/logging_setup.py   ./logging_setup.py
COPY call-analysis-api/schemas.py         ./schemas.py
COPY call-analysis-api/models/            ./models/
COPY call-analysis-api/pipeline/          ./pipeline/
COPY call-analysis-api/prompts/           ./prompts/
COPY call-analysis-api/tests/             ./tests/
# Shared core package (config/db/embeddings) — single source of truth.
COPY packages/ragcore/                    ./ragcore/

ENV PYTHONPATH=/app

EXPOSE 8000

# ── Runtime env defaults (all overridable in docker-compose / .env) ──────────
ENV WHISPER_MODEL_ID=base \
    WHISPER_COMPUTE=float16 \
    DIARIZATION_MODEL_ID=pyannote/speaker-diarization-community-1 \
    SENTIMENT_MODEL_ID=cardiffnlp/twitter-xlm-roberta-base-sentiment \
    SENTIMENT_BATCH_SIZE=32 \
    SER_MODEL_ID=iic/emotion2vec_plus_large \
    SER_BATCH_SIZE=8 \
    SER_SEGMENT_CAP=30 \
    MAX_PROSODY_SECONDS=300 \
    NER_MODEL_ID=urchade/gliner_multi-v2.1 \
    NER_THRESHOLD=0.45

CMD ["uvicorn", "app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--timeout-keep-alive", "300"]

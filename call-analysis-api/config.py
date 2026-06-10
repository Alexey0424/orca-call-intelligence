"""Centralised env-var parsing and tunable constants.

Every module reads config from here — never directly from ``os.environ``.
The defaults below are the recommended production picks (May 2026):
Whisper-base for ASR, emotion2vec+ for speech emotion, Qwen 2.5-14B
for the KPI/coaching LLM, Qwen3-Embedding-4B truncated to 1024-d via
MRL for retrieval, GLiNER for entity extraction.
"""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

import torch

from ragcore.config import (  # noqa: F401 — re-exported for existing call sites
    RUNTIME_OVERRIDES_PATH, _load_overrides, _resolve, _runtime_get,
    OLLAMA_HOST,
    EMBEDDING_MODEL_ID, EMBEDDING_NATIVE_DIM, EMBEDDING_DIM,
    EMBEDDING_TIMEOUT_SEC, embedding_model_id,
    SER_RENDER_URL, SER_RENDER_TIMEOUT_SEC, SER_API_PUBLIC_URL,
    GOTENBERG_URL, GOTENBERG_TIMEOUT_SEC,
    BRAND_MARK, BRAND_NAME, BRAND_SUB, BRAND_FOOTER_LEFT, BRAND_FOOTER_RIGHT,
    DATABASE_URL, DB_INIT_RETRIES, DB_INIT_BACKOFF_SEC, LOG_LEVEL,
)

_logger = logging.getLogger(__name__)

# ── Device detection ─────────────────────────────────────────────────────────
USE_GPU      = torch.cuda.is_available()
TORCH_DEVICE = torch.device("cuda" if USE_GPU else "cpu")
TORCH_DTYPE  = torch.float16 if USE_GPU else torch.float32

# ── Model lifecycle ──────────────────────────────────────────────────────────
# PRELOAD_MODELS=1: load every audio model at startup and keep it resident —
#   fastest back-to-back calls, but holds ~14 GB of VRAM at idle (best on a
#   dedicated box). This is the `preload-concurrent-mode` branch's behaviour.
# PRELOAD_MODELS=0 (default, "serialized"): do NOT keep audio models resident.
#   They lazy-load per call and are evicted again after each call, so the GPU
#   is free for other work when the pipeline is idle. Cost: a ~15-30 s model
#   reload on every call (the accepted cold-start trade for a shared workstation).
PRELOAD_MODELS = os.getenv("PRELOAD_MODELS", "0") == "1"


# ── Resource sampling (control-panel per-stage GPU/CPU/RAM peaks) ────────────
# The sampler thread polls CPU/RAM at this interval to track per-sub-stage
# high-water marks. GPU VRAM peak is exact (torch peak-stats) and does not
# depend on this. Set RESOURCE_SAMPLING_ENABLED=0 to disable the thread cheaply.
RESOURCE_SAMPLING_ENABLED   = os.getenv("RESOURCE_SAMPLING_ENABLED", "1") == "1"
RESOURCE_SAMPLE_INTERVAL_MS = int(float(os.getenv("RESOURCE_SAMPLE_INTERVAL_MS", "150")))


# ── ASR — Whisper-base via faster-whisper (English-only deployment) ──────────
# Pulled from Hugging Face on first build into WHISPER_CACHE (which is
# bound to the `hf_cache` named volume). Every call is treated as English
# (ASR_LANGUAGE is passed to model.transcribe), so Whisper skips its own
# language-ID step.
WHISPER_MODEL_ID     = _resolve("WHISPER_MODEL_ID", "base")
WHISPER_COMPUTE      = _resolve("WHISPER_COMPUTE",  "float16")
WHISPER_CACHE        = os.getenv("WHISPER_CACHE",
                                 "/root/.cache/huggingface/whisper")
ASR_LANGUAGE         = "en"


# ── Diarization ──────────────────────────────────────────────────────────────
# We default to pyannote/speaker-diarization-3.1 (not community-1) because
# on phone-call audio community-1 has a cold-start instability: in the first
# few seconds it can split a single speaker's voice across both cluster
# labels (lab measurement on the reference call: 13 speaker switches in
# the first 16 s vs 1 with 3.1). 3.1 also produces ~50% fewer micro-fragment
# turns (sub-500 ms turns: 39 with 3.1 vs 83 with community-1), which makes
# downstream reconciliation cleaner.
DIARIZATION_MODEL_ID = _resolve("DIARIZATION_MODEL_ID",
                                 "pyannote/speaker-diarization-3.1")
# pyannote batches its segmentation + embedding steps on the GPU. Left at the
# library defaults (segmentation 32 / embedding 32) it can transiently allocate
# the whole 16 GB card on a long call. These caps bound that peak — batch size
# only affects throughput, NOT diarization output, so lowering it is free
# accuracy-wise (a little slower for a much smaller VRAM peak).
DIARIZATION_SEG_BATCH = int(os.getenv("DIARIZATION_SEG_BATCH", "16"))
DIARIZATION_EMB_BATCH = int(os.getenv("DIARIZATION_EMB_BATCH", "8"))


# ── Text sentiment (English-only RoBERTa, 3-class) ───────────────────────────
# All calls in this deployment are English. The English-only Twitter
# RoBERTa is stronger than the multilingual XLM variant on English text
# and ~3× smaller. Same label set (negative/neutral/positive), drop-in.
# When you fine-tune on labeled call utterances (phase 2), set
# SENTIMENT_MODEL_ID in .env to your fine-tuned model id; model_versions
# JSONB will record the swap automatically.
SENTIMENT_MODEL_ID   = _resolve("SENTIMENT_MODEL_ID",
                                 "cardiffnlp/twitter-roberta-base-sentiment-latest")
SENTIMENT_BATCH_SIZE = int(float(os.getenv("SENTIMENT_BATCH_SIZE", "32")))


# ── Speech emotion (emotion2vec+ large, 9-class multilingual) ────────────────
# Apache-2.0, multilingual, Spanish-capable out of the box. Replaces the
# English-only ehcalabres wav2vec2 head-patch.
SER_MODEL_ID         = _resolve("SER_MODEL_ID",
                                 "iic/emotion2vec_plus_large")
SER_BATCH_SIZE       = int(float(os.getenv("SER_BATCH_SIZE", "8")))
SER_SEGMENT_CAP_SEC  = float(os.getenv("SER_SEGMENT_CAP",   "30"))


# (Voice-ID + client voice-linking config removed 2026-06-10 — the whole
#  voice-identity apparatus was retired; roles come from diarization + the
#  LLM opening relabel. History on branch turn-relabel-mode + git.)


# ── Entity extraction (GLiNER multilingual) ──────────────────────────────────
NER_MODEL_ID         = os.getenv("NER_MODEL_ID", "urchade/gliner_multi-v2.1")
# Per-call entity types to extract. Edit freely — GLiNER is zero-shot, so
# changing this list takes effect immediately without retraining.
NER_ENTITY_TYPES     = [
    "competitor",       # rival agencies / brokerage names
    "neighborhood",     # places mentioned ("San Diego", "Chula Vista")
    "mortgage_broker",  # lenders / mortgage companies
    "price_point",      # listed prices ("$650,000", "$2.4M")
    "property_type",    # "condo", "single-family", "duplex"
    "agent_name",       # named agents (closer's own name shows up here)
]
NER_THRESHOLD        = float(os.getenv("NER_THRESHOLD", "0.45"))


# ── LLM (KPI extraction + coaching rubric, Ollama) ───────────────────────────
# Default: gemma4:e4b (~9.6 GB) — fits a 16 GB card alongside the audio
# models. It is a reasoning model: kpis.py / coaching.py pin ``think: false``
# (thinking burned ~200 s/call on scratch tokens for identical JSON) and keep
# a content→thinking fallback when parsing. Both are required — don't remove.
# (OLLAMA_HOST is re-exported from ragcore.config above.)
# Module-level default — for snapshots / logs / metadata. Pipeline code
# that calls Ollama should use ``llm_model_id()`` so runtime swaps from
# the control UI take effect without restarting the service.
LLM_MODEL_ID         = _resolve("LLM_MODEL_ID", "gemma4:e4b")
LLM_TIMEOUT_SEC      = float(os.getenv("LLM_TIMEOUT_SEC",  "180"))
LLM_NUM_CTX          = int(os.getenv("LLM_NUM_CTX",        "16384"))
LLM_NUM_PREDICT      = int(os.getenv("LLM_NUM_PREDICT",     "4096"))


def llm_model_id() -> str:
    """Resolve the LLM model id at call time. Honours runtime overrides."""
    return _runtime_get("LLM_MODEL_ID", "gemma4:e4b")


# ── Opening-of-call LLM speaker→role relabel (pipeline/opening_relabel.py) ────
# DiarizationLM-style post-processing: read the opening transcript and let the
# LLM decide which diarized speaker is the closer/client, fixing pyannote's
# cold-start mislabel that the interim "first speaker = client" rule gets wrong.
# ON by default since 2026-06-10 — this is the validated core of role
# assignment (acoustic tweaks topped out at client-recall 0.375 on the
# annotated calls; the LLM relabel scores 1.0 — see
# experiments/asr_diarization_eval + data/diag/OPENING_RESEARCH.md).
# Env default for the enabled flag; the live value is resolved per request via
# opening_relabel_enabled() so the control UI can toggle it through the shared
# overrides file (same hot-swap path as the LLM/embedding model picks).
_OPENING_RELABEL_ENABLED_ENV = os.getenv("OPENING_RELABEL_ENABLED", "1") == "1"
OPENING_RELABEL_WINDOW_SEC   = float(os.getenv("OPENING_RELABEL_WINDOW_SEC", "60"))


def opening_relabel_enabled() -> bool:
    """Per-request resolver — honours a runtime override so the control UI can
    flip the feature on/off without a restart. Falls back to the env default."""
    v = _runtime_get("OPENING_RELABEL_ENABLED",
                     "1" if _OPENING_RELABEL_ENABLED_ENV else "0")
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def opening_relabel_model() -> str:
    """Model for the opening relabel. qwen2.5:14b is the research winner for
    this task (client recall 1.0, agent correct every run); gemma4 (the KPI
    model) occasionally emits a null agent, so it serves as the in-code retry
    fallback instead (pipeline/opening_relabel.py). Resolution: override file
    (control UI) > OPENING_RELABEL_MODEL env > KPI model."""
    return _runtime_get("OPENING_RELABEL_MODEL", llm_model_id())


# Per-role lookup table for the audio-model hot-swap registry. Keys must
# stay in sync with the slots registered in ``models/registry.py`` and the
# form field names on the Models page (``WHISPER_MODEL_ID`` etc.).
_AUDIO_ROLE_TO_ENV: dict = {
    "asr":         ("WHISPER_MODEL_ID",     "base"),
    "diarization": ("DIARIZATION_MODEL_ID", "pyannote/speaker-diarization-3.1"),
    "sentiment":   ("SENTIMENT_MODEL_ID",   "cardiffnlp/twitter-roberta-base-sentiment-latest"),
    "ser":         ("SER_MODEL_ID",         "iic/emotion2vec_plus_large"),
}


def current_audio_model_id(role: str) -> str:
    """Per-request resolver for audio model IDs. Honours runtime overrides.

    Unlike the module-level constants (``WHISPER_MODEL_ID`` etc.) which are
    frozen at import, this re-reads the override file on every call so the
    control UI can hot-swap audio models without restarting ser-api.
    """
    if role not in _AUDIO_ROLE_TO_ENV:
        raise KeyError(f"unknown audio role: {role!r}")
    env_name, default = _AUDIO_ROLE_TO_ENV[role]
    return _runtime_get(env_name, default)


# (EMBEDDING_MODEL_ID, EMBEDDING_NATIVE_DIM, EMBEDDING_DIM, EMBEDDING_TIMEOUT_SEC,
#  embedding_model_id() are re-exported from ragcore.config above.)

# ── HF auth ──────────────────────────────────────────────────────────────────
HF_TOKEN             = os.getenv("HF_TOKEN", "") or None


# (SER_RENDER_URL, SER_RENDER_TIMEOUT_SEC, SER_API_PUBLIC_URL, GOTENBERG_URL,
#  GOTENBERG_TIMEOUT_SEC, BRAND_*, DATABASE_URL, DB_INIT_RETRIES,
#  DB_INIT_BACKOFF_SEC, LOG_LEVEL are re-exported from ragcore.config above.)

# ── Pipeline thresholds (utterance grouping — core reconciliation, not tweaks) ─
MAX_INTRA_SPEAKER_GAP_SEC  = 2.0    # >2 s gap inside one speaker → new utterance
SENTENCE_END_GAP_SEC       = 0.3    # gap after sentence-end punctuation
SENTENCE_END_PATTERN       = r"[.?!]$"

# ── Reconciliation tweaks (OFF by default since 2026-06-10) ──────────────────
# A/B'd against the hand-labeled calls (experiments/asr_diarization_eval run
# 20260602_120508): the short-run collapse HURTS opening client recall (it
# absorbs real short client turns into the closer's run — Alicia 0.50 → 0.00),
# and the word silence-split is marginal (+0.25 client recall on one call).
# Both are superseded by the LLM opening relabel (client recall 1.0). The
# production methodology is therefore bone-stock reconciliation (max-overlap +
# midpoint/nearest fallback + grouping) + the LLM opening relabel. The code
# stays env-tunable as an escape hatch: WORD_SILENCE_SPLIT_ENABLED=1 /
# MIN_INTRA_TURN_RUN_WORDS=3 re-enable the old behaviour.
WORD_SILENCE_SPLIT_ENABLED          = os.getenv("WORD_SILENCE_SPLIT_ENABLED", "0") == "1"
WORD_SILENCE_MIN_SUSPICIOUS_DUR_SEC = float(os.getenv("WORD_SILENCE_MIN_SUSPICIOUS_DUR_SEC", "0.7"))
WORD_SILENCE_GAP_SEC                = float(os.getenv("WORD_SILENCE_GAP_SEC", "0.20"))
WORD_SILENCE_PEAK_THRESHOLD         = float(os.getenv("WORD_SILENCE_PEAK_THRESHOLD", "0.15"))
WORD_SILENCE_FRAME_MS               = float(os.getenv("WORD_SILENCE_FRAME_MS", "10"))
MIN_INTRA_TURN_RUN_WORDS            = int(os.getenv("MIN_INTRA_TURN_RUN_WORDS", "1"))

# (TURN_RELABEL_* / ERES2NETV2_* config removed 2026-06-10 with the voice-ID
#  teardown — the per-turn voice relabel lives on branch turn-relabel-mode.)

MIN_SER_SLICE_SEC          = 0.5    # too-short slice → skip SER classification

SENTIMENT_SLOPE_THRESHOLD  = 0.005  # slope cut-off for "improving/declining"
AROUSAL_SLOPE_THRESHOLD    = 0.01

MAX_PROSODY_SECONDS        = float(os.getenv("MAX_PROSODY_SECONDS", "300"))

# Last-resort closer label when both the LLM opening relabel and the
# first-speaker rule fail (this is the diarizer label, not a name).
DEFAULT_CLOSER_LABEL       = "SPEAKER_01"

# Hard caps on /scorecard intake. Beyond these the API returns 413 instead
# of attempting to load the audio (protects GPU + disk under hostile or
# accidentally-large uploads).
MAX_UPLOAD_BYTES           = int(os.getenv("MAX_UPLOAD_BYTES",       str(200 * 1024 * 1024)))  # 200 MB
MAX_AUDIO_DURATION_SEC     = float(os.getenv("MAX_AUDIO_DURATION_SEC", "7200"))                # 2 hours

# Shared-secret header for the small set of admin endpoints
# (right-to-be-forgotten + annotation writes). When unset (empty) the
# endpoints stay open — fine for the loopback-only deployment.
ADMIN_API_KEY              = (os.getenv("ADMIN_API_KEY", "") or "").strip()

# ── Chat priority (call-processing yields to an active live chat) ────────────
# A chat front-end marks `runtime_settings.chat_active_until` while handling a
# message; /scorecard waits (async, BEFORE the GPU lock so the chat's own API
# calls aren't blocked) until it clears, so the live human gets the GPU +
# Ollama first. Capped so a crashed chat service can't stall calls beyond the
# TTL it set. Defers a call's START; doesn't preempt a call already in flight.
# No bundled service sets the key in this stack — compose disables the gate.
CHAT_PRIORITY_ENABLED      = (os.getenv("CHAT_PRIORITY_ENABLED", "1").strip().lower()
                              not in ("0", "false", "no"))
CHAT_PRIORITY_POLL_SEC     = float(os.getenv("CHAT_PRIORITY_POLL_SEC",     "2"))
CHAT_PRIORITY_MAX_WAIT_SEC = float(os.getenv("CHAT_PRIORITY_MAX_WAIT_SEC", "300"))

# Directories that contain per-call artefacts. The RTBF endpoint sweeps
# both when it removes a call. Falls back to /data — same as where the
# pipeline writes today.
DATA_DIR                   = os.getenv("DATA_DIR", "/data")

# VAD projection version stamp. Bump whenever the coefficients in
# pipeline.emotion.derive_vad change so historical rows remain auditable.
VAD_VERSION                = "vad-v1-2026-05"


# ── Paths ────────────────────────────────────────────────────────────────────
PACKAGE_DIR = Path(__file__).parent
PROMPTS_DIR = PACKAGE_DIR / "prompts"


# ── Prompt / projection version hashes (CF-9) ────────────────────────────────
# Persisted alongside model_ids so we can prove which prompt produced any
# historical row. Editing prompts/kpi_system.txt shifts the hash; editing
# pipeline.coaching._SYSTEM_PROMPT shifts the coaching hash. Both are
# computed lazily so an import error in coaching.py never blocks startup.

def _file_sha(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    except Exception:  # noqa: BLE001 — best-effort lineage stamp
        return "unknown"


def kpi_prompt_sha() -> str:
    return _file_sha(PROMPTS_DIR / "kpi_system.txt")


def coaching_prompt_sha() -> str:
    """Hash the in-code coaching system prompt. Lazy import to avoid cycles."""
    try:
        from pipeline.coaching import _SYSTEM_PROMPT
        return hashlib.sha256(_SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:12]
    except Exception:  # noqa: BLE001
        return "unknown"


# ── Model version stamp (written to every `calls` row) ──────────────────────
def model_versions() -> dict:
    """Snapshot of the model IDs + prompt hashes that produced this row.

    Persisted into ``calls.model_versions`` so historical rows always know
    which pipeline produced them. When a model is swapped or a prompt
    edited, future rows carry the new id/hash; old rows stay accurate.

    Every audio + Ollama model is resolved live (via the runtime override
    file) so a hot-swap is reflected in the snapshot for the next call,
    without needing a process restart.
    """
    return {
        "asr":              f"whisper-{current_audio_model_id('asr')}",
        "diarization":      current_audio_model_id("diarization"),
        "sentiment":        current_audio_model_id("sentiment"),
        "ser":              current_audio_model_id("ser"),
        "ner":              NER_MODEL_ID,
        "llm":              llm_model_id(),
        "embedding":        f"{embedding_model_id()}@{EMBEDDING_DIM}",
        "kpi_prompt":       kpi_prompt_sha(),
        "coaching_prompt":  coaching_prompt_sha(),
        "vad":              VAD_VERSION,
    }

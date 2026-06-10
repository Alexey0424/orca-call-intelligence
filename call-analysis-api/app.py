"""FastAPI entrypoint for ``call-analysis-api``.

Audio models live in a single-slot-per-role registry (``models.registry``)
that supports hot-swap from the control UI without a container restart.
Each handler calls ``registry.get(role)`` inside the GPU lock; if the
configured model id has changed since the last call, the registry
evicts the previous tenant and loads the new one in place.

English-only deployment: ASR is Whisper-base (via faster-whisper).
There is no language detection step — every call is dispatched with
``language="en"`` so Whisper skips its own language-ID and decodes as
English directly.
"""
from __future__ import annotations

import asyncio
import base64
import binascii
import gc
import hashlib
import logging
import os
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import hmac
import json
import shutil

import httpx
import torch
from fastapi import (Body, Depends, FastAPI, File, Header, HTTPException,
                     Path, Query, UploadFile)
from fastapi.responses import Response

import config
import db
import logging_setup
from models import registry
from pipeline import (audio, chat_priority,
                      diarization, jobs, persistence, pii, reconciliation,
                      resource_sampler, scorecard, transcription)

logger = logging.getLogger(__name__)


# Audio models are now owned by ``models.registry`` (single-slot per role,
# hot-swappable from the control UI without a container restart). The four
# old module globals (_asr_model, _diar_pipe, _sentiment_pipeline,
# _sentiment_id2label, _ser_model, _ser_labels) were removed during the
# May 2026 hot-swap refactor — read via ``registry.get(role)`` instead.

# CF-10: Serialize the GPU-bound section of /scorecard. Models are loaded
# as module singletons and pyannote/Whisper/emotion2vec aren't safe for
# concurrent inference on the same CUDA device. The lock is held across
# the whole pipeline (cheap relative to ASR/LLM cost) so cross-talk is
# impossible. Concurrent requests just queue.
_GPU_LOCK = threading.Lock()


def _enable_tf32_if_gpu() -> None:
    if config.USE_GPU:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
        torch.backends.cudnn.benchmark        = True


# ── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(_app: FastAPI):
    logging_setup.configure()
    logger.info("starting service=call-analysis-api use_gpu=%s asr=whisper:%s",
                config.USE_GPU, config.WHISPER_MODEL_ID)
    db.ensure_schema()
    _enable_tf32_if_gpu()

    # Model lifecycle (see config.PRELOAD_MODELS):
    #   PRELOAD_MODELS=1 → load every audio model now and keep it resident
    #     (fast back-to-back calls; ~14 GB held at idle). preload-concurrent-mode.
    #   PRELOAD_MODELS=0 (default, serialized) → don't keep them resident; models
    #     lazy-load per call and are evicted after each call (see /scorecard), so
    #     the GPU is free for other work when idle.
    if config.PRELOAD_MODELS:
        preload_summary = registry.preload_all()
        for role, info in preload_summary.items():
            if info.get("ok"):
                logger.info("preloaded role=%s id=%s load_sec=%.2f",
                            role, info["model_id"], info["load_sec"])
            else:
                logger.error("preload_failed role=%s err=%s",
                             role, info.get("error"))
    else:
        logger.info("serialized_model_mode PRELOAD_MODELS=0 — audio models "
                    "lazy-load + evict per call; GPU free when idle")

    # GPU warm-up (both modes): drive one throwaway Whisper inference so the
    # process/system pays the cold first-inference cost ONCE, here. Measured:
    # the FIRST call after a full host reboot took ~735 s in ASR vs ~22 s warm,
    # the penalty concentrated entirely in ASR. That cold cost is system-level
    # (a warm container restart's first transcribe is already fast), so warming
    # once at startup makes every later call fast. In serialized mode we warm
    # and then immediately evict: the system stays warm but idle VRAM is freed,
    # so the first real call's reload is a quick ~load, not the 735 s cold init.
    # vad_filter off so the synthetic noise isn't skipped as silence.
    # Best-effort: any failure is logged and never blocks startup.
    try:
        import numpy as _np
        _t0 = time.time()
        _noise = (0.02 * _np.random.randn(16000 * 5)).astype("float32")
        with _GPU_LOCK:
            _segs, _ = registry.asr().transcribe(
                _noise, language=config.ASR_LANGUAGE,
                vad_filter=False, word_timestamps=True,
                beam_size=5, temperature=[0.0],
            )
            for _ in _segs:          # generator is lazy — consume to run it
                pass
        logger.info("warmup_asr_done sec=%.1f", time.time() - _t0)
        if not config.PRELOAD_MODELS:
            registry.unload_all()    # free the VRAM; system CUDA stays warm
    except Exception:
        logger.exception("warmup_asr_failed (non-fatal)")

    # PII warm-up: presidio + spaCy cold-load costs ~45 s and was previously
    # paid by the FIRST call's persist stage. Loading here moves that cost to
    # startup. CPU/RAM-resident only — holds no VRAM, so it doesn't conflict
    # with the serialized GPU lifecycle. Best-effort.
    try:
        _t0 = time.time()
        pii.redact_one("warmup: John Smith, 555-123-4567, j.smith@example.com")
        logger.info("warmup_pii_done sec=%.1f", time.time() - _t0)
    except Exception:
        logger.exception("warmup_pii_failed (non-fatal)")

    # Stuck-jobs reaper: demotes orphan `running` rows left behind by
    # SIGKILL'd /scorecard requests so the visibility table stays honest.
    loop = asyncio.get_running_loop()
    reaper_task = jobs.start_reaper(loop, stuck_minutes=30, interval_sec=300.0)

    logger.info("startup_complete")
    try:
        yield
    finally:
        reaper_task.cancel()
        try:
            await reaper_task
        except asyncio.CancelledError:
            pass
        logger.info("shutdown")


app = FastAPI(title="call-analysis-api", version="7.0", lifespan=lifespan)


# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/")
def root() -> Dict[str, Any]:
    return {"ok": True, "service": "call-analysis-api", "version": "7.0"}


@app.get("/health")
def health() -> Dict[str, Any]:
    """Deep health check: GPU + DB + Ollama reachability.

    Returns 200 only when all dependencies are reachable. Caller can
    treat any non-200 as "do not route work to me right now."
    """
    gpu: Dict[str, Any] = {}
    if config.USE_GPU:
        p = torch.cuda.get_device_properties(0)
        gpu = {
            "name":             p.name,
            "vram_total_gb":    round(p.total_memory / 1e9, 2),
            "vram_alloc_gb":    round(torch.cuda.memory_allocated(0) / 1e9, 2),
            "vram_reserved_gb": round(torch.cuda.memory_reserved(0)  / 1e9, 2),
        }

    deps: Dict[str, Any] = {}

    try:
        with db.connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        deps["postgres"] = {"ok": True}
    except Exception as exc:  # noqa: BLE001
        deps["postgres"] = {"ok": False, "error": str(exc)[:200]}

    try:
        resp = httpx.get(f"{config.OLLAMA_HOST}/api/tags", timeout=3.0)
        resp.raise_for_status()
        deps["ollama"] = {"ok": True, "models": len((resp.json() or {}).get("models", []))}
    except Exception as exc:  # noqa: BLE001
        deps["ollama"] = {"ok": False, "error": str(exc)[:200]}

    all_ok = all(d.get("ok") for d in deps.values())
    body: Dict[str, Any] = {
        "ok":     all_ok,
        "gpu":    gpu,
        "deps":   deps,
        "models": config.model_versions(),
    }
    if not all_ok:
        # Use HTTPException so monitors see a non-200.
        raise HTTPException(status_code=503, detail=body)
    return body


@app.get("/jobs")
def jobs_endpoint(
    status: Optional[str] = Query(None, pattern="^(running|done|failed)$"),
    limit:  int           = Query(50, ge=1, le=500),
) -> Dict[str, Any]:
    """Most-recent jobs, optionally filtered by status.

    Operator uses this to see whether a /scorecard is wedged (status
    'running' for unusually long) or which stage failures cluster in.
    """
    return {"ok": True, "jobs": jobs.list_recent(status=status, limit=limit)}


@app.get("/jobs/all")
def jobs_all_endpoint(
    status: Optional[str] = Query(None, pattern="^(running|done|failed)$"),
    limit:  int           = Query(500, ge=1, le=5000),
    offset: int           = Query(0,   ge=0),
) -> Dict[str, Any]:
    """Every job (paginated). Used by the control UI's jobs table."""
    return {"ok": True,
            "jobs": jobs.list_all(status=status, limit=limit, offset=offset)}


@app.get("/jobs/current")
def jobs_current_endpoint() -> Dict[str, Any]:
    """Best candidate for the control UI's 'Current' panel:
    most-recent running job, or — if none — the most-recent job overall."""
    row = jobs.current()
    return {"ok": True, "job": row}


@app.get("/jobs/run-index")
def jobs_run_index_endpoint(limit: int = Query(500, ge=1, le=5000)) -> Dict[str, Any]:
    """Compact list of (run_no, status, file_name, started_at) for the
    control UI's History dropdown. Latest run first."""
    return {"ok": True, "runs": jobs.run_no_index(limit=limit)}


@app.get("/jobs/by-run/{run_no}")
def job_by_run_endpoint(run_no: int) -> Dict[str, Any]:
    """Single job by sequential run number."""
    row = jobs.get_by_run_no(run_no)
    if row is None:
        raise HTTPException(status_code=404, detail="run not found")
    return {"ok": True, "job": row}


@app.get("/jobs/{job_id}")
def job_one_endpoint(job_id: str) -> Dict[str, Any]:
    """Single job by id (for control-UI drill-down)."""
    row = jobs.get_one(job_id)
    if row is None:
        raise HTTPException(status_code=404, detail="job not found")
    return {"ok": True, "job": row}


# ── Admin: wipe SER data ────────────────────────────────────────────────────
# CF-7: gate hard-destructive admin actions behind an explicit token in
# the request body. The control UI prompts the operator to type "DELETE"
# into a textbox; that string flows here and must match. Anything else
# returns 400 without touching the DB.
@app.post("/admin/wipe")
def admin_wipe_endpoint(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    confirm = (payload or {}).get("confirm") or ""
    if confirm != "DELETE":
        raise HTTPException(
            status_code=400,
            detail="confirmation required: POST {\"confirm\": \"DELETE\"}",
        )

    before: Dict[str, int] = {}
    after:  Dict[str, int] = {}
    try:
        with db.connect() as conn:
            conn.autocommit = True
            with conn.cursor() as cur:
                for tbl in ("calls", "utterances", "call_events", "pipeline_runs"):
                    cur.execute(f"SELECT COUNT(*) FROM {tbl}")
                    before[tbl] = int(cur.fetchone()[0])
                # CASCADE handles the utterances/call_events FKs in one shot.
                # RESTART IDENTITY resets pipeline_runs_run_no_seq + the
                # utterances/call_events PK sequences.
                cur.execute(
                    "TRUNCATE call_events, utterances, calls, pipeline_runs "
                    "RESTART IDENTITY CASCADE;"
                )
                for tbl in ("calls", "utterances", "call_events", "pipeline_runs"):
                    cur.execute(f"SELECT COUNT(*) FROM {tbl}")
                    after[tbl] = int(cur.fetchone()[0])
    except Exception as exc:  # noqa: BLE001
        logger.exception("admin_wipe_failed")
        raise HTTPException(status_code=500, detail=repr(exc)) from exc

    logger.warning("admin_wipe_done before=%s after=%s", before, after)
    return {"ok": True, "before": before, "after": after}


# ── Admin: hot-swap audio models without a container restart ────────────────
@app.post("/admin/reload-model")
def admin_reload_model(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """Re-resolve the configured audio model for ``role`` and swap if changed.

    The control UI calls this after writing the runtime-overrides file —
    one POST per role that the operator just changed. The registry handles
    eviction + load; the GPU lock keeps any concurrent /scorecard from
    seeing a half-evicted model.
    """
    role = (payload or {}).get("role") or ""
    if role not in ("asr", "diarization", "sentiment", "ser"):
        raise HTTPException(status_code=400,
                            detail=f"invalid role: {role!r}")
    try:
        # Hold the GPU lock for the duration of the swap so we never
        # evict a tensor mid-inference for another request.
        with _GPU_LOCK:
            result = registry.reload_if_changed(role)
    except Exception as exc:  # noqa: BLE001
        logger.exception("reload_model_failed role=%s", role)
        raise HTTPException(status_code=500, detail=repr(exc)) from exc
    return {"ok": True, **result, "loaded": registry.snapshot()}


# ── Admin: HF / FunASR disk-cache status for the audio dropdowns ────────────
# Cache layouts inside ser-api (probed at first boot):
#
#   /root/.cache/huggingface/hub/         — pyannote, sentiment, GLiNER
#   /root/.cache/huggingface/whisper/     — faster-whisper (its own download_root)
#   /root/.cache/modelscope/hub/models/   — emotion2vec via FunASR
#   /root/.cache/modelscope/modelscope/hub/models/  — older modelscope layout
#
# We probe every candidate directory for the requested model id and return
# the first hit. Misses are reported as ``cached=False`` with no path hint.
_HF_HUB_DIR     = os.getenv("HF_CACHE_DIR",
                             "/root/.cache/huggingface/hub")
_WHISPER_DIR    = config.WHISPER_CACHE   # /root/.cache/huggingface/whisper
_MODELSCOPE_DIRS = (
    "/root/.cache/modelscope/hub/models",
    "/root/.cache/modelscope/modelscope/hub/models",   # older modelscope
)


def _dir_size_bytes(path: str) -> int:
    """Best-effort recursive directory size. Returns 0 on any error."""
    if not os.path.isdir(path):
        return 0
    total = 0
    try:
        for root, _dirs, files in os.walk(path):
            for name in files:
                fp = os.path.join(root, name)
                try:
                    total += os.path.getsize(fp)
                except OSError:
                    pass
    except OSError:
        return total
    return total


def _hf_cache_status(model_id: str) -> Dict[str, Any]:
    """For an HF model id (``org/repo`` or a bare Whisper variant like
    ``base``), return whether it's downloaded and on-disk size.

    Whisper bare names live in their own ``WHISPER_CACHE`` root because
    faster-whisper sets ``cache_dir=WHISPER_CACHE`` when calling
    ``snapshot_download``. Other HF models use the standard hub layout.
    """
    if "/" not in model_id:
        # Whisper variant. faster-whisper pulls Systran's pre-converted
        # CT2 weights into its own cache root.
        slug = f"models--Systran--faster-whisper-{model_id}"
        for base in (_WHISPER_DIR, _HF_HUB_DIR):
            path = os.path.join(base, slug)
            if os.path.isdir(path):
                return {"cached": True,
                        "size_bytes": _dir_size_bytes(path),
                        "path_hint": path}
        return {"cached": False, "size_bytes": 0, "path_hint": None}

    # Plain HF repo id.
    slug = "models--" + model_id.replace("/", "--")
    path = os.path.join(_HF_HUB_DIR, slug)
    if os.path.isdir(path):
        return {"cached": True,
                "size_bytes": _dir_size_bytes(path),
                "path_hint": path}
    return {"cached": False, "size_bytes": 0, "path_hint": None}


def _modelscope_cache_status(model_id: str) -> Dict[str, Any]:
    """FunASR / emotion2vec checkpoints land in the modelscope hub layout
    at ``{base}/{repo_id}/`` — where ``repo_id`` is e.g. ``iic/emotion2vec_plus_large``.
    """
    for base in _MODELSCOPE_DIRS:
        path = os.path.join(base, model_id)
        if os.path.isdir(path):
            return {"cached": True,
                    "size_bytes": _dir_size_bytes(path),
                    "path_hint": path}
    return {"cached": False, "size_bytes": 0, "path_hint": None}


@app.get("/admin/cache-status")
def admin_cache_status(
    asr_ids:    Optional[str] = Query(None,
        description="Comma-separated Whisper variant names to probe (e.g. 'base,small,large-v3')"),
    diar_ids:   Optional[str] = Query(None,
        description="Comma-separated diarization repo ids"),
    sent_ids:   Optional[str] = Query(None,
        description="Comma-separated sentiment repo ids"),
    ser_ids:    Optional[str] = Query(None,
        description="Comma-separated emotion2vec repo ids"),
) -> Dict[str, Any]:
    """Return per-model cache state so the control UI can show
    'downloaded ✓' / 'needs download' next to each audio dropdown option.

    Designed for one call per page load — the UI passes every option in
    one of its dropdowns and gets back one verdict per id.
    """
    def _split(s: Optional[str]) -> list:
        return [x.strip() for x in (s or "").split(",") if x.strip()]

    asr_list  = _split(asr_ids)
    diar_list = _split(diar_ids)
    sent_list = _split(sent_ids)
    ser_list  = _split(ser_ids)

    return {
        "ok": True,
        "loaded": registry.snapshot(),
        "asr":         {mid: _hf_cache_status(mid)         for mid in asr_list},
        "diarization": {mid: _hf_cache_status(mid)         for mid in diar_list},
        "sentiment":   {mid: _hf_cache_status(mid)         for mid in sent_list},
        "ser":         {mid: _modelscope_cache_status(mid) for mid in ser_list},
    }


@app.post("/scorecard")
async def scorecard_endpoint(
    file:                   UploadFile = File(...),
    num_speakers:           int  = 2,
    closer_speaker:         str  = config.DEFAULT_CLOSER_LABEL,
    include_utterances:     bool = True,
    include_audio_features: bool = True,
    persist:                bool = Query(False, description="Upsert into `calls` + embed summary"),
    index_utterances:       bool = Query(False, description="Also embed + index every utterance"),
    pdf_url:                Optional[str] = Query(None, description="Render PDF URL to store on the row"),
    html_url:               Optional[str] = Query(None, description="Render HTML URL to store on the row"),
    call_id:                Optional[str] = Query(None,
        description="Optional caller-supplied call_id. If absent, derived as "
                    "sha256(file_bytes)[:16] for deterministic re-ingestion."),
) -> Dict[str, Any]:
    if num_speakers < 1:
        raise HTTPException(status_code=400, detail="num_speakers must be >= 1")

    # Chat priority: yield to an active live chat before any GPU work
    # (async, so the chat's own API calls aren't blocked). Defers the
    # call's start; doesn't preempt a call already in flight. Disabled by
    # default in this stack (CHAT_PRIORITY_ENABLED=0).
    await chat_priority.await_chat_idle()

    t0 = time.time()
    t0_mono = time.monotonic()
    job_id = jobs.start(kind="scorecard",
                        metadata={"file_name": file.filename,
                                  "models":    config.model_versions()})

    try:
        with tempfile.TemporaryDirectory() as td:
            raw_path = audio.save_upload(file, td)
            # CF-7: hard cap on upload size — protects GPU + disk from DoS.
            try:
                raw_size = os.path.getsize(raw_path)
            except OSError:
                raw_size = -1
            if raw_size > config.MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=(f"file_size={raw_size} bytes exceeds "
                            f"MAX_UPLOAD_BYTES={config.MAX_UPLOAD_BYTES}"),
                )

            # CF-4: deterministic call_id from raw upload content hash. The
            # caller may override (e.g. to backfill a known canonical id);
            # otherwise uploading the same file twice updates the same row
            # instead of producing duplicates.
            file_hash = _sha256_file(raw_path)
            if not call_id:
                call_id = file_hash[:16]

            logger.info("scorecard_start call_id=%s job_id=%s file=%s "
                        "size_bytes=%d hash=%s",
                        call_id, job_id, file.filename, raw_size, file_hash[:16])
            jobs.update_stage(job_id, "convert")

            with resource_sampler.track() as _tr:
                wav_path = audio.to_wav_16k_mono(raw_path, td)
            jobs.record_stage_peaks(job_id, "convert", _tr.peaks)
            t_convert = time.time()

            duration_sec = audio.wav_duration(wav_path)
            # CF-7: hard cap on call duration — protects GPU from multi-hour OOMs.
            if duration_sec > config.MAX_AUDIO_DURATION_SEC:
                raise HTTPException(
                    status_code=413,
                    detail=(f"duration_sec={duration_sec:.1f} exceeds "
                            f"MAX_AUDIO_DURATION_SEC={config.MAX_AUDIO_DURATION_SEC}"),
                )

            # CF-10: serialize GPU work. The lock covers ASR, diarization,
            # SER, sentiment, prosody — every model that touches the GPU.
            # Concurrent /scorecard requests queue here instead of racing.
            with _GPU_LOCK:
                # Per-stage model lifecycle. Each audio model is loaded right
                # before its stage and (in serialized mode) evicted right after,
                # so at most ONE big model sits in VRAM at a time. This bounds
                # the whole-call peak — the old code resolved all four audio
                # models up front and held them resident for the whole call, so
                # diarization (and later the gemma KPI call) ran on top of a
                # ~5-6 GB model stack and pushed the card past 14 GB. ``get()``
                # is still the swap point for control-UI hot-swaps; in preload
                # mode (PRELOAD_MODELS=1) the models are resident and we keep
                # them — no per-stage eviction.
                _serialized = not config.PRELOAD_MODELS

                jobs.update_stage(job_id, "asr")
                asr_model = registry.asr()
                with resource_sampler.track() as _tr:
                    words = transcription.transcribe(asr_model, wav_path)
                jobs.record_model(job_id, "asr",
                                  f"whisper-{config.current_audio_model_id('asr')}",
                                  "asr", _tr.elapsed, peaks=_tr.peaks)
                t_asr = time.time()
                asr_model = None
                if _serialized:
                    registry.evict("asr")
                _relax_memory()

                jobs.update_stage(job_id, "diarize")
                diar_pipe = registry.diarization()
                with resource_sampler.track() as _tr:
                    turns = diarization.diarize(diar_pipe, wav_path, num_speakers=num_speakers)
                jobs.record_model(job_id, "diarize",
                                  config.current_audio_model_id("diarization"),
                                  "diarization", _tr.elapsed, peaks=_tr.peaks)
                t_diarize = time.time()
                diar_pipe = None
                if _serialized:
                    registry.evict("diarization")
                _relax_memory()

                # Load the waveform once and reuse it for the scorecard stages.
                y, sr = audio.load_wav_float32(wav_path)

                utterances = reconciliation.merge(words, turns, y, sr)

                # scorecard.build pulls sentiment + emotion from the registry
                # itself and evicts each after use (serialized mode), then runs
                # the gemma KPI call with no audio model resident.
                jobs.update_stage(job_id, "scorecard")
                sc = scorecard.build(
                    wav_path=wav_path,
                    y=y, sr=sr, duration_sec=duration_sec,
                    utterances=utterances, turns=turns,
                    closer_speaker=closer_speaker,
                    include_prosody=include_audio_features,
                    job_id=job_id,
                )
                t_scorecard = time.time()

        # CF-5: do not persist with NULL KPIs when Ollama failed — surface
        # the failure explicitly so the caller can retry or alert.
        kpi_status = (sc.get("kpis") or {}).get("ok")
        if kpi_status is False:
            kpi_err = (sc.get("kpis") or {}).get("error", "unknown")
            logger.error("scorecard_kpi_failure call_id=%s error=%s",
                         call_id, kpi_err)
            raise HTTPException(
                status_code=503,
                detail=f"LLM KPI extraction failed: {kpi_err}",
            )

        identity = sc.get("identity", {}) or {}
        response: Dict[str, Any] = {
            "ok": True,
            "metadata": {
                "call_id":                call_id,
                "job_id":                 job_id,
                "file_name":              file.filename,
                "file_hash":              file_hash,
                "language":               config.ASR_LANGUAGE,
                "num_speakers_requested": num_speakers,
                "closer_speaker":         identity.get("closer_speaker", closer_speaker),
                "closer_name":            identity.get("closer_name"),
                "closer_confidence":      identity.get("closer_confidence"),
                "client_speaker":         identity.get("client_speaker"),
                "client_name":            identity.get("client_name"),
                "model_versions":         config.model_versions(),
            },
            "timings_sec": {
                "convert":      round(t_convert  - t0,         3),
                "asr":          round(t_asr      - t_convert,  3),
                "diarize":      round(t_diarize  - t_asr,      3),
                "scorecard_ml": round(t_scorecard - t_diarize, 3),
                "total":        round(time.time() - t0,        3),
            },
            "scorecard": sc,
        }
        if include_utterances:
            response["utterances"] = utterances

        # Serialized mode: the audio models (ASR/diarization/sentiment/SER) have
        # finished and are NOT used in persist. Evict them now, before persist,
        # so the GPU embedder + GLiNER can run on the GPU alongside the still-
        # resident gemma (used by KPI earlier + coaching shortly) without
        # overcommitting the 16 GB card. Without this, loading the embedder on
        # GPU evicts gemma → coaching then reloads it into a contended card and
        # spills to CPU (~300 s). gemma is left resident on purpose.
        if persist and not config.PRELOAD_MODELS:
            try:
                with _GPU_LOCK:
                    registry.unload_all()
                    import gc as _gc
                    import torch as _torch
                    _gc.collect()
                    if _torch.cuda.is_available():
                        _torch.cuda.empty_cache()
                logger.info("serialized_pre_persist_unload call_id=%s", call_id)
            except Exception:
                logger.exception("serialized_pre_persist_unload_failed (non-fatal)")

        if persist:
            jobs.update_stage(job_id, "persist")
            # Per-model peaks (embedding/ner/coaching) are sampled inside
            # persistence.persist; no stage-level wrap here (it would nest the
            # GPU peak-stats reset and corrupt the inner sub-stage readings).
            response["persistence"] = persistence.persist(
                call_id=call_id, file_name=file.filename,
                file_hash=file_hash,
                scorecard=sc, utterances=utterances,
                pdf_url=pdf_url, html_url=html_url,
                index_utterances=index_utterances,
                job_id=job_id,
            )

        jobs.finish(job_id, status="done", call_id=call_id,
                    started_monotonic=t0_mono)
        logger.info("scorecard_done call_id=%s job_id=%s total_sec=%.2f persisted=%s",
                    call_id, job_id, response["timings_sec"]["total"], persist)
        return response

    except HTTPException as exc:
        jobs.finish(job_id, status="failed",
                    call_id=call_id, started_monotonic=t0_mono,
                    error=f"{exc.status_code}: {exc.detail}")
        raise
    except Exception as exc:
        jobs.finish(job_id, status="failed",
                    call_id=call_id, started_monotonic=t0_mono,
                    error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        # Serialized mode: hand the GPU back after every call (success OR
        # failure) so the box is free for other work when idle. The audio
        # models lazy-reload (~15-30 s) on the next call — the accepted
        # cold-start trade. Runs after persist+coaching are done.
        if not config.PRELOAD_MODELS:
            try:
                from pipeline import ner as _ner
                with _GPU_LOCK:
                    registry.unload_all()
                    _ner.release()
                    import gc as _gc
                    import torch as _torch
                    _gc.collect()
                    if _torch.cuda.is_available():
                        _torch.cuda.empty_cache()
                        _torch.cuda.ipc_collect()
                # Also release the Ollama models that hold VRAM: the KPI/coaching
                # LLM (gemma, ~10 GB) and the embedder (~3.7 GB when on GPU, i.e.
                # EMBEDDING_NUM_GPU != 0). keep_alive:0 unloads them immediately
                # instead of after OLLAMA_KEEP_ALIVE, so the GPU goes fully free
                # between calls. Best-effort. (A CPU embedder holds no VRAM, but
                # unloading it is harmless and frees its RAM too.)
                try:
                    import httpx as _httpx
                    _httpx.post(f"{config.OLLAMA_HOST}/api/generate",
                                json={"model": config.LLM_MODEL_ID,
                                      "keep_alive": 0},
                                timeout=10)
                    _httpx.post(f"{config.OLLAMA_HOST}/api/embeddings",
                                json={"model": config.embedding_model_id(),
                                      "prompt": "x", "keep_alive": 0},
                                timeout=10)
                except Exception:
                    pass
                logger.info("serialized_unload_done call_id=%s", call_id)
            except Exception:
                logger.exception("serialized_unload_failed (non-fatal)")


def _sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def _relax_memory() -> None:
    """Best-effort release of CUDA cache + Python heap.

    Called between heavy GPU stages so the WSL2 host doesn't accumulate
    pinned buffers and SIGKILL the process from outside the cgroup.
    """
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()
    except Exception:  # noqa: BLE001 — never block a request over GC
        pass


# ── Admin: right-to-be-forgotten ─────────────────────────────────────────────
# Auth is a shared key compared constant-time. When ADMIN_API_KEY is empty
# the gate is open (single-user loopback deployment); set it in .env once
# the surface is exposed beyond localhost.
def _verify_admin(x_admin_key: Optional[str] = Header(None,
                                                       alias="x-admin-key")
                   ) -> None:
    expected = config.ADMIN_API_KEY
    if not expected:
        return
    presented = (x_admin_key or "").strip()
    if not presented or not hmac.compare_digest(presented, expected):
        raise HTTPException(status_code=401, detail="invalid_admin_key")


@app.delete("/admin/call/{call_id}")
def delete_call(
    call_id: str = Path(..., min_length=1),
    purge_files: bool = Query(True,
                              description="Also delete on-disk audio + PDF."),
    _auth: None = Depends(_verify_admin),
) -> Dict[str, Any]:
    """Right-to-be-forgotten.

    Removes:
      * calls / utterances / call_events rows (CASCADE from calls).
      * pipeline_runs rows referring to this call_id (FK is not declared,
        so we DELETE explicitly).
      * Optionally: on-disk audio + the rendered report directory at
        ``/data/reports/<call_id>/``.

    Idempotent: deleting a non-existent call returns 200 with deleted=False.
    """
    deleted_rows = 0
    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM calls WHERE call_id = %s", (call_id,))
            deleted_rows += cur.rowcount
            cur.execute("DELETE FROM pipeline_runs WHERE call_id = %s", (call_id,))
        conn.commit()

    files_removed: List[str] = []
    if purge_files:
        # Rendered report directory.
        report_dir = os.path.join(config.DATA_DIR, "reports", call_id)
        if os.path.isdir(report_dir):
            try:
                shutil.rmtree(report_dir)
                files_removed.append(report_dir)
            except OSError as exc:
                logger.warning("rtbf_rmtree_failed dir=%s err=%r",
                               report_dir, exc)
        # Audio cache directory (if any).
        audio_dir = os.path.join(config.DATA_DIR, "audio", call_id)
        if os.path.isdir(audio_dir):
            try:
                shutil.rmtree(audio_dir)
                files_removed.append(audio_dir)
            except OSError as exc:
                logger.warning("rtbf_rmtree_failed dir=%s err=%r",
                               audio_dir, exc)

    logger.info("rtbf_deleted call_id=%s db_rows=%d files=%d",
                call_id, deleted_rows, len(files_removed))
    return {
        "ok":            True,
        "call_id":       call_id,
        "deleted":       deleted_rows > 0,
        "db_rows":       deleted_rows,
        "files_removed": files_removed,
    }


# (Admin client-management endpoints removed 2026-06-10 — per-client voice
#  identity is not used; the `clients` table no longer exists.)


# (The /objections endpoint (C3 objection library) was removed 2026-06-10 —
#  it had zero consumers. The objection events themselves are still persisted
#  into call_events; query them via shared/queries/ or rebuild the endpoint
#  from git history if a consumer ever appears.)


# ── Prometheus metrics (F2) ──────────────────────────────────────────────────
@app.get("/metrics")
def metrics_endpoint() -> Response:
    """Hand-rolled Prometheus exposition (no client dep). Scraped by the
    prometheus container. Exposes GPU VRAM + corpus + job-status gauges."""
    lines: List[str] = []
    if config.USE_GPU:
        try:
            p = torch.cuda.get_device_properties(0)
            lines += [
                "# HELP ser_api_gpu_vram_bytes GPU VRAM in bytes by kind.",
                "# TYPE ser_api_gpu_vram_bytes gauge",
                f'ser_api_gpu_vram_bytes{{kind="allocated"}} {torch.cuda.memory_allocated(0)}',
                f'ser_api_gpu_vram_bytes{{kind="reserved"}} {torch.cuda.memory_reserved(0)}',
                f'ser_api_gpu_vram_bytes{{kind="total"}} {p.total_memory}',
            ]
        except Exception:  # noqa: BLE001
            pass
    try:
        with db.connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM calls")
                calls_n = cur.fetchone()[0]
                cur.execute("SELECT status, count(*) FROM pipeline_runs GROUP BY status")
                job_rows = cur.fetchall()
        lines += [
            "# HELP ser_api_calls_total Calls persisted.",
            "# TYPE ser_api_calls_total gauge",
            f"ser_api_calls_total {calls_n}",
            "# HELP ser_api_jobs_total Pipeline runs by status.",
            "# TYPE ser_api_jobs_total gauge",
        ]
        for status, n in job_rows:
            lines.append(f'ser_api_jobs_total{{status="{status}"}} {n}')
    except Exception:  # noqa: BLE001
        pass
    return Response("\n".join(lines) + "\n",
                    media_type="text/plain; version=0.0.4")


# ── Live resource snapshot (control-panel summary box) ───────────────────────
@app.get("/resources")
def resources_endpoint() -> Dict[str, Any]:
    """Live instantaneous GPU/CPU/RAM as rel/global/other %, for the control
    panel's summary box. Cheap + lock-free (no _GPU_LOCK) so it stays
    responsive while a /scorecard holds the GPU."""
    return {"ok": True, **resource_sampler.snapshot()}


# ── Chat-grade transcription (voice input for chat front-ends) ───────────────
@app.post("/transcribe")
async def transcribe_endpoint(
    file:     UploadFile    = File(...),
) -> Dict[str, Any]:
    t0 = time.time()
    with tempfile.TemporaryDirectory() as td:
        raw_path = audio.save_upload(file, td)
        wav_path = audio.to_wav_16k_mono(raw_path, td)
        with _GPU_LOCK:
            asr_model = registry.asr()
            words     = transcription.transcribe(asr_model, wav_path)
        duration = audio.wav_duration(wav_path)

    text = " ".join(w["text"] for w in words).strip()
    logger.info("transcribe_done file=%s words=%d total_sec=%.2f",
                file.filename, len(words), time.time() - t0)
    return {"ok": True, "text": text, "language": config.ASR_LANGUAGE,
            "duration_sec": duration, "elapsed_sec": round(time.time() - t0, 3)}


_MIME_EXT = {
    "audio/ogg":  ".ogg",  "audio/oga":  ".ogg",  "audio/opus": ".ogg",
    "audio/mpeg": ".mp3",  "audio/mp3":  ".mp3",
    "audio/wav":  ".wav",  "audio/x-wav":".wav",
    "audio/webm": ".webm", "audio/m4a":  ".m4a",  "audio/mp4":  ".m4a",
}


@app.post("/transcribe_b64")
async def transcribe_b64_endpoint(payload: Dict[str, Any] = Body(...)
                                  ) -> Dict[str, Any]:
    """Same as ``/transcribe`` but the audio comes inline as base64."""
    audio_b64 = payload.get("audio_b64") or ""
    if not audio_b64:
        raise HTTPException(status_code=400, detail="audio_b64 is required")
    if "," in audio_b64 and audio_b64.lstrip().startswith("data:"):
        audio_b64 = audio_b64.split(",", 1)[1]
    try:
        raw = base64.b64decode(audio_b64, validate=False)
    except (binascii.Error, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"invalid base64: {e}") from e
    if not raw:
        raise HTTPException(status_code=400, detail="audio_b64 decoded empty")

    mime_type = (payload.get("mime_type") or "audio/ogg").lower()
    ext       = _MIME_EXT.get(mime_type, ".ogg")

    t0 = time.time()
    with tempfile.TemporaryDirectory() as td:
        raw_path = os.path.join(td, f"voice{ext}")
        with open(raw_path, "wb") as f:
            f.write(raw)
        wav_path = audio.to_wav_16k_mono(raw_path, td)
        with _GPU_LOCK:
            asr_model = registry.asr()
            words     = transcription.transcribe(asr_model, wav_path)
        duration = audio.wav_duration(wav_path)

    text     = " ".join(w["text"] for w in words).strip()
    logger.info("transcribe_b64_done bytes=%d words=%d total_sec=%.2f",
                len(raw), len(words), time.time() - t0)
    return {"ok": True, "text": text, "language": config.ASR_LANGUAGE,
            "duration_sec": duration, "elapsed_sec": round(time.time() - t0, 3)}


# NOTE: /dashboard and /dashboard.png have moved to rag-api (CPU read/serve service).

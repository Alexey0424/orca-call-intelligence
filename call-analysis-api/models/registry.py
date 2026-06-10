"""Hot-swap registry for the four audio models.

Each role has exactly one slot. ``get(role)`` returns the currently
loaded model for that slot, lazily swapping in a fresh one when the
configured model ID changes (via the control-UI overrides file). The
previous model is evicted (Python reference dropped + CUDA cache freed)
before the new one is loaded, so VRAM stays bounded.

Roles
-----
* ``asr``         — faster-whisper ``WhisperModel``
* ``diarization`` — pyannote ``Pipeline``
* ``sentiment``   — HF text-classification pipeline + ``id2label`` dict
* ``ser``         — FunASR ``AutoModel`` (emotion2vec+) + label list

Concurrency
-----------
A swap holds ``_SWAP_LOCK`` and is wrapped in the caller's existing
GPU mutex (``app._GPU_LOCK``). Concurrent ``get()`` calls during a
swap block until the swap completes. Concurrent ``get()`` calls when
no swap is needed just read the slot — no contention.

Why a registry and not module-globals?
--------------------------------------
The previous design had each model in a module global written once at
lifespan startup. Swapping required a container restart because nothing
re-read the override file or rebuilt the model. The registry moves
loader resolution to call-time, which is exactly what runtime overrides
need to act on.
"""
from __future__ import annotations

import gc
import logging
import threading
import time
from typing import Any, Dict, Optional, Tuple

import config
from models import asr           as asr_loader
from models import diarization   as diar_loader
from models import emotion       as ser_loader
from models import sentiment     as sentiment_loader

logger = logging.getLogger(__name__)


_VALID_ROLES = ("asr", "diarization", "sentiment", "ser")

# slot["model"] is whatever the loader returns (heterogeneous shape — see
# ``_load_for_role``). slot["model_id"] is the model id that was loaded.
_slots: Dict[str, Dict[str, Any]] = {
    "asr":         {"model_id": None, "model": None},
    "diarization": {"model_id": None, "model": None},
    "sentiment":   {"model_id": None, "model": None},
    "ser":         {"model_id": None, "model": None},
}

_SWAP_LOCK = threading.Lock()


def _validate_role(role: str) -> None:
    if role not in _VALID_ROLES:
        raise KeyError(f"unknown role {role!r} — expected one of {_VALID_ROLES}")


# ── Loaders ──────────────────────────────────────────────────────────────────
def _load_for_role(role: str, model_id: str) -> Any:
    """Dispatch to the per-role loader. Returns the loader's native shape."""
    if role == "asr":
        return asr_loader.load(model_id)
    if role == "diarization":
        return diar_loader.load(model_id)
    if role == "sentiment":
        # Returns (pipeline, id2label).
        return sentiment_loader.load(model_id)
    if role == "ser":
        # Returns (model, labels).
        return ser_loader.load(model_id)
    raise KeyError(role)  # unreachable — _validate_role gates entry


def _release(model: Any) -> None:
    """Best-effort eviction. Pyannote / faster-whisper / HF pipelines all
    hold CUDA buffers via tensors owned by the model object. Dropping the
    Python reference is enough for the GC + ``empty_cache`` pair to claim
    the VRAM back. We swallow any exception because eviction MUST NOT
    block the swap — at worst we leak a tensor until the next GC pass.
    """
    if model is None:
        return
    try:
        # For HF pipeline (sentiment) — pin to CPU first so the GPU side
        # is fully detached before the GC reclaims it.
        pipe = model[0] if isinstance(model, tuple) and len(model) >= 1 else model
        inner = getattr(pipe, "model", None)
        if inner is not None and hasattr(inner, "to"):
            try:
                inner.to("cpu")
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        pass
    try:
        del model
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:  # noqa: BLE001
        pass


# ── Public API ──────────────────────────────────────────────────────────────
def get(role: str) -> Any:
    """Return the model for ``role``, swapping if the configured ID changed.

    Returns the same shape the per-role loader returns:
      * asr / diarization     → single object
      * sentiment             → (pipeline, id2label) tuple
      * ser                   → (model, labels)    tuple
    """
    _validate_role(role)
    desired_id = config.current_audio_model_id(role)
    slot = _slots[role]
    if slot["model"] is not None and slot["model_id"] == desired_id:
        return slot["model"]

    # Swap path. Double-checked inside the lock so concurrent callers
    # don't each kick off their own load.
    with _SWAP_LOCK:
        if slot["model"] is not None and slot["model_id"] == desired_id:
            return slot["model"]
        _swap(role, desired_id)
        return slot["model"]


def _swap(role: str, new_id: str) -> None:
    """Evict the current model for ``role`` and load ``new_id``.

    Caller MUST hold ``_SWAP_LOCK``. On loader error the slot is restored
    to whatever was loaded before so the next ``get()`` call doesn't
    return ``None``.
    """
    slot = _slots[role]
    prev_id = slot["model_id"]
    prev    = slot["model"]
    logger.info("model_swap_start role=%s old=%s new=%s",
                role, prev_id, new_id)

    # Step 1: evict the previous tenant before loading the new one.
    # Loading-then-evicting would double-peak VRAM, which 16 GB can't
    # afford alongside the Ollama 9.5 GB tenant.
    slot["model"]    = None
    slot["model_id"] = None
    _release(prev)

    # Step 2: load. If this raises, leave the slot empty and surface the
    # error to the caller — the next /scorecard will reload at that
    # point. We do NOT silently fall back to the previous model because
    # then the operator would see the override but get the old behaviour.
    t0 = time.monotonic()
    try:
        model = _load_for_role(role, new_id)
    except Exception:
        logger.exception("model_swap_failed role=%s id=%s", role, new_id)
        raise
    slot["model"]    = model
    slot["model_id"] = new_id
    logger.info("model_swap_done role=%s id=%s load_sec=%.2f",
                role, new_id, time.monotonic() - t0)


def reload(role: str) -> Dict[str, Any]:
    """Force a reload even if the configured ID matches the loaded one.

    Useful when the on-disk weights have been replaced (e.g. cache wipe).
    Returns the swap summary for the admin endpoint to surface.
    """
    _validate_role(role)
    desired_id = config.current_audio_model_id(role)
    t0 = time.monotonic()
    with _SWAP_LOCK:
        _swap(role, desired_id)
    return {"role": role, "model_id": desired_id,
            "load_sec": round(time.monotonic() - t0, 2)}


def reload_if_changed(role: str) -> Dict[str, Any]:
    """Reload only if config now resolves to a different ID. No-op if same.

    The caller (control-UI ``/models`` POST handler) sends one of these
    per audio role after writing the overrides file; we re-resolve here
    so a no-op rename of the same model doesn't pointlessly evict +
    reload the same weights.
    """
    _validate_role(role)
    desired_id = config.current_audio_model_id(role)
    slot = _slots[role]
    if slot["model_id"] == desired_id and slot["model"] is not None:
        return {"role": role, "model_id": desired_id,
                "load_sec": 0.0, "swapped": False}
    t0 = time.monotonic()
    with _SWAP_LOCK:
        if slot["model_id"] == desired_id and slot["model"] is not None:
            return {"role": role, "model_id": desired_id,
                    "load_sec": 0.0, "swapped": False}
        _swap(role, desired_id)
    return {"role": role, "model_id": desired_id,
            "load_sec": round(time.monotonic() - t0, 2),
            "swapped": True}


def snapshot() -> Dict[str, Optional[str]]:
    """Currently-loaded model id per role. Used by /health + /admin/cache-status."""
    return {role: slot["model_id"] for role, slot in _slots.items()}


def preload_all() -> Dict[str, Any]:
    """Lifespan helper — load every role's currently-configured model.

    Order matters slightly: ASR first (lightest), then diarization,
    sentiment, SER (biggest). On a fresh image these will trigger
    Hugging-Face / FunASR downloads; subsequent restarts read from the
    on-disk caches and complete in seconds.
    """
    results: Dict[str, Any] = {}
    for role in _VALID_ROLES:
        try:
            t0 = time.monotonic()
            _ = get(role)
            results[role] = {"ok": True,
                              "model_id": _slots[role]["model_id"],
                              "load_sec": round(time.monotonic() - t0, 2)}
        except Exception as exc:  # noqa: BLE001
            logger.exception("preload_failed role=%s", role)
            results[role] = {"ok": False, "error": str(exc)[:300]}

    # Defensive: after bulk-loading every model, hand any reserved-but-unused
    # caching-allocator blocks from the preload peak back to CUDA. Mirrors the
    # per-swap eviction cleanup. (Note: most of call-analysis-api's idle VRAM
    # is non-torch — CTranslate2 / FunASR / ONNX — so empty_cache() won't shrink
    # the idle footprint much; it just bounds any transient torch peak.)
    try:
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            logger.info("preload_all_cache_released")
    except Exception:  # noqa: BLE001
        pass
    return results


def evict(role: str) -> None:
    """Evict a single role's slot (drop the ref + free its CUDA buffers).

    Used by serialized mode to release a model the moment its pipeline stage
    finishes, so the four audio models don't accumulate in VRAM across one
    call (which stacked ~5-6 GB on top of whatever else was resident). No-op
    if the slot is already empty. The next ``get(role)`` lazily reloads it —
    free within a single call, since each role is used exactly once.
    """
    _validate_role(role)
    with _SWAP_LOCK:
        slot = _slots[role]
        prev = slot["model"]
        if prev is None:
            return
        slot["model"]    = None
        slot["model_id"] = None
        _release(prev)
    logger.info("registry_evict role=%s done", role)


def unload_all() -> None:
    """Evict every loaded slot and hand the VRAM back to the GPU.

    Used by serialized mode (``PRELOAD_MODELS=0``) after each /scorecard so
    the audio models don't sit resident while the pipeline is idle. The next
    ``get(role)`` lazily reloads the slot. Safe to call when slots are already
    empty. ``_release`` does the gc + ``empty_cache`` + ``ipc_collect`` dance.
    """
    with _SWAP_LOCK:
        for role in _VALID_ROLES:
            slot = _slots[role]
            prev = slot["model"]
            if prev is None:
                continue
            slot["model"]    = None
            slot["model_id"] = None
            _release(prev)
    logger.info("registry_unload_all done")


# Convenience accessors keep call-sites tidy. The tuple-returning loaders
# get unpacked here so callers don't repeat the shape knowledge.
def asr():
    return get("asr")


def diarization():
    return get("diarization")


def sentiment() -> Tuple[Any, Dict[int, str]]:
    return get("sentiment")


def ser() -> Tuple[Any, list]:
    return get("ser")

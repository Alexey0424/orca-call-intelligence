"""Read / write the shared runtime-overrides JSON file.

The control UI writes new model picks here; ser-api reads this file at
import (for audio models) and at request time (for Ollama models). The
file is on a bind-mount shared between the two containers.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from typing import Any, Dict, Optional

import config

logger = logging.getLogger(__name__)


# Audio-model keys — swapping any of these requires a ser-api restart.
RESTART_REQUIRED_KEYS = {
    "WHISPER_MODEL_ID",
    "DIARIZATION_MODEL_ID",
    "SENTIMENT_MODEL_ID",
    "SER_MODEL_ID",
}

# Ollama keys — hot-swappable, picked up on the next request.
HOT_SWAP_KEYS = {
    "LLM_MODEL_ID",
    "EMBEDDING_MODEL_ID",
    # Opening-of-call LLM speaker→role relabel (call-analysis-api reads both
    # per request via config.opening_relabel_enabled()/opening_relabel_model()).
    "OPENING_RELABEL_ENABLED",
    "OPENING_RELABEL_MODEL",
}

ALL_KEYS = RESTART_REQUIRED_KEYS | HOT_SWAP_KEYS


def requires_restart(key: str) -> bool:
    return key in RESTART_REQUIRED_KEYS


def read() -> Dict[str, Any]:
    """Return the current overrides dict (empty if file missing)."""
    p = config.RUNTIME_OVERRIDES_PATH
    if not p.exists():
        return {}
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("overrides_read_failed path=%s err=%r", p, exc)
        return {}


def write(updates: Dict[str, Optional[str]]) -> Dict[str, Any]:
    """Merge ``updates`` into the overrides file, atomically.

    Passing ``None`` for a value deletes that override (so ser-api falls
    back to its env / default). Unknown keys are dropped to avoid typos
    silently doing nothing.
    """
    current = read()
    changed: Dict[str, Optional[str]] = {}
    for k, v in updates.items():
        if k not in ALL_KEYS:
            logger.warning("overrides_unknown_key key=%s — dropped", k)
            continue
        if v is None or v == "":
            if k in current:
                del current[k]
                changed[k] = None
        else:
            if current.get(k) != v:
                current[k] = v
                changed[k] = v

    if not changed:
        return {"ok": True, "changed": {}, "current": current}

    # Atomic write — temp file in the same dir, then os.replace.
    p = config.RUNTIME_OVERRIDES_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".overrides.", suffix=".json",
                                     dir=str(p.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(current, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp_path, p)
    except Exception:
        try:
            os.unlink(tmp_path)
        except Exception:  # noqa: BLE001
            pass
        raise

    logger.info("overrides_written changed=%s", changed)
    return {"ok": True, "changed": changed, "current": current}

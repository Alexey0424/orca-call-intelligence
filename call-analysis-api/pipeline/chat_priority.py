"""Chat-priority gate — call-processing yields to an active live-chat consumer.

A chat front-end (any service serving a live human) marks
``runtime_settings.chat_active_until`` (epoch seconds) while it handles a
message. ``/scorecard`` awaits this **before** acquiring the GPU lock, so a
live human's chat gets the GPU + Ollama first. Disabled by default in this
stack (``CHAT_PRIORITY_ENABLED=0``) — no bundled service sets the key.

The wait is **async** on purpose: the event loop stays free while we wait, so
the chat's own call into this API (``/transcribe_b64`` for an inbound voice
note, which acquires the same ``_GPU_LOCK``) isn't blocked — that would
otherwise deadlock (the scorecard waiting on the chat that's waiting on the
scorecard's lock). The wait is capped at ``CHAT_PRIORITY_MAX_WAIT_SEC`` so a
crashed chat service can't stall calls beyond the TTL it set. A missing
``runtime_settings`` table (no chat service installed) means no gate.

**v1 limitation:** this defers the *start* of a call while a chat is active;
it does NOT pause a call already in flight. True mid-call preemption needs the
``_GPU_LOCK`` released at stage boundaries — deferred (see the
``gpu-contention-coordination`` memory). Fine for low/moderate call volume.
"""
from __future__ import annotations

import asyncio
import logging
import time

import config
import db

logger = logging.getLogger(__name__)

_KEY = "chat_active_until"


def _chat_active_until() -> float:
    """Epoch-seconds until which a chat holds priority, or 0.0 if none / error."""
    try:
        with db.connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT value FROM runtime_settings WHERE key = %s",
                            (_KEY,))
                row = cur.fetchone()
        if not row or not row[0]:
            return 0.0
        return float(row[0])
    except Exception:  # noqa: BLE001 — table missing / DB blip → no gate
        return 0.0


async def await_chat_idle() -> float:
    """Block (async) while a live chat holds priority. Returns secs waited."""
    if not config.CHAT_PRIORITY_ENABLED:
        return 0.0
    waited = 0.0
    while True:
        until = await asyncio.to_thread(_chat_active_until)
        now = time.time()
        if until <= now:
            break
        if waited >= config.CHAT_PRIORITY_MAX_WAIT_SEC:
            logger.warning("chat_priority_wait_capped waited=%.0fs", waited)
            break
        if waited == 0.0:
            logger.info("chat_priority_yielding active_for=%.0fs", until - now)
        await asyncio.sleep(config.CHAT_PRIORITY_POLL_SEC)
        waited += config.CHAT_PRIORITY_POLL_SEC
    if waited > 0:
        logger.info("chat_priority_resumed waited=%.1fs", waited)
    return waited

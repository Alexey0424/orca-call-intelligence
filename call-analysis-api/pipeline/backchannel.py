"""Backchannel detection — flag "mhm" / "yeah" / "okay" interjections.

Real turns (a buyer answering a question, a closer pitching) carry
substantive content; backchannels are the small active-listening cues
that show *engagement* without taking a turn. They look identical in
the schema to short statements unless we flag them, which pollutes
``wpm`` and ``question_count`` and biases talk-ratio.

The rule is deliberately conservative — over-flagging is worse than
under-flagging because backchannel-flagged utterances are excluded from
downstream aggregates:

  1. Duration ≤ ``MAX_BACKCHANNEL_SEC`` (default 1.5 s).
  2. Text content normalises to one of the ack tokens (or a 2-token
     ack pair like "yeah yeah", "okay okay", "mm hm").
  3. No sentence-ending punctuation (a 1-second "Right!" with emphasis
     could still be a substantive agreement and stays as a real turn).

Phase 2: if real calls show false positives, tighten by adding context
(e.g. "yeah" right after a closer question is a backchannel; "yeah"
followed by 8 seconds of silence then a long turn may be agreement).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List


# Conservative ack vocabulary. Single tokens that on their own are
# backchannels in conversational English.
_ACK_TOKENS = {
    "mhm", "mm", "mmhm", "mmhmm", "uhhuh", "uh-huh", "uh huh",
    "yeah", "yep", "yup", "yes",
    "okay", "ok", "alright", "all right",
    "right", "sure", "totally", "exactly", "absolutely",
    "got it", "gotcha", "i see", "i hear you",
    "cool", "nice",
}

# Tokens that often pair (e.g. "yeah yeah", "okay okay", "mm hm")
_ACK_DOUBLES = {(a, b) for a in _ACK_TOKENS for b in _ACK_TOKENS}

_PUNCT_STRIP_RE = re.compile(r"[\.,!?;:\-\"'…]+$|^[\"'\-]+")

# Anything ending in ! or strong emphasis isn't a backchannel by our rule.
_EMPHATIC_TAIL = re.compile(r"[!?]+$")


MAX_BACKCHANNEL_SEC = 1.5
MAX_BACKCHANNEL_TOKENS = 3   # safety: never flag anything >3 tokens


def _normalise(text: str) -> str:
    return _PUNCT_STRIP_RE.sub("", (text or "").strip().lower()).strip()


def is_backchannel(utterance: Dict[str, Any]) -> bool:
    """Apply the rule. Returns True only when all three conditions hold."""
    text = utterance.get("text") or ""
    if _EMPHATIC_TAIL.search(text.strip()):
        return False

    norm = _normalise(text)
    if not norm:
        return False

    tokens = norm.split()
    if len(tokens) > MAX_BACKCHANNEL_TOKENS:
        return False

    duration = float(utterance.get("end", 0.0)) - float(utterance.get("start", 0.0))
    if duration > MAX_BACKCHANNEL_SEC:
        return False

    # Single-token match.
    if len(tokens) == 1 and tokens[0] in _ACK_TOKENS:
        return True
    # Two-token match against the catalogue (handles "yeah yeah", "okay okay").
    if len(tokens) == 2 and tuple(tokens) in _ACK_DOUBLES:
        return True
    # Three-token: only if the whole string is a known compound ack.
    if len(tokens) == 3 and " ".join(tokens) in {"yeah yeah yeah", "okay okay okay"}:
        return True
    # Multi-word atoms ("got it", "i see") matched as a single key.
    if norm in _ACK_TOKENS:
        return True
    return False


def annotate(utterances: List[Dict[str, Any]]) -> int:
    """Tag each utterance in-place with ``is_backchannel`` bool.

    Returns the count flagged so the caller can log it.
    """
    flagged = 0
    for u in utterances:
        bc = is_backchannel(u)
        u["is_backchannel"] = bc
        if bc:
            flagged += 1
    return flagged

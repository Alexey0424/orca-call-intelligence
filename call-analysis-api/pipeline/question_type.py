"""Rule-based question-type classification for English utterances.

Categories:
  open     — "what brings you in today?", "tell me about..."
  closed   — "do you live here?", "is the price right?" (yes/no answerable)
  probing  — "and then what happened?", "can you give me an example?"
  none     — utterance contains no question

Phase 2 will look at false-positive rate on real calls and (if needed)
either tighten the patterns or replace this with a small classifier.
For now: every utterance that ends with "?" is classified, plus a few
implicit-question phrasings ("tell me about", "walk me through").
The aggregate counts land on ``calls`` as per-role columns the
dashboards and coaching prompt can both read.

Why bother now: ``question_count`` alone is misleading — a closer who
asked "did you" / "are you" / "is it" 20 times has very different
coaching needs than one who asked "what" / "how" / "why" 20 times.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List


# Closed-question starters (yes/no answerable in English).
_CLOSED_STARTERS = {
    "is", "are", "was", "were",
    "do", "does", "did",
    "have", "has", "had",
    "can", "could", "will", "would", "should", "shall",
    "may", "might", "must",
    "am",
}

# Open-question starters (require an explanatory answer).
_OPEN_STARTERS = {
    "what", "how", "why", "when", "where", "who", "which",
    "whose", "whom",
}

# Probing patterns — follow-ups designed to deepen an answer just given.
_PROBING_PATTERNS = (
    re.compile(r"^and (then |after |before |so )?\w+", re.IGNORECASE),
    re.compile(r"\b(tell me more|can you give me an example|"
               r"walk me through|what do you mean by|"
               r"can you elaborate|how so|in what way)\b", re.IGNORECASE),
    re.compile(r"^(so |and )(why|how|what)\b", re.IGNORECASE),
)

# Implicit-question phrasings — declarative form but functions as a question.
# Only flagged when not already ending in "?".
_IMPLICIT_OPEN = re.compile(
    r"^\s*(tell me|walk me through|describe|explain|"
    r"i'?d (like|love) to (hear|know)|"
    r"help me understand)\b",
    re.IGNORECASE,
)

# Tag questions ("you bought the house, right?") are closed — they ask
# confirmation of a stated assertion.
_TAG_TAIL = re.compile(
    r",\s*(right|correct|okay|ok|isn'?t it|aren'?t (you|they|we)|"
    r"don'?t (you|we|they)|wouldn'?t (you|we|they))\s*\??\s*$",
    re.IGNORECASE,
)

_PUNCT_TAIL = re.compile(r"[\.,!?;:\-\"'…]+$")


def _classify_sentence(sent: str) -> str:
    """Classify one sentence; returns 'open'|'closed'|'probing'|'none'."""
    s = sent.strip()
    if not s:
        return "none"

    # Probing wins over open/closed because probing patterns are specific.
    for pat in _PROBING_PATTERNS:
        if pat.search(s):
            return "probing"

    has_qmark = s.endswith("?")
    s_clean = _PUNCT_TAIL.sub("", s).strip()
    if not s_clean:
        return "none"

    first_word = s_clean.split()[0].lower().strip("'\"")

    if has_qmark:
        if _TAG_TAIL.search(s):
            return "closed"
        if first_word in _OPEN_STARTERS:
            return "open"
        if first_word in _CLOSED_STARTERS:
            return "closed"
        # Question mark but no recognised starter (e.g. "the price?") — call
        # it closed; it's a verification-style fragment.
        return "closed"

    # No question mark — only flag specific implicit-question phrasings.
    if _IMPLICIT_OPEN.match(s):
        return "open"
    return "none"


def classify_utterance(text: str) -> Dict[str, int]:
    """Count question-type occurrences in one utterance.

    An utterance may contain multiple sentences; we classify each and
    return a counts dict {open, closed, probing}.
    """
    out = {"open": 0, "closed": 0, "probing": 0}
    if not text:
        return out
    # Split on sentence terminators while keeping reasonably long fragments.
    sentences = re.split(r"(?<=[\.\?\!])\s+", text.strip())
    for sent in sentences:
        kind = _classify_sentence(sent)
        if kind in out:
            out[kind] += 1
    return out


def annotate(utterances: List[Dict[str, Any]]) -> None:
    """Tag each utterance in-place with a ``question_types`` count dict."""
    for u in utterances:
        u["question_types"] = classify_utterance(u.get("text") or "")


def aggregate(utterances: List[Dict[str, Any]],
              speaker: str) -> Dict[str, int]:
    """Sum per-utterance counts for one speaker. Skips backchannels."""
    totals = {"open": 0, "closed": 0, "probing": 0}
    for u in utterances:
        if u.get("speaker") != speaker:
            continue
        if u.get("is_backchannel"):
            continue
        qt = u.get("question_types") or {}
        for k in totals:
            totals[k] += int(qt.get(k, 0))
    return totals

"""Deterministic counts (words, questions, fillers, rapport, politeness, buying signals).

Lives outside ``kpis.py`` because it is regex-based and reproducible — no
LLM call, no nondeterminism. Both Spanish and English markers are
recognised since the project handles bilingual sales calls.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List


# Cheap whole-word match. Markers are pre-lowercased.
_FILLER_MARKERS = (
    "um", "uh", "erm",
    "you know", "i mean", "like", "kind of", "sort of",
    "basically", "literally", "actually",
    # Spanish
    "este", "eh", "o sea", "pues", "bueno",
)

_RAPPORT_MARKERS = (
    "thank you", "thanks", "i appreciate", "appreciate it",
    "i understand", "great point", "good question",
    "absolutely", "exactly", "of course",
    # Spanish
    "gracias", "te entiendo", "entiendo",
    "buen punto", "buena pregunta", "claro", "por supuesto",
)

_POLITENESS_MARKERS = (
    "please", "thank you", "thanks",
    "would you mind", "could you", "may i",
    # Spanish
    "por favor", "gracias", "podrías", "podría", "me permite",
)

_BUYING_SIGNALS = (
    "how much", "how soon", "when can",
    "ready to", "let's do it", "sounds good",
    "i'm interested", "interested in", "sign me up",
    "what's the next step",
    # Spanish
    "cuánto cuesta", "cuanto cuesta",
    "cuándo puedo", "cuando puedo",
    "estoy listo", "estoy lista",
    "me interesa", "me apunto",
)

_QUESTION_RE = re.compile(r"\?")


def _count_phrases(haystack: str, needles: Iterable[str]) -> int:
    return sum(haystack.count(n) for n in needles)


def _speaker_text(segments: List[Dict[str, Any]], speaker: str) -> str:
    return " ".join(s.get("text", "") for s in segments
                    if s.get("speaker") == speaker)


def for_speaker(segments: List[Dict[str, Any]], speaker: str) -> Dict[str, int]:
    text = _speaker_text(segments, speaker).lower()
    word_count = len(text.split())
    return {
        "word_count":         word_count,
        "question_count":     len(_QUESTION_RE.findall(text)),
        "filler_count":       _count_phrases(text, _FILLER_MARKERS),
        "rapport_markers":    _count_phrases(text, _RAPPORT_MARKERS),
        "politeness_markers": _count_phrases(text, _POLITENESS_MARKERS),
        "buying_signals":     _count_phrases(text, _BUYING_SIGNALS),
    }

"""LLM coaching rubric — six-dimension score, weighted to 0–100.

Produces a per-call coaching score persisted into ``calls.coaching_score``.

Design:
  * Weights are fixed in code; the LLM only outputs raw 0–100 per dimension.
    We compute the weighted average ourselves so the model doesn't have to
    do arithmetic (a known failure mode of small LLMs).
  * The model sees pre-extracted KPIs alongside the transcript so its scores
    are grounded in evidence we trust.
  * Failures are non-fatal: ``score_call`` returns ``None``, persistence
    leaves the column NULL, the dashboard renders "awaiting".
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Sequence

import httpx

import config

logger = logging.getLogger(__name__)


# ── Rubric ───────────────────────────────────────────────────────────────────
_WEIGHTS: Dict[str, float] = {
    "opening":            0.10,
    "discovery":          0.25,
    "objection_handling": 0.20,
    "buying_signals":     0.15,
    "next_step":          0.20,
    "rapport":            0.10,
}

_SYSTEM_PROMPT = """You are a senior sales coach reviewing a sales-call transcript.

Score the CLOSER (the sales rep, not the client) on six dimensions, each on a 0–100 scale:

1. opening              — set context, introduced themselves, set an agenda
2. discovery            — quality of probing questions, active listening, summarising the client's needs
3. objection_handling   — addressed objections directly and recovered momentum (not just deflected)
4. buying_signals       — picked up on AND capitalised on client buying cues (turning them into commitments)
5. next_step            — secured a clear, concrete next step (date, action, channel) before closing
6. rapport              — warmth, professionalism, tone, pacing, mutual respect

You will also receive pre-computed counts (objection_count, buying_signal_count, next_step_secured, rapport_quality, etc.).
Use them as evidence — do not re-count them.

Calibration anchors (use these to keep scores grounded):
  - 90+: outstanding, would coach others on this
  - 70–89: solid, with one or two small misses
  - 50–69: serviceable, with clear room to improve
  - 30–49: weak on this dimension, several missed plays
  - 0–29: significant gap, urgent coaching needed

Output STRICT JSON only — no markdown fences, no prose before or after. Schema:

{
  "scores": {
    "opening":            <integer 0-100>,
    "discovery":          <integer 0-100>,
    "objection_handling": <integer 0-100>,
    "buying_signals":     <integer 0-100>,
    "next_step":          <integer 0-100>,
    "rapport":            <integer 0-100>
  },
  "notes": "<one short sentence: what worked best AND what to improve>"
}
"""


# ── Input shaping ────────────────────────────────────────────────────────────
def _format_kpis(kpis: Dict[str, Any]) -> str:
    out: List[str] = []
    for k in ("summary", "rapport_quality", "objection_count",
              "buying_signal_count", "next_step_secured",
              "next_step_text", "predicted_outcome",
              "talk_ratio_closer", "closer_wpm", "client_wpm",
              "filler_word_count_closer", "politeness_score_closer"):
        if k in kpis and kpis[k] not in (None, ""):
            out.append(f"  {k}: {kpis[k]}")
    for key, label in (("objection_events", "Objections raised by client"),
                       ("buying_signal_events", "Buying signals from client")):
        events = kpis.get(key) or []
        if events:
            out.append(f"\n  {label}:")
            for ev in events[:8]:
                t = ev.get("start_sec", 0)
                txt = (ev.get("text") or "").strip()
                if txt:
                    out.append(f"    [{t:.0f}s] {txt}")
    return "\n".join(out)


def _format_transcript(utterances: Sequence[Dict[str, Any]],
                       max_lines: int = 180) -> str:
    out: List[str] = []
    for u in utterances[:max_lines]:
        role = u.get("role") or u.get("speaker") or "?"
        text = (u.get("text") or "").strip()
        if not text:
            continue
        out.append(f"{role}: {text}")
    if len(utterances) > max_lines:
        out.append(f"… [{len(utterances) - max_lines} more utterances omitted]")
    return "\n".join(out)


def _build_user_message(scorecard: Dict[str, Any],
                        utterances: Sequence[Dict[str, Any]]) -> str:
    md = scorecard.get("metadata", {}) or {}
    sc = scorecard.get("scorecard", scorecard) or {}
    kpis = sc.get("kpis", {}) or {}

    parts = [
        "## CALL META",
        f"closer: {md.get('closer_name') or sc.get('identity', {}).get('closer_name', 'unknown')}",
        f"client: {md.get('client_name') or kpis.get('client_name', 'unknown')}",
        f"duration_sec: {md.get('duration_sec') or sc.get('duration_sec', '?')}",
        f"language: {sc.get('language', 'en')}",
        "",
        "## PRE-COMPUTED SIGNALS",
        _format_kpis(kpis),
        "",
        "## TRANSCRIPT (role-labelled)",
        _format_transcript(utterances),
    ]
    return "\n".join(parts)


# ── Core scorer ──────────────────────────────────────────────────────────────
def score_call(scorecard: Dict[str, Any],
               utterances: Optional[Sequence[Dict[str, Any]]] = None,
               ) -> Optional[Dict[str, Any]]:
    """Run the coaching rubric over a scorecard.

    Returns ``{overall_score, breakdown, notes}`` or ``None`` on any failure.
    """
    if utterances is None:
        sc = scorecard.get("scorecard", scorecard) or {}
        utterances = (scorecard.get("utterances")
                      or sc.get("utterances")
                      or [])

    if not utterances:
        logger.warning("coaching_score_no_utterances — skipping")
        return None

    user_msg = _build_user_message(scorecard, utterances)

    payload = {
        "model": config.llm_model_id(),
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        "stream": False,
        "format": "json",
        # Disable gemma4 thinking — we want the structured rubric JSON, not a
        # chain of thought (which adds ~200 s and risks the timeout). Same
        # rationale as kpis.py; the content->thinking fallback below is kept.
        "think":  False,
        "options": {
            "num_ctx":     config.LLM_NUM_CTX,
            "temperature": 0.1,
        },
    }
    try:
        r = httpx.post(f"{config.OLLAMA_HOST}/api/chat",
                       json=payload, timeout=config.LLM_TIMEOUT_SEC)
        r.raise_for_status()
        body = r.json()
    except httpx.HTTPError as exc:
        logger.warning("coaching_score_http_failed err=%r", exc)
        return None

    msg = body.get("message", {}) or {}
    raw = (msg.get("content") or "").strip() or (msg.get("thinking") or "").strip()
    if not raw:
        logger.warning("coaching_score_empty_response body_keys=%s",
                       list(body.keys()))
        return None

    parsed = _parse_json(raw)
    if not parsed:
        logger.warning("coaching_score_unparseable head=%r", raw[:200])
        return None

    scores = parsed.get("scores") or {}
    if not isinstance(scores, dict):
        logger.warning("coaching_score_no_scores parsed=%s", parsed)
        return None

    breakdown: Dict[str, float] = {}
    for dim in _WEIGHTS:
        try:
            v = float(scores.get(dim, 50))
        except (TypeError, ValueError):
            v = 50.0
        breakdown[dim] = max(0.0, min(100.0, round(v, 1)))

    weight_sum = sum(_WEIGHTS.values()) or 1.0
    overall = sum(breakdown[d] * w for d, w in _WEIGHTS.items()) / weight_sum

    return {
        "overall_score": round(overall, 1),
        "breakdown":     breakdown,
        "notes":         str(parsed.get("notes", "")).strip()[:400],
    }


# ── Tolerant JSON parsing ────────────────────────────────────────────────────
_JSON_OBJ_RE = re.compile(r"\{[\s\S]*\}")
_FENCE_RE    = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _parse_json(raw: str) -> Optional[Dict[str, Any]]:
    s = _FENCE_RE.sub("", raw.strip()).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    m = _JSON_OBJ_RE.search(s)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None

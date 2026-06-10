"""Build the executive view dict that downstream modules consume.

The exec view is a flat, render-ready summary of the inbound analysis.
Charts, the HTML template, and the JSON ``executive_summary`` returned to
the API caller are all derived from this same dict.

**Schema-aware (v6 / May 2026 rebuild).** Reads the new ser-api JSON shape:
``utterances`` (not legacy ``segments``), per-utterance ``valence`` /
``arousal`` / ``dominance`` (not ``arousal_proxy``), and the renamed top-
level KPI keys ``next_step_secured`` / ``predicted_outcome`` /
``interruptions``. Old keys are still consulted as a fallback so legacy
input.json files in ``data/reports/`` still re-render cleanly.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import config
import insights
from utils import fnum, format_duration, nullable_round


# ── Risk score ───────────────────────────────────────────────────────────────
def _risk(*, next_step: bool, objections: int, interruptions: int,
          silence_ratio: float, buying_signals: int,
          client_sentiment: Dict[str, Any]) -> Tuple[int, str]:
    score = 0
    if not next_step:                                                    score += 2
    if objections >= config.OBJECTIONS_HEAVY:                            score += 2
    if interruptions >= config.INTERRUPTIONS_HEAVY:                      score += 1
    if silence_ratio >= config.SILENCE_HEAVY_RATIO:                      score += 1
    if str(client_sentiment.get("sentiment_trajectory", "")).lower() == "declining":
        score += 1
    if buying_signals == 0:                                              score += 1

    if score >= config.RISK_HIGH_AT:
        level = "high"
    elif score >= config.RISK_MEDIUM_AT:
        level = "medium"
    else:
        level = "low"
    return score, level


# ── Field-name compat helper ─────────────────────────────────────────────────
def _first(d: Dict[str, Any], *keys, default=None):
    """Return the first key that exists and is non-None in ``d``."""
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


# ── Helper: find the client speaker label from roles map ─────────────────────
def _client_speaker(roles: Dict[str, str]) -> Optional[str]:
    for spk, role in roles.items():
        if role == "client":
            return spk
    return None


# ── Per-utterance timeline (used by charts and transcript) ───────────────────
def _timeline_utterances(utterances: List[Dict[str, Any]],
                         closer_speaker: str,
                         client_speaker: Optional[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for u in utterances:
        spk = u.get("speaker", "UNKNOWN")
        if spk == closer_speaker:
            role = "closer"
        elif client_speaker and spk == client_speaker:
            role = "client"
        else:
            role = "other"

        sentiment = (u.get("sentiment_label") or u.get("sentiment") or "neutral").lower()
        emotion   = (_first(u, "emotion_label", "audio_emotion_label", default="") or "").lower()
        # New schema stores `arousal` as a real value in [-1, +1].
        # Old schema stored `arousal_proxy` in [0, 1]. Accept either.
        arousal   = fnum(_first(u, "arousal", "arousal_proxy"), 0.0)

        out.append({
            "start":           fnum(u.get("start"), 0.0),
            "end":             fnum(u.get("end"),   0.0),
            "speaker":         spk,
            "role":            role,
            "text":            (u.get("text") or "").strip(),
            "sentiment":       sentiment,
            "sentiment_score": fnum(u.get("sentiment_score"), 0.0),
            "sentiment_probs": u.get("sentiment_probs") or {},
            "emotion":         emotion,
            "emotion_probs":   _first(u, "emotion_probs", "audio_emotion_probs",
                                        default={}) or {},
            "valence":         fnum(u.get("valence"),   0.0),
            "arousal":         arousal,
            "dominance":       fnum(u.get("dominance"), 0.0),
        })
    out.sort(key=lambda s: s["start"])
    return out


def _role_rows(*, talk_pct: Tuple[float, float], wpm: Tuple[Any, Any],
               text_metrics: Dict[str, Any],
               sentiment: Dict[str, Any], emotion: Dict[str, Any]
               ) -> List[List[Any]]:
    closer_t = text_metrics.get("closer", {}) or {}
    client_t = text_metrics.get("client", {}) or {}
    closer_s = sentiment.get("closer", {})  or {}
    client_s = sentiment.get("client", {})  or {}
    closer_e = emotion.get("closer", {})    or {}
    client_e = emotion.get("client", {})    or {}

    return [
        ["Closer", f"{talk_pct[0]:.1f}%", nullable_round(wpm[0], 1),
         int(fnum(closer_t.get("word_count"),     0)),
         int(fnum(closer_t.get("question_count"), 0)),
         int(fnum(closer_t.get("filler_count"),   0)),
         closer_s.get("sentiment_trajectory", "n/a"),
         closer_e.get("dominant_emotion",     "n/a")],
        ["Client", f"{talk_pct[1]:.1f}%", nullable_round(wpm[1], 1),
         int(fnum(client_t.get("word_count"),     0)),
         int(fnum(client_t.get("question_count"), 0)),
         int(fnum(client_t.get("filler_count"),   0)),
         client_s.get("sentiment_trajectory", "n/a"),
         client_e.get("dominant_emotion",     "n/a")],
    ]


def _chart_data(*, talk_pct: Tuple[float, float],
                silence_pct: float, overlap_pct: float,
                text_metrics: Dict[str, Any],
                sentiment: Dict[str, Any],
                wpm: Tuple[Any, Any],
                utterances_timeline: List[Dict[str, Any]],
                duration_sec: float,
                events: Dict[str, Any]) -> Dict[str, Any]:
    closer_t = text_metrics.get("closer", {}) or {}
    client_t = text_metrics.get("client", {}) or {}
    closer_s = sentiment.get("closer", {})  or {}
    client_s = sentiment.get("client", {})  or {}
    return {
        "talk_share": {"closer_pct": talk_pct[0], "client_pct": talk_pct[1]},
        "conversation_flow": {
            "closer_talk_pct": talk_pct[0], "client_talk_pct": talk_pct[1],
            "silence_pct": silence_pct, "overlap_pct": overlap_pct,
        },
        "role_metrics": {
            "roles":     ["Closer", "Client"],
            "words":     [int(fnum(closer_t.get("word_count"), 0)),
                          int(fnum(client_t.get("word_count"), 0))],
            "questions": [int(fnum(closer_t.get("question_count"), 0)),
                          int(fnum(client_t.get("question_count"), 0))],
            "fillers":   [int(fnum(closer_t.get("filler_count"), 0)),
                          int(fnum(client_t.get("filler_count"), 0))],
            "wpm":       [fnum(wpm[0], 0.0), fnum(wpm[1], 0.0)],
        },
        "sentiment": {
            "roles": ["Closer", "Client"],
            "positive_ratio": [fnum(closer_s.get("positive_ratio"), 0.0) * 100.0,
                               fnum(client_s.get("positive_ratio"), 0.0) * 100.0],
            "negative_ratio": [fnum(closer_s.get("negative_ratio"), 0.0) * 100.0,
                               fnum(client_s.get("negative_ratio"), 0.0) * 100.0],
        },
        # NOTE: existing chart code consumes ``timeline_segments`` — keep that
        # key name even though the upstream concept is ``utterances`` now.
        "timeline_segments": utterances_timeline,
        "duration_sec":      duration_sec,
        "events":            events,
    }


# ── Public entrypoint ────────────────────────────────────────────────────────
def build(analysis: Dict[str, Any]) -> Dict[str, Any]:
    metadata     = analysis.get("metadata", {})  or {}
    scorecard    = analysis.get("scorecard", {}) or {}
    # New ser-api returns ``utterances``; legacy input.json files have ``segments``.
    utterances   = (analysis.get("utterances")
                    or analysis.get("segments")
                    or []) or []
    timeline     = scorecard.get("timeline", {})     or {}
    text_metrics = scorecard.get("text_metrics", {}) or {}
    kpis         = scorecard.get("kpis", {})         or {}
    events       = scorecard.get("events", {})       or {}
    sentiment    = scorecard.get("sentiment", {})    or {}
    emotion      = scorecard.get("emotion", {})      or {}
    roles        = scorecard.get("roles", {})        or {}

    duration_sec   = fnum(scorecard.get("duration_sec"), 0.0)
    closer_speaker = metadata.get("closer_speaker", "SPEAKER_01")
    client_speaker = _client_speaker(roles)

    # ── Talk ratio: new column is ``closer_talk_ratio``; the LLM JSON keeps
    # the legacy name ``talk_ratio_closer`` since it's part of the prompt
    # contract. Accept either.
    closer_pct = round(
        fnum(_first(kpis, "closer_talk_ratio", "talk_ratio_closer"), 0.0)
        * 100.0, 1)
    client_pct = round(max(0.0, 100.0 - closer_pct), 1)

    silence_ratio = fnum(timeline.get("silence_ratio"), 0.0)
    overlap_ratio = fnum(timeline.get("overlap_ratio"), 0.0)
    silence_pct   = round(silence_ratio * 100.0, 1)
    overlap_pct   = round(overlap_ratio * 100.0, 1)

    # New timeline uses ``interruptions`` (count). Legacy used ``interruptions_count``.
    interruptions  = int(fnum(_first(timeline, "interruptions", "interruptions_count"), 0))
    objections     = int(fnum(kpis.get("objection_count"),     0))
    buying_signals = int(fnum(kpis.get("buying_signal_count"), 0))
    # New name first, legacy fallback.
    next_step      = bool(_first(kpis,   "next_step_secured", "next_step_detected",
                                 default=_first(events, "next_step_secured",
                                                "next_step_detected",
                                                default=False)))
    next_step_text = _first(kpis, "next_step_text", "next_step_description",
                            default=_first(events, "next_step_text",
                                           "next_step_description"))
    predicted_outcome = _first(kpis, "predicted_outcome", "call_outcome_prediction")

    closer_sent = sentiment.get("closer", {}) or {}
    client_sent = sentiment.get("client", {}) or {}
    closer_em   = emotion.get("closer", {})   or {}
    client_em   = emotion.get("client", {})   or {}

    score, level = _risk(
        next_step=next_step, objections=objections, interruptions=interruptions,
        silence_ratio=silence_ratio, buying_signals=buying_signals,
        client_sentiment=client_sent,
    )

    head = insights.headline(
        next_step=next_step, objections=objections,
        buying_signals=buying_signals, risk_level=level,
        client_sentiment=client_sent,
    )
    verdict_line = insights.verdict(
        next_step=next_step, objections=objections, buying_signals=buying_signals,
        interruptions=interruptions,
        rapport=kpis.get("rapport_quality"),
        outcome=predicted_outcome,
        client_sentiment=client_sent,
    )
    coach_score, coach_grade, coach_breakdown = insights.coaching_score(
        next_step=next_step, objections=objections, interruptions=interruptions,
        silence_ratio=silence_ratio, buying_signals=buying_signals,
        rapport=kpis.get("rapport_quality"),
        client_sentiment=client_sent,
    )
    recs = insights.recommendations(
        next_step=next_step, objections=objections, interruptions=interruptions,
        silence_ratio=silence_ratio, buying_signals=buying_signals,
        client_sentiment=client_sent, closer_text=text_metrics.get("closer", {}),
    )
    highlight_strip = insights.highlights(
        events, utterances,
        next_step=next_step, next_step_text=next_step_text,
    )
    pull_quote      = insights.pull_quote(events)

    timeline_utts = _timeline_utterances(utterances, closer_speaker, client_speaker)
    role_rows     = _role_rows(
        talk_pct=(closer_pct, client_pct),
        wpm=(kpis.get("closer_wpm"), kpis.get("client_wpm")),
        text_metrics=text_metrics, sentiment=sentiment, emotion=emotion,
    )

    summary_json = {
        "headline":            head,
        "verdict":             verdict_line,
        "coaching_score":      coach_score,
        "coaching_grade":      coach_grade,
        "risk_level":          level,
        "risk_score":          score,
        "call_duration_sec":   round(duration_sec, 2),
        "call_duration_human": format_duration(duration_sec),
        "closer_talk_pct":     closer_pct,
        "client_talk_pct":     client_pct,
        "silence_pct":         silence_pct,
        "overlap_pct":         overlap_pct,
        "interruptions":       interruptions,
        "objections":          objections,
        "buying_signals":      buying_signals,
        "next_step_secured":   next_step,
        "next_step_text":      next_step_text,
        "rapport_quality":     kpis.get("rapport_quality"),
        "predicted_outcome":   predicted_outcome,
        "closer_wpm":          nullable_round(kpis.get("closer_wpm"), 2),
        "client_wpm":          nullable_round(kpis.get("client_wpm"), 2),
        "closer_sentiment_trajectory": closer_sent.get("sentiment_trajectory"),
        "client_sentiment_trajectory": client_sent.get("sentiment_trajectory"),
        "closer_dominant_emotion":     closer_em.get("dominant_emotion"),
        "client_dominant_emotion":     client_em.get("dominant_emotion"),
        # New dimensional axes — surfaced for future chart work.
        "closer_valence_mean":         closer_em.get("valence_mean"),
        "closer_arousal_mean":         closer_em.get("arousal_mean"),
        "closer_dominance_mean":       closer_em.get("dominance_mean"),
        "client_valence_mean":         client_em.get("valence_mean"),
        "client_arousal_mean":         client_em.get("arousal_mean"),
        "client_dominance_mean":       client_em.get("dominance_mean"),
    }

    # Phase strip — three boxes (opening / middle / closing) per role
    phase_strip = {
        "closer": closer_em.get("phase_emotions") or {},
        "client": client_em.get("phase_emotions") or {},
    }

    chart_data = _chart_data(
        talk_pct=(closer_pct, client_pct),
        silence_pct=silence_pct, overlap_pct=overlap_pct,
        text_metrics=text_metrics, sentiment=sentiment,
        wpm=(kpis.get("closer_wpm"), kpis.get("client_wpm")),
        utterances_timeline=timeline_utts,
        duration_sec=duration_sec, events=events,
    )

    # Hero tiles — four big call-out values for the top of the report.
    hero_tiles = [
        {"label": "Duration",   "value": format_duration(duration_sec),
         "tone": "neutral"},
        {"label": "Talk split", "value": f"{closer_pct:.0f} / {client_pct:.0f}",
         "tone": "good" if 40 <= closer_pct <= 60 else "warn" if closer_pct < 75 else "bad",
         "hint": "closer / client"},
        {"label": "Sentiment",
         "value": _trajectory_arrow(client_sent.get("sentiment_trajectory")),
         "tone": _trajectory_tone(client_sent.get("sentiment_trajectory")),
         "hint": "client trajectory"},
        {"label": "Outcome",
         "value": (predicted_outcome or "unknown").replace("_", " "),
         "tone": _outcome_tone(predicted_outcome),
         "hint": "model prediction"},
    ]

    return {
        "call_id":         metadata.get("call_id",   "unknown-call"),
        "file_name":       metadata.get("file_name", "unknown-file"),
        "report_title":    f"Executive Call Report — {metadata.get('call_id','unknown-call')}",
        "generated_at":    time.strftime("%Y-%m-%d %H:%M:%S"),
        "company_name":    config.COMPANY_NAME,
        "closer_speaker":  closer_speaker,
        "client_speaker":  client_speaker,
        "summary_json":    summary_json,
        "headline":        head,
        "verdict":         verdict_line,
        "risk_level":      level,
        "coaching_score":  coach_score,
        "coaching_grade":  coach_grade,
        "coaching_breakdown": coach_breakdown,
        "hero_tiles":      hero_tiles,
        "highlights":      highlight_strip,
        "pull_quote":      pull_quote,
        "phase_strip":     phase_strip,
        "recommendations": recs,
        "role_rows":       role_rows,
        "chart_data":      chart_data,
        "metadata":        metadata,
        "utterances":      utterances,
        "next_step_text":  next_step_text,
    }


# ── Tone helpers (used by hero tiles) ────────────────────────────────────────
def _trajectory_arrow(trajectory: Optional[str]) -> str:
    t = (trajectory or "stable").lower()
    if t == "improving": return "↗ improving"
    if t == "declining": return "↘ declining"
    return "→ stable"


def _trajectory_tone(trajectory: Optional[str]) -> str:
    t = (trajectory or "stable").lower()
    if t == "improving": return "good"
    if t == "declining": return "bad"
    return "neutral"


def _outcome_tone(outcome: Optional[str]) -> str:
    o = (outcome or "").lower()
    if o == "likely_close":      return "good"
    if o == "lost":              return "bad"
    if o == "follow_up_needed":  return "warn"
    return "neutral"

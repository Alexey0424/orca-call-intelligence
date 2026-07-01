"""Build the render-api dashboard view-model from calls.

Parameterised SQL against ``calls`` produces a clean JSON view-model that
``render-api`` turns into a custom-designed static HTML page; Gotenberg
then screenshots it to PNG for the chat-image path.

The contract this module emits matches the field shapes documented in
``render-api/dashboard.py`` and ``render-api/templates/dashboard.html``.
Keep them in sync.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import psycopg2.extras

from ragcore import config, db

logger = logging.getLogger(__name__)


# ── Date range presets ───────────────────────────────────────────────────────
# Each preset returns (since, until) as Python datetimes. ``until`` is exclusive.
# ``None`` means open-ended on that side.
def _range_bounds(date_range: str) -> Tuple[Optional[datetime], Optional[datetime]]:
    now = datetime.now(timezone.utc)
    today = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
    month_start = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
    if date_range == "today":
        return today, None
    if date_range == "yesterday":
        return today - timedelta(days=1), today
    if date_range == "7d":
        return now - timedelta(days=7), None
    if date_range == "30d":
        return now - timedelta(days=30), None
    if date_range == "mtd":
        return month_start, None
    if date_range == "last_month":
        # Calendar-aligned previous month: [first day of last month, first day of this month).
        if now.month == 1:
            prev = datetime(now.year - 1, 12, 1, tzinfo=timezone.utc)
        else:
            prev = datetime(now.year, now.month - 1, 1, tzinfo=timezone.utc)
        return prev, month_start
    if date_range == "all":
        return None, None
    # default
    return now - timedelta(days=7), None


_RANGE_LABELS = {
    "today":      "Today",
    "yesterday":  "Yesterday",
    "7d":         "Last 7 Days",
    "30d":        "Last 30 Days",
    "mtd":        "Month-to-Date",
    "last_month": "Last Month",
    "all":        "All Time",
}


def _where_received_at(since: Optional[datetime], until: Optional[datetime]
                   ) -> Tuple[str, List[Any]]:
    """Build a ``received_at`` WHERE clause + bind values."""
    parts, vals = [], []
    if since is not None:
        parts.append("received_at >= %s")
        vals.append(since)
    if until is not None:
        parts.append("received_at < %s")
        vals.append(until)
    return (" AND ".join(parts) if parts else "TRUE"), vals


# ── Helpers ──────────────────────────────────────────────────────────────────
def _initial(name: Optional[str]) -> str:
    if not name:
        return "?"
    parts = name.replace(".", " ").split()
    return parts[0][0].upper() if parts and parts[0] else "?"


def _human_name(name: Optional[str]) -> Tuple[str, str, bool]:
    """Return (display_name, handle, unattributed) for a closer slug.

    ``Elias.Clower`` → ("Elias Clower", "elias.clower", False).
    ``None``/empty   → ("Unattributed", "voice-id pending", True).
    """
    if not name:
        return ("Unattributed", "voice-id pending", True)
    pretty = name.replace(".", " ").replace("_", " ").strip()
    handle = name.lower()
    return (pretty, handle, False)


def _fmt_duration(seconds: float) -> Tuple[str, int]:
    total = int(round(seconds or 0))
    return (f"{total // 60:02d}:{total % 60:02d}", total)


_OUTCOME_LABELS = {
    "likely_close":    "Likely close",
    "follow_up_needed": "Follow-up needed",
    "lost":            "Lost",
    "unknown":         "Unknown",
}
_OUTCOME_ORDER  = ["likely_close", "follow_up_needed", "lost"]
_OUTCOME_COLORS = {
    "likely_close":     "var(--good)",
    "follow_up_needed": "var(--accent)",
    "lost":             "var(--bad)",
}

_SENTIMENT_LABELS = {
    "improving": "Improving",
    "stable":    "Stable",
    "declining": "Declining",
}
_SENTIMENT_ORDER  = ["improving", "stable", "declining"]
_SENTIMENT_COLORS = {
    "improving": "var(--sentiment-improving)",
    "stable":    "var(--sentiment-stable)",
    "declining": "var(--sentiment-declining)",
}


# ── Core builder ─────────────────────────────────────────────────────────────
def build_view_model(*, scope: str = "team",
                     date_range: str = "7d",
                     closer_name: Optional[str] = None,
                     ) -> Dict[str, Any]:
    """Run the dashboard queries and return a render-api-ready dict."""
    scope = scope or "team"
    date_range = date_range or "7d"
    since, until = _range_bounds(date_range)
    range_label = _RANGE_LABELS.get(date_range, "Last 7 Days")

    # Period filter (without scope restriction)
    period_where, period_vals = _where_received_at(since, until)

    # Scope filter (separate so prior-period queries can use the same scope)
    # Case-insensitive: closer_name is LLM-extracted from speech since
    # 2026-06-10, so capitalisation may vary between calls and chat queries.
    if scope == "closer" and closer_name:
        scope_clause = "lower(closer_name) = lower(%s)"
        scope_vals: List[Any] = [closer_name]
    else:
        scope_clause = "TRUE"
        scope_vals = []

    conn = db.connect()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            # ── KPI 1: Total calls (all-time, same scope) ───────────────────
            cur.execute(
                f"SELECT COUNT(*) AS n FROM calls WHERE {scope_clause};",
                scope_vals,
            )
            total_calls = int(cur.fetchone()["n"] or 0)

            # ── KPI 2: Calls in period + prior-period delta ────────────────
            cur.execute(
                f"SELECT COUNT(*) AS n FROM calls "
                f"WHERE {period_where} AND {scope_clause};",
                period_vals + scope_vals,
            )
            calls_in_period = int(cur.fetchone()["n"] or 0)

            # Prior comparable window
            if since is not None and until is None:
                window = datetime.now(timezone.utc) - since
                prior_since = since - window
                prior_until = since
            elif since is not None and until is not None:
                window = until - since
                prior_since = since - window
                prior_until = since
            else:
                prior_since = prior_until = None
            prior_where, prior_vals = _where_received_at(prior_since, prior_until)
            cur.execute(
                f"SELECT COUNT(*) AS n FROM calls "
                f"WHERE {prior_where} AND {scope_clause};",
                prior_vals + scope_vals,
            )
            calls_prior_period = int(cur.fetchone()["n"] or 0)

            # ── KPI 3: Avg coaching score (NULL until rubric runs) ─────────
            cur.execute(
                f"SELECT AVG(coaching_score) AS avg_score, "
                f"       COUNT(coaching_score) AS scored_n, "
                f"       COUNT(*) AS total_n "
                f"FROM calls WHERE {period_where} AND {scope_clause};",
                period_vals + scope_vals,
            )
            row = cur.fetchone()
            avg_coaching_score = (float(row["avg_score"])
                                  if row["avg_score"] is not None else None)
            coaching_scored_n  = int(row["scored_n"] or 0)
            coaching_total_n   = int(row["total_n"] or 0)

            # Per-day coaching averages for the trailing 7 days — drives
            # the KPI 3 sparkline. Days without scored calls are absent.
            cur.execute(
                """
                SELECT DATE_TRUNC('day', received_at AT TIME ZONE 'UTC') AS day,
                       AVG(coaching_score) AS avg_score
                FROM calls
                WHERE received_at >= NOW() - INTERVAL '7 days'
                  AND coaching_score IS NOT NULL
                  AND %s
                GROUP BY day
                ORDER BY day;
                """ % scope_clause,
                scope_vals,
            )
            coaching_day_scores: Dict[str, float] = {
                r["day"].strftime("%Y-%m-%d"): float(r["avg_score"])
                for r in cur.fetchall()
            }

            # ── KPI 4: Next-step rate (in period) ──────────────────────────
            cur.execute(
                f"SELECT "
                f"  SUM(CASE WHEN next_step_secured THEN 1 ELSE 0 END) AS yes_n, "
                f"  COUNT(*) AS total_n "
                f"FROM calls WHERE {period_where} AND {scope_clause};",
                period_vals + scope_vals,
            )
            row = cur.fetchone()
            next_step_yes   = int(row["yes_n"] or 0)
            next_step_total = int(row["total_n"] or 0)

            # ── Calls per day (last 7 buckets) ─────────────────────────────
            # Always render a 7-day strip ending today, regardless of date_range.
            cur.execute(
                """
                SELECT DATE_TRUNC('day', received_at AT TIME ZONE 'UTC') AS day,
                       COUNT(*) AS n
                FROM calls
                WHERE received_at >= NOW() - INTERVAL '7 days' AND %s
                GROUP BY day
                ORDER BY day;
                """ % scope_clause,
                scope_vals,
            )
            day_counts: Dict[str, int] = {
                r["day"].strftime("%Y-%m-%d"): int(r["n"]) for r in cur.fetchall()
            }

            # ── Outcome distribution ───────────────────────────────────────
            cur.execute(
                f"SELECT COALESCE(predicted_outcome, 'unknown') AS k, "
                f"       COUNT(*) AS n "
                f"FROM calls WHERE {period_where} AND {scope_clause} "
                f"GROUP BY k;",
                period_vals + scope_vals,
            )
            outcome_counts: Dict[str, int] = {r["k"]: int(r["n"]) for r in cur.fetchall()}

            # ── Sentiment trajectory ───────────────────────────────────────
            cur.execute(
                f"SELECT COALESCE(client_sentiment_trajectory, 'unknown') AS k, "
                f"       COUNT(*) AS n "
                f"FROM calls WHERE {period_where} AND {scope_clause} "
                f"GROUP BY k;",
                period_vals + scope_vals,
            )
            sentiment_counts: Dict[str, int] = {r["k"]: int(r["n"]) for r in cur.fetchall()}

            # ── Top closers (volume) ───────────────────────────────────────
            cur.execute(
                f"SELECT COALESCE(closer_name, '') AS closer, COUNT(*) AS n "
                f"FROM calls WHERE {period_where} "
                f"GROUP BY closer "
                f"ORDER BY n DESC, closer "
                f"LIMIT 6;",
                period_vals,
            )
            closer_rows = [(r["closer"], int(r["n"])) for r in cur.fetchall()]

            # ── Recent calls (8 most recent in period) ─────────────────────
            cur.execute(
                f"SELECT call_id, received_at, closer_name, client_name, "
                f"       duration_sec, objection_count, buying_signal_count, "
                f"       next_step_secured "
                f"FROM calls WHERE {period_where} AND {scope_clause} "
                f"ORDER BY received_at DESC LIMIT 8;",
                period_vals + scope_vals,
            )
            recent_rows = cur.fetchall()
    finally:
        conn.close()

    # ── Compose the view-model ──────────────────────────────────────────────
    now = datetime.now(timezone.utc)
    generated_stamp = "Generated · " + now.strftime("%d %b %Y · %H:%M UTC")

    # — Title / verdict (template-driven; safe with zero data) —
    title_line_1, title_line_2, verdict_html = _compose_narrative(
        calls_in_period=calls_in_period,
        next_step_yes=next_step_yes,
        next_step_total=next_step_total,
        outcome_counts=outcome_counts,
        sentiment_counts=sentiment_counts,
        range_label=range_label,
        scope=scope,
        scope_subject=(closer_name if scope == "closer" else "team"),
        avg_coaching_score=avg_coaching_score,
    )

    # — Scope chip —
    if scope == "closer" and closer_name:
        chip_text = f"{_human_name(closer_name)[0]} · {range_label}"
        scope_pip = _initial(closer_name)
        eyebrow = f"{_human_name(closer_name)[0]} · {range_label}"
    else:
        chip_text = f"Team · {range_label}"
        scope_pip = "T"
        eyebrow = f"Team Overview · {range_label}"

    # — KPIs —
    delta = calls_in_period - calls_prior_period
    if delta > 0:
        kpi2_tone, kpi2_delta = "pos", f"+{delta}"
    elif delta < 0:
        kpi2_tone, kpi2_delta = "neg", f"{delta}"
    else:
        kpi2_tone, kpi2_delta = "flat", "·"

    next_step_pct = (round(100.0 * next_step_yes / next_step_total)
                     if next_step_total else None)
    if next_step_pct is None:
        kpi4 = {
            "label": "Next-Step Rate", "null_text": "no calls",
            "delta": "—", "tone": "flat",
            "trend_text": "Awaiting first call",
            "spark": {"kind": "flat-dashed"},
        }
    else:
        kpi4 = {
            "label": "Next-Step Rate",
            "value": str(next_step_pct), "suffix": "%",
            "delta": f"{next_step_yes}/{next_step_total}",
            "tone": "pos" if next_step_pct >= 60 else ("flat" if next_step_pct >= 30 else "neg"),
            "trend_text": "calls with follow-up",
            "spark": {"kind": "fill-bar",
                      "tint": "good" if next_step_pct >= 50 else "bad"},
        }

    kpis = [
        {
            "label": "Total Calls",
            "value": str(total_calls),
            "delta": "·",
            "tone": "flat",
            "trend_text": "All-time",
            "spark": _sparkline_points_for_total(total_calls),
        },
        {
            "label": f"Calls · {range_label}",
            "value": str(calls_in_period),
            "delta": kpi2_delta,
            "tone": kpi2_tone,
            "trend_text": "vs prior window",
            "spark": _sparkline_points_for_period(day_counts),
        },
        _kpi_coaching(avg_coaching_score, coaching_scored_n,
                       coaching_total_n, coaching_day_scores),
        kpi4,
    ]

    # — Calls per day (always the trailing 7 days, ending today) —
    cpd_days = _build_cpd_days(day_counts)
    cpd_max = max([d["value"] for d in cpd_days] + [1])
    # Round max up to a nice integer (1, 2, 3, 5, 10, …)
    cpd_max_y = max(3, cpd_max + (1 if cpd_max == cpd_max else 0))

    active_days = sum(1 for d in cpd_days if d["active"])
    peak = max(cpd_days, key=lambda d: d["value"]) if cpd_days else None
    quiet = sum(1 for d in cpd_days if not d["active"])

    cpd_side_stats = [
        {"label": "Active Days",
         "value": str(active_days), "unit": "/ 7",
         "hint": (f"First call landed {_first_active_label(cpd_days)}."
                  if active_days else "No calls in this window.")},
        {"label": "Peak Day",
         "value": (peak["label"] if peak and peak["value"] > 0 else "—"),
         "unit": None,
         "hint": (f"{peak['value']} call{'s' if peak['value'] != 1 else ''} on this day."
                  if peak and peak["value"] > 0 else "No peak in window.")},
        {"label": "Quiet Stretch",
         "value": str(quiet), "unit": "days",
         "hint": "No calls logged on these days."},
    ]

    calls_per_day = {
        "note": "A 7-day strip of recorded calls. Dotted line is the trailing average.",
        "max_y": cpd_max_y, "show_avg_line": True,
        "days": cpd_days, "side_stats": cpd_side_stats,
    }

    # — Trio of donuts —
    trio = {
        "note": "How calls landed, where momentum is leaning, and where the model can't tell yet.",
        "donuts": [
            _outcome_donut(outcome_counts, calls_in_period),
            _risk_donut(calls_in_period),
            _sentiment_donut(sentiment_counts, calls_in_period),
        ],
    }

    # — Leaderboard —
    max_n = max([n for _, n in closer_rows] or [1])
    leaderboard_rows = []
    for i, (closer, n) in enumerate(closer_rows, start=1):
        display, handle, unattributed = _human_name(closer)
        leaderboard_rows.append({
            "rank": f"{i:02d}",
            "initial": _initial(closer) if not unattributed else "?",
            "name": display, "handle": handle,
            "bar_pct": round(100 * n / max_n),
            "count": str(n), "unit": "call" if n == 1 else "calls",
            "unattributed": unattributed,
        })
    leaderboard = {
        "note": "Volume ranking. \"Unattributed\" buckets calls the diarizer "
                "couldn't link to a known voice profile.",
        "rows": leaderboard_rows or [{
            "rank": "01", "initial": "·", "name": "No calls yet",
            "handle": "—", "bar_pct": 0, "count": "0", "unit": "calls",
            "unattributed": True,
        }],
    }

    # — Recent calls table —
    recent_calls = {
        "note": "Most recent first. Open the call report from chat for the full transcript.",
        "rows": [_recent_row(r) for r in recent_rows],
    }

    return {
        "doc_title": (config.BRAND_NAME + " · " +
                      ("Team Overview" if scope == "team"
                       else f"{_human_name(closer_name)[0]} · Performance")),
        "theme": "paper", "density": "regular",

        # Brand block — surfaced verbatim by render-api's template.
        "brand_mark":         config.BRAND_MARK,
        "brand_name":         config.BRAND_NAME,
        "brand_sub":          config.BRAND_SUB,
        "brand_footer_left":  config.BRAND_FOOTER_LEFT,
        "brand_footer_right": config.BRAND_FOOTER_RIGHT,

        "scope_pip": scope_pip,
        "scope_chip_text": chip_text,
        "generated_stamp": generated_stamp,
        "eyebrow_text": eyebrow,

        "title_line_1": title_line_1,
        "title_line_2": title_line_2,
        "verdict_html": verdict_html,

        "kpis": kpis,
        "calls_per_day": calls_per_day,
        "trio": trio,
        "leaderboard": leaderboard,
        "recent_calls": recent_calls,
    }


# ── Composition helpers ──────────────────────────────────────────────────────
def _compose_narrative(*, calls_in_period: int, next_step_yes: int,
                       next_step_total: int, outcome_counts: Dict[str, int],
                       sentiment_counts: Dict[str, int], range_label: str,
                       scope: str, scope_subject: Optional[str],
                       avg_coaching_score: Optional[float] = None,
                       ) -> Tuple[str, str, str]:
    """Hand-shaped title + verdict from raw counts. Template, not LLM."""
    if calls_in_period == 0:
        return ("Quiet window.",
                "Nothing to score yet.",
                f"No calls landed in the <b>{range_label.lower()}</b> window. "
                f"Once recordings arrive, the team metrics will populate here automatically.")

    plural = "" if calls_in_period == 1 else "s"
    title_1 = f"{_spell(calls_in_period)} call{plural}."

    # title line 2 — pick by next-step rate + sentiment
    rate = next_step_yes / next_step_total if next_step_total else 0
    decl = sentiment_counts.get("declining", 0)

    if rate >= 0.75:
        title_2 = "Pointed forward." if calls_in_period <= 3 else "Strong momentum."
    elif rate >= 0.5 and decl == 0:
        title_2 = "Mixed but holding."
    elif rate < 0.3:
        title_2 = "Soft week."
    else:
        title_2 = "Steady cadence."

    # verdict paragraph
    subj_html = ("the team" if scope == "team"
                 else f"<b>{scope_subject}</b>")
    parts = [
        f"{subj_html.capitalize() if scope == 'team' else subj_html} closed "
        f"<b>{calls_in_period} call{plural}</b> across "
        f"<b>{range_label.lower()}</b>"
    ]
    if next_step_total:
        if rate >= 0.99:
            parts.append("a next step was secured on <b>every one</b>")
        elif rate > 0:
            parts.append(
                f"a next step was secured on <b>{next_step_yes}/{next_step_total}</b>"
            )
        else:
            parts.append("no next steps were secured")
    if sentiment_counts:
        top = max(sentiment_counts.items(), key=lambda kv: kv[1])
        if top[0] in _SENTIMENT_LABELS:
            parts.append(
                f"client sentiment ran <b>{_SENTIMENT_LABELS[top[0]].lower()}</b> overall"
            )
    verdict = ". ".join(parts) + "."
    if avg_coaching_score is None:
        verdict += (" Coaching score sits in <b>cold-start</b> until the "
                    "scoring backfill lands.")
    else:
        verdict += (f" Average coaching score landed at "
                    f"<b>{round(avg_coaching_score)}</b>.")
    return title_1, title_2, verdict


def _spell(n: int) -> str:
    """Spell small integers; numbers >9 stay as digits — keeps the title elegant."""
    names = ["Zero", "One", "Two", "Three", "Four", "Five", "Six",
             "Seven", "Eight", "Nine"]
    return names[n] if 0 <= n < len(names) else str(n)


def _build_cpd_days(day_counts: Dict[str, int]) -> List[Dict[str, Any]]:
    """Return a 7-element list of {label, value, active} ending today."""
    today = datetime.now(timezone.utc).date()
    out: List[Dict[str, Any]] = []
    for offset in range(6, -1, -1):
        d = today - timedelta(days=offset)
        key = d.strftime("%Y-%m-%d")
        n = day_counts.get(key, 0)
        out.append({
            "label": d.strftime("%a %d").replace(" 0", " "),
            "value": int(n),
            "active": n > 0,
        })
    return out


def _first_active_label(days: List[Dict[str, Any]]) -> str:
    for d in days:
        if d["active"]:
            return d["label"]
    return "n/a"


def _kpi_coaching(avg: Optional[float], scored_n: int,
                  total_n: int,
                  day_scores: Optional[Dict[str, float]] = None,
                  ) -> Dict[str, Any]:
    """Build the Coaching Score KPI from real numbers (or fall back to awaiting).

    Sparkline behaviour:
      - 2+ distinct days with scores → real per-day line drawn from those values
      - otherwise (one day, or empty) → flat-dashed (honest: "not enough history")
    """
    if avg is None or scored_n == 0:
        return {
            "label":      "Avg Coaching Score",
            "null_text":  "awaiting",
            "delta":      "—",
            "tone":       "flat",
            "trend_text": ("Scorer not run yet" if total_n
                            else "No calls in window"),
            "spark":      {"kind": "flat-dashed"},
        }
    score = round(avg)
    if score >= 80:
        tone = "pos"
    elif score >= 60:
        tone = "flat"
    else:
        tone = "neg"
    coverage = (f"{scored_n}/{total_n} scored"
                if scored_n < total_n else "all calls scored")

    spark = _coaching_sparkline(day_scores or {})

    return {
        "label":      "Avg Coaching Score",
        "value":      str(score),
        "delta":      f"{scored_n}/{total_n}" if total_n else "—",
        "tone":       tone,
        "trend_text": coverage,
        "spark":      spark,
    }


def _coaching_sparkline(day_scores: Dict[str, float]) -> Dict[str, Any]:
    """Build sparkline points from real per-day averages, or fall back.

    Maps coaching scores in [0, 100] onto the sparkline's [4, 22] y-axis
    (lower y = higher score). Returns ``{kind: "flat-dashed"}`` when there
    aren't at least two distinct day values to plot.
    """
    if len(day_scores) < 2:
        return {"kind": "flat-dashed"}

    today = datetime.now(timezone.utc).date()
    values: List[Optional[float]] = []
    for offset in range(6, -1, -1):
        key = (today - timedelta(days=offset)).strftime("%Y-%m-%d")
        values.append(day_scores.get(key))

    # Forward-fill missing days with the prior known value so the line
    # stays continuous; leading missing days take the first known value.
    filled: List[float] = []
    last_known: Optional[float] = None
    for v in values:
        if v is not None:
            last_known = v
        if last_known is not None:
            filled.append(last_known)
    if len(filled) < 7:
        # Some leading days had no scores yet — pad to the front with the
        # first known value so the sparkline still has 7 anchor points.
        first = filled[0] if filled else 50.0
        filled = [first] * (7 - len(filled)) + filled

    points = [round(22 - (max(0.0, min(100.0, v)) / 100.0) * 18, 1)
              for v in filled]
    return {
        "kind":         "line",
        "stroke":       "var(--accent)",
        "endpoint_dot": True,
        "stroke_width": 1.4,
        "points":       points,
    }


def _sparkline_points_for_total(total: int) -> Dict[str, Any]:
    """Flat-ish ramp ending high — visually says 'accumulating'."""
    if total <= 0:
        return {"kind": "flat-dashed"}
    return {"kind": "line", "stroke": "var(--accent)", "endpoint_dot": True,
            "stroke_width": 1.4,
            "points": [22, 22, 22, 22, 12, 4, 4]}


def _sparkline_points_for_period(day_counts: Dict[str, int]) -> Dict[str, Any]:
    """Map the last-7-days counts onto a 0..26 vertical scale (inverted)."""
    today = datetime.now(timezone.utc).date()
    values: List[int] = []
    for offset in range(6, -1, -1):
        key = (today - timedelta(days=offset)).strftime("%Y-%m-%d")
        values.append(day_counts.get(key, 0))
    if not any(values):
        return {"kind": "flat-dashed"}
    max_v = max(values) or 1
    # invert: 22 = baseline, 4 = top
    points = [22 - round((v / max_v) * 18) for v in values]
    return {"kind": "line", "stroke": "var(--ink)", "endpoint_dot": False,
            "stroke_width": 1.2, "points": points}


def _outcome_donut(counts: Dict[str, int], total: int) -> Dict[str, Any]:
    if total <= 0:
        return _empty_donut("Outcome", "Distribution", "0", "calls")
    segs: List[Dict[str, Any]] = []
    legend: List[Dict[str, Any]] = []
    seen = 0
    for key in _OUTCOME_ORDER:
        n = counts.get(key, 0)
        pct = round(100 * n / total)
        if n > 0:
            segs.append({"color": _OUTCOME_COLORS[key], "pct": pct})
            seen += n
        legend.append({"color": _OUTCOME_COLORS[key],
                       "name": _OUTCOME_LABELS[key], "pct": pct,
                       "dim": (n == 0)})
    # leftover bucket — anything not in OUTCOME_ORDER
    other = total - seen
    if other > 0:
        pct = round(100 * other / total)
        segs.append({"color": "var(--ink-mute)", "pct": pct, "opacity": 0.5})
        legend.append({"color": "var(--ink-mute)", "name": "Other",
                       "pct": pct, "dim_swatch": True})
    return {"label": "Outcome", "title": "Distribution",
            "center_value": str(total), "center_sub": "calls",
            "segments": segs, "legend": legend}


def _risk_donut(total: int) -> Dict[str, Any]:
    """Schema has no risk classification yet — show 100% unknown."""
    if total <= 0:
        return _empty_donut("Risk", "Distribution", "0", "calls")
    return {
        "label": "Risk", "title": "Distribution",
        "center_value": str(total), "center_sub": "unscored",
        "segments": [{"color": "var(--ink-mute)", "pct": 100, "opacity": 0.5}],
        "legend": [
            {"color": "var(--ink-mute)", "name": "Unknown", "pct": 100,
             "dim_swatch": True},
            {"color": "var(--good)", "name": "Low",  "pct": 0, "dim": True},
            {"color": "var(--bad)",  "name": "High", "pct": 0, "dim": True},
        ],
    }


def _sentiment_donut(counts: Dict[str, int], total: int) -> Dict[str, Any]:
    if total <= 0:
        return _empty_donut("Client Sentiment", "Trajectory", "0", "clients")
    segs: List[Dict[str, Any]] = []
    legend: List[Dict[str, Any]] = []
    seen = 0
    for key in _SENTIMENT_ORDER:
        n = counts.get(key, 0)
        pct = round(100 * n / total)
        if n > 0:
            segs.append({"color": _SENTIMENT_COLORS[key], "pct": pct})
            seen += n
        legend.append({"color": _SENTIMENT_COLORS[key],
                       "name": _SENTIMENT_LABELS[key], "pct": pct,
                       "dim": (n == 0)})
    if (other := total - seen) > 0:
        pct = round(100 * other / total)
        segs.append({"color": "var(--ink-mute)", "pct": pct, "opacity": 0.5})
        legend.append({"color": "var(--ink-mute)", "name": "Unknown",
                       "pct": pct, "dim_swatch": True})
    return {"label": "Client Sentiment", "title": "Trajectory",
            "center_value": str(total), "center_sub": "clients",
            "segments": segs, "legend": legend}


def _empty_donut(label: str, title: str, center: str, sub: str) -> Dict[str, Any]:
    return {"label": label, "title": title,
            "center_value": center, "center_sub": sub,
            "segments": [], "legend": [
                {"color": "var(--rule)", "name": "No data", "pct": 0, "dim": True},
            ]}


def _recent_row(r: Sequence[Any]) -> Dict[str, Any]:
    call_id: str = r["call_id"]
    ts: datetime = r["received_at"]
    # Match the design's split: "20260512-" + "213411-535f9b8e"
    parts = call_id.split("-", 1)
    id_pre = (parts[0] + "-") if len(parts) == 2 else ""
    id_tail = parts[1] if len(parts) == 2 else call_id

    closer_raw = r["closer_name"]
    closer_unattributed = not bool(closer_raw)
    closer_display = (_human_name(closer_raw)[0] if closer_raw
                      else "unattributed")

    dur_str, dur_secs = _fmt_duration(r["duration_sec"])

    obj_n = int(r["objection_count"] or 0)
    buy_n = int(r["buying_signal_count"] or 0)

    return {
        "id_pre": id_pre, "id_tail": id_tail,
        "date": ts.strftime("%d %b %Y"),
        "time": ts.strftime("%H:%M"),
        "closer": closer_display,
        "closer_unattributed": closer_unattributed,
        "client": r["client_name"] or "—",
        "duration": dur_str, "duration_secs": dur_secs,
        "objections": obj_n,
        "obj_class": "obj" if obj_n > 0 else "zero",
        "buying_signals": buy_n,
        "buy_class": "buy" if buy_n > 0 else "zero",
        "next_step": "yes" if r["next_step_secured"] else "no",
    }

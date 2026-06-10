"""Pure-Python unit tests for render-api.

Run inside the container::

    docker compose exec render-api python -m pytest -q
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ── payload ──────────────────────────────────────────────────────────────────
def test_payload_unwrap_accepts_plain_dict():
    import payload
    obj = {"scorecard": {"x": 1}}
    assert payload.unwrap(obj) is obj


def test_payload_unwrap_unboxes_data_wrapper():
    import payload
    inner = {"scorecard": {"x": 1}}
    assert payload.unwrap({"data": inner}) is inner


def test_payload_unwrap_handles_single_item_list():
    import payload
    inner = {"scorecard": {"x": 1}}
    assert payload.unwrap([inner]) is inner


# ── utils ────────────────────────────────────────────────────────────────────
def test_format_duration():
    from utils import format_duration
    assert format_duration(45)    == "0m 45s"
    assert format_duration(125)   == "2m 5s"
    assert format_duration(3725)  == "1h 2m 5s"


def test_he_escapes_quotes_and_brackets():
    from utils import he
    assert he("a < b & c \"d\" 'e'") == "a &lt; b &amp; c &quot;d&quot; &#x27;e&#x27;"


def test_slugify_strips_punctuation():
    from utils import slugify
    assert slugify("Hello, World!") == "hello-world"
    assert slugify("   ")           == "report"


# ── insights ─────────────────────────────────────────────────────────────────
def test_recommendations_empty_call_falls_back():
    from insights import recommendations
    out = recommendations(
        next_step=True, objections=0, interruptions=0, silence_ratio=0.0,
        buying_signals=1, client_sentiment={}, closer_text={"question_count": 5},
    )
    assert len(out) >= 1
    # Recommendations are dicts {priority, headline, rationale}.
    assert isinstance(out[0], dict)
    assert "No major risk patterns" in out[0]["rationale"]


def test_recommendations_high_risk_returns_concrete_advice():
    from insights import recommendations
    out = recommendations(
        next_step=False, objections=3, interruptions=8, silence_ratio=0.3,
        buying_signals=0, client_sentiment={"sentiment_trajectory": "declining"},
        closer_text={"question_count": 0},
    )
    assert len(out) <= 5
    assert any("next step" in r["headline"].lower() or "next step" in r["rationale"].lower()
               for r in out)


def test_coaching_score_strong_call():
    from insights import coaching_score
    score, grade, breakdown = coaching_score(
        next_step=True, objections=0, interruptions=0, silence_ratio=0.0,
        buying_signals=2, rapport="high",
        client_sentiment={"sentiment_trajectory": "improving"},
    )
    assert 0 <= score <= 100
    assert score >= 100   # all positive bonuses, no penalties → clamps to 100
    assert grade == "A"


def test_coaching_score_disaster_call():
    from insights import coaching_score
    score, grade, _ = coaching_score(
        next_step=False, objections=4, interruptions=10, silence_ratio=0.4,
        buying_signals=0, rapport="low",
        client_sentiment={"sentiment_trajectory": "declining"},
    )
    assert score < 50
    assert grade == "F"


def test_verdict_picks_strong_close():
    from insights import verdict
    line = verdict(
        next_step=True, objections=0, buying_signals=2, interruptions=0,
        rapport="high", outcome="likely_close",
        client_sentiment={"sentiment_trajectory": "improving"},
    )
    assert "strong" in line.lower()


def test_highlights_picks_objection_and_buying_signal():
    from insights import highlights
    events = {
        "objection_events":     [{"start_sec": 12.0,
                                  "text": "Honestly, that price is way too expensive for us right now."}],
        "buying_signal_events": [{"start_sec": 30.5,
                                  "text": "Yeah, that works for me — sign me up and let us know the next step."}],
    }
    segments = []
    out = highlights(events, segments, next_step=True,
                     next_step_text="Schedule walk-through Tuesday at 4pm.")
    kinds = {h["kind"] for h in out}
    assert "objection" in kinds and "buying_signal" in kinds


# ── charts smoke (just verify they return non-empty SVG) ─────────────────────
def test_donut_returns_svg():
    from charts import donut
    out = donut.render(60.0, 40.0)
    assert out.startswith("<svg")
    assert out.endswith("</svg>")


def test_sentiment_timeline_handles_empty_segments():
    from charts import sentiment_timeline
    out = sentiment_timeline.render([], duration_sec=10.0, events={})
    assert out.startswith("<svg")

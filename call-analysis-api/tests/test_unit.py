"""Pure-Python unit tests — no model loading, run in milliseconds.

Run inside the container::

    docker compose exec call-analysis-api python -m pytest -q

These cover the deterministic glue logic (reconciliation, timeline,
text metrics). The end-to-end smoke test against ``/scorecard`` lives in
``smoke_endpoint.sh`` and requires a real audio file.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running pytest from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ── reconciliation ───────────────────────────────────────────────────────────
def test_reconciliation_groups_words_by_speaker_change():
    from pipeline import reconciliation as rec

    words = [
        {"start": 0.0, "end": 0.5, "text": "Hi"},
        {"start": 0.6, "end": 1.0, "text": "there"},
        {"start": 1.5, "end": 2.0, "text": "Hello"},
    ]
    turns = [
        {"start": 0.0, "end": 1.2, "speaker": "S1"},
        {"start": 1.3, "end": 2.5, "speaker": "S2"},
    ]
    out = rec.merge(words, turns)
    assert len(out) == 2
    assert out[0]["speaker"] == "S1" and out[0]["text"] == "Hi there"
    assert out[1]["speaker"] == "S2" and out[1]["text"] == "Hello"


def test_reconciliation_no_diarization_falls_back_to_single_speaker():
    from pipeline import reconciliation as rec
    words = [
        {"start": 0.0, "end": 0.3, "text": "one"},
        {"start": 0.4, "end": 0.7, "text": "two"},
    ]
    out = rec.merge(words, [])
    assert len(out) == 1
    assert out[0]["speaker"] == "SPEAKER_0"


# ── timeline ─────────────────────────────────────────────────────────────────
def test_timeline_computes_silence_and_overlap():
    from pipeline import timeline as tl
    turns = [
        {"start": 0.0, "end": 5.0, "speaker": "A"},
        {"start": 4.0, "end": 8.0, "speaker": "B"},  # 1s overlap with A
    ]
    out = tl.compute(turns, duration_sec=10.0)
    assert out["silence_sec"] == 2.0     # 8-10s
    assert out["overlap_sec"] == 1.0
    assert out["interruptions"] == 1
    assert len(out["interruption_events"]) == 1


# ── text_metrics ─────────────────────────────────────────────────────────────
def test_text_metrics_counts_questions_and_fillers():
    from pipeline import text_metrics as tm
    segs = [
        {"speaker": "A", "text": "How much is it? I mean, like, how much, you know?"},
        {"speaker": "B", "text": "It's expensive."},
    ]
    counts = tm.for_speaker(segs, "A")
    assert counts["question_count"] == 2
    assert counts["filler_count"]  >= 3   # "i mean", "like", "you know"
    assert counts["buying_signals"] >= 2  # "how much" appears twice


# ── resource_sampler ─────────────────────────────────────────────────────────
def test_resource_pct_clamps_and_rounds():
    from pipeline import resource_sampler as rs
    assert rs._pct(0, 0) is None             # no capacity → None
    assert rs._pct(8_000_000, 16_000_000) == 50.0
    assert rs._pct(20_000_000, 16_000_000) == 100.0   # clamp high
    assert rs._pct(-5, 16_000_000) == 0.0             # clamp low


def test_resource_triple_computes_difference():
    from pipeline import resource_sampler as rs
    assert rs._triple(40.0, 62.0) == {"rel": 40.0, "glob": 62.0, "other": 22.0}
    assert rs._triple(None, 62.0) == {"rel": None, "glob": 62.0, "other": None}


def test_resource_track_returns_peaks_dict():
    from pipeline import resource_sampler as rs
    with rs.track() as tr:
        _ = sum(range(10000))
    assert set(tr.peaks) == {"gpu", "vram", "cpu", "ram"}
    assert isinstance(tr.elapsed, float)


# ── jobs._model_entry ────────────────────────────────────────────────────────
def test_jobs_model_entry_includes_peaks_when_given():
    from pipeline import jobs
    e = jobs._model_entry("whisper-base", "asr", 2.345,
                          {"vram": {"rel": 18.0, "glob": 55.0, "other": 37.0}})
    assert e["model"] == "whisper-base"
    assert e["duration_sec"] == 2.345
    assert e["peaks"]["vram"]["glob"] == 55.0


def test_jobs_model_entry_omits_peaks_when_none():
    from pipeline import jobs
    e = jobs._model_entry("whisper-base", "asr", 2.0, None)
    assert "peaks" not in e


# ── jobs._concurrent_entry ─────────────────────────────────────────────────────
def test_jobs_concurrent_entry_keeps_per_member_time_one_shared_peak():
    """A concurrent group keeps each member's OWN duration (time is
    attributable per step) but carries a SINGLE shared peak (resource % is
    not separable on one GPU)."""
    from pipeline import jobs
    shared = {"gpu":  {"rel": None, "glob": 99.0, "other": None},
              "vram": {"rel": 38.5, "glob": 50.5, "other": 12.0}}
    e = jobs._concurrent_entry(
        members=[
            {"model": "iic/emotion2vec_plus_large", "kind": "emotion",  "duration_sec": 43.184},
            {"model": "gemma4:e4b",                 "kind": "kpis_llm", "duration_sec": 40.1},
            {"model": "opensmile-eGeMAPSv02",        "kind": "prosody",  "duration_sec": 3.0},
        ],
        duration_sec=44.0,
        peaks=shared,
    )
    assert e["kind"] == "concurrent"
    assert e["duration_sec"] == 44.0
    # one shared peak for the whole window
    assert e["peaks"]["gpu"]["glob"] == 99.0
    # each member keeps its own individual duration
    assert [m["kind"] for m in e["members"]] == ["emotion", "kpis_llm", "prosody"]
    assert e["members"][0]["duration_sec"] == 43.184
    assert e["members"][1]["duration_sec"] == 40.1
    # members do NOT carry their own peaks (shared above)
    assert all("peaks" not in m for m in e["members"])



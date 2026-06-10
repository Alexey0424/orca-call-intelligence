"""SVG chart bundle.

Each module exposes a render function returning an SVG string.
``build_bundle`` orchestrates them into the final dict consumed by the
HTML template and ``flush_to_disk``.

v5: stacked-sentiment chart removed (info redundant with sentiment_timeline);
``conversation_flow`` is now a single horizontal stack; ``role_metrics`` is
a paired-bar scorecard; new ``phase_strip`` chart.
"""
from __future__ import annotations

from typing import Any, Dict

from charts import bars, donut, emotion_profile, phase_strip, sentiment_timeline


def build_bundle(executive: Dict[str, Any]) -> Dict[str, str]:
    cd = executive["chart_data"]
    timeline_segs = cd.get("timeline_segments", []) or []
    duration_sec  = cd.get("duration_sec", 1.0) or 1.0
    events        = cd.get("events", {})  or {}
    rm            = cd["role_metrics"]
    flow          = cd["conversation_flow"]

    return {
        "talk_share": donut.render(
            cd["talk_share"]["closer_pct"],
            cd["talk_share"]["client_pct"],
        ),
        "conversation_flow": bars.horizontal_stack(
            segments=[
                ("Closer talk",  flow["closer_talk_pct"], "closer"),
                ("Client talk",  flow["client_talk_pct"], "client"),
                ("Silence",      flow["silence_pct"],     "subtle"),
                ("Overlap",      flow["overlap_pct"],     "accent"),
            ],
            title="Conversation Dynamics",
        ),
        "role_metrics": bars.scorecard(
            metrics=[
                ("Words",     float(rm["words"][0]),     float(rm["words"][1])),
                ("Questions", float(rm["questions"][0]), float(rm["questions"][1])),
                ("Fillers",   float(rm["fillers"][0]),   float(rm["fillers"][1])),
                ("WPM",       float(rm["wpm"][0]),       float(rm["wpm"][1])),
            ],
            title="Communication Metrics · Closer vs Client",
        ),
        "sentiment_timeline": sentiment_timeline.render(
            timeline_segs, duration_sec, events,
        ),
        "emotion_profile": emotion_profile.render(timeline_segs),
        "phase_strip":     phase_strip.render(executive.get("phase_strip", {})),
    }


def flush_to_disk(charts: Dict[str, str], assets_dir) -> Dict[str, Any]:
    """Write each SVG string to ``assets_dir/<name>.svg`` and return the paths."""
    paths = {}
    for name, svg in charts.items():
        path = assets_dir / f"{name}.svg"
        path.write_text(svg, encoding="utf-8")
        paths[name] = path
    return paths

"""Three-panel sentiment + activity timeline — pure SVG.

Replaces the matplotlib PNG so the report is fully vector. Three panels
stacked vertically:

    1. Closer sentiment arc      (smoothed line, green/red fill at zero)
    2. Client sentiment arc      (same)
    3. Speaker activity swim lane (rect bars per turn, ⚠/✓ event marks)
"""
from __future__ import annotations

from typing import Any, Dict, List

from charts.base import PAL, svg_close, svg_open, text_escape


# ── Layout constants ─────────────────────────────────────────────────────────
W = 1100
H = 700
PAD_LEFT, PAD_RIGHT = 70, 38
PAD_TOP             = 50
ROW_HEIGHTS = (230, 230, 140)
ROW_GAP = 18


# ── Pure-Python smoother (replaces scipy.uniform_filter1d) ───────────────────
def _boxcar(values: List[float], window: int) -> List[float]:
    if window <= 1 or len(values) < window:
        return list(values)
    out: List[float] = []
    half = window // 2
    n = len(values)
    for i in range(n):
        a, b = max(0, i - half), min(n, i + half + 1)
        chunk = values[a:b]
        out.append(sum(chunk) / len(chunk))
    return out


# ── Per-segment sentiment value: -1..+1 ──────────────────────────────────────
def _segment_score(seg: Dict[str, Any]) -> float:
    label = str(seg.get("sentiment", "neutral")).lower()
    raw = float(seg.get("sentiment_score") or 0.0)
    v = raw if abs(raw) <= 1.0 else raw / 100.0
    if label == "positive":
        return max(0.05, abs(v))
    if label == "negative":
        return -max(0.05, abs(v))
    return 0.0


# ── Sentiment line panel (closer or client) ──────────────────────────────────
def _sentiment_panel(*, segments: List[Dict[str, Any]], role: str,
                     duration_sec: float, title_text: str,
                     y_top: float, y_bottom: float) -> str:
    parts: List[str] = []
    plot_height = y_bottom - y_top
    plot_left   = PAD_LEFT
    plot_right  = W - PAD_RIGHT
    plot_width  = plot_right - plot_left
    zero_y      = y_top + plot_height / 2

    # Frame
    parts.append(
        f'<line x1="{plot_left}" y1="{zero_y}" x2="{plot_right}" y2="{zero_y}" '
        f'stroke="{PAL["border"]}" stroke-width="0.9" stroke-dasharray="4 3"/>'
    )

    # Title (top-left of panel) — bigger and bolder than v4
    parts.append(
        f'<text x="{plot_left}" y="{y_top - 12}" font-size="13" '
        f'font-weight="800" fill="{PAL["brand"]}" letter-spacing="0.02em">'
        f'{text_escape(title_text)}</text>'
    )

    role_segs = [s for s in segments if s.get("role") == role]
    if not role_segs:
        parts.append(
            f'<text x="{(plot_left + plot_right) / 2}" y="{zero_y + 6}" '
            f'text-anchor="middle" font-size="11" fill="{PAL["subtle"]}" '
            f'font-style="italic">No data</text>'
        )
        return "".join(parts)

    # Y-axis tick labels at -1, 0, +1
    for tick_val in (-1, 0, 1):
        ty = zero_y - tick_val * (plot_height / 2)
        parts.append(
            f'<text x="{plot_left - 6}" y="{ty + 3:.1f}" text-anchor="end" '
            f'font-size="8" fill="{PAL["subtle"]}">{tick_val:+.1f}</text>'
        )

    raw_xs = [(s["start"] + s["end"]) / 2 for s in role_segs]
    raw_ys = [_segment_score(s) for s in role_segs]
    smooth_ys = _boxcar(raw_ys, min(5, max(1, len(raw_ys))))

    def to_px(t: float, v: float):
        x = plot_left + (t / duration_sec) * plot_width if duration_sec > 0 else plot_left
        y = zero_y - v * (plot_height / 2)
        return x, y

    # Filled regions: positive (green) and negative (red), each clipped at zero.
    # Build a polygon under the smoothed curve.
    def _fill_polygon(positive: bool) -> str:
        pts = [(to_px(t, max(0, v) if positive else min(0, v))) for t, v in zip(raw_xs, smooth_ys)]
        if not pts:
            return ""
        # Close along the zero line
        path = f"M{pts[0][0]:.1f} {zero_y:.1f} "
        for x, y in pts:
            path += f"L{x:.1f} {y:.1f} "
        path += f"L{pts[-1][0]:.1f} {zero_y:.1f} Z"
        color = PAL["green"] if positive else PAL["red"]
        return f'<path d="{path}" fill="{color}" fill-opacity="0.20"/>'

    parts.append(_fill_polygon(positive=True))
    parts.append(_fill_polygon(positive=False))

    # Smoothed curve
    line_color = PAL["closer"] if role == "closer" else PAL["client"]
    pts = [to_px(t, v) for t, v in zip(raw_xs, smooth_ys)]
    if pts:
        path = "M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in pts)
        parts.append(
            f'<path d="{path}" stroke="{line_color}" stroke-width="2" '
            f'fill="none" opacity="0.9"/>'
        )

    # Per-segment dots
    for t, v in zip(raw_xs, raw_ys):
        x, y = to_px(t, v)
        dot_color = PAL["green"] if v > 0 else PAL["red"] if v < 0 else PAL["subtle"]
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.5" fill="{dot_color}" opacity="0.75"/>'
        )

    return "".join(parts)


# ── Swim-lane panel (speaker activity + event markers) ───────────────────────
def _swim_lane_panel(*, segments: List[Dict[str, Any]], duration_sec: float,
                     events: Dict[str, Any],
                     y_top: float, y_bottom: float) -> str:
    parts: List[str] = []
    plot_left   = PAD_LEFT
    plot_right  = W - PAD_RIGHT
    plot_width  = plot_right - plot_left
    plot_height = y_bottom - y_top

    closer_y = y_top + plot_height * 0.20
    client_y = y_top + plot_height * 0.65
    bar_h    = plot_height * 0.18

    # Title + axis labels — bigger row labels, clearer hierarchy
    parts.append(
        f'<text x="{plot_left}" y="{y_top - 12}" font-size="13" '
        f'font-weight="800" fill="{PAL["brand"]}" letter-spacing="0.02em">'
        f'Speaker Activity</text>'
        f'<text x="{plot_left - 8}" y="{closer_y + bar_h/2 + 3:.1f}" '
        f'text-anchor="end" font-size="10" font-weight="700" '
        f'fill="{PAL["closer"]}">CLOSER</text>'
        f'<text x="{plot_left - 8}" y="{client_y + bar_h/2 + 3:.1f}" '
        f'text-anchor="end" font-size="10" font-weight="700" '
        f'fill="{PAL["client"]}">CLIENT</text>'
    )

    def x_of(t: float) -> float:
        if duration_sec <= 0:
            return plot_left
        return plot_left + (t / duration_sec) * plot_width

    for seg in segments:
        x_start = x_of(seg["start"])
        x_end   = x_of(seg["end"])
        width   = max(1.5, x_end - x_start)
        if seg.get("role") == "closer":
            y, color = closer_y, PAL["closer"]
        elif seg.get("role") == "client":
            y, color = client_y, PAL["client"]
        else:
            continue
        parts.append(
            f'<rect x="{x_start:.1f}" y="{y:.1f}" width="{width:.1f}" '
            f'height="{bar_h:.1f}" fill="{color}" opacity="0.65"/>'
        )

    # Event markers (objections + buying signals)
    def _events_of(key: str) -> List[Dict]:
        return events.get(key) or []

    for evt in _events_of("objection_events"):
        t = float(evt.get("start_sec", evt.get("start", 0.0)))
        x = x_of(t)
        parts.append(
            f'<line x1="{x:.1f}" y1="{y_top + 4}" x2="{x:.1f}" y2="{y_bottom}" '
            f'stroke="{PAL["red"]}" stroke-width="1.2" stroke-dasharray="2 2" '
            f'opacity="0.7"/>'
            f'<text x="{x:.1f}" y="{y_top + 12}" text-anchor="middle" '
            f'font-size="10" fill="{PAL["red"]}">⚠</text>'
        )
    for evt in _events_of("buying_signal_events"):
        t = float(evt.get("start_sec", evt.get("start", 0.0)))
        x = x_of(t)
        parts.append(
            f'<line x1="{x:.1f}" y1="{y_top + 4}" x2="{x:.1f}" y2="{y_bottom}" '
            f'stroke="{PAL["green"]}" stroke-width="1.2" stroke-dasharray="2 2" '
            f'opacity="0.7"/>'
            f'<text x="{x:.1f}" y="{y_top + 12}" text-anchor="middle" '
            f'font-size="10" fill="{PAL["green"]}">✓</text>'
        )

    # X-axis tick labels (seconds)
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        t = duration_sec * frac
        x = x_of(t)
        parts.append(
            f'<text x="{x:.1f}" y="{y_bottom + 14:.1f}" text-anchor="middle" '
            f'font-size="8" fill="{PAL["subtle"]}">{int(round(t))}s</text>'
        )

    return "".join(parts)


def render(timeline_segments: List[Dict[str, Any]], duration_sec: float,
           events: Dict[str, Any]) -> str:
    duration_sec = max(float(duration_sec or 1.0), 1.0)
    y_cursor = PAD_TOP
    panels: List[str] = []

    panels.append(_sentiment_panel(
        segments=timeline_segments, role="closer", duration_sec=duration_sec,
        title_text="Closer · Sentiment Arc",
        y_top=y_cursor, y_bottom=y_cursor + ROW_HEIGHTS[0],
    ))
    y_cursor += ROW_HEIGHTS[0] + ROW_GAP

    panels.append(_sentiment_panel(
        segments=timeline_segments, role="client", duration_sec=duration_sec,
        title_text="Client · Sentiment Arc",
        y_top=y_cursor, y_bottom=y_cursor + ROW_HEIGHTS[1],
    ))
    y_cursor += ROW_HEIGHTS[1] + ROW_GAP

    panels.append(_swim_lane_panel(
        segments=timeline_segments, duration_sec=duration_sec, events=events,
        y_top=y_cursor, y_bottom=y_cursor + ROW_HEIGHTS[2],
    ))

    return "".join([svg_open(W, H), *panels, svg_close()])

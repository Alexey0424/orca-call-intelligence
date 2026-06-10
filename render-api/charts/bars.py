"""Bar charts.

In v5 the old vertical/grouped/stacked-sentiment trio was replaced with two
denser, more legible variants:

- ``horizontal_stack`` — a single horizontal bar split into proportional
  segments. Used for "where did the call's time go" (closer / client /
  silence / overlap). Conveys in one line what 4 vertical bars used to.

- ``scorecard``       — paired bars per metric (closer vs client) with a
  ``±`` delta tag rendered to the right of each pair. Used for the
  Performance Breakdown section.
"""
from __future__ import annotations

from typing import List, Tuple

from charts.base import PAL, svg_close, svg_open, text_escape
from charts.base import title as svg_title


# ── Horizontal stacked bar ──────────────────────────────────────────────────
def horizontal_stack(*, segments: List[Tuple[str, float, str]],
                     title: str, ylabel: str = "% of call") -> str:
    """Render a single horizontal bar split by ``segments``.

    Each segment is ``(label, percent, palette_key)``. Percents do not need
    to sum to 100; the bar fills in proportion to whatever you pass.
    """
    W, H = 540, 110
    pl, pr, pt, pb = 18, 18, 36, 28
    pw, ph = W - pl - pr, H - pt - pb
    bar_y, bar_h = pt + 4, ph - 8
    total = max(1.0, sum(s[1] for s in segments))

    # Left-to-right segments
    bars: List[str] = []
    cursor_x = pl
    for label, value, color_key in segments:
        seg_w = (value / total) * pw
        if seg_w < 0.5:
            cursor_x += seg_w
            continue
        color = PAL.get(color_key, PAL["subtle"])
        bars.append(
            f'<rect x="{cursor_x:.1f}" y="{bar_y:.1f}" '
            f'width="{seg_w:.1f}" height="{bar_h:.1f}" '
            f'fill="{color}"/>'
        )
        if seg_w > 36:
            bars.append(
                f'<text x="{cursor_x + seg_w / 2:.1f}" y="{bar_y + bar_h / 2 + 4:.1f}" '
                f'text-anchor="middle" font-size="11" font-weight="700" '
                f'fill="white">{value:.0f}%</text>'
            )
        cursor_x += seg_w

    # Legend (below bar)
    legend_parts: List[str] = []
    cursor_x = pl
    for label, value, color_key in segments:
        color = PAL.get(color_key, PAL["subtle"])
        legend_parts.append(
            f'<rect x="{cursor_x:.0f}" y="{H - 14}" width="9" height="9" '
            f'rx="2" fill="{color}"/>'
            f'<text x="{cursor_x + 12:.0f}" y="{H - 6}" font-size="10" '
            f'fill="{PAL["text"]}">{text_escape(label)} '
            f'<tspan fill="{PAL["subtle"]}">{value:.0f}%</tspan></text>'
        )
        cursor_x += len(label) * 7 + 60

    return "".join([
        svg_open(W, H),
        svg_title(title, W / 2, y=20),
        f'<rect x="{pl}" y="{bar_y}" width="{pw}" height="{bar_h}" '
        f'fill="{PAL["panel"]}" stroke="{PAL["border"]}" stroke-width="0.5"/>',
        "".join(bars),
        "".join(legend_parts),
        svg_close(),
    ])


# ── Side-by-side scorecard ──────────────────────────────────────────────────
def scorecard(*, metrics: List[Tuple[str, float, float]],
              title: str = "Closer vs Client") -> str:
    """Render a vertical stack of metric rows.

    Each metric is ``(label, closer_value, client_value)``. We draw two
    horizontal bars (closer above, client below) of length proportional to
    the value, plus a ``±delta`` chip on the right.
    """
    W = 540
    row_h = 40
    pt    = 36
    H     = pt + len(metrics) * row_h + 14

    pl, pr = 96, 88
    pw     = W - pl - pr

    # 10% headroom so the value label after a max-width bar can't slide under
    # the Δ chip on the right edge (chip starts at W - pr + 6).
    max_v = (max((max(c, l) for _, c, l in metrics), default=1) or 1) * 1.10

    rows: List[str] = []
    for i, (label, closer_v, client_v) in enumerate(metrics):
        ry      = pt + i * row_h
        bw_c    = (closer_v / max_v) * pw if max_v > 0 else 0
        bw_l    = (client_v / max_v) * pw if max_v > 0 else 0
        delta   = closer_v - client_v
        chip_bg = (PAL["green"] if delta > 0
                   else PAL["red"] if delta < 0
                   else PAL["subtle"])
        delta_text = (f"+{delta:.0f}" if delta > 0
                      else f"{delta:.0f}" if delta < 0
                      else "—")

        rows.append(
            # Metric label
            f'<text x="{pl - 8}" y="{ry + 14:.0f}" text-anchor="end" '
            f'font-size="10" font-weight="600" fill="{PAL["text"]}">'
            f'{text_escape(label)}</text>'
            # Closer bar (top)
            f'<rect x="{pl}" y="{ry + 4}" width="{bw_c:.1f}" height="11" '
            f'fill="{PAL["closer"]}" rx="1.5"/>'
            f'<text x="{pl + bw_c + 4:.1f}" y="{ry + 13}" font-size="9" '
            f'fill="{PAL["brand"]}">{closer_v:.0f}</text>'
            # Client bar (bottom)
            f'<rect x="{pl}" y="{ry + 19}" width="{bw_l:.1f}" height="11" '
            f'fill="{PAL["client"]}" rx="1.5"/>'
            f'<text x="{pl + bw_l + 4:.1f}" y="{ry + 28}" font-size="9" '
            f'fill="{PAL["brand"]}">{client_v:.0f}</text>'
            # Delta chip on the right
            f'<rect x="{W - pr + 6}" y="{ry + 9}" width="64" height="16" '
            f'rx="8" fill="{chip_bg}"/>'
            f'<text x="{W - pr + 38}" y="{ry + 21}" text-anchor="middle" '
            f'font-size="10" font-weight="700" fill="white">'
            f'Δ {text_escape(delta_text)}</text>'
        )

    # Legend pinned to the top-right column (above the Δ chips), so the
    # centred title at y=14 has the whole header row to itself. Mirrors the
    # layout used in emotion_profile.py.
    legend = (
        f'<rect x="{W - pr + 6}" y="2"  width="9" height="9" rx="2" fill="{PAL["closer"]}"/>'
        f'<text x="{W - pr + 18}" y="10" font-size="9" fill="{PAL["text"]}">Closer</text>'
        f'<rect x="{W - pr + 6}" y="14" width="9" height="9" rx="2" fill="{PAL["client"]}"/>'
        f'<text x="{W - pr + 18}" y="22" font-size="9" fill="{PAL["text"]}">Client</text>'
    )

    return "".join([
        svg_open(W, H),
        svg_title(title, W / 2, y=14),
        legend,
        "".join(rows),
        svg_close(),
    ])

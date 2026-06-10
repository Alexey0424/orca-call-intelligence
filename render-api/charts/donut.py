"""Talk-share donut chart.

Centre text reads "Closer dominates" / "Balanced" / "Client-led" depending on
how lopsided the split is — saves the reader from squinting at percentages.
"""
from __future__ import annotations

import math

from charts.base import PAL, svg_close, svg_open, text_escape, title


def _wedge(*, fraction_start: float, fraction_end: float, color: str,
           cx: float, cy: float, R: float, r: float, panel: str,
           label_pct: float) -> str:
    # Skip wedges that collapse to a point — at fraction ~0 the SVG arc
    # degenerates into a stub at 12 o'clock and the floating white "0%"
    # label lands on empty card background.
    if (fraction_end - fraction_start) < 0.005:
        return ""

    a0 = math.radians(fraction_start * 360 - 90)
    a1 = math.radians(fraction_end   * 360 - 90)
    large_arc = 1 if (fraction_end - fraction_start) > 0.5 else 0

    x0o, y0o = cx + R * math.cos(a0), cy + R * math.sin(a0)
    x1o, y1o = cx + R * math.cos(a1), cy + R * math.sin(a1)
    x0i, y0i = cx + r * math.cos(a1), cy + r * math.sin(a1)
    x1i, y1i = cx + r * math.cos(a0), cy + r * math.sin(a0)

    d = (f"M{x0o:.1f} {y0o:.1f} A{R} {R} 0 {large_arc} 1 {x1o:.1f} {y1o:.1f} "
         f"L{x0i:.1f} {y0i:.1f} A{r} {r} 0 {large_arc} 0 {x1i:.1f} {y1i:.1f}Z")

    mid_angle = math.radians((fraction_start + fraction_end) / 2 * 360 - 90)
    mid_r     = (R + r) / 2
    tx, ty    = cx + mid_r * math.cos(mid_angle), cy + mid_r * math.sin(mid_angle)

    return (
        f'<path d="{d}" fill="{color}" stroke="{panel}" stroke-width="2.5"/>'
        f'<text x="{tx:.1f}" y="{ty:.1f}" text-anchor="middle" '
        f'dominant-baseline="central" fill="white" '
        f'font-size="10" font-weight="bold">{label_pct:.0f}%</text>'
    )


def _verdict(closer_pct: float) -> str:
    if closer_pct >= 65:
        return "Closer dominates"
    if closer_pct <= 35:
        return "Client-led"
    return "Balanced"


def render(closer_pct: float, client_pct: float) -> str:
    W, H = 290, 220
    cx, cy = 145, 110
    R, r = 82, 50

    closer_fraction = max(0.0, min(1.0, closer_pct / 100.0))
    verdict = _verdict(closer_pct)

    parts = [
        svg_open(W, H),
        title("Talk Share", W / 2),
        # Wedges
        _wedge(fraction_start=0, fraction_end=closer_fraction,
               color=PAL["closer"], cx=cx, cy=cy, R=R, r=r,
               panel=PAL["panel"], label_pct=closer_pct),
        _wedge(fraction_start=closer_fraction, fraction_end=1,
               color=PAL["client"], cx=cx, cy=cy, R=R, r=r,
               panel=PAL["panel"], label_pct=client_pct),
        # Centre text — 2 lines
        f'<text x="{cx}" y="{cy - 4}" text-anchor="middle" font-size="11" '
        f'font-weight="700" fill="{PAL["brand"]}">{text_escape(verdict)}</text>'
        f'<text x="{cx}" y="{cy + 11}" text-anchor="middle" font-size="9" '
        f'fill="{PAL["subtle"]}">{closer_pct:.0f}% / {client_pct:.0f}%</text>',
        # Legend (bottom)
        f'<rect x="55"  y="{H - 22}" width="10" height="10" rx="2" fill="{PAL["closer"]}"/>'
        f'<text x="69"  y="{H - 13}" font-size="10" fill="{PAL["text"]}">Closer</text>'
        f'<rect x="155" y="{H - 22}" width="10" height="10" rx="2" fill="{PAL["client"]}"/>'
        f'<text x="169" y="{H - 13}" font-size="10" fill="{PAL["text"]}">Client</text>',
        svg_close(),
    ]
    return "".join(parts)

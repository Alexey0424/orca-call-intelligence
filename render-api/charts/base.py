"""Shared SVG primitives — escaping, header, axis ticks.

The old code generated SVG by f-string concatenation with no escaping,
which breaks the moment a label contains ``<`` or ``&``. Everything here
escapes; chart code should never call ``str.format`` against raw user
input.
"""
from __future__ import annotations

from html import escape as _html_escape
from typing import Iterable, List, Tuple

import config


PAL = config.PAL
FONT_FAMILY = config.SVG_FONT_FAMILY


def text_escape(value: object) -> str:
    """Escape a string safe for SVG text/attribute context."""
    return _html_escape(str(value), quote=True)


def svg_open(width: int, height: int) -> str:
    # Transparent background so the chart blends into the page paper instead
    # of stamping its own lighter "card" rectangle on top. The visible cream
    # box that used to ring every chart came from this background + a second
    # CSS-side card background; both are gone now.
    return (f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'font-family="{FONT_FAMILY}" '
            f'viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
            f'style="background:transparent;display:block">')


def svg_close() -> str:
    return "</svg>"


def title(text: str, x: float, y: float = 15) -> str:
    return (f'<text x="{x}" y="{y}" text-anchor="middle" font-size="11" '
            f'font-weight="bold" fill="{PAL["brand"]}">{text_escape(text)}</text>')


def axis_y_label(text: str, x: float, y: float) -> str:
    return (f'<text x="{x}" y="{y}" text-anchor="middle" font-size="8" '
            f'fill="{PAL["subtle"]}" transform="rotate(-90 {x} {y})">'
            f'{text_escape(text)}</text>')


def horizontal_grid(*, x_left: float, x_right: float, y_top: float, y_bottom: float,
                    max_value: float, ticks: Iterable[float]) -> str:
    """Horizontal grid lines + y-axis tick labels at percentages of ``max_value``."""
    if max_value <= 0:
        return ""
    ph = y_bottom - y_top
    parts: List[str] = []
    for tick in ticks:
        if tick > max_value * 1.01:
            continue
        y = y_bottom - (tick / max_value) * ph
        parts.append(
            f'<line x1="{x_left}" y1="{y:.1f}" x2="{x_right}" y2="{y:.1f}" '
            f'stroke="{PAL["border"]}" stroke-width="0.5"/>'
            f'<text x="{x_left - 3}" y="{y + 3:.1f}" text-anchor="end" '
            f'font-size="7" fill="{PAL["subtle"]}">{tick:.0f}</text>'
        )
    return "".join(parts)


def vertical_grid(*, y_top: float, y_bottom: float, x_left: float, x_right: float,
                  max_value: float, ticks: Iterable[float]) -> str:
    """Vertical grid lines along the x-axis (used by horizontal-bar charts)."""
    if max_value <= 0:
        return ""
    pw = x_right - x_left
    parts: List[str] = []
    for tick in ticks:
        if tick > max_value * 1.01:
            continue
        x = x_left + (tick / max_value) * pw
        parts.append(
            f'<line x1="{x:.1f}" y1="{y_top}" x2="{x:.1f}" y2="{y_bottom}" '
            f'stroke="{PAL["border"]}" stroke-width="0.5"/>'
            f'<text x="{x:.1f}" y="{y_bottom + 12}" text-anchor="middle" '
            f'font-size="7" fill="{PAL["subtle"]}">{tick:.0f}%</text>'
        )
    return "".join(parts)


def legend(items: List[Tuple[str, str]], *, x: float, y: float,
           font_size: int = 9, gap: int = 6) -> str:
    """Render a horizontal legend: each entry is (label, color)."""
    parts: List[str] = []
    cursor = x
    for label, color in items:
        parts.append(
            f'<rect x="{cursor:.1f}" y="{y - font_size:.1f}" width="9" height="9" '
            f'rx="2" fill="{color}"/>'
            f'<text x="{cursor + 12:.1f}" y="{y:.1f}" font-size="{font_size}" '
            f'fill="{PAL["text"]}">{text_escape(label)}</text>'
        )
        cursor += len(label) * (font_size - 2) + 12 + gap + 3
    return "".join(parts)

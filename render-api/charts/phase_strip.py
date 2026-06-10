"""Phase emotion strip — three boxes (Opening / Middle / Closing) per role.

A tiny, dense visual that answers "what was the emotional arc of the call?"
in three glances. Each box names the dominant emotion in that phase for the
role, color-coded by valence so the reader can sweep across the strip and
read the emotional shape: e.g. green→amber→green = recovered close.
"""
from __future__ import annotations

from typing import Any, Dict

from charts.base import PAL, svg_close, svg_open, text_escape, title


_VALENCE = {
    "happy":     PAL["green"],
    "calm":      PAL["green"],
    "surprised": PAL["accent"],
    "neutral":   PAL["subtle"],
    "sad":       PAL["red"],
    "angry":     PAL["red"],
    "fearful":   PAL["red"],
    "disgust":   PAL["red"],
    "disgusted": PAL["red"],
}


def _color(emotion: Any) -> str:
    if not emotion:
        return PAL["subtle"]
    return _VALENCE.get(str(emotion).lower(), PAL["subtle"])


def _box(*, x: float, y: float, w: float, h: float,
         phase_label: str, emotion: Any) -> str:
    color = _color(emotion)
    text  = (str(emotion) if emotion else "—").title()
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'rx="8" fill="{color}" fill-opacity="0.18" '
        f'stroke="{color}" stroke-width="1.4"/>'
        f'<text x="{x + w / 2:.1f}" y="{y + 18}" text-anchor="middle" '
        f'font-size="9" letter-spacing="0.08em" fill="{PAL["subtle"]}">'
        f'{text_escape(phase_label.upper())}</text>'
        f'<text x="{x + w / 2:.1f}" y="{y + h / 2 + 6:.1f}" text-anchor="middle" '
        f'font-size="14" font-weight="700" fill="{PAL["brand"]}">'
        f'{text_escape(text)}</text>'
    )


def _row(*, y: float, w_box: float, gap: float, x_start: float,
         role_label: str, phases: Dict[str, Any]) -> str:
    parts = [
        f'<text x="{x_start - 12:.1f}" y="{y + 38}" text-anchor="end" '
        f'font-size="10" font-weight="700" fill="{PAL["brand"]}">'
        f'{text_escape(role_label)}</text>'
    ]
    for i, (key, label) in enumerate((("opening", "Opening"),
                                        ("middle",  "Middle"),
                                        ("closing", "Closing"))):
        x = x_start + i * (w_box + gap)
        parts.append(_box(
            x=x, y=y, w=w_box, h=64,
            phase_label=label, emotion=phases.get(key),
        ))
    return "".join(parts)


def render(phase_strip: Dict[str, Dict[str, Any]]) -> str:
    closer_phases = phase_strip.get("closer", {}) or {}
    client_phases = phase_strip.get("client", {}) or {}

    W   = 540
    H   = 200
    box_w = 132
    gap   = 14
    x_start = 78

    parts = [
        svg_open(W, H),
        title("Emotional Arc by Phase", W / 2, y=18),
        _row(y=40,  w_box=box_w, gap=gap, x_start=x_start,
             role_label="Closer", phases=closer_phases),
        _row(y=120, w_box=box_w, gap=gap, x_start=x_start,
             role_label="Client", phases=client_phases),
        svg_close(),
    ]
    return "".join(parts)

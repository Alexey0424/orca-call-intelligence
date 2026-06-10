"""Top-emotions horizontal bar chart, color-coded by emotional valence.

Sorting: combined closer+client share descending so the most prominent
emotion is at the top. Each row carries a Δ chip showing the gap between
roles, useful for spotting "the closer is calm but the client is anxious"
type asymmetries at a glance.
"""
from __future__ import annotations

from typing import Dict, List

from charts.base import PAL, svg_close, svg_open, text_escape, title


_EMOTIONS = ["neutral", "happy", "surprised", "angry", "sad",
             "fearful", "disgust", "disgusted", "calm"]


# Color by valence (positive = green, negative = red, neutral/calm = subtle).
_VALENCE_COLOR = {
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


def _counts_per_role(timeline_segments: List[Dict]) -> Dict[str, Dict[str, int]]:
    cts: Dict[str, Dict[str, int]] = {
        "closer": {e: 0 for e in _EMOTIONS},
        "client": {e: 0 for e in _EMOTIONS},
    }
    for seg in timeline_segments:
        role = seg.get("role")
        emotion = str(seg.get("emotion", "neutral")).lower()
        if role in cts:
            cts[role][emotion if emotion in cts[role] else "neutral"] += 1
    return cts


def render(timeline_segments: List[Dict]) -> str:
    cts = _counts_per_role(timeline_segments)
    closer_total = max(sum(cts["closer"].values()), 1)
    client_total = max(sum(cts["client"].values()), 1)

    # Top 6 by combined share (descending).
    combined = sorted(
        ((e,
          100.0 * cts["closer"][e] / closer_total,
          100.0 * cts["client"][e] / client_total)
         for e in _EMOTIONS),
        key=lambda triple: triple[1] + triple[2],
        reverse=True,
    )
    rows = combined[:6]

    W, H = 540, 40 + 32 * len(rows) + 14
    pl, pr = 92, 92
    pt     = 30
    pw     = W - pl - pr
    max_v  = max([max(c, l) for _, c, l in rows] + [1]) * 1.10

    # Title + axis line
    parts: List[str] = [
        svg_open(W, H),
        title("Emotion Profile · Top Emotions by Role", W / 2, y=14),
        # Tick line at the bottom
        f'<line x1="{pl}" y1="{H - 18}" x2="{pl + pw}" y2="{H - 18}" '
        f'stroke="{PAL["border"]}" stroke-width="0.6"/>',
    ]
    for tick in (0, 25, 50, 75, 100):
        if tick > max_v:
            continue
        x = pl + (tick / max_v) * pw
        parts.append(
            f'<line x1="{x:.1f}" y1="{pt}" x2="{x:.1f}" y2="{H - 18}" '
            f'stroke="{PAL["border"]}" stroke-width="0.4" stroke-dasharray="2 2"/>'
            f'<text x="{x:.1f}" y="{H - 6}" text-anchor="middle" '
            f'font-size="8" fill="{PAL["subtle"]}">{tick}%</text>'
        )

    # Rows
    for i, (emotion, c_val, l_val) in enumerate(rows):
        ry = pt + i * 32
        color = _VALENCE_COLOR.get(emotion, PAL["subtle"])
        bw_c  = (c_val / max_v) * pw if max_v > 0 else 0
        bw_l  = (l_val / max_v) * pw if max_v > 0 else 0
        delta = c_val - l_val
        chip_bg = (PAL["green"] if delta > 0
                   else PAL["red"] if delta < 0
                   else PAL["subtle"])
        delta_text = (f"+{delta:.0f}" if delta > 0
                      else f"{delta:.0f}" if delta < 0
                      else "—")

        parts.append(
            # Emotion label (left)
            f'<text x="{pl - 8}" y="{ry + 17}" text-anchor="end" '
            f'font-size="10" font-weight="600" fill="{PAL["text"]}">'
            f'{text_escape(emotion.title())}</text>'
            # Closer bar
            f'<rect x="{pl}" y="{ry + 5}" width="{bw_c:.1f}" height="9" '
            f'fill="{color}" rx="1.5"/>'
            f'<text x="{pl + bw_c + 4:.1f}" y="{ry + 13}" font-size="8" '
            f'fill="{PAL["brand"]}">{c_val:.0f}%</text>'
            # Client bar (slightly faded for visual layering)
            f'<rect x="{pl}" y="{ry + 16}" width="{bw_l:.1f}" height="9" '
            f'fill="{color}" fill-opacity="0.5" rx="1.5"/>'
            f'<text x="{pl + bw_l + 4:.1f}" y="{ry + 24}" font-size="8" '
            f'fill="{PAL["brand"]}">{l_val:.0f}%</text>'
            # Delta chip
            f'<rect x="{W - pr + 6}" y="{ry + 7}" width="62" height="16" '
            f'rx="8" fill="{chip_bg}"/>'
            f'<text x="{W - pr + 37}" y="{ry + 19}" text-anchor="middle" '
            f'font-size="10" font-weight="700" fill="white">'
            f'Δ {text_escape(delta_text)}</text>'
        )

    # Legend (top right)
    parts.append(
        f'<rect x="{W - pr + 6}" y="2" width="9" height="9" rx="2" '
        f'fill="{PAL["closer"]}"/>'
        f'<text x="{W - pr + 18}" y="10" font-size="9" '
        f'fill="{PAL["text"]}">Closer</text>'
        f'<rect x="{W - pr + 6}" y="14" width="9" height="9" rx="2" '
        f'fill="{PAL["closer"]}" fill-opacity="0.5"/>'
        f'<text x="{W - pr + 18}" y="22" font-size="9" '
        f'fill="{PAL["text"]}">Client</text>'
    )

    parts.append(svg_close())
    return "".join(parts)

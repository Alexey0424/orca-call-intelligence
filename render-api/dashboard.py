"""Dashboard renderer — converts a dashboard data dict into HTML.

The endpoint accepts a clean data contract (see ``DASHBOARD_SCHEMA`` below)
and returns a static HTML page sized for a 1200px Gotenberg screenshot.

Design source: Claude Design bundle ``orca-dashboards/project/Dashboard.html``.
All visual decisions (palette, typography, layout, micro-charts) come from
that file; this module just computes the SVG geometry from real numbers and
hands a flat view-model to Jinja2.

SVG geometry lives here, not in the template, because:
  - donut segments need cumulative ``stroke-dashoffset`` math
  - the calls-per-day path is a step-line whose points depend on max_y
  - sparklines accept multiple ``kind`` values, each with different markup

Everything user-controlled is HTML-escaped at insertion time (Jinja's
autoescape handles plain values; explicit ``Markup`` is used for the SVG
fragments we built ourselves).
"""
from __future__ import annotations

import logging
from html import escape as he
from pathlib import Path
from typing import Any, Dict, List, Sequence

from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import Markup

import config

logger = logging.getLogger(__name__)


_ENV = Environment(
    loader=FileSystemLoader(str(config.TEMPLATES_DIR)),
    autoescape=select_autoescape(["html", "htm"]),
    keep_trailing_newline=True,
)


# ── KPI sparkline ────────────────────────────────────────────────────────────
def _spark_svg_inner(spark: Dict[str, Any] | None) -> Markup | None:
    """Build the inner-SVG markup for a KPI sparkline.

    ``spark`` shapes:
      {"kind": "line", "points": [22,22,22,22,12,4,4],
       "stroke": "var(--accent)", "endpoint_dot": true, "stroke_width": 1.4}
      {"kind": "flat-dashed"}
      {"kind": "fill-bar", "tint": "good"}   # 100%-saturated bar
      {"kind": "none"}                       # omit the SVG entirely
    """
    if not spark:
        return None
    kind = spark.get("kind", "line")

    if kind == "none":
        return None

    if kind == "flat-dashed":
        return Markup(
            '<line x1="0" y1="13" x2="240" y2="13" '
            'stroke="var(--rule)" stroke-width="1" stroke-dasharray="3 4"/>'
        )

    if kind == "fill-bar":
        tint = spark.get("tint", "good")
        bg = "var(--good-soft)" if tint == "good" else "var(--bad-soft)"
        fg = "var(--good)"      if tint == "good" else "var(--bad)"
        return Markup(
            f'<rect x="0" y="2" width="240" height="22" rx="3" fill="{bg}"/>'
            f'<rect x="0" y="2" width="240" height="22" rx="3" fill="{fg}" opacity="0.55"/>'
        )

    # kind == "line"
    points: Sequence[float] = spark.get("points") or [22] * 7
    if len(points) < 2:
        return None
    step = 240.0 / (len(points) - 1)
    coords = [(round(i * step, 1), round(float(y), 1)) for i, y in enumerate(points)]
    path_d = "M " + " L ".join(f"{x},{y}" for x, y in coords)
    stroke = spark.get("stroke", "var(--accent)")
    stroke_width = spark.get("stroke_width", 1.4)
    endpoint_dot = spark.get("endpoint_dot", True)
    last_x, last_y = coords[-1]

    out = (
        f'<path d="{path_d}" fill="none" stroke="{stroke}" '
        f'stroke-width="{stroke_width}" stroke-linecap="round" '
        f'stroke-linejoin="round"/>'
    )
    if endpoint_dot:
        out += (
            f'<circle cx="{last_x}" cy="{last_y}" r="2.4" fill="{stroke}"/>'
        )
    return Markup(out)


# ── Calls-per-day step-line ──────────────────────────────────────────────────
def _cpd_chart_svg_inner(cpd: Dict[str, Any]) -> Markup:
    """Build the inner-SVG markup for the calls-per-day chart.

    Layout (matches Dashboard.html):
      viewBox 720 × 220; usable plot y ∈ [30, 180]; x slots at
      [80, 180, 280, 380, 480, 580, 680] (7 days, step=100, start=80).
    """
    days: List[Dict[str, Any]] = cpd.get("days") or []
    max_y_raw = float(cpd.get("max_y") or 3)
    max_y = max(1.0, max_y_raw)

    y_top, y_bot = 30.0, 180.0
    plot_h = y_bot - y_top

    def y_for(v: float) -> float:
        return round(y_bot - (max(0.0, v) / max_y) * plot_h, 1)

    # Choose tick values: 0..max_y in unit steps for max_y ≤ 5, else 4 evenly-
    # spaced labels.
    if max_y <= 5:
        ticks = list(range(int(max_y), -1, -1))
    else:
        ticks = [int(round(max_y)), int(round(max_y * 2 / 3)),
                 int(round(max_y / 3)), 0]

    grid_parts: List[str] = []
    label_parts: List[str] = []
    for idx, t in enumerate(ticks):
        y = y_for(t)
        # Top + bottom are solid; middle gridlines are dashed.
        dash = '' if (idx == 0 or idx == len(ticks) - 1) else 'stroke-dasharray="2 4"'
        grid_parts.append(
            f'<line x1="40" y1="{y}" x2="720" y2="{y}" '
            f'stroke="var(--rule-soft)" stroke-width="1" {dash}/>'
        )
        label_parts.append(
            f'<text x="32" y="{y + 4}" text-anchor="end">{t}</text>'
        )

    # Day x-coords
    xs: List[float] = [80.0 + i * 100.0 for i in range(7)]
    if len(days) < 7:
        days = list(days) + [{"label": "", "value": 0, "active": False}] * (7 - len(days))
    days = days[:7]

    ys = [y_for(float(d.get("value") or 0)) for d in days]

    # Step path through the seven points
    path_d = "M " + " L ".join(f"{x},{y}" for x, y in zip(xs, ys))
    area_d = path_d + f" L {xs[-1]},{y_bot} L {xs[0]},{y_bot} Z"

    # Dotted average line (horizontal, at the mean of values)
    values = [float(d.get("value") or 0) for d in days]
    avg_y = y_for(sum(values) / len(values)) if values else y_for(0)
    avg_line = (
        f'<line x1="40" y1="{avg_y}" x2="720" y2="{avg_y}" '
        f'stroke="var(--ink-3)" stroke-width="1" stroke-dasharray="2 5"/>'
    ) if cpd.get("show_avg_line", True) else ""

    # Dots + value annotations
    dot_parts: List[str] = []
    annot_parts: List[str] = []
    xlabel_parts: List[str] = []
    for d, x, y in zip(days, xs, ys):
        active = bool(d.get("active"))
        if active:
            dot_parts.append(
                f'<circle cx="{x}" cy="{y}" r="4.5" '
                f'fill="var(--accent)" stroke="var(--paper)" stroke-width="1.6"/>'
            )
            annot_parts.append(
                f'<text x="{x}" y="{y - 16}" text-anchor="middle">{int(d.get("value") or 0)}</text>'
            )
            xlabel_parts.append(
                f'<text x="{x}" y="208" fill="var(--ink-2)" font-weight="500">'
                f'{he(str(d.get("label", "")))}</text>'
            )
        else:
            dot_parts.append(
                f'<circle cx="{x}" cy="{y}" r="3.5" '
                f'fill="var(--paper)" stroke="var(--accent)" stroke-width="1.4"/>'
            )
            xlabel_parts.append(
                f'<text x="{x}" y="208">{he(str(d.get("label", "")))}</text>'
            )

    svg = (
        # grid + y-labels
        f'<g>{"".join(grid_parts)}</g>'
        f'<g font-family="JetBrains Mono" font-size="10" fill="#968A78">'
        f'{"".join(label_parts)}</g>'
        # area
        f'<path d="{area_d}" fill="var(--accent-soft)" opacity="0.45"/>'
        # step line
        f'<path d="{path_d}" fill="none" stroke="var(--accent)" '
        f'stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>'
        # avg dotted
        f'{avg_line}'
        # dots
        f'<g>{"".join(dot_parts)}</g>'
        # value annotations
        f'<g font-family="Instrument Serif" font-style="italic" '
        f'font-size="20" fill="var(--ink)">{"".join(annot_parts)}</g>'
        # x-labels
        f'<g font-family="JetBrains Mono" font-size="10" fill="#968A78" '
        f'text-anchor="middle">{"".join(xlabel_parts)}</g>'
    )
    return Markup(svg)


# ── Donut ────────────────────────────────────────────────────────────────────
def _donut_svg_inner(segments: List[Dict[str, Any]]) -> Markup:
    """Build the inner-SVG markup for one donut chart.

    Each segment is rendered as a separate ``<circle>`` element with a
    ``stroke-dasharray`` of ``<pct> <100-pct>`` and a cumulative
    ``stroke-dashoffset`` so segments butt up against each other starting
    from 12 o'clock and going clockwise.

    A faint background ring is drawn first so a sub-100% donut still reads
    as a complete shape.
    """
    parts: List[str] = [
        '<circle cx="21" cy="21" r="15.915" fill="transparent" '
        'stroke="var(--paper-shade)" stroke-width="6"/>'
    ]
    offset = 25.0  # start at 12 o'clock
    for seg in segments or []:
        pct = max(0.0, float(seg.get("pct") or 0))
        if pct <= 0:
            continue
        color = seg.get("color", "var(--accent)")
        opacity = float(seg.get("opacity", 1.0))
        parts.append(
            f'<circle cx="21" cy="21" r="15.915" fill="transparent" '
            f'stroke="{color}" stroke-width="6" '
            f'stroke-dasharray="{pct:.2f} {100 - pct:.2f}" '
            f'stroke-dashoffset="{offset:.2f}" '
            + (f'opacity="{opacity}" ' if opacity != 1.0 else "")
            + f'transform="rotate(-90 21 21)"/>'
        )
        offset -= pct
    return Markup("".join(parts))


# ── Accent override CSS (when caller provides a custom accent) ───────────────
def _accent_overrides_css(accent: str | None) -> str | None:
    """Return inline CSS that re-binds --accent + derived shades.

    Mirrors the tweaks-panel JS in the source design: derives ``--accent-soft``
    (35% mix with white) and ``--accent-deep`` (75% darken) from ``accent``.
    Returns ``None`` for the default coral; the template skips the inline
    style attribute when ``None``.
    """
    if not accent:
        return None
    hex_ = accent.lstrip("#")
    if len(hex_) != 6:
        return None
    try:
        r, g, b = (int(hex_[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return None
    mix = lambda c, w: round(c * w + 255 * (1 - w))
    dark = lambda c, w: round(c * w)
    soft = f"rgb({mix(r, 0.35)},{mix(g, 0.35)},{mix(b, 0.35)})"
    deep = f"rgb({dark(r, 0.75)},{dark(g, 0.75)},{dark(b, 0.75)})"
    return f"--accent:{accent};--accent-soft:{soft};--accent-deep:{deep};"


# ── Public renderer ──────────────────────────────────────────────────────────
def render_dashboard_html(*, html_path: Path, data: Dict[str, Any],
                          render_id: str) -> None:
    """Render the dashboard view-model to ``html_path``.

    Caller owns the data contract; see this module's docstring + the
    template (templates/dashboard.html) for the exact field shapes.
    """
    # ── KPIs: precompute spark markup ───────────────────────────────────────
    kpis = []
    for k in data.get("kpis", []):
        kpis.append({
            **k,
            "spark_svg_inner": _spark_svg_inner(k.get("spark")),
        })

    # ── Calls per day ───────────────────────────────────────────────────────
    cpd_in = data.get("calls_per_day") or {}
    cpd = {
        "note":        cpd_in.get("note", ""),
        "chart_svg_inner": _cpd_chart_svg_inner(cpd_in),
        "side_stats":  cpd_in.get("side_stats", []),
    }

    # ── Trio of donuts ──────────────────────────────────────────────────────
    trio_in = data.get("trio") or {}
    donuts = []
    for d in trio_in.get("donuts", []):
        donuts.append({
            **d,
            "donut_svg_inner": _donut_svg_inner(d.get("segments") or []),
        })
    trio = {"note": trio_in.get("note", ""), "donuts": donuts}

    leaderboard = data.get("leaderboard") or {"note": "", "rows": []}
    recent = data.get("recent_calls") or {"note": "", "rows": []}

    template = _ENV.get_template("dashboard.html")
    html = template.render(
        doc_title=data.get("doc_title", "ORCA · Team Overview"),
        theme=data.get("theme", "paper"),
        density=data.get("density", "regular"),
        accent_overrides=_accent_overrides_css(data.get("accent")),

        # Brand fields are overridable by the caller (the view-model carries
        # the BRAND_* env values from rag-api); these are last-resort defaults.
        brand_mark=data.get("brand_mark", "O"),
        brand_name=data.get("brand_name", "ORCA"),
        brand_sub=data.get("brand_sub", "Call Intelligence"),
        brand_footer_left=data.get(
            "brand_footer_left", "ORCA · Call Intelligence Report"),
        brand_footer_right=data.get(
            "brand_footer_right", "Powered by ORCA"),

        scope_pip=data.get("scope_pip", "T"),
        scope_chip_text=data.get("scope_chip_text", "Team"),
        generated_stamp=data.get("generated_stamp", ""),
        eyebrow_text=data.get("eyebrow_text", ""),

        title_line_1=data.get("title_line_1", ""),
        title_line_2=data.get("title_line_2", ""),
        verdict_html=data.get("verdict_html", ""),

        kpis=kpis,
        cpd=cpd,
        trio=trio,
        leaderboard=leaderboard,
        recent=recent,

        render_short=render_id[-4:].lower() if render_id else "",
    )
    html_path.write_text(html, encoding="utf-8")
    logger.info("dashboard_html_written path=%s render_id=%s",
                html_path, render_id)

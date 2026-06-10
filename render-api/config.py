"""Centralised configuration: env vars, palette, paths.

The palette is the single source of truth for both SVG charts and the
HTML/CSS template. Don't introduce ad-hoc colour literals elsewhere.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict


# ── App ──────────────────────────────────────────────────────────────────────
APP_TITLE        = "render-api"
APP_VERSION      = "4.0.0"

OUTPUT_DIR       = Path(os.getenv("OUTPUT_DIR", "/data/reports"))
PUBLIC_BASE_URL  = os.getenv("PUBLIC_BASE_URL", "http://localhost:8010").rstrip("/")
COMPANY_NAME     = os.getenv("REPORT_COMPANY_NAME", "Sales Emotion Report")

LOG_LEVEL        = os.getenv("LOG_LEVEL", "INFO").upper()


# ── Paths ────────────────────────────────────────────────────────────────────
PACKAGE_DIR      = Path(__file__).parent
TEMPLATES_DIR    = PACKAGE_DIR / "templates"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Palette ── editorial / Financial-Times longread (v6, May 2026) ──────────
# Charts in charts/* pull everything from PAL, so updating values here re-skins
# every rendered SVG automatically — no edits inside charts/*.py needed.
PAL: Dict[str, str] = {
    "brand":       os.getenv("REPORT_BRAND_COLOR",  "#1A1915"),  # deep ink
    "accent":      os.getenv("REPORT_ACCENT_COLOR", "#8C2F1F"),  # terracotta
    "panel":       "#FBF7EF",   # warm card cream
    "border":      "#D9D2C4",   # warm hairline rule
    "text":        "#1A1915",   # ink
    "subtle":      "#8C857A",   # warm muted gray
    "green":       "#3B6B4A",   # muted forest (positive)
    "red":         "#A23B30",   # muted brick (negative)
    "closer":      "#6E8A6B",   # sage
    "client":      "#9A7B4C",   # umber
    "risk_low":    "#3B6B4A",
    "risk_medium": "#C08838",
    "risk_high":   "#A23B30",
    "paper":       "#F5F0E6",   # paper tone (used by chart backgrounds)
}


# ── SVG defaults ─────────────────────────────────────────────────────────────
# Manrope for body/UI, Newsreader for editorial display. WeasyPrint fetches
# both from Google Fonts at render time; if offline, the system falls back
# to the listed DejaVu/Arial chain.
SVG_FONT_FAMILY = ("Manrope,'Segoe UI',DejaVu Sans,Arial,sans-serif")


# ── Risk thresholds ──────────────────────────────────────────────────────────
RISK_HIGH_AT     = 6
RISK_MEDIUM_AT   = 3
INTERRUPTIONS_HEAVY  = 6
SILENCE_HEAVY_RATIO  = 0.20
OBJECTIONS_HEAVY     = 2

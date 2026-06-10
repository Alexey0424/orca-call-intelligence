"""Small, reusable helpers — no domain logic."""
from __future__ import annotations

import re
from html import escape as _html_escape
from pathlib import Path
from typing import Any, Optional

import config


def fnum(value: Any, default: float = 0.0) -> float:
    """Coerce to float, returning ``default`` on None or unparseable input."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def nullable_round(value: Any, ndigits: int = 2) -> Any:
    """Round if numeric, return ``None`` for None, pass through anything else."""
    if value is None:
        return None
    try:
        return round(float(value), ndigits)
    except (TypeError, ValueError):
        return value


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower().strip()).strip("-") or "report"


def format_duration(seconds: float) -> str:
    total = int(round(seconds))
    minutes, sec = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m {sec}s" if hours else f"{minutes}m {sec}s"


def trim(text: str, max_chars: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= max_chars else text[:max_chars - 1].rstrip() + "…"


def he(value: Any) -> str:
    """HTML-escape a value, including ``'`` (which the stdlib does only when ``quote=True``)."""
    return _html_escape(str(value), quote=True)


def public_url(path: Path) -> str:
    relative = path.relative_to(config.OUTPUT_DIR).as_posix()
    return f"{config.PUBLIC_BASE_URL}/reports/{relative}"


def risk_hex(level: Optional[str]) -> str:
    return config.PAL.get(f"risk_{(level or 'medium').lower()}", config.PAL["subtle"])

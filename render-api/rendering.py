"""HTML and PDF rendering.

The HTML is the single source of truth — the PDF is just WeasyPrint over
the same HTML. SVG charts are referenced by relative path from the
``assets/`` directory so WeasyPrint embeds them as true vectors.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import weasyprint
from jinja2 import Environment, FileSystemLoader, select_autoescape

import config
from utils import format_duration, fnum, he, risk_hex

logger = logging.getLogger(__name__)


_ENV = Environment(
    loader=FileSystemLoader(str(config.TEMPLATES_DIR)),
    autoescape=select_autoescape(["html", "htm"]),
    keep_trailing_newline=True,
)


# ── Sentiment pill helper ────────────────────────────────────────────────────
def _pill_breakdown(seg: Dict[str, Any]) -> Dict[str, Any]:
    """Return ``{POS, NEU, NEG, dominant}`` from any segment shape we accept."""
    probs = seg.get("sentiment_probs") or {}
    pos = fnum(probs.get("positive", probs.get("pos")), 0.0)
    neu = fnum(probs.get("neutral",  probs.get("neu")), 0.0)
    neg = fnum(probs.get("negative", probs.get("neg")), 0.0)
    total = pos + neu + neg

    if total <= 0:
        # Fallback: derive from sentiment label + score.
        label = str(seg.get("sentiment", "neutral")).lower()
        raw = fnum(seg.get("sentiment_score"), 0.0)
        strength = min(1.0, max(0.0, abs(raw) if abs(raw) <= 1 else abs(raw) / 100.0))
        if label == "positive":
            p = strength * 100; rest = 100 - p
            pos, neu, neg = p, rest * 0.7, rest * 0.3
        elif label == "negative":
            n = strength * 100; rest = 100 - n
            pos, neu, neg = rest * 0.3, rest * 0.7, n
        else:
            pos = max(0.0, (strength - 0.5) * 40)
            neg = max(0.0, (0.5 - strength) * 40)
            neu = max(0.0, 100 - pos - neg)
        total = pos + neu + neg

    scale = 100.0 / total
    values = {"POS": pos * scale, "NEU": neu * scale, "NEG": neg * scale}
    rounded = {k: int(round(v)) for k, v in values.items()}
    delta = 100 - sum(rounded.values())
    if delta:
        rounded[max(values, key=values.get)] += delta
    rounded["dominant"] = max(rounded, key=lambda k: rounded[k] if k != "dominant" else -1)
    return rounded


def _pills_html(seg: Dict[str, Any]) -> str:
    bd = _pill_breakdown(seg)
    return "".join(
        f'<span class="pill pill-{cls}{" active" if label == bd["dominant"] else ""}">'
        f'<b>{label}</b>&nbsp;{bd[label]:.0f}%</span>'
        for label, cls in (("POS", "pos"), ("NEU", "neu"), ("NEG", "neg"))
    )


# ── Variance table ───────────────────────────────────────────────────────────
def _variance_table_html(executive: Dict[str, Any]) -> str:
    summary = executive["summary_json"]
    rr = executive["role_rows"]
    closer_wpm = fnum(summary.get("closer_wpm"), 0)
    client_wpm = fnum(summary.get("client_wpm"), 0)

    def _diff(a: float, b: float) -> Tuple[str, str]:
        d = a - b
        return f"{int(d):+d}", f"{abs(d) / max(abs(b), 1) * 100:.0f}%"

    rows: List[Tuple[Any, ...]] = [
        ("Talk time",
         f"{summary['closer_talk_pct']:.1f}%", f"{summary['client_talk_pct']:.1f}%",
         *_diff(summary["closer_talk_pct"], summary["client_talk_pct"])),
        ("WPM",
         str(round(closer_wpm, 1)) if closer_wpm else "n/a",
         str(round(client_wpm, 1)) if client_wpm else "n/a",
         *(_diff(closer_wpm, client_wpm) if closer_wpm and client_wpm else ("—", "—"))),
        ("Words",        rr[0][3], rr[1][3], *_diff(fnum(rr[0][3]), fnum(rr[1][3]))),
        ("Questions",    rr[0][4], rr[1][4], *_diff(fnum(rr[0][4]), fnum(rr[1][4]))),
        ("Fillers",      rr[0][5], rr[1][5], *_diff(fnum(rr[0][5]), fnum(rr[1][5]))),
        ("Interruptions", summary["interruptions"], "—", "—", "—"),
    ]
    header = ("<tr><th>Metric</th><th>Closer</th><th>Client</th>"
              "<th>Diff</th><th>% diff</th></tr>")
    body = "".join(
        "<tr>" + "".join(f"<td>{he(c)}</td>" for c in row) + "</tr>"
        for row in rows
    )
    return f'<table class="var-table"><thead>{header}</thead><tbody>{body}</tbody></table>'


# ── Transcript view-model ────────────────────────────────────────────────────
def _transcript_turns(timeline_segs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for seg in timeline_segs:
        if not seg["text"]:
            continue
        role = seg["role"]
        sentiment = (seg.get("sentiment") or "neutral").lower()
        if sentiment not in ("positive", "negative"):
            sentiment = "neutral"
        emotion = (seg.get("emotion") or "")
        emotion_suffix = f" · {emotion.title()}" if emotion and emotion != "neutral" else ""
        timestamp = format_duration(seg["start"])
        speaker_label = ("Closer" if role == "closer"
                         else "Client" if role == "client"
                         else seg.get("speaker", ""))
        bubble_class = "bubble-closer" if role == "closer" else "bubble-client"
        out.append({
            "bubble_class":   bubble_class,
            "sent_class":     sentiment,
            "speaker_label":  speaker_label,
            "timestamp":      timestamp,
            "emotion_suffix": emotion_suffix,
            "text":           seg["text"],
            "pills_html":     _pills_html(seg),
        })
    return out


def _meta_items(metadata: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Operational metadata grid (`metadata.model_versions` JSONB-driven).

    May 2026 rebuild dropped the legacy ``metadata.whisper_model`` / ``bert_model`` /
    ``ser_model`` flat keys in favour of one ``model_versions`` JSONB carrying
    every model the pipeline used for this row. Surface them all so the
    operator can audit lineage at a glance.
    """
    mv = metadata.get("model_versions") or {}

    def _short(model_id: Any) -> str:
        s = str(model_id or "")
        if "/" in s:
            return s.rsplit("/", 1)[-1]
        return s

    items: List[Tuple[str, str]] = [
        ("Call ID",         str(metadata.get("call_id",  "n/a"))),
        ("Language",        str(metadata.get("language",
                                  metadata.get("language_hint",  "auto")))),
        ("Closer",          str(metadata.get("closer_name") or "unknown")),
        ("Client",          str(metadata.get("client_name") or "unknown")),
    ]

    if mv:
        items += [
            ("ASR backend",     str(mv.get("asr_backend", "n/a"))),
            ("ASR model",       _short(mv.get("asr"))),
            ("Whisper fallback", _short(mv.get("whisper_fallback"))),
            ("Diarization",     _short(mv.get("diarization"))),
            ("Sentiment",       _short(mv.get("sentiment"))),
            ("Speech emotion",  _short(mv.get("ser"))),
            ("Voice-ID",        _short(mv.get("voice_id"))),
            ("Entities (NER)",  _short(mv.get("ner"))),
            ("LLM",             _short(mv.get("llm"))),
            ("Embedding",       _short(mv.get("embedding"))),
        ]
    else:
        # Legacy input.json compat
        items += [
            ("Whisper",     str(metadata.get("whisper_model", "n/a"))),
            ("BERT",        str(metadata.get("bert_model",    "n/a"))),
            ("SER model",   str(metadata.get("ser_model",     "n/a"))),
            ("Diarization", str(metadata.get("diarization_model",
                              metadata.get("diarization_pipeline", "n/a")))),
        ]
    return items


# ── Public renderers ─────────────────────────────────────────────────────────
def render_html(*, html_path: Path, executive: Dict[str, Any],
                chart_paths: Dict[str, Path], render_id: str) -> None:
    """Render the report to ``html_path``."""
    timeline_segs = executive["chart_data"].get("timeline_segments", []) or []

    template = _ENV.get_template("report.html")
    css_template = _ENV.get_template("report.css")

    css = css_template.render(
        company_name=he(config.COMPANY_NAME),
        generated_at=he(executive["generated_at"]),
    )

    html = template.render(
        executive=executive,
        summary=executive["summary_json"],
        pal=config.PAL,
        risk_color=risk_hex(executive["risk_level"]),
        asset_url=lambda name: f"./assets/{chart_paths[name].name}",
        variance_table=_variance_table_html(executive),
        transcript_turns=_transcript_turns(timeline_segs),
        meta_items=_meta_items(executive.get("metadata", {})),
        render_id=render_id,
        stylesheet=css,
    )
    html_path.write_text(html, encoding="utf-8")


def render_pdf(*, pdf_path: Path, html_path: Path) -> None:
    """Convert ``html_path`` to ``pdf_path`` via WeasyPrint."""
    weasyprint.HTML(filename=str(html_path)).write_pdf(str(pdf_path))

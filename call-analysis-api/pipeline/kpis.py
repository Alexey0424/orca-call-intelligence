"""LLM-derived KPIs via Ollama (Qwen 2.5-14B-Instruct by default).

Sends a role-tagged, timestamped transcript and asks for a flat JSON
object of sales KPIs. The prompt lives in ``prompts/kpi_system.txt`` so
it can be edited without touching code.

Output JSON keys match the new ``calls`` table column names directly
(``next_step_secured``, ``predicted_outcome``, …) so persistence is a
1:1 mapping with no rename layer.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

import httpx

import config

logger = logging.getLogger(__name__)


_PROMPT_PATH = config.PROMPTS_DIR / "kpi_system.txt"
_FENCE_RE    = re.compile(r"^```(?:json)?\s*|\s*```$")
_JSON_OBJ_RE = re.compile(r"\{[\s\S]*\}")


def _system_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


def _format_transcript(utterances: List[Dict[str, Any]],
                       closer_speaker: str,
                       client_speaker: Optional[str]) -> str:
    lines: List[str] = []
    for u in utterances:
        spk = u.get("speaker", "UNKNOWN")
        if spk == closer_speaker:
            role = "CLOSER"
        elif client_speaker and spk == client_speaker:
            role = "CLIENT"
        else:
            role = spk
        mm, ss = divmod(int(float(u.get("start", 0))), 60)
        lines.append(f"[{mm:02d}:{ss:02d}] {role}: {u.get('text', '')}")
    return "\n".join(lines)


_LANGUAGE_NOTE = "The call is in English."


def call_llm(utterances: List[Dict[str, Any]],
             closer_speaker: str,
             client_speaker: Optional[str]) -> Dict[str, Any]:
    """Call Ollama with the KPI prompt and return the parsed JSON.

    Returns ``{"ok": False, "error": "..."}`` on any failure (network,
    schema mismatch, JSON parse error). Successful results are merged
    with ``{"ok": True}``.
    """
    body = {
        "model":  config.llm_model_id(),
        "stream": False,
        "format": "json",
        # gemma4 is a reasoning model: left to think, it spends ~200 s emitting
        # scratch tokens before the JSON (measured), which blows past LLM_TIMEOUT
        # under GPU load. We want structured JSON, not a chain of thought, so
        # disable thinking — drops KPI extraction from ~220 s to ~5 s. The
        # content->thinking fallback below stays as a safety net for model swaps.
        "think":  False,
        "options": {
            "temperature": 0.0,
            "num_ctx":     config.LLM_NUM_CTX,
            "num_predict": config.LLM_NUM_PREDICT,
        },
        "messages": [
            {"role": "system", "content": _system_prompt()},
            {"role": "user",   "content":
                f"{_LANGUAGE_NOTE}\n\n"
                f"TRANSCRIPT:\n"
                f"{_format_transcript(utterances, closer_speaker, client_speaker)}"},
        ],
    }
    try:
        logger.info("kpi_llm request model=%s msg_chars=%d",
                    body["model"],
                    sum(len(m["content"]) for m in body["messages"]))
        resp = httpx.post(f"{config.OLLAMA_HOST}/api/chat",
                          json=body, timeout=config.LLM_TIMEOUT_SEC)
        resp.raise_for_status()
        msg = (resp.json() or {}).get("message") or {}
        raw = (msg.get("content") or "").strip()
        # Some reasoning-mode models put structured output in `thinking`
        # when `content` is empty. Qwen 2.5-instruct does not, but we keep
        # the fallback for robustness across model swaps.
        source = "content"
        if not raw:
            raw = (msg.get("thinking") or "").strip()
            source = "thinking"
        raw = _FENCE_RE.sub("", raw).strip()
        obj_match = _JSON_OBJ_RE.search(raw)
        if obj_match:
            raw = obj_match.group(0)
        if not raw:
            raise ValueError("Ollama returned empty content and empty thinking")
        kpis = json.loads(raw)
        logger.info("kpi_llm parsed source=%s bytes=%d", source, len(raw))
    except httpx.HTTPError as exc:
        logger.error("kpi_llm http_error error=%r", exc)
        return {"ok": False, "error": f"http: {exc}"}
    except (KeyError, ValueError) as exc:
        try:
            body = resp.json()
            mm = (body or {}).get("message") or {}
            c = (mm.get("content")  or "")
            t = (mm.get("thinking") or "")
            logger.error("kpi_llm parse_error error=%r content_len=%d "
                         "thinking_len=%d content_head=%r thinking_head=%r",
                         exc, len(c), len(t), c[:160], t[:160])
        except Exception:
            logger.error("kpi_llm parse_error error=%r (response inspect failed)", exc)
        return {"ok": False, "error": f"parse: {exc}"}

    kpis["ok"] = True
    return kpis

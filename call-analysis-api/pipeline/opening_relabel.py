"""Opening-of-call speaker→role relabel via the LLM (DiarizationLM-style).

WHY: pyannote mislabels the first ~30 s of 2-speaker phone calls (cold-start
cluster instability + bloated opening ASR word timestamps), so the interim
"first speaker = client" rule in scorecard.build picks the wrong role. Acoustic
fixes can't recover it; reading the OPENING TEXT can — the agent pitches and
self-identifies, the client answered the phone and gives short replies.

WHAT: on an opening window we ask the LLM (the already-resident KPI model,
gemma4:e4b) to assign each opening utterance to speaker A/B and name the AGENT.
We then (a) decide which *diarization label* is the closer vs client and
(b) rewrite any individually-mislabelled opening utterances to the correct
diarization label. We keep working in diarization-label space (NOT role
strings) so prosody/timeline — which aggregate over the raw diarization
`turns` by label — are unaffected.

This is a post-processing step on frozen ASR + diarization (Wang et al. 2024,
"DiarizationLM"). It is text-in/decision-out: it NEVER edits the transcript,
only speaker labels. Safety rails:
  * anti-collapse guard — if the LLM lumps the whole opening onto one speaker
    while diarization found two, we reject it and keep the baseline rule;
  * strict agent validation — the agent label must be one of the speaker
    labels the LLM actually used (rejects ``"agent": null`` / hallucinations,
    which previously flipped every opening utterance to client);
  * retry + model fallback — the primary model is tried twice (transient
    Ollama 500s), then the KPI model once; only after that do we no-op;
  * any remaining failure returns None → caller falls back to the existing
    first-speaker rule. So this can only help or no-op.

Validated in experiments/asr_diarization_eval + data/diag/OPENING_RESEARCH.md:
qwen2.5:14b-instruct-q4_K_M is the reliable winner (client recall 1.0 on both
annotated calls, correct agent every run); gemma4:e4b is a usable fallback but
occasionally emits a null agent. ON by default (config.OPENING_RELABEL_ENABLED);
model via OPENING_RELABEL_MODEL (override file > env > KPI model).
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

import config

logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$")
_JSON_OBJ_RE = re.compile(r"\{[\s\S]*\}")

_SYSTEM = (
    "You label speakers in the opening of a 2-person OUTBOUND real-estate sales "
    "phone call. One speaker is the AGENT (the caller from a home-buying "
    "company: runs the pitch, asks qualifying questions, self-identifies like "
    "'this is <name> with <company>'). The other is the CLIENT (the property "
    "owner who ANSWERED the phone: the very first 'Hello' is the client; gives "
    "short replies about their own property). Assign each numbered utterance to "
    "speaker A or B (the two distinct voices) and decide which of A/B is the "
    'AGENT. Output ONLY JSON: {"turns":[{"i":0,"spk":"A"}, ...], "agent":"A"}'
)


def _call_ollama(lines: str, model: str) -> Dict[str, Any]:
    body = {
        "model": model, "stream": False, "format": "json", "think": False,
        # keep_alive 0: the relabel model may differ from the KPI model
        # (qwen for relabel, gemma for KPIs). Unload it right after the reply
        # so the audio stages + the KPI LLM never contend with it for VRAM.
        "keep_alive": 0,
        "options": {"temperature": 0.0, "num_ctx": 8192, "num_predict": 1024},
        "messages": [{"role": "system", "content": _SYSTEM},
                     {"role": "user", "content": "Opening utterances:\n" + lines}],
    }
    resp = httpx.post(f"{config.OLLAMA_HOST}/api/chat", json=body,
                      timeout=config.LLM_TIMEOUT_SEC)
    resp.raise_for_status()
    msg = (resp.json() or {}).get("message") or {}
    # gemma4 reasoning fallback: answer may land in content or thinking.
    raw = (msg.get("content") or "").strip() or (msg.get("thinking") or "").strip()
    raw = _FENCE_RE.sub("", raw).strip()
    m = _JSON_OBJ_RE.search(raw)
    if m:
        raw = m.group(0)
    if not raw:
        raise ValueError("empty completion")
    return json.loads(raw)


def _parse_and_validate(out: Dict[str, Any]) -> Optional[Tuple[Dict[int, str], str]]:
    """Parse the LLM reply into (spk_by_i, agent). Returns None when the reply
    is unusable: <2 distinct speakers (collapse), or an agent that is not one
    of the speakers the LLM actually used — this catches ``"agent": null``
    (str(None) == "None" is truthy but matches no speaker, which previously
    flipped EVERY opening utterance to client) and hallucinated labels alike.
    """
    try:
        spk_by_i = {int(t["i"]): str(t["spk"]).strip()
                    for t in out.get("turns", [])}
    except (KeyError, TypeError, ValueError):
        return None
    used = set(spk_by_i.values())
    if len(used) < 2:
        return None   # anti-collapse: diarization found two voices, LLM must too
    agent_raw = out.get("agent")
    agent = agent_raw.strip() if isinstance(agent_raw, str) else ""
    if agent not in used:
        return None   # null / empty / hallucinated agent label
    return spk_by_i, agent


def apply(utterances: List[Dict[str, Any]],
          turns: List[Dict[str, Any]]) -> Optional[Tuple[str, str]]:
    """Relabel the opening and return (closer_speaker, client_speaker) as
    diarization labels, or None if the relabel was not applied (caller should
    then keep the existing first-speaker rule). Mutates ``utterances`` in place:
    individually-mislabelled opening utterances get their ``speaker`` rewritten
    to the correct diarization label.
    """
    present: List[str] = []
    for u in utterances:
        s = u.get("speaker")
        if s and s not in present:
            present.append(s)
    if len(present) < 2:
        return None   # need two diarization labels to map two roles

    window = config.OPENING_RELABEL_WINDOW_SEC
    opening = [u for u in utterances
               if float(u.get("start", 0.0)) <= window and u.get("text")]
    if len(opening) < 2:
        return None

    lines = "\n".join(f'{i}: "{u.get("text", "")}"' for i, u in enumerate(opening))

    # Attempt policy: a TRANSPORT failure (Ollama 500 under load) is transient,
    # so the same model gets one retry — a single hiccup must not silently drop
    # the whole correction (that left short client turns inside the closer
    # cluster, Jhonatan 2026-06-10). A VALIDATION rejection (collapse / null
    # agent) at temperature 0 is effectively deterministic — re-asking the same
    # model just re-buys the same answer, so we move straight to the fallback
    # model instead (saves ~20 s on non-sales audio, where rejecting is the
    # CORRECT outcome).
    primary = config.opening_relabel_model()
    fallback = config.llm_model_id()
    models = [primary] + ([fallback] if fallback != primary else [])

    parsed: Optional[Tuple[Dict[int, str], str]] = None
    model = primary
    dt = 0.0
    for model in models:
        for attempt in (1, 2):           # 2nd attempt only after transport error
            try:
                t0 = time.time()
                out = _call_ollama(lines, model)
                dt = time.time() - t0
            except Exception as exc:  # noqa: BLE001 — transient; retry once
                logger.warning("opening_relabel model=%s attempt=%d transport "
                               "error err=%r", model, attempt, exc)
                time.sleep(2.0)
                continue
            parsed = _parse_and_validate(out)
            if parsed is None:
                logger.info("opening_relabel model=%s rejected (collapse or "
                            "invalid agent); trying next model", model)
            break                        # valid → done; rejected → next model
        if parsed is not None:
            break
    if parsed is None:
        logger.warning("opening_relabel no valid reply from %s; kept baseline",
                       models)
        return None
    spk_by_i, agent = parsed

    # Per-opening-utterance role from the LLM.
    role_by_i: Dict[int, str] = {}
    for i in range(len(opening)):
        spk = spk_by_i.get(i)
        if spk is not None:
            role_by_i[i] = "closer" if spk == agent else "client"
    if not role_by_i:
        return None

    # Decide which diarization label is closer vs client by net vote over the
    # opening (each label's closer-votes minus client-votes).
    net: Dict[str, int] = {lbl: 0 for lbl in present}
    for i, u in enumerate(opening):
        lbl = u.get("speaker")
        if lbl in net and i in role_by_i:
            net[lbl] += 1 if role_by_i[i] == "closer" else -1
    ranked = sorted(present, key=lambda l: net[l])
    client_speaker, closer_speaker = ranked[0], ranked[-1]
    if closer_speaker == client_speaker:
        return None

    # Rewrite individually-mislabelled opening utterances to the role-correct
    # diarization label. Non-opening utterances keep their (reliable) labels.
    rewrites = 0
    for i, u in enumerate(opening):
        role = role_by_i.get(i)
        if role is None:
            continue
        target = closer_speaker if role == "closer" else client_speaker
        if u.get("speaker") != target:
            u["speaker"] = target
            rewrites += 1

    logger.info("opening_relabel applied model=%s window=%.0fs utts=%d "
                "rewrites=%d closer=%s client=%s agent=%s %.1fs",
                model, window, len(opening), rewrites,
                closer_speaker, client_speaker, agent, dt)
    return closer_speaker, client_speaker

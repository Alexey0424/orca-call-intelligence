"""End-to-end scorecard assembly.

    audio  →  ASR (Whisper-base, English-only)
           →  diarize
           →  reconcile (words → utterances)
           →  text sentiment (XLM-R)
           →  speech emotion (emotion2vec+, VAD)
           →  prosody (eGeMAPS)
           →  text metrics (regex bilingual counts)
           →  LLM KPIs (Qwen)
           →  assemble

The heavy steps (emotion, KPI LLM, prosody) run SERIALLY — one model on the
GPU at a time — so the VRAM peak stays bounded and each step gets a clean,
uncontended resource reading in the control panel.

Returned dict matches what the renderer + n8n workflow expect.
"""
from __future__ import annotations

import gc
import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

import config
from models import registry
from pipeline import (audio_quality, backchannel, coaching, emotion, jobs, kpis,
                      opening_relabel, prosody, question_type, resource_sampler,
                      sentiment, text_metrics, timeline)

logger = logging.getLogger(__name__)


# WSL2 OOM mitigation. Heavy stages (Whisper, pyannote, emotion2vec, Qwen
# embedding) each pin sizeable intermediate buffers on host RAM + GPU VRAM
# during inference. Without explicit reclamation between stages the buffers
# pile up and trip the WSL2 host OOM (signature: silent SIGKILL, exit 0,
# OOMKilled=false). Cheap to call (~50 ms) — worth it.
def _relax_memory(stage: str) -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()
        logger.debug("memory_relaxed after=%s", stage)
    except Exception as exc:  # noqa: BLE001 — never block a call over GC
        logger.warning("memory_relax_failed stage=%s err=%r", stage, exc)


# ── Helpers ──────────────────────────────────────────────────────────────────
def _identify_client(speakers: List[str], closer_speaker: str) -> Optional[str]:
    return next((s for s in speakers if s != closer_speaker), None)


def _build_events(llm_kpis: Dict[str, Any]) -> Dict[str, Any]:
    """Lift event lists + next-step from the LLM JSON into a flat block.

    Renderer and n8n both read from ``scorecard.events``.
    """
    return {
        "objection_events":     llm_kpis.get("objection_events", []) or [],
        "buying_signal_events": llm_kpis.get("buying_signal_events", []) or [],
        "next_step_secured":    bool(llm_kpis.get("next_step_secured", False)),
        "next_step_text":       llm_kpis.get("next_step_text"),
    }


def _flat_features(*, timeline_d: Dict[str, Any],
                   llm_kpis: Dict[str, Any],
                   text_closer: Dict[str, int], text_client: Dict[str, int],
                   sent_closer: Dict[str, Any], sent_client: Dict[str, Any],
                   emo_closer: Dict[str, Any],  emo_client: Dict[str, Any],
                   pro_closer: Dict[str, Any],  pro_client: Dict[str, Any],
                   ) -> Dict[str, Any]:
    """Denormalised flat dict for downstream ML / CSV consumers."""
    def pf(d: Dict[str, Any], k: str) -> Any:
        return d.get(k)

    return {
        "duration_sec":              timeline_d.get("duration_sec"),
        "silence_ratio":             timeline_d.get("silence_ratio"),
        "overlap_ratio":             timeline_d.get("overlap_ratio"),
        "interruptions":             timeline_d.get("interruptions"),

        "closer_talk_ratio":         llm_kpis.get("talk_ratio_closer"),
        "closer_wpm":                llm_kpis.get("closer_wpm"),
        "client_wpm":                llm_kpis.get("client_wpm"),
        "objection_count":           llm_kpis.get("objection_count"),
        "buying_signal_count":       llm_kpis.get("buying_signal_count"),
        "next_step_secured":         1 if llm_kpis.get("next_step_secured") else 0,
        "rapport_quality":           llm_kpis.get("rapport_quality"),
        "predicted_outcome":         llm_kpis.get("predicted_outcome"),

        "closer_word_count":         text_closer.get("word_count"),
        "closer_question_count":     text_closer.get("question_count"),
        "closer_filler_count":       text_closer.get("filler_count"),
        "closer_rapport_markers":    text_closer.get("rapport_markers"),
        "closer_politeness_markers": text_closer.get("politeness_markers"),
        "client_word_count":         text_client.get("word_count"),
        "client_question_count":     text_client.get("question_count"),
        "client_buying_signals":     text_client.get("buying_signals"),

        "closer_sentiment_mean":         sent_closer.get("sentiment_mean"),
        "closer_sentiment_std":          sent_closer.get("sentiment_std"),
        "closer_sentiment_trajectory":   sent_closer.get("sentiment_trajectory"),
        "client_sentiment_mean":         sent_client.get("sentiment_mean"),
        "client_sentiment_trajectory":   sent_client.get("sentiment_trajectory"),

        "closer_dominant_emotion":   emo_closer.get("dominant_emotion"),
        "closer_valence_mean":       emo_closer.get("valence_mean"),
        "closer_arousal_mean":       emo_closer.get("arousal_mean"),
        "closer_dominance_mean":     emo_closer.get("dominance_mean"),
        "closer_emotion_entropy":    emo_closer.get("emotion_entropy"),
        "closer_arousal_trend":      emo_closer.get("arousal_trend"),
        "client_dominant_emotion":   emo_client.get("dominant_emotion"),
        "client_valence_mean":       emo_client.get("valence_mean"),
        "client_arousal_mean":       emo_client.get("arousal_mean"),
        "client_dominance_mean":     emo_client.get("dominance_mean"),
        "client_emotion_entropy":    emo_client.get("emotion_entropy"),
        "client_arousal_trend":      emo_client.get("arousal_trend"),

        "closer_loudness_mean":      pf(pro_closer, "loudness_mean"),
        "closer_pitch_hz_mean":      pf(pro_closer, "pitch_hz_mean"),
        "closer_jitter_mean":        pf(pro_closer, "jitter_mean"),
        "closer_shimmer_mean":       pf(pro_closer, "shimmer_mean"),
        "client_loudness_mean":      pf(pro_client, "loudness_mean"),
        "client_pitch_hz_mean":      pf(pro_client, "pitch_hz_mean"),
    }


# ── Main entrypoint ──────────────────────────────────────────────────────────
def build(*, wav_path: str,
          y: np.ndarray, sr: int, duration_sec: float,
          utterances: List[Dict[str, Any]],
          turns: List[Dict[str, Any]],
          closer_speaker: str,
          include_prosody: bool,
          job_id: Optional[str] = None) -> Dict[str, Any]:
    """Run every stage and return the assembled scorecard.

    When ``job_id`` is provided, the major model calls (sentiment,
    emotion, KPI LLM, prosody) record their per-model durations into
    ``jobs.metadata.stage_models["scorecard"]`` so the control UI can
    show what each model contributed to the total stage time.
    """
    # Optional: LLM opening relabel (DiarizationLM-style). Reads the opening
    # transcript to fix pyannote's cold-start mislabel, rewriting mislabelled
    # opening utterances in place and returning the closer/client diarization
    # labels. Gated OFF by default; any failure returns None and we fall back to
    # the first-speaker rule below. Runs before `speakers` is computed so the
    # set reflects any rewrites.
    relabel_result = None
    if config.opening_relabel_enabled():
        try:
            with resource_sampler.track() as _tr:
                relabel_result = opening_relabel.apply(utterances, turns)
            if job_id:
                jobs.record_model(job_id, "scorecard", config.opening_relabel_model(),
                                  "opening_relabel", _tr.elapsed, peaks=_tr.peaks)
        except Exception as exc:  # noqa: BLE001 — never block a call over relabel
            logger.warning("opening_relabel error (kept baseline) err=%r", exc)

    speakers = sorted({u.get("speaker") for u in utterances if u.get("speaker")})

    # Speaker roles come straight from diarization (voice-ID was fully retired
    # 2026-06-10): the LLM opening relabel decides closer/client when it
    # applies; otherwise the speaker who talks FIRST is the client and the
    # other diarized speaker is the closer.
    closer_name: Optional[str] = None
    closer_confidence: Optional[float] = None

    if relabel_result is not None:
        # LLM decided the closer/client mapping (overrides the first-speaker rule).
        closer_speaker, client_speaker = relabel_result
    else:
        _ref_for_order = turns or utterances
        _first_speaker = (min(_ref_for_order, key=lambda t: t.get("start", 0.0)).get("speaker")
                          if _ref_for_order else None)
        if _first_speaker and _first_speaker in speakers:
            client_speaker = _first_speaker
            closer_speaker = _identify_client(speakers, client_speaker) or closer_speaker
        else:
            # No usable speaking order — fall back to the caller-supplied label.
            client_speaker = _identify_client(speakers, closer_speaker)

    timeline_d = timeline.compute(turns or utterances, duration_sec)

    # In serialized mode each audio model is evicted the moment its stage ends,
    # so only one big model is resident at a time and the gemma KPI call below
    # runs with no audio model in VRAM. In preload mode the models stay resident.
    _serialized = not config.PRELOAD_MODELS

    # ── Text sentiment (RoBERTa, GPU) — load, run, evict ─────────────────────
    sentiment_pipeline, sentiment_id2label = registry.sentiment()
    with resource_sampler.track() as _tr:
        sentiment.run(sentiment_pipeline, utterances, sentiment_id2label)
    if job_id:
        jobs.record_model(job_id, "scorecard", config.SENTIMENT_MODEL_ID,
                          "sentiment", _tr.elapsed, peaks=_tr.peaks)
    sentiment_pipeline = sentiment_id2label = None
    if _serialized:
        registry.evict("sentiment")
    sent_closer = sentiment.aggregate(utterances, closer_speaker)
    sent_client = (sentiment.aggregate(utterances, client_speaker)
                   if client_speaker else {"ok": False})
    _relax_memory("sentiment")

    # ── SERIAL: emotion → KPI LLM → prosody, one model on the GPU at a time.
    #    These used to run concurrently; serializing them keeps the VRAM peak
    #    bounded (emotion2vec is evicted/idle before gemma runs) and gives each
    #    step a clean, uncontended resource reading in the control panel.

    # Speech emotion (emotion2vec+, GPU) — load, run (in-place on utterances),
    # evict. After this, NO audio model is resident for the gemma KPI call.
    ser_model, _ser_labels = registry.ser()
    with resource_sampler.track() as _tr:
        emotion.run(ser_model, utterances, y, sr)
    if job_id:
        jobs.record_model(job_id, "scorecard", config.SER_MODEL_ID,
                          "emotion", _tr.elapsed, peaks=_tr.peaks)
    ser_model = _ser_labels = None
    if _serialized:
        registry.evict("ser")
    emo_closer = emotion.aggregate(utterances, closer_speaker)
    emo_client = (emotion.aggregate(utterances, client_speaker)
                  if client_speaker else {"ok": False})
    _relax_memory("emotion")

    # Deterministic CPU annotations — cheap, not shown as their own rows.
    # Backchannel + question-type annotate in place; the LLM gets an immutable
    # snapshot below so it never sees the annotated transcript.
    text_closer = text_metrics.for_speaker(utterances, closer_speaker)
    text_client = (text_metrics.for_speaker(utterances, client_speaker)
                   if client_speaker else {})
    bc_count = backchannel.annotate(utterances)
    question_type.annotate(utterances)
    q_closer = question_type.aggregate(utterances, closer_speaker)
    q_client = (question_type.aggregate(utterances, client_speaker)
                if client_speaker else {"open": 0, "closed": 0, "probing": 0})
    audio_q = audio_quality.measure(y, sr)

    # KPI LLM (gemma4 in the Ollama process). Reads only speaker/start/text, so
    # we hand it a snapshot (text/speaker/start are untouched by emotion above).
    _llm_utts = [{"speaker": u.get("speaker"), "start": u.get("start"),
                  "text": u.get("text")} for u in utterances]
    with resource_sampler.track() as _tr:
        llm_kpis = kpis.call_llm(_llm_utts, closer_speaker, client_speaker)
    if job_id:
        jobs.record_model(job_id, "scorecard", config.llm_model_id(),
                          "kpis_llm", _tr.elapsed, peaks=_tr.peaks)

    # Closer name from the transcript: closers self-identify in the opening
    # ("this is Elias with Acme Homes"), so the KPI LLM extracts it the
    # same way it extracts client_name. It feeds the dashboard leaderboard
    # and the scope=closer dashboard filter.
    closer_name = (str(llm_kpis.get("closer_name") or "").strip() or None)

    # Coaching rubric — runs back-to-back with the KPI call while gemma is
    # still resident, so it pays no model reload. (It previously ran
    # post-persist, by which point gemma had been unloaded per the strict-serial
    # rule, costing a ~10-40 s reload every call.) Strict-serial is preserved:
    # still exactly one model in VRAM; the unload below now covers both LLM
    # calls. Best-effort: any failure leaves coaching None → persistence leaves
    # the calls columns NULL and the dashboard shows "awaiting".
    coaching_result: Optional[Dict[str, Any]] = None
    try:
        _coach_sc = {"metadata": {"closer_name": closer_name,
                                  "client_name": llm_kpis.get("client_name"),
                                  "duration_sec": duration_sec},
                     "scorecard": {"kpis": llm_kpis,
                                   "language": config.ASR_LANGUAGE}}
        _coach_utts = [{"role": ("closer" if u.get("speaker") == closer_speaker
                                 else "client" if u.get("speaker") == client_speaker
                                 else u.get("speaker")),
                        "text": u.get("text")} for u in utterances]
        with resource_sampler.track() as _tr:
            coaching_result = coaching.score_call(_coach_sc, _coach_utts)
        if job_id:
            jobs.record_model(job_id, "scorecard", config.llm_model_id(),
                              "coaching", _tr.elapsed, peaks=_tr.peaks)
    except Exception as exc:  # noqa: BLE001 — never block a call over coaching
        logger.warning("coaching_score_unhandled (non-fatal) err=%r", exc)

    # Strict serial (user request 2026-06-01): unload gemma the moment its LLM
    # work (KPIs + coaching) finishes. keep_alive:0 tells Ollama to drop the
    # model immediately instead of honouring OLLAMA_KEEP_ALIVE. Best-effort;
    # never block the pipeline.
    if _serialized:
        try:
            import httpx as _httpx
            _httpx.post(f"{config.OLLAMA_HOST}/api/generate",
                        json={"model": config.llm_model_id(), "keep_alive": 0},
                        timeout=10)
            logger.info("kpis_llm_unloaded gemma keep_alive=0 (serialized)")
        except Exception as exc:  # noqa: BLE001
            logger.warning("kpis_llm_unload_failed (non-fatal) err=%r", exc)
    _relax_memory("kpis_llm")

    # Prosody (openSMILE eGeMAPS, CPU).
    if include_prosody:
        with resource_sampler.track() as _tr:
            ref = turns or utterances
            pro: Dict[str, Dict[str, Any]] = {
                "closer": prosody.speaker_features(y, sr, ref, closer_speaker),
            }
            if client_speaker:
                pro["client"] = prosody.speaker_features(y, sr, ref, client_speaker)
        if job_id:
            jobs.record_model(job_id, "scorecard", "opensmile-eGeMAPSv02",
                              "prosody", _tr.elapsed, peaks=_tr.peaks)
    else:
        pro = {}

    events = _build_events(llm_kpis)
    pro_closer = pro.get("closer", {})
    pro_client = pro.get("client", {})

    flat = _flat_features(
        timeline_d=timeline_d, llm_kpis=llm_kpis,
        text_closer=text_closer, text_client=text_client,
        sent_closer=sent_closer, sent_client=sent_client,
        emo_closer=emo_closer,   emo_client=emo_client,
        pro_closer=pro_closer,   pro_client=pro_client,
    )
    # Layer the new rule-based + audio-quality features into the flat dict
    # so any CSV consumer / dashboards see them next to the existing
    # features. Kept separate from _flat_features above so the original
    # signature stays small.
    flat.update({
        "backchannel_count":         bc_count,
        "closer_open_q_count":       q_closer.get("open", 0),
        "closer_closed_q_count":     q_closer.get("closed", 0),
        "closer_probing_q_count":    q_closer.get("probing", 0),
        "client_open_q_count":       q_client.get("open", 0),
        "client_closed_q_count":     q_client.get("closed", 0),
        "client_probing_q_count":    q_client.get("probing", 0),
        **audio_q,
    })

    client_name = llm_kpis.get("client_name") or None

    roles = {closer_speaker: "closer"}
    if client_speaker:
        roles[client_speaker] = "client"

    return {
        "duration_sec":      timeline_d.get("duration_sec"),
        "language":          config.ASR_LANGUAGE,
        "speakers":          speakers,
        "roles":             roles,
        "identity": {
            "closer_speaker":    closer_speaker,
            "closer_name":       closer_name,
            "closer_confidence": closer_confidence,
            "client_speaker":    client_speaker,
            "client_name":       client_name,
        },
        "timeline":      timeline_d,
        "kpis":          llm_kpis,
        "events":        events,
        "text_metrics":  {"closer": text_closer,
                          **({"client": text_client} if client_speaker else {})},
        "sentiment":     {"closer": sent_closer,
                          "client": sent_client,
                          "model":  config.SENTIMENT_MODEL_ID},
        "emotion":       {"closer": emo_closer,
                          "client": emo_client,
                          "model":  config.SER_MODEL_ID},
        "prosody":       {"closer": pro_closer,
                          **({"client": pro_client} if client_speaker else {})},
        "audio_quality": audio_q,
        "question_types": {"closer": q_closer,
                           **({"client": q_client} if client_speaker else {})},
        "backchannel_count": bc_count,
        # Precomputed while gemma was resident (see above); persistence only
        # writes it. None on failure → calls columns stay NULL ("awaiting").
        "coaching":      coaching_result,
        "flat_features": flat,
    }

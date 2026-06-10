"""Persist a finished scorecard into Postgres + pgvector.

Tables written (atomic, one transaction):
  calls        — one row (upserted on call_id)
  utterances   — one row per turn (always wiped + reinserted for this call)
  call_events  — LLM-extracted events (objections, buying signals, next-step)
                 + GLiNER-extracted entities

Coaching score runs **after** the main commit (best-effort): it can't
delay the user-facing /scorecard response shape from being durable.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

import config
import db
from pipeline import embeddings, jobs, ner, pii, resource_sampler

logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────
def _vec_literal(vec: Optional[List[float]]) -> Optional[str]:
    if vec is None:
        return None
    return "[" + ",".join(f"{v:.7f}" for v in vec) + "]"


def _talk_ratio_from_timeline(timeline: Dict[str, Any],
                              closer_speaker: Optional[str]
                              ) -> Optional[float]:
    """Deterministic closer-talk-ratio from diarization activity.

    Returns ``closer_active_sec / total_active_sec`` clamped to [0, 1], or
    ``None`` when diarization didn't yield a usable breakdown (in which
    case the caller falls back to the LLM's value). Arithmetic is far more
    reliable from diarization than from a 14B model — and being a
    deterministic function of an existing model output, it's also
    cleanly auditable.
    """
    per = (timeline or {}).get("speaker_active_sec") or {}
    if not per or not closer_speaker or closer_speaker not in per:
        return None
    total = sum(float(v) for v in per.values())
    if total <= 0:
        return None
    ratio = float(per[closer_speaker]) / total
    return max(0.0, min(1.0, round(ratio, 4)))


def _build_summary_text(*, call_id: str, scorecard: Dict[str, Any]) -> str:
    """Compact, embedding-friendly summary of the call."""
    kpis = scorecard.get("kpis", {}) or {}
    summary = kpis.get("summary") or ""

    parts = [summary] if summary else []
    parts.append(
        f"Call {call_id}. "
        f"Talk-ratio closer: {kpis.get('talk_ratio_closer')}. "
        f"Objections: {kpis.get('objection_count')}. "
        f"Buying signals: {kpis.get('buying_signal_count')}. "
        f"Next step: {kpis.get('next_step_secured')}. "
        f"Rapport: {kpis.get('rapport_quality')}. "
        f"Outcome: {kpis.get('predicted_outcome')}."
    )

    objection_events = (scorecard.get("events", {}) or {}).get("objection_events") or []
    if objection_events:
        parts.append("Objections quoted: " + " | ".join(
            (e.get("text") or "")[:200] for e in objection_events[:3]
        ))

    buying_events = (scorecard.get("events", {}) or {}).get("buying_signal_events") or []
    if buying_events:
        parts.append("Buying signals quoted: " + " | ".join(
            (e.get("text") or "")[:200] for e in buying_events[:3]
        ))

    return "\n".join(p for p in parts if p)


def _llm_events(scorecard: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten the LLM's objection/buying-signal lists into events rows."""
    out: List[Dict[str, Any]] = []
    raw_events = scorecard.get("events", {}) or {}
    for kind_key, kind in (("objection_events",     "objection"),
                           ("buying_signal_events", "buying_signal")):
        for ev in raw_events.get(kind_key, []) or []:
            text = (ev.get("text") or "").strip()
            if not text:
                continue
            out.append({
                "kind":         kind,
                "text":         text,
                "normalized":   text.lower(),
                "confidence":   None,
                "start_sec":    float(ev.get("start_sec", 0.0)),
                "speaker_role": "client",
                "metadata":     None,
            })

    # Next-step is special: a single event, attributed to the closer.
    if raw_events.get("next_step_secured") and raw_events.get("next_step_text"):
        out.append({
            "kind":         "next_step",
            "text":         raw_events["next_step_text"],
            "normalized":   raw_events["next_step_text"].lower(),
            "confidence":   None,
            "start_sec":    None,
            "speaker_role": "closer",
            "metadata":     None,
        })
    return out


# ── Main entrypoint ──────────────────────────────────────────────────────────
def persist(*, call_id: str, file_name: Optional[str],
            scorecard: Dict[str, Any], utterances: List[Dict[str, Any]],
            file_hash: Optional[str] = None,
            pdf_url: Optional[str] = None, html_url: Optional[str] = None,
            index_utterances: bool = False,
            job_id: Optional[str] = None) -> Dict[str, Any]:
    """Upsert ``calls`` row + wipe-and-replace utterances + events.

    Returns ``{ok: True, utterances_indexed: N, events_indexed: M}`` on
    success, or ``{ok: False, error: "..."}`` on failure.

    When ``job_id`` is supplied, the major model calls in this stage
    (embeddings + NER) record their per-model durations into
    ``jobs.metadata.stage_models["persist"]`` for the control UI.
    """
    timeline   = scorecard.get("timeline", {})  or {}
    kpis       = scorecard.get("kpis", {})      or {}
    sentiment  = scorecard.get("sentiment", {}) or {}
    emotion    = scorecard.get("emotion", {})   or {}
    identity   = scorecard.get("identity", {})  or {}
    closer_s   = sentiment.get("closer", {})    or {}
    client_s   = sentiment.get("client", {})    or {}
    closer_e   = emotion.get("closer", {})      or {}
    client_e   = emotion.get("client", {})      or {}
    roles      = scorecard.get("roles", {})     or {}

    summary_text = _build_summary_text(call_id=call_id, scorecard=scorecard)
    with resource_sampler.track() as _tr:
        try:
            summary_vec = embeddings.embed_one(summary_text) if summary_text else None
        except Exception as exc:  # noqa: BLE001
            logger.error("embedding_failed call_id=%s error=%r", call_id, exc)
            return {"ok": False, "error": f"embedding: {exc}"}
    if job_id:
        jobs.record_model(job_id, "persist", config.embedding_model_id(),
                          "embedding", _tr.elapsed, peaks=_tr.peaks)

    # GLiNER entity extraction — non-fatal: a failure here just means we
    # persist the call with no NER events.
    with resource_sampler.track() as _tr:
        try:
            ner_events = ner.extract(utterances, roles)
        except Exception as exc:  # noqa: BLE001
            logger.warning("ner_failed call_id=%s error=%r", call_id, exc)
            ner_events = []
    if job_id:
        jobs.record_model(job_id, "persist", config.NER_MODEL_ID,
                          "ner", _tr.elapsed, peaks=_tr.peaks)

    all_events = _llm_events(scorecard) + ner_events

    # PII redaction (Presidio + spaCy) — adds text_redacted + pii_entities to
    # each utterance in-place and returns the per-call summary. SERIAL (runs
    # alone), so a clean per-step reading. Non-fatal: pii.annotate logs and
    # degrades gracefully if Presidio failed to load.
    with resource_sampler.track() as _tr:
        try:
            pii_summary = pii.annotate(utterances, language="en")
        except Exception as exc:  # noqa: BLE001
            logger.warning("pii_annotate_failed call_id=%s error=%r", call_id, exc)
            pii_summary = {}
    if job_id:
        jobs.record_model(job_id, "persist", "presidio/spacy-en_core_web_sm",
                          "pii", _tr.elapsed, peaks=_tr.peaks)

    # ── Single transaction ──────────────────────────────────────────────────
    # Time the whole DB transaction so the persist breakdown can show a
    # "db_write" row. The utterance-embedding time (recorded separately inside
    # _index_utterances) is subtracted out below so the two don't double-count.
    t_txn0 = time.monotonic()
    utt_embed_elapsed = 0.0
    try:
        with db.connect() as conn:
            with conn.cursor() as cur:
                # Prefer the deterministic talk-ratio (from diarization) over
                # the LLM's count. Falls back to the LLM only when diarization
                # didn't produce a per-speaker breakdown.
                closer_speaker = identity.get("closer_speaker")
                talk_ratio_det = _talk_ratio_from_timeline(timeline, closer_speaker)
                talk_ratio     = talk_ratio_det if talk_ratio_det is not None \
                                                else kpis.get("talk_ratio_closer")

                flat = scorecard.get("flat_features", {}) or {}
                cur.execute("""
                    INSERT INTO calls (
                        call_id, file_name, file_hash, language,
                        closer_name, closer_confidence, client_name,
                        duration_sec, silence_ratio, overlap_ratio, interruptions,
                        closer_talk_ratio, closer_wpm, client_wpm,
                        objection_count, buying_signal_count,
                        next_step_secured, next_step_text,
                        rapport_quality, predicted_outcome,
                        closer_sentiment_mean, closer_sentiment_trajectory,
                        client_sentiment_mean, client_sentiment_trajectory,
                        closer_dominant_emotion, client_dominant_emotion,
                        closer_valence_mean, closer_arousal_mean, closer_dominance_mean,
                        client_valence_mean, client_arousal_mean, client_dominance_mean,
                        summary, summary_embedding,
                        pdf_url, html_url,
                        audio_snr_db, audio_peak_dbfs, audio_clipping_rate,
                        audio_silence_front_sec, audio_silence_back_sec,
                        closer_open_q_count, closer_closed_q_count, closer_probing_q_count,
                        client_open_q_count, client_closed_q_count, client_probing_q_count,
                        backchannel_count,
                        pii_detected,
                        model_versions, scorecard
                    )
                    VALUES (
                        %s, %s, %s, %s,
                        %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s,
                        %s, %s,
                        %s, %s,
                        %s, %s,
                        %s, %s,
                        %s, %s,
                        %s, %s,
                        %s, %s, %s,
                        %s, %s, %s,
                        %s, %s::halfvec,
                        %s, %s,
                        %s, %s, %s,
                        %s, %s,
                        %s, %s, %s,
                        %s, %s, %s,
                        %s,
                        %s::jsonb,
                        %s::jsonb, %s::jsonb
                    )
                    ON CONFLICT (call_id) DO UPDATE SET
                        file_name                   = EXCLUDED.file_name,
                        file_hash                   = EXCLUDED.file_hash,
                        language                    = EXCLUDED.language,
                        closer_name                 = EXCLUDED.closer_name,
                        closer_confidence           = EXCLUDED.closer_confidence,
                        client_name                 = EXCLUDED.client_name,
                        duration_sec                = EXCLUDED.duration_sec,
                        silence_ratio               = EXCLUDED.silence_ratio,
                        overlap_ratio               = EXCLUDED.overlap_ratio,
                        interruptions               = EXCLUDED.interruptions,
                        closer_talk_ratio           = EXCLUDED.closer_talk_ratio,
                        closer_wpm                  = EXCLUDED.closer_wpm,
                        client_wpm                  = EXCLUDED.client_wpm,
                        objection_count             = EXCLUDED.objection_count,
                        buying_signal_count         = EXCLUDED.buying_signal_count,
                        next_step_secured           = EXCLUDED.next_step_secured,
                        next_step_text              = EXCLUDED.next_step_text,
                        rapport_quality             = EXCLUDED.rapport_quality,
                        predicted_outcome           = EXCLUDED.predicted_outcome,
                        closer_sentiment_mean       = EXCLUDED.closer_sentiment_mean,
                        closer_sentiment_trajectory = EXCLUDED.closer_sentiment_trajectory,
                        client_sentiment_mean       = EXCLUDED.client_sentiment_mean,
                        client_sentiment_trajectory = EXCLUDED.client_sentiment_trajectory,
                        closer_dominant_emotion     = EXCLUDED.closer_dominant_emotion,
                        client_dominant_emotion     = EXCLUDED.client_dominant_emotion,
                        closer_valence_mean         = EXCLUDED.closer_valence_mean,
                        closer_arousal_mean         = EXCLUDED.closer_arousal_mean,
                        closer_dominance_mean       = EXCLUDED.closer_dominance_mean,
                        client_valence_mean         = EXCLUDED.client_valence_mean,
                        client_arousal_mean         = EXCLUDED.client_arousal_mean,
                        client_dominance_mean       = EXCLUDED.client_dominance_mean,
                        summary                     = EXCLUDED.summary,
                        summary_embedding           = EXCLUDED.summary_embedding,
                        pdf_url                     = COALESCE(EXCLUDED.pdf_url,  calls.pdf_url),
                        html_url                    = COALESCE(EXCLUDED.html_url, calls.html_url),
                        audio_snr_db                = EXCLUDED.audio_snr_db,
                        audio_peak_dbfs             = EXCLUDED.audio_peak_dbfs,
                        audio_clipping_rate         = EXCLUDED.audio_clipping_rate,
                        audio_silence_front_sec     = EXCLUDED.audio_silence_front_sec,
                        audio_silence_back_sec      = EXCLUDED.audio_silence_back_sec,
                        closer_open_q_count         = EXCLUDED.closer_open_q_count,
                        closer_closed_q_count       = EXCLUDED.closer_closed_q_count,
                        closer_probing_q_count      = EXCLUDED.closer_probing_q_count,
                        client_open_q_count         = EXCLUDED.client_open_q_count,
                        client_closed_q_count       = EXCLUDED.client_closed_q_count,
                        client_probing_q_count      = EXCLUDED.client_probing_q_count,
                        backchannel_count           = EXCLUDED.backchannel_count,
                        pii_detected                = EXCLUDED.pii_detected,
                        model_versions              = EXCLUDED.model_versions,
                        scorecard                   = EXCLUDED.scorecard,
                        -- Ground-truth fields are populated by an external
                        -- CRM-pull job, never by /scorecard. Re-scoring
                        -- the audio must NOT overwrite them, so we don't
                        -- list them here at all.
                        updated_at                  = NOW()
                """, (
                    call_id, file_name, file_hash, scorecard.get("language"),
                    identity.get("closer_name"), identity.get("closer_confidence"),
                    identity.get("client_name"),
                    timeline.get("duration_sec"),
                    timeline.get("silence_ratio"), timeline.get("overlap_ratio"),
                    timeline.get("interruptions"),
                    talk_ratio,
                    kpis.get("closer_wpm"), kpis.get("client_wpm"),
                    kpis.get("objection_count"), kpis.get("buying_signal_count"),
                    bool(kpis.get("next_step_secured")),
                    kpis.get("next_step_text"),
                    kpis.get("rapport_quality"),
                    kpis.get("predicted_outcome"),
                    closer_s.get("sentiment_mean"), closer_s.get("sentiment_trajectory"),
                    client_s.get("sentiment_mean"), client_s.get("sentiment_trajectory"),
                    closer_e.get("dominant_emotion"), client_e.get("dominant_emotion"),
                    closer_e.get("valence_mean"), closer_e.get("arousal_mean"),
                    closer_e.get("dominance_mean"),
                    client_e.get("valence_mean"), client_e.get("arousal_mean"),
                    client_e.get("dominance_mean"),
                    summary_text, _vec_literal(summary_vec),
                    pdf_url, html_url,
                    flat.get("audio_snr_db"),
                    flat.get("audio_peak_dbfs"),
                    flat.get("audio_clipping_rate"),
                    flat.get("audio_silence_front_sec"),
                    flat.get("audio_silence_back_sec"),
                    flat.get("closer_open_q_count"),
                    flat.get("closer_closed_q_count"),
                    flat.get("closer_probing_q_count"),
                    flat.get("client_open_q_count"),
                    flat.get("client_closed_q_count"),
                    flat.get("client_probing_q_count"),
                    flat.get("backchannel_count"),
                    json.dumps(pii_summary),
                    json.dumps(config.model_versions()),
                    json.dumps(scorecard, ensure_ascii=False),
                ))

                utterances_indexed = 0
                if index_utterances and utterances:
                    utterances_indexed, utt_embed_elapsed = _index_utterances(
                        cur, call_id=call_id, utterances=utterances,
                        roles=roles, job_id=job_id,
                    )

                events_indexed = _index_events(cur, call_id=call_id, events=all_events)

            conn.commit()
    except Exception as exc:  # noqa: BLE001
        logger.error("persist_failed call_id=%s error=%r", call_id, exc)
        return {"ok": False, "error": f"db: {exc}"}

    logger.info("persisted call_id=%s utterances=%d events=%d",
                call_id, utterances_indexed, events_indexed)

    # Record the un-itemized DB-write time (whole transaction minus the
    # utterance embedding already recorded above) so the persist sub-steps add
    # up to the stage total in the control panel.
    if job_id:
        db_write_elapsed = max(0.0, (time.monotonic() - t_txn0) - utt_embed_elapsed)
        jobs.record_model(job_id, "persist", "postgres",
                          "db_write", db_write_elapsed)

    # Strict serial + VRAM budget: free the GPU embedder (Ollama) and GLiNER's
    # VRAM at the end of persist so the card returns to idle-free. (Coaching no
    # longer reloads gemma here — it is precomputed in scorecard.build while
    # gemma is still resident from the KPI call.) Best-effort; never block.
    try:
        ner.release()
    except Exception as exc:  # noqa: BLE001
        logger.warning("ner_release_failed err=%r", exc)
    try:
        import httpx as _httpx
        _httpx.post(f"{config.OLLAMA_HOST}/api/embeddings",
                    json={"model": config.embedding_model_id(),
                          "prompt": "x", "keep_alive": 0}, timeout=10)
    except Exception as exc:  # noqa: BLE001
        logger.warning("embedder_unload_failed err=%r", exc)

    # ── Coaching score (precomputed in scorecard.build — write-only here) ────
    sc_block = scorecard.get("scorecard", scorecard) or {}
    coach: Optional[Dict[str, Any]] = sc_block.get("coaching")

    if coach:
        try:
            with db.connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE calls
                           SET coaching_score     = %s,
                               coaching_breakdown = %s::jsonb,
                               coaching_notes     = %s
                         WHERE call_id = %s
                        """,
                        (coach["overall_score"],
                         json.dumps(coach["breakdown"]),
                         coach["notes"],
                         call_id),
                    )
                conn.commit()
            logger.info("coaching_score_persisted call_id=%s overall=%.1f",
                        call_id, coach["overall_score"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("coaching_score_persist_failed call_id=%s err=%r",
                           call_id, exc)

    return {
        "ok": True,
        "utterances_indexed": utterances_indexed,
        "events_indexed":     events_indexed,
        "coaching_score":     (coach or {}).get("overall_score"),
    }


def _index_utterances(cur, *, call_id: str,
                      utterances: List[Dict[str, Any]],
                      roles: Dict[str, str],
                      job_id: Optional[str] = None) -> tuple[int, float]:
    """Wipe and reinsert utterance rows.

    Embeds in batches via Ollama /api/embed (CF-15). On batch failure we
    automatically fall back to per-row embedding so one bad utterance
    doesn't drop the rest. Returns ``(inserted_count, embed_elapsed_sec)`` and
    records the embed time as the ``utterance_embedding`` persist sub-step so
    the control-panel breakdown accounts for it — a real cost now that the
    embedder runs on CPU (see config.EMBEDDING_NUM_GPU).
    """
    cur.execute("DELETE FROM utterances WHERE call_id = %s", (call_id,))

    texts = [(u.get("text") or "").strip() for u in utterances]
    with resource_sampler.track() as _tr:
        vecs = embeddings.embed_batch_safe(texts)
    if job_id:
        jobs.record_model(job_id, "persist", config.embedding_model_id(),
                          "utterance_embedding", _tr.elapsed, peaks=_tr.peaks)

    inserted = 0
    skipped  = 0
    for i, (u, text, vec) in enumerate(zip(utterances, texts, vecs)):
        if not text:
            continue
        if vec is None:
            skipped += 1
            continue

        speaker = u.get("speaker", "")
        role    = roles.get(speaker, "other")
        cur.execute("""
            INSERT INTO utterances (
                call_id, idx, start_sec, end_sec, speaker, role,
                text, text_redacted, pii_entities,
                asr_confidence, is_backchannel, question_types,
                sentiment_label, sentiment_score,
                emotion_label, emotion_probs,
                valence, arousal, dominance,
                text_embedding
            )
            VALUES (%s, %s, %s, %s, %s, %s,
                    %s, %s, %s::jsonb,
                    %s, %s, %s::jsonb,
                    %s, %s,
                    %s, %s::jsonb,
                    %s, %s, %s,
                    %s::halfvec)
        """, (
            call_id, i,
            u.get("start"), u.get("end"),
            speaker, role,
            text,
            u.get("text_redacted") or text,
            json.dumps(u.get("pii_entities") or {}),
            u.get("asr_confidence"),
            bool(u.get("is_backchannel", False)),
            json.dumps(u.get("question_types") or {}),
            u.get("sentiment_label"), u.get("sentiment_score"),
            u.get("emotion_label"),
            json.dumps(u.get("emotion_probs") or {}),
            u.get("valence"), u.get("arousal"), u.get("dominance"),
            _vec_literal(vec),
        ))
        inserted += 1

    if skipped:
        logger.warning("utterances_index_skipped call_id=%s skipped=%d inserted=%d",
                       call_id, skipped, inserted)
    return inserted, _tr.elapsed


def _index_events(cur, *, call_id: str,
                  events: List[Dict[str, Any]]) -> int:
    """Wipe and reinsert call_events for this call."""
    cur.execute("DELETE FROM call_events WHERE call_id = %s", (call_id,))
    if not events:
        return 0

    for ev in events:
        cur.execute("""
            INSERT INTO call_events (
                call_id, kind, text, normalized, confidence,
                start_sec, speaker_role, metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)
        """, (
            call_id, ev["kind"], ev["text"], ev.get("normalized"),
            ev.get("confidence"), ev.get("start_sec"), ev.get("speaker_role"),
            json.dumps(ev["metadata"]) if ev.get("metadata") is not None else None,
        ))
    return len(events)


# ── Search ────────────────────────────────────────────────────────────────────
def search_calls(*, query: str, top_k: int = 5,
                 min_similarity: float = 0.0) -> List[Dict[str, Any]]:
    """Top-k calls by cosine similarity of summary_embedding to ``query``."""
    if not query.strip():
        return []
    qvec_literal = _vec_literal(embeddings.embed_one(query))

    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    call_id, file_name, received_at, duration_sec,
                    closer_name, closer_confidence, client_name, client_id,
                    closer_talk_ratio, closer_wpm, client_wpm,
                    objection_count, buying_signal_count, next_step_secured,
                    rapport_quality, predicted_outcome,
                    summary, pdf_url, html_url,
                    1 - (summary_embedding <=> %s::halfvec) AS similarity
                FROM calls
                WHERE summary_embedding IS NOT NULL
                  AND 1 - (summary_embedding <=> %s::halfvec) >= %s
                ORDER BY summary_embedding <=> %s::halfvec
                LIMIT %s
            """, (qvec_literal, qvec_literal, min_similarity, qvec_literal, top_k))
            cols = [c[0] for c in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]


def search_utterances(*, query: str, top_k: int = 10,
                      min_similarity: float = 0.0,
                      role: Optional[str] = None) -> List[Dict[str, Any]]:
    """Top-k utterances by cosine similarity to ``query``. Optional role filter."""
    if not query.strip():
        return []
    qvec_literal = _vec_literal(embeddings.embed_one(query))

    sql = """
        SELECT
            call_id, idx, start_sec, end_sec, speaker, role,
            text, sentiment_label, emotion_label, valence, arousal, dominance,
            1 - (text_embedding <=> %s::halfvec) AS similarity
        FROM utterances
        WHERE text_embedding IS NOT NULL
          AND 1 - (text_embedding <=> %s::halfvec) >= %s
    """
    params: list = [qvec_literal, qvec_literal, min_similarity]
    if role:
        sql += " AND role = %s "
        params.append(role)
    sql += " ORDER BY text_embedding <=> %s::halfvec LIMIT %s "
    params.extend([qvec_literal, top_k])

    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            cols = [c[0] for c in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

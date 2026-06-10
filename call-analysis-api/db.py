"""Postgres schema bootstrap + connection helper.

Lives in the dedicated ``orca`` database (``orca`` for our app, ``n8n``
for n8n itself). Only four tables — small, semantic, no third-party noise.

  calls         — one row per processed call. Wide-and-flat: every flat
                  KPI, dimensional affect (valence/arousal/dominance),
                  embedding, model lineage, and the full pipeline payload
                  live here so a single SELECT serves dashboards + replay.
  utterances    — one row per turn taken. Sentiment + emotion + dimensional
                  affect at utterance granularity, plus an embedding for
                  "find the moment where X was said" retrieval.
  call_events   — every timestamped finding in a call: objections, buying
                  signals, next-step, plus GLiNER-extracted entities
                  (competitors, neighborhoods, mortgage brokers, price
                  points, …) in one shape.
  pipeline_runs — one row per /scorecard invocation; surfaces stuck or
                  failed work. ``run_no`` is the human-friendly sequence.

Plus the ``call_features_v1`` VIEW — the leak-free ML projection of calls,
excluding every LLM-prediction column.

Schema is created at startup via plain ``CREATE TABLE IF NOT EXISTS``;
Alembic was retired during the rebuild — five revisions for an
about-to-be-rewritten schema were just dead weight against a fresh DB.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import psycopg2

import config
from ragcore.db import connect  # noqa: F401 — re-exported for existing call sites

logger = logging.getLogger(__name__)


# ── DDL ──────────────────────────────────────────────────────────────────────
def _ddl() -> str:
    dim = config.EMBEDDING_DIM
    return f"""
    CREATE EXTENSION IF NOT EXISTS vector;

    -- (The voice-linked `clients` table + calls.client_id/client_voice_* were
    --  removed 2026-06-10 — per-client voice identity is not used. Existing
    --  DBs are migrated by the DROP statements at the end of this DDL.)

    -- ========================================================================
    -- calls — one row per processed call
    -- ========================================================================
    CREATE TABLE IF NOT EXISTS calls (
        call_id              TEXT PRIMARY KEY,
        file_name            TEXT,
        file_hash            TEXT,
        received_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        language             TEXT,

        -- Speakers
        closer_name          TEXT,
        closer_confidence    REAL,
        client_name          TEXT,

        -- Conversation dynamics
        duration_sec         REAL,
        silence_ratio        REAL,
        overlap_ratio        REAL,
        interruptions        INT,
        closer_talk_ratio    REAL,
        closer_wpm           REAL,
        client_wpm           REAL,

        -- LLM-extracted KPIs
        objection_count      INT,
        buying_signal_count  INT,
        next_step_secured    BOOLEAN,
        next_step_text       TEXT,
        rapport_quality      TEXT,
        predicted_outcome    TEXT,

        -- Coaching rubric (LLM, six dimensions, weighted 0-100)
        coaching_score       REAL,
        coaching_breakdown   JSONB,
        coaching_notes       TEXT,

        -- Per-speaker text sentiment (RoBERTa)
        closer_sentiment_mean        REAL,
        closer_sentiment_trajectory  TEXT,
        client_sentiment_mean        REAL,
        client_sentiment_trajectory  TEXT,

        -- Per-speaker audio emotion (emotion2vec+)
        closer_dominant_emotion      TEXT,
        client_dominant_emotion      TEXT,
        -- Dimensional VAD (dashboards build on these, not the labels)
        closer_valence_mean          REAL,
        closer_arousal_mean          REAL,
        closer_dominance_mean        REAL,
        client_valence_mean          REAL,
        client_arousal_mean          REAL,
        client_dominance_mean        REAL,

        -- Search
        summary              TEXT,
        summary_embedding    halfvec({dim}),

        -- Output artefacts. Lazily populated by rag-api /report/{{call_id}} on first
        -- view (render-once-then-cache). NULL until a report is requested.
        pdf_url              TEXT,
        html_url             TEXT,

        -- Ground-truth outcome. Populated by an external CRM-pull job,
        -- NOT by /scorecard. UPSERT path COALESCEs these so a re-score
        -- never clobbers what the CRM wrote.
        actual_outcome       TEXT,
        deal_value           NUMERIC(12,2),
        outcome_confirmed_at TIMESTAMPTZ,
        outcome_source       TEXT,

        -- Audio quality (deterministic, single-pass over the converted WAV)
        audio_snr_db             REAL,
        audio_peak_dbfs          REAL,
        audio_clipping_rate      REAL,
        audio_silence_front_sec  REAL,
        audio_silence_back_sec   REAL,

        -- Question-type counts (rule-based, English)
        closer_open_q_count      INT,
        closer_closed_q_count    INT,
        closer_probing_q_count   INT,
        client_open_q_count      INT,
        client_closed_q_count    INT,
        client_probing_q_count   INT,

        -- Backchannel total ("mhm" / "yeah" interjections — excluded from
        -- wpm and question_count aggregates downstream).
        backchannel_count        INT,

        -- PII presence summary: {{entity_type: count}}. Raw transcripts on
        -- utterances.text; redacted variants on utterances.text_redacted.
        pii_detected             JSONB,

        -- Lineage + full payload for replay
        model_versions       JSONB,
        scorecard            JSONB,

        -- Bounded-value guards. Buggy pipeline writes are caught here
        -- before they corrupt dashboards or training sets.
        CONSTRAINT ck_calls_closer_talk_ratio CHECK (closer_talk_ratio    IS NULL OR closer_talk_ratio    BETWEEN 0.0 AND 1.0),
        CONSTRAINT ck_calls_silence_ratio     CHECK (silence_ratio        IS NULL OR silence_ratio        BETWEEN 0.0 AND 1.0),
        CONSTRAINT ck_calls_overlap_ratio     CHECK (overlap_ratio        IS NULL OR overlap_ratio        BETWEEN 0.0 AND 1.0),
        CONSTRAINT ck_calls_closer_confidence CHECK (closer_confidence    IS NULL OR closer_confidence    BETWEEN 0.0 AND 1.0),
        CONSTRAINT ck_calls_coaching_score    CHECK (coaching_score       IS NULL OR coaching_score       BETWEEN 0.0 AND 100.0),
        CONSTRAINT ck_calls_valence_closer    CHECK (closer_valence_mean   IS NULL OR closer_valence_mean   BETWEEN -1.0 AND 1.0),
        CONSTRAINT ck_calls_arousal_closer    CHECK (closer_arousal_mean   IS NULL OR closer_arousal_mean   BETWEEN -1.0 AND 1.0),
        CONSTRAINT ck_calls_dominance_closer  CHECK (closer_dominance_mean IS NULL OR closer_dominance_mean BETWEEN -1.0 AND 1.0),
        CONSTRAINT ck_calls_valence_client    CHECK (client_valence_mean   IS NULL OR client_valence_mean   BETWEEN -1.0 AND 1.0),
        CONSTRAINT ck_calls_arousal_client    CHECK (client_arousal_mean   IS NULL OR client_arousal_mean   BETWEEN -1.0 AND 1.0),
        CONSTRAINT ck_calls_dominance_client  CHECK (client_dominance_mean IS NULL OR client_dominance_mean BETWEEN -1.0 AND 1.0),
        CONSTRAINT ck_calls_rapport_quality   CHECK (rapport_quality   IS NULL OR rapport_quality   IN ('low','medium','high')),
        CONSTRAINT ck_calls_predicted_outcome CHECK (predicted_outcome IS NULL OR predicted_outcome IN ('likely_close','follow_up_needed','lost','unknown')),
        CONSTRAINT ck_calls_closer_traj       CHECK (closer_sentiment_trajectory IS NULL OR closer_sentiment_trajectory IN ('improving','stable','declining','unknown')),
        CONSTRAINT ck_calls_client_traj       CHECK (client_sentiment_trajectory IS NULL OR client_sentiment_trajectory IN ('improving','stable','declining','unknown')),
        CONSTRAINT ck_calls_actual_outcome    CHECK (actual_outcome IS NULL OR actual_outcome IN ('closed','lost','no_decision','pending')),
        CONSTRAINT ck_calls_outcome_source    CHECK (outcome_source IS NULL OR outcome_source IN ('crm','human_review','external','manual')),
        CONSTRAINT ck_calls_clipping_rate     CHECK (audio_clipping_rate IS NULL OR audio_clipping_rate BETWEEN 0.0 AND 1.0),
        CONSTRAINT ck_calls_silence_front     CHECK (audio_silence_front_sec IS NULL OR audio_silence_front_sec >= 0.0),
        CONSTRAINT ck_calls_silence_back      CHECK (audio_silence_back_sec  IS NULL OR audio_silence_back_sec  >= 0.0)
    );

    CREATE INDEX IF NOT EXISTS calls_received_at_idx     ON calls (received_at DESC);
    CREATE INDEX IF NOT EXISTS calls_closer_idx          ON calls (closer_name);
    CREATE INDEX IF NOT EXISTS calls_file_hash_idx       ON calls (file_hash);
    CREATE INDEX IF NOT EXISTS calls_updated_at_idx      ON calls (updated_at DESC);
    CREATE INDEX IF NOT EXISTS calls_actual_outcome_idx  ON calls (actual_outcome);
    CREATE INDEX IF NOT EXISTS calls_summary_hnsw
        ON calls USING hnsw (summary_embedding halfvec_cosine_ops);

    -- Annotate LLM-prediction vs. observation vs. label columns so ML code
    -- knows what's leaky. The call_features_v1 view below is the official
    -- leak-free projection — supervised models should JOIN against that.
    DO $$ BEGIN
        EXECUTE 'COMMENT ON COLUMN calls.predicted_outcome    IS ''LLM PREDICTION — do not use as feature for outcome models. See call_features_v1.''';
        EXECUTE 'COMMENT ON COLUMN calls.rapport_quality      IS ''LLM PREDICTION — same warning.''';
        EXECUTE 'COMMENT ON COLUMN calls.coaching_score       IS ''LLM PREDICTION (weighted rubric).''';
        EXECUTE 'COMMENT ON COLUMN calls.coaching_breakdown   IS ''LLM PREDICTION — six-dimension JSONB.''';
        EXECUTE 'COMMENT ON COLUMN calls.coaching_notes       IS ''LLM PREDICTION — short coaching note.''';
        EXECUTE 'COMMENT ON COLUMN calls.next_step_secured    IS ''LLM PREDICTION — derived from transcript by KPI prompt.''';
        EXECUTE 'COMMENT ON COLUMN calls.next_step_text       IS ''LLM PREDICTION.''';
        EXECUTE 'COMMENT ON COLUMN calls.summary              IS ''LLM PREDICTION — embedded into summary_embedding for search.''';
        EXECUTE 'COMMENT ON COLUMN calls.actual_outcome       IS ''GROUND-TRUTH LABEL (CRM/human). Populated outside the audio pipeline.''';
        EXECUTE 'COMMENT ON COLUMN calls.deal_value           IS ''GROUND-TRUTH LABEL.''';
        EXECUTE 'COMMENT ON COLUMN calls.outcome_confirmed_at IS ''GROUND-TRUTH LABEL — when actual_outcome was confirmed.''';
        EXECUTE 'COMMENT ON COLUMN calls.outcome_source       IS ''GROUND-TRUTH LABEL provenance.''';
    END $$;

    -- Safe-for-ML view: every observation + derived feature, EXCLUDING all
    -- LLM-prediction columns. This is the surface a supervised model should
    -- JOIN against actual_outcome / deal_value.
    DROP VIEW IF EXISTS call_features_v1;
    CREATE VIEW call_features_v1 AS
        SELECT
            call_id, file_name, file_hash, received_at, language,
            closer_name, closer_confidence, client_name,
            duration_sec, silence_ratio, overlap_ratio, interruptions,
            closer_talk_ratio, closer_wpm, client_wpm,
            objection_count, buying_signal_count,
            closer_sentiment_mean, closer_sentiment_trajectory,
            client_sentiment_mean, client_sentiment_trajectory,
            closer_dominant_emotion, client_dominant_emotion,
            closer_valence_mean, closer_arousal_mean, closer_dominance_mean,
            client_valence_mean, client_arousal_mean, client_dominance_mean,
            model_versions,
            actual_outcome, deal_value, outcome_confirmed_at, outcome_source
          FROM calls;
    COMMENT ON VIEW call_features_v1 IS
        'Safe-for-ML projection of calls: observations + derived features '
        '+ labels. Excludes every column commented as LLM PREDICTION on the '
        'base table.';

    -- ========================================================================
    -- utterances — one row per turn taken in a call
    -- ========================================================================
    CREATE TABLE IF NOT EXISTS utterances (
        id                BIGSERIAL PRIMARY KEY,
        call_id           TEXT NOT NULL REFERENCES calls(call_id) ON DELETE CASCADE,
        idx               INT  NOT NULL,
        start_sec         REAL,
        end_sec           REAL,
        speaker           TEXT,
        role              TEXT,
        text              TEXT,
        text_redacted     TEXT,
        pii_entities      JSONB,

        asr_confidence    REAL,
        is_backchannel    BOOLEAN NOT NULL DEFAULT FALSE,
        question_types    JSONB,

        sentiment_label   TEXT,
        sentiment_score   REAL,

        emotion_label     TEXT,
        emotion_probs     JSONB,
        valence           REAL,
        arousal           REAL,
        dominance         REAL,

        text_embedding    halfvec({dim}),
        created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        UNIQUE (call_id, idx),

        CONSTRAINT ck_utt_role            CHECK (role            IS NULL OR role            IN ('closer','client','other')),
        CONSTRAINT ck_utt_sentiment_label CHECK (sentiment_label IS NULL OR sentiment_label IN ('positive','neutral','negative')),
        CONSTRAINT ck_utt_valence         CHECK (valence   IS NULL OR valence   BETWEEN -1.0 AND 1.0),
        CONSTRAINT ck_utt_arousal         CHECK (arousal   IS NULL OR arousal   BETWEEN -1.0 AND 1.0),
        CONSTRAINT ck_utt_dominance       CHECK (dominance IS NULL OR dominance BETWEEN -1.0 AND 1.0),
        CONSTRAINT ck_utt_asr_confidence  CHECK (asr_confidence IS NULL OR asr_confidence BETWEEN 0.0 AND 1.0)
    );

    CREATE INDEX IF NOT EXISTS utterances_call_idx   ON utterances (call_id);
    CREATE INDEX IF NOT EXISTS utterances_role_idx   ON utterances (role);
    CREATE INDEX IF NOT EXISTS utterances_text_hnsw
        ON utterances USING hnsw (text_embedding halfvec_cosine_ops);

    -- ========================================================================
    -- call_events — every timestamped finding in a call (one table, one shape)
    -- ========================================================================
    -- Subsumes:
    --   • LLM-extracted events (objections, buying signals, next-step)
    --   • GLiNER-extracted entities (competitors, neighborhoods, brokers,
    --     price points, property types, …)
    CREATE TABLE IF NOT EXISTS call_events (
        id            BIGSERIAL PRIMARY KEY,
        call_id       TEXT NOT NULL REFERENCES calls(call_id) ON DELETE CASCADE,
        kind          TEXT NOT NULL,
        text          TEXT NOT NULL,
        normalized    TEXT,
        confidence    REAL,
        start_sec     REAL,
        speaker_role  TEXT,
        metadata      JSONB,
        created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),

        CONSTRAINT ck_call_events_kind         CHECK (kind IN ('objection','buying_signal','next_step',
                                                               'competitor','neighborhood','mortgage_broker',
                                                               'price_point','property_type','agent_name')),
        CONSTRAINT ck_call_events_speaker_role CHECK (speaker_role IS NULL OR speaker_role IN ('closer','client','other')),
        CONSTRAINT ck_call_events_confidence   CHECK (confidence IS NULL OR confidence BETWEEN 0.0 AND 1.0)
    );

    CREATE INDEX IF NOT EXISTS call_events_call_idx        ON call_events (call_id);
    CREATE INDEX IF NOT EXISTS call_events_kind_norm_idx   ON call_events (kind, normalized);
    CREATE INDEX IF NOT EXISTS call_events_kind_time_idx   ON call_events (kind, call_id, start_sec);

    -- ========================================================================
    -- pipeline_runs — one row per /scorecard invocation
    -- ========================================================================
    -- Lightweight tracker: status transitions running → done | failed.
    -- ``stage`` records the last stage that started (asr / diarize / kpis /
    -- persist / coaching / done) so the operator can spot pipeline hot spots.
    CREATE SEQUENCE IF NOT EXISTS pipeline_runs_run_no_seq;

    CREATE TABLE IF NOT EXISTS pipeline_runs (
        job_id        TEXT PRIMARY KEY,
        run_no        INTEGER UNIQUE NOT NULL DEFAULT nextval('pipeline_runs_run_no_seq'),
        call_id       TEXT,
        kind          TEXT NOT NULL,
        status        TEXT NOT NULL,
        stage         TEXT,
        attempts      INT  NOT NULL DEFAULT 1,
        started_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        finished_at   TIMESTAMPTZ,
        duration_sec  REAL,
        last_error    TEXT,
        metadata      JSONB,
        CONSTRAINT ck_pipeline_runs_status CHECK (status IN ('running','done','failed'))
    );
    ALTER SEQUENCE pipeline_runs_run_no_seq OWNED BY pipeline_runs.run_no;

    CREATE INDEX IF NOT EXISTS pipeline_runs_started_at_idx ON pipeline_runs (started_at DESC);
    CREATE INDEX IF NOT EXISTS pipeline_runs_status_idx     ON pipeline_runs (status, started_at DESC);
    CREATE INDEX IF NOT EXISTS pipeline_runs_call_id_idx    ON pipeline_runs (call_id);

    -- ── Migration: remove the voice-linked client identity (2026-06-10) ─────
    -- Safe on fresh DBs (IF EXISTS everywhere). call_features_v1 was already
    -- recreated above without client_id, so the column drops cleanly.
    ALTER TABLE calls DROP COLUMN IF EXISTS client_id;
    ALTER TABLE calls DROP COLUMN IF EXISTS client_voice_embedding;
    ALTER TABLE calls DROP COLUMN IF EXISTS client_match_confidence;
    ALTER TABLE calls DROP COLUMN IF EXISTS client_match_action;
    DROP TABLE IF EXISTS clients CASCADE;
    """


def ensure_schema() -> bool:
    """Create extensions, tables, indexes, and the call_features_v1 view.

    Retries cover the case where postgres hasn't finished starting yet.
    """
    last_err: Optional[Exception] = None
    for attempt in range(1, config.DB_INIT_RETRIES + 1):
        try:
            conn = psycopg2.connect(config.DATABASE_URL)
            conn.autocommit = True
            try:
                with conn.cursor() as cur:
                    cur.execute(_ddl())
            finally:
                conn.close()
            logger.info("db_schema ready attempt=%d embedding_dim=%d",
                        attempt, config.EMBEDDING_DIM)
            return True
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            logger.warning("db_schema init_failed attempt=%d error=%r",
                           attempt, exc)
            time.sleep(config.DB_INIT_BACKOFF_SEC)

    logger.error("db_schema giving_up retries=%d last_error=%r",
                 config.DB_INIT_RETRIES, last_err)
    return False


# connect() is re-exported from ragcore.db above.

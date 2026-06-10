-- ────────────────────────────────────────────────────────────────────────────
--  Drill into a single call.
--
--  Replace <CALL_ID> with the call_id you want to inspect (e.g. found from
--  01_recent_calls.sql).
-- ────────────────────────────────────────────────────────────────────────────

\set call_id '20260508-115207-9cb0d5dd'        -- ← edit me


-- ── 1. Headline KPIs for the call ──────────────────────────────────────────

SELECT call_id, file_name, received_at,
       closer_name, client_name, duration_sec,
       closer_talk_ratio, closer_wpm, client_wpm,
       silence_ratio, overlap_ratio, interruptions,
       objection_count, buying_signal_count, next_step_secured,
       rapport_quality, predicted_outcome, coaching_score,
       closer_dominant_emotion, closer_valence_mean,
       closer_arousal_mean, closer_dominance_mean,
       client_dominant_emotion, client_valence_mean,
       client_arousal_mean, client_dominance_mean,
       closer_sentiment_trajectory, client_sentiment_trajectory,
       summary, pdf_url, html_url
FROM calls
WHERE call_id = :'call_id';


-- ── 2. Every utterance, in order ───────────────────────────────────────────

SELECT idx,
       role,
       ROUND(start_sec)::INT AS at_sec,
       ROUND(end_sec - start_sec)::NUMERIC(5,2) AS dur,
       sentiment_label,
       emotion_label,
       valence, arousal, dominance,
       text
FROM utterances
WHERE call_id = :'call_id'
ORDER BY idx;


-- ── 3. Every event in the call (objections + buying signals + entities) ───

SELECT kind, normalized, text, confidence,
       ROUND(start_sec)::INT AS at_sec, speaker_role
FROM call_events
WHERE call_id = :'call_id'
ORDER BY kind, start_sec;


-- ── 4. Coaching breakdown for the call ─────────────────────────────────────

SELECT call_id, coaching_score,
       jsonb_pretty(coaching_breakdown) AS breakdown,
       coaching_notes
FROM calls
WHERE call_id = :'call_id';


-- ── 5. Model lineage — which models produced this row? ─────────────────────

SELECT call_id, jsonb_pretty(model_versions) AS model_versions
FROM calls
WHERE call_id = :'call_id';


-- ── 6. The full raw payload, for replay / future features ──────────────────

SELECT call_id, jsonb_pretty(scorecard) AS scorecard
FROM calls
WHERE call_id = :'call_id';

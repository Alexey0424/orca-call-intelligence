-- ────────────────────────────────────────────────────────────────────────────
--  Cross-call aggregates — useful for dashboards / week-in-review style reports.
--  Pick one block at a time.
-- ────────────────────────────────────────────────────────────────────────────


-- ── 1. Top-line counters across the corpus ─────────────────────────────────

SELECT
    COUNT(*)                                                AS total_calls,
    SUM(CASE WHEN next_step_secured  THEN 1 ELSE 0 END)     AS calls_with_next_step,
    SUM(CASE WHEN buying_signal_count > 0 THEN 1 ELSE 0 END) AS calls_with_buying_signal,
    SUM(CASE WHEN predicted_outcome = 'likely_close' THEN 1 ELSE 0 END) AS likely_close,
    SUM(CASE WHEN predicted_outcome = 'lost'         THEN 1 ELSE 0 END) AS lost,
    ROUND(AVG(closer_talk_ratio)::NUMERIC, 2)               AS avg_closer_talk,
    ROUND(AVG(closer_wpm)::NUMERIC, 1)                      AS avg_closer_wpm,
    ROUND(AVG(client_wpm)::NUMERIC, 1)                      AS avg_client_wpm,
    ROUND(AVG(duration_sec)::NUMERIC, 0)                    AS avg_duration_sec,
    ROUND(AVG(coaching_score)::NUMERIC, 1)                  AS avg_coaching_score
FROM calls;


-- ── 2. Distribution by predicted outcome ───────────────────────────────────

-- SELECT predicted_outcome,
--        COUNT(*)                                  AS calls,
--        ROUND(AVG(closer_talk_ratio)::NUMERIC,2)  AS avg_closer_talk,
--        ROUND(AVG(buying_signal_count)::NUMERIC,1) AS avg_buying_signals,
--        ROUND(AVG(objection_count)::NUMERIC,1)    AS avg_objections
-- FROM calls
-- GROUP BY predicted_outcome
-- ORDER BY calls DESC;


-- ── 3. Closer dimensional-affect frequency ─────────────────────────────────

-- SELECT closer_dominant_emotion,
--        COUNT(*)                            AS calls,
--        ROUND(AVG(closer_valence_mean)::NUMERIC, 3) AS avg_valence,
--        ROUND(AVG(closer_arousal_mean)::NUMERIC, 3) AS avg_arousal,
--        ROUND(AVG(closer_dominance_mean)::NUMERIC, 3) AS avg_dominance,
--        SUM(CASE WHEN next_step_secured THEN 1 ELSE 0 END) AS with_next_step
-- FROM calls
-- GROUP BY closer_dominant_emotion
-- ORDER BY calls DESC;


-- ── 4. Calls per day (last 30 days) ────────────────────────────────────────

-- SELECT (received_at AT TIME ZONE 'UTC')::DATE AS day,
--        COUNT(*)                              AS calls,
--        SUM(CASE WHEN next_step_secured THEN 1 ELSE 0 END) AS with_next_step,
--        SUM(CASE WHEN buying_signal_count > 0 THEN 1 ELSE 0 END) AS with_buying
-- FROM calls
-- WHERE received_at >= now() - INTERVAL '30 days'
-- GROUP BY day
-- ORDER BY day DESC;


-- ── 5. Utterance-level averages per role ───────────────────────────────────
-- Useful for comparing how closer vs client typically behave.

-- SELECT role,
--        COUNT(*)                                  AS utterances,
--        ROUND(AVG(end_sec - start_sec)::NUMERIC, 2) AS avg_utt_sec,
--        ROUND(AVG(valence)::NUMERIC, 3)           AS avg_valence,
--        ROUND(AVG(arousal)::NUMERIC, 3)           AS avg_arousal,
--        SUM(CASE WHEN sentiment_label = 'positive' THEN 1 ELSE 0 END) AS pos,
--        SUM(CASE WHEN sentiment_label = 'neutral'  THEN 1 ELSE 0 END) AS neu,
--        SUM(CASE WHEN sentiment_label = 'negative' THEN 1 ELSE 0 END) AS neg
-- FROM utterances
-- WHERE role IN ('closer', 'client')
-- GROUP BY role;


-- ── 6. Top-mentioned entities (competitors / neighbourhoods / brokers) ─────
-- The `call_events` table fed by GLiNER lets you slice by mention type.

-- SELECT kind, normalized, COUNT(*) AS mentions
-- FROM call_events
-- WHERE kind IN ('competitor', 'neighborhood', 'mortgage_broker', 'price_point')
-- GROUP BY kind, normalized
-- ORDER BY kind, mentions DESC
-- LIMIT 40;

-- ────────────────────────────────────────────────────────────────────────────
--  Filtered call lists by KPI — pick whichever query you need.
--  Run them one at a time (uncomment one block).
-- ────────────────────────────────────────────────────────────────────────────


-- ── 1. Calls that ended without a confirmed next step ──────────────────────
-- These are the prime "follow-up at risk" cases.

SELECT call_id, received_at, closer_name, closer_talk_ratio,
       objection_count, buying_signal_count,
       rapport_quality, summary, pdf_url
FROM calls
WHERE next_step_secured = false
ORDER BY received_at DESC;


-- ── 2. Heavy objection calls (≥ 2 objections) ──────────────────────────────
-- Look for objection-handling coaching opportunities.

-- SELECT call_id, received_at, closer_name, objection_count, buying_signal_count,
--        client_sentiment_trajectory, summary, pdf_url
-- FROM calls
-- WHERE objection_count >= 2
-- ORDER BY objection_count DESC, received_at DESC;


-- ── 3. Low-rapport calls ───────────────────────────────────────────────────
-- The LLM marked rapport as 'low'. Pair with the closer's emotion to see if
-- there's a pattern (e.g. low valence on the closer → low rapport).

-- SELECT call_id, received_at, closer_name, rapport_quality,
--        closer_dominant_emotion, closer_valence_mean,
--        client_dominant_emotion, summary, pdf_url
-- FROM calls
-- WHERE rapport_quality = 'low'
-- ORDER BY received_at DESC;


-- ── 4. Sentiment-declining calls with no buying signals ────────────────────
-- High-risk pattern.

-- SELECT call_id, received_at, closer_name, client_sentiment_trajectory,
--        buying_signal_count, next_step_secured, summary, pdf_url
-- FROM calls
-- WHERE client_sentiment_trajectory = 'declining'
--   AND buying_signal_count = 0
-- ORDER BY received_at DESC;


-- ── 5. Closer-dominated calls (>70% talk time) ─────────────────────────────
-- Often correlates with low discovery / monologue style.

-- SELECT call_id, received_at, closer_name, closer_talk_ratio,
--        closer_wpm, client_wpm, summary, pdf_url
-- FROM calls
-- WHERE closer_talk_ratio > 0.70
-- ORDER BY closer_talk_ratio DESC;


-- ── 6. Likely-close calls — your wins ──────────────────────────────────────

-- SELECT call_id, received_at, closer_name, closer_talk_ratio,
--        buying_signal_count, rapport_quality, summary, pdf_url
-- FROM calls
-- WHERE predicted_outcome = 'likely_close'
-- ORDER BY received_at DESC;


-- ── 7. Coaching score leaderboard (this week, scored calls only) ───────────

-- SELECT closer_name,
--        COUNT(*)              AS calls_scored,
--        ROUND(AVG(coaching_score)::numeric, 1) AS avg_score
-- FROM calls
-- WHERE coaching_score IS NOT NULL
--   AND received_at >= NOW() - INTERVAL '7 days'
-- GROUP BY closer_name
-- ORDER BY avg_score DESC;

-- ────────────────────────────────────────────────────────────────────────────
--  Recent calls — daily-driver overview
--
--  Returns the most recent 25 calls with headline KPIs and the PDF link.
--  Use this when you want a quick "what happened today" view.
-- ────────────────────────────────────────────────────────────────────────────

SELECT
    call_id,
    received_at        AS at,
    file_name,
    closer_name,
    client_name,
    duration_sec,
    ROUND(closer_talk_ratio * 100)::INT AS closer_talk_pct,
    objection_count,
    buying_signal_count,
    next_step_secured,
    rapport_quality,
    predicted_outcome,
    coaching_score,
    closer_sentiment_trajectory,
    client_sentiment_trajectory,
    client_dominant_emotion,
    pdf_url
FROM calls
ORDER BY received_at DESC
LIMIT 25;

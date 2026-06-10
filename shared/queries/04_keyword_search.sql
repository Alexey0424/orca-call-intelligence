-- ────────────────────────────────────────────────────────────────────────────
--  Keyword text search — exact substring (ILIKE) matching, no embeddings.
--  Use this when you know the literal word/phrase. For meaning-based search,
--  see 06_semantic_search.sql or hit /search on ser-api.
--
--  Replace the placeholder text and run.
-- ────────────────────────────────────────────────────────────────────────────


-- ── 1. Search the LLM-written summary ──────────────────────────────────────
-- Wide net: matches if the keyword appears anywhere in the summary text.

SELECT call_id, received_at, closer_name, summary, pdf_url
FROM calls
WHERE summary ILIKE '%price%'
ORDER BY received_at DESC;


-- ── 2. Search the raw transcript (utterance by utterance) ──────────────────
-- Returns specific utterances + timestamps. Drill down with 05_single_call_drilldown.

-- SELECT call_id, idx, role,
--        ROUND(start_sec)::INT AS at_sec, text,
--        sentiment_label, emotion_label, valence, arousal
-- FROM utterances
-- WHERE text ILIKE '%vacant%'
-- ORDER BY call_id, idx;


-- ── 3. Filter to client-only utterances containing a phrase ────────────────
-- Useful for "what does the client say when …".

-- SELECT call_id, ROUND(start_sec)::INT AS at_sec, text, sentiment_label, valence
-- FROM utterances
-- WHERE role = 'client'
--   AND text ILIKE '%expensive%'
-- ORDER BY call_id, idx;


-- ── 4. Search the call_events table for objection quotes ──────────────────
-- The LLM-extracted objection events are stored verbatim in `call_events`.

-- SELECT call_id, ROUND(start_sec)::INT AS at_sec, text
-- FROM call_events
-- WHERE kind = 'objection'
--   AND text ILIKE '%price%'
-- ORDER BY call_id, start_sec;


-- ── 5. Find every mention of a specific competitor ─────────────────────────
-- GLiNER puts these in `call_events.kind = 'competitor'` with both verbatim text
-- and a lowercase `normalized` column for grouping.

-- SELECT call_id, text, normalized, confidence, ROUND(start_sec)::INT AS at_sec
-- FROM call_events
-- WHERE kind = 'competitor'
--   AND normalized ILIKE '%compass%'
-- ORDER BY call_id, start_sec;

-- ────────────────────────────────────────────────────────────────────────────
--  Semantic search examples — pgvector cosine similarity.
--
--  These reproduce what the /search HTTP endpoint does, but directly in SQL
--  so you can prototype filters and joins.
--
--  pgvector operators:
--    <=>   cosine distance         (lower = more similar)
--    <->   euclidean (L2) distance
--    <#>   negative inner product
--
--  Similarity = 1 - cosine_distance.
-- ────────────────────────────────────────────────────────────────────────────


-- ── 1. "Find calls similar to a known reference call" ─────────────────────
-- Pick a known good/bad call and surface its 5 nearest neighbours.

WITH q AS (
    SELECT summary_embedding AS vec
    FROM calls
    WHERE call_id = '20260508-115207-9cb0d5dd'   -- ← edit me
)
SELECT call_id,
       ROUND((1 - (summary_embedding <=> (SELECT vec FROM q)))::NUMERIC, 3) AS similarity,
       predicted_outcome,
       summary
FROM calls
WHERE summary_embedding IS NOT NULL
ORDER BY summary_embedding <=> (SELECT vec FROM q)
LIMIT 5;


-- ── 2. "Find calls similar to a query string" ─────────────────────────────
-- We can't call Ollama from inside SQL, so use ser-api's /search endpoint:
--   curl -fsS "http://localhost:8000/search?q=price+objection&top_k=5"


-- ── 3. Per-utterance search across calls ──────────────────────────────────
-- Find moments anywhere in the corpus that are semantically close to a
-- reference utterance. Useful for "find all client objections similar to X".

WITH q AS (
    SELECT text_embedding AS vec
    FROM utterances
    WHERE call_id = '20260508-115207-9cb0d5dd' AND idx = 5   -- ← edit me
)
SELECT call_id,
       idx,
       role,
       ROUND(start_sec)::INT AS at_sec,
       ROUND((1 - (text_embedding <=> (SELECT vec FROM q)))::NUMERIC, 3) AS similarity,
       text
FROM utterances
WHERE text_embedding IS NOT NULL
  AND role = 'client'
ORDER BY text_embedding <=> (SELECT vec FROM q)
LIMIT 10;


-- ── 4. Combine semantic + structured filters (the killer feature) ─────────
-- "Find calls semantically similar to a reference, but only those with
--  objection_count >= 2 AND from the last 30 days."

-- WITH q AS (
--     SELECT summary_embedding AS vec
--     FROM calls WHERE call_id = '<KNOWN_CALL_ID>'
-- )
-- SELECT call_id,
--        objection_count,
--        ROUND((1 - (summary_embedding <=> (SELECT vec FROM q)))::NUMERIC, 3) AS similarity,
--        summary
-- FROM calls
-- WHERE summary_embedding IS NOT NULL
--   AND objection_count >= 2
--   AND received_at >= now() - INTERVAL '30 days'
-- ORDER BY summary_embedding <=> (SELECT vec FROM q)
-- LIMIT 10;

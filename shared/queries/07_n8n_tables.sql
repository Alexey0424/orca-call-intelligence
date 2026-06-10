-- ────────────────────────────────────────────────────────────────────────────
--  n8n internal tables — workflow state, history, sharing, publish status.
--
--  Useful when troubleshooting "why isn't my workflow showing up" or
--  "how many versions of this workflow exist".
-- ────────────────────────────────────────────────────────────────────────────


-- ── 1. Workflows currently registered ──────────────────────────────────────

SELECT id,
       name,
       active,
       "isArchived",
       "activeVersionId" IS NOT NULL AS published,
       "versionCounter"
FROM workflow_entity
ORDER BY name;


-- ── 2. Sharing / ownership ─────────────────────────────────────────────────

SELECT sw."workflowId",
       we.name        AS workflow_name,
       p.name         AS project,
       sw.role,
       sw."createdAt"
FROM shared_workflow sw
JOIN workflow_entity we ON we.id = sw."workflowId"
JOIN project p          ON p.id  = sw."projectId"
ORDER BY sw."createdAt" DESC;


-- ── 3. Version history per workflow ────────────────────────────────────────

SELECT we.name,
       COUNT(wh."versionId")          AS versions,
       MAX(wh."createdAt")            AS latest_version_at,
       we."activeVersionId"
FROM workflow_entity we
LEFT JOIN workflow_history wh ON wh."workflowId" = we.id
GROUP BY we.id, we.name, we."activeVersionId"
ORDER BY we.name;


-- ── 4. Recent execution history (n8n keeps this for the UI) ────────────────

SELECT id,
       "workflowId",
       finished,
       "startedAt",
       "stoppedAt",
       mode,
       status
FROM execution_entity
ORDER BY "startedAt" DESC NULLS LAST
LIMIT 20;


-- ── 5. Published-version sanity (after CLI imports, this should match
--     workflow_entity.activeVersionId for every workflow you can see). ─────

SELECT we.name,
       we."activeVersionId" = wpv."publishedVersionId" AS in_sync,
       we."activeVersionId",
       wpv."publishedVersionId"
FROM workflow_entity we
LEFT JOIN workflow_published_version wpv ON wpv."workflowId" = we.id
ORDER BY we.name;

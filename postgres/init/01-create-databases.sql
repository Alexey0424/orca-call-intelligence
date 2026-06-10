-- ─────────────────────────────────────────────────────────────────────────────
--  Two-DB layout for the stack:
--
--    orca  — app data (call-analysis-api, rag-api, orca-ui-ser-control)
--    n8n   — n8n itself (TypeORM-managed; created from POSTGRES_DB env var)
--
--  Postgres only runs files in /docker-entrypoint-initdb.d when the data
--  directory is empty (first boot of the container against a fresh
--  postgres_storage volume). After that, schema migration is each
--  service's own responsibility — call-analysis-api creates the orca
--  tables at startup (db.ensure_schema()).
--
--  To force a clean rebuild:
--      docker compose down
--      docker volume rm <project>_postgres_storage
--      docker compose --profile gpu-nvidia up -d
-- ─────────────────────────────────────────────────────────────────────────────

CREATE DATABASE orca;

# Postgres query starter pack

Drop these into Adminer's **SQL command** box (`http://localhost:8081` → SQL command), or run via shell:

```bash
docker compose exec -T postgres psql -U root -d orca < shared/queries/01_recent_calls.sql
```

| File | Purpose |
|---|---|
| `01_recent_calls.sql` | A daily-driver overview of the latest calls + their headline KPIs |
| `02_calls_by_kpi.sql` | Common filters: no next step, heavy objections, low rapport, declining sentiment |
| `03_aggregate_metrics.sql` | Cross-call aggregates: per-emotion call counts, average WPM, etc. |
| `04_keyword_search.sql` | ILIKE text search over `summary`, `utterances.text`, and `call_events.text` |
| `05_single_call_drilldown.sql` | Replace `<CALL_ID>` and inspect everything about that one call |
| `06_semantic_search.sql` | pgvector cosine-similarity examples (find similar calls / similar moments) |
| `07_n8n_tables.sql` | n8n's internal tables — workflows, history, sharing, publish state |

Each file has a header comment that explains what each query does and which fields to tweak. Most are read-only `SELECT`s; nothing here modifies data.

To wipe SER test data without dropping schema, see `shared/wipe_test_data.sh`.

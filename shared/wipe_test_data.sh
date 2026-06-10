#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  wipe_test_data.sh — clear SER test data without dropping the schema.
#
#  By default this only TRUNCATEs the SER tables in the `orca` DB
#  (calls + utterances + call_events + pipeline_runs). Pass --files to also
#  delete the rendered PDF/HTML artefacts under data/reports/. The n8n DB
#  is never touched.
#
#  Usage (run from repo root, with the stack up):
#
#      bash shared/wipe_test_data.sh           # interactive, DB only
#      bash shared/wipe_test_data.sh --files   # also clear data/reports/*
#      bash shared/wipe_test_data.sh --yes     # skip confirmation
#      bash shared/wipe_test_data.sh --files --yes
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PG_USER="${POSTGRES_USER:-root}"
PG_DB="${POSTGRES_ORCA_DB:-orca}"
ALSO_FILES=false
SKIP_CONFIRM=false

usage() {
    cat <<EOF
wipe_test_data.sh — clear SER test rows + (optionally) rendered files.

Options:
  --files     Also delete everything under data/reports/
  --yes, -y   Skip the confirmation prompt
  -h, --help  This help text
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --files)    ALSO_FILES=true;    shift ;;
        --yes|-y)   SKIP_CONFIRM=true;  shift ;;
        -h|--help)  usage; exit 0 ;;
        *)          echo "unknown arg: $1" >&2; usage; exit 1 ;;
    esac
done

# ── Show what will happen ────────────────────────────────────────────────────
echo "── What this will do ──────────────────────────────────────────"
echo "  • TRUNCATE calls + utterances + call_events + pipeline_runs (RESTART IDENTITY CASCADE)"
echo "    – schema and indexes stay intact; the n8n DB is untouched"
if $ALSO_FILES; then
    echo "  • DELETE everything under ./data/reports/* (PDFs, HTMLs, input.json)"
fi
echo "───────────────────────────────────────────────────────────────"

if ! $SKIP_CONFIRM; then
    read -r -p "Continue? [y/N] " ans
    case "$ans" in
        y|Y|yes|YES) ;;
        *) echo "aborted."; exit 1 ;;
    esac
fi

# ── DB truncate ──────────────────────────────────────────────────────────────
echo
echo "→ truncating calls + utterances + call_events + pipeline_runs…"
docker compose exec -T postgres psql -U "$PG_USER" -d "$PG_DB" <<'SQL'
TRUNCATE TABLE call_events, utterances, calls, pipeline_runs RESTART IDENTITY CASCADE;
SELECT 'calls'         AS table, COUNT(*) AS rows FROM calls
UNION ALL
SELECT 'utterances'    AS table, COUNT(*) AS rows FROM utterances
UNION ALL
SELECT 'call_events'   AS table, COUNT(*) AS rows FROM call_events
UNION ALL
SELECT 'pipeline_runs' AS table, COUNT(*) AS rows FROM pipeline_runs;
SQL
echo "✓ DB tables empty"

# ── On-disk cleanup (optional) ───────────────────────────────────────────────
if $ALSO_FILES; then
    echo
    echo "→ clearing data/reports/* …"
    if [[ -d data/reports ]]; then
        find data/reports -mindepth 1 -delete
        echo "✓ data/reports cleared"
    else
        echo "⚠ data/reports does not exist; skipping"
    fi
fi

echo
echo "Done."

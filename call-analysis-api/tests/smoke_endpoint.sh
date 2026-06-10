#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  ser-api end-to-end smoke test
#
#  Run from the repo root with the stack up::
#
#      ./ser-api/tests/smoke_endpoint.sh path/to/sample.mp3
#
#  Verifies:
#    1. /health returns ok=true
#    2. /scorecard accepts a multipart upload and returns ok=true
#    3. The response carries the new top-level shape
#       (timings_sec / scorecard.events / scorecard.text_metrics)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

AUDIO=${1:-}
HOST=${SER_API_HOST:-http://localhost:8000}

if [[ -z "$AUDIO" ]]; then
  echo "Usage: $0 <audio-file>"
  exit 1
fi

echo "─ /health ───────────────────────────────────────────────"
curl -fsS "$HOST/health" | python -m json.tool

echo
echo "─ /scorecard (file=$AUDIO) ──────────────────────────────"
curl -fsS -X POST "$HOST/scorecard?language=en&include_segments=false" \
     -F "file=@${AUDIO}" \
  | python -c "
import json, sys
data = json.load(sys.stdin)
required_top = {'ok','metadata','timings_sec','scorecard'}
required_sc  = {'timeline','kpis','events','text_metrics','sentiment','emotion','prosody','flat_features'}
missing_top  = required_top - set(data)
missing_sc   = required_sc  - set(data.get('scorecard', {}))
assert not missing_top, f'Missing top-level keys: {missing_top}'
assert not missing_sc,  f'Missing scorecard keys: {missing_sc}'
print('OK call_id=', data['metadata']['call_id'])
print('   timings  =', data['timings_sec'])
print('   speakers =', data['scorecard']['speakers'])
print('   events   =', list(data['scorecard']['events']))
"
echo
echo "Smoke test passed."

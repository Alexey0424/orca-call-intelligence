#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  ser-api-render end-to-end smoke test
#
#  Usage: ./tests/smoke_endpoint.sh path/to/scorecard.json
#  (You can use any data/reports/<id>/input.json as the test payload.)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PAYLOAD=${1:-}
HOST=${RENDER_HOST:-http://localhost:8010}

if [[ -z "$PAYLOAD" ]]; then
  echo "Usage: $0 <scorecard.json>"
  exit 1
fi

echo "─ /health ───────────────────────────────────────────────"
curl -fsS "$HOST/health" | python -m json.tool

echo
echo "─ /render?output_format=both (payload=$PAYLOAD) ─────────"
curl -fsS -X POST "$HOST/render?output_format=both" \
     -H "Content-Type: application/json" \
     --data-binary "@${PAYLOAD}" \
  | python -c "
import json, sys
data = json.load(sys.stdin)
required = {'ok','service','render_id','call_id','report_title','executive_summary','files'}
missing = required - set(data)
assert not missing, f'Missing keys: {missing}'
files = data['files']
assert files['pdf_url'],  'pdf_url missing'
assert files['html_url'], 'html_url missing'
print('OK render_id=', data['render_id'])
print('   pdf_url =', files['pdf_url'])
print('   html_url=', files['html_url'])
"
echo
echo "Smoke test passed."

# Golden Test Harness

Locks in current /scorecard behavior on a small set of hand-picked
calls so that future prompt / model / pipeline changes can't silently
shift KPI outputs.

This directory is **scaffolding only** today. You populate it in **phase 2**
(after ~1 week of real calls) with audio you've hand-labeled.

## Files

* `manifest.json` — the gold set. One entry per call:
  ```json
  {
    "name":  "elias_strong_close",
    "audio": "/data/golden/elias_strong_close.wav",
    "kwargs": { "num_speakers": 2, "include_audio_features": true },
    "expected": {
      "kpis": {
        "objection_count":     { "min": 0, "max": 2 },
        "buying_signal_count": { "min": 2, "max": 6 },
        "next_step_secured":   true,
        "predicted_outcome":   "likely_close"
      },
      "timeline": {
        "interruptions": { "min": 0, "max": 3 }
      },
      "audio_quality": {
        "audio_snr_db": { "min": 15 }
      }
    }
  }
  ```
* `run_golden.py` — the runner. POSTs each entry to `/scorecard`, then
  walks `expected` and asserts each leaf either matches exactly (scalars)
  or falls inside `{min, max}` (numerics). Prints a pass/fail per call.

## Running

```bash
# Inside the container so /data/golden is reachable:
docker compose exec ser-api python /app/tests/golden/run_golden.py

# Or from the host with a different base URL:
SER_API_URL=http://127.0.0.1:8000 \
  docker compose exec ser-api python /app/tests/golden/run_golden.py
```

Exit code: 0 if every call passes; non-zero on any failure.
Suitable for a CI step once the manifest is populated.

## Recommended phase-2 workflow

1. After ~1 week of real calls land, scan dashboards for **representative
   call shapes**: at least one strong close, one weak close, one
   objection-heavy call, one quiet/mostly-listening call, one with bad
   audio. 5–10 calls is plenty.
2. For each: copy the audio into `/data/golden/`, **listen to it**, write
   down what you'd expect the KPIs to look like (with tolerance bands
   for the noisy ones). Add an entry to `manifest.json`.
3. Run `run_golden.py` once. Tighten the tolerance bands so all entries
   pass cleanly on the current pipeline.
4. From that point on: re-run after every prompt edit or model swap.
   If a previously-passing call starts failing, something material
   shifted — investigate before merging.

## What this is NOT

Not a model accuracy benchmark. The point is **regression detection**,
not measuring how good the pipeline is in absolute terms. For absolute
accuracy work, hand-label calls externally and compare against the
LLM-prediction columns on `calls` (`predicted_outcome`, `coaching_score`,
etc.) — the dedicated `call_annotations` table from the v6 schema was
dropped in the May 2026 rebuild.

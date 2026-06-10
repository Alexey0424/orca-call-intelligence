"""Run the golden /scorecard regression suite.

See ./README.md for the manifest schema and recommended workflow. This
script is *scaffolding* — it does the right thing on whatever you put in
``manifest.json``, but you supply the cases.

Exit codes:
  0  — every case passed (or manifest is empty: vacuously true).
  1  — at least one case failed.
  2  — manifest was missing or malformed.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx


HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "manifest.json"

SER_API_URL = os.getenv("SER_API_URL", "http://127.0.0.1:8000")
TIMEOUT_SEC = float(os.getenv("GOLDEN_TIMEOUT_SEC", "600"))


def _load_manifest() -> List[Dict[str, Any]]:
    if not MANIFEST.exists():
        print(f"FAIL: manifest not found at {MANIFEST}", file=sys.stderr)
        sys.exit(2)
    try:
        with MANIFEST.open(encoding="utf-8") as f:
            data = json.load(f)
        return list(data.get("cases") or [])
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: manifest parse error: {exc}", file=sys.stderr)
        sys.exit(2)


def _drill(d: Dict[str, Any], path: List[str]) -> Any:
    """Walk a dot-path through a nested dict; None on missing key."""
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _check_value(actual: Any, expected: Any, path: str
                  ) -> Tuple[bool, str]:
    """Compare one leaf.

    Rules:
      * dict with min/max keys → numeric tolerance band
      * plain scalar → exact equality
      * list → element-wise (rare; mostly used to assert ordered IDs)
    """
    if isinstance(expected, dict) and ("min" in expected or "max" in expected):
        if actual is None:
            return False, f"{path}: expected numeric, got None"
        if not isinstance(actual, (int, float)):
            return False, f"{path}: expected number, got {type(actual).__name__}"
        if "min" in expected and actual < expected["min"]:
            return False, f"{path}: {actual} < min {expected['min']}"
        if "max" in expected and actual > expected["max"]:
            return False, f"{path}: {actual} > max {expected['max']}"
        return True, ""

    if expected != actual:
        return False, f"{path}: expected {expected!r}, got {actual!r}"
    return True, ""


def _walk(expected: Dict[str, Any], scorecard: Dict[str, Any],
          prefix: str = "") -> List[str]:
    """Recursively check every leaf in ``expected`` against ``scorecard``."""
    failures: List[str] = []
    for k, v in expected.items():
        path = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict) and not ("min" in v or "max" in v):
            sub = _drill(scorecard, path.split("."))
            if not isinstance(sub, dict):
                failures.append(f"{path}: expected dict, got {sub!r}")
                continue
            failures.extend(_walk(v, scorecard, prefix=path))
        else:
            actual = _drill(scorecard, path.split("."))
            ok, msg = _check_value(actual, v, path)
            if not ok:
                failures.append(msg)
    return failures


def _run_case(case: Dict[str, Any]) -> Tuple[bool, str, float]:
    """Run one case end-to-end. Returns (passed, message, elapsed_sec)."""
    name  = case.get("name") or case.get("audio") or "<unnamed>"
    audio = case.get("audio")
    kwargs = case.get("kwargs") or {}
    expected = case.get("expected") or {}

    if not audio or not Path(audio).exists():
        return False, f"audio not found: {audio}", 0.0

    t0 = time.monotonic()
    try:
        with open(audio, "rb") as f:
            files = {"file": (Path(audio).name, f.read(), "application/octet-stream")}
        params = dict(kwargs)
        # We don't want golden runs to persist into the live calls table.
        params.setdefault("persist", "false")
        params.setdefault("include_utterances", "false")
        resp = httpx.post(f"{SER_API_URL}/scorecard", params=params,
                          files=files, timeout=TIMEOUT_SEC)
        resp.raise_for_status()
        body = resp.json()
    except Exception as exc:  # noqa: BLE001
        return False, f"{name}: HTTP failure {exc!r}", time.monotonic() - t0

    scorecard = body.get("scorecard") or {}
    failures = _walk(expected, scorecard)
    elapsed = time.monotonic() - t0
    if failures:
        return False, f"{name}: " + "; ".join(failures), elapsed
    return True, f"{name}: OK", elapsed


def main() -> int:
    cases = _load_manifest()
    if not cases:
        print("golden: manifest is empty — nothing to check (phase 1).")
        return 0

    passed = failed = 0
    for case in cases:
        ok, msg, elapsed = _run_case(case)
        status = "PASS" if ok else "FAIL"
        print(f"{status}  ({elapsed:5.1f}s)  {msg}")
        if ok:
            passed += 1
        else:
            failed += 1

    print(f"\ngolden: {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

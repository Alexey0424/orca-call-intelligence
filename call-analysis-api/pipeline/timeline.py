"""Speaker-activity timeline derived from diarization turns.

Sweep through the start/end events ordered by time. Track the set of
currently-active speakers and accumulate per-speaker active time, total
silence, and total overlap. An "interruption" is when a new speaker
joins while at least one other speaker is already active.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple


def _accumulate(active: set, dt: float, silence_acc: float, overlap_acc: float,
                per_speaker: Dict[str, float]
                ) -> Tuple[float, float]:
    n = len(active)
    if dt <= 0:
        return silence_acc, overlap_acc
    if n == 0:
        return silence_acc + dt, overlap_acc
    if n == 1:
        spk = next(iter(active))
        per_speaker[spk] = per_speaker.get(spk, 0.0) + dt
        return silence_acc, overlap_acc
    for spk in active:
        per_speaker[spk] = per_speaker.get(spk, 0.0) + dt
    return silence_acc, overlap_acc + dt


def compute(turns: List[Dict[str, Any]], duration_sec: float) -> Dict[str, Any]:
    events: List[Tuple[float, int, str]] = []
    for t in turns:
        a, b, spk = float(t["start"]), float(t["end"]), str(t["speaker"])
        if b > a:
            events.append((a, +1, spk))
            events.append((b, -1, spk))
    events.sort(key=lambda x: (x[0], -x[1]))

    active: set = set()
    silence = overlap = 0.0
    per_speaker: Dict[str, float] = {}
    interruptions: List[Dict[str, Any]] = []
    prev_t = 0.0

    for t, kind, spk in events:
        t = max(0.0, min(float(t), duration_sec))
        silence, overlap = _accumulate(active, t - prev_t, silence, overlap, per_speaker)
        if kind == +1:
            if active and spk not in active:
                interruptions.append({"time": round(t, 3),
                                      "speaker": spk,
                                      "overlapping_with": sorted(active)})
            active.add(spk)
        else:
            active.discard(spk)
        prev_t = t

    silence, overlap = _accumulate(active, duration_sec - prev_t,
                                    silence, overlap, per_speaker)

    silence_ratio = (silence / duration_sec) if duration_sec > 0 else 0.0
    overlap_ratio = (overlap / duration_sec) if duration_sec > 0 else 0.0

    return {
        "duration_sec":        round(duration_sec, 2),
        "silence_sec":         round(silence,    2),
        "overlap_sec":         round(overlap,    2),
        "silence_ratio":       round(silence_ratio, 4),
        "overlap_ratio":       round(overlap_ratio, 4),
        "speaker_active_sec":  {k: round(v, 2) for k, v in per_speaker.items()},
        "interruption_events": interruptions,
        "interruptions":       len(interruptions),
    }

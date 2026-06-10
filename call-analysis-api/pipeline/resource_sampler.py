"""Per-stage / per-sub-stage resource-peak sampler.

Captures the high-water mark of GPU VRAM, CPU, and RAM while a block of
pipeline work runs, expressed as % of capacity and split into:
  rel  — ser-api's own process (this pipeline)
  glob — the whole box (incl. Ollama, Postgres, OS)
  other = glob - rel (everything that isn't us)

GPU-relative peak is exact (torch peak-stats); CPU/RAM peaks and the
GPU-global peak come from a lightweight daemon thread polling psutil +
torch.cuda.mem_get_info every RESOURCE_SAMPLE_INTERVAL_MS.

Best-effort by contract: nothing here may raise into the pipeline.
"""
from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from typing import Dict, Optional

import psutil

import config

logger = logging.getLogger(__name__)

try:
    import torch
    _HAS_GPU = bool(torch.cuda.is_available())
except Exception:  # noqa: BLE001
    torch = None  # type: ignore
    _HAS_GPU = False

# NVML for whole-card GPU compute utilization (= nvidia-smi's utilization.gpu).
# Device-wide only — NVML cannot attribute compute util per process, so the
# GPU metric is global-only (no rel/other split).
try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:  # noqa: BLE001
    pynvml = None  # type: ignore
    _NVML_HANDLE = None

_PROC = psutil.Process()
_NCPU = psutil.cpu_count() or 1


def _gpu_total_bytes() -> int:
    if _HAS_GPU:
        try:
            return int(torch.cuda.get_device_properties(0).total_memory)
        except Exception:  # noqa: BLE001
            return 0
    return 0


def _gpu_util() -> Optional[float]:
    """Whole-card GPU compute utilization % via NVML. None if unavailable."""
    if _NVML_HANDLE is None:
        return None
    try:
        return float(pynvml.nvmlDeviceGetUtilizationRates(_NVML_HANDLE).gpu)
    except Exception:  # noqa: BLE001
        return None


def _ram_total_bytes() -> int:
    try:
        return int(psutil.virtual_memory().total)
    except Exception:  # noqa: BLE001
        return 0


_GPU_TOTAL = _gpu_total_bytes()
_RAM_TOTAL = _ram_total_bytes()

Triple = Dict[str, Optional[float]]
Peaks = Dict[str, Triple]


def _pct(used: float, total: float) -> Optional[float]:
    """used/total as a 0..100 percentage, clamped + 1-dp. None if no capacity."""
    if not total or total <= 0:
        return None
    return round(max(0.0, min(100.0, used / total * 100.0)), 1)


def _triple(rel: Optional[float], glob: Optional[float]) -> Triple:
    other = (round(glob - rel, 1)
             if rel is not None and glob is not None else None)
    return {"rel": rel, "glob": glob, "other": other}


class _Tracker:
    """Holds running maxima during a tracked block; populated on exit."""

    def __init__(self) -> None:
        self.elapsed: float = 0.0
        self.peaks: Peaks = {"vram": _triple(None, None),
                             "cpu": _triple(None, None),
                             "ram": _triple(None, None)}
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._t0 = 0.0
        self._cpu_rel = 0.0
        self._cpu_glob = 0.0
        self._ram_rel = 0.0
        self._ram_glob = 0.0
        self._gpu_glob = 0.0
        self._gpu_util: Optional[float] = None  # whole-card compute util %

    def _loop(self, interval_s: float) -> None:
        # Prime the cpu_percent counters so subsequent reads return a real delta.
        try:
            _PROC.cpu_percent(interval=None)
            psutil.cpu_percent(interval=None)
        except Exception:  # noqa: BLE001
            pass
        # Sample immediately, then on each interval — so a sub-interval stage
        # (e.g. ffmpeg convert) still gets a RAM + global-VRAM reading. CPU
        # needs a delta between calls, so it fills in from the second tick on.
        while True:
            try:
                self._cpu_rel = max(self._cpu_rel,
                                    _PROC.cpu_percent(interval=None) / _NCPU)
                self._cpu_glob = max(self._cpu_glob,
                                     psutil.cpu_percent(interval=None))
                self._ram_rel = max(self._ram_rel, _PROC.memory_info().rss)
                self._ram_glob = max(self._ram_glob,
                                     psutil.virtual_memory().used)
                if _HAS_GPU:
                    free, total = torch.cuda.mem_get_info()
                    self._gpu_glob = max(self._gpu_glob, total - free)
                u = _gpu_util()
                if u is not None:
                    self._gpu_util = (u if self._gpu_util is None
                                      else max(self._gpu_util, u))
            except Exception:  # noqa: BLE001
                pass
            if self._stop.wait(interval_s):
                break

    def _finalize(self) -> None:
        gpu_rel = gpu_glob = None
        if _HAS_GPU and _GPU_TOTAL:
            try:
                gpu_rel = _pct(torch.cuda.max_memory_allocated(), _GPU_TOTAL)
            except Exception:  # noqa: BLE001
                pass
            gpu_glob = _pct(self._gpu_glob, _GPU_TOTAL) if self._gpu_glob else None
        cpu_rel = round(min(100.0, self._cpu_rel), 1) if self._cpu_rel else None
        cpu_glob = round(min(100.0, self._cpu_glob), 1) if self._cpu_glob else None
        ram_rel = _pct(self._ram_rel, _RAM_TOTAL) if self._ram_rel else None
        ram_glob = _pct(self._ram_glob, _RAM_TOTAL) if self._ram_glob else None
        gpu_util = round(self._gpu_util, 1) if self._gpu_util is not None else None
        self.peaks = {"gpu":  _triple(None, gpu_util),
                      "vram": _triple(gpu_rel, gpu_glob),
                      "cpu":  _triple(cpu_rel, cpu_glob),
                      "ram":  _triple(ram_rel, ram_glob)}


@contextmanager
def track():
    """Context manager: resets GPU peak stats, runs a sampler thread, and on
    exit fills `.peaks` (the high-water marks) and `.elapsed` (seconds)."""
    tr = _Tracker()
    if _HAS_GPU:
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:  # noqa: BLE001
            pass
    tr._t0 = time.monotonic()
    if config.RESOURCE_SAMPLING_ENABLED:
        interval_s = max(0.02, config.RESOURCE_SAMPLE_INTERVAL_MS / 1000.0)
        tr._thread = threading.Thread(target=tr._loop, args=(interval_s,),
                                      daemon=True, name="resource_sampler")
        tr._thread.start()
    try:
        yield tr
    finally:
        tr.elapsed = round(time.monotonic() - tr._t0, 3)
        tr._stop.set()
        if tr._thread is not None:
            tr._thread.join(timeout=0.5)
        try:
            tr._finalize()
        except Exception:  # noqa: BLE001
            logger.warning("resource_sampler_finalize_failed", exc_info=True)

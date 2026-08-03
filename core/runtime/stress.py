"""Dependency-light helpers for lifecycle and memory stress diagnostics.

The application does not depend on psutil.  Linux RSS values are read from
``/proc`` and all reports contain only JSON-serialisable primitives so stress
runs can be archived with a bug report.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Iterable, Sequence

_MIB = 1024.0


@dataclass(frozen=True, slots=True)
class StressSample:
    """One process/model lifecycle observation."""

    cycle: int
    phase: str
    worker_pid: int | None
    worker_generation: int
    parent_rss_mib: float | None = None
    worker_rss_mib: float | None = None
    vram_allocated_gib: float | None = None
    vram_reserved_gib: float | None = None
    vram_device_used_gib: float | None = None
    timestamp: float = 0.0

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class StressSummary:
    """Compact memory/process summary computed from stress samples."""

    sample_count: int
    completed_cycles: int
    unique_worker_pids: tuple[int, ...]
    parent_rss_first_mib: float | None
    parent_rss_last_mib: float | None
    parent_rss_peak_mib: float | None
    parent_rss_drift_mib: float | None
    worker_rss_first_mib: float | None
    worker_rss_last_mib: float | None
    worker_rss_peak_mib: float | None
    worker_rss_drift_mib: float | None
    max_vram_allocated_gib: float | None
    max_vram_reserved_gib: float | None
    max_vram_device_used_gib: float | None

    def as_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["unique_worker_pids"] = list(self.unique_worker_pids)
        return data


def read_process_rss_mib(pid: int | None) -> float | None:
    """Return current resident memory for ``pid`` on Linux.

    ``None`` is returned when the process no longer exists or the platform does
    not expose ``/proc/<pid>/status``.  Stress diagnostics must never make the
    application fail merely because an optional metric is unavailable.
    """
    if pid is None or pid <= 0:
        return None
    status = Path(f"/proc/{int(pid)}/status")
    try:
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                fields = line.split()
                return float(fields[1]) / _MIB
    except (FileNotFoundError, PermissionError, OSError, ValueError, IndexError):
        return None
    return None


def process_exists(pid: int | None) -> bool:
    """Check process existence without sending a terminating signal."""
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def wait_for_process_exit(pid: int | None, timeout_s: float = 3.0) -> bool:
    """Wait until ``pid`` is gone, returning ``False`` on timeout."""
    deadline = time.monotonic() + max(0.0, float(timeout_s))
    while process_exists(pid):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.02)
    return True


def make_sample(
    *,
    cycle: int,
    phase: str,
    worker_pid: int | None,
    worker_generation: int,
    metadata: dict[str, object] | None = None,
) -> StressSample:
    """Build a sample from process RSS and optional model cleanup telemetry."""
    metadata = metadata or {}
    return StressSample(
        cycle=int(cycle),
        phase=str(phase),
        worker_pid=worker_pid,
        worker_generation=int(worker_generation),
        parent_rss_mib=read_process_rss_mib(os.getpid()),
        worker_rss_mib=read_process_rss_mib(worker_pid),
        vram_allocated_gib=_optional_float(metadata, "after_allocated_gib"),
        vram_reserved_gib=_optional_float(metadata, "after_reserved_gib"),
        vram_device_used_gib=_optional_float(metadata, "after_device_used_gib"),
        timestamp=time.time(),
    )


def summarize_samples(samples: Sequence[StressSample]) -> StressSummary:
    """Summarise completed lifecycle samples without hiding missing metrics."""
    parent = _available(sample.parent_rss_mib for sample in samples)
    worker = _available(sample.worker_rss_mib for sample in samples)
    allocated = _available(sample.vram_allocated_gib for sample in samples)
    reserved = _available(sample.vram_reserved_gib for sample in samples)
    device = _available(sample.vram_device_used_gib for sample in samples)
    pids = tuple(sorted({sample.worker_pid for sample in samples if sample.worker_pid}))
    cycles = max((sample.cycle for sample in samples), default=0)
    return StressSummary(
        sample_count=len(samples),
        completed_cycles=cycles,
        unique_worker_pids=pids,
        parent_rss_first_mib=_first(parent),
        parent_rss_last_mib=_last(parent),
        parent_rss_peak_mib=max(parent) if parent else None,
        parent_rss_drift_mib=_drift(parent),
        worker_rss_first_mib=_first(worker),
        worker_rss_last_mib=_last(worker),
        worker_rss_peak_mib=max(worker) if worker else None,
        worker_rss_drift_mib=_drift(worker),
        max_vram_allocated_gib=max(allocated) if allocated else None,
        max_vram_reserved_gib=max(reserved) if reserved else None,
        max_vram_device_used_gib=max(device) if device else None,
    )


def tail_drift(values: Iterable[float | None], window: int = 3) -> float | None:
    """Median last-window minus median first-window.

    Using small medians is less sensitive to allocator jitter than comparing one
    first and one last sample.  Positive values represent growth.
    """
    available = _available(values)
    if len(available) < 2:
        return None
    size = max(1, min(int(window), len(available) // 2 or 1))
    return median(available[-size:]) - median(available[:size])


def write_stress_report(
    path: str | Path,
    *,
    kind: str,
    settings: dict[str, object],
    samples: Sequence[StressSample],
    checks: dict[str, object],
) -> Path:
    """Write an atomic JSON report and return the final path."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = summarize_samples(samples)
    payload = {
        "kind": str(kind),
        "settings": dict(settings),
        "summary": summary.as_dict(),
        "checks": dict(checks),
        "samples": [sample.as_dict() for sample in samples],
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(output)
    return output


def _optional_float(metadata: dict[str, object], key: str) -> float | None:
    value = metadata.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _available(values: Iterable[float | None]) -> list[float]:
    return [float(value) for value in values if value is not None]


def _first(values: Sequence[float]) -> float | None:
    return values[0] if values else None


def _last(values: Sequence[float]) -> float | None:
    return values[-1] if values else None


def _drift(values: Sequence[float]) -> float | None:
    return values[-1] - values[0] if len(values) >= 2 else None

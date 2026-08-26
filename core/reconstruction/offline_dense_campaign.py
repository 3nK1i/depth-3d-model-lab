"""Reusable Phase I offline Dense RGB quality-campaign contracts.

This module is intentionally dependency-light.  It owns campaign planning,
pass-only monotonic path selection, and host/GPU telemetry.  Geometry remains
owned by the already-qualified reconstruction/fusion/filtering/persistence
kernels invoked by the Phase I runner.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import subprocess
import threading
import time
from typing import Any, Mapping, Sequence


I1_CAMPAIGN_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class QualifiedPath:
    orders: tuple[int, ...]
    worst_consistency_ratio: float
    worst_median_relative_residual: float

    @property
    def view_count(self) -> int:
        return len(self.orders)

    @property
    def span(self) -> int:
        return 0 if len(self.orders) < 2 else int(self.orders[-1] - self.orders[0])

    def as_dict(self) -> dict[str, Any]:
        median = float(self.worst_median_relative_residual)
        return {
            "orders": list(self.orders),
            "view_count": self.view_count,
            "span": self.span,
            "worst_consistency_ratio": float(self.worst_consistency_ratio),
            # Empty/single-view paths have no edge residual by definition.
            # Keep the internal +inf sentinel for scoring, but serialize it as
            # JSON null so strict allow_nan=False manifests always remain valid.
            "worst_median_relative_residual": median if math.isfinite(median) else None,
        }


def pair_key(order_a: int, order_b: int) -> tuple[int, int]:
    a, b = int(order_a), int(order_b)
    return (a, b) if a < b else (b, a)


def _pair_quality(pair: Mapping[str, Any]) -> tuple[float, float]:
    """Return the exact E3/D2 bidirectional quality tuple for a PASS edge.

    The real diagnostic schema uses ``consistent_ratio_of_comparable`` and
    ``median_absolute_relative_depth_residual``. I1 must consume those exact
    fields rather than a synthetic shorthand, otherwise a valid PASS edge can
    silently degrade to (0, +inf) and poison path tie-breaking/reporting.
    """

    if str(pair.get("status", "")) != "pass":
        return 0.0, math.inf

    directions = [pair.get("a_to_b"), pair.get("b_to_a")]
    consistencies: list[float] = []
    medians: list[float] = []
    for direction in directions:
        if not isinstance(direction, Mapping):
            raise ValueError("PASS dense-consistency edge is missing a direction payload")
        consistency = direction.get("consistent_ratio_of_comparable")
        median = direction.get("median_absolute_relative_depth_residual")
        if consistency is None or median is None:
            raise ValueError("PASS dense-consistency edge is missing required quality metrics")
        consistency_value = float(consistency)
        median_value = float(median)
        if not math.isfinite(consistency_value) or not 0.0 <= consistency_value <= 1.0:
            raise ValueError("PASS dense-consistency edge has invalid consistency")
        if not math.isfinite(median_value) or median_value < 0.0:
            raise ValueError("PASS dense-consistency edge has invalid median residual")
        consistencies.append(consistency_value)
        medians.append(median_value)

    if len(consistencies) != 2 or len(medians) != 2:
        raise ValueError("PASS dense-consistency edge requires two directed quality records")
    return min(consistencies), max(medians)


def _path_score(path: QualifiedPath) -> tuple[Any, ...]:
    # Prefer more qualified views first.  When view count ties, prefer broader
    # temporal coverage, stronger worst-case dense consistency, lower residual,
    # then deterministic earlier/lexicographic order.
    return (
        path.view_count,
        path.span,
        float(path.worst_consistency_ratio),
        -float(path.worst_median_relative_residual),
        tuple(-value for value in path.orders),
    )


def select_longest_monotonic_pass_path(
    available_orders: Sequence[int],
    pair_summaries: Mapping[tuple[int, int], Mapping[str, Any]],
    *,
    max_bridge_gap: int,
) -> QualifiedPath:
    """Select the strongest longest monotonic path using PASS edges only.

    The graph is a DAG in workspace-order space.  Degraded and inconclusive
    edges are never relaxed.  This generalizes the validated E3 bridge idea to
    a full offline campaign without imposing a hand-authored ten-view seed.
    """

    if max_bridge_gap < 1:
        raise ValueError("max_bridge_gap must be positive")
    orders = tuple(sorted({int(value) for value in available_orders}))
    if not orders:
        return QualifiedPath((), 0.0, math.inf)

    best: dict[int, QualifiedPath] = {}
    for order in orders:
        current = QualifiedPath((order,), 1.0, 0.0)
        for previous in orders:
            if previous >= order:
                break
            if order - previous > max_bridge_gap:
                continue
            previous_path = best.get(previous)
            if previous_path is None:
                continue
            pair = pair_summaries.get(pair_key(previous, order))
            if not isinstance(pair, Mapping) or str(pair.get("status", "")) != "pass":
                continue
            consistency, median = _pair_quality(pair)
            candidate = QualifiedPath(
                previous_path.orders + (order,),
                min(previous_path.worst_consistency_ratio, consistency),
                max(previous_path.worst_median_relative_residual, median),
            )
            if _path_score(candidate) > _path_score(current):
                current = candidate
        best[order] = current
    return max(best.values(), key=_path_score)


def uniform_order_subset(orders: Sequence[int], maximum: int) -> tuple[int, ...]:
    values = tuple(sorted({int(value) for value in orders}))
    if maximum <= 0:
        raise ValueError("maximum must be positive")
    if len(values) <= maximum:
        return values
    if maximum == 1:
        return (values[len(values) // 2],)
    selected_indices = {
        int(round(index * (len(values) - 1) / float(maximum - 1)))
        for index in range(maximum)
    }
    selected = tuple(values[index] for index in sorted(selected_indices))
    # Rounding can only collapse indices in pathological tiny cases; fill any
    # missing slots deterministically from the remaining sequence.
    if len(selected) < maximum:
        missing = [value for value in values if value not in set(selected)]
        selected = tuple(sorted(selected + tuple(missing[: maximum - len(selected)])))
    return selected


def read_host_memory_used_mib() -> float | None:
    path = Path("/proc/meminfo")
    try:
        values: dict[str, float] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if ":" not in line:
                continue
            key, rest = line.split(":", 1)
            token = rest.strip().split()[0]
            values[key] = float(token) / 1024.0
        total = values.get("MemTotal")
        available = values.get("MemAvailable")
        if total is None or available is None:
            return None
        return max(0.0, total - available)
    except (OSError, ValueError, IndexError):
        return None


def read_gpu_memory_used_mib() -> float | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    values: list[float] = []
    for line in completed.stdout.splitlines():
        try:
            values.append(float(line.strip()))
        except ValueError:
            continue
    return sum(values) if values else None


class StageTelemetry:
    """Best-effort host/GPU telemetry for one campaign stage.

    Values are host-observed totals, not exclusive process accounting.  This is
    deliberate for I1: detailed per-process profiling is reserved for I5.
    """

    def __init__(self, stage: str, *, poll_seconds: float = 0.5) -> None:
        self.stage = str(stage)
        self.poll_seconds = max(0.1, float(poll_seconds))
        self.started_at = 0.0
        self.elapsed_seconds = 0.0
        self.peak_host_memory_used_mib: float | None = None
        self.peak_gpu_memory_used_mib: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        host = read_host_memory_used_mib()
        gpu = read_gpu_memory_used_mib()
        if host is not None:
            self.peak_host_memory_used_mib = host if self.peak_host_memory_used_mib is None else max(self.peak_host_memory_used_mib, host)
        if gpu is not None:
            self.peak_gpu_memory_used_mib = gpu if self.peak_gpu_memory_used_mib is None else max(self.peak_gpu_memory_used_mib, gpu)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.poll_seconds)

    def __enter__(self) -> "StageTelemetry":
        self.started_at = time.monotonic()
        self._sample()
        self._thread = threading.Thread(target=self._loop, name=f"i1-telemetry-{self.stage}", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.poll_seconds * 3.0))
        self._sample()
        self.elapsed_seconds = max(0.0, time.monotonic() - self.started_at)

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "elapsed_seconds": float(self.elapsed_seconds),
            "peak_host_memory_used_mib": self.peak_host_memory_used_mib,
            "peak_gpu_memory_used_mib": self.peak_gpu_memory_used_mib,
            "resource_scope": "host_observed_total_not_process_exclusive",
        }

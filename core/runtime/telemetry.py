"""Pure runtime telemetry helpers used by the UI and worker contracts.

The helpers in this module deliberately avoid Qt, CUDA, and model imports so
that rate/counter behaviour is deterministic and easy to unit test.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


class RateMeter:
    """Sliding-window event-rate estimator.

    The first event has no measurable rate. Once at least two events exist,
    ``rate`` reports transitions per second over the retained time window.
    """

    def __init__(self, window_seconds: float = 2.0) -> None:
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        self._window_seconds = float(window_seconds)
        self._timestamps: deque[float] = deque()

    def mark(self, timestamp: float) -> float | None:
        self._timestamps.append(float(timestamp))
        self._trim(float(timestamp))
        return self.rate

    def reset(self) -> None:
        self._timestamps.clear()

    @property
    def rate(self) -> float | None:
        if len(self._timestamps) < 2:
            return None
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return None
        return (len(self._timestamps) - 1) / elapsed

    def _trim(self, now: float) -> None:
        cutoff = now - self._window_seconds
        while len(self._timestamps) > 2 and self._timestamps[0] < cutoff:
            self._timestamps.popleft()


@dataclass(slots=True)
class RuntimeCounters:
    """Per-session counters for latest-frame/backpressure diagnostics."""

    source_frames_seen: int = 0
    pending_frames_replaced: int = 0
    frames_submitted: int = 0
    submit_rejected: int = 0
    results_received: int = 0
    pending_results_replaced: int = 0
    results_rendered: int = 0
    stale_results_ignored: int = 0
    inference_errors: int = 0

    @property
    def dropped_before_inference(self) -> int:
        return self.pending_frames_replaced + self.submit_rejected

    @property
    def dropped_before_render(self) -> int:
        return self.pending_results_replaced + self.stale_results_ignored

    @property
    def total_dropped(self) -> int:
        return self.dropped_before_inference + self.dropped_before_render

    def reset(self) -> None:
        self.source_frames_seen = 0
        self.pending_frames_replaced = 0
        self.frames_submitted = 0
        self.submit_rejected = 0
        self.results_received = 0
        self.pending_results_replaced = 0
        self.results_rendered = 0
        self.stale_results_ignored = 0
        self.inference_errors = 0


@dataclass(frozen=True, slots=True)
class GeometryTiming:
    """Measured UI-side point-cloud/depth update timings for one result."""

    point_count: int
    depth_view_ms: float = 0.0
    geometry_cpu_ms: float = 0.0
    canvas_update_ms: float = 0.0
    total_ms: float = 0.0

"""Presentation-only live reconstruction tracking diagnostics for Phase 8.8C5.

C5 deliberately derives diagnostics only from already-authoritative live SLAM
observations.  It does not invent confidence scores, metric quality, scale, or
metrology authority.  Missing native feature statistics stay explicitly
unavailable.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from math import ceil, isfinite
from statistics import median
from typing import Iterable


_TRACKING_KIND = {
    "initializing": "warning",
    "tracking": "ready",
    "degraded": "warning",
    "lost": "error",
    "recovering": "warning",
    "stopped": "neutral",
    "unknown": "neutral",
}

_TRACKING_LABEL = {
    "initializing": "INITIALIZING",
    "tracking": "TRACKING",
    "degraded": "DEGRADED",
    "lost": "LOST",
    "recovering": "RECOVERING",
    "stopped": "STOPPED",
    "unknown": "WAITING",
}


def _state_value(value: object) -> str:
    candidate = getattr(value, "value", value)
    text = str(candidate or "unknown").strip().lower()
    return text if text in _TRACKING_KIND else "unknown"


def _percentile(values: Iterable[float], percentile: float) -> float | None:
    samples = sorted(float(value) for value in values if isfinite(float(value)))
    if not samples:
        return None
    if len(samples) == 1:
        return samples[0]
    p = min(1.0, max(0.0, float(percentile)))
    rank = p * (len(samples) - 1)
    low = int(rank)
    high = min(len(samples) - 1, ceil(rank))
    if low == high:
        return samples[low]
    fraction = rank - low
    return samples[low] * (1.0 - fraction) + samples[high] * fraction


@dataclass(frozen=True, slots=True)
class LiveTrackingDiagnosticsPresentation:
    tracking_state: str = "unknown"
    tracking_label: str = "WAITING"
    status_kind: str = "neutral"
    latest_latency_ms: float | None = None
    median_latency_ms: float | None = None
    p95_latency_ms: float | None = None
    latency_sample_count: int = 0
    latency_text: str = "Waiting for samples"
    map_revision: int | None = None
    keyframe_count: int | None = None
    landmark_count: int | None = None
    map_text: str = "No live map yet"
    map_growth_text: str = "—"
    tracked_features: int | None = None
    inlier_features: int | None = None
    inlier_ratio: float | None = None
    feature_quality_available: bool = False
    feature_text: str = "Not reported by Stella"
    reason: str = ""
    observation_count: int = 0
    terminal: bool = False

    @property
    def display_tracking_label(self) -> str:
        if self.terminal and self.tracking_state != "unknown":
            return f"LAST • {self.tracking_label}"
        return self.tracking_label


class LiveTrackingDiagnosticsAccumulator:
    """Bounded C5 presentation accumulator for real live SLAM diagnostics."""

    MAX_LATENCY_SAMPLES = 120

    def __init__(self) -> None:
        self._latencies_ms: deque[float] = deque(maxlen=self.MAX_LATENCY_SAMPLES)
        self._observation_count = 0
        self._last_map: tuple[int, int, int] | None = None
        self._current = LiveTrackingDiagnosticsPresentation()

    @property
    def current(self) -> LiveTrackingDiagnosticsPresentation:
        return self._current

    @property
    def latency_sample_count(self) -> int:
        return len(self._latencies_ms)

    def reset_session(self) -> LiveTrackingDiagnosticsPresentation:
        self._latencies_ms.clear()
        self._observation_count = 0
        self._last_map = None
        self._current = LiveTrackingDiagnosticsPresentation()
        return self._current

    def mark_terminal(self, terminal: bool = True) -> LiveTrackingDiagnosticsPresentation:
        self._current = replace(self._current, terminal=bool(terminal))
        return self._current

    def update(self, observation: object) -> LiveTrackingDiagnosticsPresentation:
        self._observation_count += 1
        state = _state_value(getattr(observation, "tracking_state", "unknown"))
        quality = getattr(observation, "quality", None)

        latest_latency_ms: float | None = None
        if quality is not None:
            raw_latency = getattr(quality, "processing_latency_ns", None)
            if isinstance(raw_latency, (int, float)) and isfinite(float(raw_latency)) and raw_latency >= 0:
                latest_latency_ms = float(raw_latency) / 1_000_000.0
                self._latencies_ms.append(latest_latency_ms)

        latency_values = tuple(self._latencies_ms)
        median_ms = float(median(latency_values)) if latency_values else None
        p95_ms = _percentile(latency_values, 0.95)
        if latest_latency_ms is None:
            latency_text = "Waiting for samples"
        elif median_ms is None or p95_ms is None:
            latency_text = f"{latest_latency_ms:.1f} ms"
        else:
            latency_text = (
                f"{latest_latency_ms:.1f} ms  •  med {median_ms:.1f}  •  p95 {p95_ms:.1f}"
            )

        map_revision: int | None = None
        keyframes: int | None = None
        landmarks: int | None = None
        map_text = "No live map yet"
        map_growth_text = "—"
        snapshot = getattr(observation, "map_snapshot", None)
        if snapshot is not None:
            raw_revision = getattr(snapshot, "revision", None)
            raw_keyframes = getattr(snapshot, "keyframe_count", None)
            raw_landmarks = getattr(snapshot, "landmark_count", None)
            if isinstance(raw_revision, int) and raw_revision >= 0:
                map_revision = raw_revision
            if isinstance(raw_keyframes, int) and raw_keyframes >= 0:
                keyframes = raw_keyframes
            if isinstance(raw_landmarks, int) and raw_landmarks >= 0:
                landmarks = raw_landmarks
            if map_revision is not None and keyframes is not None and landmarks is not None:
                map_text = f"r{map_revision}  •  {keyframes} KF  •  {landmarks:,} LM"
                if self._last_map is not None and map_revision != self._last_map[0]:
                    delta_kf = keyframes - self._last_map[1]
                    delta_lm = landmarks - self._last_map[2]
                    map_growth_text = f"{delta_kf:+d} KF  •  {delta_lm:+,d} LM"
                elif self._last_map is not None:
                    map_growth_text = "No revision change"
                self._last_map = (map_revision, keyframes, landmarks)

        tracked: int | None = None
        inliers: int | None = None
        ratio: float | None = None
        feature_available = False
        feature_text = "Not reported by Stella"
        reason = ""
        if quality is not None:
            reason = str(getattr(quality, "reason", "") or "").strip()
            raw_tracked = getattr(quality, "tracked_features", None)
            raw_inliers = getattr(quality, "inlier_features", None)
            raw_ratio = getattr(quality, "inlier_ratio", None)
            if isinstance(raw_tracked, int) and raw_tracked > 0:
                tracked = raw_tracked
                inliers = int(raw_inliers) if isinstance(raw_inliers, int) and raw_inliers >= 0 else 0
                if isinstance(raw_ratio, (int, float)) and isfinite(float(raw_ratio)):
                    ratio = min(1.0, max(0.0, float(raw_ratio)))
                elif tracked > 0:
                    ratio = min(1.0, max(0.0, inliers / tracked))
                feature_available = True
                feature_text = f"{inliers}/{tracked}  •  {ratio * 100.0:.0f}% inliers"

        self._current = LiveTrackingDiagnosticsPresentation(
            tracking_state=state,
            tracking_label=_TRACKING_LABEL[state],
            status_kind=_TRACKING_KIND[state],
            latest_latency_ms=latest_latency_ms,
            median_latency_ms=median_ms,
            p95_latency_ms=p95_ms,
            latency_sample_count=len(self._latencies_ms),
            latency_text=latency_text,
            map_revision=map_revision,
            keyframe_count=keyframes,
            landmark_count=landmarks,
            map_text=map_text,
            map_growth_text=map_growth_text,
            tracked_features=tracked,
            inlier_features=inliers,
            inlier_ratio=ratio,
            feature_quality_available=feature_available,
            feature_text=feature_text,
            reason=reason,
            observation_count=self._observation_count,
            terminal=False,
        )
        return self._current

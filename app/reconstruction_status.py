"""Presentation-only Reconstruction runtime status helpers for Phase 8.8B4.

This module does not own reconstruction lifecycle. It converts already-authoritative
live/offline controller state and worker events into compact UI presentation data.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping

from core.reconstruction import LiveCaptureRuntimeState, LiveCaptureRuntimeStats


@dataclass(frozen=True, slots=True)
class ReconstructionStatusPresentation:
    """Dependency-light snapshot used by the Reconstruction page."""

    stage_label: str
    message: str
    status_kind: str = "neutral"
    progress_fraction: float | None = None
    indeterminate: bool = False


_OFFLINE_STAGE_LABELS = {
    "source": "Preparing source",
    "source_ingestion": "Ingesting source",
    "camera_pose": "Camera reconstruction",
    "cancellation": "Cancelling",
    "offline": "Offline reconstruction",
}


def format_elapsed_seconds(seconds: float) -> str:
    value = max(0, int(float(seconds)))
    hours, remainder = divmod(value, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def present_live_state(
    state: LiveCaptureRuntimeState,
    stats: LiveCaptureRuntimeStats | None = None,
) -> ReconstructionStatusPresentation:
    if not isinstance(state, LiveCaptureRuntimeState):
        raise TypeError("state must be LiveCaptureRuntimeState.")

    labels = {
        LiveCaptureRuntimeState.IDLE: ("Idle", "Live reconstruction is idle.", "neutral", False),
        LiveCaptureRuntimeState.STARTING: (
            "Starting live SLAM",
            "Launching the isolated Stella worker.",
            "running",
            True,
        ),
        LiveCaptureRuntimeState.RUNNING: (
            "Live tracking",
            "Live camera tracking is active.",
            "running",
            True,
        ),
        LiveCaptureRuntimeState.PAUSED: (
            "Paused",
            "Live reconstruction is paused; capture ownership remains with Live 3D.",
            "warning",
            False,
        ),
        LiveCaptureRuntimeState.RECOVERING: (
            "Recovering",
            "The isolated live worker is recovering within its bounded restart policy.",
            "warning",
            True,
        ),
        LiveCaptureRuntimeState.STOPPING: (
            "Stopping live SLAM",
            "Waiting for deterministic live-worker cleanup.",
            "running",
            True,
        ),
        LiveCaptureRuntimeState.STOPPED: (
            "Stopped",
            "Live reconstruction stopped cleanly.",
            "ready",
            False,
        ),
        LiveCaptureRuntimeState.FAILED: (
            "Live reconstruction failed",
            "Live reconstruction ended with an error.",
            "error",
            False,
        ),
    }
    stage, message, kind, indeterminate = labels[state]
    if stats is not None:
        if not isinstance(stats, LiveCaptureRuntimeStats):
            raise TypeError("stats must be LiveCaptureRuntimeStats or None.")
        if state is LiveCaptureRuntimeState.RUNNING and stats.latest_tracking_state:
            tracking = stats.latest_tracking_state.replace("_", " ").strip().title()
            message = f"Tracking: {tracking}."
        if state is LiveCaptureRuntimeState.RECOVERING and stats.last_error:
            message = f"Recovering after: {stats.last_error}"
        if state is LiveCaptureRuntimeState.FAILED and stats.last_error:
            message = stats.last_error
    return ReconstructionStatusPresentation(
        stage_label=stage,
        message=message,
        status_kind=kind,
        indeterminate=indeterminate,
    )


def present_offline_state(
    state: object,
) -> ReconstructionStatusPresentation:
    value = str(getattr(state, "value", state)).strip().lower()
    mapping = {
        "idle": ReconstructionStatusPresentation(
            "Idle", "Offline reconstruction is idle."
        ),
        "starting": ReconstructionStatusPresentation(
            "Preparing offline job",
            "Launching the isolated reconstruction worker.",
            "running",
            indeterminate=True,
        ),
        "running": ReconstructionStatusPresentation(
            "Offline reconstruction",
            "Waiting for the next reconstruction stage update.",
            "running",
            indeterminate=True,
        ),
        "cancelling": ReconstructionStatusPresentation(
            "Cancelling",
            "Cancellation requested; waiting for bounded worker cleanup.",
            "warning",
            indeterminate=True,
        ),
        "completed": ReconstructionStatusPresentation(
            "Completed",
            "Offline reconstruction completed.",
            "ready",
            progress_fraction=1.0,
        ),
        "failed": ReconstructionStatusPresentation(
            "Offline reconstruction failed",
            "Offline reconstruction ended with an error.",
            "error",
        ),
        "cancelled": ReconstructionStatusPresentation(
            "Cancelled",
            "Offline reconstruction was cancelled.",
            "warning",
        ),
    }
    if value not in mapping:
        raise ValueError(f"Unsupported offline reconstruction state: {value or state!r}")
    return mapping[value]


def present_offline_event(payload: Mapping[str, object]) -> ReconstructionStatusPresentation:
    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping.")
    event = str(payload.get("event", "status")).strip().lower()
    stage = str(payload.get("stage", "offline")).strip().lower()
    message = str(payload.get("message", "")).strip() or "Offline reconstruction update."
    label = _OFFLINE_STAGE_LABELS.get(stage, stage.replace("_", " ").title() or "Offline reconstruction")

    raw_progress = payload.get("progress_fraction")
    progress: float | None = None
    if raw_progress is not None:
        try:
            progress = float(raw_progress)
        except (TypeError, ValueError):
            progress = None
        if progress is not None and (not isfinite(progress) or progress < 0.0 or progress > 1.0):
            progress = None

    kind = "running"
    indeterminate = progress is None
    if event == "completed":
        kind = "ready"
        progress = 1.0
        indeterminate = False
    elif event == "failed":
        kind = "error"
        indeterminate = False
    elif event in {"cancelled", "cancellation"} or stage == "cancellation":
        kind = "warning"
        indeterminate = event not in {"cancelled"}

    return ReconstructionStatusPresentation(
        stage_label=label,
        message=message,
        status_kind=kind,
        progress_fraction=progress,
        indeterminate=indeterminate,
    )


def format_live_telemetry(
    stats: LiveCaptureRuntimeStats | None,
    *,
    pose_count: int = 0,
    keyframe_count: int = 0,
    landmark_count: int | None = None,
) -> str:
    if stats is None:
        return "No live telemetry"
    parts = [
        f"{stats.processed_frames} processed",
        f"{stats.observations} obs",
        f"{max(0, int(pose_count))} poses",
    ]
    if keyframe_count > 0:
        parts.append(f"{int(keyframe_count)} keyframes")
    if landmark_count is not None and landmark_count > 0:
        parts.append(f"{int(landmark_count):,} landmarks")
    if stats.mean_processing_latency_ns is not None:
        parts.append(f"{stats.mean_processing_latency_ns / 1_000_000.0:.1f} ms")
    return " • ".join(parts)


__all__ = [
    "ReconstructionStatusPresentation",
    "format_elapsed_seconds",
    "format_live_telemetry",
    "present_live_state",
    "present_offline_event",
    "present_offline_state",
]

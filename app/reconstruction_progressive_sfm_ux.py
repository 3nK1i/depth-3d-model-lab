"""I6.6E professional progressive sparse-SfM UX and bounded performance telemetry.

This module is dependency-light and presentation-only. It summarizes the already
validated I6.6A-D provisional sparse stream without granting final, persistent,
metric, measurement, or metrology authority. Viewer timing is the UI-thread cost
of projecting/applying one bounded preview update; it is not GPU render latency.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

from core.reconstruction.reconstruction_progressive_sfm_contract import (
    ProgressiveSfmSnapshot,
    ProgressiveSfmStage,
)

I66E_PHASE = "I6.6E"
I66E_VIEWER_UPDATE_TARGET_MS = 50.0
I66E_PREVIEW_PAYLOAD_BUDGET_BYTES = 128 * 1024


def _nonnegative_int(value: object) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, parsed)


def _run_state_value(value: object) -> str:
    raw = getattr(value, "value", value)
    return str(raw or "idle").strip().lower()


def _format_bytes(value: int) -> str:
    amount = max(0, int(value))
    if amount < 1024:
        return f"{amount} B"
    return f"{amount / 1024.0:.1f} KiB"


@dataclass(frozen=True, slots=True)
class ProgressiveSfmViewerPerformance:
    """Bounded aggregate UI-update telemetry; no per-revision history is retained."""

    applied_updates: int = 0
    rejected_updates: int = 0
    last_update_ms: float | None = None
    max_update_ms: float = 0.0
    last_payload_bytes: int = 0
    max_payload_bytes: int = 0

    def record_success(
        self,
        *,
        update_ms: float,
        payload_bytes: int,
    ) -> "ProgressiveSfmViewerPerformance":
        elapsed = float(update_ms)
        if elapsed < 0.0:
            raise ValueError("update_ms must be non-negative.")
        size = int(payload_bytes)
        if size < 0:
            raise ValueError("payload_bytes must be non-negative.")
        return replace(
            self,
            applied_updates=self.applied_updates + 1,
            last_update_ms=elapsed,
            max_update_ms=max(self.max_update_ms, elapsed),
            last_payload_bytes=size,
            max_payload_bytes=max(self.max_payload_bytes, size),
        )

    def record_rejection(self) -> "ProgressiveSfmViewerPerformance":
        return replace(self, rejected_updates=self.rejected_updates + 1)

    @property
    def update_target_met(self) -> bool:
        return self.max_update_ms <= I66E_VIEWER_UPDATE_TARGET_MS

    @property
    def payload_budget_met(self) -> bool:
        return self.max_payload_bytes <= I66E_PREVIEW_PAYLOAD_BUDGET_BYTES


@dataclass(frozen=True, slots=True)
class ProgressiveSfmUxPresentation:
    status_text: str
    status_kind: str
    stage_text: str
    cameras_text: str
    landmarks_text: str
    quality_text: str
    revision_text: str
    transport_text: str
    viewer_text: str
    payload_text: str
    authority_text: str
    boundary_text: str
    active: bool


def progressive_viewer_payload_bytes(frame: object) -> int:
    """Return the bounded NumPy payload bytes held for one I6.6D viewer frame."""

    total = 0
    for name in (
        "camera_matrices_reconstruction",
        "landmark_points_reconstruction",
        "landmark_colors_rgb",
    ):
        value = getattr(frame, name, None)
        if value is not None:
            total += int(getattr(value, "nbytes", 0))
    return total


def build_progressive_sfm_ux_presentation(
    *,
    active_snapshot: ProgressiveSfmSnapshot | None,
    last_snapshot: ProgressiveSfmSnapshot | None,
    run_state: object,
    transport_stats: Mapping[str, object] | None = None,
    performance: ProgressiveSfmViewerPerformance | None = None,
) -> ProgressiveSfmUxPresentation:
    """Build one honest product presentation over provisional sparse-SfM state."""

    if active_snapshot is not None and not isinstance(active_snapshot, ProgressiveSfmSnapshot):
        raise TypeError("active_snapshot must be ProgressiveSfmSnapshot or None.")
    if last_snapshot is not None and not isinstance(last_snapshot, ProgressiveSfmSnapshot):
        raise TypeError("last_snapshot must be ProgressiveSfmSnapshot or None.")

    perf = performance or ProgressiveSfmViewerPerformance()
    state = _run_state_value(run_state)
    snapshot = active_snapshot or last_snapshot
    active = active_snapshot is not None

    if active:
        if active_snapshot.stage is ProgressiveSfmStage.FINALIZING:
            status_text = "Finalizing provisional"
            status_kind = "warning"
            stage_text = "Finalizing"
        else:
            status_text = "Live provisional"
            status_kind = "running"
            stage_text = "Mapping"
    elif state == "starting":
        status_text = "Starting"
        status_kind = "running"
        stage_text = "Waiting for sparse mapping"
    elif state == "running":
        status_text = "Waiting for sparse map"
        status_kind = "running"
        stage_text = "Feature / matching pipeline"
    elif state == "cancelling":
        status_text = "Cancelling"
        status_kind = "warning"
        stage_text = "Clearing provisional preview"
    elif state == "completed":
        status_text = "Progressive phase complete"
        status_kind = "ready"
        stage_text = "Final sparse result admitted separately"
    elif state == "failed":
        status_text = "Stopped without admission"
        status_kind = "error"
        stage_text = "Provisional preview cleared"
    elif state == "cancelled":
        status_text = "Cancelled"
        status_kind = "warning"
        stage_text = "Provisional preview cleared"
    else:
        status_text = "Waiting for reconstruction"
        status_kind = "neutral"
        stage_text = "Not active"

    if snapshot is None:
        cameras_text = "—"
        landmarks_text = "—"
        quality_text = "—"
        revision_text = "—"
    else:
        geometry = snapshot.geometry
        camera_preview = len(geometry.camera_samples) if geometry is not None else 0
        landmark_preview = len(geometry.landmark_samples) if geometry is not None else 0
        prefix = "" if active else "Last provisional: "
        cameras_text = (
            f"{prefix}{snapshot.statistics.registered_cameras} registered"
            f" • {camera_preview} preview"
        )
        landmarks_text = (
            f"{prefix}{snapshot.statistics.sparse_landmarks:,} sparse"
            f" • {landmark_preview:,} preview"
        )
        error = snapshot.statistics.mean_reprojection_error_px
        quality_text = "Not reported" if error is None else f"{error:.3f} px reprojection"
        progress = snapshot.progress_fraction
        progress_suffix = "" if progress is None else f" • {progress * 100.0:.0f}%"
        revision_text = f"r{snapshot.revision}{progress_suffix}"

    stats = dict(transport_stats or {})
    accepted = _nonnegative_int(stats.get("accepted"))
    coalesced = _nonnegative_int(stats.get("coalesced"))
    rejected = _nonnegative_int(stats.get("rejected"))
    delivered = _nonnegative_int(stats.get("delivered"))
    transport_text = (
        f"{delivered} delivered • {coalesced} coalesced • {rejected} rejected"
    )
    if delivered == 0 and accepted > 0:
        transport_text = (
            f"{accepted} accepted • {coalesced} coalesced • {rejected} rejected"
        )

    if perf.applied_updates <= 0 or perf.last_update_ms is None:
        viewer_text = "No viewer update yet"
        payload_text = "No preview payload yet"
    else:
        target_suffix = "" if perf.update_target_met else " • above 50 ms target"
        viewer_text = (
            f"{perf.last_update_ms:.1f} ms latest • {perf.max_update_ms:.1f} ms max"
            f" • {perf.applied_updates} applied{target_suffix}"
        )
        payload_suffix = "" if perf.payload_budget_met else " • above 128 KiB budget"
        payload_text = (
            f"{_format_bytes(perf.last_payload_bytes)} latest"
            f" • {_format_bytes(perf.max_payload_bytes)} max{payload_suffix}"
        )

    return ProgressiveSfmUxPresentation(
        status_text=status_text,
        status_kind=status_kind,
        stage_text=stage_text,
        cameras_text=cameras_text,
        landmarks_text=landmarks_text,
        quality_text=quality_text,
        revision_text=revision_text,
        transport_text=transport_text,
        viewer_text=viewer_text,
        payload_text=payload_text,
        authority_text="PyCOLMAP provisional • arbitrary scale",
        boundary_text="Not final • not persistent • not metric",
        active=active,
    )


__all__ = [
    "I66E_PHASE",
    "I66E_PREVIEW_PAYLOAD_BUDGET_BYTES",
    "I66E_VIEWER_UPDATE_TARGET_MS",
    "ProgressiveSfmUxPresentation",
    "ProgressiveSfmViewerPerformance",
    "build_progressive_sfm_ux_presentation",
    "progressive_viewer_payload_bytes",
]

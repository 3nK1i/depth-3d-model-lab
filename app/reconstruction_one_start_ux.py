"""I6.2D professional UX presentation for the offline One-Start workflow.

This module is presentation-only.  It translates already-authoritative offline
runtime and result-handoff state into concise action labels/tooltips and a
terminal product message.  It does not start/cancel reconstruction, change
controller lifecycle, select providers, activate production Auto, promote
sparse SfM into canonical geometry, or claim metric/metrology authority.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.reconstruction_offline_result_handoff import OfflineReconstructionResultHandoff


@dataclass(frozen=True, slots=True)
class OfflineOneStartUxPresentation:
    start_text: str
    start_tooltip: str
    pause_tooltip: str
    stop_text: str
    stop_tooltip: str
    terminal_stage_label: str = ""
    terminal_message: str = ""
    terminal_status_kind: str = ""
    terminal_progress_fraction: float | None = None


def present_offline_one_start_ux(
    state: object,
    *,
    source_ready: bool,
    handoff: OfflineReconstructionResultHandoff | None = None,
    last_error: str = "",
) -> OfflineOneStartUxPresentation:
    """Return professional product copy for one offline reconstruction state."""

    value = str(getattr(state, "value", state)).strip().lower()
    supported = {"idle", "starting", "running", "cancelling", "completed", "failed", "cancelled"}
    if value not in supported:
        raise ValueError(f"Unsupported offline reconstruction state: {value or state!r}")
    if handoff is not None and not isinstance(handoff, OfflineReconstructionResultHandoff):
        raise TypeError("handoff must be OfflineReconstructionResultHandoff or None.")

    if value in {"starting", "running", "cancelling"}:
        return OfflineOneStartUxPresentation(
            start_text="Start",
            start_tooltip=(
                "An offline reconstruction is already active. Duplicate Start is blocked until "
                "the current session reaches a terminal state."
            ),
            pause_tooltip=(
                "Pause/resume is not available for the qualified offline reconstruction path in I6.2. "
                "Use Cancel to stop the active isolated worker safely."
            ),
            stop_text="Cancel",
            stop_tooltip=(
                "Request cancellation of the active offline reconstruction. The worker uses the "
                "existing bounded cleanup path; no completed result is fabricated."
            ),
        )

    if value == "completed":
        terminal = _completed_terminal(handoff=handoff, last_error=last_error)
        return OfflineOneStartUxPresentation(
            start_text="Run again",
            start_tooltip=(
                "Start a new isolated sparse SfM reconstruction session from the selected source. "
                "The completed session remains preserved in its own output directory."
            ) if source_ready else "Select a ready offline source before starting another reconstruction.",
            pause_tooltip="Pause/resume applies only to live reconstruction; the offline run is complete.",
            stop_text="Stop",
            stop_tooltip="No offline reconstruction is active.",
            **terminal,
        )

    if value == "cancelled":
        return OfflineOneStartUxPresentation(
            start_text="Start again",
            start_tooltip=(
                "Start a new offline sparse reconstruction session from the selected source. "
                "The cancelled session is not treated as a completed result."
            ) if source_ready else "Select a ready offline source before starting another reconstruction.",
            pause_tooltip="Pause/resume applies only to live reconstruction.",
            stop_text="Stop",
            stop_tooltip="No offline reconstruction is active.",
        )

    if value == "failed":
        return OfflineOneStartUxPresentation(
            start_text="Start",
            start_tooltip=(
                "The previous offline reconstruction failed. Review the failure/recovery guidance "
                "before starting another session."
            ),
            pause_tooltip="Pause/resume applies only to live reconstruction.",
            stop_text="Stop",
            stop_tooltip="No offline reconstruction is active.",
        )

    if source_ready:
        start_tooltip = (
            "Start one isolated sparse SfM reconstruction using the validated source. "
            "This I6.2 workflow does not run dense/canonical reconstruction or production Auto."
        )
    else:
        start_tooltip = "Select a ready image sequence or recorded video before starting reconstruction."
    return OfflineOneStartUxPresentation(
        start_text="Start",
        start_tooltip=start_tooltip,
        pause_tooltip="Pause/resume applies only to live reconstruction.",
        stop_text="Stop",
        stop_tooltip="No offline reconstruction is active.",
    )


def _completed_terminal(
    *,
    handoff: OfflineReconstructionResultHandoff | None,
    last_error: str,
) -> dict[str, object]:
    error = str(last_error).strip()
    if handoff is None:
        if error and "result handoff" in error.lower():
            return {
                "terminal_stage_label": "Result handoff rejected",
                "terminal_message": error,
                "terminal_status_kind": "error",
                "terminal_progress_fraction": 1.0,
            }
        # Preserve the historical worker-state presentation for synthetic/legacy
        # state-only transitions that do not carry I6.2C product handoff evidence.
        return {
            "terminal_stage_label": "Completed",
            "terminal_message": "Offline reconstruction completed.",
            "terminal_status_kind": "ready",
            "terminal_progress_fraction": 1.0,
        }

    if handoff.partial_result_available:
        scale = _scale_phrase(handoff.scale_status)
        if handoff.viewer_ready:
            viewer = "Sparse geometry and camera trajectory are available in the viewport."
        else:
            viewer = (
                "The scientific result remains admitted, but the sparse viewer artifact is unavailable."
            )
        warning_suffix = ""
        if handoff.warnings:
            warning_suffix = " " + " ".join(handoff.warnings)
        return {
            "terminal_stage_label": "Sparse reconstruction complete",
            "terminal_message": (
                "Sparse SfM completed and was admitted as a partial result. "
                f"{viewer} Dense/canonical reconstruction has not run. Scale: {scale}."
                f"{warning_suffix}"
            ).strip(),
            "terminal_status_kind": "warning",
            "terminal_progress_fraction": 1.0,
        }

    if handoff.persistent_scene_ready:
        return {
            "terminal_stage_label": "Reconstruction complete",
            "terminal_message": (
                "The completed reconstruction result is backed by persistent canonical scene evidence."
            ),
            "terminal_status_kind": "ready",
            "terminal_progress_fraction": 1.0,
        }

    return {
        "terminal_stage_label": "Completed • no scene",
        "terminal_message": (
            "The worker completed, but the admitted result does not expose a partial or persistent scene."
        ),
        "terminal_status_kind": "warning",
        "terminal_progress_fraction": 1.0,
    }


def _scale_phrase(value: str) -> str:
    normalized = str(value).strip().lower()
    labels = {
        "arbitrary": "arbitrary SfM gauge",
        "metric": "metric coordinates",
        "calibrated_metric": "calibrated metric",
        "unqualified": "unqualified",
    }
    return labels.get(normalized, normalized.replace("_", " ") or "unqualified")


__all__ = [
    "OfflineOneStartUxPresentation",
    "present_offline_one_start_ux",
]

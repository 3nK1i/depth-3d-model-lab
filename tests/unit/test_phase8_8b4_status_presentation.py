from __future__ import annotations

from app.reconstruction_status import (
    format_elapsed_seconds,
    format_live_telemetry,
    present_live_state,
    present_offline_event,
    present_offline_state,
)
from core.reconstruction import LiveCaptureRuntimeState, LiveCaptureRuntimeStats


def _stats(**overrides) -> LiveCaptureRuntimeStats:
    values = dict(
        offered_frames=20,
        processed_frames=18,
        replaced_frames=1,
        rejected_frames=1,
        observations=17,
        last_sequence_index=19,
        latest_tracking_state="tracking",
        last_error="",
        state=LiveCaptureRuntimeState.RUNNING,
        mean_processing_latency_ns=12_500_000.0,
    )
    values.update(overrides)
    return LiveCaptureRuntimeStats(**values)


def test_b4_elapsed_formatter_is_compact_and_stable() -> None:
    assert format_elapsed_seconds(0) == "00:00"
    assert format_elapsed_seconds(65.9) == "01:05"
    assert format_elapsed_seconds(3661) == "01:01:01"


def test_b4_live_running_is_ongoing_not_fake_percentage() -> None:
    presentation = present_live_state(LiveCaptureRuntimeState.RUNNING, _stats())
    assert presentation.stage_label == "Live tracking"
    assert presentation.indeterminate is True
    assert presentation.progress_fraction is None
    assert "Tracking" in presentation.message


def test_b4_live_failure_surfaces_authoritative_runtime_error() -> None:
    presentation = present_live_state(
        LiveCaptureRuntimeState.FAILED,
        _stats(state=LiveCaptureRuntimeState.FAILED, last_error="worker protocol failed"),
    )
    assert presentation.status_kind == "error"
    assert presentation.message == "worker protocol failed"


def test_b4_offline_completed_is_finite_100_percent() -> None:
    presentation = present_offline_state("completed")
    assert presentation.stage_label == "Completed"
    assert presentation.progress_fraction == 1.0
    assert presentation.indeterminate is False


def test_b4_offline_event_maps_worker_stage_and_fraction_without_global_guessing() -> None:
    presentation = present_offline_event(
        {
            "event": "progress",
            "stage": "source_ingestion",
            "message": "Selected 50 reconstruction keyframes.",
            "progress_fraction": 0.75,
        }
    )
    assert presentation.stage_label == "Ingesting source"
    assert presentation.progress_fraction == 0.75
    assert presentation.message == "Selected 50 reconstruction keyframes."


def test_b4_invalid_worker_progress_becomes_indeterminate_instead_of_lie() -> None:
    presentation = present_offline_event(
        {
            "event": "progress",
            "stage": "camera_pose",
            "message": "Matching features.",
            "progress_fraction": 4.2,
        }
    )
    assert presentation.stage_label == "Camera reconstruction"
    assert presentation.progress_fraction is None
    assert presentation.indeterminate is True


def test_b4_cancelled_event_is_warning_not_failure() -> None:
    presentation = present_offline_event(
        {"event": "cancelled", "stage": "cancellation", "message": "Cancelled."}
    )
    assert presentation.status_kind == "warning"
    assert presentation.stage_label == "Cancelling"


def test_b4_live_telemetry_is_compact_and_uses_runtime_stats() -> None:
    text = format_live_telemetry(
        _stats(),
        pose_count=16,
        keyframe_count=4,
        landmark_count=1234,
    )
    assert "18 processed" in text
    assert "17 obs" in text
    assert "16 poses" in text
    assert "4 keyframes" in text
    assert "1,234 landmarks" in text
    assert "12.5 ms" in text

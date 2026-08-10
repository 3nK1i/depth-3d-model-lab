from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.main_window import MainWindow
from app.reconstruction_controller import ReconstructionSourceSelection, ReconstructionWorkflow
from app.offline_reconstruction_controller import OfflineReconstructionRunState
from app.widgets.mode_tabs import AppMode
from core.reconstruction import LiveCaptureRuntimeState, LiveCaptureRuntimeStats


@pytest.fixture
def window(qtbot, monkeypatch):
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    win = MainWindow()
    qtbot.addWidget(win)
    yield win
    win.close()


def _images(tmp_path: Path) -> Path:
    root = tmp_path / "images"
    root.mkdir()
    for index in range(4):
        (root / f"frame-{index:02d}.jpg").write_bytes(b"image")
    return root


def _stats() -> LiveCaptureRuntimeStats:
    return LiveCaptureRuntimeStats(
        offered_frames=22,
        processed_frames=20,
        replaced_frames=1,
        rejected_frames=1,
        observations=19,
        last_sequence_index=21,
        latest_tracking_state="tracking",
        last_error="",
        state=LiveCaptureRuntimeState.RUNNING,
        mean_processing_latency_ns=15_200_000.0,
    )


def test_b4_initial_runtime_presentation_waits_for_source(window) -> None:
    page = window._reconstruction_page
    assert page._stage_value.text() == "Waiting for source"
    assert page._elapsed_value.text() == "—"
    assert page._progress_bar.maximum() == 100
    assert page._runtime_message.text() == "Choose a Reconstruction workflow to begin."


def test_b4_ready_source_presents_ready_to_start_without_launching(window, tmp_path: Path) -> None:
    page = window._reconstruction_page
    controller = window._reconstruction_controller
    controller.select_image_sequence(_images(tmp_path))

    assert page._stage_value.text() == "Ready to start"
    assert page._stage_value.property("statusKind") == "ready"
    assert controller.offline_is_running is False


def test_b4_offline_worker_event_updates_stage_fraction_and_message(window) -> None:
    page = window._reconstruction_page
    controller = window._reconstruction_controller
    controller._offline_controller._state = OfflineReconstructionRunState.RUNNING
    controller._offline_controller._run_finalized = False
    controller.offline_state_changed.emit(OfflineReconstructionRunState.RUNNING)
    controller.offline_event_received.emit(
        {
            "event": "progress",
            "stage": "camera_pose",
            "message": "Registering sparse cameras.",
            "progress_fraction": 0.42,
        }
    )

    assert page._stage_value.text() == "Camera reconstruction"
    assert page._progress_bar.maximum() == 100
    assert page._progress_bar.value() == 42
    assert page._runtime_message.text() == "Registering sparse cameras."


def test_b4_offline_completion_is_100_percent_and_freezes_elapsed(window, qtbot) -> None:
    page = window._reconstruction_page
    controller = window._reconstruction_controller
    controller._offline_controller._state = OfflineReconstructionRunState.STARTING
    controller._offline_controller._run_finalized = False
    controller.offline_state_changed.emit(OfflineReconstructionRunState.STARTING)
    qtbot.wait(20)
    controller._offline_controller._state = OfflineReconstructionRunState.COMPLETED
    controller._offline_controller._run_finalized = True
    controller.offline_state_changed.emit(OfflineReconstructionRunState.COMPLETED)

    assert page._stage_value.text() == "Completed"
    assert page._progress_bar.maximum() == 100
    assert page._progress_bar.value() == 100
    assert page._elapsed_value.text() != "—"


def test_b4_offline_result_summary_is_compact_not_raw_backend_log(window) -> None:
    page = window._reconstruction_page
    window._reconstruction_controller.offline_result_ready.emit(
        {
            "sfm_summary": {
                "registered_keyframes": 50,
                "input_keyframes": 50,
                "landmarks": 28435,
                "mean_reprojection_error_px": 0.4948,
            }
        },
        "/tmp/result",
    )

    assert page._telemetry_value.text() == "50/50 keyframes • 28,435 landmarks • 0.49 px"
    assert "Traceback" not in page._telemetry_value.text()


def test_b4_live_runtime_uses_indeterminate_progress_and_compact_telemetry(window) -> None:
    page = window._reconstruction_page
    controller = window._reconstruction_controller
    fake_runtime = SimpleNamespace(is_active=True, stats=_stats())
    controller.live_slam_controller._runtime = fake_runtime
    page.set_live_observation(SimpleNamespace(pose=object(), is_keyframe=True, map_snapshot=None))
    controller.live_state_changed.emit(LiveCaptureRuntimeState.RUNNING)
    page._refresh_runtime_telemetry()

    assert page._stage_value.text() == "Live tracking"
    assert page._progress_bar.minimum() == 0
    assert page._progress_bar.maximum() == 0
    assert "20 processed" in page._telemetry_value.text()
    assert "1 poses" in page._telemetry_value.text()
    controller.live_slam_controller._runtime = None


def test_b4_controller_error_is_user_facing_after_b5_navigation_enablement(window) -> None:
    page = window._reconstruction_page
    window._reconstruction_controller.error_occurred.emit("External worker failed safely.")

    assert page._runtime_message.text() == "External worker failed safely."
    assert page._runtime_message.property("statusKind") == "error"
    assert window._mode_tabs._buttons[AppMode.RECONSTRUCTION].isEnabled() is True


def test_b4_status_timer_is_ui_only_with_b5_navigation_available(window) -> None:
    page = window._reconstruction_page
    assert page._telemetry_timer.isActive() is True
    assert page._telemetry_timer.interval() == 500
    assert window._mode_tabs._buttons[AppMode.RECONSTRUCTION].isEnabled() is True


def test_b4_terminal_failure_preserves_specific_controller_error(window) -> None:
    page = window._reconstruction_page
    controller = window._reconstruction_controller
    controller.error_occurred.emit("PyCOLMAP worker exited before producing a result.")
    controller._offline_controller._state = OfflineReconstructionRunState.FAILED
    controller._offline_controller._run_finalized = True
    controller.offline_state_changed.emit(OfflineReconstructionRunState.FAILED)

    assert page._stage_value.text() == "Offline reconstruction failed"
    assert page._runtime_message.text() == "PyCOLMAP worker exited before producing a result."
    assert page._runtime_message.property("statusKind") == "error"


def test_b4_live_stopped_stage_survives_post_stop_source_disconnect_refresh(window) -> None:
    page = window._reconstruction_page
    controller = window._reconstruction_controller
    controller._set_source_selection(
        ReconstructionSourceSelection(
            workflow=ReconstructionWorkflow.LIVE_SOURCE,
            ready=True,
            display_name="simulated-live.avi",
            detail="Recorded video as simulated live • Connected",
            live_source_kind="recorded_video",
        )
    )
    controller.live_state_changed.emit(LiveCaptureRuntimeState.STOPPED)
    assert page._stage_value.text() == "Stopped"

    controller._set_source_selection(
        ReconstructionSourceSelection(
            workflow=ReconstructionWorkflow.LIVE_SOURCE,
            ready=False,
            display_name="No compatible Live 3D source",
            detail="Connect a Camera, RTSP stream, or Video in Live 3D first.",
            live_source_kind="none",
        )
    )

    assert page._source_readiness_value.text() == "Needs attention"
    assert page._stage_value.text() == "Stopped"
    assert page._runtime_message.text() == "Live reconstruction stopped cleanly."

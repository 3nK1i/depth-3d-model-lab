from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.main_window import MainWindow
from app.offline_reconstruction_controller import OfflineReconstructionRunState
from app.reconstruction_controller import ReconstructionWorkflow
from app.widgets.mode_tabs import AppMode
from core.capture.base_source import SourceStatus
from core.capture.camera_source import CameraSource
from core.reconstruction import (
    CameraIntrinsicsPolicy,
    LiveCaptureRuntimeState,
    live_camera_calibration_dict,
)
from tests.reconstruction_live_helpers import make_calibrated_live_camera


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


def test_b3_ready_source_enables_start_but_navigation_stays_locked(window, tmp_path: Path) -> None:
    page = window._reconstruction_page
    controller = window._reconstruction_controller
    controller.select_image_sequence(_images(tmp_path))

    assert page._start_button.isEnabled() is True
    assert page._pause_button.isEnabled() is False
    assert page._stop_button.isEnabled() is False
    assert window._mode_tabs._buttons[AppMode.RECONSTRUCTION].isEnabled() is False
    assert controller.is_active is False


def test_b3_offline_runtime_button_states_are_start_cancel_and_restart_safe(window, tmp_path: Path) -> None:
    page = window._reconstruction_page
    controller = window._reconstruction_controller
    controller.select_image_sequence(_images(tmp_path))

    controller._offline_controller._state = OfflineReconstructionRunState.RUNNING
    controller._offline_controller._run_finalized = False
    controller.offline_state_changed.emit(OfflineReconstructionRunState.RUNNING)

    assert page._start_button.isEnabled() is False
    assert page._pause_button.isEnabled() is False
    assert page._stop_button.isEnabled() is True
    assert page._stop_button.text() == "Cancel"
    assert all(not button.isEnabled() for button in page._source_buttons.values())

    controller._offline_controller._state = OfflineReconstructionRunState.COMPLETED
    controller._offline_controller._run_finalized = True
    controller.offline_state_changed.emit(OfflineReconstructionRunState.COMPLETED)

    assert page._start_button.isEnabled() is True
    assert page._stop_button.isEnabled() is False
    assert all(button.isEnabled() for button in page._source_buttons.values())


def test_b3_offline_start_button_delegates_to_selected_controller_workflow(
    window, monkeypatch, tmp_path: Path
) -> None:
    page = window._reconstruction_page
    controller = window._reconstruction_controller
    controller.select_image_sequence(_images(tmp_path))
    calls: list[str] = []
    monkeypatch.setattr(
        controller,
        "start_selected_offline",
        lambda: calls.append("offline") or "00000000-0000-4000-8000-0000000088b3",
    )

    page._start_button.click()

    assert calls == ["offline"]
    assert page._session_value.text() == "00000000-0000-4000-8000-0000000088b3"
    assert page._runtime_output_value.text() == "Preparing output"


def test_b3_live_start_calibrated_mode_delegates_to_live_bridge(
    window, monkeypatch, tmp_path: Path
) -> None:
    page = window._reconstruction_page
    controller = window._reconstruction_controller
    source = CameraSource(0, width=1280, height=720, fps=30)
    source._status = SourceStatus.CONNECTED
    window._controller._active_source = source
    controller.select_live_source()

    calibration = tmp_path / "camera-calibration.json"
    calibration.write_text(
        json.dumps(live_camera_calibration_dict(make_calibrated_live_camera())),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "app.widgets.reconstruction_page.QFileDialog.getOpenFileName",
        lambda *_args, **_kwargs: (str(calibration), "Camera calibration"),
    )
    monkeypatch.setattr(page, "_choose_live_camera_mode", lambda: "calibrated")
    captured = []

    def fake_start_selected_live(*, camera):
        captured.append(camera)
        return "00000000-0000-4000-8000-00000000b3b3"

    monkeypatch.setattr(controller, "start_selected_live", fake_start_selected_live)
    page._start_button.click()

    assert len(captured) == 1
    assert captured[0].intrinsics.calibration_status.value == "calibrated"
    assert page._runtime_prerequisite_value.text() == calibration.name
    assert page._camera_geometry_value.text() == "Calibrated"
    assert page._session_value.text() == "00000000-0000-4000-8000-00000000b3b3"


def test_b3_live_start_test_mode_uses_explicit_estimated_policy(window, monkeypatch) -> None:
    page = window._reconstruction_page
    controller = window._reconstruction_controller
    source = CameraSource(0, width=1280, height=720, fps=30)
    source._status = SourceStatus.CONNECTED
    window._controller._active_source = source
    controller.select_live_source()
    monkeypatch.setattr(page, "_choose_live_camera_mode", lambda: "estimated")
    captured: dict[str, object] = {}

    def fake_start_selected_live(*, camera, camera_intrinsics_policy):
        captured["camera"] = camera
        captured["policy"] = camera_intrinsics_policy
        return "00000000-0000-4000-8000-00000000b3e3"

    monkeypatch.setattr(controller, "start_selected_live", fake_start_selected_live)
    page._start_button.click()

    camera = captured["camera"]
    assert camera.intrinsics.source.value == "estimated"
    assert camera.intrinsics.calibration_status.value == "not_calibrated"
    assert captured["policy"] is CameraIntrinsicsPolicy.ALLOW_ESTIMATED
    assert page._runtime_prerequisite_value.text() == "Estimated intrinsics • Test mode"
    assert page._camera_geometry_value.text() == "Estimated / test"
    assert page._metric_scale_value.text() == "Unavailable (test)"
    assert page._metrology_value.text() == "Not certified"
    assert page._readiness_badge.text() == "TEST MODE • NON-METRIC"
    assert page._session_value.text() == "00000000-0000-4000-8000-00000000b3e3"


def test_b3_live_start_mode_cancel_never_starts_backend(window, monkeypatch) -> None:
    page = window._reconstruction_page
    controller = window._reconstruction_controller
    source = CameraSource(0, width=1280, height=720, fps=30)
    source._status = SourceStatus.CONNECTED
    window._controller._active_source = source
    controller.select_live_source()
    monkeypatch.setattr(page, "_choose_live_camera_mode", lambda: None)
    calls: list[str] = []
    monkeypatch.setattr(
        controller,
        "start_selected_live",
        lambda **_kwargs: calls.append("started") or "unused",
    )

    page._start_button.click()

    assert calls == []
    assert controller.live_slam_controller.runtime is None


def test_b3_live_pause_resume_and_stop_delegate_without_blocking_ui(window, monkeypatch) -> None:
    page = window._reconstruction_page
    controller = window._reconstruction_controller
    source = CameraSource(0, width=1280, height=720, fps=30)
    source._status = SourceStatus.CONNECTED
    window._controller._active_source = source
    controller.select_live_source()

    calls: list[str] = []
    monkeypatch.setattr(controller, "pause_live", lambda: calls.append("pause") or True)
    monkeypatch.setattr(controller, "resume_live", lambda: calls.append("resume") or True)
    monkeypatch.setattr(controller, "request_stop_live", lambda: calls.append("stop") or True)

    page.set_live_state(LiveCaptureRuntimeState.RUNNING)
    assert page._pause_button.text() == "Pause"
    assert page._pause_button.isEnabled() is True
    assert page._stop_button.isEnabled() is True
    page._pause_button.click()

    page.set_live_state(LiveCaptureRuntimeState.PAUSED)
    assert page._pause_button.text() == "Resume"
    page._pause_button.click()
    page._stop_button.click()

    assert calls == ["pause", "resume", "stop"]


def test_b3_source_selection_alone_still_never_starts_any_backend(window, tmp_path: Path) -> None:
    controller = window._reconstruction_controller
    controller.select_image_sequence(_images(tmp_path))

    assert controller.live_slam_controller.runtime is None
    assert controller.offline_is_running is False
    assert controller.offline_state is OfflineReconstructionRunState.IDLE

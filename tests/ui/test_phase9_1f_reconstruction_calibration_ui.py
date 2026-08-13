from __future__ import annotations

from pathlib import Path

import pytest

from app.main_window import MainWindow
from core.capture.base_source import SourceStatus
from core.capture.camera_source import CameraSource
from core.reconstruction.calibration_management import (
    CameraCalibrationProfileStore,
    create_manual_camera_calibration_profile,
)
from core.reconstruction.calibration_profile import CameraCalibrationValidationState
from app.widgets.mode_tabs import AppMode


@pytest.fixture
def window(qtbot, monkeypatch):
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    win = MainWindow()
    qtbot.addWidget(win)
    yield win
    win.close()


def _profile_1080p():
    return create_manual_camera_calibration_profile(
        width=1920,
        height=1080,
        fx=1400.0,
        fy=1404.0,
        cx=960.0,
        cy=540.0,
        camera_id="phase9-camera",
        profile_id="phase9-profile",
        calibration_record_id="phase9-record",
    )


def test_1f_reconstruction_keeps_existing_mode_count_and_adds_calibration_actions(window) -> None:
    page = window._reconstruction_page
    assert window._mode_stack.count() == 6
    assert page._calibration_manage_button.text() == "Manage Calibration"
    assert page._calibration_clear_button.text() == "Clear"
    assert page._calibration_profile_value.text() == "No profile selected"
    assert page._metric_scale_value.text() == "Evidence-driven"


def test_1f_selected_profile_updates_authority_without_metric_promotion(window) -> None:
    page = window._reconstruction_page
    profile = _profile_1080p()
    page.set_camera_calibration_profile(profile)

    assert page.selected_camera_calibration_profile == profile
    assert page._calibration_profile_value.text() == "phase9-camera"
    assert page._calibration_resolution_value.text() == "1920 × 1080"
    assert "Unvalidated" in page._camera_geometry_value.text()
    assert page._metric_scale_value.text() == "Not qualified"
    assert page._metrology_value.text() == "Not certified"
    assert page._calibration_clear_button.isEnabled()


def test_1f_selected_profile_is_safely_rescaled_for_live_start(window, monkeypatch) -> None:
    page = window._reconstruction_page
    controller = window._reconstruction_controller
    source = CameraSource(0, width=1280, height=720, fps=30)
    source._status = SourceStatus.CONNECTED
    window._controller._active_source = source
    controller.select_live_source()
    page.set_camera_calibration_profile(_profile_1080p())

    captured = []

    def fake_start_selected_live(*, camera):
        captured.append(camera)
        return "00000000-0000-4000-8000-0000000091f0"

    monkeypatch.setattr(controller, "start_selected_live", fake_start_selected_live)
    page._start_button.click()

    assert len(captured) == 1
    camera = captured[0]
    assert camera.image_size.width == 1280
    assert camera.image_size.height == 720
    assert camera.intrinsics.matrix[0, 0] == pytest.approx(1400.0 * (1280.0 / 1920.0))
    assert camera.intrinsics.matrix[1, 1] == pytest.approx(1404.0 * (720.0 / 1080.0))
    assert "safely rescaled" in page._runtime_prerequisite_value.text()
    assert page._metric_scale_value.text() == "Not qualified"
    assert page._metrology_value.text() == "Not certified"


def test_1f_clear_profile_does_not_delete_reusable_library(window, tmp_path: Path) -> None:
    page = window._reconstruction_page
    profile = _profile_1080p()
    page._calibration_profile_store = CameraCalibrationProfileStore(tmp_path / "profiles")
    page._calibration_profile_store.save(profile)
    page.set_camera_calibration_profile(profile)

    page.clear_camera_calibration_profile()

    assert page.selected_camera_calibration_profile is None
    assert page._calibration_profile_store.contains(profile.profile_id)
    assert page._calibration_profile_value.text() == "No profile selected"


def test_1f_reconstruction_sidebar_remains_vertical_only_at_1366x768(window, qtbot) -> None:
    window._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)
    page = window._reconstruction_page
    window.resize(1366, 768)
    window.show()
    qtbot.wait(20)

    assert page._settings_scroll.horizontalScrollBar().maximum() == 0
    assert page._settings_scroll.horizontalScrollBarPolicy().name == "ScrollBarAlwaysOff"

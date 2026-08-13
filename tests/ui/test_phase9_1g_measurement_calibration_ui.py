from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from app.main_window import MainWindow
from app.measurement_inspection_controller import MeasurementInspectionController
from app.widgets.inspection_page import InspectionPage
from core.capture.base_source import SourceStatus
from core.capture.camera_source import CameraSource
from core.reconstruction.calibration_management import create_manual_camera_calibration_profile
from core.reconstruction.calibration_profile import CameraCalibrationValidationState


def _profile():
    return create_manual_camera_calibration_profile(
        width=640,
        height=480,
        fx=520.0,
        fy=522.0,
        cx=320.0,
        cy=240.0,
        camera_id="phase9-1g-camera",
        profile_id="phase9-1g-profile",
        calibration_record_id="phase9-1g-record",
    )


@pytest.fixture
def window(qtbot, monkeypatch):
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    win = MainWindow()
    qtbot.addWidget(win)
    yield win
    win.close()


def test_1g_inspection_page_presents_shared_calibration_without_metric_promotion(qtbot) -> None:
    page = InspectionPage()
    qtbot.addWidget(page)
    page.set_camera_calibration_context(_profile())

    assert "phase9-1g-profile" in page._calibration_state.text()
    assert "Unvalidated" in page._calibration_state.text()
    assert "Metric scale remains separate" in page._calibration_detail.text()


def test_1g_inspection_manage_button_requests_shared_workspace(qtbot) -> None:
    page = InspectionPage()
    qtbot.addWidget(page)
    emitted = []
    page.calibration_requested.connect(lambda: emitted.append(True))
    # The button is the only secondary button inside the calibration card with this text.
    button = next(value for value in page.findChildren(type(page._capture_btn)) if value.text() == "Manage Calibration")
    button.click()
    assert emitted == [True]


def test_1g_reconstruction_selection_is_visible_in_measure_and_inspect(window) -> None:
    profile = _profile()
    window._reconstruction_page.set_camera_calibration_profile(profile)
    window._refresh_inspection_context()

    assert window._reconstruction_page.selected_camera_calibration_profile is profile
    assert window._inspection_page._camera_calibration_profile is profile
    assert "phase9-1g-profile" in window._inspection_page._calibration_state.text()


def test_1g_measurement_controller_serializes_selected_profile(qtbot, monkeypatch) -> None:
    controller = MeasurementInspectionController(Path.cwd())
    captured = {}
    monkeypatch.setattr(
        controller,
        "_start_process",
        lambda args, state_value: captured.update(args=args, state=state_value),
    )
    controller.start_analysis(
        image_bgr=np.zeros((48, 64, 3), dtype=np.uint8),
        depth_model_id="depth-pro",
        depth_config={},
        perception_model_id="development-perception-mock",
        depth_profile="efficient",
        perception_profile="efficient",
        max_objects=1,
        calibration_profile=_profile(),
    )
    args = captured["args"]
    assert "--calibration-profile" in args
    path = Path(args[args.index("--calibration-profile") + 1])
    assert path.is_file()
    assert "phase9-1g-profile" in path.read_text(encoding="utf-8")
    controller.shutdown()


def test_1g_measurement_controller_rejects_rejected_profile(qtbot) -> None:
    controller = MeasurementInspectionController(Path.cwd())
    rejected = replace(
        _profile(),
        validation_state=CameraCalibrationValidationState.REJECTED,
    )
    with pytest.raises(ValueError, match="Rejected camera-calibration"):
        controller.start_analysis(
            image_bgr=np.zeros((48, 64, 3), dtype=np.uint8),
            depth_model_id="depth-pro",
            depth_config={},
            perception_model_id="development-perception-mock",
            depth_profile="efficient",
            perception_profile="efficient",
            max_objects=1,
            calibration_profile=rejected,
        )
    controller.shutdown()


def test_1g_current_application_source_context_does_not_require_reconstruction_workflow(window) -> None:
    source = CameraSource(2, width=1280, height=720, fps=30)
    source._status = SourceStatus.CONNECTED
    window._controller._active_source = source

    size = window._reconstruction_controller.current_application_image_size()
    assert window._reconstruction_controller.current_application_source_label() == "USB camera 2"
    assert size.width == 1280
    assert size.height == 720

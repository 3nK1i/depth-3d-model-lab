from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QScrollArea

from app.dialogs.camera_calibration_dialog import (
    CameraCalibrationDialog,
    MODE_IMPORT,
    MODE_MANUAL,
)
from core.reconstruction.calibration_management import CameraCalibrationProfileStore
from core.reconstruction.calibration_profile import (
    CameraCalibrationMethod,
    CameraCalibrationValidationState,
)
from core.reconstruction.live_calibration import live_camera_calibration_dict
from tests.reconstruction_live_helpers import make_calibrated_live_camera


def _dialog(qtbot, tmp_path: Path) -> CameraCalibrationDialog:
    dialog = CameraCalibrationDialog(
        profile_store=CameraCalibrationProfileStore(tmp_path / "profiles"),
        source_label="Camera 0",
        target_resolution_text="1280 × 720",
        metric_status_text="Not qualified",
        allow_test_mode=True,
    )
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.wait(5)
    return dialog


def test_1f_manual_entry_is_explicit_and_remains_unvalidated(qtbot, tmp_path: Path) -> None:
    dialog = _dialog(qtbot, tmp_path)
    dialog._set_mode(MODE_MANUAL)
    dialog._manual_width.setValue(1280)
    dialog._manual_height.setValue(720)
    dialog._manual_fx.setValue(900.0)
    dialog._manual_fy.setValue(905.0)
    dialog._manual_cx.setValue(640.0)
    dialog._manual_cy.setValue(360.0)
    dialog._build_manual_profile()

    profile = dialog.selected_profile
    assert profile is not None
    assert profile.method is CameraCalibrationMethod.MANUAL
    assert profile.validation_state is CameraCalibrationValidationState.UNVALIDATED
    assert profile.calibration_image_size.width == 1280
    assert profile.calibration_image_size.height == 720
    assert dialog._metric_value.text() == "Not qualified"
    assert dialog._use_button.isEnabled()


def test_1f_manual_profile_can_be_saved_and_reloaded(qtbot, tmp_path: Path) -> None:
    dialog = _dialog(qtbot, tmp_path)
    dialog._set_mode(MODE_MANUAL)
    dialog._manual_width.setValue(640)
    dialog._manual_height.setValue(480)
    dialog._manual_fx.setValue(520.0)
    dialog._manual_fy.setValue(521.0)
    dialog._manual_cx.setValue(320.0)
    dialog._manual_cy.setValue(240.0)
    dialog._build_manual_profile()
    profile = dialog.selected_profile
    assert profile is not None

    dialog._save_current_profile()
    assert dialog.profile_store.contains(profile.profile_id)
    saved = dialog.profile_store.list_profiles()
    assert len(saved) == 1
    # CameraIntrinsics contains a NumPy matrix, so dataclass ``==`` is not a
    # valid round-trip assertion (NumPy elementwise equality has ambiguous
    # truth semantics). Compare the canonical persisted representation instead.
    assert saved[0].as_dict() == profile.as_dict()


def test_1f_legacy_phase8_import_preserves_legacy_evidence(qtbot, tmp_path: Path) -> None:
    dialog = _dialog(qtbot, tmp_path)
    legacy = tmp_path / "legacy-camera.json"
    legacy.write_text(
        json.dumps(live_camera_calibration_dict(make_calibrated_live_camera())),
        encoding="utf-8",
    )
    dialog._set_mode(MODE_IMPORT)
    dialog._import_path.setText(str(legacy))
    dialog._import_profile()

    profile = dialog.selected_profile
    assert profile is not None
    assert profile.method is CameraCalibrationMethod.LEGACY_LIVE_JSON
    assert profile.validation_state is CameraCalibrationValidationState.LEGACY_COMPATIBLE
    assert profile.has_phase9_validation_evidence is False



def test_1f_rejected_profile_can_be_saved_but_not_used(qtbot, tmp_path: Path) -> None:
    dialog = _dialog(qtbot, tmp_path)
    dialog._set_mode(MODE_MANUAL)
    dialog._manual_width.setValue(640)
    dialog._manual_height.setValue(480)
    dialog._manual_fx.setValue(520.0)
    dialog._manual_fy.setValue(521.0)
    dialog._manual_cx.setValue(320.0)
    dialog._manual_cy.setValue(240.0)
    dialog._build_manual_profile()
    assert dialog.selected_profile is not None
    dialog._current_profile = replace(
        dialog.selected_profile,
        validation_state=CameraCalibrationValidationState.REJECTED,
    )
    dialog._refresh_profile_presentation()

    assert dialog._save_button.isEnabled()
    assert not dialog._use_button.isEnabled()


def test_1f_dialog_keeps_metric_scale_visibly_separate(qtbot, tmp_path: Path) -> None:
    dialog = _dialog(qtbot, tmp_path)
    assert dialog._calibration_value.text() == "No profile selected"
    assert dialog._metric_value.text() == "Not qualified"
    assert "metric scale" in dialog._status_message.text().lower()
    assert dialog._test_mode_button is not None
    assert dialog._test_mode_button.toolTip()


def test_1f_dialog_scrolls_vertically_without_horizontal_overflow(qtbot, tmp_path: Path) -> None:
    dialog = _dialog(qtbot, tmp_path)
    dialog.resize(760, 560)
    qtbot.wait(10)
    for index in range(dialog._stack.count()):
        host = dialog._stack.widget(index)
        # Each page owns one QScrollArea; inspect it by object name to avoid
        # depending on layout implementation details.
        scroll = host.findChild(QScrollArea, "cameraCalibrationScroll")
        assert scroll is not None
        assert scroll.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff

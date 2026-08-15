from __future__ import annotations

from pathlib import Path

import pytest
from PyQt6.QtCore import Qt

from app.da3_canonical_integration_controller import DA3CanonicalRunState
from app.main_window import MainWindow
from app.widgets.mode_tabs import AppMode
from core.reconstruction.da3 import DA3_PROVIDER_ID
from tests.phase9_3h_helpers import integrate_phase9_3h_fixture, write_ready_image_sequence


@pytest.fixture
def window(qtbot, monkeypatch):
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    win = MainWindow()
    qtbot.addWidget(win)
    win._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)
    yield win
    win.close()


def test_3h_explicit_integrate_action_is_enabled_without_calibration_and_does_not_auto_start(
    window, qtbot, monkeypatch, tmp_path: Path
) -> None:
    controller = window._reconstruction_controller
    page = window._reconstruction_page
    assert page.selected_camera_calibration_profile is None
    controller.select_image_sequence(write_ready_image_sequence(tmp_path))
    qtbot.wait(10)

    assert page._da3_integrate_button.isEnabled() is True
    assert page._da3_cancel_button.isEnabled() is False
    assert "calibration optional" in page._da3_job_value.text().lower()
    assert controller.da3_canonical_state is DA3CanonicalRunState.IDLE
    assert controller.is_active is False

    calls: list[bool] = []
    monkeypatch.setattr(controller, "start_selected_da3_canonical", lambda **_kwargs: calls.append(True) or "sid")
    qtbot.mouseClick(page._da3_integrate_button, Qt.MouseButton.LeftButton)
    assert calls == [True]
    assert controller.is_active is False


def test_3h_provider_selection_remains_inspection_only_while_explicit_button_is_separate(
    window, qtbot, monkeypatch, tmp_path: Path
) -> None:
    controller = window._reconstruction_controller
    page = window._reconstruction_page
    controller.select_image_sequence(write_ready_image_sequence(tmp_path / "source"))
    _integration, scene_manifest, _root = integrate_phase9_3h_fixture(tmp_path / "scene")
    controller.load_persistent_scene(scene_manifest)
    qtbot.wait(10)

    calls: list[bool] = []
    monkeypatch.setattr(controller, "start_selected_da3_canonical", lambda **_kwargs: calls.append(True) or "sid")
    index = page._dense_provider_combo.findData(DA3_PROVIDER_ID)
    assert index >= 0
    page._dense_provider_combo.setCurrentIndex(index)
    qtbot.wait(5)

    assert calls == []
    entry = page.selected_dense_provider_entry
    assert entry is not None and entry.scene_bound is True
    assert "DA3 support bound" in page._dense_integration_value.text()
    assert "Supporting geometry only" in page._dense_authority_value.text()
    assert page._da3_integrate_button.isEnabled() is True


def test_3h_loaded_da3_scene_is_canonical_arbitrary_scale_and_semantics_remain_unavailable(
    window, qtbot, tmp_path: Path
) -> None:
    controller = window._reconstruction_controller
    page = window._reconstruction_page
    _integration, scene_manifest, _root = integrate_phase9_3h_fixture(tmp_path)
    scene = controller.load_persistent_scene(scene_manifest)
    qtbot.wait(20)

    assert scene.canonical_geometry_authoritative is True
    assert scene.certified_metrology is False
    assert page._canonical_scene_scale_value.text() == "Not qualified"
    assert page._reconstruction_canvas.canonical_mesh_loaded is True
    assert page._canonical_camera_presentation is not None
    assert page._canonical_camera_presentation.persisted_pose_count == 3
    assert page._semantic_object_combo.isEnabled() is False
    assert "unavailable" in page._semantic_structure_value.text().lower()
    assert page._inspection_scope_value.text() == "Scene"


def test_3h_da3_activity_uses_existing_application_resource_interlock(window, qtbot) -> None:
    controller = window._reconstruction_controller
    page = window._reconstruction_page
    da3 = controller.da3_canonical_controller

    da3._run_finalized = False
    da3._state = DA3CanonicalRunState.RUNNING
    controller.da3_canonical_state_changed.emit(DA3CanonicalRunState.RUNNING)
    qtbot.wait(5)
    assert controller.is_active is True
    assert page._da3_runtime_value.text() == "Running"
    assert page._stage_value.text() == "DA3 canonical integration"
    assert page._progress_bar.maximum() == 0
    for mode, button in window._mode_tabs._buttons.items():
        assert button.isEnabled() is (mode is AppMode.RECONSTRUCTION)

    da3._run_finalized = True
    da3._state = DA3CanonicalRunState.COMPLETED
    controller.da3_canonical_state_changed.emit(DA3CanonicalRunState.COMPLETED)
    controller.da3_canonical_run_finished.emit({"state": DA3CanonicalRunState.COMPLETED})
    qtbot.wait(5)
    assert controller.is_active is False
    assert page._da3_runtime_value.text() == "Completed"
    assert page._stage_value.text() == "DA3 canonical scene ready"
    assert page._progress_bar.maximum() == 100
    assert all(button.isEnabled() for button in window._mode_tabs._buttons.values())


def test_3h_advanced_reconstruction_controls_fit_1366x768(window, qtbot, tmp_path: Path) -> None:
    page = window._reconstruction_page
    window.resize(1366, 768)
    window.show()
    window._reconstruction_controller.select_image_sequence(write_ready_image_sequence(tmp_path))
    page._advanced_reconstruction_group.set_expanded(True)
    page._settings_scroll.ensureWidgetVisible(page._advanced_reconstruction_group)
    qtbot.wait(20)

    assert page._settings_scroll.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    assert page._settings_scroll.horizontalScrollBar().maximum() == 0
    assert page._advanced_reconstruction_group.width() <= page._settings_scroll.viewport().width()
    assert page._da3_integrate_button.width() > 0

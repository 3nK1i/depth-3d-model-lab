from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt

from app.main_window import MainWindow
from app.widgets.mode_tabs import AppMode
from core.reconstruction.da3 import DA3_PROVIDER_ID
from tests.phase9_3f_helpers import make_phase9_3f_scene


def _window(qtbot, monkeypatch):
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    window = MainWindow()
    qtbot.addWidget(window)
    return window


def test_3g_advanced_reconstruction_provider_inspector_is_read_only(qtbot, monkeypatch, tmp_path: Path) -> None:
    window = _window(qtbot, monkeypatch)
    page = window._reconstruction_page
    scene, _semantic, _surface = make_phase9_3f_scene(
        tmp_path / "providers", request_id="phase9-3g-ui"
    )
    page.set_persistent_scene(scene, "persistent_reconstruction_scene.json")

    presentation = page.dense_provider_presentation
    assert presentation is not None
    assert presentation.available is True
    assert presentation.backend_started is False
    assert page._advanced_reconstruction_group.title == "ADVANCED RECONSTRUCTION"
    assert page._dense_provider_combo.isEnabled() is True
    assert page._dense_provider_combo.count() == 4
    assert page._dense_provider_combo.currentData() == "canonical-dense-fusion"
    assert "canonical-dense-fusion" in page._dense_pipeline_value.text()
    assert "quality-aware-surface-mesh" in page._dense_pipeline_value.text()
    assert page._dense_role_value.text() == "Canonical dense geometry"
    assert page._dense_execution_value.text() == "In Process"
    assert page._dense_scene_use_value.text() == "Bound to current persistent scene"
    assert page._dense_authority_value.text() == "Canonical geometry authority"
    assert scene.canonical_geometry_authoritative is True
    assert scene.certified_metrology is False
    window.close()


def test_3g_da3_entry_is_visible_but_not_activated_or_probed(qtbot, monkeypatch, tmp_path: Path) -> None:
    window = _window(qtbot, monkeypatch)
    page = window._reconstruction_page
    scene, _semantic, _surface = make_phase9_3f_scene(
        tmp_path / "da3", request_id="phase9-3g-da3"
    )
    page.set_persistent_scene(scene, "persistent_reconstruction_scene.json")

    index = page._dense_provider_combo.findData(DA3_PROVIDER_ID)
    assert index >= 0
    page._dense_provider_combo.setCurrentIndex(index)
    entry = page.selected_dense_provider_entry
    assert entry is not None
    assert entry.provider_id == DA3_PROVIDER_ID
    assert entry.scene_bound is False
    assert entry.external is True
    assert "External process" in page._dense_execution_value.text()
    assert "Not used" in page._dense_scene_use_value.text()
    assert "runtime not probed" in page._dense_state_value.text().lower()
    assert "9.3H" in page._dense_state_value.text()
    assert "Supporting geometry only" in page._dense_authority_value.text()
    assert page.dense_provider_presentation.backend_started is False
    assert scene.canonical_geometry_authoritative is True
    assert scene.certified_metrology is False
    window.close()


def test_3g_provider_ui_remains_responsive_and_fails_closed_without_scene(qtbot, monkeypatch, tmp_path: Path) -> None:
    window = _window(qtbot, monkeypatch)
    window._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)
    window.resize(1366, 768)
    window.show()
    qtbot.wait(20)
    page = window._reconstruction_page
    scene, _semantic, _surface = make_phase9_3f_scene(
        tmp_path / "layout", request_id="phase9-3g-layout"
    )
    page.set_persistent_scene(scene, "persistent_reconstruction_scene.json")
    page._advanced_reconstruction_group.set_expanded(True)
    page._settings_scroll.ensureWidgetVisible(page._advanced_reconstruction_group)
    qtbot.wait(20)

    assert page._settings_scroll.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    assert page._settings_scroll.horizontalScrollBar().maximum() == 0
    assert page._advanced_reconstruction_group.width() <= page._settings_scroll.viewport().width()

    page.set_persistent_scene(None, "")
    assert page._dense_provider_combo.isEnabled() is False
    assert page._dense_provider_combo.currentText() == "No persistent scene"
    assert page.dense_provider_presentation is not None
    assert page.dense_provider_presentation.available is False
    assert page.dense_provider_presentation.backend_started is False
    window.close()

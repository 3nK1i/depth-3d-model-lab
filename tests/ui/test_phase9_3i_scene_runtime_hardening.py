from __future__ import annotations

from PyQt6.QtCore import Qt

from app.main_window import MainWindow
from app.widgets.mode_tabs import AppMode
from tests.phase9_3i_helpers import make_phase9_3i_scene


def _window(qtbot, monkeypatch):
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    window = MainWindow()
    qtbot.addWidget(window)
    return window


def test_3i_new_scene_coalesces_mesh_camera_fit_into_one_final_fit(qtbot, monkeypatch, tmp_path) -> None:
    window = _window(qtbot, monkeypatch)
    page = window._reconstruction_page
    canvas = page._reconstruction_canvas
    scene, _semantic, _surface = make_phase9_3i_scene(tmp_path, request_id="phase9-3i-fit")
    before = canvas.fit_calls

    page.set_persistent_scene(scene, "persistent_reconstruction_scene.json")

    assert canvas.canonical_mesh_loaded is True
    assert canvas.canonical_camera_loaded is True
    assert canvas.fit_calls == before + 1
    health = page.scene_runtime_health
    assert health is not None
    assert health.bind_count == 1
    assert health.same_scene_reload_count == 0
    assert health.view_preserved is False
    assert health.certified_metrology is False
    window.close()


def test_3i_same_scene_reload_preserves_view_selection_and_reuses_mesh_resource(qtbot, monkeypatch, tmp_path) -> None:
    window = _window(qtbot, monkeypatch)
    page = window._reconstruction_page
    canvas = page._reconstruction_canvas
    scene, semantic, _surface = make_phase9_3i_scene(tmp_path, request_id="phase9-3i-reload")
    page.set_persistent_scene(scene, "persistent_reconstruction_scene.json")
    page._semantic_object_combo.setCurrentIndex(1)
    selected = semantic.objects[0].object_id
    assert canvas.selected_semantic_object_id == selected
    fit_after_first = canvas.fit_calls

    page.set_persistent_scene(scene, "persistent_reconstruction_scene.json")

    assert canvas.fit_calls == fit_after_first
    assert canvas.selected_semantic_object_id == selected
    assert page._semantic_object_combo.currentIndex() == 1
    health = page.scene_runtime_health
    assert health is not None
    assert health.bind_count == 2
    assert health.same_scene_reload_count == 1
    assert health.mesh_visual_reused is True
    assert health.view_preserved is True
    assert health.selection_preserved is True
    assert "Same-scene reload" in page._scene_health_lifecycle_value.text()
    assert "selection preserved" in page._scene_health_lifecycle_value.text()
    window.close()


def test_3i_different_scene_fits_once_clears_selection_and_scene_health_fits_1366(qtbot, monkeypatch, tmp_path) -> None:
    window = _window(qtbot, monkeypatch)
    window._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)
    window.resize(1366, 768)
    window.show()
    qtbot.wait(20)
    page = window._reconstruction_page
    canvas = page._reconstruction_canvas
    first, _semantic, _surface = make_phase9_3i_scene(tmp_path, request_id="phase9-3i-first")
    second, _semantic2, _surface2 = make_phase9_3i_scene(tmp_path, request_id="phase9-3i-second")
    page.set_persistent_scene(first, "first.json")
    page._semantic_object_combo.setCurrentIndex(1)
    fit_before_switch = canvas.fit_calls

    page.set_persistent_scene(second, "second.json")
    qtbot.wait(20)

    assert canvas.fit_calls == fit_before_switch + 1
    assert canvas.selected_semantic_object_id is None
    health = page.scene_runtime_health
    assert health is not None
    assert health.view_preserved is False
    assert health.selection_preserved is False
    assert health.canonical_geometry_authoritative is True
    assert "one final Fit" in page._scene_health_lifecycle_value.text()
    page._scene_health_group.set_expanded(True)
    page._settings_scroll.ensureWidgetVisible(page._scene_health_group)
    qtbot.wait(20)
    assert page._settings_scroll.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    assert page._settings_scroll.horizontalScrollBar().maximum() == 0
    assert page._scene_health_group.width() <= page._settings_scroll.viewport().width()
    window.close()

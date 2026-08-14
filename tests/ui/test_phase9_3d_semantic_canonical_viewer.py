from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt

from app.main_window import MainWindow
from app.widgets.mode_tabs import AppMode
from tests.phase9_3d_helpers import make_phase9_3d_scene


def _window(qtbot, monkeypatch):
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    window = MainWindow()
    qtbot.addWidget(window)
    return window


def test_3d_persistent_semantic_objects_bind_to_canonical_surface_and_selection(qtbot, monkeypatch, tmp_path: Path) -> None:
    window = _window(qtbot, monkeypatch)
    page = window._reconstruction_page
    scene, semantic, _surface = make_phase9_3d_scene(tmp_path)

    page.set_persistent_scene(scene, "persistent_reconstruction_scene.json")

    canvas = page._reconstruction_canvas
    presentation = page.canonical_semantic_presentation
    assert canvas.canonical_mesh_loaded is True
    assert presentation is not None
    assert presentation.object_ids == tuple(item.object_id for item in semantic.objects)
    assert presentation.surface_bound_object_count == 3
    assert presentation.surface_highlight_object_count == 3
    assert canvas.canonical_semantic_loaded is True
    assert canvas.canonical_semantic_scene_id == scene.scene_id
    assert canvas.canonical_semantic_object_ids == presentation.object_ids
    assert canvas.canonical_semantic_surface_binding_count == 3
    assert canvas.canonical_semantic_label_display_count == 3
    assert canvas.canonical_semantic_labels_visible is True
    assert page._viewport_state.text() == "Canonical scene • semantic objects"

    page._semantic_object_combo.setCurrentIndex(1)
    assert canvas.selected_semantic_object_id == semantic.objects[0].object_id
    assert canvas.selected_canonical_semantic_surface_visible is True
    assert canvas.selected_canonical_semantic_label_visible is True
    assert "Exact canonical-surface membership persisted and validated" in page._semantic_support_value.toolTip()
    assert "canonical geometry" in page._semantic_support_value.toolTip().lower()
    window.close()


def test_3d_existing_semantic_visibility_control_hides_labels_and_selected_surface(qtbot, monkeypatch, tmp_path: Path) -> None:
    window = _window(qtbot, monkeypatch)
    page = window._reconstruction_page
    scene, _semantic, _surface = make_phase9_3d_scene(tmp_path)
    page.set_persistent_scene(scene, "scene.json")
    page._semantic_object_combo.setCurrentIndex(1)
    canvas = page._reconstruction_canvas
    button = page._viewport_tool_buttons["semantic_objects"]

    button.setChecked(False)
    assert canvas.semantic_objects_visible is False
    assert canvas.canonical_semantic_labels_visible is False
    assert canvas.selected_canonical_semantic_surface_visible is False
    assert canvas.selected_canonical_semantic_label_visible is False

    button.setChecked(True)
    assert canvas.semantic_objects_visible is True
    assert canvas.canonical_semantic_labels_visible is True
    assert canvas.selected_canonical_semantic_surface_visible is True
    assert canvas.selected_canonical_semantic_label_visible is True
    window.close()


def test_3d_legacy_semantic_snapshot_keeps_context_without_fabricated_surface_highlight(qtbot, monkeypatch, tmp_path: Path) -> None:
    window = _window(qtbot, monkeypatch)
    page = window._reconstruction_page
    scene, semantic, _surface = make_phase9_3d_scene(tmp_path, legacy_membership=True)

    page.set_persistent_scene(scene, "legacy.json")
    presentation = page.canonical_semantic_presentation
    canvas = page._reconstruction_canvas

    assert presentation is not None
    assert len(presentation.objects) == len(semantic.objects)
    assert presentation.surface_bound_object_count == 0
    assert canvas.canonical_semantic_loaded is True
    assert canvas.canonical_semantic_surface_binding_count == 0
    page._semantic_object_combo.setCurrentIndex(1)
    assert canvas.selected_canonical_semantic_surface_visible is False
    assert canvas.selected_canonical_semantic_label_visible is True
    assert "exact surface membership was not persisted" in page._semantic_support_value.toolTip()
    assert scene.canonical_geometry_authoritative is True
    assert scene.certified_metrology is False
    window.close()


def test_3d_invalid_semantic_membership_fails_presentation_only_and_preserves_canonical_mesh(qtbot, monkeypatch, tmp_path: Path) -> None:
    def corrupt(payload, surface) -> None:
        payload["objects"][0]["face_indices"] = [surface.mesh.face_count]
        payload["objects"][0]["face_count"] = 1

    window = _window(qtbot, monkeypatch)
    page = window._reconstruction_page
    scene, _semantic, _surface = make_phase9_3d_scene(tmp_path, mutate_semantic=corrupt)

    page.set_persistent_scene(scene, "bad-semantic.json")

    canvas = page._reconstruction_canvas
    assert canvas.canonical_mesh_loaded is True
    assert canvas.semantic_objects_available is True
    assert canvas.canonical_semantic_loaded is False
    assert page.canonical_semantic_presentation is None
    assert "face_indices exceed" in page._canonical_semantic_error
    assert "association rejected" in page._semantic_structure_value.toolTip().lower()
    assert scene.canonical_geometry_authoritative is True
    assert scene.certified_metrology is False
    window.close()


def test_3d_scene_switch_1366_layout_and_shutdown_clear_semantic_resources(qtbot, monkeypatch, tmp_path: Path) -> None:
    window = _window(qtbot, monkeypatch)
    window._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)
    window.resize(1366, 768)
    window.show()
    qtbot.wait(20)
    page = window._reconstruction_page
    scene, _semantic, _surface = make_phase9_3d_scene(tmp_path / "first", request_id="phase9-3d-ui-first")
    legacy, _semantic2, _surface2 = make_phase9_3d_scene(
        tmp_path / "legacy",
        request_id="phase9-3d-ui-legacy",
        include_semantic_snapshot=False,
    )

    page.set_persistent_scene(scene, "scene.json")
    assert page._settings_scroll.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    assert page._settings_scroll.horizontalScrollBar().maximum() == 0
    assert page._reconstruction_canvas.canonical_semantic_loaded is True

    page.set_persistent_scene(legacy, "legacy.json")
    assert page._reconstruction_canvas.canonical_mesh_loaded is True
    assert page._reconstruction_canvas.canonical_semantic_loaded is False
    assert page.canonical_semantic_presentation is None

    page.set_persistent_scene(scene, "scene.json")
    assert page._reconstruction_canvas.canonical_semantic_loaded is True
    page.shutdown_viewport()
    assert page.canonical_semantic_presentation is None
    assert page._reconstruction_canvas.canonical_semantic_loaded is False
    assert page._reconstruction_canvas.canonical_mesh_loaded is False
    assert page._reconstruction_canvas.closed is True
    window.close()

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt

from app.main_window import MainWindow
from app.widgets.mode_tabs import AppMode
from tests.phase9_3e_helpers import inspection_report, measurement_context, measurement_payload
from tests.phase9_3f_helpers import make_phase9_3f_scene


def _window(qtbot, monkeypatch):
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    window = MainWindow()
    qtbot.addWidget(window)
    return window


def _load_qualified_object(page, tmp_path: Path, *, request_id: str):
    scene, semantic, _surface = make_phase9_3f_scene(tmp_path / request_id, request_id=request_id)
    page.set_persistent_scene(scene, "persistent_reconstruction_scene.json")
    page._semantic_object_combo.setCurrentIndex(1)
    object_id = semantic.objects[0].object_id
    evidence = measurement_payload()
    page.set_reconstruction_measurement_evidence(evidence)
    page.set_reconstruction_measurement_validity_evidence(evidence, context=measurement_context())
    page.set_reconstruction_inspection_qc_evidence(
        inspection_report(session_id=scene.session_id, object_id=object_id, include_object=True)
    )
    return scene, semantic


def test_3f_professional_scene_layers_control_existing_and_canonical_resources(qtbot, monkeypatch, tmp_path: Path) -> None:
    window = _window(qtbot, monkeypatch)
    page = window._reconstruction_page
    scene, _semantic = _load_qualified_object(page, tmp_path, request_id="phase9-3f-layers")
    canvas = page._reconstruction_canvas
    layers = page._scene_layer_toggles

    expected = {
        "canonical_surface", "grid", "axes", "camera_frustums", "camera_trajectory",
        "sparse_landmarks", "sparse_envelope", "semantic_objects", "semantic_labels",
        "semantic_selection", "inspection_evidence",
    }
    assert set(layers) == expected
    assert page._scene_layers_group.title == "SCENE LAYERS"
    assert canvas.canonical_mesh_loaded is True
    assert canvas.canonical_camera_loaded is True
    assert canvas.canonical_semantic_loaded is True
    assert page.canonical_inspection_overlay_presentation is not None
    assert scene.canonical_geometry_authoritative is True
    assert scene.certified_metrology is False

    # Independent canonical layers hide presentation only; loaded evidence remains intact.
    layers["canonical_surface"].setChecked(False)
    assert canvas.canonical_mesh_loaded is True
    assert canvas.canonical_mesh_visible is False
    layers["canonical_surface"].setChecked(True)
    assert canvas.canonical_mesh_visible is True

    layers["axes"].setChecked(False)
    assert canvas.axes_visible is False
    layers["axes"].setChecked(True)
    assert canvas.axes_visible is True

    # Existing quick controls are mirrored bidirectionally for compatibility.
    layers["camera_trajectory"].setChecked(False)
    assert page._viewport_tool_buttons["trajectory"].isChecked() is False
    assert canvas.canonical_camera_trajectory_visible is False
    page._viewport_tool_buttons["trajectory"].setChecked(True)
    assert layers["camera_trajectory"].isChecked() is True
    assert canvas.canonical_camera_trajectory_visible is True

    layers["camera_frustums"].setChecked(False)
    assert page._viewport_tool_buttons["keyframes"].isChecked() is False
    assert canvas.canonical_camera_frustums_visible is False
    layers["camera_frustums"].setChecked(True)
    assert canvas.canonical_camera_frustums_visible is True

    # 9.3D/E child layers remain independently controllable under the semantic master gate.
    assert canvas.canonical_semantic_labels_visible is True
    assert canvas.selected_canonical_semantic_surface_visible is True
    assert canvas.canonical_inspection_overlay_visible is True

    layers["semantic_labels"].setChecked(False)
    assert canvas.canonical_semantic_labels_visible is False
    assert canvas.selected_canonical_semantic_surface_visible is True
    layers["semantic_labels"].setChecked(True)

    layers["semantic_selection"].setChecked(False)
    assert canvas.selected_canonical_semantic_surface_visible is False
    assert canvas.canonical_semantic_labels_visible is True
    layers["semantic_selection"].setChecked(True)

    layers["inspection_evidence"].setChecked(False)
    assert page.canonical_inspection_overlay_presentation is not None
    assert canvas.canonical_inspection_overlay_visible is False
    layers["inspection_evidence"].setChecked(True)
    assert canvas.canonical_inspection_overlay_visible is True

    layers["semantic_objects"].setChecked(False)
    assert page._viewport_tool_buttons["semantic_objects"].isChecked() is False
    assert canvas.semantic_objects_visible is False
    assert canvas.canonical_semantic_labels_visible is False
    assert canvas.selected_canonical_semantic_surface_visible is False
    assert canvas.canonical_inspection_overlay_visible is False
    layers["semantic_objects"].setChecked(True)
    assert canvas.semantic_objects_visible is True
    assert canvas.canonical_semantic_labels_visible is True
    assert canvas.selected_canonical_semantic_surface_visible is True
    assert canvas.canonical_inspection_overlay_visible is True
    window.close()


def test_3f_layer_preferences_survive_scene_reload_without_mutating_evidence(qtbot, monkeypatch, tmp_path: Path) -> None:
    window = _window(qtbot, monkeypatch)
    page = window._reconstruction_page
    scene, semantic = _load_qualified_object(page, tmp_path, request_id="phase9-3f-reload")
    layers = page._scene_layer_toggles
    canvas = page._reconstruction_canvas

    for key in ("canonical_surface", "semantic_labels", "semantic_selection", "inspection_evidence"):
        layers[key].setChecked(False)

    page.set_persistent_scene(None, "")
    assert canvas.canonical_mesh_loaded is False
    for key in ("canonical_surface", "semantic_labels", "semantic_selection", "inspection_evidence"):
        assert layers[key].isChecked() is False

    page.set_persistent_scene(scene, "persistent_reconstruction_scene.json")
    page._semantic_object_combo.setCurrentIndex(1)
    object_id = semantic.objects[0].object_id
    evidence = measurement_payload()
    page.set_reconstruction_measurement_evidence(evidence)
    page.set_reconstruction_measurement_validity_evidence(evidence, context=measurement_context())
    page.set_reconstruction_inspection_qc_evidence(
        inspection_report(session_id=scene.session_id, object_id=object_id, include_object=True)
    )

    assert canvas.canonical_mesh_loaded is True
    assert canvas.canonical_mesh_visible is False
    assert canvas.canonical_semantic_loaded is True
    assert canvas.canonical_semantic_labels_visible is False
    assert canvas.selected_canonical_semantic_surface_visible is False
    assert page.canonical_inspection_overlay_presentation is not None
    assert canvas.canonical_inspection_overlay_visible is False
    assert scene.canonical_geometry_authoritative is True
    assert scene.certified_metrology is False
    window.close()


def test_3f_unavailable_layers_fail_closed_without_fabricating_semantics(qtbot, monkeypatch, tmp_path: Path) -> None:
    window = _window(qtbot, monkeypatch)
    page = window._reconstruction_page
    scene, _semantic, _surface = make_phase9_3f_scene(
        tmp_path / "legacy", request_id="phase9-3f-legacy", include_semantic_snapshot=False
    )
    page.set_persistent_scene(scene, "legacy.json")
    layers = page._scene_layer_toggles

    assert layers["canonical_surface"].isEnabled() is True
    assert layers["semantic_objects"].isEnabled() is False
    assert layers["semantic_labels"].isEnabled() is False
    assert layers["semantic_selection"].isEnabled() is False
    assert layers["inspection_evidence"].isEnabled() is False
    assert page._reconstruction_canvas.canonical_semantic_loaded is False
    assert scene.canonical_geometry_authoritative is True
    assert scene.certified_metrology is False
    window.close()


def test_3f_scene_layers_remain_responsive_at_1366x768(qtbot, monkeypatch, tmp_path: Path) -> None:
    window = _window(qtbot, monkeypatch)
    window._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)
    window.resize(1366, 768)
    window.show()
    qtbot.wait(20)
    page = window._reconstruction_page
    _load_qualified_object(page, tmp_path, request_id="phase9-3f-layout")
    page._scene_layers_group.set_expanded(True)
    page._settings_scroll.ensureWidgetVisible(page._scene_layers_group)
    qtbot.wait(20)

    assert page._settings_scroll.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    assert page._settings_scroll.horizontalScrollBar().maximum() == 0
    assert page._scene_layers_group.width() <= page._settings_scroll.viewport().width()
    assert all(toggle.width() == 32 and toggle.height() == 18 for toggle in page._scene_layer_toggles.values())
    window.close()

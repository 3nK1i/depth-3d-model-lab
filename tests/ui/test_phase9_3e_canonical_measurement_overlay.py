from __future__ import annotations

from pathlib import Path

from app.main_window import MainWindow
from tests.phase9_3d_helpers import make_phase9_3d_scene
from tests.phase9_3e_helpers import inspection_report, measurement_context, measurement_payload


def _window(qtbot, monkeypatch):
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    window = MainWindow()
    qtbot.addWidget(window)
    return window


def test_3e_selected_canonical_object_requires_e3_before_numeric_viewport_overlay(qtbot, monkeypatch, tmp_path: Path) -> None:
    window = _window(qtbot, monkeypatch)
    page = window._reconstruction_page
    scene, semantic, _surface = make_phase9_3d_scene(tmp_path, request_id="phase9-3e-ui")
    page.set_persistent_scene(scene, "scene.json")
    page._semantic_object_combo.setCurrentIndex(1)
    canvas = page._reconstruction_canvas
    first_id = semantic.objects[0].object_id
    assert canvas.selected_semantic_object_id == first_id
    assert canvas.canonical_inspection_overlay_visible is False

    evidence = measurement_payload()
    page.set_reconstruction_measurement_evidence(evidence)
    assert page.reconstruction_measurement_presentation is not None
    assert page.reconstruction_measurement_validity_presentation is None
    assert page.canonical_inspection_overlay_presentation is None
    assert canvas.canonical_inspection_overlay_visible is False
    assert canvas.canonical_inspection_overlay_text == ""

    page.set_reconstruction_measurement_validity_evidence(evidence, context=measurement_context())
    overlay = page.canonical_inspection_overlay_presentation
    assert overlay is not None
    assert overlay.object_id == first_id
    assert canvas.canonical_inspection_overlay_visible is True
    assert canvas.canonical_inspection_overlay_object_id == first_id
    # Keep the viewport evidence readable at the 1366x768 minimum target.
    assert float(canvas.CANONICAL_INSPECTION_OVERLAY_FONT_SIZE) >= 14.0
    assert "42.0 mm ± 1.5 mm" in canvas.canonical_inspection_overlay_text
    assert "Depth: REJECTED" in canvas.canonical_inspection_overlay_text
    assert "Excessive uncertainty" in canvas.canonical_inspection_overlay_text
    assert "canonical geometry unchanged" in canvas.canonical_inspection_overlay_text
    assert "metrology not certified" in canvas.canonical_inspection_overlay_text
    window.close()


def test_3e_qc_is_exact_object_only_and_visibility_follows_semantic_layer(qtbot, monkeypatch, tmp_path: Path) -> None:
    window = _window(qtbot, monkeypatch)
    page = window._reconstruction_page
    scene, semantic, _surface = make_phase9_3d_scene(tmp_path, request_id="phase9-3e-qc")
    page.set_persistent_scene(scene, "scene.json")
    page._semantic_object_combo.setCurrentIndex(1)
    canvas = page._reconstruction_canvas
    first_id = semantic.objects[0].object_id

    page.set_reconstruction_inspection_qc_evidence(
        inspection_report(session_id=scene.session_id, object_id=first_id, include_object=True)
    )
    assert canvas.canonical_inspection_overlay_visible is True
    assert "QC: WARN" in canvas.canonical_inspection_overlay_text
    assert "Width near tolerance" in canvas.canonical_inspection_overlay_text

    # Same attached source report, different selected object: never copy the
    # scene-level WARN decision onto an object without an exact feature result.
    page._semantic_object_combo.setCurrentIndex(2)
    assert canvas.canonical_inspection_overlay_visible is True
    assert "QC: NOT EVALUATED" in canvas.canonical_inspection_overlay_text
    assert "QC: WARN" not in canvas.canonical_inspection_overlay_text

    button = page._viewport_tool_buttons["semantic_objects"]
    button.setChecked(False)
    assert canvas.canonical_inspection_overlay_visible is False
    button.setChecked(True)
    assert canvas.canonical_inspection_overlay_visible is True

    page._semantic_object_combo.setCurrentIndex(0)
    assert canvas.canonical_inspection_overlay_visible is False
    assert canvas.canonical_inspection_overlay_text == ""
    window.close()


def test_3e_scene_switch_and_shutdown_clear_overlay_resources(qtbot, monkeypatch, tmp_path: Path) -> None:
    window = _window(qtbot, monkeypatch)
    page = window._reconstruction_page
    scene, semantic, _surface = make_phase9_3d_scene(tmp_path / "first", request_id="phase9-3e-first")
    legacy, _semantic2, _surface2 = make_phase9_3d_scene(
        tmp_path / "legacy", request_id="phase9-3e-legacy", include_semantic_snapshot=False
    )
    page.set_persistent_scene(scene, "scene.json")
    page._semantic_object_combo.setCurrentIndex(1)
    evidence = measurement_payload()
    page.set_reconstruction_measurement_evidence(evidence)
    page.set_reconstruction_measurement_validity_evidence(evidence, context=measurement_context())
    assert page._reconstruction_canvas.canonical_inspection_overlay_visible is True

    page.set_persistent_scene(legacy, "legacy.json")
    assert page._reconstruction_canvas.canonical_inspection_overlay_visible is False
    assert page.canonical_inspection_overlay_presentation is None

    page.set_persistent_scene(scene, "scene.json")
    page._semantic_object_combo.setCurrentIndex(1)
    # Existing exact evidence remains session/object keyed and may rebind only
    # when the scene carries the same exact session/object identity.
    page._refresh_canonical_inspection_overlay()
    page.shutdown_viewport()
    assert page.canonical_inspection_overlay_presentation is None
    assert page._reconstruction_canvas.canonical_inspection_overlay_visible is False
    assert page._reconstruction_canvas.closed is True
    window.close()

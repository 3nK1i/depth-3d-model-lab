from __future__ import annotations

import numpy as np

from app.main_window import MainWindow
from core.reconstruction.measurement_registration import CanonicalMeasurementRegistrationState
from tests.phase9_3j_helpers import (
    OBSERVATION_ID,
    make_phase9_3j_inspection_bundle,
    make_phase9_3j_qualified_scene,
)


def _window(qtbot, monkeypatch):
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    window = MainWindow()
    qtbot.addWidget(window)
    return window


def _load_inspection(window: MainWindow, tmp_path):
    summary, obj = make_phase9_3j_inspection_bundle(tmp_path)
    height, width = summary.image_height, summary.image_width
    arrays = {
        "aligned_rgb": np.zeros((height, width, 3), dtype=np.uint8),
        "mask:0": np.zeros((height, width), dtype=bool),
        "raw:0": np.zeros((0, 3), dtype=np.float32),
        "filtered:0": np.zeros((0, 3), dtype=np.float32),
    }
    window._inspection_page.load_bundle(summary, arrays)
    assert window._inspection_page.selected_object is not None
    assert window._inspection_page.selected_object.observation_id == OBSERVATION_ID
    return summary, obj


def test_3j_shared_bridge_exposes_qualified_registration_in_both_workspaces(qtbot, monkeypatch, tmp_path) -> None:
    window = _window(qtbot, monkeypatch)
    page = window._reconstruction_page
    inspection = window._inspection_page
    scene, authority = make_phase9_3j_qualified_scene(tmp_path / "scene")
    page.set_persistent_scene(scene, "scene.json")
    page.set_persistent_authority_snapshot(authority, "authority.json")
    _summary, _obj = _load_inspection(window, tmp_path / "inspection")

    semantic = page.canonical_semantic_presentation
    assert semantic is not None
    target = next(item for item in semantic.objects if OBSERVATION_ID in item.observation_ids)
    assert page.select_semantic_object_by_id(target.object_id) is True
    window._refresh_measurement_canonical_registration()

    presentation = inspection.canonical_registration_presentation
    assert presentation is not None
    assert presentation.state is CanonicalMeasurementRegistrationState.QUALIFIED
    assert page.canonical_measurement_registration_presentation is presentation
    assert page._canonical_registration_state_value.text() == "Qualified"
    assert "42.0 mm" in page._canonical_registration_dimensions_value.text()
    assert page._reconstruction_canvas.canonical_registered_dimension_count == 1
    assert page._reconstruction_canvas.canonical_registered_dimensions_visible is True
    assert any(
        "Width: 42.0 mm" in text
        for text in page._reconstruction_canvas.canonical_registered_dimension_text
    )
    assert "42.0 mm" in inspection._canonical_registration_dimensions.text()

    layer = page._scene_layer_toggles["inspection_evidence"]
    layer.setChecked(False)
    assert page._reconstruction_canvas.canonical_registered_dimensions_visible is False
    layer.setChecked(True)
    assert page._reconstruction_canvas.canonical_registered_dimensions_visible is True
    window.close()


def test_3j_missing_metric_authority_is_unavailable_without_breaking_measure_or_reconstruction(qtbot, monkeypatch, tmp_path) -> None:
    window = _window(qtbot, monkeypatch)
    page = window._reconstruction_page
    inspection = window._inspection_page
    scene, _authority = make_phase9_3j_qualified_scene(tmp_path / "scene")
    page.set_persistent_scene(scene, "scene.json")
    _summary, _obj = _load_inspection(window, tmp_path / "inspection")

    semantic = page.canonical_semantic_presentation
    assert semantic is not None
    target = next(item for item in semantic.objects if OBSERVATION_ID in item.observation_ids)
    page.select_semantic_object_by_id(target.object_id)
    window._refresh_measurement_canonical_registration()

    presentation = inspection.canonical_registration_presentation
    assert presentation is not None
    assert presentation.state is CanonicalMeasurementRegistrationState.UNAVAILABLE
    assert presentation.dimensions == ()
    assert page.canonical_measurement_registration_presentation is presentation
    assert page._canonical_registration_state_value.text() == "Unavailable"
    assert page._reconstruction_canvas.canonical_registered_dimension_count == 0
    assert page._reconstruction_canvas.canonical_registered_dimensions_visible is False
    assert inspection.selected_object is not None
    assert inspection.selected_object.measurements[0]["value_si"] == 0.042
    assert page._reconstruction_canvas.canonical_mesh_loaded is True
    assert scene.canonical_geometry_authoritative is True
    assert scene.certified_metrology is False
    window.close()

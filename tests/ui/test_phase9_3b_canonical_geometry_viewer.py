from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt

from app.main_window import MainWindow
from app.widgets.mode_tabs import AppMode
from core.reconstruction import (
    MetricEvidenceValidationState,
    build_persistent_scene_authority_snapshot,
    persistent_scene_from_pipeline_result,
    reference_target_metric_evidence,
    write_persistent_reconstruction_scene,
)
from core.reconstruction.calibration_management import create_manual_camera_calibration_profile
from tests.reconstruction_f5_helpers import run_f5_fixture


def _scene(tmp_path: Path):
    _fixture, _providers, result, snapshots = run_f5_fixture(
        tmp_path / "run", request_id="phase9-3b-ui"
    )
    surface = (tmp_path / "canonical_surface.obj").resolve()
    surface.write_text(
        "v -1 -1 0 1 0 0\n"
        "v 1 -1 0 0 1 0\n"
        "v 1 1 0 0 0 1\n"
        "v -1 1 0 1 1 0\n"
        "f 1 2 3\n"
        "f 1 3 4\n",
        encoding="utf-8",
    )
    snapshots = dict(snapshots)
    snapshots["surface_mesh"] = {
        "surface_obj_path": str(surface),
        "surface_vertex_count": 4,
        "surface_face_count": 2,
        "metric_geometry_status": "arbitrary",
        "canonical_geometry_authoritative": True,
        "certified_metrology": False,
    }
    return persistent_scene_from_pipeline_result(result, payload_snapshots_by_kind=snapshots), surface


def _qualified_authority(scene):
    profile = create_manual_camera_calibration_profile(
        width=640,
        height=480,
        fx=520.0,
        fy=521.0,
        cx=320.0,
        cy=240.0,
        profile_id="phase9-3b-ui-profile",
        calibration_record_id="phase9-3b-ui-record",
        camera_id="phase9-3b-ui-camera",
        provenance="phase9_3b_ui",
    )
    evidence = reference_target_metric_evidence(
        evidence_id="phase9-3b-ui-target",
        reconstruction_frame_id="reconstruction-world",
        reference_id="phase9-3b-ui-reference",
        target_length_m=0.25,
        measured_length_reconstruction_units=5.0,
        target_relative_uncertainty=0.005,
        measurement_relative_uncertainty=0.02,
        provenance="phase9_3b_ui",
        validation_state=MetricEvidenceValidationState.VALIDATED,
        validation_record_id="phase9-3b-ui-validation",
    )
    return build_persistent_scene_authority_snapshot(
        scene,
        camera_calibration_profile=profile,
        metric_qualification_evidence=evidence,
    )


def _window(qtbot, monkeypatch):
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    window = MainWindow()
    qtbot.addWidget(window)
    return window


def test_3b_persistent_scene_loads_canonical_surface_into_existing_viewport(qtbot, monkeypatch, tmp_path: Path) -> None:
    window = _window(qtbot, monkeypatch)
    page = window._reconstruction_page
    scene, surface = _scene(tmp_path)

    page.set_persistent_scene(scene, str(tmp_path / "persistent_reconstruction_scene.json"))

    canvas = page._reconstruction_canvas
    assert canvas.canonical_mesh_loaded is True
    assert canvas.canonical_mesh_visible is True
    assert canvas.canonical_mesh_vertex_count == 4
    assert canvas.canonical_mesh_face_count == 2
    assert canvas.canonical_mesh_scene_id == scene.scene_id
    assert canvas.canonical_mesh_artifact_id == scene.canonical_geometry_artifact_id
    assert canvas.canonical_mesh_source_path == str(surface)
    assert page.canonical_viewer_mesh is not None
    assert page.canonical_viewer_mesh.certified_metrology is False
    assert page._viewport_placeholder.isHidden()
    assert page._viewport_state.text() == "Canonical surface mesh"
    assert "Interactive canonical surface" in page._canonical_scene_status_value.text()
    assert page.active_view_representation == "canonical_geometry"
    assert canvas.fit_calls >= 1
    window.close()


def test_3b_viewer_uses_z_up_presentation_transform_without_mutating_authority(qtbot, monkeypatch, tmp_path: Path) -> None:
    window = _window(qtbot, monkeypatch)
    page = window._reconstruction_page
    scene, _surface = _scene(tmp_path)
    page.set_persistent_scene(scene, "scene.json")

    vertices = page._reconstruction_canvas._canonical_mesh_vertices
    np.testing.assert_allclose(
        vertices,
        np.asarray([[-1, 0, 1], [1, 0, 1], [1, 0, -1], [-1, 0, -1]], dtype=np.float32),
    )
    assert scene.canonical_geometry_authoritative is True
    assert scene.certified_metrology is False
    assert page._viewport_safety.text() == "VIEWER Z-UP • ARBITRARY GAUGE"
    window.close()


def test_3b_metric_authority_updates_banner_without_reloading_or_certifying_mesh(qtbot, monkeypatch, tmp_path: Path) -> None:
    window = _window(qtbot, monkeypatch)
    page = window._reconstruction_page
    scene, _surface = _scene(tmp_path)
    page.set_persistent_scene(scene, "scene.json")
    before = page._reconstruction_canvas._canonical_mesh_vertices.copy()

    page.set_persistent_authority_snapshot(_qualified_authority(scene), "authority.json")

    np.testing.assert_array_equal(page._reconstruction_canvas._canonical_mesh_vertices, before)
    assert page._viewport_safety.text() == "VIEWER Z-UP • METRIC SCALE QUALIFIED"
    assert page._metrology_value.text() == "Not certified"
    assert "Interactive canonical surface" in page._canonical_scene_status_value.text()
    assert page.canonical_viewer_mesh is not None
    assert page.canonical_viewer_mesh.certified_metrology is False
    window.close()


def test_3b_viewer_failure_clears_stale_mesh_but_retains_canonical_authority(qtbot, monkeypatch, tmp_path: Path) -> None:
    window = _window(qtbot, monkeypatch)
    page = window._reconstruction_page
    scene, surface = _scene(tmp_path)
    original = surface.read_bytes()
    surface.write_bytes(bytes([original[0] ^ 1]) + original[1:])

    page.set_persistent_scene(scene, "scene.json")

    assert scene.canonical_geometry_authoritative is True
    assert page.canonical_scene_presentation is not None
    assert page.canonical_scene_presentation.canonical_geometry_authoritative is True
    assert page.canonical_viewer_mesh is None
    assert page._reconstruction_canvas.canonical_mesh_loaded is False
    assert page._viewport_state.text() == "Canonical surface unavailable"
    assert "canonical authority retained" in page._canonical_scene_status_value.text().lower()
    window.close()


def test_3b_clear_shutdown_and_1366_layout_release_viewer_resource(qtbot, monkeypatch, tmp_path: Path) -> None:
    window = _window(qtbot, monkeypatch)
    window._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)
    window.resize(1366, 768)
    window.show()
    qtbot.wait(20)
    page = window._reconstruction_page
    scene, _surface = _scene(tmp_path)
    page.set_persistent_scene(scene, "scene.json")
    assert page._reconstruction_canvas.canonical_mesh_loaded
    assert page._settings_scroll.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    assert page._settings_scroll.horizontalScrollBar().maximum() == 0

    page.set_persistent_scene(None, "")
    assert page.canonical_viewer_mesh is None
    assert page._reconstruction_canvas.canonical_mesh_loaded is False

    page.set_persistent_scene(scene, "scene.json")
    assert page._reconstruction_canvas.canonical_mesh_loaded
    page.shutdown_viewport()
    assert page.canonical_viewer_mesh is None
    assert page._reconstruction_canvas.closed is True
    assert page._reconstruction_canvas.canonical_mesh_loaded is False
    window.close()


def test_3b_controller_load_and_reload_bind_verified_persistent_surface(
    qtbot, monkeypatch, tmp_path: Path
) -> None:
    window = _window(qtbot, monkeypatch)
    controller = window._reconstruction_controller
    page = window._reconstruction_page
    scene, _surface = _scene(tmp_path)
    manifest = write_persistent_reconstruction_scene(scene, tmp_path / "persistent")

    loaded = controller.load_persistent_scene(manifest, verify_file_references=True)
    assert loaded.scene_id == scene.scene_id
    assert page._reconstruction_canvas.canonical_mesh_loaded is True
    first_source = page._reconstruction_canvas.canonical_mesh_source_path

    reloaded = controller.reload_persistent_scene()
    assert reloaded.scene_fingerprint_sha256 == scene.scene_fingerprint_sha256
    assert page._reconstruction_canvas.canonical_mesh_loaded is True
    assert page._reconstruction_canvas.canonical_mesh_source_path == first_source
    assert page.active_view_representation == "canonical_geometry"
    assert reloaded.canonical_geometry_authoritative is True
    assert reloaded.certified_metrology is False
    window.close()


def test_3b_scene_without_materialized_surface_keeps_authority_without_inventing_viewer_geometry(
    qtbot, monkeypatch, tmp_path: Path
) -> None:
    window = _window(qtbot, monkeypatch)
    page = window._reconstruction_page
    _fixture, _providers, result, snapshots = run_f5_fixture(
        tmp_path / "no-surface-run", request_id="phase9-3b-no-surface"
    )
    scene = persistent_scene_from_pipeline_result(result, payload_snapshots_by_kind=snapshots)

    page.set_persistent_scene(scene, "scene.json")

    assert scene.canonical_geometry_authoritative is True
    assert page.canonical_viewer_mesh is None
    assert page._reconstruction_canvas.canonical_mesh_loaded is False
    assert page.canonical_scene_presentation is not None
    assert page.canonical_scene_presentation.viewer_resource_ready is False
    assert "no materialized interactive surface" in page._canonical_scene_status_value.text().lower()
    assert page._viewport_safety.text() == "VIEWER Z-UP • ARBITRARY GAUGE"
    window.close()

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt

from app.main_window import MainWindow
from app.widgets.mode_tabs import AppMode
from core.reconstruction import (
    MetricEvidenceValidationState,
    build_persistent_scene_authority_snapshot,
    persistent_scene_from_pipeline_result,
    reference_target_metric_evidence,
)
from core.reconstruction.calibration_management import create_manual_camera_calibration_profile
from tests.reconstruction_f5_helpers import run_f5_fixture


def _scene(tmp_path: Path):
    _fixture, _providers, result, snapshots = run_f5_fixture(
        tmp_path / "run", request_id="phase9-3a-ui"
    )
    surface = (tmp_path / "canonical_surface.obj").resolve()
    surface.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    snapshots = dict(snapshots)
    snapshots["surface_mesh"] = {
        "surface_obj_path": str(surface),
        "metric_geometry_status": "arbitrary",
        "canonical_geometry_authoritative": True,
        "certified_metrology": False,
    }
    return persistent_scene_from_pipeline_result(result, payload_snapshots_by_kind=snapshots)


def _qualified_authority(scene):
    profile = create_manual_camera_calibration_profile(
        width=640,
        height=480,
        fx=520.0,
        fy=521.0,
        cx=320.0,
        cy=240.0,
        profile_id="phase9-3a-ui-profile",
        calibration_record_id="phase9-3a-ui-record",
        camera_id="phase9-3a-ui-camera",
        provenance="phase9_3a_ui",
    )
    evidence = reference_target_metric_evidence(
        evidence_id="phase9-3a-ui-target",
        reconstruction_frame_id="reconstruction-world",
        reference_id="phase9-3a-ui-target-ref",
        target_length_m=0.25,
        measured_length_reconstruction_units=5.0,
        target_relative_uncertainty=0.005,
        measurement_relative_uncertainty=0.02,
        provenance="phase9_3a_ui",
        validation_state=MetricEvidenceValidationState.VALIDATED,
        validation_record_id="phase9-3a-ui-validation",
    )
    return build_persistent_scene_authority_snapshot(
        scene,
        camera_calibration_profile=profile,
        metric_qualification_evidence=evidence,
    )


def test_3a_sidebar_binds_scene_without_changing_canonical_authority_contract(qtbot, monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    window = MainWindow()
    qtbot.addWidget(window)
    page = window._reconstruction_page
    scene = _scene(tmp_path)

    page.set_persistent_scene(scene, str(tmp_path / "persistent_reconstruction_scene.json"))

    assert page._canonical_scene_state_value.text() == "Ready persistent scene"
    assert page._canonical_scene_geometry_value.text() == "Surface Mesh"
    assert page._canonical_scene_resource_value.text() == "Materialized • 9.3B ready"
    assert page._canonical_scene_scale_value.text() == "Not qualified"
    assert page._canonical_scene_authority_value.text() == "Canonical geometry • metrology not certified"
    assert page.canonical_scene_presentation is not None
    assert page.canonical_scene_presentation.viewer_resource_ready is True
    assert page.canonical_scene_presentation.canonical_geometry_authoritative is True
    assert page.canonical_scene_presentation.certified_metrology is False
    assert page._viewport_safety.text() == "VIEWER Z-UP • ARBITRARY GAUGE"
    window.close()


def test_3a_bound_authority_updates_metric_wording_without_certifying_metrology(
    qtbot, monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    window = MainWindow()
    qtbot.addWidget(window)
    page = window._reconstruction_page
    scene = _scene(tmp_path)
    authority = _qualified_authority(scene)

    page.set_persistent_scene(scene, str(tmp_path / "persistent_reconstruction_scene.json"))
    page.set_persistent_authority_snapshot(authority, str(tmp_path / "persistent_scene_authority.json"))

    assert page._canonical_scene_scale_value.text().startswith("Qualified metric • ±")
    assert page._metric_scale_value.text() == "Qualified"
    assert page._metrology_value.text() == "Not certified"
    assert page._viewport_safety.text() == "VIEWER Z-UP • METRIC SCALE QUALIFIED"
    assert page.canonical_scene_presentation is not None
    assert page.canonical_scene_presentation.certified_metrology is False
    window.close()


def test_3a_clearing_scene_restores_neutral_banner_and_summary(qtbot, monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    window = MainWindow()
    qtbot.addWidget(window)
    page = window._reconstruction_page
    scene = _scene(tmp_path)
    authority = _qualified_authority(scene)
    page.set_persistent_scene(scene, "scene.json")
    page.set_persistent_authority_snapshot(authority, "authority.json")
    assert "METRIC SCALE QUALIFIED" in page._viewport_safety.text()

    page.set_persistent_scene(None, "")

    assert page._canonical_scene_state_value.text() == "No persistent scene"
    assert page._canonical_scene_resource_value.text() == "—"
    assert page._canonical_scene_scale_value.text() == "—"
    assert page._viewport_safety.text() == "VIEWER Z-UP • ARBITRARY GAUGE"
    window.close()


def test_3a_sidebar_remains_vertical_scroll_only_at_1366x768(qtbot, monkeypatch) -> None:
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    window = MainWindow()
    qtbot.addWidget(window)
    window._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)
    window.resize(1366, 768)
    window.show()
    qtbot.wait(20)
    page = window._reconstruction_page

    assert page._settings_scroll.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    assert page._settings_scroll.horizontalScrollBar().maximum() == 0
    assert page._settings_scroll.verticalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAsNeeded
    assert page._canonical_scene_authority_value.wordWrap() is True
    window.close()

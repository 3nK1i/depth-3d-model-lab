from __future__ import annotations

from pathlib import Path

from app.main_window import MainWindow
from core.reconstruction import (
    MetricEvidenceValidationState,
    build_persistent_scene_authority_snapshot,
    reference_target_metric_evidence,
)
from core.reconstruction.calibration_management import create_manual_camera_calibration_profile
from tests.reconstruction_f5_helpers import run_f5_fixture


def _profile():
    return create_manual_camera_calibration_profile(
        width=640,
        height=480,
        fx=520.0,
        fy=521.0,
        cx=320.0,
        cy=240.0,
        profile_id="phase9-1h-ui-profile",
        calibration_record_id="phase9-1h-ui-record",
        camera_id="phase9-1h-ui-camera",
        provenance="phase9_1h_ui",
    )


def _evidence():
    return reference_target_metric_evidence(
        evidence_id="phase9-1h-ui-target",
        reconstruction_frame_id="reconstruction-world",
        reference_id="ui-target",
        target_length_m=0.25,
        measured_length_reconstruction_units=5.0,
        target_relative_uncertainty=0.005,
        measurement_relative_uncertainty=0.02,
        provenance="phase9_1h_ui",
        validation_state=MetricEvidenceValidationState.VALIDATED,
        validation_record_id="phase9-1h-ui-validation",
    )


def test_1h_loaded_scene_restores_calibration_and_metric_authority(qtbot, monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    window = MainWindow()
    qtbot.addWidget(window)
    controller = window._reconstruction_controller
    page = window._reconstruction_page
    _fixture, _providers, result, snapshots = run_f5_fixture(tmp_path / "run")
    scene = controller.adopt_pipeline_result(
        result,
        (tmp_path / "source").resolve(),
        payload_snapshots_by_kind=snapshots,
    )
    authority = build_persistent_scene_authority_snapshot(
        scene,
        camera_calibration_profile=_profile(),
        metric_qualification_evidence=_evidence(),
    )
    controller.persistence_controller.save_scene_with_authority(
        (tmp_path / "saved").resolve(),
        authority_snapshot=authority,
    )
    controller.clear_persistent_scene()
    controller.load_persistent_scene((tmp_path / "saved" / "persistent_reconstruction_scene.json").resolve())

    assert page.selected_camera_calibration_profile is not None
    assert page.selected_camera_calibration_profile.profile_id == "phase9-1h-ui-profile"
    assert page._metric_scale_value.text() == "Qualified"
    assert page._metrology_value.text() == "Not certified"
    window.close()


def test_1h_loading_legacy_scene_clears_unrelated_selected_profile(qtbot, monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    window = MainWindow()
    qtbot.addWidget(window)
    controller = window._reconstruction_controller
    page = window._reconstruction_page
    page.set_camera_calibration_profile(_profile())
    assert page.selected_camera_calibration_profile is not None

    _fixture, _providers, result, snapshots = run_f5_fixture(tmp_path / "run")
    controller.adopt_pipeline_result(
        result,
        (tmp_path / "legacy").resolve(),
        payload_snapshots_by_kind=snapshots,
    )
    controller.load_persistent_scene((tmp_path / "legacy" / "persistent_reconstruction_scene.json").resolve())
    assert page.selected_camera_calibration_profile is None
    assert page._metrology_value.text() == "Not certified"
    window.close()

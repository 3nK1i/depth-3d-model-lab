from __future__ import annotations

from pathlib import Path

import pytest

from app.reconstruction_persistence_controller import ReconstructionPersistenceController
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
        profile_id="phase9-1h-integration-profile",
        calibration_record_id="phase9-1h-integration-record",
        camera_id="phase9-1h-integration-camera",
        provenance="phase9_1h_integration",
    )


def _evidence():
    return reference_target_metric_evidence(
        evidence_id="phase9-1h-integration-target",
        reconstruction_frame_id="reconstruction-world",
        reference_id="reference-target",
        target_length_m=0.20,
        measured_length_reconstruction_units=4.0,
        target_relative_uncertainty=0.005,
        measurement_relative_uncertainty=0.015,
        provenance="phase9_1h_integration",
        validation_state=MetricEvidenceValidationState.VALIDATED,
        validation_record_id="phase9-1h-integration-validation",
    )


def test_1h_controller_save_reload_restores_authority_snapshot(qapp, tmp_path: Path) -> None:
    controller = ReconstructionPersistenceController()
    _fixture, _providers, result, snapshots = run_f5_fixture(tmp_path / "run")
    scene = controller.adopt_pipeline_result(
        result,
        (tmp_path / "source").resolve(),
        payload_snapshots_by_kind=snapshots,
    )
    snapshot = build_persistent_scene_authority_snapshot(
        scene,
        camera_calibration_profile=_profile(),
        metric_qualification_evidence=_evidence(),
    )
    saved = controller.save_scene_with_authority(
        (tmp_path / "saved").resolve(),
        authority_snapshot=snapshot,
    )
    assert controller.authority_snapshot is not None
    assert controller.authority_manifest is not None

    controller.clear_scene()
    assert controller.authority_snapshot is None
    loaded_scene = controller.load_scene(saved)
    assert loaded_scene.scene_id == scene.scene_id
    assert controller.authority_snapshot is not None
    assert controller.authority_snapshot.as_dict() == snapshot.as_dict()
    assert controller.authority_snapshot.certified_metrology is False


def test_1h_controller_legacy_scene_reload_clears_previous_authority(qapp, tmp_path: Path) -> None:
    controller = ReconstructionPersistenceController()
    _fixture, _providers, result, snapshots = run_f5_fixture(tmp_path / "run")
    scene = controller.adopt_pipeline_result(
        result,
        (tmp_path / "legacy").resolve(),
        payload_snapshots_by_kind=snapshots,
    )
    # No Phase 9.1H sidecar exists for this untouched F5 scene.
    assert controller.authority_snapshot is None
    controller.load_scene((tmp_path / "legacy" / "persistent_reconstruction_scene.json").resolve())
    assert controller.scene is not None
    assert controller.scene.scene_id == scene.scene_id
    assert controller.authority_snapshot is None
    assert controller.authority_manifest is None

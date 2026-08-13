from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.reconstruction import (
    MetricEvidenceValidationState,
    ReconstructionValidationError,
    build_persistent_scene_authority_snapshot,
    persistent_scene_from_pipeline_result,
    read_optional_persistent_scene_authority_snapshot,
    read_persistent_scene_authority_snapshot,
    reference_target_metric_evidence,
    write_persistent_reconstruction_scene,
    write_persistent_scene_authority_snapshot,
)
from core.reconstruction.calibration_management import create_manual_camera_calibration_profile
from tests.reconstruction_f5_helpers import run_f5_fixture


def _scene(tmp_path: Path):
    _fixture, _providers, result, snapshots = run_f5_fixture(tmp_path / "run")
    return persistent_scene_from_pipeline_result(result, payload_snapshots_by_kind=snapshots)


def _profile():
    return create_manual_camera_calibration_profile(
        width=640,
        height=480,
        fx=520.0,
        fy=521.0,
        cx=320.0,
        cy=240.0,
        profile_id="phase9-1h-profile",
        calibration_record_id="phase9-1h-record",
        camera_id="phase9-1h-camera",
        provenance="phase9_1h_unit_test",
        created_at_utc="2026-08-13T18:00:00+00:00",
    )


def _metric_evidence():
    return reference_target_metric_evidence(
        evidence_id="phase9-1h-reference-target",
        reconstruction_frame_id="reconstruction-world",
        reference_id="target-250mm",
        target_length_m=0.25,
        measured_length_reconstruction_units=5.0,
        target_relative_uncertainty=0.005,
        measurement_relative_uncertainty=0.02,
        provenance="phase9_1h_reference_target_test",
        validation_state=MetricEvidenceValidationState.VALIDATED,
        validation_record_id="phase9-1h-target-validation",
        created_at_utc="2026-08-13T18:01:00+00:00",
    )


def test_1h_authority_snapshot_round_trip_preserves_calibration_and_metric_evidence(tmp_path: Path) -> None:
    scene = _scene(tmp_path)
    profile = _profile()
    evidence = _metric_evidence()
    snapshot = build_persistent_scene_authority_snapshot(
        scene,
        camera_calibration_profile=profile,
        metric_qualification_evidence=evidence,
        created_at_utc="2026-08-13T18:02:00+00:00",
    )
    assert snapshot.camera_calibration_profile is not None
    assert snapshot.camera_calibration_profile.as_dict() == profile.as_dict()
    assert snapshot.metric_qualification_evidence is not None
    assert snapshot.metric_qualification_evidence.as_dict() == evidence.as_dict()
    assert snapshot.metric_coordinates_allowed is True
    assert snapshot.calibrated_metric_scale is True
    assert snapshot.certified_metrology is False

    root = (tmp_path / "persistent").resolve()
    scene_manifest = write_persistent_reconstruction_scene(scene, root)
    authority_manifest = write_persistent_scene_authority_snapshot(snapshot, root)
    loaded = read_persistent_scene_authority_snapshot(authority_manifest, expected_scene=scene)
    assert loaded.as_dict() == snapshot.as_dict()
    assert scene_manifest.name == "persistent_reconstruction_scene.json"
    assert json.loads(scene_manifest.read_text())["schema_version"] == 1


def test_1h_snapshot_is_self_contained_and_does_not_require_profile_store_file(tmp_path: Path) -> None:
    scene = _scene(tmp_path)
    snapshot = build_persistent_scene_authority_snapshot(
        scene,
        camera_calibration_profile=_profile(),
        created_at_utc="2026-08-13T18:02:00+00:00",
    )
    root = (tmp_path / "persistent").resolve()
    scene_manifest = write_persistent_reconstruction_scene(scene, root)
    write_persistent_scene_authority_snapshot(snapshot, root)

    loaded, authority_path = read_optional_persistent_scene_authority_snapshot(
        scene_manifest,
        expected_scene=scene,
    )
    assert authority_path is not None
    assert loaded is not None
    assert loaded.camera_calibration_profile is not None
    assert loaded.camera_calibration_profile.profile_id == "phase9-1h-profile"


def test_1h_legacy_scene_without_authority_sidecar_loads_as_no_persisted_authority(tmp_path: Path) -> None:
    scene = _scene(tmp_path)
    root = (tmp_path / "legacy").resolve()
    scene_manifest = write_persistent_reconstruction_scene(scene, root)
    loaded, authority_path = read_optional_persistent_scene_authority_snapshot(
        scene_manifest,
        expected_scene=scene,
    )
    assert loaded is None
    assert authority_path is None
    assert json.loads(scene_manifest.read_text())["schema_version"] == 1


def test_1h_authority_sidecar_tampering_fails_closed(tmp_path: Path) -> None:
    scene = _scene(tmp_path)
    root = (tmp_path / "tamper").resolve()
    write_persistent_reconstruction_scene(scene, root)
    manifest = write_persistent_scene_authority_snapshot(
        build_persistent_scene_authority_snapshot(scene, camera_calibration_profile=_profile()),
        root,
    )
    raw = json.loads(manifest.read_text())
    raw["camera_calibration_profile"]["profile_id"] = "tampered-profile"
    manifest.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ReconstructionValidationError, match="checksum mismatch"):
        read_persistent_scene_authority_snapshot(manifest, expected_scene=scene)


def test_1h_authority_sidecar_is_bound_to_exact_scene_identity(tmp_path: Path) -> None:
    first = _scene(tmp_path / "first")
    second = _scene(tmp_path / "second")
    # The fixture normally receives a fresh session/request identity and therefore
    # produces a different persistent-scene identity.
    assert first.scene_id != second.scene_id or first.session_id != second.session_id
    root = (tmp_path / "bound").resolve()
    manifest = write_persistent_scene_authority_snapshot(
        build_persistent_scene_authority_snapshot(first, camera_calibration_profile=_profile()),
        root,
    )
    with pytest.raises(ReconstructionValidationError, match="does not match"):
        read_persistent_scene_authority_snapshot(manifest, expected_scene=second)


def test_1h_persisted_metric_authority_never_certifies_metrology(tmp_path: Path) -> None:
    scene = _scene(tmp_path)
    snapshot = build_persistent_scene_authority_snapshot(
        scene,
        metric_qualification_evidence=_metric_evidence(),
    )
    assert snapshot.metric_qualification_assessment is not None
    assert snapshot.metric_qualification_assessment.authorizes_calibrated_metric_scale
    assert snapshot.certified_metrology is False
    payload = snapshot.as_dict()
    payload["certified_metrology"] = True
    # Recompute the checksum is deliberately irrelevant here; normalized contract
    # validation itself must reject any metrology promotion.
    with pytest.raises(ReconstructionValidationError, match="cannot claim certified metrology"):
        from core.reconstruction import persistent_scene_authority_snapshot_from_dict

        persistent_scene_authority_snapshot_from_dict(payload)


def test_1h_recomputed_file_checksum_cannot_hide_authority_fingerprint_tamper(tmp_path: Path) -> None:
    import hashlib

    scene = _scene(tmp_path)
    root = (tmp_path / "fingerprint-tamper").resolve()
    write_persistent_reconstruction_scene(scene, root)
    manifest = write_persistent_scene_authority_snapshot(
        build_persistent_scene_authority_snapshot(scene, camera_calibration_profile=_profile()),
        root,
    )
    raw = json.loads(manifest.read_text())
    raw["camera_calibration_profile"]["notes"] = "tampered after persistence"
    payload = json.dumps(raw, indent=2, sort_keys=True) + "\n"
    manifest.write_text(payload)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    (root / "persistent_reconstruction_authority.json.sha256").write_text(
        f"{digest}  persistent_reconstruction_authority.json\n"
    )
    with pytest.raises(ReconstructionValidationError, match="fingerprint"):
        read_persistent_scene_authority_snapshot(manifest, expected_scene=scene)


def test_1h_rejected_metric_evidence_remains_non_authoritative_after_round_trip(tmp_path: Path) -> None:
    from core.reconstruction import MetricEvidenceKind, MetricQualificationEvidence

    scene = _scene(tmp_path)
    evidence = MetricQualificationEvidence(
        evidence_id="phase9-1h-rejected-target",
        kind=MetricEvidenceKind.REFERENCE_TARGET,
        validation_state=MetricEvidenceValidationState.REJECTED,
        provenance="phase9_1h_rejected_test",
        rejection_reasons=("target_not_detected",),
    )
    snapshot = build_persistent_scene_authority_snapshot(
        scene,
        metric_qualification_evidence=evidence,
    )
    root = (tmp_path / "rejected").resolve()
    write_persistent_reconstruction_scene(scene, root)
    manifest = write_persistent_scene_authority_snapshot(snapshot, root)
    loaded = read_persistent_scene_authority_snapshot(manifest, expected_scene=scene)
    assert loaded.metric_qualification_assessment is not None
    assert loaded.metric_qualification_assessment.state.value == "rejected"
    assert loaded.metric_coordinates_allowed is False
    assert loaded.calibrated_metric_scale is False
    assert loaded.certified_metrology is False


def test_1h_rejected_camera_profile_cannot_be_persisted_as_active_scene_calibration(tmp_path: Path) -> None:
    from dataclasses import replace
    from core.reconstruction.calibration_profile import CameraCalibrationValidationState

    scene = _scene(tmp_path)
    rejected = replace(
        _profile(),
        validation_state=CameraCalibrationValidationState.REJECTED,
    )
    with pytest.raises(ReconstructionValidationError, match="Rejected camera-calibration evidence"):
        build_persistent_scene_authority_snapshot(
            scene,
            camera_calibration_profile=rejected,
        )

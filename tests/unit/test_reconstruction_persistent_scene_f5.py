from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.reconstruction import (
    PersistentReconstructionSceneState,
    ReconstructionArtifactAuthority,
    ReconstructionArtifactKind,
    ReconstructionPipelineRunStatus,
    ReconstructionValidationError,
    persistent_scene_from_pipeline_payload,
    persistent_scene_from_pipeline_result,
    read_persistent_reconstruction_scene,
    verify_persistent_scene_file_references,
    write_persistent_reconstruction_scene,
)
from tests.reconstruction_f5_helpers import run_f5_fixture


def test_f5_materializes_ready_scene_with_full_authority_hierarchy(tmp_path: Path) -> None:
    _fixture, _providers, result, snapshots = run_f5_fixture(tmp_path / "run")
    scene = persistent_scene_from_pipeline_result(
        result,
        payload_snapshots_by_kind=snapshots,
        created_at_utc="2026-08-09T12:00:00+00:00",
    )
    assert scene.state is PersistentReconstructionSceneState.READY
    assert scene.pipeline_status is ReconstructionPipelineRunStatus.COMPLETED
    assert scene.canonical_geometry_authoritative
    assert scene.certified_metrology is False
    assert scene.reconstruction_ui_enabled is False
    assert scene.gaussian_artifact is not None
    assert scene.nerf_artifact is not None
    assert scene.gaussian_artifact.authority is ReconstructionArtifactAuthority.APPEARANCE_ONLY
    assert scene.nerf_artifact.authority is ReconstructionArtifactAuthority.APPEARANCE_ONLY
    assert scene.gaussian_artifact.canonical_geometry_authoritative is False
    assert scene.nerf_artifact.canonical_geometry_authoritative is False
    assert len(scene.artifacts) == 9


def test_f5_scene_identity_is_deterministic_for_same_pipeline_content(tmp_path: Path) -> None:
    _fixture, _providers, result, snapshots = run_f5_fixture(tmp_path / "run", request_id="f5-stable")
    first = persistent_scene_from_pipeline_result(result, payload_snapshots_by_kind=snapshots, created_at_utc="2026-08-09T12:00:00+00:00")
    second = persistent_scene_from_pipeline_result(result, payload_snapshots_by_kind=snapshots, created_at_utc="2026-08-09T13:00:00+00:00")
    assert first.scene_id == second.scene_id
    assert first.scene_fingerprint_sha256 == second.scene_fingerprint_sha256
    assert first.created_at_utc != second.created_at_utc


def test_f5_round_trip_and_file_integrity(tmp_path: Path) -> None:
    _fixture, _providers, result, snapshots = run_f5_fixture(tmp_path / "run")
    scene = persistent_scene_from_pipeline_result(result, payload_snapshots_by_kind=snapshots)
    manifest = write_persistent_reconstruction_scene(scene, (tmp_path / "persistent").resolve())
    assert manifest.is_file()
    loaded = read_persistent_reconstruction_scene(manifest)
    assert loaded.scene_id == scene.scene_id
    assert loaded.scene_fingerprint_sha256 == scene.scene_fingerprint_sha256
    assert len(verify_persistent_scene_file_references(loaded)) >= 3


def test_f5_manifest_tampering_fails_closed(tmp_path: Path) -> None:
    _fixture, _providers, result, snapshots = run_f5_fixture(tmp_path / "run")
    scene = persistent_scene_from_pipeline_result(result, payload_snapshots_by_kind=snapshots)
    manifest = write_persistent_reconstruction_scene(scene, (tmp_path / "persistent").resolve())
    raw = json.loads(manifest.read_text())
    raw["canonical_geometry_artifact_id"] = "tampered"
    manifest.write_text(json.dumps(raw, indent=2) + "\n")
    with pytest.raises(ReconstructionValidationError):
        read_persistent_reconstruction_scene(manifest)


def test_f5_degraded_optional_appearance_is_persistable_and_explicit(tmp_path: Path) -> None:
    _fixture, _providers, result, snapshots = run_f5_fixture(tmp_path / "run", degrade_gaussian=True, request_id="f5-degraded")
    assert result.status is ReconstructionPipelineRunStatus.DEGRADED
    scene = persistent_scene_from_pipeline_result(result, payload_snapshots_by_kind=snapshots)
    assert scene.state is PersistentReconstructionSceneState.DEGRADED
    assert scene.gaussian_artifact is None
    assert scene.nerf_artifact is not None
    assert any(item.stage.value == "gaussian_appearance" for item in scene.fallback_evidence)
    assert scene.canonical_geometry_authoritative


def test_f5_failed_or_cancelled_pipeline_payload_cannot_be_promoted_to_scene(tmp_path: Path) -> None:
    _fixture, _providers, result, _snapshots = run_f5_fixture(tmp_path / "run")
    payload = result.as_dict()
    payload["status"] = "failed"
    with pytest.raises(ReconstructionValidationError):
        persistent_scene_from_pipeline_payload(payload)


def test_f5_requires_semantics_and_qc_core_stages(tmp_path: Path) -> None:
    _fixture, _providers, result, _snapshots = run_f5_fixture(tmp_path / "run")
    payload = result.as_dict()
    payload["stages"] = [stage for stage in payload["stages"] if stage["stage"] != "inspection_qc"]
    with pytest.raises(ReconstructionValidationError):
        persistent_scene_from_pipeline_payload(payload)


def test_f5_neural_payload_snapshots_do_not_promote_authority(tmp_path: Path) -> None:
    _fixture, _providers, result, snapshots = run_f5_fixture(tmp_path / "run")
    snapshots["gaussian_scene"] = dict(snapshots["gaussian_scene"], canonical_geometry_authoritative=True)
    scene = persistent_scene_from_pipeline_result(result, payload_snapshots_by_kind=snapshots)
    gaussian = next(item for item in scene.artifacts if item.kind is ReconstructionArtifactKind.GAUSSIAN_SCENE)
    assert gaussian.authority is ReconstructionArtifactAuthority.APPEARANCE_ONLY
    assert gaussian.canonical_geometry_authoritative is False

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.reconstruction import (
    GaussianOptimizationState,
    GaussianSceneArtifact,
    GaussianSceneRole,
    MetricGeometryStatus,
    current_process_gaussian_heavy_modules,
    persist_gaussian_representation,
    read_persistent_reconstruction_scene,
    reload_persistent_gaussian_representation,
    verify_persistent_scene_file_references,
)
from core.reconstruction.errors import ReconstructionValidationError
from tests.phase9_4d_helpers import make_phase9_4d_source


def _persist(tmp_path: Path):
    scene, artifact, preview, metrics, manifest, summary = make_phase9_4d_source(tmp_path)
    result = persist_gaussian_representation(
        base_scene=scene,
        gaussian_artifact=artifact,
        preview_path=preview,
        qualification_metrics_path=metrics,
        source_artifact_manifest_path=manifest,
        source_optimization_summary_path=summary,
        expected_dataset_fingerprint_sha256=artifact.dataset_fingerprint_sha256,
        output_dir=(tmp_path / "persistent").resolve(),
    )
    return scene, artifact, result


def test_4d_persists_gaussian_into_existing_scene_contract_without_authority_promotion(tmp_path: Path) -> None:
    base, artifact, result = _persist(tmp_path)
    assert result.scene.scene_id != base.scene_id
    assert result.parent_scene_id == base.scene_id
    assert result.scene.canonical_geometry_artifact_id == base.canonical_geometry_artifact_id
    assert result.scene.gaussian_artifact is not None
    assert result.scene.gaussian_artifact.authority.value == "appearance_only"
    assert result.artifact.scene_role is GaussianSceneRole.APPEARANCE_ONLY
    assert result.artifact.dataset_fingerprint_sha256 == artifact.dataset_fingerprint_sha256
    assert result.canonical_geometry_authoritative is True
    assert result.gaussian_canonical_geometry_authoritative is False
    assert result.metric_scale_authoritative is False
    assert result.measurement_authoritative is False
    assert result.qc_authoritative is False
    assert result.certified_metrology is False
    assert result.backend_started is False


def test_4d_materializes_self_owned_gaussian_files_and_durable_artifact_manifest(tmp_path: Path) -> None:
    _base, source_artifact, result = _persist(tmp_path)
    root = Path(result.scene_manifest_path).parent
    gaussian = root / "appearance" / "gaussian"
    expected = {
        "scene.pt",
        "preview.png",
        "qualification_metrics.json",
        "artifact_manifest.json",
        "source_artifact_manifest.json",
        "source_optimization_summary.json",
    }
    assert expected.issubset({item.name for item in gaussian.iterdir()})
    payload = json.loads((gaussian / "artifact_manifest.json").read_text(encoding="utf-8"))
    durable = GaussianSceneArtifact.from_payload(payload)
    assert durable.artifact_dir == str(gaussian.resolve())
    assert durable.scene_path == str((gaussian / "scene.pt").resolve())
    assert durable.request_id == source_artifact.request_id
    assert durable.dataset_fingerprint_sha256 == source_artifact.dataset_fingerprint_sha256
    assert Path(durable.scene_path).read_bytes() == Path(source_artifact.scene_path).read_bytes()


def test_4d_roundtrip_reload_and_file_reference_integrity_pass(tmp_path: Path) -> None:
    base, _artifact, result = _persist(tmp_path)
    loaded_scene = read_persistent_reconstruction_scene(result.scene_manifest_path)
    refs = verify_persistent_scene_file_references(loaded_scene)
    reloaded = reload_persistent_gaussian_representation(result.scene_manifest_path)
    assert reloaded.state.value == "reloaded"
    assert reloaded.restart_reload_verified is True
    assert reloaded.parent_scene_id == base.scene_id
    assert reloaded.scene.scene_id == result.scene.scene_id
    assert len(reloaded.gaussian_file_references) >= 5
    assert len(refs) >= len(base.file_references) + 5
    assert current_process_gaussian_heavy_modules() == ()


def test_4d_preserves_base_scene_object_and_canonical_artifact_exactly(tmp_path: Path) -> None:
    base, _artifact, result = _persist(tmp_path)
    canonical_before = next(item for item in base.artifacts if item.artifact_id == base.canonical_geometry_artifact_id)
    canonical_after = next(
        item for item in result.scene.artifacts if item.artifact_id == result.scene.canonical_geometry_artifact_id
    )
    assert canonical_after == canonical_before
    assert base.gaussian_artifact is None
    assert result.source_scene_unchanged is True
    assert base.scene_fingerprint_sha256 == result.parent_scene_fingerprint_sha256


def test_4d_tampered_persisted_scene_pt_is_rejected_on_restart(tmp_path: Path) -> None:
    _base, _artifact, result = _persist(tmp_path)
    Path(result.artifact.scene_path).write_bytes(b"tampered\n")
    with pytest.raises(ReconstructionValidationError, match="size changed|checksum"):
        reload_persistent_gaussian_representation(result.scene_manifest_path)


def test_4d_rejects_dataset_fingerprint_mismatch(tmp_path: Path) -> None:
    scene, artifact, preview, metrics, manifest, summary = make_phase9_4d_source(tmp_path)
    with pytest.raises(ReconstructionValidationError, match="dataset fingerprint"):
        persist_gaussian_representation(
            base_scene=scene,
            gaussian_artifact=artifact,
            preview_path=preview,
            qualification_metrics_path=metrics,
            source_artifact_manifest_path=manifest,
            source_optimization_summary_path=summary,
            expected_dataset_fingerprint_sha256="b" * 64,
            output_dir=(tmp_path / "persistent").resolve(),
        )


def test_4d_rejects_artifact_manifest_that_does_not_match_source_artifact(tmp_path: Path) -> None:
    scene, artifact, preview, metrics, manifest, summary = make_phase9_4d_source(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["training_steps"] = 25
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ReconstructionValidationError, match="does not match"):
        persist_gaussian_representation(
            base_scene=scene,
            gaussian_artifact=artifact,
            preview_path=preview,
            qualification_metrics_path=metrics,
            source_artifact_manifest_path=manifest,
            source_optimization_summary_path=summary,
            output_dir=(tmp_path / "persistent").resolve(),
        )


def test_4d_rejects_non_cuda_qualification_metrics(tmp_path: Path) -> None:
    scene, artifact, preview, metrics, manifest, summary = make_phase9_4d_source(tmp_path)
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    payload["real_cuda_execution"] = False
    metrics.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ReconstructionValidationError, match="real CUDA"):
        persist_gaussian_representation(
            base_scene=scene,
            gaussian_artifact=artifact,
            preview_path=preview,
            qualification_metrics_path=metrics,
            source_artifact_manifest_path=manifest,
            source_optimization_summary_path=summary,
            output_dir=(tmp_path / "persistent").resolve(),
        )


def test_4d_rejects_scale_mismatch_against_canonical_scene(tmp_path: Path) -> None:
    scene, artifact, preview, metrics, manifest, summary = make_phase9_4d_source(tmp_path)
    mismatched = GaussianSceneArtifact(
        request_id=artifact.request_id,
        backend_id=artifact.backend_id,
        backend_version=artifact.backend_version,
        backend_provenance=artifact.backend_provenance,
        optimization_state=GaussianOptimizationState.OPTIMIZED,
        scene_role=GaussianSceneRole.APPEARANCE_ONLY,
        artifact_dir=artifact.artifact_dir,
        scene_path=artifact.scene_path,
        dataset_fingerprint_sha256=artifact.dataset_fingerprint_sha256,
        input_frame_count=artifact.input_frame_count,
        gaussian_count=artifact.gaussian_count,
        training_steps=artifact.training_steps,
        training_duration_s=artifact.training_duration_s,
        render_capable=True,
        device=artifact.device,
        peak_vram_mib=artifact.peak_vram_mib,
        pose_convention=artifact.pose_convention,
        reconstruction_frame_id=artifact.reconstruction_frame_id,
        metric_geometry_status=MetricGeometryStatus.CALIBRATED_METRIC,
        unit_to_metre=1.0,
        relative_scale_uncertainty=0.01,
    )
    manifest.write_text(json.dumps(mismatched.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ReconstructionValidationError, match="scale state"):
        persist_gaussian_representation(
            base_scene=scene,
            gaussian_artifact=mismatched,
            preview_path=preview,
            qualification_metrics_path=metrics,
            source_artifact_manifest_path=manifest,
            source_optimization_summary_path=summary,
            output_dir=(tmp_path / "persistent").resolve(),
        )


def test_4d_refuses_silent_overwrite_of_existing_gaussian_artifact(tmp_path: Path) -> None:
    _base, artifact, first = _persist(tmp_path)
    with pytest.raises(ReconstructionValidationError, match="overwrite"):
        persist_gaussian_representation(
            base_scene=first.scene,
            gaussian_artifact=artifact,
            preview_path=Path(artifact.artifact_dir) / "preview.png",
            qualification_metrics_path=Path(artifact.artifact_dir) / "qualification_metrics.json",
            source_artifact_manifest_path=Path(artifact.artifact_dir) / "artifact_manifest.json",
            output_dir=(tmp_path / "second").resolve(),
        )

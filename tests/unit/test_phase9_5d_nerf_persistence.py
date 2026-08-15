from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.reconstruction import (
    NerfSceneRole,
    current_process_nerf_heavy_modules,
    persist_nerf_representation,
    read_persistent_reconstruction_scene,
    reload_persistent_nerf_representation,
    verify_persistent_scene_file_references,
)
from core.reconstruction.errors import ReconstructionValidationError
from tests.phase9_5d_helpers import make_phase9_5d_source


def _persist(tmp_path: Path):
    scene, artifact, preview, runtime, manifest, summary, console = make_phase9_5d_source(tmp_path)
    result = persist_nerf_representation(
        base_scene=scene,
        nerf_artifact=artifact,
        preview_path=preview,
        runtime_evidence_path=runtime,
        source_optimization_manifest_path=manifest,
        source_optimization_summary_path=summary,
        source_worker_console_path=console,
        expected_dataset_fingerprint_sha256=artifact.dataset_fingerprint_sha256,
        output_dir=(tmp_path / "persistent").resolve(),
    )
    return scene, artifact, result


def test_5d_persists_nerf_without_authority_promotion(tmp_path: Path) -> None:
    base, artifact, result = _persist(tmp_path)
    assert result.scene.scene_id != base.scene_id
    assert result.parent_scene_id == base.scene_id
    assert result.scene.canonical_geometry_artifact_id == base.canonical_geometry_artifact_id
    assert result.scene.nerf_artifact is not None
    assert result.scene.nerf_artifact.authority.value == "appearance_only"
    assert result.artifact.scene_role is NerfSceneRole.APPEARANCE_ONLY
    assert result.artifact.dataset_fingerprint_sha256 == artifact.dataset_fingerprint_sha256
    assert result.canonical_geometry_authoritative is True
    assert result.nerf_canonical_geometry_authoritative is False
    assert result.metric_scale_authoritative is False
    assert result.measurement_authoritative is False
    assert result.qc_authoritative is False
    assert result.certified_metrology is False
    assert result.backend_started is False
    assert result.rendering_started is False


def test_5d_materializes_self_owned_checkpoint_config_preview_dataset_and_manifests(tmp_path: Path) -> None:
    _base, source, result = _persist(tmp_path)
    root = Path(result.scene_manifest_path).parent / "appearance" / "nerf"
    expected = {
        "artifact_manifest.json",
        "config.yml",
        "preview.png",
        "e2_runtime_evidence.json",
        "source_runtime_evidence.json",
        "source_optimization_manifest.json",
        "source_optimization_summary.json",
        "source_worker_console.log",
        "runtime_load_spec.json",
        "dataset_integrity.json",
    }
    assert expected.issubset({item.name for item in root.iterdir()})
    assert Path(result.artifact.checkpoint_path).is_file()
    assert Path(result.dataset_root, "transforms.json").is_file()
    assert Path(result.dataset_root, "images", "0000.png").is_file()
    assert Path(result.artifact.config_path).read_bytes() == Path(source.config_path).read_bytes()
    assert Path(result.artifact.checkpoint_path).read_bytes() == Path(source.checkpoint_path).read_bytes()


def test_5d_runtime_evidence_and_load_spec_are_durable_and_backend_deferred(tmp_path: Path) -> None:
    _base, _source, result = _persist(tmp_path)
    root = Path(result.artifact.artifact_dir)
    evidence = json.loads((root / "e2_runtime_evidence.json").read_text(encoding="utf-8"))
    spec = json.loads(Path(result.runtime_load_spec_path).read_text(encoding="utf-8"))
    assert Path(evidence["preview_path"]).parent == root
    assert evidence["phase9_5d_backend_started"] is False
    assert evidence["phase9_5d_rendering_started"] is False
    assert spec["load_strategy"] == "isolated_staged_rebind_from_immutable_training_config"
    assert spec["backend_start_performed"] is False
    assert spec["rendering_performed"] is False
    assert spec["interactive_renderer_deferred_to"] == "9.5E"


def test_5d_roundtrip_reload_and_integrity_pass_without_heavy_backend(tmp_path: Path) -> None:
    base, _artifact, result = _persist(tmp_path)
    loaded = read_persistent_reconstruction_scene(result.scene_manifest_path)
    refs = verify_persistent_scene_file_references(loaded)
    reloaded = reload_persistent_nerf_representation(result.scene_manifest_path)
    assert reloaded.state.value == "reloaded"
    assert reloaded.restart_reload_verified is True
    assert reloaded.parent_scene_id == base.scene_id
    assert reloaded.scene.scene_id == result.scene.scene_id
    assert len(reloaded.nerf_file_references) >= 8
    assert len(refs) >= len(reloaded.nerf_file_references)
    assert current_process_nerf_heavy_modules() == ()
    assert reloaded.backend_started is False


def test_5d_preserves_base_scene_and_canonical_artifact_exactly(tmp_path: Path) -> None:
    base, _artifact, result = _persist(tmp_path)
    canonical_before = next(item for item in base.artifacts if item.artifact_id == base.canonical_geometry_artifact_id)
    canonical_after = next(item for item in result.scene.artifacts if item.artifact_id == result.scene.canonical_geometry_artifact_id)
    assert canonical_after == canonical_before
    assert base.nerf_artifact is None
    assert result.source_scene_unchanged is True
    assert base.scene_fingerprint_sha256 == result.parent_scene_fingerprint_sha256


def test_5d_tampered_checkpoint_is_rejected_on_reload(tmp_path: Path) -> None:
    _base, _artifact, result = _persist(tmp_path)
    Path(result.artifact.checkpoint_path).write_bytes(b"tampered\n")
    with pytest.raises(ReconstructionValidationError, match="size changed|checksum"):
        reload_persistent_nerf_representation(result.scene_manifest_path)


def test_5d_tampered_dataset_file_is_rejected_even_when_not_top_level_scene_reference(tmp_path: Path) -> None:
    _base, _artifact, result = _persist(tmp_path)
    Path(result.dataset_root, "images", "0000.png").write_bytes(b"tampered-image")
    with pytest.raises(ReconstructionValidationError, match="dataset file size changed|dataset file checksum changed"):
        reload_persistent_nerf_representation(result.scene_manifest_path)


def test_5d_rejects_dataset_fingerprint_mismatch(tmp_path: Path) -> None:
    scene, artifact, preview, runtime, manifest, summary, console = make_phase9_5d_source(tmp_path)
    with pytest.raises(ReconstructionValidationError, match="dataset fingerprint"):
        persist_nerf_representation(
            base_scene=scene,
            nerf_artifact=artifact,
            preview_path=preview,
            runtime_evidence_path=runtime,
            source_optimization_manifest_path=manifest,
            source_optimization_summary_path=summary,
            source_worker_console_path=console,
            expected_dataset_fingerprint_sha256="b" * 64,
            output_dir=(tmp_path / "persistent").resolve(),
        )


def test_5d_rejects_tampered_9_5c_optimization_manifest_hash(tmp_path: Path) -> None:
    scene, artifact, preview, runtime, manifest, summary, console = make_phase9_5d_source(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["checkpoint_sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ReconstructionValidationError, match="checkpoint_sha256 mismatch"):
        persist_nerf_representation(
            base_scene=scene,
            nerf_artifact=artifact,
            preview_path=preview,
            runtime_evidence_path=runtime,
            source_optimization_manifest_path=manifest,
            output_dir=(tmp_path / "persistent").resolve(),
        )


def test_5d_refuses_silent_overwrite_of_existing_nerf(tmp_path: Path) -> None:
    _base, source, first = _persist(tmp_path)
    root = Path(source.artifact_dir)
    with pytest.raises(ReconstructionValidationError, match="overwrite"):
        persist_nerf_representation(
            base_scene=first.scene,
            nerf_artifact=source,
            preview_path=root / "preview.png",
            runtime_evidence_path=root / "e2_runtime_evidence.json",
            source_optimization_manifest_path=tmp_path / "phase9_5c_optimization_manifest.json",
            output_dir=(tmp_path / "second").resolve(),
        )

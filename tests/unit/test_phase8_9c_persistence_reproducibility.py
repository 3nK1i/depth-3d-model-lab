from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from core.reconstruction import (
    PERSISTENT_RECONSTRUCTION_SCENE_CHECKSUM,
    PERSISTENT_RECONSTRUCTION_SCENE_MANIFEST,
    ReconstructionArtifactAuthority,
    ReconstructionValidationError,
    check_reconstruction_engine_resume_compatibility,
    read_persistent_reconstruction_scene,
    read_reconstruction_engine_checkpoint,
    verify_persistent_scene_file_references,
    write_persistent_reconstruction_scene,
)
from tests.reconstruction_f5_helpers import run_f5_fixture
from tests.reconstruction_g3_helpers import write_checkpoint_fixture


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rewrite_scene_checksum(manifest: Path) -> None:
    digest = _sha256(manifest)
    (manifest.parent / PERSISTENT_RECONSTRUCTION_SCENE_CHECKSUM).write_text(
        f"{digest}  {PERSISTENT_RECONSTRUCTION_SCENE_MANIFEST}\n",
        encoding="utf-8",
    )


def test_c_scene_serialization_is_byte_reproducible_and_preserves_authority(tmp_path: Path) -> None:
    _fixture, _providers, result, snapshots = run_f5_fixture(
        tmp_path / "run", request_id="phase8-9c-reproducible"
    )
    from core.reconstruction import persistent_scene_from_pipeline_result

    scene = persistent_scene_from_pipeline_result(
        result,
        payload_snapshots_by_kind=snapshots,
        created_at_utc="2026-08-12T11:30:00+00:00",
    )
    first = write_persistent_reconstruction_scene(scene, (tmp_path / "scene-a").resolve())
    second = write_persistent_reconstruction_scene(scene, (tmp_path / "scene-b").resolve())

    assert first.read_bytes() == second.read_bytes()
    assert _sha256(first) == _sha256(second)
    assert (first.parent / PERSISTENT_RECONSTRUCTION_SCENE_CHECKSUM).read_text() == (
        second.parent / PERSISTENT_RECONSTRUCTION_SCENE_CHECKSUM
    ).read_text()

    loaded = read_persistent_reconstruction_scene(second)
    assert loaded.scene_id == scene.scene_id
    assert loaded.scene_fingerprint_sha256 == scene.scene_fingerprint_sha256
    assert loaded.canonical_geometry_authoritative is True
    assert loaded.certified_metrology is False
    assert loaded.gaussian_artifact is not None
    assert loaded.nerf_artifact is not None
    assert loaded.gaussian_artifact.authority is ReconstructionArtifactAuthority.APPEARANCE_ONLY
    assert loaded.nerf_artifact.authority is ReconstructionArtifactAuthority.APPEARANCE_ONLY
    assert verify_persistent_scene_file_references(loaded)


def test_c_manifest_tamper_fails_checksum_before_scene_adoption(tmp_path: Path) -> None:
    _fixture, _providers, result, snapshots = run_f5_fixture(tmp_path / "run")
    from core.reconstruction import persistent_scene_from_pipeline_result

    scene = persistent_scene_from_pipeline_result(result, payload_snapshots_by_kind=snapshots)
    manifest = write_persistent_reconstruction_scene(scene, (tmp_path / "scene").resolve())
    manifest.write_text(manifest.read_text(encoding="utf-8") + " ", encoding="utf-8")

    with pytest.raises(ReconstructionValidationError, match="checksum mismatch"):
        read_persistent_reconstruction_scene(manifest)


def test_c_semantic_tamper_fails_even_if_manifest_sidecar_is_recomputed(tmp_path: Path) -> None:
    _fixture, _providers, result, snapshots = run_f5_fixture(tmp_path / "run")
    from core.reconstruction import persistent_scene_from_pipeline_result

    scene = persistent_scene_from_pipeline_result(result, payload_snapshots_by_kind=snapshots)
    manifest = write_persistent_reconstruction_scene(scene, (tmp_path / "scene").resolve())
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["canonical_geometry_artifact_id"] = "tampered-canonical-id"
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _rewrite_scene_checksum(manifest)

    with pytest.raises(ReconstructionValidationError, match="canonical geometry ID"):
        read_persistent_reconstruction_scene(manifest)


def test_c_referenced_artifact_missing_size_and_same_size_content_changes_fail_closed(tmp_path: Path) -> None:
    _fixture, _providers, result, snapshots = run_f5_fixture(tmp_path / "run")
    from core.reconstruction import persistent_scene_from_pipeline_result

    scene = persistent_scene_from_pipeline_result(result, payload_snapshots_by_kind=snapshots)
    manifest = write_persistent_reconstruction_scene(scene, (tmp_path / "scene").resolve())
    loaded = read_persistent_reconstruction_scene(manifest)
    refs = verify_persistent_scene_file_references(loaded)
    assert refs

    target = Path(refs[0].path)
    original = target.read_bytes()

    target.unlink()
    with pytest.raises(ReconstructionValidationError, match="referenced file is missing"):
        verify_persistent_scene_file_references(loaded)
    target.write_bytes(original)

    target.write_bytes(original + b"size-drift")
    with pytest.raises(ReconstructionValidationError, match="file size changed"):
        verify_persistent_scene_file_references(loaded)
    target.write_bytes(original)

    replacement = bytes((value ^ 0x01) for value in original)
    assert len(replacement) == len(original)
    target.write_bytes(replacement)
    with pytest.raises(ReconstructionValidationError, match="file checksum changed"):
        verify_persistent_scene_file_references(loaded)
    target.write_bytes(original)
    verify_persistent_scene_file_references(loaded)


def test_c_checkpoint_reload_is_stable_and_provider_or_payload_drift_is_incompatible(tmp_path: Path) -> None:
    providers, _request, plan, checkpoint, manifest = write_checkpoint_fixture(tmp_path / "checkpoint")
    loaded = read_reconstruction_engine_checkpoint(manifest)
    assert loaded.checkpoint_fingerprint_sha256 == checkpoint.checkpoint_fingerprint_sha256
    assert loaded.stage_results == checkpoint.stage_results

    compatible = check_reconstruction_engine_resume_compatibility(
        loaded,
        plan=plan,
        providers=providers,
        available_payload_refs=loaded.payload_refs,
    )
    assert compatible.compatible

    drifted = list(providers)
    descriptor = drifted[-1]
    from dataclasses import replace

    drifted[-1] = replace(descriptor, provider_version=descriptor.provider_version + "-drift")
    provider_drift = check_reconstruction_engine_resume_compatibility(
        loaded,
        plan=plan,
        providers=tuple(drifted),
        available_payload_refs=loaded.payload_refs,
    )
    assert not provider_drift.compatible
    assert any("Provider versions" in reason for reason in provider_drift.reasons)

    missing_payload = check_reconstruction_engine_resume_compatibility(
        loaded,
        plan=plan,
        providers=providers,
        available_payload_refs=loaded.payload_refs[:-1],
    )
    assert not missing_payload.compatible
    assert any("payload" in reason.lower() for reason in missing_payload.reasons)


def test_c_checkpoint_manifest_tamper_fails_closed(tmp_path: Path) -> None:
    _providers, _request, _plan, _checkpoint, manifest = write_checkpoint_fixture(tmp_path / "checkpoint")
    raw = bytearray(manifest.read_bytes())
    raw[-2] = ord(" ") if raw[-2] != ord(" ") else ord("\t")
    manifest.write_bytes(bytes(raw))

    with pytest.raises(ReconstructionValidationError, match="SHA-256 verification failed"):
        read_reconstruction_engine_checkpoint(manifest)

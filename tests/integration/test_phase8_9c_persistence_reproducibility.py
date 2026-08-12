from __future__ import annotations

from pathlib import Path

from core.reconstruction import (
    ReconstructionArtifactAuthority,
    persistent_scene_from_pipeline_result,
    read_persistent_reconstruction_scene,
    verify_persistent_scene_file_references,
    write_persistent_reconstruction_scene,
)
from tests.reconstruction_f5_helpers import run_f5_fixture


def test_c_integrated_scene_save_reload_resave_keeps_identity_refs_and_authority(tmp_path: Path) -> None:
    _fixture, _providers, result, snapshots = run_f5_fixture(
        tmp_path / "run", request_id="phase8-9c-integrated"
    )
    original = persistent_scene_from_pipeline_result(
        result,
        payload_snapshots_by_kind=snapshots,
        created_at_utc="2026-08-12T11:45:00+00:00",
    )

    first_manifest = write_persistent_reconstruction_scene(original, (tmp_path / "first").resolve())
    first = read_persistent_reconstruction_scene(first_manifest)
    first_refs = verify_persistent_scene_file_references(first)

    second_manifest = write_persistent_reconstruction_scene(first, (tmp_path / "second").resolve())
    second = read_persistent_reconstruction_scene(second_manifest)
    second_refs = verify_persistent_scene_file_references(second)

    assert second.scene_id == first.scene_id == original.scene_id
    assert second.scene_fingerprint_sha256 == first.scene_fingerprint_sha256
    assert second.as_dict() == first.as_dict()
    assert second_refs == first_refs
    assert second.canonical_geometry_authoritative
    assert second.certified_metrology is False
    assert second.gaussian_artifact is not None
    assert second.nerf_artifact is not None
    assert second.gaussian_artifact.authority is ReconstructionArtifactAuthority.APPEARANCE_ONLY
    assert second.nerf_artifact.authority is ReconstructionArtifactAuthority.APPEARANCE_ONLY

from pathlib import Path

from core.reconstruction import (
    PersistentReconstructionSceneState,
    persistent_scene_from_pipeline_result,
    read_persistent_reconstruction_scene,
    verify_persistent_scene_file_references,
    write_persistent_reconstruction_scene,
)
from tests.reconstruction_f5_helpers import run_f5_fixture


def test_f5_full_f1_to_f4_result_persists_and_reloads(tmp_path: Path) -> None:
    _fixture, providers, result, snapshots = run_f5_fixture(tmp_path / "run", request_id="phase8-7f5-integration")
    scene = persistent_scene_from_pipeline_result(result, payload_snapshots_by_kind=snapshots)
    manifest = write_persistent_reconstruction_scene(scene, (tmp_path / "scene").resolve())
    reloaded = read_persistent_reconstruction_scene(manifest)
    verify_persistent_scene_file_references(reloaded)
    assert reloaded.state is PersistentReconstructionSceneState.READY
    assert reloaded.canonical_geometry_authoritative
    assert reloaded.gaussian_artifact is not None
    assert reloaded.nerf_artifact is not None
    assert {item.provider_id for item in reloaded.provider_provenance} == {item.provider_id for item in result.provider_provenance}
    assert len(providers) >= len(reloaded.provider_provenance)

from pathlib import Path

from core.reconstruction import (
    persistent_scene_from_pipeline_result,
    read_persistent_reconstruction_scene,
    write_persistent_reconstruction_scene,
)
from tests.reconstruction_f5_helpers import run_f5_fixture


def test_f5_repeated_persist_reload_keeps_stable_scene_identity(tmp_path: Path) -> None:
    _fixture, _providers, result, snapshots = run_f5_fixture(tmp_path / "run", request_id="phase8-7f5-stress")
    expected = None
    for index in range(20):
        scene = persistent_scene_from_pipeline_result(result, payload_snapshots_by_kind=snapshots)
        manifest = write_persistent_reconstruction_scene(scene, (tmp_path / f"scene-{index}").resolve())
        loaded = read_persistent_reconstruction_scene(manifest)
        expected = expected or loaded.scene_fingerprint_sha256
        assert loaded.scene_fingerprint_sha256 == expected

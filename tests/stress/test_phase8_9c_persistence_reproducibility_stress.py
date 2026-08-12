from __future__ import annotations

from pathlib import Path

from core.reconstruction import (
    persistent_scene_from_pipeline_result,
    read_persistent_reconstruction_scene,
    verify_persistent_scene_file_references,
    write_persistent_reconstruction_scene,
)
from tests.reconstruction_f5_helpers import run_f5_fixture


def test_c_repeated_save_reload_verify_keeps_exact_scene_identity(tmp_path: Path) -> None:
    _fixture, _providers, result, snapshots = run_f5_fixture(
        tmp_path / "run", request_id="phase8-9c-stress"
    )
    scene = persistent_scene_from_pipeline_result(
        result,
        payload_snapshots_by_kind=snapshots,
        created_at_utc="2026-08-12T12:00:00+00:00",
    )
    expected_id = scene.scene_id
    expected_fingerprint = scene.scene_fingerprint_sha256

    current = scene
    for index in range(30):
        manifest = write_persistent_reconstruction_scene(
            current, (tmp_path / f"scene-{index:02d}").resolve()
        )
        current = read_persistent_reconstruction_scene(manifest)
        verify_persistent_scene_file_references(current)
        assert current.scene_id == expected_id
        assert current.scene_fingerprint_sha256 == expected_fingerprint
        assert current.certified_metrology is False

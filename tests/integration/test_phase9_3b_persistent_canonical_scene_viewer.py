from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.reconstruction_canonical_geometry import load_canonical_viewer_mesh
from core.reconstruction import (
    persistent_scene_from_pipeline_result,
    read_persistent_reconstruction_scene,
    verify_persistent_scene_file_references,
    write_persistent_reconstruction_scene,
)
from tests.reconstruction_f5_helpers import run_f5_fixture


def _persisted_scene(tmp_path: Path):
    _fixture, _providers, result, snapshots = run_f5_fixture(
        tmp_path / "run", request_id="phase9-3b-integration"
    )
    surface = (tmp_path / "canonical.obj").resolve()
    surface.write_text(
        "# canonical viewer fixture\n"
        "v -1 -1 0 1 0 0\n"
        "v 1 -1 0 0 1 0\n"
        "v 1 1 0 0 0 1\n"
        "v -1 1 0 1 1 0\n"
        "f 1 2 3\n"
        "f 1 3 4\n",
        encoding="utf-8",
    )
    snapshots = dict(snapshots)
    snapshots["surface_mesh"] = {
        "surface_obj_path": str(surface),
        "surface_vertex_count": 4,
        "surface_face_count": 2,
        "metric_geometry_status": "arbitrary",
        "canonical_geometry_authoritative": True,
        "certified_metrology": False,
    }
    scene = persistent_scene_from_pipeline_result(result, payload_snapshots_by_kind=snapshots)
    manifest = write_persistent_reconstruction_scene(scene, (tmp_path / "persistent").resolve())
    return manifest


def test_3b_persistent_roundtrip_verifies_then_loads_same_canonical_surface(tmp_path: Path) -> None:
    manifest = _persisted_scene(tmp_path)
    loaded = read_persistent_reconstruction_scene(manifest)
    refs = verify_persistent_scene_file_references(loaded)
    assert any(item.role == "surface_obj_path" for item in refs)
    mesh = load_canonical_viewer_mesh(loaded)
    assert mesh is not None
    assert mesh.scene_id == loaded.scene_id
    assert mesh.artifact_id == loaded.canonical_geometry_artifact_id
    assert mesh.vertex_count == 4
    assert mesh.face_count == 2
    assert loaded.canonical_geometry_authoritative is True
    assert loaded.certified_metrology is False


def test_3b_reloaded_scene_produces_deterministic_presentation_arrays(tmp_path: Path) -> None:
    manifest = _persisted_scene(tmp_path)
    first = read_persistent_reconstruction_scene(manifest)
    second = read_persistent_reconstruction_scene(manifest)
    first_mesh = load_canonical_viewer_mesh(first)
    second_mesh = load_canonical_viewer_mesh(second)
    assert first_mesh is not None and second_mesh is not None
    assert first.scene_fingerprint_sha256 == second.scene_fingerprint_sha256
    np.testing.assert_array_equal(first_mesh.vertices_viewer, second_mesh.vertices_viewer)
    np.testing.assert_array_equal(first_mesh.faces, second_mesh.faces)
    np.testing.assert_array_equal(first_mesh.vertex_colors_rgba, second_mesh.vertex_colors_rgba)


def test_3b_missing_surface_fails_closed_without_changing_scene_authority(tmp_path: Path) -> None:
    manifest = _persisted_scene(tmp_path)
    loaded = read_persistent_reconstruction_scene(manifest)
    reference = next(item for item in loaded.file_references if item.role == "surface_obj_path")
    Path(reference.path).unlink()
    with pytest.raises(OSError, match="missing"):
        load_canonical_viewer_mesh(loaded)
    assert loaded.canonical_geometry_authoritative is True
    assert loaded.certified_metrology is False

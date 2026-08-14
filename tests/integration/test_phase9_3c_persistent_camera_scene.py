from __future__ import annotations

from pathlib import Path

import numpy as np

from app.reconstruction_canonical_cameras import (
    build_canonical_camera_trajectory_snapshot,
    load_persistent_canonical_camera_trajectory,
)
from app.reconstruction_canonical_geometry import load_canonical_viewer_mesh
from core.reconstruction import (
    persistent_scene_from_pipeline_result,
    read_persistent_reconstruction_scene,
    write_persistent_reconstruction_scene,
)
from core.reconstruction.orchestration import ReconstructionArtifactKind
from tests.reconstruction_f5_helpers import run_f5_fixture


def _scene(tmp_path: Path, *, with_camera_snapshot: bool = True):
    _fixture, _providers, result, snapshots = run_f5_fixture(
        tmp_path / "run", request_id="phase9-3c-integration"
    )
    base = persistent_scene_from_pipeline_result(result, payload_snapshots_by_kind=snapshots)
    camera_artifact = next(
        item for item in base.artifacts if item.kind is ReconstructionArtifactKind.CAMERA_TRAJECTORY
    )

    surface = (tmp_path / "canonical_surface.obj").resolve()
    surface.write_text(
        "v -1 -1 -0.5 1 0 0\n"
        "v 1 -1 -0.5 0 1 0\n"
        "v 1 1 -0.5 0 0 1\n"
        "v -1 1 -0.5 1 1 0\n"
        "v -1 -1 0.5 1 0 1\n"
        "v 1 -1 0.5 0 1 1\n"
        "v 1 1 0.5 1 1 1\n"
        "v -1 1 0.5 0.5 0.5 0.5\n"
        "f 1 2 3\nf 1 3 4\nf 5 7 6\nf 5 8 7\n"
        "f 1 5 6\nf 1 6 2\nf 2 6 7\nf 2 7 3\n"
        "f 3 7 8\nf 3 8 4\nf 4 8 5\nf 4 5 1\n",
        encoding="utf-8",
    )
    updated = dict(snapshots)
    updated["surface_mesh"] = {
        "surface_obj_path": str(surface),
        "surface_vertex_count": 8,
        "surface_face_count": 12,
        "metric_geometry_status": "arbitrary",
        "canonical_geometry_authoritative": True,
        "certified_metrology": False,
    }
    if with_camera_snapshot:
        matrices = []
        for index, x in enumerate((-1.5, -0.75, 0.0, 0.75, 1.5)):
            matrix = np.eye(4, dtype=np.float64)
            matrix[:3, 3] = np.asarray([x, 0.0, -1.4 + 0.15 * index])
            matrices.append(matrix)
        matrices = np.asarray(matrices)
        updated["camera_trajectory"] = build_canonical_camera_trajectory_snapshot(
            matrices_reconstruction=matrices,
            camera_ids=tuple(f"cam-{index}" for index in range(matrices.shape[0])),
            pose_layers=("optimized",) * matrices.shape[0],
            provider_id=camera_artifact.provider_id,
            reconstruction_frame_id="reconstruction-world",
        )
    return persistent_scene_from_pipeline_result(result, payload_snapshots_by_kind=updated)


def test_3c_persistent_roundtrip_preserves_surface_and_camera_context(tmp_path: Path) -> None:
    scene = _scene(tmp_path)
    manifest = write_persistent_reconstruction_scene(scene, (tmp_path / "persistent").resolve())
    loaded = read_persistent_reconstruction_scene(manifest)

    mesh = load_canonical_viewer_mesh(loaded)
    cameras = load_persistent_canonical_camera_trajectory(loaded)
    assert mesh is not None
    assert cameras is not None
    assert mesh.scene_id == cameras.scene_id == loaded.scene_id
    assert mesh.certified_metrology is False
    assert cameras.certified_metrology is False
    assert cameras.persisted_pose_count == 5
    assert loaded.canonical_geometry_authoritative is True


def test_3c_camera_snapshot_remains_supporting_geometry_after_roundtrip(tmp_path: Path) -> None:
    scene = _scene(tmp_path)
    manifest = write_persistent_reconstruction_scene(scene, (tmp_path / "persistent").resolve())
    loaded = read_persistent_reconstruction_scene(manifest)
    camera_artifact = next(
        item for item in loaded.artifacts if item.kind is ReconstructionArtifactKind.CAMERA_TRAJECTORY
    )

    assert camera_artifact.authority.value == "supporting_geometry"
    assert camera_artifact.canonical_geometry_authoritative is False
    assert loaded.canonical_geometry_authoritative is True
    assert loaded.certified_metrology is False


def test_3c_legacy_persistent_scene_without_pose_snapshot_remains_compatible(tmp_path: Path) -> None:
    scene = _scene(tmp_path, with_camera_snapshot=False)
    manifest = write_persistent_reconstruction_scene(scene, (tmp_path / "persistent").resolve())
    loaded = read_persistent_reconstruction_scene(manifest)

    assert load_canonical_viewer_mesh(loaded) is not None
    assert load_persistent_canonical_camera_trajectory(loaded) is None
    assert loaded.canonical_geometry_authoritative is True
    assert loaded.certified_metrology is False

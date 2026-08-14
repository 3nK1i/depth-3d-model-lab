#!/usr/bin/env python3
"""Dependency-light Phase 9.3C persistent world/camera presentation smoke."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


def _write_box(path: Path) -> None:
    path.write_text(
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


def _poses() -> np.ndarray:
    matrices = []
    for index in range(7):
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, 3] = np.asarray([-1.8 + index * 0.6, 0.0, -1.2 + index * 0.22])
        matrices.append(matrix)
    return np.asarray(matrices, dtype=np.float64)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="phase9_3c_smoke_") as tmp:
        root = Path(tmp).resolve()
        _fixture, _providers, result, snapshots = run_f5_fixture(
            root / "run", request_id="phase9-3c-smoke"
        )
        base = persistent_scene_from_pipeline_result(result, payload_snapshots_by_kind=snapshots)
        camera_artifact = next(
            item for item in base.artifacts if item.kind is ReconstructionArtifactKind.CAMERA_TRAJECTORY
        )
        surface = (root / "canonical_surface.obj").resolve()
        _write_box(surface)
        poses = _poses()
        updated = dict(snapshots)
        updated["surface_mesh"] = {
            "surface_obj_path": str(surface),
            "surface_vertex_count": 8,
            "surface_face_count": 12,
            "metric_geometry_status": "arbitrary",
            "canonical_geometry_authoritative": True,
            "certified_metrology": False,
        }
        updated["camera_trajectory"] = build_canonical_camera_trajectory_snapshot(
            matrices_reconstruction=poses,
            camera_ids=tuple(f"camera-{index:03d}" for index in range(poses.shape[0])),
            pose_layers=("optimized",) * poses.shape[0],
            provider_id=camera_artifact.provider_id,
            reconstruction_frame_id="reconstruction-world",
        )
        scene = persistent_scene_from_pipeline_result(result, payload_snapshots_by_kind=updated)
        manifest = write_persistent_reconstruction_scene(scene, (root / "persistent").resolve())
        loaded = read_persistent_reconstruction_scene(manifest)
        mesh = load_canonical_viewer_mesh(loaded)
        cameras = load_persistent_canonical_camera_trajectory(loaded)
        assert mesh is not None and cameras is not None
        assert loaded.canonical_geometry_authoritative is True
        assert cameras.certified_metrology is False

        print(json.dumps({
            "status": "PASS",
            "scene_id": loaded.scene_id,
            "scene_fingerprint_sha256": loaded.scene_fingerprint_sha256,
            "canonical_geometry_authoritative": loaded.canonical_geometry_authoritative,
            "canonical_mesh_vertex_count": mesh.vertex_count,
            "canonical_mesh_face_count": mesh.face_count,
            "camera_artifact_id": cameras.artifact_id,
            "camera_provider_id": cameras.provider_id,
            "camera_pose_convention": cameras.pose_convention,
            "camera_reconstruction_frame_id": cameras.reconstruction_frame_id,
            "persisted_camera_pose_count": cameras.persisted_pose_count,
            "camera_trajectory_available": cameras.trajectory_available,
            "first_camera_position_viewer": cameras.positions_viewer[0].tolist(),
            "last_camera_position_viewer": cameras.positions_viewer[-1].tolist(),
            "camera_geometry_authority": "supporting_geometry",
            "certified_metrology": cameras.certified_metrology,
            "camera_pose_invented": False,
            "heavy_backend_started": False,
        }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

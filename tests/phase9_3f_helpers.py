from __future__ import annotations

from pathlib import Path

import numpy as np

from app.reconstruction_canonical_cameras import build_canonical_camera_trajectory_snapshot
from core.reconstruction import (
    SemanticGeometryResult,
    SurfaceMeshResult,
    persistent_scene_from_pipeline_result,
)
from core.reconstruction.orchestration import ReconstructionArtifactKind
from tests.phase9_3d_helpers import write_surface_obj
from tests.reconstruction_f5_helpers import run_f5_fixture


def _camera_matrices(count: int = 6) -> np.ndarray:
    poses = []
    for index in range(count):
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, 3] = np.asarray(
            [-1.2 + index * 0.48, 0.06 * (index % 2), -0.9 + 0.20 * index],
            dtype=np.float64,
        )
        poses.append(matrix)
    return np.asarray(poses, dtype=np.float64)


def make_phase9_3f_scene(
    tmp_path: Path,
    *,
    request_id: str = "phase9-3f-test",
    include_semantic_snapshot: bool = True,
):
    fixture, _providers, result, snapshots = run_f5_fixture(
        tmp_path / "run", request_id=request_id
    )
    surface = fixture.store.find_unique(SurfaceMeshResult)
    semantic = fixture.store.find_unique(SemanticGeometryResult)
    surface_path = write_surface_obj(surface, tmp_path / "canonical_surface.obj")

    updated = dict(snapshots)
    updated["surface_mesh"] = {
        "surface_obj_path": str(surface_path),
        "surface_vertex_count": surface.mesh.vertex_count,
        "surface_face_count": surface.mesh.face_count,
        "metric_geometry_status": surface.metric_geometry_status.value,
        "canonical_geometry_authoritative": True,
        "certified_metrology": False,
    }
    if include_semantic_snapshot:
        updated["semantic_geometry"] = semantic.as_dict()

    base = persistent_scene_from_pipeline_result(result, payload_snapshots_by_kind=updated)
    camera_artifact = next(
        item for item in base.artifacts if item.kind is ReconstructionArtifactKind.CAMERA_TRAJECTORY
    )
    matrices = _camera_matrices()
    updated["camera_trajectory"] = build_canonical_camera_trajectory_snapshot(
        matrices_reconstruction=matrices,
        camera_ids=tuple(f"cam-{index:03d}" for index in range(matrices.shape[0])),
        pose_layers=("optimized",) * matrices.shape[0],
        provider_id=camera_artifact.provider_id,
        reconstruction_frame_id="reconstruction-world",
    )
    scene = persistent_scene_from_pipeline_result(result, payload_snapshots_by_kind=updated)
    return scene, semantic, surface

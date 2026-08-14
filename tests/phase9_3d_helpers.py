from __future__ import annotations

from pathlib import Path

import numpy as np

from core.reconstruction import (
    SemanticGeometryResult,
    SurfaceMeshResult,
    persistent_scene_from_pipeline_result,
)
from tests.reconstruction_f5_helpers import run_f5_fixture


def write_surface_obj(surface: SurfaceMeshResult, path: Path) -> Path:
    """Materialize the exact fixture surface without changing its geometry."""

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.asarray(surface.mesh.vertices_reconstruction, dtype=np.float64)
    faces = np.asarray(surface.mesh.faces, dtype=np.int64)
    colors = np.asarray(surface.mesh.colors_rgb)
    lines = ["# Phase 9.3D canonical semantic fixture"]
    for index, vertex in enumerate(vertices):
        color = colors[index] if colors.shape == (vertices.shape[0], 3) else None
        if color is None:
            lines.append("v " + " ".join(f"{value:.9g}" for value in vertex))
            continue
        rgb = np.asarray(color, dtype=np.float64)
        if np.any(rgb > 1.0):
            rgb = rgb / 255.0
        lines.append(
            "v "
            + " ".join(f"{value:.9g}" for value in vertex)
            + " "
            + " ".join(f"{value:.9g}" for value in rgb)
        )
    for face in faces:
        lines.append("f " + " ".join(str(int(value) + 1) for value in face))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def make_phase9_3d_scene(
    tmp_path: Path,
    *,
    request_id: str = "phase9-3d-test",
    include_semantic_snapshot: bool = True,
    legacy_membership: bool = False,
    mutate_semantic=None,
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
        semantic_payload = semantic.as_dict()
        if legacy_membership:
            for item in semantic_payload["objects"]:
                item.pop("vertex_indices", None)
                item.pop("face_indices", None)
        if mutate_semantic is not None:
            mutate_semantic(semantic_payload, surface)
        updated["semantic_geometry"] = semantic_payload

    scene = persistent_scene_from_pipeline_result(result, payload_snapshots_by_kind=updated)
    return scene, semantic, surface

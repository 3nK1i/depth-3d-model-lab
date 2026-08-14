#!/usr/bin/env python3
"""Dependency-light Phase 9.3B persistent canonical-mesh smoke."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.reconstruction_canonical_geometry import load_canonical_viewer_mesh
from core.reconstruction import (
    persistent_scene_from_pipeline_result,
    read_persistent_reconstruction_scene,
    verify_persistent_scene_file_references,
    write_persistent_reconstruction_scene,
)
from tests.reconstruction_f5_helpers import run_f5_fixture


def _write_cube(path: Path) -> None:
    vertices = (
        (-1, -1, -1, 1, 0, 0),
        (1, -1, -1, 0, 1, 0),
        (1, 1, -1, 0, 0, 1),
        (-1, 1, -1, 1, 1, 0),
        (-1, -1, 1, 1, 0, 1),
        (1, -1, 1, 0, 1, 1),
        (1, 1, 1, 1, 1, 1),
        (-1, 1, 1, 0.35, 0.55, 0.95),
    )
    faces = (
        (1, 2, 3), (1, 3, 4),
        (5, 7, 6), (5, 8, 7),
        (1, 5, 6), (1, 6, 2),
        (2, 6, 7), (2, 7, 3),
        (3, 7, 8), (3, 8, 4),
        (4, 8, 5), (4, 5, 1),
    )
    lines = ["# Phase 9.3B canonical viewer smoke"]
    lines.extend("v " + " ".join(str(value) for value in vertex) for vertex in vertices)
    lines.extend("f " + " ".join(str(value) for value in face) for face in faces)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="phase9_3b_smoke_") as tmp:
        root = Path(tmp).resolve()
        _fixture, _providers, result, snapshots = run_f5_fixture(
            root / "run", request_id="phase9-3b-smoke"
        )
        surface = (root / "canonical_surface.obj").resolve()
        _write_cube(surface)
        snapshots = dict(snapshots)
        snapshots["surface_mesh"] = {
            "surface_obj_path": str(surface),
            "surface_vertex_count": 8,
            "surface_face_count": 12,
            "metric_geometry_status": "arbitrary",
            "canonical_geometry_authoritative": True,
            "certified_metrology": False,
        }
        scene = persistent_scene_from_pipeline_result(result, payload_snapshots_by_kind=snapshots)
        manifest = write_persistent_reconstruction_scene(scene, (root / "persistent").resolve())
        loaded = read_persistent_reconstruction_scene(manifest)
        refs = verify_persistent_scene_file_references(loaded)
        mesh = load_canonical_viewer_mesh(loaded)
        assert mesh is not None
        assert mesh.scene_id == loaded.scene_id
        assert mesh.artifact_id == loaded.canonical_geometry_artifact_id
        assert mesh.vertex_count == 8 and mesh.face_count == 12
        assert loaded.canonical_geometry_authoritative is True
        assert loaded.certified_metrology is False

        print(json.dumps({
            "status": "PASS",
            "scene_id": loaded.scene_id,
            "scene_fingerprint_sha256": loaded.scene_fingerprint_sha256,
            "canonical_artifact_id": mesh.artifact_id,
            "canonical_geometry_authoritative": loaded.canonical_geometry_authoritative,
            "surface_reference_verified": any(item.role == "surface_obj_path" for item in refs),
            "canonical_mesh_vertex_count": mesh.vertex_count,
            "canonical_mesh_face_count": mesh.face_count,
            "viewer_bounds_min": mesh.bounds_min_viewer.tolist(),
            "viewer_bounds_max": mesh.bounds_max_viewer.tolist(),
            "certified_metrology": mesh.certified_metrology,
            "surface_mesh_result_reconstructed": False,
            "heavy_backend_started": False,
        }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

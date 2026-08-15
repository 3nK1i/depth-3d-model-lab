#!/usr/bin/env python3
"""Dependency-light Phase 9.3F scene-layer availability smoke."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.reconstruction_canonical_cameras import load_persistent_canonical_camera_trajectory
from app.reconstruction_canonical_geometry import load_canonical_viewer_mesh
from app.reconstruction_canonical_semantics import load_persistent_canonical_semantics
from app.reconstruction_scene_layers import build_reconstruction_scene_layer_availability
from tests.phase9_3f_helpers import make_phase9_3f_scene


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="phase9_3f_smoke_") as tmp:
        scene, semantic, _surface = make_phase9_3f_scene(
            Path(tmp).resolve(), request_id="phase9-3f-smoke"
        )
        mesh = load_canonical_viewer_mesh(scene)
        if mesh is None:
            raise RuntimeError("9.3F smoke did not materialize the canonical surface.")
        cameras = load_persistent_canonical_camera_trajectory(scene)
        if cameras is None:
            raise RuntimeError("9.3F smoke did not recover persisted canonical cameras.")
        semantics = load_persistent_canonical_semantics(scene, mesh)
        if semantics is None:
            raise RuntimeError("9.3F smoke did not recover persisted canonical semantics.")

        availability = build_reconstruction_scene_layer_availability(
            canonical_surface_loaded=True,
            semantic_objects_available=bool(semantics.objects),
            canonical_semantics_loaded=True,
            semantic_selection_active=True,
            inspection_evidence_available=True,
        )
        payload = {
            "status": "PASS",
            "scene_id": scene.scene_id,
            "canonical_geometry_authoritative": scene.canonical_geometry_authoritative,
            "certified_metrology": scene.certified_metrology,
            "canonical_vertex_count": mesh.vertex_count,
            "canonical_face_count": mesh.face_count,
            "canonical_camera_count": len(cameras.camera_ids),
            "semantic_object_count": len(semantic.objects),
            "scene_layers": availability.as_dict(),
            "visibility_changes_authority": False,
            "perception_rerun": False,
            "heavy_backend_started": False,
        }
        if not all(
            (
                availability.canonical_surface,
                availability.semantic_objects,
                availability.semantic_labels,
                availability.semantic_selection,
                availability.inspection_evidence,
            )
        ):
            raise RuntimeError("9.3F smoke unexpectedly lost materialized layer availability.")
        if payload["certified_metrology"] or not payload["canonical_geometry_authoritative"]:
            raise RuntimeError("9.3F smoke changed canonical/metrology authority unexpectedly.")
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

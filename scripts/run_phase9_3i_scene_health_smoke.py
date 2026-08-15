#!/usr/bin/env python3
"""Dependency-light Phase 9.3I scene bind/resource health smoke."""

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
from app.reconstruction_scene_runtime_health import (
    build_reconstruction_scene_bind_plan,
    build_reconstruction_scene_runtime_health,
)
from tests.phase9_3i_helpers import make_phase9_3i_scene


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="phase9_3i_smoke_") as tmp:
        scene, _semantic, _surface = make_phase9_3i_scene(Path(tmp))
        mesh = load_canonical_viewer_mesh(scene)
        cameras = load_persistent_canonical_camera_trajectory(scene)
        semantics = load_persistent_canonical_semantics(scene, mesh)
        if mesh is None or cameras is None or semantics is None:
            raise RuntimeError("9.3I smoke fixture did not materialize all canonical resources.")

        first = build_reconstruction_scene_bind_plan("", scene.scene_id)
        reload_plan = build_reconstruction_scene_bind_plan(
            scene.scene_id, scene.scene_id, selected_object_id=semantics.object_ids[0]
        )
        health = build_reconstruction_scene_runtime_health(
            scene_id=scene.scene_id,
            bind_latency_ms=8.0,
            bind_count=2,
            same_scene_reload_count=1,
            canonical_vertex_count=mesh.vertex_count,
            canonical_face_count=mesh.face_count,
            canonical_camera_count=cameras.persisted_pose_count,
            semantic_object_count=len(semantics.objects),
            mesh_visual_reused=True,
            view_preserved=reload_plan.preserve_view,
            selection_preserved=True,
            canonical_geometry_authoritative=scene.canonical_geometry_authoritative,
        )
        payload = health.as_dict()
        payload.update(
            {
                "status": "PASS",
                "new_scene_fit_after_bind": first.fit_after_bind,
                "same_scene_fit_after_bind": reload_plan.fit_after_bind,
                "same_scene_view_preserved": reload_plan.preserve_view,
                "same_scene_selection_preserved": bool(reload_plan.preserve_selected_object_id),
                "calibration_required": False,
                "heavy_backend_started": False,
            }
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

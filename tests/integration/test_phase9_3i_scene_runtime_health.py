from __future__ import annotations

from app.reconstruction_canonical_cameras import load_persistent_canonical_camera_trajectory
from app.reconstruction_canonical_geometry import load_canonical_viewer_mesh
from app.reconstruction_canonical_semantics import load_persistent_canonical_semantics
from app.reconstruction_scene_runtime_health import build_reconstruction_scene_runtime_health
from tests.phase9_3i_helpers import make_phase9_3i_scene


def test_3i_runtime_health_reports_existing_persistent_resources_without_authority_promotion(tmp_path) -> None:
    scene, _semantic, _surface = make_phase9_3i_scene(tmp_path)
    mesh = load_canonical_viewer_mesh(scene)
    cameras = load_persistent_canonical_camera_trajectory(scene)
    semantics = load_persistent_canonical_semantics(scene, mesh)
    assert mesh is not None and cameras is not None and semantics is not None

    health = build_reconstruction_scene_runtime_health(
        scene_id=scene.scene_id,
        bind_latency_ms=7.25,
        bind_count=1,
        same_scene_reload_count=0,
        canonical_vertex_count=mesh.vertex_count,
        canonical_face_count=mesh.face_count,
        canonical_camera_count=cameras.persisted_pose_count,
        semantic_object_count=len(semantics.objects),
        mesh_visual_reused=False,
        view_preserved=False,
        selection_preserved=False,
        canonical_geometry_authoritative=scene.canonical_geometry_authoritative,
    )
    assert health.canonical_vertex_count == mesh.vertex_count
    assert health.canonical_camera_count == 6
    assert health.semantic_object_count == len(semantics.objects)
    assert health.canonical_geometry_authoritative is True
    assert health.certified_metrology is False
    assert health.as_dict()["diagnostics_only"] is True

from __future__ import annotations

from app.reconstruction_canonical_cameras import load_persistent_canonical_camera_trajectory
from app.reconstruction_canonical_geometry import load_canonical_viewer_mesh
from app.reconstruction_canonical_semantics import load_persistent_canonical_semantics
from app.reconstruction_scene_layers import build_reconstruction_scene_layer_availability
from tests.phase9_3f_helpers import make_phase9_3f_scene


def test_3f_persistent_scene_exposes_only_materialized_layers(tmp_path) -> None:
    scene, semantic, _surface = make_phase9_3f_scene(tmp_path, request_id="phase9-3f-integration")
    mesh = load_canonical_viewer_mesh(scene)
    assert mesh is not None
    cameras = load_persistent_canonical_camera_trajectory(scene)
    assert cameras is not None
    semantics = load_persistent_canonical_semantics(scene, mesh)
    assert semantics is not None

    state = build_reconstruction_scene_layer_availability(
        canonical_surface_loaded=True,
        semantic_objects_available=bool(semantic.objects),
        canonical_semantics_loaded=True,
        semantic_selection_active=True,
        inspection_evidence_available=True,
    )
    assert state.canonical_surface is True
    assert state.semantic_objects is True
    assert state.semantic_labels is True
    assert state.semantic_selection is True
    assert state.inspection_evidence is True
    assert scene.canonical_geometry_authoritative is True
    assert scene.certified_metrology is False
    assert semantics.certified_metrology is False
    assert len(cameras.camera_ids) == 6
    assert len(semantic.objects) == 3

from pathlib import Path

from app.reconstruction_canonical_geometry import load_canonical_viewer_mesh
from core.reconstruction.da3 import DA3_PROVIDER_ID
from tests.phase9_3g_helpers import make_phase9_3g_scene


def test_3g_provider_presentation_binds_to_persistent_canonical_scene_without_backend(tmp_path: Path) -> None:
    scene, _semantic, _surface, _canonical, presentation = make_phase9_3g_scene(
        tmp_path / "integration", request_id="phase9-3g-integration"
    )
    mesh = load_canonical_viewer_mesh(scene)
    assert mesh is not None
    assert mesh.vertex_count > 0
    assert mesh.face_count > 0
    assert scene.canonical_geometry_authoritative is True
    assert scene.certified_metrology is False
    assert presentation.available is True
    assert presentation.backend_started is False
    assert any(item.provider_id == DA3_PROVIDER_ID for item in presentation.entries)
    assert all(item.provider_id for item in presentation.entries)

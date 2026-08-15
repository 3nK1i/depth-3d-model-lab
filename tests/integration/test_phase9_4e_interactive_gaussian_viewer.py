from __future__ import annotations

from pathlib import Path

from app.reconstruction_gaussian_interactive import load_interactive_gaussian_scene
from app.reconstruction_canonical_geometry import load_canonical_viewer_mesh
from app.reconstruction_canonical_scene_presentation import build_canonical_scene_presentation
from tests.phase9_4e_helpers import make_phase9_4e_persistent_scene


def test_94e_persist_reload_to_interactive_view_contract(tmp_path: Path) -> None:
    scene, manifest, _artifact = make_phase9_4e_persistent_scene(tmp_path)
    presentation = build_canonical_scene_presentation(scene, None)
    canonical_mesh = load_canonical_viewer_mesh(scene)
    assert presentation.viewer_resource_ready is True
    assert canonical_mesh is not None
    assert canonical_mesh.vertex_count == 4
    assert canonical_mesh.face_count == 2
    interactive = load_interactive_gaussian_scene(scene)
    assert manifest.is_file()
    assert interactive.display_gaussian_count == 64
    assert interactive.total_gaussian_count == 64
    assert interactive.state_text.startswith("Interactive Gaussian appearance ready")
    assert "canonical" in interactive.authority_text.lower()

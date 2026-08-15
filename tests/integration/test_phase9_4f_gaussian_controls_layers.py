from __future__ import annotations

from pathlib import Path
import sys

from app.reconstruction_canonical_geometry import load_canonical_viewer_mesh
from app.reconstruction_gaussian_controls import GaussianDisplaySettings, apply_gaussian_display_settings
from app.reconstruction_gaussian_interactive import load_interactive_gaussian_scene
from tests.phase9_4e_helpers import make_phase9_4e_persistent_scene


def test_94f_persisted_scene_to_controlled_gaussian_layer_preserves_canonical_authority(tmp_path: Path) -> None:
    scene, manifest, _artifact = make_phase9_4e_persistent_scene(tmp_path)
    canonical = load_canonical_viewer_mesh(scene)
    before = {name for name in sys.modules if name.startswith(("gsplat", "diff_gaussian_rasterization"))}
    interactive = load_interactive_gaussian_scene(scene)
    controlled = apply_gaussian_display_settings(
        interactive,
        GaussianDisplaySettings(density_percent=25, size_percent=120, opacity_percent=75),
    )
    after = {name for name in sys.modules if name.startswith(("gsplat", "diff_gaussian_rasterization"))}

    assert manifest.is_file()
    assert canonical is not None and canonical.vertex_count == 4 and canonical.face_count == 2
    assert scene.canonical_geometry_authoritative is True
    assert controlled.source_display_count == 64
    assert controlled.display_count == 16
    assert interactive.canonical_geometry_authoritative is False
    assert interactive.measurement_authoritative is False
    assert interactive.certified_metrology is False
    assert after == before

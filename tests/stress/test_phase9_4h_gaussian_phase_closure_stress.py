from __future__ import annotations

import gc
from pathlib import Path
import weakref

from app.reconstruction_gaussian_controls import GaussianDisplaySettings, apply_gaussian_display_settings
from app.reconstruction_gaussian_interactive import load_interactive_gaussian_scene
from app.reconstruction_gaussian_resources import build_gaussian_interactive_resource_policy
from tests.phase9_4e_helpers import make_phase9_4e_persistent_scene
from tests.reconstruction_helpers import make_hardware


def test_94h_repeated_interactive_load_control_release_keeps_staging_collectible(tmp_path: Path) -> None:
    scene, _manifest, _artifact = make_phase9_4e_persistent_scene(tmp_path)
    policy = build_gaussian_interactive_resource_policy(
        make_hardware(total_gib=6.0, name="NVIDIA GeForce RTX 2060")
    )
    for index in range(24):
        interactive = load_interactive_gaussian_scene(scene, max_gaussians=policy.max_interactive_gaussians)
        buffers = apply_gaussian_display_settings(
            interactive,
            GaussianDisplaySettings(
                density_percent=25 if index % 2 else 50,
                size_percent=100 + (index % 5) * 5,
                opacity_percent=65 + (index % 4) * 5,
            ),
        )
        source_ref = weakref.ref(interactive.positions_reconstruction)
        display_ref = weakref.ref(buffers.positions_reconstruction)
        del buffers
        del interactive
        gc.collect()
        assert source_ref() is None
        assert display_ref() is None

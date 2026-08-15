from __future__ import annotations

import gc
from pathlib import Path
import weakref

from app.reconstruction_gaussian_controls import GaussianDisplaySettings, apply_gaussian_display_settings
from app.reconstruction_gaussian_interactive import load_interactive_gaussian_scene
from app.reconstruction_gaussian_resources import build_gaussian_interactive_resource_policy
from core.reconstruction import current_process_gaussian_heavy_modules
from tests.phase9_4e_helpers import make_phase9_4e_persistent_scene
from tests.reconstruction_helpers import make_hardware


def test_94g_repeated_cpu_load_apply_release_does_not_retain_numpy_staging(tmp_path: Path) -> None:
    scene, _manifest, _artifact = make_phase9_4e_persistent_scene(tmp_path)
    policy = build_gaussian_interactive_resource_policy(
        make_hardware(total_gib=6.0, name="NVIDIA GeForce RTX 2060")
    )
    for _ in range(24):
        interactive = load_interactive_gaussian_scene(
            scene, max_gaussians=policy.max_interactive_gaussians
        )
        buffers = apply_gaussian_display_settings(
            interactive,
            GaussianDisplaySettings(density_percent=50, size_percent=125, opacity_percent=70),
        )
        source_ref = weakref.ref(interactive.positions_reconstruction)
        display_ref = weakref.ref(buffers.positions_reconstruction)
        del buffers
        del interactive
        gc.collect()
        assert source_ref() is None
        assert display_ref() is None
    assert current_process_gaussian_heavy_modules() == ()

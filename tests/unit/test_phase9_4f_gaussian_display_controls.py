from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.reconstruction_gaussian_controls import (
    GaussianDisplaySettings,
    apply_gaussian_display_settings,
)
from app.reconstruction_gaussian_interactive import load_interactive_gaussian_scene
from tests.phase9_4e_helpers import make_phase9_4e_persistent_scene


def _interactive(tmp_path: Path):
    scene, _manifest, _artifact = make_phase9_4e_persistent_scene(tmp_path)
    return load_interactive_gaussian_scene(scene)


def test_94f_default_controls_are_visibility_only_and_non_authoritative() -> None:
    settings = GaussianDisplaySettings()
    assert settings.density_percent == 100
    assert settings.size_percent == 100
    assert settings.opacity_percent == 100
    assert settings.visibility_only is True
    assert settings.canonical_geometry_authoritative is False
    assert settings.metric_scale_authoritative is False
    assert settings.measurement_authoritative is False
    assert settings.qc_authoritative is False
    assert settings.certified_metrology is False


def test_94f_density_sampling_is_deterministic_and_keeps_source_buffers_immutable(tmp_path: Path) -> None:
    interactive = _interactive(tmp_path)
    source_positions = interactive.positions_reconstruction.copy()
    source_colors = interactive.colors_rgba.copy()
    source_sizes = interactive.sizes_px.copy()
    settings = GaussianDisplaySettings(density_percent=50)

    first = apply_gaussian_display_settings(interactive, settings)
    second = apply_gaussian_display_settings(interactive, settings)

    assert first.source_display_count == 64
    assert first.display_count == 32
    assert np.array_equal(first.positions_reconstruction, second.positions_reconstruction)
    assert np.array_equal(interactive.positions_reconstruction, source_positions)
    assert np.array_equal(interactive.colors_rgba, source_colors)
    assert np.array_equal(interactive.sizes_px, source_sizes)


def test_94f_size_and_opacity_controls_only_scale_display_buffers(tmp_path: Path) -> None:
    interactive = _interactive(tmp_path)
    settings = GaussianDisplaySettings(
        density_percent=100,
        size_percent=140,
        opacity_percent=50,
    )
    buffers = apply_gaussian_display_settings(interactive, settings)

    assert buffers.display_count == 64
    assert np.allclose(
        buffers.colors_rgba[:, :3], interactive.colors_rgba[:, :3], rtol=0.0, atol=0.0
    )
    assert np.all(buffers.colors_rgba[:, 3] <= interactive.colors_rgba[:, 3] + 1e-7)
    assert np.all(buffers.sizes_px >= interactive.sizes_px - 1e-7)
    assert "density 100%" in buffers.status_text
    assert "size 140%" in buffers.status_text
    assert "opacity 50%" in buffers.status_text
    assert buffers.certified_metrology is False


def test_94f_invalid_display_settings_fail_closed() -> None:
    with pytest.raises(ValueError, match="density"):
        GaussianDisplaySettings(density_percent=60)
    with pytest.raises(ValueError, match="size"):
        GaussianDisplaySettings(size_percent=200)
    with pytest.raises(ValueError, match="opacity"):
        GaussianDisplaySettings(opacity_percent=10)

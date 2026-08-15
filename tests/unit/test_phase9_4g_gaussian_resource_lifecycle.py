from __future__ import annotations

from pathlib import Path

import pytest

from app.reconstruction_gaussian_controls import GaussianDisplaySettings, apply_gaussian_display_settings
from app.reconstruction_gaussian_interactive import load_interactive_gaussian_scene
from app.reconstruction_gaussian_resources import (
    GaussianViewerResidencyState,
    GaussianViewerRuntimeHealth,
    build_gaussian_interactive_resource_policy,
    build_gaussian_viewer_runtime_health,
    gaussian_display_staging_bytes,
    gaussian_scene_staging_bytes,
)
from tests.phase9_4e_helpers import make_phase9_4e_persistent_scene
from tests.reconstruction_helpers import make_hardware


def _scene(tmp_path: Path, *, maximum: int = 64):
    scene, _manifest, _artifact = make_phase9_4e_persistent_scene(tmp_path)
    return load_interactive_gaussian_scene(scene, max_gaussians=maximum)


def test_94g_resource_policy_adapts_between_rtx2060_and_high_memory_targets() -> None:
    low = build_gaussian_interactive_resource_policy(
        make_hardware(total_gib=6.0, name="NVIDIA GeForce RTX 2060")
    )
    high = build_gaussian_interactive_resource_policy(
        make_hardware(total_gib=32.0, name="NVIDIA GeForce RTX 5090", capability=(12, 0))
    )
    cpu = build_gaussian_interactive_resource_policy(make_hardware(total_gib=None))

    assert low.max_interactive_gaussians == 100_000
    assert low.tier == "conservative-6gb"
    assert "RTX 2060" in low.device_label
    assert high.max_interactive_gaussians == 200_000
    assert high.tier == "high-memory"
    assert cpu.max_interactive_gaussians == 75_000
    assert low.backend_free and high.backend_free and cpu.backend_free
    assert low.certified_metrology is False


def test_94g_staging_bytes_are_exact_for_loaded_and_controlled_buffers(tmp_path: Path) -> None:
    interactive = _scene(tmp_path)
    buffers = apply_gaussian_display_settings(
        interactive,
        GaussianDisplaySettings(density_percent=50, size_percent=125, opacity_percent=70),
    )
    assert gaussian_scene_staging_bytes(interactive) == (
        interactive.positions_reconstruction.nbytes
        + interactive.colors_rgba.nbytes
        + interactive.sizes_px.nbytes
    )
    assert gaussian_display_staging_bytes(buffers) == (
        buffers.positions_reconstruction.nbytes + buffers.colors_rgba.nbytes + buffers.sizes_px.nbytes
    )
    assert buffers.display_count == 32


def test_94g_runtime_health_distinguishes_resident_and_released_without_vram_claim(tmp_path: Path) -> None:
    interactive = _scene(tmp_path)
    buffers = apply_gaussian_display_settings(interactive, GaussianDisplaySettings(density_percent=50))
    policy = build_gaussian_interactive_resource_policy(make_hardware(total_gib=6.0, name="RTX 2060"))
    resident = build_gaussian_viewer_runtime_health(
        policy=policy,
        scene=interactive,
        buffers=buffers,
        resident=True,
        had_release=False,
        load_latency_ms=12.5,
        apply_latency_ms=1.25,
        load_count=1,
        release_count=0,
        canvas_resource_generation=1,
        canvas_release_count=0,
        canonical_geometry_authoritative=True,
    )
    assert resident.state is GaussianViewerResidencyState.RESIDENT
    assert resident.resident is True
    assert "CPU staging" in resident.resources_text
    assert "Load 12.5 ms" in resident.performance_text
    assert resident.as_dict()["gpu_vram_measured"] is False
    assert resident.as_dict()["gaussian_measurement_authoritative"] is False

    released = build_gaussian_viewer_runtime_health(
        policy=policy,
        scene=None,
        buffers=None,
        resident=False,
        had_release=True,
        load_latency_ms=12.5,
        apply_latency_ms=1.25,
        load_count=1,
        release_count=1,
        canvas_resource_generation=2,
        canvas_release_count=1,
        canonical_geometry_authoritative=True,
        last_total_gaussian_count=64,
        last_source_display_count=64,
    )
    assert released.state is GaussianViewerResidencyState.RELEASED
    assert released.resident is False
    assert released.source_staging_bytes == 0
    assert released.controlled_staging_bytes == 0
    assert "Released on hide/switch" in released.lifecycle_text
    assert released.total_gaussian_count == 64


def test_94g_runtime_health_rejects_measurement_or_metrology_authority() -> None:
    common = dict(
        state=GaussianViewerResidencyState.UNLOADED,
        device_label="RTX 2060",
        resource_tier="conservative-6gb",
        max_display_gaussians=100_000,
        total_gaussian_count=0,
        source_display_count=0,
        controlled_display_count=0,
        source_staging_bytes=0,
        controlled_staging_bytes=0,
        load_latency_ms=0.0,
        apply_latency_ms=0.0,
        load_count=0,
        release_count=0,
        canvas_resource_generation=1,
        canvas_release_count=0,
        canonical_geometry_authoritative=True,
    )
    with pytest.raises(ValueError):
        GaussianViewerRuntimeHealth(**common, gaussian_measurement_authoritative=True)
    with pytest.raises(ValueError):
        GaussianViewerRuntimeHealth(**common, certified_metrology=True)

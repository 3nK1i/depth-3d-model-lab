from __future__ import annotations

from pathlib import Path

from app.reconstruction_gaussian_controls import GaussianDisplaySettings, apply_gaussian_display_settings
from app.reconstruction_gaussian_interactive import load_interactive_gaussian_scene
from app.reconstruction_gaussian_resources import (
    build_gaussian_interactive_resource_policy,
    build_gaussian_viewer_runtime_health,
)
from tests.phase9_4e_helpers import make_phase9_4e_persistent_scene
from tests.reconstruction_helpers import make_hardware


def test_94g_persisted_scene_respects_resource_cap_and_release_contract(tmp_path: Path) -> None:
    scene, _manifest, artifact = make_phase9_4e_persistent_scene(tmp_path)
    policy = build_gaussian_interactive_resource_policy(
        make_hardware(total_gib=6.0, name="NVIDIA GeForce RTX 2060")
    )
    interactive = load_interactive_gaussian_scene(
        scene,
        max_gaussians=policy.max_interactive_gaussians,
    )
    buffers = apply_gaussian_display_settings(
        interactive,
        GaussianDisplaySettings(density_percent=25, size_percent=120, opacity_percent=70),
    )
    health = build_gaussian_viewer_runtime_health(
        policy=policy,
        scene=interactive,
        buffers=buffers,
        resident=True,
        had_release=False,
        load_latency_ms=3.0,
        apply_latency_ms=1.0,
        load_count=1,
        release_count=0,
        canvas_resource_generation=1,
        canvas_release_count=0,
        canonical_geometry_authoritative=scene.canonical_geometry_authoritative,
    )
    assert health.resident is True
    assert health.controlled_display_count == 16
    assert health.source_display_count == 64
    assert scene.gaussian_artifact is not None
    assert scene.gaussian_artifact.artifact_id == interactive.artifact_id
    assert health.canonical_geometry_authoritative is True
    assert health.gaussian_measurement_authoritative is False
    assert health.certified_metrology is False

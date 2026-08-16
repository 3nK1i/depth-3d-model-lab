from __future__ import annotations

from types import SimpleNamespace

from app.reconstruction_digital_twin_composition import build_digital_twin_scene_composition
from app.reconstruction_digital_twin_navigation import (
    DigitalTwinNavigationMode,
    build_digital_twin_navigation_sync,
)
from app.reconstruction_scene_runtime_health import build_reconstruction_scene_bind_plan


def _canonical(scene_id: str):
    return SimpleNamespace(
        available=True,
        scene_id=scene_id,
        scene_state="ready",
        artifact_id="canonical",
        canonical_geometry_authoritative=True,
        certified_metrology=False,
    )


def _appearance(scene_id: str):
    return SimpleNamespace(
        scene_id=scene_id,
        certified_metrology=False,
        entries=(
            SimpleNamespace(
                artifact_id="gaussian",
                representation_kind="gaussian_scene",
                canonical_geometry_authoritative=False,
                certified_metrology=False,
            ),
            SimpleNamespace(
                artifact_id="nerf",
                representation_kind="nerf_scene",
                canonical_geometry_authoritative=False,
                certified_metrology=False,
            ),
        ),
    )


def _camera():
    return {
        "center_viewer": (0.25, -0.5, 1.75),
        "distance": 6.0,
        "azimuth_degrees": 48.0,
        "elevation_degrees": 19.0,
        "fov_degrees": 45.0,
    }


def test_96c_gaussian_and_nerf_consume_the_same_canonical_view_without_authority_transfer() -> None:
    scene_id = "scene-96c-shared"
    appearance = _appearance(scene_id)
    gaussian_composition = build_digital_twin_scene_composition(
        _canonical(scene_id),
        canonical_surface_loaded=True,
        canonical_surface_visible=True,
        appearance_presentation=appearance,
        active_view_representation="gaussian_scene",
        gaussian_visible=True,
    )
    gaussian = build_digital_twin_navigation_sync(
        gaussian_composition,
        camera_state=_camera(),
        gaussian_interactive=True,
    )

    nerf_composition = build_digital_twin_scene_composition(
        _canonical(scene_id),
        canonical_surface_loaded=True,
        canonical_surface_visible=True,
        appearance_presentation=appearance,
        active_view_representation="nerf_scene",
        nerf_visible=True,
    )
    nerf = build_digital_twin_navigation_sync(
        nerf_composition,
        camera_state=_camera(),
        nerf_interactive=True,
    )

    assert gaussian.center_viewer == nerf.center_viewer
    assert gaussian.distance == nerf.distance
    assert gaussian.azimuth_degrees == nerf.azimuth_degrees
    assert gaussian.elevation_degrees == nerf.elevation_degrees
    assert gaussian.fov_degrees == nerf.fov_degrees
    assert gaussian.navigation_source == nerf.navigation_source == "canonical_viewport_camera"
    assert gaussian.spatial_authority == nerf.spatial_authority == "canonical_geometry"
    assert gaussian.certified_metrology is False and nerf.certified_metrology is False


def test_96c_scene_bind_policy_preserves_view_only_for_exact_same_scene() -> None:
    same = build_reconstruction_scene_bind_plan("scene-a", "scene-a")
    changed = build_reconstruction_scene_bind_plan("scene-a", "scene-b")
    assert same.preserve_view is True
    assert same.fit_after_bind is False
    assert changed.preserve_view is False
    assert changed.fit_after_bind is True

    composition = build_digital_twin_scene_composition(
        _canonical("scene-a"),
        canonical_surface_loaded=True,
        canonical_surface_visible=True,
        appearance_presentation=_appearance("scene-a"),
    )
    sync = build_digital_twin_navigation_sync(
        composition,
        camera_state=_camera(),
        view_preserved_on_same_scene_reload=same.preserve_view,
    )
    assert sync.mode is DigitalTwinNavigationMode.CANONICAL
    assert sync.scene_switch_text == "Same-scene reload • canonical view preserved"

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.reconstruction_digital_twin_navigation import (
    DigitalTwinNavigationMode,
    build_digital_twin_navigation_sync,
)

ROOT = Path(__file__).resolve().parents[2]


def _layer(*, available: bool = True, visible: bool = False):
    return SimpleNamespace(available=available, visible=visible)


def _composition(
    *,
    scene_id: str = "scene-96c",
    active: str = "canonical_geometry",
    gaussian_available: bool = True,
    gaussian_visible: bool = False,
    nerf_available: bool = True,
    nerf_visible: bool = False,
):
    layers = {
        "gaussian_scene": _layer(available=gaussian_available, visible=gaussian_visible),
        "nerf_scene": _layer(available=nerf_available, visible=nerf_visible),
    }
    return SimpleNamespace(
        state=SimpleNamespace(value="composed"),
        scene_id=scene_id,
        active_view_representation=active,
        canonical_geometry_authoritative=True,
        certified_metrology=False,
        layer=lambda key: layers.get(key),
    )


def _camera(**overrides):
    values = {
        "center_viewer": (1.0, 2.0, 3.0),
        "distance": 4.5,
        "azimuth_degrees": 32.0,
        "elevation_degrees": 24.0,
        "fov_degrees": 45.0,
    }
    values.update(overrides)
    return values


def test_96c_no_scene_fails_closed_without_camera_or_appearance() -> None:
    composition = SimpleNamespace(
        state=SimpleNamespace(value="no_scene"),
        scene_id="",
        active_view_representation="canonical_geometry",
        canonical_geometry_authoritative=False,
        certified_metrology=False,
        layer=lambda _key: None,
    )
    sync = build_digital_twin_navigation_sync(composition)
    assert sync.mode is DigitalTwinNavigationMode.NO_SCENE
    assert sync.scene_id == ""
    assert sync.center_viewer is None
    assert sync.navigation_text == "Waiting for persistent scene"
    assert sync.as_dict()["spatial_authority"] == "none"
    assert sync.certified_metrology is False


def test_96c_canonical_viewport_camera_is_the_single_navigation_source() -> None:
    sync = build_digital_twin_navigation_sync(_composition(), camera_state=_camera())
    assert sync.mode is DigitalTwinNavigationMode.CANONICAL
    assert sync.navigation_source == "canonical_viewport_camera"
    assert sync.spatial_authority == "canonical_geometry"
    assert sync.center_viewer == (1.0, 2.0, 3.0)
    assert "shared navigation source" in sync.navigation_text
    payload = sync.as_dict()
    assert payload["appearance_spatial_authority"] is False
    assert payload["picking_authority"] == "canonical_geometry"
    assert payload["measurement_authority"] == "canonical_geometry"


def test_96c_interactive_gaussian_co_navigates_without_camera_authority() -> None:
    sync = build_digital_twin_navigation_sync(
        _composition(active="gaussian_scene", gaussian_visible=True),
        camera_state=_camera(azimuth_degrees=71.0),
        gaussian_interactive=True,
    )
    assert sync.mode is DigitalTwinNavigationMode.GAUSSIAN_INTERACTIVE
    assert sync.active_view_representation == "gaussian_scene"
    assert sync.azimuth_degrees == 71.0
    assert sync.navigation_text == "Shared canonical camera • Gaussian co-navigates"
    assert "presentation state only" in sync.authority_text


def test_96c_interactive_nerf_and_static_preview_are_reported_truthfully() -> None:
    nerf = build_digital_twin_navigation_sync(
        _composition(active="nerf_scene", nerf_visible=True),
        camera_state=_camera(),
        nerf_interactive=True,
        view_preserved_on_same_scene_reload=True,
    )
    assert nerf.mode is DigitalTwinNavigationMode.NERF_INTERACTIVE
    assert "NeRF follows view requests" in nerf.navigation_text
    assert nerf.scene_switch_text == "Same-scene reload • canonical view preserved"

    preview = build_digital_twin_navigation_sync(
        _composition(active="nerf_scene", nerf_visible=True),
        camera_state=_camera(),
        nerf_preview_visible=True,
    )
    assert preview.mode is DigitalTwinNavigationMode.STATIC_PREVIEW
    assert preview.navigation_text == "Canonical navigation retained • NeRF static preview"


def test_96c_rejects_conflicting_or_unbound_appearance_navigation() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        build_digital_twin_navigation_sync(
            _composition(active="gaussian_scene", gaussian_visible=True, nerf_visible=True),
            camera_state=_camera(),
            gaussian_interactive=True,
            nerf_preview_visible=True,
        )
    with pytest.raises(ValueError, match="unavailable"):
        build_digital_twin_navigation_sync(
            _composition(
                active="gaussian_scene",
                gaussian_available=False,
                gaussian_visible=False,
            ),
            camera_state=_camera(),
            gaussian_interactive=True,
        )
    with pytest.raises(ValueError, match="distance"):
        build_digital_twin_navigation_sync(_composition(), camera_state=_camera(distance=0.0))


def test_96c_module_is_dependency_light_and_cannot_create_backend_or_spatial_authority() -> None:
    path = ROOT / "app/reconstruction_digital_twin_navigation.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".", 1)[0])
    for forbidden in (
        "PyQt6", "vispy", "torch", "pycolmap", "gsplat", "nerfstudio",
        "tinycudann", "depth_anything_3",
    ):
        assert forbidden not in imports
    assert "canonical_viewport_camera" in source
    assert "spatial_authority: str = \"canonical_geometry\"" in source
    assert "certified_metrology: bool = False" in source

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.reconstruction_digital_twin_composition import (
    DigitalTwinCompositionState,
    DigitalTwinLayerRole,
    build_digital_twin_scene_composition,
)

ROOT = Path(__file__).resolve().parents[2]


def _canonical(*, scene_id: str = "scene-96b", scene_state: str = "ready"):
    return SimpleNamespace(
        available=True,
        scene_id=scene_id,
        scene_state=scene_state,
        artifact_id="canonical-1",
        canonical_geometry_authoritative=True,
        certified_metrology=False,
    )


def _semantic(*, scene_id: str = "scene-96b"):
    return SimpleNamespace(
        scene_id=scene_id,
        semantic_artifact_id="semantic-1",
        certified_metrology=False,
    )


def _camera(*, scene_id: str = "scene-96b"):
    return SimpleNamespace(
        scene_id=scene_id,
        artifact_id="camera-1",
        certified_metrology=False,
    )


def _appearance(*kinds: str, scene_id: str = "scene-96b"):
    return SimpleNamespace(
        scene_id=scene_id,
        certified_metrology=False,
        entries=tuple(
            SimpleNamespace(
                artifact_id=f"{kind}-1",
                representation_kind=kind,
                canonical_geometry_authoritative=False,
                certified_metrology=False,
            )
            for kind in kinds
        ),
    )


def test_96b_no_scene_fails_closed_with_no_available_layers() -> None:
    composition = build_digital_twin_scene_composition()
    assert composition.state is DigitalTwinCompositionState.NO_SCENE
    assert composition.scene_id == ""
    assert composition.available_layers == ()
    assert composition.visible_layers == ()
    assert composition.active_view_representation == "canonical_geometry"
    assert composition.canonical_geometry_authoritative is False
    assert composition.certified_metrology is False


def test_96b_composes_all_scene_domains_without_authority_promotion() -> None:
    composition = build_digital_twin_scene_composition(
        _canonical(),
        canonical_surface_loaded=True,
        canonical_surface_visible=True,
        semantic_presentation=_semantic(),
        semantic_visible=True,
        camera_presentation=_camera(),
        camera_visible=True,
        appearance_presentation=_appearance("gaussian_scene", "nerf_scene"),
        active_view_representation="gaussian_scene",
        gaussian_visible=True,
    )
    assert composition.state is DigitalTwinCompositionState.COMPOSED
    assert composition.composition_text == "Canonical • Semantics • Cameras • Gaussian • NeRF"
    assert composition.visibility_text == "Visible: Canonical • Semantics • Cameras • Gaussian"
    canonical = composition.layer("canonical_surface")
    semantics = composition.layer("semantic_objects")
    cameras = composition.layer("world_cameras")
    gaussian = composition.layer("gaussian_scene")
    nerf = composition.layer("nerf_scene")
    assert canonical is not None and canonical.role is DigitalTwinLayerRole.AUTHORITATIVE_GEOMETRY
    assert canonical.canonical_geometry_authoritative is True
    assert semantics is not None and semantics.role is DigitalTwinLayerRole.SUPPORTING_CONTEXT
    assert cameras is not None and "not scale evidence" in cameras.authority_text
    assert gaussian is not None and gaussian.role is DigitalTwinLayerRole.APPEARANCE_ONLY
    assert nerf is not None and nerf.role is DigitalTwinLayerRole.APPEARANCE_ONLY
    assert all(not layer.certified_metrology for layer in composition.layers)
    payload = composition.as_dict()
    assert payload["picking_authority"] == "canonical_geometry"
    assert payload["measurement_authority"] == "canonical_geometry"


def test_96b_hidden_canonical_visual_does_not_demote_canonical_authority() -> None:
    composition = build_digital_twin_scene_composition(
        _canonical(),
        canonical_surface_loaded=True,
        canonical_surface_visible=False,
        camera_presentation=_camera(),
        camera_visible=True,
    )
    canonical = composition.layer("canonical_surface")
    assert canonical is not None
    assert canonical.available is True
    assert canonical.visible is False
    assert canonical.canonical_geometry_authoritative is True
    assert composition.canonical_geometry_authoritative is True


def test_96b_rejects_scene_identity_mismatch_in_support_or_appearance_layers() -> None:
    with pytest.raises(ValueError, match="Semantic scene identity"):
        build_digital_twin_scene_composition(
            _canonical(),
            semantic_presentation=_semantic(scene_id="other-scene"),
        )
    with pytest.raises(ValueError, match="Appearance scene identity"):
        build_digital_twin_scene_composition(
            _canonical(),
            appearance_presentation=_appearance("gaussian_scene", scene_id="other-scene"),
        )


def test_96b_rejects_cross_representation_visibility_and_authority_conflicts() -> None:
    appearance = _appearance("gaussian_scene", "nerf_scene")
    with pytest.raises(ValueError, match="cannot be visible simultaneously"):
        build_digital_twin_scene_composition(
            _canonical(),
            appearance_presentation=appearance,
            active_view_representation="gaussian_scene",
            gaussian_visible=True,
            nerf_visible=True,
        )
    with pytest.raises(ValueError, match="Appearance layer unexpectedly claims canonical geometry authority"):
        bad = _appearance("gaussian_scene")
        bad.entries[0].canonical_geometry_authoritative = True
        build_digital_twin_scene_composition(_canonical(), appearance_presentation=bad)


def test_96b_module_remains_dependency_light_and_presentation_only() -> None:
    path = ROOT / "app/reconstruction_digital_twin_composition.py"
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
    assert "core.reconstruction" not in source
    assert "read-only" in source.lower()

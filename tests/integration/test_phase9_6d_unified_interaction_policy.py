from __future__ import annotations

from types import SimpleNamespace

from app.reconstruction_digital_twin_composition import build_digital_twin_scene_composition
from app.reconstruction_digital_twin_interaction import (
    DigitalTwinInteractionMode,
    NERF_MEASUREMENT_ASSIST_MAX_OPACITY_PERCENT,
    build_digital_twin_interaction_presentation,
)


def _canonical():
    return SimpleNamespace(
        available=True,
        scene_id="scene-96d-integration",
        scene_state="ready",
        artifact_id="canonical:surface",
        canonical_geometry_authoritative=True,
        certified_metrology=False,
    )


def _semantics():
    return SimpleNamespace(
        scene_id="scene-96d-integration",
        semantic_artifact_id="semantic:scene",
        certified_metrology=False,
    )


def _appearance(kind: str):
    entry = SimpleNamespace(
        representation_kind=kind,
        artifact_id=f"{kind}:artifact",
        canonical_geometry_authoritative=False,
        certified_metrology=False,
    )
    return SimpleNamespace(
        scene_id="scene-96d-integration",
        entries=(entry,),
        certified_metrology=False,
    )


def test_gaussian_composition_preserves_canonical_inspection_authority() -> None:
    composition = build_digital_twin_scene_composition(
        _canonical(),
        canonical_surface_loaded=True,
        canonical_surface_visible=True,
        semantic_presentation=_semantics(),
        semantic_visible=True,
        appearance_presentation=_appearance("gaussian_scene"),
        active_view_representation="gaussian_scene",
        gaussian_visible=True,
    )
    interaction = build_digital_twin_interaction_presentation(
        composition,
        canonical_surface_loaded=True,
        semantic_context_available=True,
        semantic_selection_active=True,
        gaussian_interactive=True,
    )
    assert interaction.mode is DigitalTwinInteractionMode.GAUSSIAN_INTERACTIVE
    assert interaction.picking_authority == "canonical_geometry"
    assert interaction.measurement_authority == "canonical_geometry"
    assert interaction.certified_metrology is False


def test_nerf_composition_supports_canonical_ruler_without_render_authority() -> None:
    composition = build_digital_twin_scene_composition(
        _canonical(),
        canonical_surface_loaded=True,
        canonical_surface_visible=True,
        semantic_presentation=_semantics(),
        semantic_visible=True,
        appearance_presentation=_appearance("nerf_scene"),
        active_view_representation="nerf_scene",
        nerf_visible=True,
    )
    interaction = build_digital_twin_interaction_presentation(
        composition,
        canonical_surface_loaded=True,
        semantic_context_available=True,
        measurement_active=True,
        nerf_interactive=True,
    )
    assert interaction.mode is DigitalTwinInteractionMode.NERF_INTERACTIVE
    assert interaction.appearance_frozen_for_measurement is True
    assert interaction.effective_nerf_opacity_cap_percent == NERF_MEASUREMENT_ASSIST_MAX_OPACITY_PERCENT
    assert interaction.picking_authority == "canonical_geometry"
    assert interaction.certified_metrology is False

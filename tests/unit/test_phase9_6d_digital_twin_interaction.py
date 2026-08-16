from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.reconstruction_digital_twin_interaction import (
    DigitalTwinInteractionMode,
    NERF_MEASUREMENT_ASSIST_MAX_OPACITY_PERCENT,
    NERF_SEMANTIC_ASSIST_MAX_OPACITY_PERCENT,
    build_digital_twin_interaction_presentation,
)


def _composition(active: str = "canonical_geometry"):
    return SimpleNamespace(
        scene_id="scene-96d",
        active_view_representation=active,
        canonical_geometry_authoritative=True,
    )


def test_no_scene_fails_closed_without_interaction_authority() -> None:
    presentation = build_digital_twin_interaction_presentation(None)
    assert presentation.mode is DigitalTwinInteractionMode.NO_SCENE
    assert presentation.picking_authority == "none"
    assert presentation.measurement_authority == "none"
    assert presentation.certified_metrology is False


def test_canonical_scene_exposes_semantics_and_ruler_without_promoting_semantics() -> None:
    presentation = build_digital_twin_interaction_presentation(
        _composition(),
        canonical_surface_loaded=True,
        semantic_context_available=True,
    )
    assert presentation.mode is DigitalTwinInteractionMode.CANONICAL
    assert presentation.measurement_available is True
    assert "inspection + ruler" in presentation.interaction_text.lower()
    assert presentation.picking_authority == "canonical_geometry"
    assert presentation.measurement_authority == "canonical_geometry"
    assert "semantics are context only" in presentation.authority_text.lower()


def test_gaussian_interactive_keeps_semantic_inspection_and_measurement_canonical() -> None:
    presentation = build_digital_twin_interaction_presentation(
        _composition("gaussian_scene"),
        canonical_surface_loaded=True,
        semantic_context_available=True,
        semantic_selection_active=True,
        gaussian_interactive=True,
    )
    assert presentation.mode is DigitalTwinInteractionMode.GAUSSIAN_INTERACTIVE
    assert presentation.effective_nerf_opacity_cap_percent is None
    assert "gaussian remains appearance context" in presentation.interaction_text.lower()
    assert presentation.measurement_authority == "canonical_geometry"


def test_nerf_semantic_selection_uses_transient_legibility_cap() -> None:
    presentation = build_digital_twin_interaction_presentation(
        _composition("nerf_scene"),
        canonical_surface_loaded=True,
        semantic_context_available=True,
        semantic_selection_active=True,
        nerf_interactive=True,
    )
    assert presentation.mode is DigitalTwinInteractionMode.NERF_INTERACTIVE
    assert presentation.effective_nerf_opacity_cap_percent == NERF_SEMANTIC_ASSIST_MAX_OPACITY_PERCENT
    assert "dimmed" in presentation.appearance_text.lower()
    assert presentation.certified_metrology is False


def test_nerf_measurement_freezes_and_dims_without_changing_authority() -> None:
    presentation = build_digital_twin_interaction_presentation(
        _composition("nerf_scene"),
        canonical_surface_loaded=True,
        semantic_context_available=True,
        measurement_active=True,
        nerf_interactive=True,
    )
    assert presentation.effective_nerf_opacity_cap_percent == NERF_MEASUREMENT_ASSIST_MAX_OPACITY_PERCENT
    assert presentation.appearance_frozen_for_measurement is True
    assert "uncertainty/rejection preserved" in presentation.measurement_text.lower()
    assert presentation.picking_authority == "canonical_geometry"


def test_static_preview_is_never_promoted_to_picking_surface() -> None:
    presentation = build_digital_twin_interaction_presentation(
        _composition("nerf_scene"),
        canonical_surface_loaded=True,
        semantic_context_available=True,
        nerf_preview_visible=True,
    )
    assert presentation.mode is DigitalTwinInteractionMode.STATIC_PREVIEW
    assert presentation.static_preview_blocks_canonical_picking is True
    assert "not camera-synchronized" in presentation.appearance_text.lower()
    assert presentation.measurement_authority == "canonical_geometry"


def test_interaction_rejects_simultaneous_appearance_modes() -> None:
    with pytest.raises(ValueError, match="cannot coexist"):
        build_digital_twin_interaction_presentation(
            _composition("gaussian_scene"),
            canonical_surface_loaded=True,
            gaussian_interactive=True,
            nerf_preview_visible=True,
        )


def test_measurement_cannot_be_active_without_canonical_surface() -> None:
    with pytest.raises(ValueError, match="Active measurement requires"):
        build_digital_twin_interaction_presentation(
            _composition(),
            canonical_surface_loaded=False,
            measurement_active=True,
        )

from __future__ import annotations

import pytest

from app.reconstruction_scene_layers import (
    ReconstructionSceneLayerAvailability,
    build_reconstruction_scene_layer_availability,
)


def test_3f_availability_is_fail_closed_and_visibility_only() -> None:
    state = build_reconstruction_scene_layer_availability(
        canonical_surface_loaded=True,
        semantic_objects_available=True,
        canonical_semantics_loaded=True,
        semantic_selection_active=True,
        inspection_evidence_available=True,
    )
    assert state.canonical_surface is True
    assert state.semantic_objects is True
    assert state.semantic_labels is True
    assert state.semantic_selection is True
    assert state.inspection_evidence is True
    assert state.canonical_geometry_authoritative is False
    assert state.certified_metrology is False
    payload = state.as_dict()
    assert payload["visibility_only"] is True
    assert payload["canonical_geometry_authoritative"] is False
    assert payload["certified_metrology"] is False


def test_3f_semantic_children_require_real_semantic_context() -> None:
    state = build_reconstruction_scene_layer_availability(
        canonical_surface_loaded=True,
        semantic_objects_available=True,
        canonical_semantics_loaded=False,
        semantic_selection_active=True,
        inspection_evidence_available=True,
    )
    assert state.semantic_objects is True
    assert state.semantic_labels is False
    assert state.semantic_selection is False
    assert state.inspection_evidence is False


def test_3f_inspection_layer_requires_exact_selected_semantic_context() -> None:
    state = build_reconstruction_scene_layer_availability(
        canonical_surface_loaded=True,
        semantic_objects_available=True,
        canonical_semantics_loaded=True,
        semantic_selection_active=False,
        inspection_evidence_available=True,
    )
    assert state.semantic_labels is True
    assert state.semantic_selection is False
    assert state.inspection_evidence is False


def test_3f_invalid_direct_availability_cannot_promote_children() -> None:
    with pytest.raises(ValueError, match="Semantic labels"):
        ReconstructionSceneLayerAvailability(
            canonical_surface=False,
            semantic_objects=False,
            semantic_labels=True,
            semantic_selection=False,
            inspection_evidence=False,
        )
    with pytest.raises(ValueError, match="Inspection evidence"):
        ReconstructionSceneLayerAvailability(
            canonical_surface=True,
            semantic_objects=True,
            semantic_labels=True,
            semantic_selection=False,
            inspection_evidence=True,
        )

from __future__ import annotations

import pytest

from app.offline_reconstruction_semantics import semantic_presentation_from_payload
from app.reconstruction_inspection_selection import (
    ReconstructionInspectionSelectionScope,
    build_reconstruction_inspection_selection,
)


def _semantic_payload(*, session_id: str = "e1-session") -> dict[str, object]:
    return {
        "session_id": session_id,
        "metric_geometry_status": "arbitrary",
        "certified_metrology": False,
        "statistics": {
            "observations_total": 7,
            "observations_accepted": 7,
            "persistent_object_count": 2,
        },
        "objects": [
            {
                "object_id": "semantic-gear-0001",
                "class_name": "gear",
                "semantic_role": "rotating_component",
                "confidence": 0.93,
                "vertex_count": 420,
                "face_count": 260,
                "observation_ids": ["obs-g-0", "obs-g-1", "obs-g-2", "obs-g-3"],
                "mean_vertex_quality": 0.88,
                "median_vertex_uncertainty_reconstruction_units": 0.04,
                "maximum_vertex_uncertainty_reconstruction_units": 0.08,
                "centroid_reconstruction": [0.0, 0.0, 2.0],
                "bounds_min_reconstruction": [-0.4, -0.4, 1.8],
                "bounds_max_reconstruction": [0.4, 0.4, 2.2],
                "orientation_reliable": True,
                "metric_geometry_status": "arbitrary",
                "certified_metrology": False,
            },
            {
                "object_id": "semantic-shaft-0002",
                "class_name": "shaft",
                "semantic_role": "supporting_component",
                "confidence": 0.91,
                "vertex_count": 310,
                "face_count": 190,
                "observation_ids": ["obs-s-0", "obs-s-1", "obs-s-2"],
                "mean_vertex_quality": 0.85,
                "median_vertex_uncertainty_reconstruction_units": 0.05,
                "maximum_vertex_uncertainty_reconstruction_units": 0.09,
                "centroid_reconstruction": [0.8, 0.0, 2.0],
                "bounds_min_reconstruction": [0.55, -0.18, 1.7],
                "bounds_max_reconstruction": [1.05, 0.18, 2.3],
                "orientation_reliable": True,
                "metric_geometry_status": "arbitrary",
                "certified_metrology": False,
            },
        ],
    }


def _presentation(session_id: str = "e1-session"):
    return semantic_presentation_from_payload(
        _semantic_payload(session_id=session_id), source="e1-generic-semantic"
    )


def test_e1_empty_context_has_no_inspection_target() -> None:
    selection = build_reconstruction_inspection_selection()
    assert selection.scope is ReconstructionInspectionSelectionScope.NONE
    assert not selection.has_scene
    assert not selection.has_object
    assert selection.scene_text == "No reconstruction scene"
    assert selection.state_text == "No inspection target"
    assert selection.authority_text == "Selection only • measurement/QC not evaluated"
    assert not selection.certified_metrology


def test_e1_semantic_context_without_object_selects_scene_scope() -> None:
    selection = build_reconstruction_inspection_selection(
        semantic_presentation=_presentation(),
        scene_availability="partial",
    )
    assert selection.scope is ReconstructionInspectionSelectionScope.SCENE
    assert selection.session_id == "e1-session"
    assert selection.scope_text == "Scene"
    assert selection.object_text == "—"
    assert selection.geometry_text == "Scene context available"
    assert selection.state_text == "Scene target selected"


def test_e1_existing_d5_object_selection_becomes_generic_object_target() -> None:
    selection = build_reconstruction_inspection_selection(
        semantic_presentation=_presentation(),
        selected_object_index=1,
        scene_availability="partial",
    )
    assert selection.scope is ReconstructionInspectionSelectionScope.OBJECT
    assert selection.object_id == "semantic-shaft-0002"
    assert selection.class_name == "shaft"
    assert selection.semantic_role == "supporting_component"
    assert selection.object_text == "shaft • Supporting Component"
    assert selection.geometry_text == "Semantic support geometry available"
    assert selection.state_text == "Object target selected"
    assert selection.metric_geometry_status == "arbitrary"
    assert "measurement/QC not evaluated" in selection.authority_text


def test_e1_persistent_scene_identity_preserves_canonical_geometry_authority_only() -> None:
    selection = build_reconstruction_inspection_selection(
        scene_id="prs-0123456789abcdef01234567",
        session_id="e1-persistent-session",
        scene_source="persistent_scene",
        scene_availability="ready",
        canonical_geometry_authoritative=True,
    )
    assert selection.scope is ReconstructionInspectionSelectionScope.SCENE
    assert selection.canonical_geometry_authoritative is True
    assert selection.geometry_text == "Canonical scene geometry available"
    assert selection.certified_metrology is False
    assert "measurement/QC not evaluated" in selection.authority_text


def test_e1_scene_and_semantic_session_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="does not match"):
        build_reconstruction_inspection_selection(
            session_id="scene-A",
            semantic_presentation=_presentation("scene-B"),
            selected_object_index=0,
        )


def test_e1_out_of_range_object_selection_is_rejected() -> None:
    with pytest.raises(ValueError, match="outside"):
        build_reconstruction_inspection_selection(
            semantic_presentation=_presentation(),
            selected_object_index=99,
        )


def test_e1_scene_selection_does_not_claim_object_identity() -> None:
    selection = build_reconstruction_inspection_selection(
        scene_id="prs-aaaaaaaaaaaaaaaaaaaaaaaa",
        session_id="scene-session",
        scene_source="persistent_scene",
        canonical_geometry_authoritative=True,
    )
    assert selection.object_id == ""
    assert selection.class_name == ""
    assert selection.semantic_role == ""
    assert selection.semantic_source == ""


def test_e1_long_scene_identity_is_compact_for_sidebar_but_full_in_tooltip() -> None:
    scene_id = "prs-" + "1234567890" * 5
    selection = build_reconstruction_inspection_selection(
        scene_id=scene_id,
        session_id="session-long",
        scene_source="persistent_scene",
    )
    assert "…" in selection.scene_text
    assert len(selection.scene_text) <= 28
    assert scene_id in selection.scene_tooltip

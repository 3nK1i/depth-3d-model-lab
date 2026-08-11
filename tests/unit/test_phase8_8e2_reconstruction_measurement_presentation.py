from __future__ import annotations

import pytest

from app.offline_reconstruction_semantics import semantic_presentation_from_payload
from app.reconstruction_inspection_selection import build_reconstruction_inspection_selection
from app.reconstruction_measurement_presentation import (
    build_reconstruction_measurement_presentation,
    empty_reconstruction_measurement_text,
)


def _selection():
    semantic = semantic_presentation_from_payload(
        {
            "session_id": "e2-session",
            "metric_geometry_status": "arbitrary",
            "certified_metrology": False,
            "statistics": {
                "observations_total": 3,
                "observations_accepted": 3,
                "persistent_object_count": 1,
            },
            "objects": [
                {
                    "object_id": "semantic-generic-0001",
                    "class_name": "gear",
                    "semantic_role": "rotating_component",
                    "confidence": 0.92,
                    "vertex_count": 400,
                    "face_count": 240,
                    "observation_ids": ["obs-0", "obs-1", "obs-2"],
                    "mean_vertex_quality": 0.88,
                    "median_vertex_uncertainty_reconstruction_units": 0.04,
                    "maximum_vertex_uncertainty_reconstruction_units": 0.08,
                    "centroid_reconstruction": [0.0, 0.0, 2.0],
                    "bounds_min_reconstruction": [-0.3, -0.3, 1.7],
                    "bounds_max_reconstruction": [0.3, 0.3, 2.3],
                    "orientation_reliable": True,
                    "metric_geometry_status": "arbitrary",
                    "certified_metrology": False,
                }
            ],
        },
        source="e2-semantic",
    )
    return build_reconstruction_inspection_selection(
        semantic_presentation=semantic,
        selected_object_index=0,
        scene_availability="partial",
    )


def _measurement_object():
    return {
        "observation_id": "phase75-observation-0001",
        "measurements": [
            {
                "kind": "width",
                "status": "available",
                "value_si": 0.0421,
                "unit_si": "m",
                "method": "preferred_box_extent",
                "expanded_uncertainty_si": 0.0006,
                "reliability": "medium",
            },
            {
                "kind": "height",
                "status": "available",
                "value_si": 0.0182,
                "unit_si": "m",
                "method": "preferred_box_extent",
                "expanded_uncertainty_si": 0.0005,
                "reliability": "medium",
            },
            {
                "kind": "depth",
                "status": "rejected",
                "value_si": None,
                "unit_si": "m",
                "method": "preferred_box_extent",
                "rejection_reasons": ["excessive_uncertainty"],
            },
            {
                "kind": "orientation_yaw",
                "status": "available",
                "value_si": 0.5235987755982988,
                "unit_si": "rad",
                "method": "preferred_box_orientation",
                "expanded_uncertainty_si": 0.02,
                "reliability": "low",
            },
        ],
    }


def test_e2_reuses_existing_measurement_values_without_recomputing() -> None:
    presentation = build_reconstruction_measurement_presentation(
        _selection(), _measurement_object(), source="phase7_5.inspection_bundle"
    )
    assert presentation.target_object_id == "semantic-generic-0001"
    assert presentation.source == "phase7_5.inspection_bundle"
    assert presentation.source_observation_id == "phase75-observation-0001"
    assert presentation.source_measurement_count == 4
    assert [value.kind for value in presentation.values] == [
        "width",
        "height",
        "orientation_yaw",
    ]
    assert "Width: 42.1 mm" in presentation.values_text
    assert "Height: 18.2 mm" in presentation.values_text
    assert "Orientation Yaw: 30.00°" in presentation.values_text
    assert "Depth" not in presentation.values_text
    assert presentation.state_text == "3/4 values presented"
    assert "validity/confidence deferred to E3" in presentation.authority_text
    assert presentation.certified_metrology is False


def test_e2_does_not_surface_uncertainty_reliability_or_rejection_reasons() -> None:
    presentation = build_reconstruction_measurement_presentation(
        _selection(), _measurement_object()
    )
    text = "\n".join((presentation.values_text, presentation.state_text, presentation.authority_text))
    assert "±" not in text
    assert "medium" not in text.lower()
    assert "low" not in text.lower()
    assert "excessive_uncertainty" not in text


def test_e2_requires_object_scope_and_never_guesses_scene_measurements() -> None:
    semantic = semantic_presentation_from_payload(
        {
            "session_id": "e2-scene",
            "metric_geometry_status": "arbitrary",
            "certified_metrology": False,
            "statistics": {"observations_total": 0, "observations_accepted": 0, "persistent_object_count": 0},
            "objects": [],
        },
        source="e2-semantic",
    )
    selection = build_reconstruction_inspection_selection(semantic_presentation=semantic)
    with pytest.raises(ValueError, match="active E1 object"):
        build_reconstruction_measurement_presentation(selection, _measurement_object())


def test_e2_rejects_unknown_source_status() -> None:
    payload = _measurement_object()
    payload["measurements"][0]["status"] = "maybe"
    with pytest.raises(ValueError, match="Unsupported"):
        build_reconstruction_measurement_presentation(_selection(), payload)


def test_e2_rejects_nonfinite_values() -> None:
    payload = _measurement_object()
    payload["measurements"][0]["value_si"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        build_reconstruction_measurement_presentation(_selection(), payload)


def test_e2_empty_states_follow_e1_scope_without_claiming_values() -> None:
    none = build_reconstruction_inspection_selection()
    target, source, values, state, authority = empty_reconstruction_measurement_text(none)
    assert target == "No target"
    assert values == "—"
    assert "does not compute measurements" in authority

    semantic = semantic_presentation_from_payload(
        {
            "session_id": "e2-empty-scene",
            "metric_geometry_status": "arbitrary",
            "certified_metrology": False,
            "statistics": {"observations_total": 0, "observations_accepted": 0, "persistent_object_count": 0},
            "objects": [],
        },
        source="e2-semantic",
    )
    scene = build_reconstruction_inspection_selection(semantic_presentation=semantic)
    target, source, values, state, authority = empty_reconstruction_measurement_text(scene)
    assert target == "Scene"
    assert state == "Select a semantic object"
    assert values == "—"

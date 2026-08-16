from __future__ import annotations

import pytest

from app.reconstruction_canonical_scene_presentation import (
    CanonicalScenePresentation,
    CanonicalSceneResourceState,
    CanonicalSceneScaleState,
)
from app.reconstruction_estimated_metric_measurement import (
    EstimatedMetricScaleHint,
    build_reconstruction_ruler_metric_record,
    estimated_scale_from_ruler_reference,
)
from core.measurement.interactive_distance import InteractiveDistanceState


def _canonical(*, unit_to_metre: float | None, qualified: bool = False) -> CanonicalScenePresentation:
    return CanonicalScenePresentation(
        available=True,
        scene_id="scene-e0",
        scene_state="ready",
        artifact_id="surface",
        artifact_kind="surface_mesh",
        provider_id="fixture",
        canonical_geometry_authoritative=True,
        resource_state=CanonicalSceneResourceState.MATERIALIZED,
        resource_role="surface_obj_path",
        resource_path="/tmp/surface.obj",
        calibration_available=qualified,
        scale_state=(
            CanonicalSceneScaleState.QUALIFIED
            if qualified
            else (
                CanonicalSceneScaleState.PROVISIONAL
                if unit_to_metre is not None
                else CanonicalSceneScaleState.NOT_QUALIFIED
            )
        ),
        scale_relative_uncertainty=(0.03 if unit_to_metre is not None else None),
        scale_unit_to_metre=unit_to_metre,
        metric_evidence_id=("qualified-scale" if unit_to_metre is not None else ""),
        metric_assessment_state=("qualified" if qualified else "provisional"),
        certified_metrology=False,
    )


def test_estimated_scale_from_ruler_is_explicit_uncalibrated_and_non_authoritative() -> None:
    hint = estimated_scale_from_ruler_reference(
        scene_id="scene-e0",
        distance_reconstruction_units=2.0,
        approximate_distance_m=1.0,
        reference_relative_uncertainty=0.20,
    )
    assert hint.unit_to_metre_estimate == pytest.approx(0.5)
    assert hint.relative_scale_uncertainty > 0.20
    assert hint.reference_distance_reconstruction_units == 2.0
    assert hint.reference_distance_m == 1.0
    assert hint.calibrated is False
    assert hint.metric_scale_authoritative is False
    assert hint.certified_metrology is False


def test_arbitrary_scene_uses_estimated_hint_for_metre_readout_with_uncertainty() -> None:
    hint = estimated_scale_from_ruler_reference(
        scene_id="scene-e0",
        distance_reconstruction_units=2.0,
        approximate_distance_m=1.0,
        reference_relative_uncertainty=0.20,
    )
    record = build_reconstruction_ruler_metric_record(
        scene_id="scene-e0",
        point_a_reconstruction=(0.0, 0.0, 0.0),
        point_b_reconstruction=(4.0, 0.0, 0.0),
        canonical_scene=_canonical(unit_to_metre=None),
        estimated_scale_hint=hint,
    )
    assert record.current.state is InteractiveDistanceState.ESTIMATED
    assert record.current.distance_source_units == pytest.approx(4.0)
    assert record.current.distance_m == pytest.approx(2.0)
    assert record.current.expanded_uncertainty_m is not None
    assert record.current.expanded_uncertainty_m > 0.40
    assert record.original_estimate is record.current
    assert record.primary_text.startswith("Estimated 2.000")
    assert "raw 4.0000 reconstruction units" in record.secondary_text
    assert "certified metrology: no" in record.secondary_text


def test_later_qualified_scale_refines_current_value_without_overwriting_original_estimate() -> None:
    hint = estimated_scale_from_ruler_reference(
        scene_id="scene-e0",
        distance_reconstruction_units=2.0,
        approximate_distance_m=1.0,
        reference_relative_uncertainty=0.20,
    )
    original = build_reconstruction_ruler_metric_record(
        scene_id="scene-e0",
        point_a_reconstruction=(0.0, 0.0, 0.0),
        point_b_reconstruction=(4.0, 0.0, 0.0),
        canonical_scene=_canonical(unit_to_metre=None),
        estimated_scale_hint=hint,
    )
    refined = build_reconstruction_ruler_metric_record(
        scene_id="scene-e0",
        point_a_reconstruction=(0.0, 0.0, 0.0),
        point_b_reconstruction=(4.0, 0.0, 0.0),
        canonical_scene=_canonical(unit_to_metre=0.45, qualified=True),
        estimated_scale_hint=hint,
        previous=original,
    )
    assert refined.current.state is InteractiveDistanceState.QUALIFIED
    assert refined.current.distance_m == pytest.approx(1.8)
    assert refined.original_estimate is original.current
    assert refined.original_estimate.distance_m == pytest.approx(2.0)
    assert refined.original_preserved is True
    assert "original estimate preserved" in refined.secondary_text
    assert "Original estimate" in refined.provenance_tooltip


def test_no_scale_hint_keeps_arbitrary_scene_native_only() -> None:
    record = build_reconstruction_ruler_metric_record(
        scene_id="scene-e0",
        point_a_reconstruction=(0.0, 0.0, 0.0),
        point_b_reconstruction=(3.0, 4.0, 0.0),
        canonical_scene=_canonical(unit_to_metre=None),
    )
    assert record.current.state is InteractiveDistanceState.UNAVAILABLE
    assert record.current.distance_m is None
    assert record.current.distance_source_units == pytest.approx(5.0)
    assert record.primary_text == "Metric estimate unavailable"


def test_estimated_hint_must_be_scene_bound_and_cannot_claim_authority() -> None:
    with pytest.raises(ValueError, match="non-authoritative"):
        EstimatedMetricScaleHint(
            scene_id="scene-e0",
            source_kind="test",
            unit_to_metre_estimate=0.5,
            relative_scale_uncertainty=0.2,
            provenance="fixture",
            metric_scale_authoritative=True,
        )
    hint = estimated_scale_from_ruler_reference(
        scene_id="other-scene",
        distance_reconstruction_units=2.0,
        approximate_distance_m=1.0,
        reference_relative_uncertainty=0.2,
    )
    with pytest.raises(ValueError, match="does not match"):
        build_reconstruction_ruler_metric_record(
            scene_id="scene-e0",
            point_a_reconstruction=(0.0, 0.0, 0.0),
            point_b_reconstruction=(1.0, 0.0, 0.0),
            canonical_scene=_canonical(unit_to_metre=None),
            estimated_scale_hint=hint,
        )

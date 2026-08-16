from __future__ import annotations

from pathlib import Path

import pytest

from app.reconstruction_canonical_scene_presentation import build_canonical_scene_presentation
from app.reconstruction_estimated_metric_measurement import (
    build_reconstruction_ruler_metric_record,
    estimated_scale_from_ruler_reference,
)
from core.measurement.interactive_distance import InteractiveDistanceState
from tests.phase9_3j_helpers import make_phase9_3j_qualified_scene
from tests.phase9_4e_helpers import make_phase9_4e_persistent_scene


def test_e0_arbitrary_gaussian_scene_accepts_non_authoritative_estimate_without_promoting_scene(tmp_path: Path) -> None:
    scene, _manifest, _artifact = make_phase9_4e_persistent_scene(tmp_path)
    canonical = build_canonical_scene_presentation(scene, None)
    assert canonical.scale_unit_to_metre is None
    assert scene.gaussian_artifact is not None
    assert scene.gaussian_artifact.canonical_geometry_authoritative is False

    hint = estimated_scale_from_ruler_reference(
        scene_id=scene.scene_id,
        distance_reconstruction_units=2.0,
        approximate_distance_m=1.0,
        reference_relative_uncertainty=0.25,
    )
    record = build_reconstruction_ruler_metric_record(
        scene_id=scene.scene_id,
        point_a_reconstruction=(0.0, 0.0, 0.0),
        point_b_reconstruction=(3.0, 0.0, 0.0),
        canonical_scene=canonical,
        estimated_scale_hint=hint,
    )
    assert record.current.state is InteractiveDistanceState.ESTIMATED
    assert record.current.distance_m == pytest.approx(1.5)
    assert canonical.scale_unit_to_metre is None
    assert hint.metric_scale_authoritative is False
    assert record.certified_metrology is False


def test_e0_scene_bound_qualified_scale_supersedes_hint_but_preserves_original_estimate(tmp_path: Path) -> None:
    scene, authority = make_phase9_3j_qualified_scene(tmp_path)
    qualified = build_canonical_scene_presentation(scene, authority)
    # Build the original estimate against the same scene before qualified evidence is applied.
    arbitrary = build_canonical_scene_presentation(scene, None)
    hint = estimated_scale_from_ruler_reference(
        scene_id=scene.scene_id,
        distance_reconstruction_units=2.0,
        approximate_distance_m=1.2,
        reference_relative_uncertainty=0.20,
    )
    original = build_reconstruction_ruler_metric_record(
        scene_id=scene.scene_id,
        point_a_reconstruction=(0.0, 0.0, 0.0),
        point_b_reconstruction=(2.0, 0.0, 0.0),
        canonical_scene=arbitrary,
        estimated_scale_hint=hint,
    )
    refined = build_reconstruction_ruler_metric_record(
        scene_id=scene.scene_id,
        point_a_reconstruction=(0.0, 0.0, 0.0),
        point_b_reconstruction=(2.0, 0.0, 0.0),
        canonical_scene=qualified,
        estimated_scale_hint=hint,
        previous=original,
    )
    assert original.current.state is InteractiveDistanceState.ESTIMATED
    assert original.current.distance_m == pytest.approx(1.2)
    # The fixture has qualified scene scale but no camera-calibration profile;
    # the ruler therefore remains an estimate while using the better scene scale.
    assert refined.current.state is InteractiveDistanceState.ESTIMATED
    assert refined.current.distance_m == pytest.approx(2.0)
    assert refined.original_estimate is original.current
    assert refined.original_preserved is True
    assert refined.certified_metrology is False

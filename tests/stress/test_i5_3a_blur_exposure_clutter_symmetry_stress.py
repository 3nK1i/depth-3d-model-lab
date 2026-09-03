from __future__ import annotations

from pathlib import Path

from core.reconstruction.i5_blur_exposure_clutter_symmetry import I53AStressDesignPolicy, build_i5_3a_stress_plan
from core.reconstruction.i5_low_overlap_weak_parallax import build_ordered_source_identity
from tests.i5_2a_helpers import make_ordered_dataset


def test_i53a_plan_is_stable_across_repeated_construction(tmp_path: Path) -> None:
    dataset = make_ordered_dataset(tmp_path / "dataset", count=31)
    identity = build_ordered_source_identity(dataset)
    expected = build_i5_3a_stress_plan(identity)
    for _ in range(40):
        assert build_i5_3a_stress_plan(identity) == expected


def test_i53a_every_stress_preserves_full_source_membership(tmp_path: Path) -> None:
    dataset = make_ordered_dataset(tmp_path / "dataset", count=23)
    identity = build_ordered_source_identity(dataset)
    plan = build_i5_3a_stress_plan(identity)
    control_names = plan["control"]["names"]
    for scenario in plan.values():
        assert scenario["names"] == control_names
        assert scenario["image_count"] == 23


def test_i53a_default_recipes_are_bounded_and_non_destructive() -> None:
    policy = I53AStressDesignPolicy()
    assert policy.blur_kernel_size % 2 == 1
    assert 0.0 < policy.underexposure_gain < 1.0
    assert policy.overexposure_gain > 1.0
    assert 0.0 < policy.clutter_band_fraction < 0.5
    assert 0.0 < policy.symmetry_band_fraction < 0.5
    assert policy.clutter_source_offset > 0

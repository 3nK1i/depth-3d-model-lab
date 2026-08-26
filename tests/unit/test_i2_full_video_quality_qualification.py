from __future__ import annotations

import numpy as np
import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.offline_dense_quality import (
    I2QualityPolicy,
    automated_quality_gates,
    classify_peripheral_geometry,
    directional_sector_summary,
    quantiles,
    region_statistics,
    spatial_envelope,
)


def _cloud(seed: int = 0, count: int = 4000) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, size=(count, 3)).astype(np.float64)


def test_i2_policy_defaults_are_ordered_and_conservative() -> None:
    policy = I2QualityPolicy()
    assert policy.robust_lower_quantile < 0.5 < policy.robust_upper_quantile
    assert policy.minimum_extended_fraction_of_supported >= policy.minimum_core_fraction_of_supported
    assert policy.far_radial_multiplier > 1.0


def test_i2_policy_rejects_invalid_robust_quantiles() -> None:
    with pytest.raises(ReconstructionValidationError):
        I2QualityPolicy(robust_lower_quantile=0.60, robust_upper_quantile=0.90)


def test_spatial_envelope_robust_diagonal_ignores_extreme_tail_better_than_full() -> None:
    points = np.vstack([_cloud(), np.asarray([[100.0, 0.0, 0.0], [-100.0, 0.0, 0.0]])])
    env = spatial_envelope(points, lower_quantile=0.02, upper_quantile=0.98)
    assert env.full_diagonal > env.robust_diagonal * 10.0
    assert env.radial_p99 < 10.0


def test_peripheral_classifier_marks_far_tail_without_modifying_input() -> None:
    core = _cloud(count=3000)
    extended = np.vstack([core, np.asarray([[20.0, 0.0, 0.0], [25.0, 0.0, 0.0]])])
    original = extended.copy()
    peripheral, far, sector, _ = classify_peripheral_geometry(
        extended, core, voxel_size=0.1, policy=I2QualityPolicy()
    )
    np.testing.assert_array_equal(extended, original)
    assert peripheral[-2:].all()
    assert far[-2:].all()
    assert np.all(sector[-2:] == 1)


def test_peripheral_classifier_keeps_core_center_central() -> None:
    core = _cloud(count=3000)
    peripheral, far, sector, _ = classify_peripheral_geometry(
        core, core, voxel_size=0.1, policy=I2QualityPolicy()
    )
    assert np.mean(peripheral) < 0.20
    assert np.mean(far) < 0.05
    assert np.count_nonzero(sector == 0) > len(core) * 0.80


def test_directional_sector_summary_reports_positive_x_tail() -> None:
    sector = np.asarray([0, 1, 1, 2], dtype=np.uint8)
    support = np.asarray([5, 2, 3, 2])
    confidence = np.asarray([0.8, 0.4, 0.5, 0.3])
    dispersion = np.asarray([0.1, 0.4, 0.3, 0.5])
    summary = directional_sector_summary(
        sector, support=support, confidence=confidence, dispersion=dispersion
    )
    assert summary["+x"]["count"] == 2
    assert summary["-x"]["count"] == 1
    assert summary["central"]["count"] == 1


def test_region_statistics_handles_empty_region() -> None:
    stats = region_statistics(
        np.zeros(5, dtype=bool),
        support=np.arange(5),
        confidence=np.linspace(0.1, 0.5, 5),
        dispersion=np.linspace(0.2, 0.6, 5),
        component_size=np.ones(5),
    )
    assert stats["count"] == 0
    assert stats["confidence_quantiles"]["p50"] is None


def test_quantiles_ignores_nan() -> None:
    result = quantiles(np.asarray([1.0, 2.0, np.nan, 3.0]))
    assert result["p50"] == pytest.approx(2.0)


def _gates(**overrides):
    values = dict(
        input_sfm_keyframes=40,
        registered_views=40,
        candidate_dense_views=40,
        alignment_accepted_views=40,
        selected_views=33,
        source_supported=171_631,
        core_points=113_756,
        extended_points=132_395,
        rejected_supported=39_236,
        largest_component_fraction=0.910,
        exact_observation_accounting=True,
        persistence_self_contained=True,
        authority_boundaries_preserved=True,
        policy=I2QualityPolicy(),
    )
    values.update(overrides)
    return automated_quality_gates(**values)


def test_automated_quality_gates_accept_current_full_video_profile() -> None:
    gates = _gates()
    assert all(gates.values())


def test_automated_quality_gates_reject_low_selected_fraction() -> None:
    gates = _gates(selected_views=10)
    assert gates["minimum_selected_fraction_of_registered"] is False


def test_automated_quality_gates_reject_fragmented_geometry() -> None:
    gates = _gates(largest_component_fraction=0.50)
    assert gates["minimum_largest_component_fraction"] is False


def test_automated_quality_gates_reject_weak_retention() -> None:
    gates = _gates(core_points=30_000, extended_points=60_000, rejected_supported=111_631)
    assert gates["minimum_core_fraction_of_supported"] is False
    assert gates["minimum_extended_fraction_of_supported"] is False
    assert gates["maximum_rejected_supported_fraction"] is False


def test_automated_quality_gates_require_exact_accounting_and_persistence() -> None:
    gates = _gates(exact_observation_accounting=False, persistence_self_contained=False)
    assert gates["exact_observation_accounting"] is False
    assert gates["persistence_self_contained"] is False


def test_region_statistics_requires_aligned_evidence() -> None:
    with pytest.raises(ReconstructionValidationError):
        region_statistics(
            np.ones(3, dtype=bool),
            support=np.ones(2),
            confidence=np.ones(3),
            dispersion=np.ones(3),
            component_size=np.ones(3),
        )

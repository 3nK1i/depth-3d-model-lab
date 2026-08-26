from __future__ import annotations

import numpy as np
import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.offline_dense_extremum_survivor_diagnosis import (
    I251ExtremumPolicy,
    aligned_region_statistics,
    extremum_driver_mask,
    policy_envelope_decomposition,
    rejection_consensus,
    robust_directional_excess,
    source_view_attribution,
)


def _cloud() -> np.ndarray:
    rng = np.random.default_rng(7)
    central = rng.uniform(-1.0, 1.0, size=(400, 3))
    tails = np.asarray([[4.0, 0, 0], [4.2, .1, 0], [-3.8, 0, 0], [0, 4.1, 0], [0, 0, -4.0]])
    return np.vstack([central, tails])


def test_policy_rejects_invalid_quantiles() -> None:
    with pytest.raises(ReconstructionValidationError):
        I251ExtremumPolicy(robust_lower_quantile=.6, robust_upper_quantile=.9)


def test_policy_rejects_invalid_tail_fraction() -> None:
    with pytest.raises(ReconstructionValidationError):
        I251ExtremumPolicy(minimum_fraction_of_directional_max=0.0)


def test_robust_directional_excess_identifies_outside_points() -> None:
    result = robust_directional_excess(_cloud(), lower_quantile=.02, upper_quantile=.98)
    assert np.count_nonzero(result["outside_robust_box"]) > 0
    assert np.max(result["maximum_normalized_excess"]) > 1.0


def test_direction_codes_are_zero_inside_and_one_to_six_outside() -> None:
    result = robust_directional_excess(_cloud(), lower_quantile=.02, upper_quantile=.98)
    codes = np.asarray(result["dominant_direction_code"])
    assert set(np.unique(codes)).issubset(set(range(7)))
    assert np.all(codes[np.asarray(result["maximum_normalized_excess"]) <= 0] == 0)


def test_extremum_driver_mask_selects_tail_endpoints() -> None:
    result = robust_directional_excess(_cloud(), lower_quantile=.02, upper_quantile=.98)
    mask, summary = extremum_driver_mask(result["directional_excess"], policy=I251ExtremumPolicy())
    assert summary["driver_count"] == int(np.count_nonzero(mask))
    assert np.count_nonzero(mask[-5:]) >= 4


def test_extremum_driver_requires_nx6_evidence() -> None:
    with pytest.raises(ReconstructionValidationError):
        extremum_driver_mask(np.zeros((10, 3)), policy=I251ExtremumPolicy())


def test_rejection_consensus_finds_stubborn_survivors() -> None:
    result = rejection_consensus(
        [np.asarray([1, 0, 0], dtype=bool), np.asarray([0, 1, 0], dtype=bool)], length=3
    )
    assert result["survives_all_policies"].tolist() == [False, False, True]
    assert result["rejection_count"].tolist() == [1, 1, 0]


def test_rejection_consensus_requires_masks() -> None:
    with pytest.raises(ReconstructionValidationError):
        rejection_consensus([], length=3)


def test_policy_decomposition_detects_robust_denominator_shrink() -> None:
    rng = np.random.default_rng(2)
    points = rng.normal(size=(1000, 3))
    points = np.vstack([points, [[12.0, 0, 0]]])
    drivers = np.zeros(len(points), dtype=bool); drivers[-1] = True
    reject = np.zeros(len(points), dtype=bool)
    # Remove broad shoulder geometry while deliberately preserving the extreme.
    reject[:120] = True
    result = policy_envelope_decomposition(
        points, reject, lower_quantile=.02, upper_quantile=.98, extremum_driver_mask_fixed=drivers
    )
    assert result["extremum_driver_rejection_fraction"] == 0.0
    assert result["candidate_full_diagonal"] <= result["baseline_full_diagonal"] + 1e-12


def test_policy_decomposition_reports_driver_rejection_fraction() -> None:
    points = _cloud()
    metrics = robust_directional_excess(points, lower_quantile=.02, upper_quantile=.98)
    drivers, _ = extremum_driver_mask(metrics["directional_excess"], policy=I251ExtremumPolicy())
    result = policy_envelope_decomposition(
        points, drivers, lower_quantile=.02, upper_quantile=.98, extremum_driver_mask_fixed=drivers
    )
    assert result["extremum_driver_rejection_fraction"] == 1.0


def test_aligned_region_statistics_reports_boolean_fractions() -> None:
    n = 6
    result = aligned_region_statistics(
        np.asarray([1,1,0,0,0,0], dtype=bool),
        support=np.arange(n)+2,
        confidence=np.linspace(.2,.8,n),
        dispersion=np.linspace(.1,.6,n),
        high_support_neighbors=np.arange(n),
        radius2_neighbors=np.arange(n)+5,
        parallax_degrees=np.arange(n)+1,
        baseline_to_range=np.linspace(.01,.06,n),
        low_diversity=np.asarray([1,0,0,0,0,0], dtype=bool),
        is_core=np.asarray([1,1,1,0,0,0], dtype=bool),
        is_peripheral=np.asarray([0,1,0,1,0,0], dtype=bool),
        is_far=np.asarray([0,1,0,0,0,0], dtype=bool),
    )
    assert result["count"] == 2
    assert result["core_fraction"] == 1.0
    assert result["low_diversity_fraction"] == 0.5


def test_aligned_region_statistics_rejects_misalignment() -> None:
    with pytest.raises(ReconstructionValidationError):
        aligned_region_statistics(
            [True, False], support=[2], confidence=[.5,.5], dispersion=[.1,.1],
            high_support_neighbors=[1,1], radius2_neighbors=[2,2], parallax_degrees=[2,2],
            baseline_to_range=[.1,.1], low_diversity=[0,0], is_core=[1,1], is_peripheral=[0,0], is_far=[0,0]
        )


def test_source_view_attribution_ranks_strongest_source() -> None:
    matrix = np.asarray([[1,1,1,0], [0,1,0,1], [0,0,1,1]], dtype=bool)
    records = source_view_attribution(matrix, [5, 10, 15], [1,1,1,0], top_k=3)
    assert records[0]["workspace_order"] == 5
    assert records[0]["observed_extremum_voxels"] == 3


def test_source_view_attribution_rejects_bad_shape() -> None:
    with pytest.raises(ReconstructionValidationError):
        source_view_attribution(np.ones((2, 3), dtype=bool), [1], [1,1,1], top_k=2)

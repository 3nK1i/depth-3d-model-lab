from __future__ import annotations

import numpy as np
import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.offline_dense_peripheral_diagnosis import (
    I21DiagnosisPolicy,
    automatic_temporal_suspect_segments,
    pair_cosupport_records,
    removal_sensitivity,
    scale_profile,
    validate_observation_matrix,
)


def _matrix() -> tuple[np.ndarray, np.ndarray]:
    # views x voxels; baseline supports = [2,2,3,2,3,2]
    matrix = np.asarray([
        [1, 1, 1, 0, 0, 0],
        [1, 0, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1],
    ], dtype=bool)
    return matrix, matrix.sum(axis=0)


def _points() -> np.ndarray:
    return np.asarray([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [5.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [8.0, 0.0, 0.0],
    ], dtype=np.float64)


def _masks():
    peripheral = np.asarray([0, 0, 0, 1, 0, 1], dtype=bool)
    far = np.asarray([0, 0, 0, 0, 0, 1], dtype=bool)
    weak = peripheral.copy()
    core = np.asarray([1, 1, 1, 0, 1, 0], dtype=bool)
    sector = np.asarray([0, 0, 0, 1, 0, 1], dtype=np.uint8)
    return peripheral, far, weak, core, sector


def test_i21_policy_defaults_are_valid() -> None:
    policy = I21DiagnosisPolicy()
    assert policy.minimum_support_views == 2
    assert policy.robust_lower_quantile < 0.5 < policy.robust_upper_quantile


def test_i21_policy_rejects_support_below_two() -> None:
    with pytest.raises(ReconstructionValidationError):
        I21DiagnosisPolicy(minimum_support_views=1)


def test_exact_observation_accounting_accepts_matching_matrix() -> None:
    matrix, support = _matrix()
    got_matrix, got_support = validate_observation_matrix(matrix, support, minimum_support_views=2)
    np.testing.assert_array_equal(got_matrix, matrix)
    np.testing.assert_array_equal(got_support, support)


def test_exact_observation_accounting_rejects_mismatch() -> None:
    matrix, support = _matrix()
    support = support.copy(); support[0] += 1
    with pytest.raises(ReconstructionValidationError):
        validate_observation_matrix(matrix, support, minimum_support_views=2)


def test_automatic_segments_pick_strongest_views_in_each_temporal_half() -> None:
    orders = (0, 1, 2, 20, 21, 22)
    records = [
        {"workspace_order": 0, "peripheral_retained_observations": 10, "far_peripheral_retained_observations": 0},
        {"workspace_order": 1, "peripheral_retained_observations": 30, "far_peripheral_retained_observations": 2},
        {"workspace_order": 2, "peripheral_retained_observations": 20, "far_peripheral_retained_observations": 1},
        {"workspace_order": 20, "peripheral_retained_observations": 5, "far_peripheral_retained_observations": 0},
        {"workspace_order": 21, "peripheral_retained_observations": 40, "far_peripheral_retained_observations": 3},
        {"workspace_order": 22, "peripheral_retained_observations": 15, "far_peripheral_retained_observations": 1},
    ]
    segments = automatic_temporal_suspect_segments(orders, records, top_n=2)
    assert segments["early_suspect"] == (1, 2)
    assert segments["late_suspect"] == (21, 22)


def test_single_view_removal_collapses_support2_but_not_support3() -> None:
    matrix, support = _matrix()
    peripheral, far, weak, core, sector = _masks()
    metrics, collapse, remaining = removal_sensitivity(
        points=_points(), observation_matrix=matrix, baseline_support=support,
        removed_view_indices=(0,), peripheral_mask=peripheral, far_mask=far, weak_mask=weak,
        core_mask=core, sector_codes=sector, policy=I21DiagnosisPolicy(),
    )
    assert collapse[0] and collapse[1]
    assert not collapse[2]
    assert remaining[2] == 2
    assert metrics["collapsed_extended"]["collapsed_count"] == 2


def test_segment_removal_can_collapse_support3_geometry() -> None:
    matrix, support = _matrix()
    peripheral, far, weak, core, sector = _masks()
    _, collapse, remaining = removal_sensitivity(
        points=_points(), observation_matrix=matrix, baseline_support=support,
        removed_view_indices=(0, 1), peripheral_mask=peripheral, far_mask=far, weak_mask=weak,
        core_mask=core, sector_codes=sector, policy=I21DiagnosisPolicy(),
    )
    assert collapse[2]
    assert remaining[2] == 1


def test_removal_sensitivity_reports_peripheral_and_far_collapse() -> None:
    matrix, support = _matrix()
    peripheral, far, weak, core, sector = _masks()
    metrics, _, _ = removal_sensitivity(
        points=_points(), observation_matrix=matrix, baseline_support=support,
        removed_view_indices=(2,), peripheral_mask=peripheral, far_mask=far, weak_mask=weak,
        core_mask=core, sector_codes=sector, policy=I21DiagnosisPolicy(),
    )
    assert metrics["collapsed_peripheral"]["collapsed_count"] == 2
    assert metrics["collapsed_far"]["collapsed_count"] == 1


def test_removal_sensitivity_does_not_mutate_points() -> None:
    matrix, support = _matrix(); points = _points(); before = points.copy()
    peripheral, far, weak, core, sector = _masks()
    removal_sensitivity(
        points=points, observation_matrix=matrix, baseline_support=support,
        removed_view_indices=(2,), peripheral_mask=peripheral, far_mask=far, weak_mask=weak,
        core_mask=core, sector_codes=sector, policy=I21DiagnosisPolicy(),
    )
    np.testing.assert_array_equal(points, before)


def test_pair_cosupport_identifies_support2_peripheral_pair() -> None:
    matrix, support = _matrix(); peripheral, far, weak, _, _ = _masks()
    records = pair_cosupport_records(matrix, support, (0, 1, 2, 3), peripheral_mask=peripheral, far_mask=far, weak_mask=weak)
    matching = [item for item in records if (item["order_a"], item["order_b"]) == (1, 2)][0]
    assert matching["support2_peripheral_pair"] >= 1


def test_pair_cosupport_records_all_pairs() -> None:
    matrix, support = _matrix(); peripheral, far, weak, _, _ = _masks()
    records = pair_cosupport_records(matrix, support, (0, 1, 2, 3), peripheral_mask=peripheral, far_mask=far, weak_mask=weak)
    assert len(records) == 6


def test_scale_profile_is_available_and_finite() -> None:
    result = scale_profile({0: 4.0, 1: 4.2, 2: 6.0, 3: 4.1}, (1, 2))
    assert result["available"] is True
    assert np.isfinite(result["maximum_absolute_robust_log_z"])


def test_scale_profile_unavailable_without_valid_scales() -> None:
    result = scale_profile({}, (1, 2))
    assert result["available"] is False


def test_invalid_removed_view_index_is_rejected() -> None:
    matrix, support = _matrix(); peripheral, far, weak, core, sector = _masks()
    with pytest.raises(ReconstructionValidationError):
        removal_sensitivity(
            points=_points(), observation_matrix=matrix, baseline_support=support,
            removed_view_indices=(99,), peripheral_mask=peripheral, far_mask=far, weak_mask=weak,
            core_mask=core, sector_codes=sector, policy=I21DiagnosisPolicy(),
        )

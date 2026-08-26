from __future__ import annotations

import numpy as np
import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.offline_dense_support_diversity import (
    I24SupportDiversityPolicy,
    classify_low_diversity,
    derive_reference_thresholds,
    diagnostic_signal,
    diversity_region_statistics,
    support2_pair_diversity_records,
    support_diversity_metrics,
)


def _scene():
    # Four cameras and five retained voxels.  Every voxel has >=2 supporters.
    centers = np.asarray([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [4.0, 0.0, 0.0],
    ], dtype=np.float64)
    points = np.asarray([
        [0.5, 1.0, 0.0],
        [0.5, 10.0, 0.0],
        [2.0, 4.0, 0.0],
        [0.2, 2.0, 0.0],
        [2.0, 2.0, 0.0],
    ], dtype=np.float64)
    matrix = np.asarray([
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 0],
        [0, 0, 1, 0, 1],
        [0, 0, 0, 0, 1],
    ], dtype=bool)
    orders = (0, 1, 2, 10)
    return points, matrix, centers, orders


def test_i24_policy_defaults_valid() -> None:
    policy = I24SupportDiversityPolicy()
    assert 0.0 < policy.reference_quantile < 0.5
    assert policy.maximum_support_for_low_diversity >= 2


def test_i24_policy_rejects_invalid_reference_quantile() -> None:
    with pytest.raises(ReconstructionValidationError):
        I24SupportDiversityPolicy(reference_quantile=0.75)


def test_support_diversity_rejects_support_below_two() -> None:
    points, matrix, centers, orders = _scene()
    matrix = matrix.copy(); matrix[:, 0] = False; matrix[0, 0] = True
    with pytest.raises(ReconstructionValidationError):
        support_diversity_metrics(points, matrix, centers, orders)


def test_known_two_view_parallax_is_ninety_degrees() -> None:
    points = np.asarray([[1.0, 1.0, 0.0]])
    centers = np.asarray([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    matrix = np.asarray([[1], [1]], dtype=bool)
    metrics = support_diversity_metrics(points, matrix, centers, (0, 1))
    assert metrics["maximum_pair_parallax_degrees"][0] == pytest.approx(90.0, abs=1e-8)


def test_baseline_to_range_is_scale_invariant() -> None:
    points, matrix, centers, orders = _scene()
    a = support_diversity_metrics(points, matrix, centers, orders)
    b = support_diversity_metrics(points * 7.5, matrix, centers * 7.5, orders)
    np.testing.assert_allclose(a["maximum_pair_parallax_degrees"], b["maximum_pair_parallax_degrees"], atol=1e-10)
    np.testing.assert_allclose(a["maximum_baseline_to_range_ratio"], b["maximum_baseline_to_range_ratio"], atol=1e-10)


def test_maximum_pair_metric_uses_diverse_pair_when_three_views_support() -> None:
    points, matrix, centers, orders = _scene()
    metrics = support_diversity_metrics(points, matrix, centers, orders)
    # voxel 2 is supported by cameras 0/1/2, so its max baseline is sqrt(2)
    assert metrics["support"][2] == 3
    assert metrics["maximum_camera_baseline_reconstruction_units"][2] == pytest.approx(np.sqrt(2.0))


def test_supporting_rank_and_order_span_are_reported() -> None:
    points, matrix, centers, orders = _scene()
    metrics = support_diversity_metrics(points, matrix, centers, orders)
    # voxel 4 is supported by rows 0,2,3 => rank span 3 and order span 10
    assert metrics["supporting_rank_span"][4] == 3
    assert metrics["supporting_order_span"][4] == 10


def _reference_metrics():
    return {
        "support": np.asarray([2, 2, 2, 3, 2, 2], dtype=np.int64),
        "maximum_pair_parallax_degrees": np.asarray([10, 20, 30, 40, 2, 3], dtype=np.float64),
        "maximum_baseline_to_range_ratio": np.asarray([.10, .20, .30, .40, .01, .02], dtype=np.float64),
        "maximum_camera_baseline_reconstruction_units": np.ones(6),
        "supporting_rank_span": np.asarray([1, 1, 2, 3, 1, 1]),
        "supporting_order_span": np.asarray([1, 1, 2, 3, 1, 1]),
    }


def test_reference_thresholds_use_central_core_low_support() -> None:
    metrics = _reference_metrics()
    core = np.asarray([1, 1, 1, 1, 0, 0], dtype=bool)
    peripheral = np.asarray([0, 0, 0, 0, 1, 1], dtype=bool)
    thresholds, reference = derive_reference_thresholds(
        metrics, core_mask=core, peripheral_mask=peripheral,
        policy=I24SupportDiversityPolicy(reference_quantile=.25, minimum_reference_points=2),
    )
    assert thresholds["source"] == "central_core_low_support"
    assert np.count_nonzero(reference) == 4
    assert thresholds["maximum_pair_parallax_degrees_threshold"] > 10.0


def test_low_diversity_requires_angle_and_baseline_ratio() -> None:
    metrics = _reference_metrics()
    thresholds = {
        "maximum_pair_parallax_degrees_threshold": 5.0,
        "maximum_baseline_to_range_ratio_threshold": .05,
    }
    low = classify_low_diversity(metrics, thresholds, policy=I24SupportDiversityPolicy(minimum_reference_points=2))
    assert low.tolist() == [False, False, False, False, True, True]


def test_region_statistics_report_low_diversity_fraction() -> None:
    metrics = _reference_metrics()
    mask = np.asarray([0, 0, 0, 0, 1, 1], dtype=bool)
    low = mask.copy()
    stats = diversity_region_statistics(mask, metrics=metrics, low_diversity_mask=low)
    assert stats["count"] == 2
    assert stats["low_diversity_fraction"] == 1.0
    assert stats["support2_fraction"] == 1.0


def test_support2_pair_records_identify_exact_pair() -> None:
    points, matrix, centers, orders = _scene()
    peripheral = np.asarray([0, 1, 0, 1, 0], dtype=bool)
    far = np.asarray([0, 1, 0, 0, 0], dtype=bool)
    weak = peripheral.copy()
    core = ~peripheral
    low = np.asarray([0, 1, 0, 1, 0], dtype=bool)
    records = support2_pair_diversity_records(
        points=points, observation_matrix=matrix, camera_centers=centers, view_orders=orders,
        peripheral_mask=peripheral, far_mask=far, weak_mask=weak, core_mask=core, low_diversity_mask=low,
    )
    pair01 = next(item for item in records if (item["order_a"], item["order_b"]) == (0, 1))
    assert pair01["support2_peripheral_count"] == 2
    assert pair01["support2_far_count"] == 1


def test_support2_pair_records_do_not_count_support3_as_exact_pair() -> None:
    points, matrix, centers, orders = _scene()
    mask = np.ones(len(points), dtype=bool)
    records = support2_pair_diversity_records(
        points=points, observation_matrix=matrix, camera_centers=centers, view_orders=orders,
        peripheral_mask=mask, far_mask=mask, weak_mask=mask, core_mask=mask, low_diversity_mask=mask,
    )
    total_support2 = sum(item["support2_extended_count"] for item in records)
    assert total_support2 == int(np.count_nonzero(matrix.sum(axis=0) == 2))


def test_diagnostic_signal_can_be_strongly_enriched() -> None:
    ref = {"low_diversity_fraction": .10}
    peripheral = {"low_diversity_fraction": .30}
    far = {"low_diversity_fraction": .40}
    result = diagnostic_signal(
        reference_stats=ref, peripheral_stats=peripheral, far_stats=far,
        policy=I24SupportDiversityPolicy(minimum_reference_points=2),
    )
    assert result["strength"] == "strong"
    assert result["supports_support_diversity_hypothesis"] is True


def test_diagnostic_signal_rejects_no_enrichment() -> None:
    result = diagnostic_signal(
        reference_stats={"low_diversity_fraction": .20},
        peripheral_stats={"low_diversity_fraction": .22},
        far_stats={"low_diversity_fraction": .21},
        policy=I24SupportDiversityPolicy(minimum_reference_points=2),
    )
    assert result["strength"] == "weak"
    assert result["supports_support_diversity_hypothesis"] is False

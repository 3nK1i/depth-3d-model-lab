from __future__ import annotations

import math

import pytest

from core.reconstruction.offline_dense_campaign import (
    I1_CAMPAIGN_SCHEMA_VERSION,
    pair_key,
    select_longest_monotonic_pass_path,
    uniform_order_subset,
)


def _pair(a: int, b: int, *, status: str = "pass", consistency: float = 0.8, median: float = 0.05):
    # Mirror the real E3/D2 pair-diagnostic schema exactly. This guards I1
    # against accidentally drifting to synthetic shorthand field names.
    return pair_key(a, b), {
        "workspace_orders": [min(a, b), max(a, b)],
        "status": status,
        "a_to_b": {
            "consistent_ratio_of_comparable": consistency,
            "median_absolute_relative_depth_residual": median,
        },
        "b_to_a": {
            "consistent_ratio_of_comparable": min(1.0, consistency + 0.05),
            "median_absolute_relative_depth_residual": median * 0.9,
        },
    }


def test_i1_schema_version_is_locked() -> None:
    assert I1_CAMPAIGN_SCHEMA_VERSION == 1


def test_pair_key_is_order_independent() -> None:
    assert pair_key(9, 7) == (7, 9)
    assert pair_key(7, 9) == (7, 9)


def test_longest_path_bridges_one_degraded_adjacent_edge_without_relaxing_it() -> None:
    pairs = dict([
        _pair(0, 1),
        _pair(1, 2),
        _pair(2, 3, status="degraded"),
        _pair(2, 4, consistency=0.76, median=0.08),
        _pair(3, 4),
        _pair(4, 5),
    ])
    path = select_longest_monotonic_pass_path([0, 1, 2, 3, 4, 5], pairs, max_bridge_gap=2)
    assert path.orders == (0, 1, 2, 4, 5)
    assert 3 not in path.orders
    assert path.worst_consistency_ratio == pytest.approx(0.76)
    assert path.worst_median_relative_residual == pytest.approx(0.08)


def test_longest_path_never_uses_degraded_or_inconclusive_edges() -> None:
    pairs = dict([
        _pair(0, 1, status="degraded"),
        _pair(0, 2, status="inconclusive"),
        _pair(1, 2, status="pass"),
    ])
    path = select_longest_monotonic_pass_path([0, 1, 2], pairs, max_bridge_gap=2)
    assert path.orders == (1, 2)


def test_longest_path_prefers_more_views_before_better_quality() -> None:
    pairs = dict([
        _pair(0, 1, consistency=0.55, median=0.09),
        _pair(1, 2, consistency=0.55, median=0.09),
        _pair(0, 2, consistency=0.99, median=0.01),
    ])
    path = select_longest_monotonic_pass_path([0, 1, 2], pairs, max_bridge_gap=2)
    assert path.orders == (0, 1, 2)


def test_longest_path_prefers_stronger_consistency_when_view_count_and_span_tie() -> None:
    pairs = dict([
        _pair(0, 2, consistency=0.70, median=0.05),
        _pair(1, 3, consistency=0.90, median=0.05),
    ])
    path = select_longest_monotonic_pass_path([0, 1, 2, 3], pairs, max_bridge_gap=2)
    assert path.orders == (1, 3)


def test_empty_path_is_explicit() -> None:
    path = select_longest_monotonic_pass_path([], {}, max_bridge_gap=3)
    assert path.orders == ()
    assert path.view_count == 0
    assert math.isinf(path.worst_median_relative_residual)
    assert path.as_dict()["worst_median_relative_residual"] is None


def test_uniform_order_subset_preserves_endpoints_and_bound() -> None:
    subset = uniform_order_subset(tuple(range(40)), 10)
    assert len(subset) == 10
    assert subset[0] == 0
    assert subset[-1] == 39
    assert subset == tuple(sorted(set(subset)))


def test_uniform_order_subset_returns_all_when_under_budget() -> None:
    assert uniform_order_subset([5, 1, 3], 5) == (1, 3, 5)


def test_uniform_order_subset_rejects_non_positive_budget() -> None:
    with pytest.raises(ValueError):
        uniform_order_subset([0, 1], 0)

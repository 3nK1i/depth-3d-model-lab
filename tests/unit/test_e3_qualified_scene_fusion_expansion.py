from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.reconstruction.errors import ReconstructionValidationError
from scripts.run_e3_qualified_scene_fusion_expansion import (
    _best_growth_path,
    _expanded_support_masks,
    _candidate_workspace_orders,
    _pair_quality,
    _robust_diagonal,
    _validate_e2_summary,
    _validate_thresholds,
)


def _e2_summary(*, passed: bool = True, authorized: bool = True):
    orders = [5, 6, 7, 9, 10, 11, 12, 13, 14, 15]
    return {
        "phase": "E2",
        "pass": passed,
        "status": "pass" if passed else "degraded",
        "decision": {"authorizes_next_scene_fusion_expansion_milestone": authorized},
        "selected_workspace_orders": orders,
        "selected_keyframe_ids": [f"kf-{order}" for order in orders],
    }


def _thresholds():
    return {
        "consistency_relative_tolerance": 0.12,
        "occlusion_relative_margin": 0.20,
        "minimum_comparable_samples": 1000,
        "minimum_consistent_ratio": 0.50,
        "maximum_median_relative_residual": 0.10,
        "maximum_p90_relative_residual": 0.25,
        "maximum_front_conflict_ratio": 0.25,
        "nn_support_scene_fraction": 0.05,
    }


def _pair(status: str, consistency: float = 0.8, median: float = 0.05):
    return {
        "status": status,
        "a_to_b": {
            "consistent_ratio_of_comparable": consistency,
            "median_absolute_relative_depth_residual": median,
        },
        "b_to_a": {
            "consistent_ratio_of_comparable": consistency - 0.02,
            "median_absolute_relative_depth_residual": median + 0.01,
        },
    }


def test_validate_e2_summary_requires_passing_authorized_exact_ten_view_seed():
    _validate_e2_summary(_e2_summary())
    with pytest.raises(ReconstructionValidationError):
        _validate_e2_summary(_e2_summary(passed=False))
    with pytest.raises(ReconstructionValidationError):
        _validate_e2_summary(_e2_summary(authorized=False))
    bad = _e2_summary()
    bad["selected_workspace_orders"] = bad["selected_workspace_orders"][:-1]
    with pytest.raises(ReconstructionValidationError):
        _validate_e2_summary(bad)


def test_threshold_validation_requires_complete_unchanged_d4_3_evidence():
    assert _validate_thresholds(_thresholds()) == _thresholds()
    bad = _thresholds()
    del bad["maximum_p90_relative_residual"]
    with pytest.raises(ReconstructionValidationError):
        _validate_thresholds(bad)
    bad = _thresholds()
    bad["consistency_relative_tolerance"] = 0.30
    with pytest.raises(ReconstructionValidationError):
        _validate_thresholds(bad)


def test_candidate_pool_expands_only_outside_seed_and_never_reopens_interior_order_8():
    registered = range(40)
    seed = [5, 6, 7, 9, 10, 11, 12, 13, 14, 15]
    left, right = _candidate_workspace_orders(
        registered, seed, left_count=5, right_count=8
    )
    assert left == (0, 1, 2, 3, 4)
    assert right == (16, 17, 18, 19, 20, 21, 22, 23)
    assert 8 not in left + right


def test_candidate_pool_respects_registered_boundaries_and_requested_counts():
    left, right = _candidate_workspace_orders(
        [2, 4, 5, 6, 7, 9, 10, 15, 17, 20],
        [5, 6, 7, 9, 10, 15],
        left_count=1,
        right_count=1,
    )
    assert left == (4,)
    assert right == (17,)


def test_growth_path_bridges_known_bad_adjacent_transition_without_threshold_relaxation():
    pairs = {
        (15, 16): _pair("degraded", 0.4, 0.14),
        (15, 17): _pair("pass", 0.83, 0.07),
        (16, 17): _pair("pass", 0.91, 0.03),
        (17, 18): _pair("pass", 0.90, 0.04),
    }
    path = _best_growth_path(
        anchor_order=15,
        candidate_orders=[16, 17, 18],
        pair_summaries=pairs,
        direction="right",
    )
    assert path.orders_anchor_outward == (15, 17, 18)
    assert 16 not in path.added_orders


def test_growth_path_prefers_more_qualified_views_before_edge_quality():
    pairs = {
        (5, 4): _pair("pass", 0.99, 0.01),
        (5, 3): _pair("pass", 0.75, 0.07),
        (4, 3): _pair("degraded", 0.40, 0.15),
        (3, 2): _pair("pass", 0.76, 0.06),
    }
    path = _best_growth_path(
        anchor_order=5,
        candidate_orders=[4, 3, 2],
        pair_summaries=pairs,
        direction="left",
    )
    assert path.orders_anchor_outward == (5, 3, 2)




def test_expanded_support_masks_accept_support_counts_above_ten_views():
    counts = np.asarray([1, 2, 3, 11, 20], dtype=np.int64)
    supported, high = _expanded_support_masks(
        counts, minimum_supported_views=2, high_support_views=3, maximum_views=20
    )
    assert supported.tolist() == [False, True, True, True, True]
    assert high.tolist() == [False, False, True, True, True]

def test_pair_quality_uses_worst_bidirectional_consistency_and_median():
    consistency, median = _pair_quality(_pair("pass", 0.82, 0.04))
    assert consistency == pytest.approx(0.80)
    assert median == pytest.approx(0.05)
    assert _pair_quality(_pair("degraded"))[0] == 0.0


def test_robust_diagonal_ignores_extreme_endpoint_outliers_better_than_raw_bounds():
    line = np.column_stack([
        np.linspace(0.0, 10.0, 1000),
        np.zeros(1000),
        np.zeros(1000),
    ])
    points = np.vstack([line, [[1000.0, 0.0, 0.0]]])
    value = _robust_diagonal(points)
    assert 9.0 < value < 11.0


def test_e3_preview_uses_proven_concrete_canvas_and_sampling_imports():
    preview = Path(__file__).resolve().parents[2] / "scripts" / "run_e3_qualified_scene_fusion_preview.py"
    source = preview.read_text(encoding="utf-8")
    assert "from app.widgets.viewport.vispy_canvas import PointCloudCanvas" in source
    assert "from app.widgets.viewport import PointCloudCanvas" not in source
    assert "from core.geometry.sampling import enforce_point_budget" in source
    assert "from core.visualization_sampling import enforce_point_budget" not in source


def test_e3_scope_keeps_f_g_h_j_authority_out_of_expansion_stage():
    script = Path(__file__).resolve().parents[2] / "scripts" / "run_e3_qualified_scene_fusion_expansion.py"
    source = script.read_text(encoding="utf-8")
    assert '"confidence_artifact_filtering_formalized": False' in source
    assert '"canonical_persistence_promoted": False' in source
    assert '"surface_mesh_generated": False' in source
    assert '"authorizes_next_confidence_filtering_milestone": passed' in source

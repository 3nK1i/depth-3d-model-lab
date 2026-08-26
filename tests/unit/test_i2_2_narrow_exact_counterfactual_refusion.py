from __future__ import annotations

import numpy as np
import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.offline_dense_counterfactual_refusion import (
    CounterfactualScenario,
    comparison_against_baseline,
    pareto_front,
    scenario_quality_profile,
    scenarios_from_i21,
    validate_scenario_against_baseline,
)
from core.reconstruction.offline_dense_quality import I2QualityPolicy


def _i21() -> dict:
    return {
        "diagnostic_findings": {
            "top_single_view_dependencies": [
                {"workspace_order": 5}, {"workspace_order": 1}, {"workspace_order": 4},
                {"workspace_order": 30}, {"workspace_order": 31},
            ]
        },
        "top_support2_cosupport_pairs": [
            {"order_a": 30, "order_b": 31},
            {"order_a": 25, "order_b": 26},
            {"order_a": 3, "order_b": 5},
            {"order_a": 4, "order_b": 5},
            {"order_a": 1, "order_b": 2},
        ],
    }


def test_scenarios_from_i21_matches_ranked_evidence() -> None:
    selected = tuple(range(40))
    scenarios = scenarios_from_i21(_i21(), selected)
    assert [s.name for s in scenarios[:5]] == ["minus_5", "minus_1", "minus_4", "minus_30", "minus_31"]
    assert scenarios[5].excluded_orders == (30, 31)


def test_scenarios_are_deduplicated() -> None:
    payload = _i21(); payload["top_support2_cosupport_pairs"].insert(0, {"order_a": 5, "order_b": 5})
    scenarios = scenarios_from_i21(payload, tuple(range(40)))
    assert all(len(set(s.excluded_orders)) == len(s.excluded_orders) for s in scenarios)


def test_scenario_contract_sorts_exclusions() -> None:
    scenario = CounterfactualScenario("pair", (31, 30), "test")
    assert scenario.excluded_orders == (30, 31)


def test_scenario_contract_rejects_empty() -> None:
    with pytest.raises(ReconstructionValidationError):
        CounterfactualScenario("bad", (), "test")


def test_validate_scenario_requires_selected_order() -> None:
    with pytest.raises(ReconstructionValidationError):
        validate_scenario_against_baseline(CounterfactualScenario("x", (99,), "test"), tuple(range(20)))


def test_validate_scenario_preserves_order() -> None:
    remaining = validate_scenario_against_baseline(CounterfactualScenario("x", (5, 7), "test"), tuple(range(15)))
    assert 5 not in remaining and 7 not in remaining and remaining == tuple(v for v in range(15) if v not in {5, 7})


def _cloud() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(4)
    core = rng.normal(0.0, 0.2, size=(120, 3))
    tail = np.column_stack([np.linspace(1.0, 4.0, 30), np.zeros(30), np.zeros(30)])
    return np.vstack([core, tail]), core


def _profile(points: np.ndarray, core: np.ndarray):
    n = len(points)
    return scenario_quality_profile(
        extended_points=points,
        core_points=core,
        support=np.full(n, 2, dtype=np.uint32),
        confidence=np.full(n, 0.6, dtype=np.float32),
        dispersion=np.full(n, 0.2, dtype=np.float32),
        component_size=np.full(n, n, dtype=np.int32),
        voxel_size=0.05,
        baseline_core_points=core,
        baseline_core_confidence=np.full(len(core), 0.65, dtype=np.float32),
        baseline_core_dispersion=np.full(len(core), 0.15, dtype=np.float32),
        policy=I2QualityPolicy(),
    )


def test_quality_profile_detects_peripheral_tail() -> None:
    points, core = _cloud(); profile, masks = _profile(points, core)
    assert profile["baseline_anchor"]["peripheral_points"] > 0
    assert np.count_nonzero(masks["peripheral"]) == profile["baseline_anchor"]["peripheral_points"]


def test_quality_profile_detects_weak_support2_periphery() -> None:
    points, core = _cloud(); profile, _ = _profile(points, core)
    assert profile["baseline_anchor"]["weak_peripheral_points"] == profile["baseline_anchor"]["peripheral_points"]


def test_quality_profile_reports_dominant_component_fraction() -> None:
    points, core = _cloud(); profile, _ = _profile(points, core)
    assert profile["dominant_component_fraction_of_extended"] == pytest.approx(1.0)


def test_comparison_reports_artifact_reduction_and_core_loss() -> None:
    points, core = _cloud(); baseline, _ = _profile(points, core)
    scenario, _ = _profile(points[:-20], core)
    result = comparison_against_baseline(baseline, scenario)
    assert result["peripheral_reduction_fraction"] > 0
    assert result["core_loss_fraction"] == pytest.approx(0.0)


def test_comparison_marks_promising_surgical_change() -> None:
    baseline = {
        "extended_points": 1000, "core_points": 800,
        "baseline_anchor": {"peripheral_points": 100, "far_peripheral_points": 50, "weak_peripheral_points": 90},
        "extended_envelope": {"full_to_robust_diagonal_ratio": 2.0, "robust_diagonal": 10.0},
    }
    scenario = {
        "extended_points": 960, "core_points": 790,
        "baseline_anchor": {"peripheral_points": 75, "far_peripheral_points": 30, "weak_peripheral_points": 68},
        "extended_envelope": {"full_to_robust_diagonal_ratio": 1.6, "robust_diagonal": 9.8},
    }
    result = comparison_against_baseline(baseline, scenario)
    assert result["promising_for_manual_exact_review"] is True


def test_comparison_rejects_destructive_core_loss() -> None:
    baseline = {
        "extended_points": 1000, "core_points": 800,
        "baseline_anchor": {"peripheral_points": 100, "far_peripheral_points": 50, "weak_peripheral_points": 90},
        "extended_envelope": {"full_to_robust_diagonal_ratio": 2.0, "robust_diagonal": 10.0},
    }
    scenario = {
        "extended_points": 700, "core_points": 600,
        "baseline_anchor": {"peripheral_points": 20, "far_peripheral_points": 5, "weak_peripheral_points": 15},
        "extended_envelope": {"full_to_robust_diagonal_ratio": 1.2, "robust_diagonal": 9.0},
    }
    assert comparison_against_baseline(baseline, scenario)["promising_for_manual_exact_review"] is False


def test_pareto_front_removes_dominated_scenario() -> None:
    records = [
        {"name": "a", "comparison": {"core_loss_fraction": 0.02, "far_peripheral_reduction_fraction": 0.30, "peripheral_reduction_fraction": 0.20}},
        {"name": "b", "comparison": {"core_loss_fraction": 0.03, "far_peripheral_reduction_fraction": 0.20, "peripheral_reduction_fraction": 0.10}},
        {"name": "c", "comparison": {"core_loss_fraction": 0.01, "far_peripheral_reduction_fraction": 0.10, "peripheral_reduction_fraction": 0.08}},
    ]
    front = pareto_front(records)
    assert "b" not in front and {"a", "c"}.issubset(front)

def test_scenario_subset_limits_are_respected() -> None:
    scenarios = scenarios_from_i21(_i21(), tuple(range(40)), top_single_views=2, top_pairs=1)
    assert [item.name for item in scenarios] == ["minus_5", "minus_1", "minus_30_31"]

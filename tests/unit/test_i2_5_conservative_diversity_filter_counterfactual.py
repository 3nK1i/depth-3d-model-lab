from __future__ import annotations

import numpy as np
import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.offline_dense_diversity_filter_counterfactual import (
    DiversityFilterScenario,
    confidence_reference_threshold,
    counterfactual_quality_comparison,
    default_diversity_filter_scenarios,
    diversity_filter_rejection_mask,
    pareto_front,
    policy_effect_counts,
)


def _evidence():
    return {
        "support": np.asarray([2, 2, 2, 3, 3, 2], dtype=np.int64),
        "angle": np.asarray([2.0, 4.0, 8.0, 2.0, 8.0, 1.0]),
        "ratio": np.asarray([0.02, 0.05, 0.12, 0.02, 0.12, 0.01]),
        "high": np.asarray([1, 2, 3, 1, 4, 1], dtype=np.int64),
        "radius2": np.asarray([8, 10, 16, 9, 20, 14], dtype=np.int64),
        "confidence": np.asarray([0.35, 0.45, 0.70, 0.30, 0.80, 0.20]),
        "reference": np.asarray([0, 1, 1, 0, 1, 0], dtype=bool),
    }


def _mask(scenario: DiversityFilterScenario):
    e = _evidence()
    return diversity_filter_rejection_mask(
        scenario=scenario,
        support=e["support"],
        maximum_pair_parallax_degrees=e["angle"],
        maximum_baseline_to_range_ratio=e["ratio"],
        high_support_neighbors=e["high"],
        radius2_neighbors=e["radius2"],
        confidence=e["confidence"],
        reference_mask=e["reference"],
        parallax_reference_threshold_degrees=5.0,
        baseline_to_range_reference_threshold=0.06,
    )


def test_i25_default_policy_ladder_is_fixed_and_unique() -> None:
    scenarios = default_diversity_filter_scenarios()
    assert len(scenarios) == 6
    assert len({item.name for item in scenarios}) == 6
    assert all(item.maximum_support in (2, 3) for item in scenarios)


def test_i25_scenario_rejects_invalid_support_limit() -> None:
    with pytest.raises(ReconstructionValidationError):
        DiversityFilterScenario("bad", 4, 1.0)


def test_i25_scenario_rejects_invalid_diversity_scale() -> None:
    with pytest.raises(ReconstructionValidationError):
        DiversityFilterScenario("bad", 2, 1.25)


def test_confidence_reference_threshold_uses_requested_quantile() -> None:
    values = np.asarray([0.1, 0.2, 0.6, 0.8])
    mask = np.asarray([0, 1, 1, 1], dtype=bool)
    assert confidence_reference_threshold(values, reference_mask=mask, quantile=0.5) == pytest.approx(0.6)


def test_rejection_requires_angle_and_baseline_ratio_to_be_low() -> None:
    scenario = DiversityFilterScenario("basic", 2, 1.0)
    reject, _ = _mask(scenario)
    assert reject.tolist() == [True, True, False, False, False, True]


def test_high_support_anchor_constraint_is_conjunctive() -> None:
    scenario = DiversityFilterScenario("anchor1", 2, 1.0, maximum_high_support_neighbors=1)
    reject, _ = _mask(scenario)
    assert reject.tolist() == [True, False, False, False, False, True]


def test_radius2_constraint_can_preserve_dense_local_geometry() -> None:
    scenario = DiversityFilterScenario(
        "radius", 2, 1.0, maximum_high_support_neighbors=2, maximum_radius2_neighbors=10
    )
    reject, _ = _mask(scenario)
    assert reject.tolist() == [True, True, False, False, False, False]


def test_confidence_constraint_is_resolved_from_reference() -> None:
    scenario = DiversityFilterScenario(
        "conf", 3, 1.0, maximum_high_support_neighbors=1, maximum_confidence_reference_quantile=0.5
    )
    reject, resolved = _mask(scenario)
    assert resolved["resolved_confidence_limit"] == pytest.approx(0.70)
    assert reject.tolist() == [True, False, False, True, False, True]


def test_policy_effect_counts_report_core_and_artifact_regions() -> None:
    reject = np.asarray([1, 0, 1, 0, 0, 1], dtype=bool)
    core = np.asarray([1, 1, 0, 1, 0, 0], dtype=bool)
    peripheral = np.asarray([0, 0, 1, 1, 1, 1], dtype=bool)
    far = np.asarray([0, 0, 1, 0, 0, 1], dtype=bool)
    weak = peripheral.copy()
    result = policy_effect_counts(
        reject, is_core=core, is_peripheral=peripheral, is_far_peripheral=far, is_weak_peripheral=weak
    )
    assert result["rejected_points"] == 3
    assert result["core_rejected_points"] == 1
    assert result["far_peripheral_reduction_fraction"] == 1.0


def test_central_core_loss_excludes_peripheral_core() -> None:
    reject = np.asarray([0, 1, 0, 0], dtype=bool)
    core = np.asarray([1, 1, 1, 0], dtype=bool)
    peripheral = np.asarray([0, 1, 0, 1], dtype=bool)
    result = policy_effect_counts(
        reject,
        is_core=core,
        is_peripheral=peripheral,
        is_far_peripheral=peripheral,
        is_weak_peripheral=peripheral,
    )
    assert result["core_loss_fraction"] == pytest.approx(1 / 3)
    assert result["central_core_loss_fraction"] == 0.0


def test_quality_comparison_can_mark_conservative_candidate_balanced() -> None:
    effects = {
        "central_core_loss_fraction": 0.01,
        "peripheral_reduction_fraction": 0.20,
        "far_peripheral_reduction_fraction": 0.35,
        "weak_peripheral_reduction_fraction": 0.25,
    }
    result = counterfactual_quality_comparison(
        baseline_extended_points=1000,
        baseline_core_points=800,
        baseline_full_to_robust_ratio=2.3,
        baseline_robust_diagonal=10.0,
        candidate_extended_points=960,
        candidate_core_points=784,
        candidate_full_to_robust_ratio=2.1,
        candidate_robust_diagonal=9.8,
        candidate_dominant_component_fraction=0.95,
        effect_counts=effects,
    )
    assert result["balanced_for_manual_review"] is True
    assert result["preferred_for_manual_review"] is True


def test_quality_comparison_rejects_excessive_core_loss() -> None:
    effects = {
        "central_core_loss_fraction": 0.05,
        "peripheral_reduction_fraction": 0.50,
        "far_peripheral_reduction_fraction": 0.70,
        "weak_peripheral_reduction_fraction": 0.50,
    }
    result = counterfactual_quality_comparison(
        baseline_extended_points=1000,
        baseline_core_points=800,
        baseline_full_to_robust_ratio=2.3,
        baseline_robust_diagonal=10.0,
        candidate_extended_points=900,
        candidate_core_points=720,
        candidate_full_to_robust_ratio=1.8,
        candidate_robust_diagonal=9.8,
        candidate_dominant_component_fraction=0.95,
        effect_counts=effects,
    )
    assert result["balanced_for_manual_review"] is False


def test_pareto_front_removes_dominated_policy() -> None:
    records = [
        {
            "name": "a",
            "comparison": {
                "core_loss_fraction": .02,
                "central_core_loss_fraction": .01,
                "far_peripheral_reduction_fraction": .30,
                "peripheral_reduction_fraction": .20,
                "full_to_robust_ratio_improvement_fraction": .05,
            },
        },
        {
            "name": "b",
            "comparison": {
                "core_loss_fraction": .03,
                "central_core_loss_fraction": .02,
                "far_peripheral_reduction_fraction": .20,
                "peripheral_reduction_fraction": .15,
                "full_to_robust_ratio_improvement_fraction": .01,
            },
        },
    ]
    assert pareto_front(records) == ("a",)


def test_rule_resolution_reports_rejected_count_without_region_labels() -> None:
    scenario = DiversityFilterScenario("basic", 2, 1.0)
    reject, resolved = _mask(scenario)
    assert resolved["rejected_points"] == int(np.count_nonzero(reject))
    # Filtering itself has no peripheral/far parameters; those are evaluation-only.
    assert "peripheral" not in resolved
    assert "far" not in resolved

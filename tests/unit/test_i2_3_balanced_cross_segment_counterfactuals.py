from __future__ import annotations

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.offline_dense_balanced_counterfactuals import (
    BalancedCrossSegmentScenario,
    balanced_pareto_front,
    derive_balanced_cross_segment_scenarios,
    evaluate_balanced_result,
    single_by_order,
)


def _single(name: str, order: int, *, core: float, far: float, peripheral: float, ratio: float) -> dict:
    return {
        "name": name,
        "excluded_orders": [order],
        "comparison": {
            "core_loss_fraction": core,
            "extended_loss_fraction": core,
            "peripheral_reduction_fraction": peripheral,
            "far_peripheral_reduction_fraction": far,
            "weak_peripheral_reduction_fraction": peripheral,
            "full_to_robust_ratio_improvement_fraction": ratio,
            "extended_robust_diagonal_change_fraction": 0.0,
            "promising_for_manual_exact_review": True,
        },
    }


def _i22() -> dict:
    return {
        "status": "exact_counterfactuals_ready",
        "scenarios": [
            _single("minus_5", 5, core=.05, far=.48, peripheral=.25, ratio=-.05),
            _single("minus_1", 1, core=.046, far=.36, peripheral=.20, ratio=-.03),
            _single("minus_4", 4, core=.046, far=.47, peripheral=.21, ratio=-.08),
            _single("minus_30", 30, core=.044, far=.05, peripheral=.18, ratio=.017),
            _single("minus_31", 31, core=.048, far=.035, peripheral=.18, ratio=.032),
        ],
    }


def test_scenario_contract_orders_temporally() -> None:
    s = BalancedCrossSegmentScenario("x", 31, 5)
    assert s.excluded_orders == (5, 31)


def test_scenario_contract_rejects_same_view() -> None:
    with pytest.raises(ReconstructionValidationError):
        BalancedCrossSegmentScenario("x", 5, 5)


def test_single_by_order_maps_exact_singles() -> None:
    assert set(single_by_order(_i22())) == {1, 4, 5, 30, 31}


def test_derivation_forms_cross_product() -> None:
    scenarios = derive_balanced_cross_segment_scenarios(_i22(), [0, 1, 4, 5, 10, 20, 30, 31, 38, 39])
    assert {s.excluded_orders for s in scenarios} == {(5, 31), (5, 30), (4, 31), (4, 30), (1, 31), (1, 30)}


def test_derivation_prioritizes_early_far_reduction() -> None:
    scenarios = derive_balanced_cross_segment_scenarios(_i22(), list(range(40)), max_early=1, max_late=1)
    assert scenarios[0].early_order == 5


def test_derivation_prioritizes_late_envelope_gain() -> None:
    scenarios = derive_balanced_cross_segment_scenarios(_i22(), list(range(40)), max_early=1, max_late=1)
    assert scenarios[0].late_order == 31


def test_derivation_requires_both_temporal_sides() -> None:
    payload = _i22(); payload["scenarios"] = payload["scenarios"][:3]
    with pytest.raises(ReconstructionValidationError):
        derive_balanced_cross_segment_scenarios(payload, list(range(40)))


def _combined(core=.06, far=.55, peripheral=.30, ratio=.10, dominant=.97) -> dict:
    return {
        "name": "minus_5_31",
        "comparison": {
            "core_loss_fraction": core,
            "extended_loss_fraction": .07,
            "peripheral_reduction_fraction": peripheral,
            "far_peripheral_reduction_fraction": far,
            "weak_peripheral_reduction_fraction": .28,
            "full_to_robust_ratio_improvement_fraction": ratio,
            "extended_robust_diagonal_change_fraction": -.02,
        },
        "quality_profile": {
            "extended_envelope": {"full_to_robust_diagonal_ratio": 2.1},
            "dominant_component_fraction_of_extended": dominant,
        },
    }


def test_balanced_evaluation_accepts_surgical_cross_segment_gain() -> None:
    singles = single_by_order(_i22())
    ev = evaluate_balanced_result(_combined(), singles[5], singles[31], {"dominant_component_fraction_of_extended": .99})
    assert ev["balanced_candidate_for_manual_review"] is True


def test_balanced_evaluation_rejects_high_core_loss() -> None:
    singles = single_by_order(_i22())
    ev = evaluate_balanced_result(_combined(core=.09), singles[5], singles[31], {"dominant_component_fraction_of_extended": .99})
    assert ev["balanced_candidate_for_manual_review"] is False


def test_balanced_evaluation_requires_envelope_improvement() -> None:
    singles = single_by_order(_i22())
    ev = evaluate_balanced_result(_combined(ratio=-.01), singles[5], singles[31], {"dominant_component_fraction_of_extended": .99})
    assert ev["balanced_candidate_for_manual_review"] is False


def test_balanced_evaluation_reports_core_overlap_savings() -> None:
    singles = single_by_order(_i22())
    ev = evaluate_balanced_result(_combined(core=.06), singles[5], singles[31], {"dominant_component_fraction_of_extended": .99})
    assert ev["core_overlap_savings_fraction"] == pytest.approx(.05 + .048 - .06)


def test_balanced_evaluation_reports_gain_over_best_single() -> None:
    singles = single_by_order(_i22())
    ev = evaluate_balanced_result(_combined(far=.60), singles[5], singles[31], {"dominant_component_fraction_of_extended": .99})
    assert ev["far_gain_over_best_single_fraction"] == pytest.approx(.60 - .48)


def test_balanced_pareto_removes_dominated_record() -> None:
    records = [
        {"name": "a", "balanced_evaluation": {"combined_core_loss_fraction": .05, "combined_far_peripheral_reduction_fraction": .5, "combined_peripheral_reduction_fraction": .3, "combined_full_to_robust_ratio_improvement_fraction": .1}},
        {"name": "b", "balanced_evaluation": {"combined_core_loss_fraction": .06, "combined_far_peripheral_reduction_fraction": .4, "combined_peripheral_reduction_fraction": .2, "combined_full_to_robust_ratio_improvement_fraction": .05}},
        {"name": "c", "balanced_evaluation": {"combined_core_loss_fraction": .04, "combined_far_peripheral_reduction_fraction": .2, "combined_peripheral_reduction_fraction": .15, "combined_full_to_robust_ratio_improvement_fraction": .12}},
    ]
    front = balanced_pareto_front(records)
    assert "b" not in front and {"a", "c"}.issubset(front)


def test_derivation_rejects_non_ready_i22() -> None:
    payload = _i22(); payload["status"] = "running"
    with pytest.raises(ReconstructionValidationError):
        derive_balanced_cross_segment_scenarios(payload, list(range(40)))

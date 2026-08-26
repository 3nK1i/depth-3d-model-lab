from __future__ import annotations

import numpy as np
import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.offline_dense_manifold_filter_counterfactual import (
    ManifoldFilterScenario,
    default_manifold_filter_scenarios,
    manifold_counterfactual_comparison,
    manifold_effect_counts,
    manifold_pareto_front,
    manifold_rejection_mask,
)


def _rule(scenario: ManifoldFilterScenario, distance, knn, linearity):
    return manifold_rejection_mask(
        scenario=scenario,
        nearest_strong_anchor_distance_voxels=np.asarray(distance, dtype=float),
        knn_median_distance_voxels=np.asarray(knn, dtype=float),
        local_linearity=np.asarray(linearity, dtype=float),
        anchor_distance_reference_q90_voxels=1.2,
        knn_spacing_reference_q90_voxels=1.6,
        linearity_reference_q90=.48,
    )


def test_scenario_rejects_invalid_distance_scale() -> None:
    with pytest.raises(ReconstructionValidationError):
        ManifoldFilterScenario("bad", .5)


def test_default_scenarios_are_small_and_unique() -> None:
    scenarios = default_manifold_filter_scenarios()
    assert len(scenarios) == 5
    assert len({s.name for s in scenarios}) == len(scenarios)


def test_scenario_contract_contains_no_spatial_or_extremum_filter_field() -> None:
    payload = default_manifold_filter_scenarios()[0].as_dict()
    assert not any("peripheral" in key or "extremum" in key or "view" in key for key in payload)


def test_rejection_requires_detachment_and_sparsity() -> None:
    reject, _ = _rule(ManifoldFilterScenario("x", 4), [6, 6, 2], [2, 1, 2], [.6, .6, .6])
    assert reject.tolist() == [True, False, False]


def test_rejection_thresholds_are_strict() -> None:
    reject, _ = _rule(ManifoldFilterScenario("x", 4), [4.8, 4.8001], [2, 2], [.6, .6])
    assert reject.tolist() == [False, True]


def test_linearity_guard_narrows_rule() -> None:
    scenario = ManifoldFilterScenario("x", 4, require_linearity=True)
    reject, _ = _rule(scenario, [8, 8], [2, 2], [.3, .8])
    assert reject.tolist() == [False, True]


def test_rejection_rejects_misaligned_arrays() -> None:
    with pytest.raises(ReconstructionValidationError):
        _rule(ManifoldFilterScenario("x", 4), [8, 8], [2], [.6, .6])


def test_effect_counts_report_extremum_and_stubborn_rejection() -> None:
    reject = np.array([1, 0, 1, 0], dtype=bool)
    effects = manifold_effect_counts(
        reject,
        is_core=[1, 1, 0, 0],
        is_peripheral=[0, 0, 1, 1],
        is_far_peripheral=[0, 0, 1, 0],
        is_weak_peripheral=[0, 0, 1, 1],
        is_extremum_driver=[0, 0, 1, 1],
        is_stubborn_extremum_survivor=[0, 0, 1, 0],
    )
    assert effects["extremum_driver_rejection_fraction"] == .5
    assert effects["stubborn_extremum_rejection_fraction"] == 1.0


def test_effect_counts_track_central_core_protection() -> None:
    effects = manifold_effect_counts(
        [0, 1, 0], is_core=[1, 1, 0], is_peripheral=[0, 1, 1], is_far_peripheral=[0, 1, 0],
        is_weak_peripheral=[0, 1, 0], is_extremum_driver=[0, 1, 0], is_stubborn_extremum_survivor=[0, 1, 0]
    )
    assert effects["central_core_rejection_fraction"] == 0.0


def _comparison(**overrides):
    effects = {
        "central_core_rejection_fraction": .003,
        "peripheral_rejection_fraction": .30,
        "far_peripheral_rejection_fraction": .40,
        "extremum_driver_rejection_fraction": .80,
        "stubborn_extremum_rejection_fraction": .75,
    }
    effects.update(overrides.pop("effects", {}))
    values = dict(
        baseline_extended_points=100000, baseline_core_points=80000,
        baseline_full_diagonal=20.0, baseline_robust_diagonal=10.0,
        candidate_extended_points=98000, candidate_core_points=79500,
        candidate_full_diagonal=17.0, candidate_robust_diagonal=9.9,
        candidate_dominant_component_fraction=.96, effects=effects,
    )
    values.update(overrides)
    return manifold_counterfactual_comparison(**values)


def test_comparison_can_be_balanced() -> None:
    result = _comparison()
    assert result["balanced_for_manual_review"]


def test_comparison_can_be_preferred() -> None:
    result = _comparison(candidate_core_points=79800, candidate_extended_points=98500)
    assert result["preferred_for_manual_review"]


def test_ratio_worsening_blocks_balanced_status() -> None:
    result = _comparison(candidate_full_diagonal=19.5, candidate_robust_diagonal=9.0)
    assert result["full_to_robust_ratio_improvement_fraction"] < 0
    assert not result["balanced_for_manual_review"]


def test_large_core_loss_blocks_preferred_status() -> None:
    result = _comparison(candidate_core_points=76000)
    assert not result["preferred_for_manual_review"]


def test_pareto_front_removes_dominated_candidate() -> None:
    strong = {"name": "strong", "comparison": _comparison()}
    weak = {"name": "weak", "comparison": _comparison(
        candidate_core_points=79000,
        effects={
            "central_core_rejection_fraction": .006,
            "peripheral_rejection_fraction": .20,
            "far_peripheral_rejection_fraction": .25,
            "extremum_driver_rejection_fraction": .60,
            "stubborn_extremum_rejection_fraction": .55,
        },
        candidate_full_diagonal=18.0,
        candidate_robust_diagonal=9.8,
    )}
    assert manifold_pareto_front([strong, weak]) == ("strong",)

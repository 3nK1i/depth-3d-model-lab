from __future__ import annotations

from core.reconstruction.i5_remaining_depth_provider_robustness import (
    I59A_B1_PROVIDERS, I59A_B2_PROVIDERS,
    build_i5_9a_remaining_depth_provider_robustness_admission,
)
from tests.i5_9a_helpers import build_args


def test_i59a_artifact_contract_binds_predecessors_inventory_batches_and_roadmap():
    result = build_i5_9a_remaining_depth_provider_robustness_admission(**build_args())
    assert result["phase"] == "I5.9A"
    assert result["status"] == "remaining_depth_provider_robustness_inventory_and_scenarios_admitted"
    assert result["pass"] is True
    assert set(result["provider_inventory"]) == set(I59A_B1_PROVIDERS + I59A_B2_PROVIDERS)
    assert result["batch_plan"]["I5.9B1"]["providers"] == list(I59A_B1_PROVIDERS)
    assert result["batch_plan"]["I5.9B2"]["providers"] == list(I59A_B2_PROVIDERS)
    assert result["historical_g4_anchors"]["low_texture"]["depth_anything_v2"] == "rejected"
    assert result["historical_g4_anchors"]["low_texture"]["marigold"] == "rejected"
    assert result["decision"]["i5_10_started"] is False
    assert result["next"] == "I5.9B1 metric-like single-image provider robustness replay."


def test_i59a_does_not_compare_or_rank_providers():
    result = build_i5_9a_remaining_depth_provider_robustness_admission(**build_args())
    assert result["scope"]["provider_comparison_started"] is False
    assert result["decision"]["provider_ranking_created"] is False
    assert result["decision"]["universal_provider_winner_declared"] is False
    assert result["decision"]["global_scalar_score_created"] is False

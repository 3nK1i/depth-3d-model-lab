from __future__ import annotations

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.offline_dense_i4_decoupled_sampling_counterfactual import (
    build_counterfactual_plans,
    build_i4_1_5_counterfactual,
    estimate_plan_pass_fraction,
    nominal_local_count,
    plan_geometry_summary,
    uniform_times,
)


def _bucket(p):
    return {"pass_fraction": p}


def _evidence(p0=.27, p1=.08, p2=0.0, p3=None):
    return {
        "pass_fraction": .12,
        "time_buckets": {
            "le_0_75s": _bucket(p0),
            "gt_0_75_le_1_5s": _bucket(p1),
            "gt_1_5_le_3_0s": _bucket(p2),
            "gt_3_0s": _bucket(p3),
        },
    }


def _summary():
    return {
        "diagnosis": {
            "classification": "mixed_capture_regime_and_local_density",
            "local_density_signal": True,
            "capture_regime_signal": True,
        },
        "current_full_span": _evidence(.03,.03,.03,0.0),
        "same_source_dense_prefix": _evidence(.269,.08,0.0,None),
        "short_control": _evidence(.679,.577,.208,None),
    }


def test_uniform_times_include_both_ends():
    values=uniform_times(10.0,5)
    assert values[0] == 0.0 and values[-1] == 10.0 and len(values) == 5


def test_nominal_local_count_matches_full_span_intent():
    assert nominal_local_count(50.5838666667, 2.0) == 102


def test_plans_preserve_same_global_anchor_backbone():
    plans=build_counterfactual_plans(duration_seconds=50.0)
    assert len({p.global_anchor_times_seconds for p in plans}) == 1
    assert all(p.global_anchor_count == 40 for p in plans)


def test_baseline_reuses_global_anchors_as_dense_candidates():
    baseline=build_counterfactual_plans(duration_seconds=50.0)[0]
    assert baseline.dense_candidate_times_seconds == baseline.global_anchor_times_seconds
    assert baseline.dense_candidates_require_localization is False


def test_localized_pool_is_bounded():
    plan=build_counterfactual_plans(duration_seconds=50.6,dense_candidate_cap=120)[2]
    assert plan.dense_candidate_count <= 120
    assert plan.dense_candidates_require_localization is True


def test_localized_pool_preserves_full_span():
    plan=build_counterfactual_plans(duration_seconds=50.6)[2]
    g=plan_geometry_summary(plan)
    assert g["source_span_fraction"] == pytest.approx(1.0)


def test_localized_pool_restores_local_gap():
    baseline,_,local=build_counterfactual_plans(duration_seconds=50.6)
    assert plan_geometry_summary(local)["dense_adjacent_gap_seconds"]["p50"] < plan_geometry_summary(baseline)["dense_adjacent_gap_seconds"]["p50"]
    assert plan_geometry_summary(local)["adjacent_local_fraction_le_threshold"] > .95


def test_estimator_does_not_invent_unsupported_gt3_pairs():
    baseline=build_counterfactual_plans(duration_seconds=50.6)[0]
    estimate=estimate_plan_pass_fraction(baseline,evidence_summary=_evidence())
    assert estimate["unscored_pair_count"] > 0
    assert estimate["actual_dense_execution"] is False


def test_estimator_scores_local_pool_from_empirical_buckets():
    local=build_counterfactual_plans(duration_seconds=50.6)[2]
    estimate=estimate_plan_pass_fraction(local,evidence_summary=_evidence())
    assert estimate["estimated_pass_fraction"] is not None
    assert 0.0 <= estimate["estimated_pass_fraction"] <= 1.0


def test_counterfactual_recommends_decoupling_when_density_signal_is_real():
    result=build_i4_1_5_counterfactual(duration_seconds=50.5838666667,i4_1_4_summary=_summary())
    assert result["decision"]["decoupled_sampling_architecture_candidate"] is True
    assert result["decision"]["i4_1_6_authorized"] is True
    assert result["decision"]["recommended_counterfactual_plan"] is not None


def test_counterfactual_keeps_capture_warning():
    result=build_i4_1_5_counterfactual(duration_seconds=50.0,i4_1_4_summary=_summary())
    assert result["decision"]["cleaner_static_capture_still_recommended"] is True


def test_counterfactual_never_authorizes_threshold_relaxation_or_mesh():
    result=build_i4_1_5_counterfactual(duration_seconds=50.0,i4_1_4_summary=_summary())
    assert result["decision"]["production_sampling_integration_authorized"] is False
    assert result["decision"]["dense_consistency_threshold_relaxation_authorized"] is False
    assert result["decision"]["mesh_authorization"] is False
    assert result["authority"]["sampling_policy_changed"] is False


def test_invalid_budget_rejected():
    with pytest.raises(ReconstructionValidationError):
        build_counterfactual_plans(duration_seconds=10.0,dense_candidate_cap=10,final_fusion_cap=40)

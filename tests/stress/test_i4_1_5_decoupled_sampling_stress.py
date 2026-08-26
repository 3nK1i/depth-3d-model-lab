from __future__ import annotations

from core.reconstruction.offline_dense_i4_decoupled_sampling_counterfactual import build_counterfactual_plans, plan_geometry_summary


def test_long_video_candidate_pool_remains_bounded():
    plans=build_counterfactual_plans(duration_seconds=3600.0,global_anchor_cap=40,dense_candidate_cap=120,local_target_fps=2.0,final_fusion_cap=40)
    assert all(p.global_anchor_count == 40 for p in plans)
    assert all(p.dense_candidate_count <= 120 for p in plans)
    assert all(plan_geometry_summary(p)["source_span_fraction"] == 1.0 for p in plans)


def test_repeated_planning_is_deterministic():
    first=build_counterfactual_plans(duration_seconds=50.5838666667)
    for _ in range(100):
        assert build_counterfactual_plans(duration_seconds=50.5838666667) == first


def test_dense_candidate_budget_never_changes_final_fusion_cap():
    for cap in (40,60,80,120,200):
        plans=build_counterfactual_plans(duration_seconds=120.0,dense_candidate_cap=cap,final_fusion_cap=40)
        assert all(p.final_fusion_cap == 40 for p in plans)

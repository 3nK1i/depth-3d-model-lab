from core.reconstruction.offline_dense_i4_decoupled_dense_ab import DenseABPlanSummary, choose_real_dense_plan


def test_large_candidate_path_remains_bounded_for_final_fusion() -> None:
    summary = DenseABPlanSummary(
        plan_name="uniform", planned_candidates=120, alignment_accepted=120,
        pair_edges=350, pair_pass=300, pair_degraded=50, pair_inconclusive=0,
        selected_qualified_views=120, full_span=True, final_fusion_cap=40,
    )
    assert summary.dense_gate_pass
    assert summary.as_dict()["fusion_eligible_view_count"] == 40


def test_many_inconclusive_pairs_keep_exact_accounting() -> None:
    summary = DenseABPlanSummary(
        plan_name="neighbors", planned_candidates=118, alignment_accepted=100,
        pair_edges=250, pair_pass=50, pair_degraded=50, pair_inconclusive=150,
        selected_qualified_views=20, full_span=True,
    )
    assert summary.pair_pass_fraction == 0.5
    assert summary.dense_gate_pass


def test_winner_selection_is_deterministic_under_tie() -> None:
    a = DenseABPlanSummary(
        plan_name="a", planned_candidates=100, alignment_accepted=90,
        pair_edges=100, pair_pass=40, pair_degraded=60, pair_inconclusive=0,
        selected_qualified_views=20, full_span=True,
    )
    b = DenseABPlanSummary(
        plan_name="b", planned_candidates=110, alignment_accepted=99,
        pair_edges=100, pair_pass=40, pair_degraded=60, pair_inconclusive=0,
        selected_qualified_views=20, full_span=True,
    )
    decision = choose_real_dense_plan([a, b], baseline_selected_views=2, baseline_pair_pass_fraction=0.02)
    assert decision["recommended_plan"] == "a"  # fewer candidates wins exact tie


def test_saturated_baseline_with_large_selected_gain_remains_fail_closed_on_pair_regression() -> None:
    saturated = DenseABPlanSummary(
        plan_name="saturated", planned_candidates=120, alignment_accepted=120,
        pair_edges=400, pair_pass=400, pair_degraded=0, pair_inconclusive=0,
        selected_qualified_views=120, full_span=True,
    )
    decision = choose_real_dense_plan([saturated], baseline_selected_views=40, baseline_pair_pass_fraction=1.0)
    assert decision["real_dense_ab_validation_pass"] is True
    assert decision["selected_view_gain_over_global_only"] == 80
    assert decision["pair_quality_regressed"] is False


def test_many_dense_records_do_not_retain_bulk_arrays_in_memory(tmp_path) -> None:
    import numpy as np

    from scripts.run_i4_1_6b_decoupled_dense_ab import _compact_dense_record

    records = []
    for index in range(176):
        record = {
            "workspace_order": index,
            "keyframe_id": f"kf-{index}",
            "cloud_npz": str(tmp_path / f"cloud-{index}.npz"),
            "points": np.ones((64, 3), dtype=np.float64),
            "colors": np.ones((64, 3), dtype=np.uint8),
            "aligned_depth": np.ones((16, 16), dtype=np.float64),
        }
        records.append(_compact_dense_record(record))

    assert len(records) == 176
    assert all(item is not None for item in records)
    assert all("points" not in item for item in records if item is not None)
    assert all("colors" not in item for item in records if item is not None)
    assert all("aligned_depth" not in item for item in records if item is not None)

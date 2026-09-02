from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.reconstruction.offline_dense_i4_decoupled_dense_ab import (
    DenseABPlanSummary,
    align_localized_candidate_depth,
    camera_estimate_from_localization,
    choose_real_dense_plan,
)
from tests.unit.test_i4_1_6_decoupled_sampling_integration import _synthetic_sfm


def _localization(sfm, *, count: int = 24):
    landmarks = sfm.landmarks[:count]
    return {
        "source_sequence_index": 15,
        "timestamp_ns": 500_000_000,
        "accepted": True,
        "keyframe_id": "i4-1-6-localized-00000015",
        "T_reconstruction_from_camera": np.eye(4).tolist(),
        "matched_landmark_ids": [item.landmark_id for item in landmarks],
        "matched_feature_xy": [[100.0 + i, 120.0] for i in range(len(landmarks))],
    }


def test_camera_estimate_keeps_authoritative_localized_pose(tmp_path: Path) -> None:
    sfm = _synthetic_sfm(tmp_path)
    loc = _localization(sfm)
    estimate = camera_estimate_from_localization(loc, camera_model=sfm.camera_models[0])
    assert estimate.backend_image_id is None
    assert estimate.camera_id == sfm.camera_models[0].camera_id
    assert np.allclose(estimate.T_reconstruction_from_camera, np.eye(4))


def test_localized_depth_alignment_accepts_real_sparse_scale_fit(tmp_path: Path) -> None:
    sfm = _synthetic_sfm(tmp_path)
    depth = np.full((480, 640), 2.0, dtype=np.float32)
    evidence = align_localized_candidate_depth(
        sfm,
        localization=_localization(sfm),
        camera_model=sfm.camera_models[0],
        dense_depth_sfm_grid=depth,
        source_image_width=640,
        source_image_height=480,
        depth_provenance="unit-test-da3",
        minimum_correspondences=8,
        maximum_relative_residual=0.35,
    )
    assert evidence.is_accepted
    assert evidence.alignment is not None
    assert 1.9 < evidence.alignment.scale_to_reconstruction_units < 2.5
    assert evidence.inlier_correspondences >= 8
    assert evidence.sfm_scale_status.value == "arbitrary"


def test_localized_depth_alignment_scales_source_pixel_coordinates(tmp_path: Path) -> None:
    sfm = _synthetic_sfm(tmp_path)
    loc = _localization(sfm)
    loc["matched_feature_xy"] = [[200.0 + 2.0 * i, 240.0] for i in range(len(loc["matched_landmark_ids"]))]
    depth = np.full((480, 640), 2.0, dtype=np.float32)
    evidence = align_localized_candidate_depth(
        sfm,
        localization=loc,
        camera_model=sfm.camera_models[0],
        dense_depth_sfm_grid=depth,
        source_image_width=1280,
        source_image_height=960,
        depth_provenance="scaled-pixel-test",
    )
    assert evidence.is_accepted


def test_localized_depth_alignment_rejects_insufficient_correspondences(tmp_path: Path) -> None:
    sfm = _synthetic_sfm(tmp_path)
    evidence = align_localized_candidate_depth(
        sfm,
        localization=_localization(sfm, count=5),
        camera_model=sfm.camera_models[0],
        dense_depth_sfm_grid=np.full((480, 640), 2.0, dtype=np.float32),
        source_image_width=640,
        source_image_height=480,
        depth_provenance="few",
        minimum_correspondences=8,
    )
    assert not evidence.is_accepted
    assert evidence.rejection_code.value == "insufficient_correspondences"


def test_localized_depth_alignment_refuses_wrong_grid(tmp_path: Path) -> None:
    sfm = _synthetic_sfm(tmp_path)
    with pytest.raises(Exception):
        align_localized_candidate_depth(
            sfm,
            localization=_localization(sfm),
            camera_model=sfm.camera_models[0],
            dense_depth_sfm_grid=np.ones((240, 320), dtype=np.float32),
            source_image_width=640,
            source_image_height=480,
            depth_provenance="bad-grid",
        )


def test_dense_ab_summary_requires_exact_pair_accounting() -> None:
    with pytest.raises(Exception):
        DenseABPlanSummary(
            plan_name="x", planned_candidates=100, alignment_accepted=90,
            pair_edges=10, pair_pass=4, pair_degraded=4, pair_inconclusive=1,
            selected_qualified_views=12, full_span=True,
        )


def test_dense_ab_gate_requires_alignment_full_span_and_ten_views() -> None:
    passing = DenseABPlanSummary(
        plan_name="uniform", planned_candidates=100, alignment_accepted=90,
        pair_edges=80, pair_pass=30, pair_degraded=50, pair_inconclusive=0,
        selected_qualified_views=12, full_span=True,
    )
    assert passing.dense_gate_pass
    assert passing.as_dict()["fusion_eligible_view_count"] == 12
    weak = DenseABPlanSummary(
        plan_name="weak", planned_candidates=100, alignment_accepted=79,
        pair_edges=80, pair_pass=30, pair_degraded=50, pair_inconclusive=0,
        selected_qualified_views=12, full_span=True,
    )
    assert not weak.dense_gate_pass


def test_dense_ab_more_than_fusion_cap_is_still_qualified_but_bounded() -> None:
    summary = DenseABPlanSummary(
        plan_name="long", planned_candidates=110, alignment_accepted=100,
        pair_edges=120, pair_pass=70, pair_degraded=50, pair_inconclusive=0,
        selected_qualified_views=55, full_span=True, final_fusion_cap=40,
    )
    assert summary.dense_gate_pass
    assert summary.as_dict()["fusion_eligible_view_count"] == 40


def test_real_dense_plan_prefers_real_gate_and_larger_path() -> None:
    a = DenseABPlanSummary(
        plan_name="uniform", planned_candidates=102, alignment_accepted=90,
        pair_edges=100, pair_pass=30, pair_degraded=70, pair_inconclusive=0,
        selected_qualified_views=14, full_span=True,
    )
    b = DenseABPlanSummary(
        plan_name="neighbors", planned_candidates=118, alignment_accepted=110,
        pair_edges=110, pair_pass=40, pair_degraded=70, pair_inconclusive=0,
        selected_qualified_views=18, full_span=True,
    )
    decision = choose_real_dense_plan([a, b], baseline_selected_views=2, baseline_pair_pass_fraction=0.02)
    assert decision["recommended_plan"] == "neighbors"
    assert decision["real_dense_ab_validation_pass"] is True
    assert decision["i4_1_6c_production_binding_authorized"] is True
    assert decision["production_sampling_integration_authorized"] is False


def test_real_dense_plan_does_not_authorize_without_real_gate() -> None:
    weak = DenseABPlanSummary(
        plan_name="uniform", planned_candidates=102, alignment_accepted=90,
        pair_edges=100, pair_pass=8, pair_degraded=92, pair_inconclusive=0,
        selected_qualified_views=7, full_span=True,
    )
    decision = choose_real_dense_plan([weak], baseline_selected_views=2, baseline_pair_pass_fraction=0.02)
    assert decision["real_dense_ab_validation_pass"] is False
    assert decision["i4_1_6c_production_binding_authorized"] is False


def test_real_dense_plan_accepts_saturated_baseline_when_selected_views_increase_without_pair_regression() -> None:
    saturated = DenseABPlanSummary(
        plan_name="neighbors", planned_candidates=118, alignment_accepted=118,
        pair_edges=348, pair_pass=348, pair_degraded=0, pair_inconclusive=0,
        selected_qualified_views=118, full_span=True,
    )
    decision = choose_real_dense_plan(
        [saturated], baseline_selected_views=40, baseline_pair_pass_fraction=1.0
    )
    assert decision["real_dense_ab_validation_pass"] is True
    assert decision["baseline_pair_pass_fraction_saturated"] is True
    assert decision["pair_quality_regressed"] is False
    assert decision["selected_view_gain_over_global_only"] == 78
    assert decision["comparison_rule"] == "saturated_baseline_no_pair_regression_plus_selected_gain"


def test_real_dense_plan_rejects_saturated_baseline_when_pair_quality_regresses() -> None:
    regressed = DenseABPlanSummary(
        plan_name="neighbors", planned_candidates=118, alignment_accepted=118,
        pair_edges=348, pair_pass=347, pair_degraded=1, pair_inconclusive=0,
        selected_qualified_views=118, full_span=True,
    )
    decision = choose_real_dense_plan(
        [regressed], baseline_selected_views=40, baseline_pair_pass_fraction=1.0
    )
    assert decision["real_dense_ab_validation_pass"] is False
    assert decision["pair_quality_regressed"] is True


def test_real_dense_plan_keeps_strict_pair_gain_for_unsaturated_baseline() -> None:
    equal_quality = DenseABPlanSummary(
        plan_name="neighbors", planned_candidates=118, alignment_accepted=118,
        pair_edges=100, pair_pass=80, pair_degraded=20, pair_inconclusive=0,
        selected_qualified_views=50, full_span=True,
    )
    decision = choose_real_dense_plan(
        [equal_quality], baseline_selected_views=40, baseline_pair_pass_fraction=0.8
    )
    assert decision["real_dense_ab_validation_pass"] is False
    assert decision["baseline_pair_pass_fraction_saturated"] is False
    assert decision["pair_quality_requirement_pass"] is False


def test_dense_ab_compacts_persisted_records_without_losing_metadata(tmp_path: Path) -> None:
    from scripts.run_i4_1_6b_decoupled_dense_ab import _compact_dense_record

    record = {
        "workspace_order": 7,
        "keyframe_id": "kf-7",
        "cloud_npz": str(tmp_path / "cloud.npz"),
        "points": np.ones((10, 3), dtype=np.float64),
        "colors": np.ones((10, 3), dtype=np.uint8),
        "aligned_depth": np.ones((4, 5), dtype=np.float64),
        "camera": object(),
    }
    compact = _compact_dense_record(record)
    assert compact is not None
    assert "points" not in compact
    assert "colors" not in compact
    assert "aligned_depth" not in compact
    assert compact["workspace_order"] == 7
    assert compact["keyframe_id"] == "kf-7"
    assert compact["camera"] is record["camera"]
    assert "points" in record


def test_dense_ab_hydrates_only_persisted_bulk_arrays(tmp_path: Path) -> None:
    from scripts.run_i4_1_6b_decoupled_dense_ab import _hydrate_dense_record

    cloud = tmp_path / "canonical_cloud.npz"
    points = np.arange(30, dtype=np.float64).reshape(10, 3)
    colors = np.arange(30, dtype=np.uint8).reshape(10, 3)
    depth = np.arange(20, dtype=np.float64).reshape(4, 5)
    np.savez_compressed(
        cloud,
        points_reconstruction=points,
        colors_rgb=colors,
        aligned_depth_reconstruction_units=depth,
    )
    compact = {
        "workspace_order": 3,
        "keyframe_id": "kf-3",
        "cloud_npz": str(cloud),
        "camera": object(),
    }
    hydrated = _hydrate_dense_record(compact)
    assert np.array_equal(hydrated["points"], points)
    assert np.array_equal(hydrated["colors"], colors)
    assert np.array_equal(hydrated["aligned_depth"], depth)
    assert hydrated["camera"] is compact["camera"]
    assert "points" not in compact

from __future__ import annotations

from core.reconstruction.adaptive_view_selection import build_adaptive_view_selection_plan
from core.reconstruction.dense_provider_architecture import DenseProviderImplementation
from tests.i4_3f1_helpers import inflate_resolution, make_f1_sfm, resources_high, resources_mid


def test_f1_4k_resource_pressure_is_provider_specific_and_explainable(tmp_path) -> None:
    sfm = inflate_resolution(make_f1_sfm(tmp_path, view_count=120))
    resources = resources_mid()

    dav2 = build_adaptive_view_selection_plan(
        sfm,
        implementation=DenseProviderImplementation.DEPTH_ANYTHING_V2,
        resources=resources,
    )
    mvs = build_adaptive_view_selection_plan(
        sfm,
        implementation=DenseProviderImplementation.COLMAP_MVS,
        resources=resources,
    )

    assert dav2.reduction_applied is True
    assert mvs.reduction_applied is True
    assert dav2.selected_view_count > mvs.selected_view_count
    assert dav2.resource_limited_target_view_count == dav2.selected_view_count
    assert mvs.resource_limited_target_view_count == mvs.selected_view_count
    assert "resource_pressure_adaptive_reduction" in dav2.reason_codes
    assert "resource_pressure_adaptive_reduction" in mvs.reason_codes
    assert dav2.permanent_global_view_cap is None
    assert mvs.permanent_global_view_cap is None


def test_f1_plan_is_complete_engineer_visible_evidence(tmp_path) -> None:
    sfm = inflate_resolution(make_f1_sfm(tmp_path, view_count=40), width=1920, height=1080)
    plan = build_adaptive_view_selection_plan(
        sfm,
        implementation=DenseProviderImplementation.MARIGOLD,
        resources=resources_high(),
    )
    payload = plan.as_dict()

    assert payload["source_view_count"] == 40
    assert len(payload["evidence"]) == 40
    assert len(payload["decisions"]) == 40
    assert all(item["reason_codes"] for item in payload["decisions"])
    assert payload["requirements"]["implementation"] == "marigold"
    assert payload["resources"]["planning_capacity_formula"]
    assert payload["authority"]["camera_pose_authority_owner"] == "pycolmap_sfm"
    assert payload["authority"]["provider_selected"] is False
    assert payload["authority"]["metric_scale_claimed"] is False


def test_f1_temporal_selection_keeps_span_and_order_under_pressure(tmp_path) -> None:
    sfm = inflate_resolution(make_f1_sfm(tmp_path, view_count=96))
    plan = build_adaptive_view_selection_plan(
        sfm,
        implementation=DenseProviderImplementation.VIDEO_DEPTH_ANYTHING,
        resources=resources_mid(),
    )
    assert plan.reduction_applied is True
    assert plan.selected_keyframe_ids[0] == "kf-0"
    assert plan.selected_keyframe_ids[-1] == "kf-95"
    indices = [int(value.split("-")[-1]) for value in plan.selected_keyframe_ids]
    assert indices == sorted(indices)
    assert "temporal_provider_source_order_preserved" in plan.reason_codes

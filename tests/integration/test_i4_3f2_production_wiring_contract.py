from __future__ import annotations

from pathlib import Path

from core.reconstruction.adaptive_production_planning import (
    build_adaptive_pre_sfm_frame_plan,
    prepare_adaptive_provider_execution,
)
from core.reconstruction.dense_provider_architecture import DenseProviderImplementation
from tests.i4_3f1_helpers import inflate_resolution, make_f1_sfm, resources_high, resources_mid
from tests.i4_3f2_helpers import available_descriptor


def test_large_source_reduces_per_run_but_small_source_does_not() -> None:
    large = build_adaptive_pre_sfm_frame_plan(
        source_kind="recorded_video",
        estimated_total_frames=2400,
        expected_frame_rate_hz=30.0,
        resources=resources_mid(),
    )
    small = build_adaptive_pre_sfm_frame_plan(
        source_kind="image_directory",
        estimated_total_frames=47,
        expected_frame_rate_hz=10.0,
        resources=resources_mid(),
    )
    assert large.target_sampled_frames == 64
    assert large.adaptive_reduction_expected is True
    assert small.target_sampled_frames == 47
    assert small.adaptive_reduction_expected is False
    assert large.permanent_global_view_cap is None is small.permanent_global_view_cap


def test_provider_specific_post_sfm_plans_differ_without_auto_selection(tmp_path: Path) -> None:
    sfm = inflate_resolution(make_f1_sfm(tmp_path, view_count=48))
    mvs = prepare_adaptive_provider_execution(
        sfm,
        provider=available_descriptor(DenseProviderImplementation.COLMAP_MVS, "mvs-test"),
        reconstruction_frame_id="offline-sfm-reconstruction",
        resources=resources_mid(),
    )
    dav2 = prepare_adaptive_provider_execution(
        sfm,
        provider=available_descriptor(DenseProviderImplementation.DEPTH_ANYTHING_V2, "dav2-test"),
        reconstruction_frame_id="offline-sfm-reconstruction",
        resources=resources_mid(),
    )
    assert mvs.selection_decision.automatic is False
    assert dav2.selection_decision.automatic is False
    assert mvs.plan.requirements.implementation is DenseProviderImplementation.COLMAP_MVS
    assert dav2.plan.requirements.implementation is DenseProviderImplementation.DEPTH_ANYTHING_V2
    assert mvs.plan.permanent_global_view_cap is None
    assert dav2.plan.permanent_global_view_cap is None
    assert mvs.plan.estimated_source_work_units > dav2.plan.estimated_source_work_units


def test_temporal_provider_projection_keeps_selected_source_order(tmp_path: Path) -> None:
    sfm = inflate_resolution(make_f1_sfm(tmp_path, view_count=40))
    prep = prepare_adaptive_provider_execution(
        sfm,
        provider=available_descriptor(DenseProviderImplementation.VIDEO_DEPTH_ANYTHING, "vda-test"),
        reconstruction_frame_id="offline-sfm-reconstruction",
        resources=resources_mid(),
    )
    source_order = {key: index for index, key in enumerate(prep.plan.source_order_keyframe_ids)}
    assert list(prep.plan.selected_keyframe_ids) == sorted(prep.plan.selected_keyframe_ids, key=source_order.__getitem__)
    assert tuple(item.keyframe_id for item in prep.provider_sfm_result.workspace.images) == prep.plan.selected_keyframe_ids
    assert "temporal_provider_source_order_preserved" in prep.plan.reason_codes

from __future__ import annotations

from core.reconstruction.adaptive_production_planning import (
    build_adaptive_pre_sfm_frame_plan,
    prepare_adaptive_provider_execution,
)
from core.reconstruction.adaptive_view_selection import AdaptiveViewOverrideMode, AdaptiveViewSelectionOverride
from core.reconstruction.dense_provider_architecture import DenseProviderImplementation
from tests.i4_3f1_helpers import inflate_resolution, make_f1_sfm
from tests.i4_3f2_helpers import available_descriptor
from tests.i4_3f3_helpers import resources_2060, resources_5090, resources_constrained


def test_pre_sfm_resource_scaling_has_no_fixed_cap() -> None:
    profiles = (resources_constrained(), resources_2060(), resources_5090())
    for total in (20, 68, 1000, 5000):
        plans = [build_adaptive_pre_sfm_frame_plan(
            source_kind="recorded_video", estimated_total_frames=total,
            expected_frame_rate_hz=30.0, resources=resource,
        ) for resource in profiles]
        targets = [plan.target_sampled_frames for plan in plans]
        assert all(target is not None and 3 <= target <= total for target in targets)
        assert targets == sorted(targets)
        assert all(plan.permanent_global_view_cap is None for plan in plans)
        if total <= targets[0]:
            assert all(target == total for target in targets)


def test_all_ten_providers_validate_without_auto_or_sfm_mutation(tmp_path) -> None:
    sfm = inflate_resolution(make_f1_sfm(tmp_path, view_count=48))
    original = sfm.as_dict()
    for implementation in DenseProviderImplementation:
        mid = prepare_adaptive_provider_execution(
            sfm,
            provider=available_descriptor(implementation, f"{implementation.value}-f3"),
            reconstruction_frame_id="offline-sfm-reconstruction",
            resources=resources_2060(),
        )
        high = prepare_adaptive_provider_execution(
            sfm,
            provider=available_descriptor(implementation, f"{implementation.value}-f3-high"),
            reconstruction_frame_id="offline-sfm-reconstruction",
            resources=resources_5090(),
        )
        assert mid.plan.permanent_global_view_cap is None
        assert high.plan.permanent_global_view_cap is None
        assert mid.selection_decision.automatic is False
        assert high.selection_decision.automatic is False
        assert mid.plan.selected_view_count >= mid.plan.requirements.minimum_views
        assert high.plan.selected_view_count >= mid.plan.selected_view_count
        if mid.plan.requirements.stateful_sequence:
            order = {key: i for i, key in enumerate(mid.plan.source_order_keyframe_ids)}
            assert list(mid.plan.selected_keyframe_ids) == sorted(mid.plan.selected_keyframe_ids, key=order.__getitem__)
    assert sfm.as_dict() == original


def test_engineer_overrides_remain_explicit_and_visible(tmp_path) -> None:
    pre = build_adaptive_pre_sfm_frame_plan(
        source_kind="recorded_video", estimated_total_frames=1000,
        expected_frame_rate_hz=30.0, resources=resources_5090(),
        engineer_max_keyframes_override=17,
    )
    assert pre.target_sampled_frames == 17
    assert pre.engineer_override_applied is True
    assert "engineer_max_keyframes_override_applied" in pre.reason_codes
    sfm = inflate_resolution(make_f1_sfm(tmp_path, view_count=24))
    prep = prepare_adaptive_provider_execution(
        sfm,
        provider=available_descriptor(DenseProviderImplementation.MARIGOLD, "marigold-f3"),
        reconstruction_frame_id="offline-sfm-reconstruction",
        resources=resources_constrained(),
        override=AdaptiveViewSelectionOverride(AdaptiveViewOverrideMode.ALL_REGISTERED, reason="engineer_f3_quality_check"),
    )
    assert prep.plan.selected_view_count == 24
    assert prep.plan.engineer_override_applied is True
    assert prep.plan.permanent_global_view_cap is None


def test_planning_is_deterministic_for_same_inputs(tmp_path) -> None:
    sfm = inflate_resolution(make_f1_sfm(tmp_path, view_count=36))
    provider = available_descriptor(DenseProviderImplementation.COLMAP_MVS, "mvs-f3")
    first = prepare_adaptive_provider_execution(
        sfm, provider=provider, reconstruction_frame_id="offline-sfm-reconstruction", resources=resources_2060()
    )
    second = prepare_adaptive_provider_execution(
        sfm, provider=provider, reconstruction_frame_id="offline-sfm-reconstruction", resources=resources_2060()
    )
    assert first.plan.selected_keyframe_ids == second.plan.selected_keyframe_ids
    assert first.plan.decisions == second.plan.decisions

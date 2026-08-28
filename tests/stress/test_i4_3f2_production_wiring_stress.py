from __future__ import annotations

from pathlib import Path

from core.reconstruction.adaptive_production_planning import build_adaptive_pre_sfm_frame_plan, prepare_adaptive_provider_execution
from core.reconstruction.dense_provider_architecture import DenseProviderImplementation
from tests.i4_3f1_helpers import inflate_resolution, make_f1_sfm, resources_high, resources_mid
from tests.i4_3f2_helpers import available_descriptor


def test_pre_sfm_targets_are_monotonic_with_source_and_resources() -> None:
    for total in (20, 68, 200, 1000, 5000):
        mid = build_adaptive_pre_sfm_frame_plan(
            source_kind="recorded_video",
            estimated_total_frames=total,
            expected_frame_rate_hz=30.0,
            resources=resources_mid(),
        )
        high = build_adaptive_pre_sfm_frame_plan(
            source_kind="recorded_video",
            estimated_total_frames=total,
            expected_frame_rate_hz=30.0,
            resources=resources_high(),
        )
        assert 3 <= mid.target_sampled_frames <= total
        assert mid.target_sampled_frames <= high.target_sampled_frames <= total
        assert mid.permanent_global_view_cap is None
        assert high.permanent_global_view_cap is None


def test_all_provider_implementations_prepare_repeatedly_without_authority_drift(tmp_path: Path) -> None:
    sfm = inflate_resolution(make_f1_sfm(tmp_path, view_count=24))
    original = sfm.as_dict()
    for cycle in range(3):
        for implementation in DenseProviderImplementation:
            prep = prepare_adaptive_provider_execution(
                sfm,
                provider=available_descriptor(implementation, f"{implementation.value}-{cycle}"),
                reconstruction_frame_id="offline-sfm-reconstruction",
                resources=resources_mid(),
            )
            assert prep.plan.permanent_global_view_cap is None
            assert prep.selection_decision.automatic is False
            assert prep.plan.selected_view_count >= prep.plan.requirements.minimum_views
            assert prep.provider_sfm_result.statistics.registered_keyframes == prep.plan.selected_view_count
        assert sfm.as_dict() == original

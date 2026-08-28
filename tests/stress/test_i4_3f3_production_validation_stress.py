from __future__ import annotations

from core.reconstruction.adaptive_production_planning import build_adaptive_pre_sfm_frame_plan, prepare_adaptive_provider_execution
from core.reconstruction.dense_provider_architecture import DenseProviderImplementation
from tests.i4_3f1_helpers import inflate_resolution, make_f1_sfm
from tests.i4_3f2_helpers import available_descriptor
from tests.i4_3f3_helpers import resources_2060, resources_5090, resources_constrained


def test_resource_monotonicity_over_many_source_sizes() -> None:
    for total in range(3, 3001, 113):
        low = build_adaptive_pre_sfm_frame_plan(source_kind="recorded_video", estimated_total_frames=total, expected_frame_rate_hz=30.0, resources=resources_constrained())
        mid = build_adaptive_pre_sfm_frame_plan(source_kind="recorded_video", estimated_total_frames=total, expected_frame_rate_hz=30.0, resources=resources_2060())
        high = build_adaptive_pre_sfm_frame_plan(source_kind="recorded_video", estimated_total_frames=total, expected_frame_rate_hz=30.0, resources=resources_5090())
        assert 3 <= low.target_sampled_frames <= mid.target_sampled_frames <= high.target_sampled_frames <= total
        assert low.permanent_global_view_cap is None is mid.permanent_global_view_cap is high.permanent_global_view_cap


def test_repeated_all_provider_planning_never_mutates_sfm(tmp_path) -> None:
    sfm = inflate_resolution(make_f1_sfm(tmp_path, view_count=30))
    original = sfm.as_dict()
    for cycle in range(4):
        for implementation in DenseProviderImplementation:
            prep = prepare_adaptive_provider_execution(
                sfm,
                provider=available_descriptor(implementation, f"{implementation.value}-f3-{cycle}"),
                reconstruction_frame_id="offline-sfm-reconstruction",
                resources=resources_2060(),
            )
            assert prep.plan.permanent_global_view_cap is None
            assert prep.selection_decision.automatic is False
        assert sfm.as_dict() == original

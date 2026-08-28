from __future__ import annotations

from core.reconstruction.adaptive_view_selection import build_adaptive_view_selection_plan
from core.reconstruction.dense_provider_architecture import DenseProviderImplementation
from tests.i4_3f1_helpers import inflate_resolution, make_f1_sfm, resources_mid


def test_f1_large_registered_view_set_has_no_fixed_cap_and_all_providers_plan(tmp_path) -> None:
    sfm = inflate_resolution(make_f1_sfm(tmp_path, view_count=180), width=2560, height=1440)
    resources = resources_mid()
    for implementation in DenseProviderImplementation:
        plan = build_adaptive_view_selection_plan(
            sfm,
            implementation=implementation,
            resources=resources,
        )
        assert plan.source_view_count == 180
        assert plan.permanent_global_view_cap is None
        assert plan.selected_view_count >= plan.requirements.minimum_views
        assert plan.selected_view_count <= 180
        assert len(plan.decisions) == 180
        assert len(plan.evidence) == 180
        assert plan.estimated_selected_work_units <= plan.estimated_source_work_units
        assert plan.authority_metadata


def test_f1_repeated_large_plans_are_deterministic(tmp_path) -> None:
    sfm = inflate_resolution(make_f1_sfm(tmp_path, view_count=128), width=3840, height=2160)
    resources = resources_mid()
    for implementation in (
        DenseProviderImplementation.COLMAP_MVS,
        DenseProviderImplementation.ROLLING_DEPTH,
        DenseProviderImplementation.MARIGOLD,
    ):
        first = build_adaptive_view_selection_plan(
            sfm,
            implementation=implementation,
            resources=resources,
        )
        second = build_adaptive_view_selection_plan(
            sfm,
            implementation=implementation,
            resources=resources,
        )
        assert first.selected_keyframe_ids == second.selected_keyframe_ids
        assert first.reason_codes == second.reason_codes

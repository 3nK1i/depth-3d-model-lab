from __future__ import annotations

from dataclasses import replace

import pytest

from core.reconstruction.adaptive_view_selection import (
    AdaptiveSelectionResourceSnapshot,
    AdaptiveViewOverrideMode,
    AdaptiveViewSelectionOverride,
    AdaptiveViewSelectionPlan,
    AdaptiveViewSelectionPolicy,
    build_adaptive_view_selection_plan,
    i4_3f1_architecture_snapshot,
    provider_view_selection_requirements,
)
from core.reconstruction.dense_provider_architecture import (
    DenseProviderImplementation,
    dense_provider_implementation_contracts,
)
from core.reconstruction.errors import ReconstructionValidationError
from tests.i4_3f1_helpers import hardware_2060, make_f1_sfm, resources_high, resources_low


def test_f1_architecture_snapshot_covers_every_provider_and_no_cap() -> None:
    snapshot = i4_3f1_architecture_snapshot()
    assert snapshot["pass"] is True
    assert snapshot["selection"]["permanent_global_view_cap"] is None
    assert snapshot["selection"]["engineer_override_supported"] is True
    assert snapshot["authority"]["camera_pose_authority_owner"] == "pycolmap_sfm"
    assert snapshot["authority"]["auto_policy_implemented"] is False
    assert snapshot["scope"]["production_wiring_performed"] is False
    assert len(snapshot["provider_requirements"]) == len(dense_provider_implementation_contracts())


@pytest.mark.parametrize("implementation", list(DenseProviderImplementation))
def test_every_provider_has_explicit_requirements(implementation) -> None:
    requirements = provider_view_selection_requirements(implementation)
    assert requirements.implementation is implementation
    assert requirements.minimum_views >= 2
    assert requirements.relative_cost_per_megapixel > 0.0
    assert abs(sum(requirements.normalized_priorities.values()) - 1.0) < 1e-9


def test_high_resources_keep_all_registered_views(tmp_path) -> None:
    sfm = make_f1_sfm(tmp_path, view_count=12)
    plan = build_adaptive_view_selection_plan(
        sfm,
        implementation=DenseProviderImplementation.METRIC3D_V2,
        resources=resources_high(),
    )
    assert plan.selected_view_count == 12
    assert plan.reduction_applied is False
    assert plan.adaptive_selection_evaluated is True
    assert plan.permanent_global_view_cap is None
    assert plan.reason_codes == ("all_registered_views_fit_resource_budget",)
    assert plan.selected_keyframe_ids == plan.source_order_keyframe_ids


def test_resource_pressure_reduces_dynamically_and_preserves_endpoints(tmp_path) -> None:
    sfm = make_f1_sfm(tmp_path, view_count=12)
    plan = build_adaptive_view_selection_plan(
        sfm,
        implementation=DenseProviderImplementation.COLMAP_MVS,
        resources=resources_low(),
    )
    assert plan.reduction_applied is True
    assert plan.selected_view_count == provider_view_selection_requirements(
        DenseProviderImplementation.COLMAP_MVS
    ).minimum_views
    assert plan.selected_keyframe_ids[0] == "kf-0"
    assert plan.selected_keyframe_ids[-1] == "kf-11"
    assert "resource_pressure_adaptive_reduction" in plan.reason_codes
    assert plan.resource_limited_target_view_count == plan.selected_view_count
    assert plan.estimated_selected_work_units < plan.estimated_source_work_units


def test_selection_is_deterministic(tmp_path) -> None:
    sfm = make_f1_sfm(tmp_path, view_count=14)
    first = build_adaptive_view_selection_plan(
        sfm,
        implementation=DenseProviderImplementation.DA3,
        resources=resources_low(),
    )
    second = build_adaptive_view_selection_plan(
        sfm,
        implementation=DenseProviderImplementation.DA3,
        resources=resources_low(),
    )
    assert first.selected_keyframe_ids == second.selected_keyframe_ids
    assert first.decisions == second.decisions


def test_temporal_provider_preserves_source_order(tmp_path) -> None:
    sfm = make_f1_sfm(tmp_path, view_count=16)
    plan = build_adaptive_view_selection_plan(
        sfm,
        implementation=DenseProviderImplementation.ROLLING_DEPTH,
        resources=resources_low(),
    )
    source_index = {key: index for index, key in enumerate(plan.source_order_keyframe_ids)}
    assert list(plan.selected_keyframe_ids) == sorted(
        plan.selected_keyframe_ids, key=source_index.__getitem__
    )
    assert "temporal_provider_source_order_preserved" in plan.reason_codes
    assert plan.requirements.stateful_sequence is True


def test_engineer_override_all_registered_is_visible(tmp_path) -> None:
    sfm = make_f1_sfm(tmp_path, view_count=10)
    plan = build_adaptive_view_selection_plan(
        sfm,
        implementation=DenseProviderImplementation.MARIGOLD,
        resources=resources_low(),
        override=AdaptiveViewSelectionOverride(
            AdaptiveViewOverrideMode.ALL_REGISTERED,
            reason="engineer_quality_diagnosis",
        ),
    )
    assert plan.selected_view_count == 10
    assert plan.engineer_override_applied is True
    assert plan.reduction_applied is False
    assert "engineer_override_applied" in plan.reason_codes
    assert all("engineer" in reason for decision in plan.decisions for reason in decision.reason_codes)


def test_engineer_explicit_override_preserves_source_order(tmp_path) -> None:
    sfm = make_f1_sfm(tmp_path, view_count=10)
    plan = build_adaptive_view_selection_plan(
        sfm,
        implementation=DenseProviderImplementation.DEPTH_ANYTHING_V2,
        resources=resources_high(),
        override=AdaptiveViewSelectionOverride(
            AdaptiveViewOverrideMode.EXPLICIT_KEYFRAMES,
            keyframe_ids=("kf-8", "kf-2", "kf-5"),
            reason="engineer_ab_subset",
        ),
    )
    assert plan.selected_keyframe_ids == ("kf-2", "kf-5", "kf-8")
    assert plan.engineer_override_applied is True
    assert plan.reduction_applied is True


def test_override_rejects_unknown_keyframe(tmp_path) -> None:
    sfm = make_f1_sfm(tmp_path, view_count=5)
    with pytest.raises(ReconstructionValidationError, match="unregistered keyframe"):
        build_adaptive_view_selection_plan(
            sfm,
            implementation=DenseProviderImplementation.METRIC3D_V2,
            resources=resources_high(),
            override=AdaptiveViewSelectionOverride(
                AdaptiveViewOverrideMode.EXPLICIT_KEYFRAMES,
                keyframe_ids=("kf-0", "missing"),
            ),
        )


def test_override_cannot_violate_provider_minimum(tmp_path) -> None:
    sfm = make_f1_sfm(tmp_path, view_count=5)
    with pytest.raises(ReconstructionValidationError, match="minimum-view"):
        build_adaptive_view_selection_plan(
            sfm,
            implementation=DenseProviderImplementation.COLMAP_MVS,
            resources=resources_high(),
            override=AdaptiveViewSelectionOverride(
                AdaptiveViewOverrideMode.EXPLICIT_KEYFRAMES,
                keyframe_ids=("kf-0", "kf-1"),
            ),
        )


def test_missing_sharpness_fails_soft_without_dropping_authority(tmp_path) -> None:
    sfm = make_f1_sfm(tmp_path, view_count=6)
    root = tmp_path / "sfm" / "images"
    (root / "frame_0002.png").unlink()
    plan = build_adaptive_view_selection_plan(
        sfm,
        implementation=DenseProviderImplementation.MOGE_2,
        resources=resources_high(),
    )
    evidence = next(item for item in plan.evidence if item.keyframe_id == "kf-2")
    assert evidence.sharpness_available is False
    assert evidence.sharpness_score == pytest.approx(0.5)
    assert dict(plan.authority_metadata)["camera_pose_authority_owner"] == "pycolmap_sfm"


def test_resource_snapshot_from_hardware_preserves_observed_resources() -> None:
    snapshot = AdaptiveSelectionResourceSnapshot.from_hardware(
        hardware_2060(), available_ram_bytes=12 * 1024**3
    )
    assert snapshot.gpu_name.endswith("RTX 2060")
    assert snapshot.free_vram_gib == pytest.approx(5.0)
    assert snapshot.available_ram_gib == pytest.approx(12.0)
    assert snapshot.planning_capacity_units() == pytest.approx(448.0)


def test_plan_validation_refuses_permanent_cap(tmp_path) -> None:
    sfm = make_f1_sfm(tmp_path, view_count=5)
    plan = build_adaptive_view_selection_plan(
        sfm,
        implementation=DenseProviderImplementation.DEPTH_PRO,
        resources=resources_high(),
    )
    with pytest.raises(ReconstructionValidationError, match="permanent global view cap"):
        replace(plan, permanent_global_view_cap=40)


def test_plan_validation_refuses_authority_promotion(tmp_path) -> None:
    sfm = make_f1_sfm(tmp_path, view_count=5)
    plan = build_adaptive_view_selection_plan(
        sfm,
        implementation=DenseProviderImplementation.DEPTH_PRO,
        resources=resources_high(),
    )
    with pytest.raises(ReconstructionValidationError, match="cannot promote"):
        replace(
            plan,
            authority_metadata={
                "camera_pose_authority_owner": "pycolmap_sfm",
                "metric_scale_claimed": True,
            },
        )


def test_policy_has_no_view_count_limit() -> None:
    payload = AdaptiveViewSelectionPolicy().as_dict()
    assert "maximum_views" not in payload
    assert payload["permanent_global_view_cap"] is None


def _a_snapshot() -> dict:
    return {
        "phase": "I4.3A",
        "status": "pass",
        "pass": True,
        "architecture": {
            "authoritative_camera_frame_owner": "pycolmap_sfm",
            "provider_family_contracts": [
                {
                    "family": "single_image_depth",
                    "adaptive_view_selection_required": True,
                    "permanent_global_view_cap": None,
                }
            ],
        },
    }


def _e6_snapshot() -> dict:
    return {
        "phase": "I4.3E6",
        "status": "pass",
        "pass": True,
        "qualification": {
            "i4_3e_provider_adapters_complete": True,
            "remaining_i4_3e_providers_pending": [],
        },
        "authority": {
            "provider_promoted": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
        },
    }


def test_f1_binding_snapshot_closes_contract_only() -> None:
    from core.reconstruction.adaptive_view_selection import build_i4_3f1_binding_snapshot

    snapshot = build_i4_3f1_binding_snapshot(
        i4_3a_snapshot=_a_snapshot(),
        i4_3e6_snapshot=_e6_snapshot(),
        resources=resources_high(),
    )
    assert snapshot["pass"] is True
    assert snapshot["decision"]["i4_3f1_adaptive_selection_contract_closed"] is True
    assert snapshot["decision"]["i4_3f2_production_wiring_authorized"] is True
    assert snapshot["decision"]["provider_selected"] is False
    assert snapshot["decision"]["auto_policy_implemented"] is False
    assert snapshot["decision"]["permanent_global_view_cap"] is None


def test_f1_binding_refuses_incomplete_e6() -> None:
    from core.reconstruction.adaptive_view_selection import build_i4_3f1_binding_snapshot

    bad = _e6_snapshot()
    bad["qualification"]["i4_3e_provider_adapters_complete"] = False
    with pytest.raises(ReconstructionValidationError, match="requires all I4.3E"):
        build_i4_3f1_binding_snapshot(
            i4_3a_snapshot=_a_snapshot(),
            i4_3e6_snapshot=bad,
            resources=resources_high(),
        )


def test_f1_binding_refuses_predecessor_cap() -> None:
    from core.reconstruction.adaptive_view_selection import build_i4_3f1_binding_snapshot

    bad = _a_snapshot()
    bad["architecture"]["provider_family_contracts"][0]["permanent_global_view_cap"] = 40
    with pytest.raises(ReconstructionValidationError, match="permanent global view caps"):
        build_i4_3f1_binding_snapshot(
            i4_3a_snapshot=bad,
            i4_3e6_snapshot=_e6_snapshot(),
            resources=resources_high(),
        )

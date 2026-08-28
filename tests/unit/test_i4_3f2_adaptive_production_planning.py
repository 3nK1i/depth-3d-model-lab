from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from core.reconstruction.adaptive_production_planning import (
    build_adaptive_pre_sfm_frame_plan,
    finalize_adaptive_provider_execution_result,
    prepare_adaptive_provider_execution,
    project_authoritative_sfm_for_provider,
)
from core.reconstruction.adaptive_view_selection import (
    AdaptiveSelectionResourceSnapshot,
    build_adaptive_view_selection_plan,
)
from core.reconstruction.dense_provider_architecture import (
    DenseProviderExecutionResult,
    DenseProviderImplementation,
    DenseProviderRunStatus,
)
from core.reconstruction.errors import ReconstructionValidationError
from core.contracts.validation import metadata_as_dict
from tests.i4_3f1_helpers import make_f1_sfm, resources_high, resources_low, resources_mid
from tests.i4_3f2_helpers import available_descriptor


def test_pre_sfm_plan_is_resource_derived_not_legacy_80() -> None:
    plan = build_adaptive_pre_sfm_frame_plan(
        source_kind="recorded_video",
        estimated_total_frames=1000,
        expected_frame_rate_hz=30.0,
        resources=resources_mid(),
    )
    assert plan.resource_target_frames == 64
    assert plan.target_sampled_frames == 64
    assert plan.target_sampled_frames != 80
    assert plan.permanent_global_view_cap is None
    assert plan.adaptive_reduction_expected is True
    assert "resource_pressure_adaptive_pre_sfm_reduction" in plan.reason_codes


def test_pre_sfm_high_resources_expand_and_small_source_keeps_all() -> None:
    low = build_adaptive_pre_sfm_frame_plan(
        source_kind="image_directory",
        estimated_total_frames=1000,
        expected_frame_rate_hz=10.0,
        resources=resources_mid(),
    )
    high = build_adaptive_pre_sfm_frame_plan(
        source_kind="image_directory",
        estimated_total_frames=1000,
        expected_frame_rate_hz=10.0,
        resources=resources_high(),
    )
    small = build_adaptive_pre_sfm_frame_plan(
        source_kind="image_directory",
        estimated_total_frames=68,
        expected_frame_rate_hz=10.0,
        resources=resources_high(),
    )
    assert high.target_sampled_frames > low.target_sampled_frames
    assert small.target_sampled_frames == 68
    assert small.adaptive_reduction_expected is False


def test_pre_sfm_explicit_max_is_engineer_override_not_global_default() -> None:
    plan = build_adaptive_pre_sfm_frame_plan(
        source_kind="image_directory",
        estimated_total_frames=120,
        expected_frame_rate_hz=10.0,
        resources=resources_high(),
        engineer_max_keyframes_override=42,
    )
    assert plan.target_sampled_frames == 42
    assert plan.engineer_override_applied is True
    assert "engineer_max_keyframes_override_applied" in plan.reason_codes
    assert plan.permanent_global_view_cap is None


def test_pre_sfm_unknown_resources_leave_source_uncapped() -> None:
    resources = AdaptiveSelectionResourceSnapshot(
        gpu_name="unknown",
        total_vram_bytes=None,
        free_vram_bytes=None,
        available_ram_bytes=None,
        cuda_ready=False,
    )
    plan = build_adaptive_pre_sfm_frame_plan(
        source_kind="recorded_video",
        estimated_total_frames=None,
        expected_frame_rate_hz=30.0,
        resources=resources,
    )
    assert plan.target_sampled_frames is None
    assert plan.resource_target_frames is None
    assert "resource_capacity_unknown_no_fixed_pre_sfm_cap" in plan.reason_codes


def test_sfm_projection_is_new_object_and_preserves_selected_camera_poses(tmp_path: Path) -> None:
    sfm = make_f1_sfm(tmp_path, view_count=12)
    before = deepcopy(sfm.as_dict())
    plan = build_adaptive_view_selection_plan(
        sfm,
        implementation=DenseProviderImplementation.MARIGOLD,
        resources=resources_low(),
    )
    assert plan.reduction_applied is True
    projected = project_authoritative_sfm_for_provider(sfm, plan)
    assert projected is not sfm
    assert sfm.as_dict() == before
    assert projected.statistics.registered_keyframes == plan.selected_view_count
    assert tuple(item.keyframe_id for item in projected.workspace.images) == plan.selected_keyframe_ids
    original_cameras = {item.keyframe_id: item for item in sfm.cameras}
    for camera in projected.cameras:
        assert np.array_equal(
            camera.T_reconstruction_from_camera,
            original_cameras[camera.keyframe_id].T_reconstruction_from_camera,
        )
    selected = set(plan.selected_keyframe_ids)
    assert all(
        len(landmark.observations) >= 2
        and all(obs.keyframe_id in selected for obs in landmark.observations)
        for landmark in projected.landmarks
    )
    projection_meta = metadata_as_dict(projected.metadata)["i4_3f2_provider_input_projection"]
    assert dict(projection_meta)["authoritative_sfm_modified"] is False


def test_no_reduction_reuses_original_sfm(tmp_path: Path) -> None:
    sfm = make_f1_sfm(tmp_path, view_count=8)
    plan = build_adaptive_view_selection_plan(
        sfm,
        implementation=DenseProviderImplementation.DEPTH_ANYTHING_V2,
        resources=resources_high(),
    )
    assert plan.reduction_applied is False
    assert project_authoritative_sfm_for_provider(sfm, plan) is sfm


def test_prepare_and_finalize_rebind_provider_local_result_to_original_source(tmp_path: Path) -> None:
    sfm = make_f1_sfm(tmp_path, view_count=12)
    provider = available_descriptor(DenseProviderImplementation.MARIGOLD, "marigold-test")
    preparation = prepare_adaptive_provider_execution(
        sfm,
        provider=provider,
        reconstruction_frame_id="offline-sfm-reconstruction",
        resources=resources_low(),
    )
    assert preparation.plan.reduction_applied is True
    assert preparation.selection_request.available_view_count == preparation.plan.selected_view_count
    assert preparation.selection_request.requested_provider_id == provider.provider_id
    assert preparation.selection_decision.automatic is False
    local = DenseProviderExecutionResult(
        provider_id=provider.provider_id,
        implementation=provider.implementation,
        status=DenseProviderRunStatus.COMPLETED,
        source_view_count=preparation.plan.selected_view_count,
        selected_view_count=preparation.plan.selected_view_count,
        adaptive_view_selection=False,
        view_selection_reason="all_projected_views",
        permanent_global_view_cap=None,
        canonical_geometry_ref="/tmp/canonical.ply",
        canonical_frame_id="offline-sfm-reconstruction",
    )
    final = finalize_adaptive_provider_execution_result(local, preparation)
    assert final.source_view_count == 12
    assert final.selected_view_count == preparation.plan.selected_view_count
    assert final.adaptive_view_selection is True
    assert final.permanent_global_view_cap is None
    assert "resource_pressure_adaptive_reduction" in final.view_selection_reason
    assert final.metric_scale_claimed is False


def test_prepare_refuses_unavailable_provider(tmp_path: Path) -> None:
    sfm = make_f1_sfm(tmp_path, view_count=6)
    provider = available_descriptor(DenseProviderImplementation.DEPTH_PRO, "depth-pro-test")
    unavailable = provider.__class__(
        provider_id=provider.provider_id,
        implementation=provider.implementation,
        execution=provider.execution,
        backend_version=provider.backend_version,
        adapter_version=provider.adapter_version,
        qualification_state=provider.qualification_state,
        available=False,
        unavailable_reason="runtime_missing",
    )
    with pytest.raises(ReconstructionValidationError, match="unavailable"):
        prepare_adaptive_provider_execution(
            sfm,
            provider=unavailable,
            reconstruction_frame_id="offline-sfm-reconstruction",
            resources=resources_high(),
        )


def test_f2_readiness_snapshot_authorizes_only_f3() -> None:
    from core.reconstruction.adaptive_production_planning import build_i4_3f2_production_readiness_snapshot

    f1 = {
        "phase": "I4.3F1",
        "status": "pass",
        "pass": True,
        "decision": {
            "i4_3f2_production_wiring_authorized": True,
            "permanent_global_view_cap": None,
        },
    }
    snapshot = build_i4_3f2_production_readiness_snapshot(i4_3f1_snapshot=f1, resources=resources_mid())
    assert snapshot["pass"] is True
    assert snapshot["decision"]["i4_3f2_production_wiring_closed"] is True
    assert snapshot["decision"]["i4_3f3_production_validation_authorized"] is True
    assert snapshot["decision"]["provider_selected"] is False
    assert snapshot["authority"]["permanent_global_view_cap"] is None


def test_f2_readiness_refuses_predecessor_global_cap() -> None:
    from core.reconstruction.adaptive_production_planning import build_i4_3f2_production_readiness_snapshot

    f1 = {
        "phase": "I4.3F1",
        "status": "pass",
        "pass": True,
        "decision": {
            "i4_3f2_production_wiring_authorized": True,
            "permanent_global_view_cap": 40,
        },
    }
    with pytest.raises(ReconstructionValidationError, match="permanent global view cap"):
        build_i4_3f2_production_readiness_snapshot(i4_3f1_snapshot=f1, resources=resources_mid())

from __future__ import annotations

import copy

import pytest

from core.reconstruction.dense_provider_architecture import (
    DenseProviderDescriptor,
    DenseProviderExecutionResult,
    DenseProviderFamily,
    DenseProviderFamilyContract,
    DenseProviderFrameBinding,
    DenseProviderImplementation,
    DenseProviderNativeOutput,
    DenseProviderProgressEvent,
    DenseProviderProgressStage,
    DenseProviderQualificationState,
    DenseProviderRunStatus,
    DenseProviderSelectionDecision,
    DenseProviderSelectionMode,
    DenseProviderSelectionRequest,
    validate_i4_3a_authorization,
)
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.orchestration import ReconstructionProviderExecution


def _closure() -> dict:
    return {
        "phase": "I4.2F",
        "status": "pass",
        "pass": True,
        "decision": {
            "i4_2_closed": True,
            "i4_2f_closed": True,
            "i4_3_provider_architecture_authorized": True,
            "dense_provider_promoted": False,
            "production_d12_policy_change_authorized": False,
        },
        "authority": {
            "authoritative_sparse_modified": False,
            "bundle_adjustment_rerun": False,
            "certified_metrology": False,
            "dense_provider_promoted": False,
            "measurement_authoritative": False,
            "metric_scale_claimed": False,
            "patch_match_rerun": False,
            "pose_refinement": False,
            "production_filter_policy_changed": False,
            "production_manifold_applied_to_authoritative_scene": False,
            "sfm_rerun": False,
            "stereo_fusion_rerun": False,
            "surface_mesh_generated": False,
        },
        "scientific_conclusion": {
            "d12_sparse_safe_on_temple_ring_within_tested_transfer": True,
            "global_universal_detail_safety_claimed": False,
            "temple_ring_detail_preservation_qualified": True,
            "temple_ring_driven_d12_policy_correction_required": False,
        },
    }


@pytest.mark.parametrize(
    "path,value",
    [
        (("decision", "i4_2_closed"), False),
        (("decision", "i4_2f_closed"), False),
        (("decision", "i4_3_provider_architecture_authorized"), False),
        (("decision", "production_d12_policy_change_authorized"), True),
        (("authority", "metric_scale_claimed"), True),
        (("authority", "production_filter_policy_changed"), True),
        (("scientific_conclusion", "temple_ring_detail_preservation_qualified"), False),
        (("scientific_conclusion", "temple_ring_driven_d12_policy_correction_required"), True),
    ],
)
def test_i43a_authorization_fails_closed_on_upstream_mutation(path: tuple[str, str], value: object) -> None:
    payload = copy.deepcopy(_closure())
    payload[path[0]][path[1]] = value
    with pytest.raises(ReconstructionValidationError):
        validate_i4_3a_authorization(payload)


@pytest.mark.parametrize(
    "family,native_output,frame_binding",
    [
        (
            DenseProviderFamily.MULTI_VIEW_AI_DEPTH,
            DenseProviderNativeOutput.DENSE_GEOMETRY,
            DenseProviderFrameBinding.AUTHORITATIVE_SFM_FRAME,
        ),
        (
            DenseProviderFamily.TEMPORAL_DEPTH,
            DenseProviderNativeOutput.DENSE_GEOMETRY,
            DenseProviderFrameBinding.AUTHORITATIVE_SFM_FRAME,
        ),
        (
            DenseProviderFamily.SINGLE_IMAGE_DEPTH,
            DenseProviderNativeOutput.DENSE_GEOMETRY,
            DenseProviderFrameBinding.AUTHORITATIVE_SFM_FRAME,
        ),
    ],
)
def test_i43a_learned_depth_families_cannot_masquerade_as_direct_mvs_geometry(
    family: DenseProviderFamily,
    native_output: DenseProviderNativeOutput,
    frame_binding: DenseProviderFrameBinding,
) -> None:
    with pytest.raises(ReconstructionValidationError, match="native-output/frame-binding"):
        DenseProviderFamilyContract(
            family=family,
            native_output=native_output,
            frame_binding=frame_binding,
        )


def test_i43a_descriptor_availability_reason_is_fail_closed() -> None:
    with pytest.raises(ReconstructionValidationError, match="unavailable_reason"):
        DenseProviderDescriptor(
            provider_id="colmap-mvs",
            implementation=DenseProviderImplementation.COLMAP_MVS,
            execution=ReconstructionProviderExecution.EXTERNAL_PROCESS,
            backend_version="4.1.1",
            adapter_version="i4.3b",
            qualification_state=DenseProviderQualificationState.CANDIDATE,
            available=False,
            unavailable_reason="",
        )


def test_i43a_selection_request_rejects_frame_without_authoritative_sfm() -> None:
    with pytest.raises(ReconstructionValidationError, match="cannot carry a frame ID"):
        DenseProviderSelectionRequest(
            mode=DenseProviderSelectionMode.AUTO,
            available_view_count=10,
            authoritative_sfm_available=False,
            authoritative_sfm_frame_id="stale-frame",
        )


def test_i43a_auto_decision_rejects_auto_as_concrete_provider() -> None:
    with pytest.raises(ReconstructionValidationError, match="never an executable"):
        DenseProviderSelectionDecision(
            request_mode=DenseProviderSelectionMode.AUTO,
            requested_provider_id="",
            selected_provider_id="auto",
            selected_implementation=DenseProviderImplementation.COLMAP_MVS,
            considered_provider_ids=("auto", "colmap-mvs"),
            automatic=True,
            engineer_override_applied=False,
            fallback_used=False,
            reason_codes=("bad",),
        )


def test_i43a_failed_result_cannot_smuggle_metric_claim() -> None:
    with pytest.raises(ReconstructionValidationError):
        DenseProviderExecutionResult(
            provider_id="colmap-mvs",
            implementation=DenseProviderImplementation.COLMAP_MVS,
            status=DenseProviderRunStatus.FAILED,
            source_view_count=47,
            selected_view_count=0,
            adaptive_view_selection=True,
            view_selection_reason="",
            permanent_global_view_cap=None,
            canonical_geometry_ref="",
            canonical_frame_id="",
            failure_reason="failed",
            canonical_geometry_authoritative=False,
            metric_scale_claimed=True,
        )


def test_i43a_progress_event_cannot_smuggle_metric_claim() -> None:
    with pytest.raises(ReconstructionValidationError, match="cannot establish"):
        DenseProviderProgressEvent(
            provider_id="depth-pro",
            implementation=DenseProviderImplementation.DEPTH_PRO,
            stage=DenseProviderProgressStage.PROVIDER_EXECUTION,
            stage_detail="depth_inference",
            progress_fraction=0.4,
            processed_views=4,
            total_views=10,
            metric_scale_claimed=True,
        )


def test_i43a_progress_event_rejects_negative_resource_usage() -> None:
    with pytest.raises(ReconstructionValidationError, match="gpu_vram_used_mib"):
        DenseProviderProgressEvent(
            provider_id="video-depth-anything",
            implementation=DenseProviderImplementation.VIDEO_DEPTH_ANYTHING,
            stage=DenseProviderProgressStage.PROVIDER_EXECUTION,
            stage_detail="temporal_depth",
            progress_fraction=0.4,
            processed_views=20,
            total_views=50,
            gpu_vram_used_mib=-1.0,
        )


def test_i43a_progress_event_rejects_preview_ref_without_availability_flag() -> None:
    with pytest.raises(ReconstructionValidationError, match="cannot carry a geometry reference"):
        DenseProviderProgressEvent(
            provider_id="colmap-mvs",
            implementation=DenseProviderImplementation.COLMAP_MVS,
            stage=DenseProviderProgressStage.FUSION_FILTERING,
            stage_detail="provisional_fusion",
            progress_fraction=0.8,
            processed_views=40,
            total_views=47,
            provisional_geometry_available=False,
            provisional_geometry_ref="memory://stale",
            preview_revision=5,
        )

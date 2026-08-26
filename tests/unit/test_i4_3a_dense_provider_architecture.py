from __future__ import annotations

from pathlib import Path

import pytest

from core.reconstruction.dense_provider_architecture import (
    I43A_AUTHORITATIVE_FRAME_OWNER,
    DenseProviderDescriptor,
    DenseProviderExecutionResult,
    DenseProviderFamily,
    DenseProviderFamilyContract,
    DenseProviderFrameBinding,
    DenseProviderImplementation,
    DenseProviderImplementationContract,
    DenseProviderNativeOutput,
    DenseProviderProgressEvent,
    DenseProviderProgressStage,
    DenseProviderQualificationState,
    DenseProviderRunStatus,
    DenseProviderSelectionDecision,
    DenseProviderSelectionMode,
    DenseProviderSelectionRequest,
    build_i4_3a_architecture_snapshot,
    dense_provider_family_contracts,
    dense_provider_implementation_contracts,
    read_i4_2f_closure,
    validate_dense_provider_catalog,
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


def _descriptor(
    provider_id: str,
    implementation: DenseProviderImplementation,
    *,
    available: bool = False,
    unavailable_reason: str = "adapter_not_bound",
) -> DenseProviderDescriptor:
    return DenseProviderDescriptor(
        provider_id=provider_id,
        implementation=implementation,
        execution=ReconstructionProviderExecution.EXTERNAL_PROCESS,
        backend_version="unbound",
        adapter_version="i4.3a-contract",
        qualification_state=DenseProviderQualificationState.CANDIDATE,
        available=available,
        unavailable_reason=unavailable_reason,
    )


def test_i43a_selection_modes_separate_policy_from_provider_identity() -> None:
    assert [item.value for item in DenseProviderSelectionMode] == ["auto", "explicit_provider"]


def test_i43a_provider_families_are_methodology_specific() -> None:
    assert [item.value for item in DenseProviderFamily] == [
        "mvs",
        "multi_view_ai_depth",
        "temporal_depth",
        "single_image_depth",
        "hybrid",
    ]


def test_i43a_provider_implementations_include_complete_locked_roadmap() -> None:
    assert [item.value for item in DenseProviderImplementation] == [
        "colmap_mvs",
        "da3",
        "rolling_depth",
        "video_depth_anything",
        "metric3d_v2",
        "unidepth_v2",
        "moge_2",
        "depth_pro",
        "depth_anything_v2",
        "marigold",
    ]


def test_i43a_family_contracts_lock_native_output_and_frame_binding() -> None:
    values = {item.family: item for item in dense_provider_family_contracts()}
    assert values[DenseProviderFamily.MVS].native_output is DenseProviderNativeOutput.DENSE_GEOMETRY
    assert values[DenseProviderFamily.MVS].frame_binding is DenseProviderFrameBinding.AUTHORITATIVE_SFM_FRAME
    for family in (
        DenseProviderFamily.MULTI_VIEW_AI_DEPTH,
        DenseProviderFamily.TEMPORAL_DEPTH,
        DenseProviderFamily.SINGLE_IMAGE_DEPTH,
    ):
        assert values[family].native_output is DenseProviderNativeOutput.DEPTH_EVIDENCE
        assert values[family].frame_binding is DenseProviderFrameBinding.ALIGN_TO_AUTHORITATIVE_SFM
    assert values[DenseProviderFamily.HYBRID].implementation_authorized is False


def test_i43a_family_contract_rejects_historical_permanent_view_cap() -> None:
    with pytest.raises(ReconstructionValidationError, match="permanent global dense-view cap"):
        DenseProviderFamilyContract(
            family=DenseProviderFamily.MVS,
            native_output=DenseProviderNativeOutput.DENSE_GEOMETRY,
            frame_binding=DenseProviderFrameBinding.AUTHORITATIVE_SFM_FRAME,
            permanent_global_view_cap=40,
        )


def test_i43a_hybrid_cannot_be_implemented_without_new_evidence_gate() -> None:
    with pytest.raises(ReconstructionValidationError, match="Hybrid implementation is not authorized"):
        DenseProviderFamilyContract(
            family=DenseProviderFamily.HYBRID,
            native_output=DenseProviderNativeOutput.COMPOSITE_GEOMETRY,
            frame_binding=DenseProviderFrameBinding.AUTHORITATIVE_SFM_FRAME,
            implementation_authorized=True,
        )


def test_i43a_family_contract_requires_progressive_preview_surface() -> None:
    with pytest.raises(ReconstructionValidationError, match="progressive status"):
        DenseProviderFamilyContract(
            family=DenseProviderFamily.MVS,
            native_output=DenseProviderNativeOutput.DENSE_GEOMETRY,
            frame_binding=DenseProviderFrameBinding.AUTHORITATIVE_SFM_FRAME,
            progressive_status_required=False,
        )


def test_i43a_family_contract_cannot_claim_metric_or_pose_authority() -> None:
    with pytest.raises(ReconstructionValidationError, match="cannot establish"):
        DenseProviderFamilyContract(
            family=DenseProviderFamily.SINGLE_IMAGE_DEPTH,
            native_output=DenseProviderNativeOutput.DEPTH_EVIDENCE,
            frame_binding=DenseProviderFrameBinding.ALIGN_TO_AUTHORITATIVE_SFM,
            establishes_metric_scale=True,
        )


def test_i43a_implementation_contracts_map_each_backend_to_expected_family_and_stage() -> None:
    values = {item.implementation: item for item in dense_provider_implementation_contracts()}
    assert values[DenseProviderImplementation.COLMAP_MVS].family is DenseProviderFamily.MVS
    assert values[DenseProviderImplementation.COLMAP_MVS].integration_stage == "I4.3B"
    assert values[DenseProviderImplementation.DA3].family is DenseProviderFamily.MULTI_VIEW_AI_DEPTH
    assert values[DenseProviderImplementation.DA3].integration_stage == "I4.3C"
    assert values[DenseProviderImplementation.ROLLING_DEPTH].family is DenseProviderFamily.TEMPORAL_DEPTH
    assert values[DenseProviderImplementation.VIDEO_DEPTH_ANYTHING].integration_stage == "I4.3D"
    assert values[DenseProviderImplementation.METRIC3D_V2].family is DenseProviderFamily.SINGLE_IMAGE_DEPTH
    assert values[DenseProviderImplementation.MARIGOLD].integration_stage == "I4.3E"


def test_i43a_rolling_depth_identity_does_not_fabricate_existing_adapter() -> None:
    values = {item.implementation: item for item in dense_provider_implementation_contracts()}
    assert values[DenseProviderImplementation.ROLLING_DEPTH].existing_adapter_expected is False


def test_i43a_implementation_contract_rejects_hybrid_registration() -> None:
    with pytest.raises(ReconstructionValidationError, match="does not register a concrete Hybrid"):
        DenseProviderImplementationContract(
            DenseProviderImplementation.DA3,
            "Bad hybrid",
            DenseProviderFamily.HYBRID,
            "I4.3C",
            True,
        )


def test_i43a_descriptor_rejects_auto_provider_id() -> None:
    with pytest.raises(ReconstructionValidationError, match="Auto policy ID"):
        _descriptor("auto", DenseProviderImplementation.COLMAP_MVS)


def test_i43a_descriptor_derives_family_from_implementation() -> None:
    value = _descriptor("da3-depth", DenseProviderImplementation.DA3)
    assert value.family is DenseProviderFamily.MULTI_VIEW_AI_DEPTH
    assert value.production_qualified is False


def test_i43a_descriptor_availability_is_fail_closed() -> None:
    with pytest.raises(ReconstructionValidationError, match="unavailable_reason"):
        _descriptor("colmap-mvs", DenseProviderImplementation.COLMAP_MVS, unavailable_reason="")


def test_i43a_catalog_rejects_duplicate_provider_ids() -> None:
    a = _descriptor("provider-a", DenseProviderImplementation.COLMAP_MVS)
    b = _descriptor("provider-a", DenseProviderImplementation.DA3)
    with pytest.raises(ReconstructionValidationError, match="globally unique"):
        validate_dense_provider_catalog((a, b))


def test_i43a_catalog_rejects_duplicate_implementation_identity() -> None:
    a = _descriptor("colmap-a", DenseProviderImplementation.COLMAP_MVS)
    b = _descriptor("colmap-b", DenseProviderImplementation.COLMAP_MVS)
    with pytest.raises(ReconstructionValidationError, match="one canonical provider descriptor"):
        validate_dense_provider_catalog((a, b))


def test_i43a_auto_request_requires_no_concrete_requested_provider() -> None:
    with pytest.raises(ReconstructionValidationError, match="cannot carry requested_provider_id"):
        DenseProviderSelectionRequest(
            mode=DenseProviderSelectionMode.AUTO,
            available_view_count=47,
            authoritative_sfm_available=True,
            authoritative_sfm_frame_id="sfm-main",
            requested_provider_id="colmap-mvs",
        )


def test_i43a_explicit_request_requires_concrete_provider() -> None:
    with pytest.raises(ReconstructionValidationError, match="require requested_provider_id"):
        DenseProviderSelectionRequest(
            mode=DenseProviderSelectionMode.EXPLICIT_PROVIDER,
            available_view_count=47,
            authoritative_sfm_available=True,
            authoritative_sfm_frame_id="sfm-main",
        )


def test_i43a_explicit_request_rejects_redundant_engineer_override() -> None:
    with pytest.raises(ReconstructionValidationError, match="do not use engineer_override"):
        DenseProviderSelectionRequest(
            mode=DenseProviderSelectionMode.EXPLICIT_PROVIDER,
            available_view_count=47,
            authoritative_sfm_available=True,
            authoritative_sfm_frame_id="sfm-main",
            requested_provider_id="colmap-mvs",
            engineer_override_provider_id="da3-depth",
        )


def test_i43a_selection_request_requires_authoritative_frame_identity_when_available() -> None:
    with pytest.raises(ReconstructionValidationError, match="frame_id"):
        DenseProviderSelectionRequest(
            mode=DenseProviderSelectionMode.AUTO,
            available_view_count=47,
            authoritative_sfm_available=True,
        )


def test_i43a_auto_decision_selects_concrete_implementation_not_family() -> None:
    decision = DenseProviderSelectionDecision(
        request_mode=DenseProviderSelectionMode.AUTO,
        requested_provider_id="",
        selected_provider_id="colmap-mvs",
        selected_implementation=DenseProviderImplementation.COLMAP_MVS,
        considered_provider_ids=("colmap-mvs", "da3-depth"),
        automatic=True,
        engineer_override_applied=False,
        fallback_used=False,
        reason_codes=("strong_sfm_overlap_texture",),
    )
    assert decision.as_dict()["selected_family"] == "mvs"
    assert decision.selected_provider_id != "auto"


def test_i43a_auto_engineer_override_is_recorded_as_nonautomatic() -> None:
    decision = DenseProviderSelectionDecision(
        request_mode=DenseProviderSelectionMode.AUTO,
        requested_provider_id="",
        selected_provider_id="da3-depth",
        selected_implementation=DenseProviderImplementation.DA3,
        considered_provider_ids=("colmap-mvs", "da3-depth"),
        automatic=False,
        engineer_override_applied=True,
        fallback_used=False,
        reason_codes=("engineer_override",),
    )
    assert decision.automatic is False


def test_i43a_explicit_decision_must_match_requested_provider_without_fallback() -> None:
    with pytest.raises(ReconstructionValidationError, match="must match requested_provider_id"):
        DenseProviderSelectionDecision(
            request_mode=DenseProviderSelectionMode.EXPLICIT_PROVIDER,
            requested_provider_id="colmap-mvs",
            selected_provider_id="da3-depth",
            selected_implementation=DenseProviderImplementation.DA3,
            considered_provider_ids=("colmap-mvs", "da3-depth"),
            automatic=False,
            engineer_override_applied=False,
            fallback_used=False,
            reason_codes=("wrong",),
        )


def test_i43a_explicit_fallback_can_select_different_provider_when_recorded() -> None:
    decision = DenseProviderSelectionDecision(
        request_mode=DenseProviderSelectionMode.EXPLICIT_PROVIDER,
        requested_provider_id="colmap-mvs",
        selected_provider_id="da3-depth",
        selected_implementation=DenseProviderImplementation.DA3,
        considered_provider_ids=("colmap-mvs", "da3-depth"),
        automatic=False,
        engineer_override_applied=False,
        fallback_used=True,
        reason_codes=("mvs_environment_unavailable", "explicit_fallback"),
    )
    assert decision.fallback_used is True


def test_i43a_unselected_decision_requires_reason() -> None:
    with pytest.raises(ReconstructionValidationError, match="explicit reason"):
        DenseProviderSelectionDecision(
            request_mode=DenseProviderSelectionMode.AUTO,
            requested_provider_id="",
            selected_provider_id="",
            selected_implementation=None,
            considered_provider_ids=("colmap-mvs",),
            automatic=True,
            engineer_override_applied=False,
            fallback_used=False,
            reason_codes=(),
        )


def test_i43a_progress_event_can_publish_replaceable_provisional_geometry() -> None:
    event = DenseProviderProgressEvent(
        provider_id="colmap-mvs",
        implementation=DenseProviderImplementation.COLMAP_MVS,
        stage=DenseProviderProgressStage.PROVIDER_EXECUTION,
        stage_detail="patch_match_stereo",
        progress_fraction=0.5,
        processed_views=24,
        total_views=47,
        provisional_geometry_available=True,
        provisional_geometry_ref="memory://preview/12",
        preview_revision=12,
        gpu_vram_used_mib=4096.0,
        process_ram_used_mib=2048.0,
    )
    payload = event.as_dict()
    assert payload["authority"]["provisional_geometry_authoritative"] is False
    assert payload["preview_revision"] == 12


def test_i43a_progress_event_rejects_unversioned_provisional_geometry() -> None:
    with pytest.raises(ReconstructionValidationError, match="positive preview_revision"):
        DenseProviderProgressEvent(
            provider_id="da3-depth",
            implementation=DenseProviderImplementation.DA3,
            stage=DenseProviderProgressStage.ALIGNMENT,
            stage_detail="canonical_alignment",
            progress_fraction=0.6,
            processed_views=12,
            total_views=20,
            provisional_geometry_available=True,
            provisional_geometry_ref="memory://preview/current",
            preview_revision=0,
        )


def test_i43a_progress_event_rejects_authoritative_preview_geometry() -> None:
    with pytest.raises(ReconstructionValidationError, match="cannot be authoritative"):
        DenseProviderProgressEvent(
            provider_id="video-depth-anything",
            implementation=DenseProviderImplementation.VIDEO_DEPTH_ANYTHING,
            stage=DenseProviderProgressStage.PROVIDER_EXECUTION,
            stage_detail="temporal_depth",
            progress_fraction=0.2,
            processed_views=10,
            total_views=50,
            provisional_geometry_authoritative=True,
        )


def test_i43a_progress_event_rejects_invalid_progress_or_counts() -> None:
    with pytest.raises(ReconstructionValidationError, match=r"within \[0, 1\]"):
        DenseProviderProgressEvent(
            provider_id="metric3d-v2",
            implementation=DenseProviderImplementation.METRIC3D_V2,
            stage=DenseProviderProgressStage.PROVIDER_EXECUTION,
            stage_detail="depth_inference",
            progress_fraction=1.2,
            processed_views=1,
            total_views=10,
        )


def test_i43a_completed_result_preserves_sfm_authority_and_nonmetric_limits() -> None:
    result = DenseProviderExecutionResult(
        provider_id="colmap-mvs",
        implementation=DenseProviderImplementation.COLMAP_MVS,
        status=DenseProviderRunStatus.COMPLETED,
        source_view_count=68,
        selected_view_count=68,
        adaptive_view_selection=True,
        view_selection_reason="all qualified views retained",
        permanent_global_view_cap=None,
        canonical_geometry_ref="file:///scene/dense.ply",
        canonical_frame_id="sfm-main",
        native_artifact_refs=("file:///scene/fused.ply",),
        progress_event_count=14,
        final_preview_revision=4,
    )
    assert result.camera_pose_authority_owner == I43A_AUTHORITATIVE_FRAME_OWNER
    assert result.family is DenseProviderFamily.MVS
    assert result.metric_scale_claimed is False


def test_i43a_reduced_view_result_requires_adaptive_explainable_selection() -> None:
    with pytest.raises(ReconstructionValidationError, match="adaptive"):
        DenseProviderExecutionResult(
            provider_id="colmap-mvs",
            implementation=DenseProviderImplementation.COLMAP_MVS,
            status=DenseProviderRunStatus.COMPLETED,
            source_view_count=68,
            selected_view_count=48,
            adaptive_view_selection=False,
            view_selection_reason="",
            permanent_global_view_cap=None,
            canonical_geometry_ref="file:///scene/dense.ply",
            canonical_frame_id="sfm-main",
        )


def test_i43a_execution_result_rejects_permanent_view_cap() -> None:
    with pytest.raises(ReconstructionValidationError, match="permanent global"):
        DenseProviderExecutionResult(
            provider_id="colmap-mvs",
            implementation=DenseProviderImplementation.COLMAP_MVS,
            status=DenseProviderRunStatus.COMPLETED,
            source_view_count=68,
            selected_view_count=40,
            adaptive_view_selection=True,
            view_selection_reason="legacy cap",
            permanent_global_view_cap=40,
            canonical_geometry_ref="file:///scene/dense.ply",
            canonical_frame_id="sfm-main",
        )


def test_i43a_degraded_result_requires_explicit_degradation_reason() -> None:
    with pytest.raises(ReconstructionValidationError, match="degradation_reasons"):
        DenseProviderExecutionResult(
            provider_id="da3-depth",
            implementation=DenseProviderImplementation.DA3,
            status=DenseProviderRunStatus.DEGRADED,
            source_view_count=20,
            selected_view_count=20,
            adaptive_view_selection=True,
            view_selection_reason="all views retained",
            permanent_global_view_cap=None,
            canonical_geometry_ref="file:///scene/dense.npz",
            canonical_frame_id="sfm-main",
        )


def test_i43a_failed_result_cannot_claim_canonical_geometry() -> None:
    result = DenseProviderExecutionResult(
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
        failure_reason="environment unavailable",
        canonical_geometry_authoritative=False,
    )
    assert result.status is DenseProviderRunStatus.FAILED


def test_i43a_final_preview_revision_requires_progress_history() -> None:
    with pytest.raises(ReconstructionValidationError, match="requires at least one progress event"):
        DenseProviderExecutionResult(
            provider_id="colmap-mvs",
            implementation=DenseProviderImplementation.COLMAP_MVS,
            status=DenseProviderRunStatus.COMPLETED,
            source_view_count=47,
            selected_view_count=47,
            adaptive_view_selection=True,
            view_selection_reason="all views retained",
            permanent_global_view_cap=None,
            canonical_geometry_ref="file:///scene/dense.ply",
            canonical_frame_id="sfm-main",
            progress_event_count=0,
            final_preview_revision=1,
        )


def test_i43a_authorization_accepts_closed_i42_without_promoting_provider() -> None:
    gate = validate_i4_3a_authorization(_closure())
    assert gate["i4_2_closed"] is True
    assert gate["i4_3_provider_architecture_authorized"] is True
    assert gate["global_universal_detail_safety_claimed"] is False


def test_i43a_authorization_rejects_provider_already_promoted() -> None:
    payload = _closure()
    payload["decision"]["dense_provider_promoted"] = True
    with pytest.raises(ReconstructionValidationError, match="dense_provider_promoted"):
        validate_i4_3a_authorization(payload)


def test_i43a_snapshot_locks_complete_room_comparison_and_progressive_preview() -> None:
    snapshot = build_i4_3a_architecture_snapshot(i4_2f_closure=_closure())
    assert snapshot["architecture"]["selection_modes"] == ["auto", "explicit_provider"]
    assert snapshot["qualification_plan"]["complete_room_mvs_vs_da3_required"] is True
    assert snapshot["progressive_preview_contract"]["required_for_v1_0"] is True
    assert snapshot["progressive_preview_contract"]["provisional_geometry_authoritative"] is False
    assert snapshot["authority"]["backend_started"] is False
    assert snapshot["authority"]["provider_promoted"] is False


def test_i43a_snapshot_locks_i43b_through_i43h_roadmap() -> None:
    snapshot = build_i4_3a_architecture_snapshot(i4_2f_closure=_closure())
    assert len(snapshot["i4_3_roadmap"]) == 7
    assert snapshot["i4_3_roadmap"][0].startswith("I4.3B")
    assert snapshot["i4_3_roadmap"][-1].startswith("I4.3H")


def test_i43a_closure_reader_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "closure.json"
    path.write_text("not-json", encoding="utf-8")
    with pytest.raises(ReconstructionValidationError, match="Unable to read"):
        read_i4_2f_closure(path)

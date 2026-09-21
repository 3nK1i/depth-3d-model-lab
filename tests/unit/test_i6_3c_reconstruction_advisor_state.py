from __future__ import annotations

from types import SimpleNamespace
from uuid import UUID

from app.reconstruction_advisor_state import (
    ReconstructionAdvisorEvidenceBinding,
    build_reconstruction_advisor_live_state,
)
from app.reconstruction_offline_workspace import (
    OfflineReconstructionWorkspaceSnapshot,
    OfflineWorkspaceEvidenceIdentityStatus,
    OfflineWorkspaceScaleStatus,
)
from app.reconstruction_session_model import (
    ReconstructionSessionFocus,
    ReconstructionSessionIdentityStatus,
    ReconstructionSessionLifecycle,
    ReconstructionSessionModel,
)
from core.reconstruction.dense_provider_architecture import (
    DenseProviderFamily,
    DenseProviderImplementation,
)
from core.reconstruction.reconstruction_advisor_contract import (
    AdvisorContextResolution,
    AdvisorDisposition,
    AdvisorEvidenceBinding,
    AdvisorEvidenceObservation,
    AdvisorEvidenceOutcome,
    AdvisorResourceObservation,
)

SESSION = "11111111-1111-4111-8111-111111111111"
SHA = "a" * 64


def _session(*, lifecycle: ReconstructionSessionLifecycle = ReconstructionSessionLifecycle.READY):
    return ReconstructionSessionModel(
        session_id=SESSION,
        focus=ReconstructionSessionFocus.LAST_RUNTIME,
        identity_status=ReconstructionSessionIdentityStatus.CONSISTENT,
        lifecycle=lifecycle,
        workflow="image_sequence",
        selected_source_label="images",
        selected_source_detail="128 supported images",
        selected_source_path="/data/images",
        scene_id="",
        scene_manifest="",
        scene_state="",
        scene_availability="none",
        checkpoint_id="",
        checkpoint_manifest="",
        checkpoint_next_stage="",
        checkpoint_resume_allowed=False,
        artifacts=(),
        providers=(),
        appearance_representation_kinds=(),
        scene_created_at_utc="",
        checkpoint_created_at_utc="",
        canonical_geometry_authoritative=False,
        certified_metrology=False,
        identity_details="consistent",
    )


def _workspace(*, partial=False, identity=OfflineWorkspaceEvidenceIdentityStatus.CONSISTENT):
    return OfflineReconstructionWorkspaceSnapshot(
        session_model=_session(
            lifecycle=(
                ReconstructionSessionLifecycle.COMPLETED
                if partial
                else ReconstructionSessionLifecycle.READY
            )
        ),
        session_id=SESSION,
        result_session_id=SESSION if partial else "",
        evidence_identity_status=identity,
        identity_details="consistent" if identity is not OfflineWorkspaceEvidenceIdentityStatus.CONFLICT else "conflict",
        workflow="image_sequence",
        offline_workflow_supported=True,
        source_ready=True,
        lifecycle="completed" if partial else "ready",
        offline_run_state="completed" if partial else "idle",
        output_dir="/tmp/output" if partial else "",
        result_path="/tmp/output/offline_result.json" if partial else "",
        result_status="completed" if partial else "",
        current_stage="",
        progress_fraction=None,
        progress_message="",
        scene_availability="partial" if partial else "none",
        partial_result_available=partial,
        persistent_scene_available=False,
        scene_manifest="",
        checkpoint_available=False,
        checkpoint_manifest="",
        checkpoint_resume_allowed=False,
        checkpoint_next_stage="",
        camera_authority_owner="pycolmap_sfm",
        camera_pose_available=partial,
        canonical_geometry_authoritative=False,
        scale_status=(OfflineWorkspaceScaleStatus.ARBITRARY if partial else OfflineWorkspaceScaleStatus.UNQUALIFIED),
        metric_coordinates_allowed=False,
        calibrated_metric_scale=False,
        certified_metrology=False,
        production_auto_policy_scope="offline_shadow_only",
        production_auto_qualified=False,
        production_auto_authorized=False,
    )


def _selection(path="/data/images"):
    return SimpleNamespace(
        workflow=SimpleNamespace(value="image_sequence"),
        ready=True,
        item_count=128,
        path=path,
    )


def _resources():
    return AdvisorResourceObservation(
        gpu_name="NVIDIA GeForce RTX 2060",
        cuda_ready=True,
        total_vram_bytes=6 * 1024**3,
        free_vram_bytes=5 * 1024**3,
        available_ram_bytes=8 * 1024**3,
    )


def _evidence():
    return AdvisorEvidenceObservation(
        provider_id="colmap-mvs-external",
        provider_family=DenseProviderFamily.MVS,
        provider_implementation=DenseProviderImplementation.COLMAP_MVS,
        outcome=AdvisorEvidenceOutcome.PASS,
        evidence_context_id="ctx-clean",
        matched_scenario_ids=("scenario-clean",),
        reason_codes=("exact_clean_evidence",),
        limitations=("frozen_context_only",),
        bindings=(
            AdvisorEvidenceBinding(
                evidence_id="i5-final",
                phase="I5.14B",
                path="/evidence/i5-final.json",
                sha256=SHA,
            ),
        ),
    )


def _binding(*, runtime_clean=True):
    return ReconstructionAdvisorEvidenceBinding(
        evidence_context_resolution=AdvisorContextResolution.EXPLICIT_FROZEN_TOKEN,
        evidence_context_id="ctx-clean",
        available_provider_ids=("colmap-mvs-external",),
        resources=_resources(),
        provider_evidence=(_evidence(),),
        runtime_integrity_clean=runtime_clean,
    )


def _result(*, source_uri="/data/images", usable=True):
    UUID(SESSION)
    return {
        "session_id": SESSION,
        "status": "completed",
        "pass": True,
        "camera_pose_available": True,
        "persistent_scene_ready": False,
        "certified_metrology": False,
        "source": {
            "source_uri": source_uri,
            "fingerprint": "source-fingerprint",
        },
        "sfm_summary": {
            "quality_assessment": {"status": "accepted", "usable": usable},
            "gauge": {"scale_status": "arbitrary"},
        },
    }


def test_i6_3c_unresolved_startup_binding_is_passive_and_non_executable() -> None:
    binding = ReconstructionAdvisorEvidenceBinding.unresolved()
    state = build_reconstruction_advisor_live_state(
        source_selection=SimpleNamespace(
            workflow=SimpleNamespace(value="none"), ready=False, item_count=None, path=None
        ),
        workspace=_workspace(),
        offline_result=None,
        evidence_binding=binding,
    )

    assert state.decision.disposition is AdvisorDisposition.ABSTAIN
    assert state.production_effect is False
    assert state.request.production_auto_authorized is False
    assert state.request.provider_activation_authorized is False
    assert state.request.execution_authorized is False
    assert state.request.settings_mutation_authorized is False


def test_i6_3c_exact_source_admitted_usable_sfm_can_feed_pure_engine_recommendation() -> None:
    state = build_reconstruction_advisor_live_state(
        source_selection=_selection(),
        workspace=_workspace(partial=True),
        offline_result=_result(),
        evidence_binding=_binding(),
    )

    assert state.source_matches_admitted_result is True
    assert state.request.source_fingerprint == "source-fingerprint"
    assert state.request.authoritative_sfm_usable is True
    assert state.request.workspace_state == "partial_result"
    assert state.decision.disposition is AdvisorDisposition.RECOMMEND
    assert state.decision.recommended_provider_id == "colmap-mvs-external"
    assert state.decision.production_effect is False


def test_i6_3c_stale_result_from_different_source_is_not_reused() -> None:
    state = build_reconstruction_advisor_live_state(
        source_selection=_selection(path="/data/new-images"),
        workspace=_workspace(partial=True),
        offline_result=_result(source_uri="/data/old-images"),
        evidence_binding=_binding(),
    )

    assert state.source_matches_admitted_result is False
    assert state.request.source_fingerprint == ""
    assert state.request.authoritative_sfm_usable is False
    assert state.decision.disposition is AdvisorDisposition.ABSTAIN
    assert "authoritative_sfm_not_usable" in state.decision.reason_codes


def test_i6_3c_quality_assessment_must_explicitly_mark_sfm_usable() -> None:
    state = build_reconstruction_advisor_live_state(
        source_selection=_selection(),
        workspace=_workspace(partial=True),
        offline_result=_result(usable=False),
        evidence_binding=_binding(),
    )

    assert state.request.authoritative_sfm_usable is False
    assert state.decision.disposition is AdvisorDisposition.ABSTAIN


def test_i6_3c_workspace_identity_conflict_forces_runtime_integrity_false() -> None:
    state = build_reconstruction_advisor_live_state(
        source_selection=_selection(),
        workspace=_workspace(
            partial=True,
            identity=OfflineWorkspaceEvidenceIdentityStatus.CONFLICT,
        ),
        offline_result=_result(),
        evidence_binding=_binding(),
    )

    assert state.request.authoritative_sfm_usable is False
    assert state.request.runtime_integrity_clean is False
    assert state.decision.disposition is AdvisorDisposition.ABSTAIN


def test_i6_3c_explicit_runtime_integrity_blocker_is_preserved() -> None:
    state = build_reconstruction_advisor_live_state(
        source_selection=_selection(),
        workspace=_workspace(partial=True),
        offline_result=_result(),
        evidence_binding=_binding(runtime_clean=False),
    )

    assert state.request.runtime_integrity_clean is False
    assert state.decision.reason_codes == ("advisor_abstain", "runtime_integrity_not_clean")


def test_i6_3c_ready_source_without_admitted_sfm_result_abstains() -> None:
    state = build_reconstruction_advisor_live_state(
        source_selection=_selection(),
        workspace=_workspace(partial=False),
        offline_result=None,
        evidence_binding=_binding(),
    )

    assert state.request.source_ready is True
    assert state.request.workspace_state == "ready"
    assert state.request.authoritative_sfm_usable is False
    assert state.decision.disposition is AdvisorDisposition.ABSTAIN


def test_i6_3c_binding_rejects_auto_and_duplicate_provider_ids() -> None:
    import pytest

    with pytest.raises(ValueError, match="concrete providers"):
        ReconstructionAdvisorEvidenceBinding(
            evidence_context_resolution=AdvisorContextResolution.UNRESOLVED,
            evidence_context_id="",
            available_provider_ids=("auto",),
            resources=_resources(),
            provider_evidence=(),
        )
    with pytest.raises(ValueError, match="unique"):
        ReconstructionAdvisorEvidenceBinding(
            evidence_context_resolution=AdvisorContextResolution.UNRESOLVED,
            evidence_context_id="",
            available_provider_ids=("provider-a", "provider-a"),
            resources=_resources(),
            provider_evidence=(),
        )

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from core.reconstruction import (
    ENGINE_ENDURANCE_PHASE,
    ENGINE_ENDURANCE_SCHEMA_VERSION,
    DeterministicMockStageExecutor,
    ReconstructionEnduranceCycleEvidence,
    ReconstructionEnduranceScenario,
    ReconstructionPath,
    ReconstructionPipelineGoal,
    ReconstructionPipelineOrchestrator,
    ReconstructionPipelineRequest,
    ReconstructionPipelineRunStatus,
    ReconstructionProviderDescriptor,
    ReconstructionProviderExecution,
    ReconstructionProviderRole,
    ReconstructionQualityProfile,
    ReconstructionResumeCompatibilityStatus,
    ReconstructionValidationError,
    build_reconstruction_endurance_campaign,
    build_reconstruction_pipeline_plan,
    check_reconstruction_engine_resume_compatibility,
    create_reconstruction_engine_checkpoint,
)
from tests.reconstruction_f5_helpers import run_f5_fixture


def _clean(cycle: int, scene: str | None = None):
    return ReconstructionEnduranceCycleEvidence(
        cycle_index=cycle,
        scenario=ReconstructionEnduranceScenario.CLEAN_RECOVERY,
        passed=True,
        pipeline_status="completed",
        persistent_scene_state="ready",
        scene_id=scene or f"prs-clean-{cycle}",
        canonical_geometry_authoritative=True,
        gaussian_authority="appearance_only",
        nerf_authority="appearance_only",
    )


def test_g3_schema_and_phase_are_explicit():
    assert ENGINE_ENDURANCE_SCHEMA_VERSION == 1
    assert ENGINE_ENDURANCE_PHASE == "8.7G3"


def test_clean_cycle_requires_ready_scene_and_authority():
    with pytest.raises(ReconstructionValidationError, match="ready scene"):
        ReconstructionEnduranceCycleEvidence(
            cycle_index=0,
            scenario=ReconstructionEnduranceScenario.CLEAN_RECOVERY,
            passed=True,
            pipeline_status="completed",
            persistent_scene_state="degraded",
            scene_id="prs-x",
            canonical_geometry_authoritative=True,
            gaussian_authority="appearance_only",
            nerf_authority="appearance_only",
        )


def test_passing_cycle_cannot_hide_process_leak():
    with pytest.raises(ReconstructionValidationError, match="clean process boundary"):
        ReconstructionEnduranceCycleEvidence(
            cycle_index=0,
            scenario=ReconstructionEnduranceScenario.CHECKPOINT_REJECTION,
            passed=True,
            process_group_clean=False,
            checkpoint_rejected_safely=True,
        )


def test_optional_degradation_keeps_canonical_geometry_and_one_appearance_backend():
    evidence = ReconstructionEnduranceCycleEvidence(
        cycle_index=1,
        scenario=ReconstructionEnduranceScenario.OPTIONAL_DEGRADATION,
        passed=True,
        pipeline_status="degraded",
        persistent_scene_state="degraded",
        canonical_geometry_authoritative=True,
        gaussian_authority="missing",
        nerf_authority="appearance_only",
    )
    assert evidence.passed


def test_optional_degradation_cannot_promote_neural_authority():
    with pytest.raises(ReconstructionValidationError, match="authority"):
        ReconstructionEnduranceCycleEvidence(
            cycle_index=1,
            scenario=ReconstructionEnduranceScenario.OPTIONAL_DEGRADATION,
            passed=True,
            pipeline_status="degraded",
            canonical_geometry_authoritative=True,
            gaussian_authority="canonical_geometry",
            nerf_authority="appearance_only",
        )


def test_required_failure_requires_fail_closed_and_no_scene():
    evidence = ReconstructionEnduranceCycleEvidence(
        cycle_index=2,
        scenario=ReconstructionEnduranceScenario.REQUIRED_FAILURE,
        passed=True,
        pipeline_status="failed",
        required_failure_fail_closed=True,
    )
    assert evidence.required_failure_fail_closed


def test_checkpoint_rejection_must_be_safe():
    with pytest.raises(ReconstructionValidationError, match="reject safely"):
        ReconstructionEnduranceCycleEvidence(
            cycle_index=3,
            scenario=ReconstructionEnduranceScenario.CHECKPOINT_REJECTION,
            passed=True,
            checkpoint_rejected_safely=False,
        )


def test_campaign_requires_unique_indices():
    with pytest.raises(ReconstructionValidationError, match="indices"):
        build_reconstruction_endurance_campaign(
            (_clean(0, "prs-a"), _clean(0, "prs-b")),
            backward_compatibility_verified=True,
            resource_cleanup_verified=True,
            authority_stability_verified=True,
            checkpoint_safety_verified=True,
        )


def test_campaign_rejects_stale_scene_identity_reuse():
    report = build_reconstruction_endurance_campaign(
        (_clean(0, "prs-same"), _clean(1, "prs-same")),
        backward_compatibility_verified=True,
        resource_cleanup_verified=True,
        authority_stability_verified=True,
        checkpoint_safety_verified=True,
    )
    assert report.passed is False
    assert report.as_dict()["unique_clean_scene_ids"] is False


def test_campaign_passes_mixed_bounded_matrix():
    cycles = (
        _clean(0),
        _clean(1),
        ReconstructionEnduranceCycleEvidence(
            cycle_index=2,
            scenario=ReconstructionEnduranceScenario.OPTIONAL_DEGRADATION,
            passed=True,
            pipeline_status="degraded",
            persistent_scene_state="degraded",
            canonical_geometry_authoritative=True,
            gaussian_authority="missing",
            nerf_authority="appearance_only",
        ),
        ReconstructionEnduranceCycleEvidence(
            cycle_index=3,
            scenario=ReconstructionEnduranceScenario.REQUIRED_FAILURE,
            passed=True,
            pipeline_status="failed",
            required_failure_fail_closed=True,
        ),
        ReconstructionEnduranceCycleEvidence(
            cycle_index=4,
            scenario=ReconstructionEnduranceScenario.CHECKPOINT_REJECTION,
            passed=True,
            checkpoint_rejected_safely=True,
        ),
    )
    report = build_reconstruction_endurance_campaign(
        cycles,
        backward_compatibility_verified=True,
        resource_cleanup_verified=True,
        authority_stability_verified=True,
        checkpoint_safety_verified=True,
    )
    assert report.passed is True
    assert report.scenario_counts["clean_recovery"] == 2


def test_real_f5_degradation_fixture_maps_to_g3_optional_degradation(tmp_path: Path):
    fixture, providers, result, snapshots = run_f5_fixture(tmp_path, degrade_gaussian=True, request_id="g3-degrade")
    assert result.status is ReconstructionPipelineRunStatus.DEGRADED
    evidence = ReconstructionEnduranceCycleEvidence(
        cycle_index=0,
        scenario=ReconstructionEnduranceScenario.OPTIONAL_DEGRADATION,
        passed=True,
        pipeline_status=result.status.value,
        persistent_scene_state="degraded",
        canonical_geometry_authoritative=result.canonical_geometry_authoritative,
        gaussian_authority="missing",
        nerf_authority="appearance_only",
    )
    assert evidence.canonical_geometry_authoritative


def _provider(provider_id: str, role: ReconstructionProviderRole, version: str = "1"):
    return ReconstructionProviderDescriptor(
        provider_id=provider_id,
        provider_version=version,
        roles=(role,),
        execution=ReconstructionProviderExecution.DEVELOPMENT_MOCK,
        priority=10,
        adapter_version="g3-test",
    )


def test_provider_drift_checkpoint_is_rejected_safely():
    providers = (
        _provider("ingestion", ReconstructionProviderRole.INGESTION),
        _provider("pose", ReconstructionProviderRole.CAMERA_POSE),
        _provider("depth", ReconstructionProviderRole.DEPTH),
        _provider("dense", ReconstructionProviderRole.DENSE_GEOMETRY),
        _provider("mesh", ReconstructionProviderRole.SURFACE_MESH),
    )
    request = ReconstructionPipelineRequest(
        request_id="g3-checkpoint",
        session_id=str(uuid4()),
        path=ReconstructionPath.OFFLINE,
        quality_profile=ReconstructionQualityProfile.EFFICIENT,
        goals=(ReconstructionPipelineGoal.CANONICAL_GEOMETRY,),
    )
    plan = build_reconstruction_pipeline_plan(request, providers)
    committed = []
    control_result = ReconstructionPipelineOrchestrator(
        plan=plan,
        providers=providers,
        executors={p.provider_id: DeterministicMockStageExecutor(p.provider_id) for p in providers},
    ).run(stage_commit_hook=lambda stages: committed.append(stages))
    checkpoint = create_reconstruction_engine_checkpoint(plan=plan, providers=providers, stage_results=committed[2])
    drifted = list(providers)
    drifted[-1] = _provider("mesh", ReconstructionProviderRole.SURFACE_MESH, version="2")
    compatibility = check_reconstruction_engine_resume_compatibility(
        checkpoint,
        plan=plan,
        providers=tuple(drifted),
        available_payload_refs=checkpoint.payload_refs,
    )
    assert compatibility.status is ReconstructionResumeCompatibilityStatus.INCOMPATIBLE

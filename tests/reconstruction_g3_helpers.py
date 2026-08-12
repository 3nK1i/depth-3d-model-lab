from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from core.reconstruction import (
    DeterministicMockStageExecutor,
    ReconstructionEnduranceCycleEvidence,
    ReconstructionEnduranceScenario,
    ReconstructionOperationControl,
    ReconstructionPath,
    ReconstructionPipelineGoal,
    ReconstructionPipelineOrchestrator,
    ReconstructionPipelineRequest,
    ReconstructionPipelineRunStatus,
    ReconstructionPipelineStage,
    ReconstructionProviderDescriptor,
    ReconstructionProviderExecution,
    ReconstructionProviderRole,
    ReconstructionQualityProfile,
    check_reconstruction_engine_resume_compatibility,
    create_reconstruction_engine_checkpoint,
    persistent_scene_from_pipeline_result,
    write_reconstruction_engine_checkpoint,
    build_reconstruction_pipeline_plan,
)
from tests.reconstruction_f3_helpers import make_f3_fixture
from tests.reconstruction_f5_helpers import run_f5_fixture


def provider(provider_id: str, role: ReconstructionProviderRole, *, version: str = "1"):
    return ReconstructionProviderDescriptor(
        provider_id=provider_id,
        provider_version=version,
        roles=(role,),
        execution=ReconstructionProviderExecution.DEVELOPMENT_MOCK,
        priority=10,
        adapter_version="phase8.8g3-test",
    )


def semantic_fixture(session_id: str | None = None):
    providers = (
        provider("ingestion", ReconstructionProviderRole.INGESTION),
        provider("pose", ReconstructionProviderRole.CAMERA_POSE),
        provider("depth", ReconstructionProviderRole.DEPTH),
        provider("dense", ReconstructionProviderRole.DENSE_GEOMETRY),
        provider("mesh", ReconstructionProviderRole.SURFACE_MESH),
        provider("semantics", ReconstructionProviderRole.SEMANTICS),
        provider("inspection", ReconstructionProviderRole.INSPECTION_QC),
    )
    request = ReconstructionPipelineRequest(
        request_id="phase8-8g3-test",
        session_id=session_id or str(uuid4()),
        path=ReconstructionPath.OFFLINE,
        quality_profile=ReconstructionQualityProfile.EFFICIENT,
        goals=(ReconstructionPipelineGoal.SEMANTIC_INSPECTION,),
    )
    plan = build_reconstruction_pipeline_plan(request, providers)
    executors = {
        item.provider_id: DeterministicMockStageExecutor(item.provider_id)
        for item in providers
    }
    return providers, request, plan, executors


def checkpoint_prefix(plan, providers, executors):
    control = ReconstructionOperationControl()
    committed = []

    def hook(stages):
        committed.append(stages)
        if stages[-1].stage is ReconstructionPipelineStage.DEPTH_EVIDENCE:
            control.request_cancel()

    result = ReconstructionPipelineOrchestrator(
        plan=plan,
        providers=providers,
        executors=executors,
    ).run(control=control, stage_commit_hook=hook)
    assert result.status is ReconstructionPipelineRunStatus.CANCELLED
    return committed[-1]


def write_checkpoint_fixture(root: Path, *, session_id: str | None = None):
    providers, request, plan, executors = semantic_fixture(session_id=session_id)
    prefix = checkpoint_prefix(plan, providers, executors)
    checkpoint = create_reconstruction_engine_checkpoint(
        plan=plan,
        providers=providers,
        stage_results=prefix,
    )
    manifest = write_reconstruction_engine_checkpoint(checkpoint, root.resolve())
    return providers, request, plan, checkpoint, manifest

# Backward-compatible Phase 8.7G3 endurance helpers.
#
# Phase 8.8G3 added checkpoint/resume UI fixtures to this shared helper module.
# The earlier endurance helpers remain part of the historical integration-test
# surface and must stay available so the full Phase 8.7/8.8 regression can be
# collected and executed together.
def clean_evidence(index: int) -> ReconstructionEnduranceCycleEvidence:
    fixture = make_f3_fixture(request_id=f"g3-clean-{index}")
    request = ReconstructionPipelineRequest(
        request_id=f"g3-clean-{index}",
        session_id=fixture.session_id,
        path=ReconstructionPath.OFFLINE,
        quality_profile=ReconstructionQualityProfile.EFFICIENT,
        goals=(ReconstructionPipelineGoal.SEMANTIC_INSPECTION,),
    )
    plan = build_reconstruction_pipeline_plan(request, fixture.providers)
    result = ReconstructionPipelineOrchestrator(
        plan=plan, providers=fixture.providers, executors=fixture.executors
    ).run()
    return ReconstructionEnduranceCycleEvidence(
        cycle_index=index,
        scenario=ReconstructionEnduranceScenario.CLEAN_RECOVERY,
        passed=True,
        pipeline_status=result.status.value,
        persistent_scene_state="ready",
        scene_id=f"prs-g3-clean-{index}-{fixture.session_id[:8]}",
        canonical_geometry_authoritative=result.canonical_geometry_authoritative,
        gaussian_authority="appearance_only",
        nerf_authority="appearance_only",
    )


def optional_degradation_evidence(index: int, root: Path) -> ReconstructionEnduranceCycleEvidence:
    fixture, _providers, result, snapshots = run_f5_fixture(
        root / f"degraded-{index}", degrade_gaussian=True, request_id=f"g3-degraded-{index}"
    )
    scene = persistent_scene_from_pipeline_result(result, payload_snapshots_by_kind=snapshots)
    appearance = scene.as_dict()["appearance"]
    return ReconstructionEnduranceCycleEvidence(
        cycle_index=index,
        scenario=ReconstructionEnduranceScenario.OPTIONAL_DEGRADATION,
        passed=True,
        pipeline_status=result.status.value,
        persistent_scene_state=scene.state.value,
        scene_id=scene.scene_id,
        canonical_geometry_authoritative=scene.canonical_geometry_authoritative,
        gaussian_authority="missing" if not appearance["gaussian_artifact_id"] else appearance["authority"],
        nerf_authority="missing" if not appearance["nerf_artifact_id"] else appearance["authority"],
    )


def required_failure_evidence(index: int) -> ReconstructionEnduranceCycleEvidence:
    fixture = make_f3_fixture(request_id=f"g3-required-failure-{index}")

    class FailDense:
        def execute(self, _context):
            raise RuntimeError("g3_injected_required_dense_failure")

    executors = dict(fixture.executors)
    dense_id = next(
        p.provider_id
        for p in fixture.providers
        if ReconstructionProviderRole.DENSE_GEOMETRY in p.roles
    )
    executors[dense_id] = FailDense()
    request = ReconstructionPipelineRequest(
        request_id=f"g3-required-failure-{index}",
        session_id=fixture.session_id,
        path=ReconstructionPath.OFFLINE,
        quality_profile=ReconstructionQualityProfile.EFFICIENT,
        goals=(ReconstructionPipelineGoal.SEMANTIC_INSPECTION,),
    )
    plan = build_reconstruction_pipeline_plan(request, fixture.providers)
    result = ReconstructionPipelineOrchestrator(
        plan=plan, providers=fixture.providers, executors=executors
    ).run()
    return ReconstructionEnduranceCycleEvidence(
        cycle_index=index,
        scenario=ReconstructionEnduranceScenario.REQUIRED_FAILURE,
        passed=result.status.value == "failed",
        pipeline_status=result.status.value,
        canonical_geometry_authoritative=False,
        required_failure_fail_closed=result.status.value == "failed",
        reason=result.stages[-1].reason,
    )


def checkpoint_rejection_evidence(index: int) -> ReconstructionEnduranceCycleEvidence:
    providers = (
        provider("ingestion", ReconstructionProviderRole.INGESTION),
        provider("pose", ReconstructionProviderRole.CAMERA_POSE),
        provider("depth", ReconstructionProviderRole.DEPTH),
        provider("dense", ReconstructionProviderRole.DENSE_GEOMETRY),
        provider("mesh", ReconstructionProviderRole.SURFACE_MESH),
    )
    request = ReconstructionPipelineRequest(
        request_id=f"g3-checkpoint-reject-{index}",
        session_id=str(uuid4()),
        path=ReconstructionPath.OFFLINE,
        quality_profile=ReconstructionQualityProfile.EFFICIENT,
        goals=(ReconstructionPipelineGoal.CANONICAL_GEOMETRY,),
    )
    plan = build_reconstruction_pipeline_plan(request, providers)
    prefixes = []
    ReconstructionPipelineOrchestrator(
        plan=plan,
        providers=providers,
        executors={
            item.provider_id: DeterministicMockStageExecutor(item.provider_id)
            for item in providers
        },
    ).run(stage_commit_hook=lambda stages: prefixes.append(stages))
    checkpoint = create_reconstruction_engine_checkpoint(
        plan=plan, providers=providers, stage_results=prefixes[2]
    )
    drifted = list(providers)
    drifted[-1] = provider(
        "mesh", ReconstructionProviderRole.SURFACE_MESH, version="g3-drift"
    )
    compatibility = check_reconstruction_engine_resume_compatibility(
        checkpoint,
        plan=plan,
        providers=tuple(drifted),
        available_payload_refs=checkpoint.payload_refs,
    )
    return ReconstructionEnduranceCycleEvidence(
        cycle_index=index,
        scenario=ReconstructionEnduranceScenario.CHECKPOINT_REJECTION,
        passed=not compatibility.compatible,
        checkpoint_rejected_safely=not compatibility.compatible,
        reason=";".join(compatibility.reasons),
    )

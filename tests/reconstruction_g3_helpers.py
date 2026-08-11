from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from core.reconstruction import (
    DeterministicMockStageExecutor,
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
    create_reconstruction_engine_checkpoint,
    write_reconstruction_engine_checkpoint,
    build_reconstruction_pipeline_plan,
)


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

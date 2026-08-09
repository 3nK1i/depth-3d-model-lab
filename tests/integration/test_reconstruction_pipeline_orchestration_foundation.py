from __future__ import annotations

from uuid import uuid4

from core.reconstruction import (
    DeterministicMockStageExecutor,
    ReconstructionArtifactAuthority,
    ReconstructionPath,
    ReconstructionPipelineExecutionScope,
    ReconstructionPipelineGoal,
    ReconstructionPipelineOrchestrator,
    ReconstructionPipelineRequest,
    ReconstructionPipelineRunStatus,
    ReconstructionPipelineStage,
    ReconstructionProviderDescriptor,
    ReconstructionProviderExecution,
    ReconstructionProviderRole,
    ReconstructionQualityProfile,
    build_reconstruction_pipeline_plan,
)


def _provider(provider_id, role, *, priority=100, available=True):
    return ReconstructionProviderDescriptor(
        provider_id=provider_id,
        provider_version="integration-1",
        roles=(role,),
        execution=ReconstructionProviderExecution.DEVELOPMENT_MOCK,
        priority=priority,
        available=available,
        unavailable_reason="" if available else "integration_unavailable",
    )


def test_whole_foundation_dag_preserves_authority_and_provider_fallback() -> None:
    providers = [
        _provider("ingestion", ReconstructionProviderRole.INGESTION),
        _provider("pose", ReconstructionProviderRole.CAMERA_POSE),
        _provider("depth-primary", ReconstructionProviderRole.DEPTH, priority=1, available=False),
        _provider("depth-fallback", ReconstructionProviderRole.DEPTH, priority=2),
        _provider("dense", ReconstructionProviderRole.DENSE_GEOMETRY),
        _provider("mesh", ReconstructionProviderRole.SURFACE_MESH),
        _provider("semantics", ReconstructionProviderRole.SEMANTICS),
        _provider("inspection", ReconstructionProviderRole.INSPECTION_QC),
        _provider("gaussian", ReconstructionProviderRole.GAUSSIAN_APPEARANCE),
        _provider("nerf", ReconstructionProviderRole.NERF_APPEARANCE),
    ]
    request = ReconstructionPipelineRequest(
        request_id="phase8-7f1-integration",
        session_id=str(uuid4()),
        path=ReconstructionPath.OFFLINE,
        quality_profile=ReconstructionQualityProfile.EFFICIENT,
        goals=(
            ReconstructionPipelineGoal.SEMANTIC_INSPECTION,
            ReconstructionPipelineGoal.GAUSSIAN_APPEARANCE,
            ReconstructionPipelineGoal.NERF_APPEARANCE,
        ),
    )
    plan = build_reconstruction_pipeline_plan(request, providers)
    executors = {
        provider.provider_id: DeterministicMockStageExecutor(provider.provider_id)
        for provider in providers
        if provider.available
    }
    result = ReconstructionPipelineOrchestrator(
        plan=plan,
        providers=providers,
        executors=executors,
    ).run()

    assert result.status is ReconstructionPipelineRunStatus.DEGRADED
    assert result.execution_scope is ReconstructionPipelineExecutionScope.DEVELOPMENT_MOCK
    assert result.canonical_geometry_authoritative is True
    assert result.certified_metrology is False
    assert result.reconstruction_ui_enabled is False

    stages = {stage.stage: stage for stage in result.stages}
    assert stages[ReconstructionPipelineStage.DEPTH_EVIDENCE].provider_id == "depth-fallback"
    assert stages[ReconstructionPipelineStage.DEPTH_EVIDENCE].fallback_used is True
    assert stages[ReconstructionPipelineStage.SURFACE_MESH].artifact.authority is ReconstructionArtifactAuthority.CANONICAL_GEOMETRY
    assert stages[ReconstructionPipelineStage.GAUSSIAN_APPEARANCE].artifact.authority is ReconstructionArtifactAuthority.APPEARANCE_ONLY
    assert stages[ReconstructionPipelineStage.NERF_APPEARANCE].artifact.authority is ReconstructionArtifactAuthority.APPEARANCE_ONLY

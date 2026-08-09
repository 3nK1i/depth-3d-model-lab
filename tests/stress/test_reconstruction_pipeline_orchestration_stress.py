from __future__ import annotations

from uuid import uuid4

from core.reconstruction import (
    DeterministicMockStageExecutor,
    ReconstructionPath,
    ReconstructionPipelineGoal,
    ReconstructionPipelineOrchestrator,
    ReconstructionPipelineRequest,
    ReconstructionPipelineRunStatus,
    ReconstructionProviderDescriptor,
    ReconstructionProviderExecution,
    ReconstructionProviderRole,
    ReconstructionQualityProfile,
    build_reconstruction_pipeline_plan,
)


def _providers():
    roles = (
        ("ingestion", ReconstructionProviderRole.INGESTION),
        ("pose", ReconstructionProviderRole.CAMERA_POSE),
        ("depth", ReconstructionProviderRole.DEPTH),
        ("dense", ReconstructionProviderRole.DENSE_GEOMETRY),
        ("mesh", ReconstructionProviderRole.SURFACE_MESH),
        ("semantics", ReconstructionProviderRole.SEMANTICS),
        ("inspection", ReconstructionProviderRole.INSPECTION_QC),
        ("gaussian", ReconstructionProviderRole.GAUSSIAN_APPEARANCE),
        ("nerf", ReconstructionProviderRole.NERF_APPEARANCE),
    )
    return [
        ReconstructionProviderDescriptor(
            provider_id=provider_id,
            provider_version="stress-1",
            roles=(role,),
            execution=ReconstructionProviderExecution.DEVELOPMENT_MOCK,
        )
        for provider_id, role in roles
    ]


def test_repeated_f1_orchestration_has_no_cross_run_state_leakage() -> None:
    providers = _providers()
    for index in range(100):
        request = ReconstructionPipelineRequest(
            request_id=f"stress-{index}",
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
        }
        result = ReconstructionPipelineOrchestrator(
            plan=plan,
            providers=providers,
            executors=executors,
        ).run()
        assert result.status is ReconstructionPipelineRunStatus.COMPLETED
        assert result.canonical_geometry_authoritative is True
        assert result.certified_metrology is False
        assert all(len(executor.calls) == 1 for executor in executors.values())

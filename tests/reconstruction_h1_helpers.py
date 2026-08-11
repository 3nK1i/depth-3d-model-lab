from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.reconstruction import (
    DeterministicMockStageExecutor,
    ReconstructionOperationControl,
    ReconstructionPath,
    ReconstructionPipelineGoal,
    ReconstructionPipelineOrchestrator,
    ReconstructionPipelineRequest,
    ReconstructionPipelineRunResult,
    ReconstructionPipelineRunStatus,
    ReconstructionPipelineStage,
    ReconstructionProviderDescriptor,
    ReconstructionProviderExecution,
    ReconstructionProviderRole,
    ReconstructionQualityProfile,
    build_reconstruction_pipeline_plan,
    create_reconstruction_engine_checkpoint,
)

H1_SESSION_ID = "81818181-8181-4818-8818-818181818181"
H1_REQUEST_ID = "phase8-8h1-whole-workflow"


def _provider(provider_id: str, role: ReconstructionProviderRole) -> ReconstructionProviderDescriptor:
    return ReconstructionProviderDescriptor(
        provider_id=provider_id,
        provider_version="1",
        roles=(role,),
        execution=ReconstructionProviderExecution.DEVELOPMENT_MOCK,
        adapter_version="phase8.8h1-integration",
        priority=10,
    )


@dataclass(frozen=True, slots=True)
class H1PipelineFixture:
    providers: tuple[ReconstructionProviderDescriptor, ...]
    request: ReconstructionPipelineRequest
    plan: object
    executors: dict[str, DeterministicMockStageExecutor]


def whole_pipeline_fixture(*, session_id: str = H1_SESSION_ID) -> H1PipelineFixture:
    providers = (
        _provider("h1-ingestion", ReconstructionProviderRole.INGESTION),
        _provider("h1-pose", ReconstructionProviderRole.CAMERA_POSE),
        _provider("h1-depth", ReconstructionProviderRole.DEPTH),
        _provider("h1-dense", ReconstructionProviderRole.DENSE_GEOMETRY),
        _provider("h1-mesh", ReconstructionProviderRole.SURFACE_MESH),
        _provider("h1-semantics", ReconstructionProviderRole.SEMANTICS),
        _provider("h1-inspection", ReconstructionProviderRole.INSPECTION_QC),
        _provider("h1-gaussian", ReconstructionProviderRole.GAUSSIAN_APPEARANCE),
        _provider("h1-nerf", ReconstructionProviderRole.NERF_APPEARANCE),
    )
    request = ReconstructionPipelineRequest(
        request_id=H1_REQUEST_ID,
        session_id=session_id,
        path=ReconstructionPath.OFFLINE,
        quality_profile=ReconstructionQualityProfile.EFFICIENT,
        goals=(
            ReconstructionPipelineGoal.SEMANTIC_INSPECTION,
            ReconstructionPipelineGoal.GAUSSIAN_APPEARANCE,
            ReconstructionPipelineGoal.NERF_APPEARANCE,
        ),
        allow_optional_degradation=True,
    )
    plan = build_reconstruction_pipeline_plan(request, providers)
    executors = {
        item.provider_id: DeterministicMockStageExecutor(
            item.provider_id,
            payload_prefix="mock://phase8-8h1",
        )
        for item in providers
    }
    return H1PipelineFixture(providers, request, plan, executors)


def run_whole_pipeline(*, session_id: str = H1_SESSION_ID) -> tuple[H1PipelineFixture, ReconstructionPipelineRunResult]:
    fixture = whole_pipeline_fixture(session_id=session_id)
    result = ReconstructionPipelineOrchestrator(
        plan=fixture.plan,
        providers=fixture.providers,
        executors=fixture.executors,
    ).run()
    if result.status is not ReconstructionPipelineRunStatus.COMPLETED:
        raise RuntimeError(f"H1 deterministic whole pipeline did not complete: {result.status.value}")
    return fixture, result


def checkpoint_prefix(fixture: H1PipelineFixture):
    control = ReconstructionOperationControl()
    committed = []

    def hook(stages):
        committed.append(stages)
        if stages[-1].stage is ReconstructionPipelineStage.DEPTH_EVIDENCE:
            control.request_cancel()

    result = ReconstructionPipelineOrchestrator(
        plan=fixture.plan,
        providers=fixture.providers,
        executors={
            item.provider_id: DeterministicMockStageExecutor(
                item.provider_id,
                payload_prefix="mock://phase8-8h1-resume",
            )
            for item in fixture.providers
        },
    ).run(control=control, stage_commit_hook=hook)
    if result.status is not ReconstructionPipelineRunStatus.CANCELLED:
        raise RuntimeError("H1 checkpoint fixture must stop at a canonical cancelled prefix.")
    return committed[-1]


def create_checkpoint_fixture(fixture: H1PipelineFixture):
    prefix = checkpoint_prefix(fixture)
    checkpoint = create_reconstruction_engine_checkpoint(
        plan=fixture.plan,
        providers=fixture.providers,
        stage_results=prefix,
    )
    return checkpoint, prefix


def write_ready_image_sequence(root: Path) -> Path:
    source = (root / "h1-sequence").resolve()
    source.mkdir(parents=True, exist_ok=True)
    for index in range(3):
        (source / f"frame-{index:02d}.png").write_bytes(b"phase8.8h1")
    return source

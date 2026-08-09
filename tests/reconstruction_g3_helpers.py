from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from core.reconstruction import (
    DeterministicMockStageExecutor,
    ReconstructionEnduranceCycleEvidence,
    ReconstructionEnduranceScenario,
    ReconstructionPath,
    ReconstructionPipelineGoal,
    ReconstructionPipelineOrchestrator,
    ReconstructionPipelineRequest,
    ReconstructionProviderDescriptor,
    ReconstructionProviderExecution,
    ReconstructionProviderRole,
    ReconstructionQualityProfile,
    build_reconstruction_pipeline_plan,
    check_reconstruction_engine_resume_compatibility,
    create_reconstruction_engine_checkpoint,
    persistent_scene_from_pipeline_result,
)
from tests.reconstruction_f3_helpers import make_f3_fixture
from tests.reconstruction_f5_helpers import run_f5_fixture


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
    dense_id = next(p.provider_id for p in fixture.providers if ReconstructionProviderRole.DENSE_GEOMETRY in p.roles)
    executors[dense_id] = FailDense()
    request = ReconstructionPipelineRequest(
        request_id=f"g3-required-failure-{index}",
        session_id=fixture.session_id,
        path=ReconstructionPath.OFFLINE,
        quality_profile=ReconstructionQualityProfile.EFFICIENT,
        goals=(ReconstructionPipelineGoal.SEMANTIC_INSPECTION,),
    )
    plan = build_reconstruction_pipeline_plan(request, fixture.providers)
    result = ReconstructionPipelineOrchestrator(plan=plan, providers=fixture.providers, executors=executors).run()
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
    def provider(pid: str, role: ReconstructionProviderRole, version: str = "1"):
        return ReconstructionProviderDescriptor(
            provider_id=pid,
            provider_version=version,
            roles=(role,),
            execution=ReconstructionProviderExecution.DEVELOPMENT_MOCK,
            priority=10,
            adapter_version="g3-helper",
        )

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
        executors={p.provider_id: DeterministicMockStageExecutor(p.provider_id) for p in providers},
    ).run(stage_commit_hook=lambda stages: prefixes.append(stages))
    checkpoint = create_reconstruction_engine_checkpoint(plan=plan, providers=providers, stage_results=prefixes[2])
    drifted = list(providers)
    drifted[-1] = provider("mesh", ReconstructionProviderRole.SURFACE_MESH, version="g3-drift")
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

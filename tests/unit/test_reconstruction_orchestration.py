from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import pytest

from core.reconstruction import (
    RECONSTRUCTION_ORCHESTRATION_SCHEMA_VERSION,
    STAGE_CONTRACTS,
    BackendProvenance,
    DeterministicMockStageExecutor,
    ReconstructionArtifactAuthority,
    ReconstructionArtifactKind,
    ReconstructionOperationControl,
    ReconstructionPath,
    ReconstructionPipelineArtifact,
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
    ReconstructionStageRunState,
    ReconstructionValidationError,
    aggregate_pipeline_backend_provenance,
    build_reconstruction_pipeline_plan,
    reconstruction_stage_contract,
    required_pipeline_stages,
    write_reconstruction_pipeline_json,
)


def _request(*goals: ReconstructionPipelineGoal, allow_optional_degradation: bool = True):
    return ReconstructionPipelineRequest(
        request_id="phase8-7f1-test",
        session_id=str(uuid4()),
        path=ReconstructionPath.OFFLINE,
        quality_profile=ReconstructionQualityProfile.EFFICIENT,
        goals=tuple(goals),
        allow_optional_degradation=allow_optional_degradation,
    )


def _provider(
    provider_id: str,
    *roles: ReconstructionProviderRole,
    priority: int = 100,
    available: bool = True,
    execution: ReconstructionProviderExecution = ReconstructionProviderExecution.DEVELOPMENT_MOCK,
):
    return ReconstructionProviderDescriptor(
        provider_id=provider_id,
        provider_version="1.0",
        roles=tuple(roles),
        execution=execution,
        priority=priority,
        available=available,
        unavailable_reason="" if available else "not_available_for_test",
        adapter_version="adapter-1",
    )


def _canonical_providers(*, include_appearance: bool = False):
    providers = [
        _provider("ingestion", ReconstructionProviderRole.INGESTION),
        _provider("pose", ReconstructionProviderRole.CAMERA_POSE),
        _provider("depth", ReconstructionProviderRole.DEPTH),
        _provider("dense", ReconstructionProviderRole.DENSE_GEOMETRY),
        _provider("mesh", ReconstructionProviderRole.SURFACE_MESH),
        _provider("semantics", ReconstructionProviderRole.SEMANTICS),
        _provider("inspection", ReconstructionProviderRole.INSPECTION_QC),
    ]
    if include_appearance:
        providers.extend(
            [
                _provider("gaussian", ReconstructionProviderRole.GAUSSIAN_APPEARANCE),
                _provider("nerf", ReconstructionProviderRole.NERF_APPEARANCE),
            ]
        )
    return providers


def _executors(providers, *, failures: dict[str, int] | None = None):
    failures = failures or {}
    return {
        provider.provider_id: DeterministicMockStageExecutor(
            provider.provider_id,
            fail_count=failures.get(provider.provider_id, 0),
        )
        for provider in providers
        if provider.available
    }


def test_schema_and_stage_contract_graph_are_stable_and_one_to_one() -> None:
    assert RECONSTRUCTION_ORCHESTRATION_SCHEMA_VERSION == 1
    assert len(STAGE_CONTRACTS) == 9
    assert len({item.stage for item in STAGE_CONTRACTS}) == 9
    assert len({item.output_kind for item in STAGE_CONTRACTS}) == 9
    assert STAGE_CONTRACTS[0].stage is ReconstructionPipelineStage.SOURCE_INGESTION
    assert STAGE_CONTRACTS[-1].stage is ReconstructionPipelineStage.NERF_APPEARANCE


def test_canonical_goal_expands_only_through_surface_mesh() -> None:
    assert required_pipeline_stages((ReconstructionPipelineGoal.CANONICAL_GEOMETRY,)) == (
        ReconstructionPipelineStage.SOURCE_INGESTION,
        ReconstructionPipelineStage.CAMERA_POSE,
        ReconstructionPipelineStage.DEPTH_EVIDENCE,
        ReconstructionPipelineStage.DENSE_GEOMETRY,
        ReconstructionPipelineStage.SURFACE_MESH,
    )


def test_semantic_inspection_goal_extends_canonical_chain_without_neural_appearance() -> None:
    stages = required_pipeline_stages((ReconstructionPipelineGoal.SEMANTIC_INSPECTION,))
    assert stages[-2:] == (
        ReconstructionPipelineStage.SEMANTIC_GEOMETRY,
        ReconstructionPipelineStage.INSPECTION_QC,
    )
    assert ReconstructionPipelineStage.GAUSSIAN_APPEARANCE not in stages
    assert ReconstructionPipelineStage.NERF_APPEARANCE not in stages


def test_nerf_goal_does_not_require_dense_or_mesh_geometry() -> None:
    stages = required_pipeline_stages((ReconstructionPipelineGoal.NERF_APPEARANCE,))
    assert stages == (
        ReconstructionPipelineStage.SOURCE_INGESTION,
        ReconstructionPipelineStage.CAMERA_POSE,
        ReconstructionPipelineStage.NERF_APPEARANCE,
    )


def test_gaussian_goal_preserves_canonical_mesh_dependency() -> None:
    stages = required_pipeline_stages((ReconstructionPipelineGoal.GAUSSIAN_APPEARANCE,))
    assert ReconstructionPipelineStage.SURFACE_MESH in stages
    assert stages[-1] is ReconstructionPipelineStage.GAUSSIAN_APPEARANCE


def test_neural_stage_authority_is_appearance_only() -> None:
    assert reconstruction_stage_contract(
        ReconstructionPipelineStage.GAUSSIAN_APPEARANCE
    ).output_authority is ReconstructionArtifactAuthority.APPEARANCE_ONLY
    assert reconstruction_stage_contract(
        ReconstructionPipelineStage.NERF_APPEARANCE
    ).output_authority is ReconstructionArtifactAuthority.APPEARANCE_ONLY


def test_artifact_rejects_authority_escalation() -> None:
    request = _request(ReconstructionPipelineGoal.NERF_APPEARANCE)
    with pytest.raises(ReconstructionValidationError):
        ReconstructionPipelineArtifact(
            artifact_id="bad",
            session_id=request.session_id,
            stage=ReconstructionPipelineStage.NERF_APPEARANCE,
            kind=ReconstructionArtifactKind.NERF_SCENE,
            authority=ReconstructionArtifactAuthority.CANONICAL_GEOMETRY,
            provider_id="nerf",
        )


def test_provider_can_declare_multiple_generic_roles_for_future_provider_integration() -> None:
    provider = _provider(
        "multi-role-provider",
        ReconstructionProviderRole.CAMERA_POSE,
        ReconstructionProviderRole.DEPTH,
    )
    assert provider.supports(ReconstructionProviderRole.CAMERA_POSE)
    assert provider.supports(ReconstructionProviderRole.DEPTH)
    assert not provider.supports(ReconstructionProviderRole.NERF_APPEARANCE)


def test_provider_validation_rejects_duplicate_roles_and_bad_availability_reason() -> None:
    with pytest.raises(ReconstructionValidationError):
        _provider("duplicate", ReconstructionProviderRole.DEPTH, ReconstructionProviderRole.DEPTH)
    with pytest.raises(ReconstructionValidationError):
        ReconstructionProviderDescriptor(
            provider_id="bad",
            provider_version="1",
            roles=(ReconstructionProviderRole.DEPTH,),
            execution=ReconstructionProviderExecution.IN_PROCESS,
            available=False,
        )


def test_plan_uses_deterministic_fallback_when_higher_priority_provider_is_unavailable() -> None:
    providers = _canonical_providers()
    providers = [
        *providers,
        _provider("depth-preferred", ReconstructionProviderRole.DEPTH, priority=1, available=False),
    ]
    request = _request(ReconstructionPipelineGoal.CANONICAL_GEOMETRY)
    plan = build_reconstruction_pipeline_plan(request, providers)
    depth_stage = next(item for item in plan.stages if item.stage is ReconstructionPipelineStage.DEPTH_EVIDENCE)
    assert depth_stage.preferred_provider_id == "depth-preferred"
    assert depth_stage.candidate_provider_ids == ("depth",)
    assert depth_stage.unavailable_provider_ids == ("depth-preferred",)
    assert depth_stage.fallback_planned is True


def test_plan_fails_closed_when_required_provider_role_is_missing() -> None:
    providers = [p for p in _canonical_providers() if not p.supports(ReconstructionProviderRole.SURFACE_MESH)]
    with pytest.raises(ReconstructionValidationError, match="surface_mesh"):
        build_reconstruction_pipeline_plan(
            _request(ReconstructionPipelineGoal.CANONICAL_GEOMETRY), providers
        )


def test_optional_appearance_stage_can_be_planned_as_skipped() -> None:
    providers = _canonical_providers()
    request = _request(
        ReconstructionPipelineGoal.CANONICAL_GEOMETRY,
        ReconstructionPipelineGoal.NERF_APPEARANCE,
    )
    plan = build_reconstruction_pipeline_plan(request, providers)
    nerf = next(stage for stage in plan.stages if stage.stage is ReconstructionPipelineStage.NERF_APPEARANCE)
    assert nerf.required is False
    assert nerf.candidate_provider_ids == ()
    assert nerf.skip_reason == "optional_provider_unavailable"


def test_requested_appearance_becomes_required_when_degradation_is_disabled() -> None:
    with pytest.raises(ReconstructionValidationError, match="nerf_appearance"):
        build_reconstruction_pipeline_plan(
            _request(
                ReconstructionPipelineGoal.NERF_APPEARANCE,
                allow_optional_degradation=False,
            ),
            [
                _provider("ingestion", ReconstructionProviderRole.INGESTION),
                _provider("pose", ReconstructionProviderRole.CAMERA_POSE),
            ],
        )


def test_full_development_mock_pipeline_completes_with_canonical_authority_and_no_metrology_claim() -> None:
    providers = _canonical_providers(include_appearance=True)
    request = _request(
        ReconstructionPipelineGoal.SEMANTIC_INSPECTION,
        ReconstructionPipelineGoal.GAUSSIAN_APPEARANCE,
        ReconstructionPipelineGoal.NERF_APPEARANCE,
    )
    plan = build_reconstruction_pipeline_plan(request, providers)
    result = ReconstructionPipelineOrchestrator(
        plan=plan, providers=providers, executors=_executors(providers)
    ).run()
    assert result.status is ReconstructionPipelineRunStatus.COMPLETED
    assert result.execution_scope is ReconstructionPipelineExecutionScope.DEVELOPMENT_MOCK
    assert result.development_mock_used is True
    assert result.canonical_geometry_authoritative is True
    assert result.canonical_geometry_artifact_id.endswith(":surface_mesh:mesh")
    assert result.certified_metrology is False
    assert result.reconstruction_ui_enabled is False
    appearance = [
        stage.artifact for stage in result.stages if stage.stage in {
            ReconstructionPipelineStage.GAUSSIAN_APPEARANCE,
            ReconstructionPipelineStage.NERF_APPEARANCE,
        }
    ]
    assert all(item is not None and item.authority is ReconstructionArtifactAuthority.APPEARANCE_ONLY for item in appearance)


def test_planned_fallback_marks_result_degraded_but_preserves_canonical_geometry() -> None:
    providers = _canonical_providers()
    providers.append(_provider("depth-preferred", ReconstructionProviderRole.DEPTH, priority=1, available=False))
    request = _request(ReconstructionPipelineGoal.CANONICAL_GEOMETRY)
    plan = build_reconstruction_pipeline_plan(request, providers)
    result = ReconstructionPipelineOrchestrator(
        plan=plan, providers=providers, executors=_executors(providers)
    ).run()
    depth = next(item for item in result.stages if item.stage is ReconstructionPipelineStage.DEPTH_EVIDENCE)
    assert result.status is ReconstructionPipelineRunStatus.DEGRADED
    assert depth.state is ReconstructionStageRunState.COMPLETED
    assert depth.provider_id == "depth"
    assert depth.fallback_used is True
    assert result.canonical_geometry_authoritative is True


def test_runtime_provider_failure_uses_next_candidate_and_records_attempts() -> None:
    providers = _canonical_providers()
    providers = [p for p in providers if p.provider_id != "depth"] + [
        _provider("depth-a", ReconstructionProviderRole.DEPTH, priority=1),
        _provider("depth-b", ReconstructionProviderRole.DEPTH, priority=2),
    ]
    request = _request(ReconstructionPipelineGoal.CANONICAL_GEOMETRY)
    plan = build_reconstruction_pipeline_plan(request, providers)
    result = ReconstructionPipelineOrchestrator(
        plan=plan,
        providers=providers,
        executors=_executors(providers, failures={"depth-a": 1}),
    ).run()
    depth = next(item for item in result.stages if item.stage is ReconstructionPipelineStage.DEPTH_EVIDENCE)
    assert result.status is ReconstructionPipelineRunStatus.DEGRADED
    assert depth.provider_attempts == ("depth-a", "depth-b")
    assert depth.provider_id == "depth-b"
    assert depth.fallback_used is True


def test_optional_appearance_runtime_failure_degrades_without_invalidating_canonical_geometry() -> None:
    providers = _canonical_providers(include_appearance=True)
    request = _request(
        ReconstructionPipelineGoal.CANONICAL_GEOMETRY,
        ReconstructionPipelineGoal.NERF_APPEARANCE,
    )
    plan = build_reconstruction_pipeline_plan(request, providers)
    result = ReconstructionPipelineOrchestrator(
        plan=plan,
        providers=providers,
        executors=_executors(providers, failures={"nerf": 1}),
    ).run()
    nerf = next(item for item in result.stages if item.stage is ReconstructionPipelineStage.NERF_APPEARANCE)
    assert result.status is ReconstructionPipelineRunStatus.DEGRADED
    assert nerf.state is ReconstructionStageRunState.SKIPPED
    assert result.canonical_geometry_authoritative is True


def test_required_provider_failure_fails_closed_and_stops_downstream_execution() -> None:
    providers = _canonical_providers()
    request = _request(ReconstructionPipelineGoal.SEMANTIC_INSPECTION)
    plan = build_reconstruction_pipeline_plan(request, providers)
    executors = _executors(providers, failures={"dense": 1})
    result = ReconstructionPipelineOrchestrator(
        plan=plan, providers=providers, executors=executors
    ).run()
    assert result.status is ReconstructionPipelineRunStatus.FAILED
    assert result.stages[-1].stage is ReconstructionPipelineStage.DENSE_GEOMETRY
    assert result.stages[-1].state is ReconstructionStageRunState.FAILED
    assert executors["mesh"].calls == []
    assert executors["semantics"].calls == []


def test_missing_executor_is_treated_as_provider_failure_not_backend_import() -> None:
    providers = _canonical_providers()
    request = _request(ReconstructionPipelineGoal.CANONICAL_GEOMETRY)
    plan = build_reconstruction_pipeline_plan(request, providers)
    executors = _executors(providers)
    executors.pop("mesh")
    result = ReconstructionPipelineOrchestrator(plan=plan, providers=providers, executors=executors).run()
    assert result.status is ReconstructionPipelineRunStatus.FAILED
    assert result.stages[-1].stage is ReconstructionPipelineStage.SURFACE_MESH
    assert "executor_missing" in result.stages[-1].reason


def test_pre_requested_cancellation_stops_before_any_provider_executes() -> None:
    providers = _canonical_providers()
    request = _request(ReconstructionPipelineGoal.CANONICAL_GEOMETRY)
    plan = build_reconstruction_pipeline_plan(request, providers)
    executors = _executors(providers)
    control = ReconstructionOperationControl()
    control.request_cancel()
    result = ReconstructionPipelineOrchestrator(
        plan=plan, providers=providers, executors=executors
    ).run(control=control)
    assert result.status is ReconstructionPipelineRunStatus.CANCELLED
    assert result.stages[0].state is ReconstructionStageRunState.CANCELLED
    assert all(executor.calls == [] for executor in executors.values())


def test_provider_provenance_aggregates_only_successful_artifact_producers() -> None:
    providers = _canonical_providers(include_appearance=True)
    request = _request(ReconstructionPipelineGoal.CANONICAL_GEOMETRY)
    plan = build_reconstruction_pipeline_plan(request, providers)
    result = ReconstructionPipelineOrchestrator(
        plan=plan, providers=providers, executors=_executors(providers)
    ).run()
    backend_provenance = aggregate_pipeline_backend_provenance(result, providers)
    assert all(isinstance(item, BackendProvenance) for item in backend_provenance)
    ids = {item.backend_id for item in backend_provenance}
    assert ids == {"ingestion", "pose", "depth", "dense", "mesh"}
    assert "gaussian" not in ids and "nerf" not in ids


def test_pipeline_json_serialization_is_explicit_and_records_future_phase_ownership(tmp_path: Path) -> None:
    providers = _canonical_providers()
    request = _request(ReconstructionPipelineGoal.CANONICAL_GEOMETRY)
    plan = build_reconstruction_pipeline_plan(request, providers)
    result = ReconstructionPipelineOrchestrator(
        plan=plan, providers=providers, executors=_executors(providers)
    ).run()
    target = write_reconstruction_pipeline_json(result, tmp_path / "result.json")
    payload = json.loads(target.read_text())
    assert payload["canonical_geometry_authoritative"] is True
    assert payload["development_mock_used"] is True
    assert payload["production_execution_claimed"] is False
    assert payload["future_phase_ownership"]["depth_anything_3_provider"] == "8.7F2"
    assert payload["future_phase_ownership"]["persistent_reconstruction_scene"] == "8.7F5"
    assert payload["future_phase_ownership"]["ui"] == "8.8"


def test_pipeline_json_requires_absolute_path(tmp_path: Path) -> None:
    providers = _canonical_providers()
    plan = build_reconstruction_pipeline_plan(
        _request(ReconstructionPipelineGoal.CANONICAL_GEOMETRY), providers
    )
    with pytest.raises(ReconstructionValidationError):
        write_reconstruction_pipeline_json(plan, Path("relative.json"))


def test_artifact_stage_kind_mapping_cannot_be_relabeled() -> None:
    request = _request(ReconstructionPipelineGoal.CANONICAL_GEOMETRY)
    mesh_contract = reconstruction_stage_contract(ReconstructionPipelineStage.SURFACE_MESH)
    artifact = ReconstructionPipelineArtifact(
        artifact_id="mesh",
        session_id=request.session_id,
        stage=mesh_contract.stage,
        kind=mesh_contract.output_kind,
        authority=mesh_contract.output_authority,
        provider_id="mesh",
    )
    with pytest.raises(ReconstructionValidationError):
        replace(artifact, kind=ReconstructionArtifactKind.NERF_SCENE)

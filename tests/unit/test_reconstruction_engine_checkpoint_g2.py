from __future__ import annotations

import inspect
from pathlib import Path
from uuid import uuid4

import pytest

from core.reconstruction import (
    ENGINE_CHECKPOINT_PHASE,
    ENGINE_CHECKPOINT_SCHEMA_VERSION,
    DeterministicMockStageExecutor,
    PipelinePayloadStore,
    ReconstructionEngineCheckpoint,
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
    ReconstructionResumeCompatibilityStatus,
    ReconstructionStageRunState,
    ReconstructionValidationError,
    build_reconstruction_pipeline_plan,
    check_reconstruction_engine_resume_compatibility,
    create_reconstruction_engine_checkpoint,
    read_reconstruction_engine_checkpoint,
    write_reconstruction_engine_checkpoint,
)


def _provider(provider_id: str, role: ReconstructionProviderRole):
    return ReconstructionProviderDescriptor(
        provider_id=provider_id,
        provider_version="1",
        roles=(role,),
        execution=ReconstructionProviderExecution.DEVELOPMENT_MOCK,
        priority=10,
        adapter_version="g2-test",
    )


def _fixture():
    providers = (
        _provider("ingestion", ReconstructionProviderRole.INGESTION),
        _provider("pose", ReconstructionProviderRole.CAMERA_POSE),
        _provider("depth", ReconstructionProviderRole.DEPTH),
        _provider("dense", ReconstructionProviderRole.DENSE_GEOMETRY),
        _provider("mesh", ReconstructionProviderRole.SURFACE_MESH),
    )
    request = ReconstructionPipelineRequest(
        request_id="phase8-g2-unit",
        session_id=str(uuid4()),
        path=ReconstructionPath.OFFLINE,
        quality_profile=ReconstructionQualityProfile.EFFICIENT,
        goals=(ReconstructionPipelineGoal.CANONICAL_GEOMETRY,),
    )
    plan = build_reconstruction_pipeline_plan(request, providers)
    executors = {p.provider_id: DeterministicMockStageExecutor(p.provider_id) for p in providers}
    return providers, request, plan, executors


def _checkpoint_after_depth():
    providers, request, plan, executors = _fixture()
    control = ReconstructionOperationControl()
    committed = []

    def hook(stages):
        committed.append(stages)
        if stages[-1].stage is ReconstructionPipelineStage.DEPTH_EVIDENCE:
            control.request_cancel()

    result = ReconstructionPipelineOrchestrator(plan=plan, providers=providers, executors=executors).run(
        control=control, stage_commit_hook=hook
    )
    assert result.status is ReconstructionPipelineRunStatus.CANCELLED
    prefix = committed[-1]
    checkpoint = create_reconstruction_engine_checkpoint(plan=plan, providers=providers, stage_results=prefix)
    return providers, request, plan, executors, checkpoint


def test_g2_schema_and_phase_are_explicit():
    assert ENGINE_CHECKPOINT_SCHEMA_VERSION == 1
    assert ENGINE_CHECKPOINT_PHASE == "8.7G2"


def test_stage_commit_hook_observes_durable_prefixes():
    providers, _request, plan, executors = _fixture()
    lengths = []
    ReconstructionPipelineOrchestrator(plan=plan, providers=providers, executors=executors).run(
        stage_commit_hook=lambda stages: lengths.append(len(stages))
    )
    assert lengths == [1, 2, 3, 4, 5]


def test_checkpoint_captures_exact_prefix_identity():
    providers, request, plan, _executors, checkpoint = _checkpoint_after_depth()
    assert checkpoint.request_id == request.request_id
    assert checkpoint.session_id == request.session_id
    assert checkpoint.completed_stage_count == 3
    assert checkpoint.last_committed_stage is ReconstructionPipelineStage.DEPTH_EVIDENCE
    assert checkpoint.next_stage is ReconstructionPipelineStage.DENSE_GEOMETRY
    assert len(checkpoint.payload_refs) == 3
    assert len(checkpoint.checkpoint_fingerprint_sha256) == 64


def test_checkpoint_roundtrip_verifies_manifest_sha(tmp_path: Path):
    _providers, _request, _plan, _executors, checkpoint = _checkpoint_after_depth()
    root = (tmp_path / "checkpoint").resolve()
    manifest = write_reconstruction_engine_checkpoint(checkpoint, root)
    loaded = read_reconstruction_engine_checkpoint(root)
    assert loaded.checkpoint_id == checkpoint.checkpoint_id
    assert loaded.checkpoint_fingerprint_sha256 == checkpoint.checkpoint_fingerprint_sha256
    assert manifest.is_file()
    assert (root / "reconstruction_engine_checkpoint.json.sha256").is_file()


def test_checkpoint_manifest_tampering_is_rejected(tmp_path: Path):
    _providers, _request, _plan, _executors, checkpoint = _checkpoint_after_depth()
    root = (tmp_path / "checkpoint").resolve()
    manifest = write_reconstruction_engine_checkpoint(checkpoint, root)
    manifest.write_text(manifest.read_text() + " ", encoding="utf-8")
    with pytest.raises(ReconstructionValidationError, match="SHA-256"):
        read_reconstruction_engine_checkpoint(root)


def test_resume_requires_all_payload_refs_rehydrated():
    providers, _request, plan, _executors, checkpoint = _checkpoint_after_depth()
    result = check_reconstruction_engine_resume_compatibility(
        checkpoint, plan=plan, providers=providers, available_payload_refs=checkpoint.payload_refs[:-1]
    )
    assert result.status is ReconstructionResumeCompatibilityStatus.INCOMPATIBLE
    assert any("not rehydrated" in reason for reason in result.reasons)


def test_resume_rejects_provider_version_drift():
    providers, _request, plan, _executors, checkpoint = _checkpoint_after_depth()
    changed = list(providers)
    p = changed[-1]
    changed[-1] = ReconstructionProviderDescriptor(
        provider_id=p.provider_id,
        provider_version="2",
        roles=p.roles,
        execution=p.execution,
        priority=p.priority,
        adapter_version=p.adapter_version,
    )
    result = check_reconstruction_engine_resume_compatibility(
        checkpoint, plan=plan, providers=tuple(changed), available_payload_refs=checkpoint.payload_refs
    )
    assert not result.compatible
    assert any("Provider versions" in reason for reason in result.reasons)


def test_compatible_resume_reuses_prefix_without_reexecuting_it():
    providers, request, plan, _executors, checkpoint = _checkpoint_after_depth()
    compatibility = check_reconstruction_engine_resume_compatibility(
        checkpoint, plan=plan, providers=providers, available_payload_refs=checkpoint.payload_refs
    )
    compatibility.require_compatible()
    executors = {p.provider_id: DeterministicMockStageExecutor(p.provider_id) for p in providers}
    result = ReconstructionPipelineOrchestrator(plan=plan, providers=providers, executors=executors).run(
        resume_stage_results=checkpoint.stage_results
    )
    assert result.status is ReconstructionPipelineRunStatus.COMPLETED
    assert executors["ingestion"].calls == []
    assert executors["pose"].calls == []
    assert executors["depth"].calls == []
    assert executors["dense"].calls == ["dense_geometry"]
    assert executors["mesh"].calls == ["surface_mesh"]
    assert result.request_id == request.request_id


def test_orchestrator_rejects_non_prefix_resume_results():
    providers, _request, plan, executors, checkpoint = _checkpoint_after_depth()
    bad = (checkpoint.stage_results[1],)
    with pytest.raises(ReconstructionValidationError, match="canonical plan prefix"):
        ReconstructionPipelineOrchestrator(plan=plan, providers=providers, executors=executors).run(
            resume_stage_results=bad
        )


def test_payload_store_exposes_stable_registered_refs():
    store = PipelinePayloadStore()
    store.put({"a": 1}, payload_ref="memory://a")
    store.put([1, 2], payload_ref="memory://b")
    assert store.contains_ref("memory://a")
    assert store.payload_refs == ("memory://a", "memory://b")


def test_checkpoint_output_dir_must_be_absolute(tmp_path: Path):
    _providers, _request, _plan, _executors, checkpoint = _checkpoint_after_depth()
    with pytest.raises(ReconstructionValidationError, match="absolute"):
        write_reconstruction_engine_checkpoint(checkpoint, Path("relative-checkpoint"))


def test_g2_orchestrator_signature_is_backward_compatible():
    signature = inspect.signature(ReconstructionPipelineOrchestrator.run)
    assert signature.parameters["control"].default is None
    assert tuple(signature.parameters["resume_stage_results"].default) == ()
    assert signature.parameters["stage_commit_hook"].default is None

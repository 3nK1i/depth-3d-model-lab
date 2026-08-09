from __future__ import annotations

from pathlib import Path

from core.reconstruction import (
    DeterministicMockStageExecutor,
    ReconstructionOperationControl,
    ReconstructionPath,
    ReconstructionPipelineGoal,
    ReconstructionPipelineOrchestrator,
    ReconstructionPipelineRequest,
    ReconstructionPipelineStage,
    ReconstructionProviderDescriptor,
    ReconstructionProviderExecution,
    ReconstructionProviderRole,
    ReconstructionQualityProfile,
    build_reconstruction_pipeline_plan,
    create_reconstruction_engine_checkpoint,
    read_reconstruction_engine_checkpoint,
    write_reconstruction_engine_checkpoint,
)
from uuid import uuid4


def _provider(pid, role):
    return ReconstructionProviderDescriptor(
        provider_id=pid, provider_version="1", roles=(role,),
        execution=ReconstructionProviderExecution.DEVELOPMENT_MOCK,
        priority=10, adapter_version="g2-stress",
    )


def _prefix():
    providers = (
        _provider("ingestion", ReconstructionProviderRole.INGESTION),
        _provider("pose", ReconstructionProviderRole.CAMERA_POSE),
        _provider("depth", ReconstructionProviderRole.DEPTH),
        _provider("dense", ReconstructionProviderRole.DENSE_GEOMETRY),
        _provider("mesh", ReconstructionProviderRole.SURFACE_MESH),
    )
    request = ReconstructionPipelineRequest(
        request_id="phase8-g2-stress", session_id=str(uuid4()), path=ReconstructionPath.OFFLINE,
        quality_profile=ReconstructionQualityProfile.EFFICIENT,
        goals=(ReconstructionPipelineGoal.CANONICAL_GEOMETRY,),
    )
    plan = build_reconstruction_pipeline_plan(request, providers)
    executors = {p.provider_id: DeterministicMockStageExecutor(p.provider_id) for p in providers}
    control = ReconstructionOperationControl(); holder = {}
    def hook(stages):
        if stages[-1].stage is ReconstructionPipelineStage.DEPTH_EVIDENCE:
            holder["stages"] = stages
            control.request_cancel()
    ReconstructionPipelineOrchestrator(plan=plan, providers=providers, executors=executors).run(
        control=control, stage_commit_hook=hook
    )
    return providers, plan, holder["stages"]


def test_atomic_checkpoint_rewrite_remains_readable_across_many_updates(tmp_path: Path):
    providers, plan, stages = _prefix()
    root = (tmp_path / "checkpoint").resolve()
    checkpoint_id = str(uuid4())
    last = None
    for _ in range(75):
        checkpoint = create_reconstruction_engine_checkpoint(
            plan=plan, providers=providers, stage_results=stages, checkpoint_id=checkpoint_id
        )
        write_reconstruction_engine_checkpoint(checkpoint, root)
        loaded = read_reconstruction_engine_checkpoint(root)
        assert loaded.checkpoint_id == checkpoint_id
        assert loaded.completed_stage_count == 3
        last = loaded.checkpoint_fingerprint_sha256
    assert isinstance(last, str) and len(last) == 64


def test_checkpoint_sidecar_missing_fails_closed(tmp_path: Path):
    providers, plan, stages = _prefix()
    root = (tmp_path / "checkpoint").resolve()
    write_reconstruction_engine_checkpoint(
        create_reconstruction_engine_checkpoint(plan=plan, providers=providers, stage_results=stages), root
    )
    (root / "reconstruction_engine_checkpoint.json.sha256").unlink()
    try:
        read_reconstruction_engine_checkpoint(root)
    except Exception as exc:
        assert "checksum" in str(exc).lower()
    else:
        raise AssertionError("missing checkpoint checksum must fail closed")

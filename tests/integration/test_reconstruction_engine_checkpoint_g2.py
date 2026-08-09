from __future__ import annotations

from pathlib import Path

from core.reconstruction import (
    ReconstructionOperationControl,
    ReconstructionPipelineOrchestrator,
    ReconstructionPipelineRunStatus,
    ReconstructionPipelineStage,
    check_reconstruction_engine_resume_compatibility,
    create_reconstruction_engine_checkpoint,
    read_reconstruction_engine_checkpoint,
    write_reconstruction_engine_checkpoint,
)
from tests.reconstruction_f3_helpers import make_f3_fixture


def test_f3_pipeline_can_resume_from_persisted_depth_prefix_without_reexecuting_bound_evidence(tmp_path: Path):
    fixture = make_f3_fixture(request_id="phase8-g2-integration")
    control = ReconstructionOperationControl()
    checkpoint_root = (tmp_path / "checkpoint").resolve()
    saved = {}

    def hook(stages):
        if stages[-1].stage is ReconstructionPipelineStage.DEPTH_EVIDENCE:
            checkpoint = create_reconstruction_engine_checkpoint(
                plan=fixture.plan, providers=fixture.providers, stage_results=stages
            )
            write_reconstruction_engine_checkpoint(checkpoint, checkpoint_root)
            saved["id"] = checkpoint.checkpoint_id
            control.request_cancel()

    interrupted = fixture.orchestrator.run(control=control, stage_commit_hook=hook)
    assert interrupted.status is ReconstructionPipelineRunStatus.CANCELLED

    fresh = make_f3_fixture(
        session_id=fixture.session_id,
        request_id="phase8-g2-integration",
    )
    checkpoint = read_reconstruction_engine_checkpoint(checkpoint_root)
    assert checkpoint.checkpoint_id == saved["id"]
    compatibility = check_reconstruction_engine_resume_compatibility(
        checkpoint,
        plan=fresh.plan,
        providers=fresh.providers,
        available_payload_refs=fresh.store.payload_refs,
    )
    compatibility.require_compatible()
    resumed = fresh.orchestrator.run(resume_stage_results=checkpoint.stage_results)
    assert resumed.status is ReconstructionPipelineRunStatus.COMPLETED
    assert resumed.canonical_geometry_authoritative is True
    assert len(resumed.stages) == len(fresh.plan.stages)
    assert resumed.stages[0:3] == checkpoint.stage_results

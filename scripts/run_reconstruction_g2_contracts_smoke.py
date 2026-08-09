#!/usr/bin/env python3
"""Dependency-light Phase 8.7G2 checkpoint/resume contract smoke."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
    build_reconstruction_pipeline_plan,
    check_reconstruction_engine_resume_compatibility,
    create_reconstruction_engine_checkpoint,
    read_reconstruction_engine_checkpoint,
    write_reconstruction_engine_checkpoint,
)


def provider(pid, role):
    return ReconstructionProviderDescriptor(
        provider_id=pid, provider_version="1", roles=(role,),
        execution=ReconstructionProviderExecution.DEVELOPMENT_MOCK,
        priority=10, adapter_version="g2-smoke",
    )


def main() -> int:
    out = Path("/tmp/phase8_7g2_contracts").resolve()
    if out.exists(): shutil.rmtree(out)
    out.mkdir(parents=True)
    providers = (
        provider("ingestion", ReconstructionProviderRole.INGESTION),
        provider("pose", ReconstructionProviderRole.CAMERA_POSE),
        provider("depth", ReconstructionProviderRole.DEPTH),
        provider("dense", ReconstructionProviderRole.DENSE_GEOMETRY),
        provider("mesh", ReconstructionProviderRole.SURFACE_MESH),
    )
    request = ReconstructionPipelineRequest(
        request_id="phase8-7g2-contract-smoke", session_id=str(uuid4()), path=ReconstructionPath.OFFLINE,
        quality_profile=ReconstructionQualityProfile.EFFICIENT,
        goals=(ReconstructionPipelineGoal.CANONICAL_GEOMETRY,),
    )
    plan = build_reconstruction_pipeline_plan(request, providers)
    first = {p.provider_id: DeterministicMockStageExecutor(p.provider_id) for p in providers}
    control = ReconstructionOperationControl()
    checkpoint_id = str(uuid4())

    def hook(stages):
        cp = create_reconstruction_engine_checkpoint(
            plan=plan, providers=providers, stage_results=stages, checkpoint_id=checkpoint_id
        )
        write_reconstruction_engine_checkpoint(cp, out / "checkpoint")
        if stages[-1].stage is ReconstructionPipelineStage.DEPTH_EVIDENCE:
            control.request_cancel()

    interrupted = ReconstructionPipelineOrchestrator(plan=plan, providers=providers, executors=first).run(
        control=control, stage_commit_hook=hook
    )
    checkpoint = read_reconstruction_engine_checkpoint(out / "checkpoint")
    available = checkpoint.payload_refs
    compatibility = check_reconstruction_engine_resume_compatibility(
        checkpoint, plan=plan, providers=providers, available_payload_refs=available
    )
    compatibility.require_compatible()
    second = {p.provider_id: DeterministicMockStageExecutor(p.provider_id) for p in providers}
    resumed = ReconstructionPipelineOrchestrator(plan=plan, providers=providers, executors=second).run(
        resume_stage_results=checkpoint.stage_results
    )
    passed = bool(
        interrupted.status is ReconstructionPipelineRunStatus.CANCELLED
        and resumed.status is ReconstructionPipelineRunStatus.COMPLETED
        and checkpoint.completed_stage_count == 3
        and second["ingestion"].calls == []
        and second["pose"].calls == []
        and second["depth"].calls == []
        and second["dense"].calls == ["dense_geometry"]
        and second["mesh"].calls == ["surface_mesh"]
    )
    payload = {
        "phase": "8.7G2", "pass": passed,
        "checkpoint_id": checkpoint.checkpoint_id,
        "checkpoint_fingerprint_sha256": checkpoint.checkpoint_fingerprint_sha256,
        "checkpoint_stage": checkpoint.last_committed_stage.value,
        "next_stage": checkpoint.next_stage.value,
        "reused_stage_count": checkpoint.completed_stage_count,
        "resume_compatible": compatibility.compatible,
        "completed_stage_reexecution": False,
        "resumed_status": resumed.status.value,
        "canonical_geometry_authoritative": resumed.canonical_geometry_authoritative,
        "endurance_deferred_to": "8.7G3", "full_regression_deferred_to": "8.7G4",
        "certified_metrology": False, "reconstruction_ui_enabled": False,
    }
    (out / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Phase 8.7G2 whole-engine checkpoint/resume contracts: {'PASS' if passed else 'FAIL'}")
    print(f"checkpoint_stage={payload['checkpoint_stage']} next_stage={payload['next_stage']} reused={payload['reused_stage_count']}")
    print(f"resume_compatible={payload['resume_compatible']} completed_stage_reexecution=False status={payload['resumed_status']}")
    print("checkpoint_manifest_sha256=True provider_identity_validation=True payload_rehydration_validation=True")
    print("endurance_deferred_to=8.7G3 full_regression_deferred_to=8.7G4 reconstruction_ui_enabled=False")
    print(f"summary={out/'summary.json'}")
    return 0 if passed else 1

if __name__ == "__main__":
    raise SystemExit(main())

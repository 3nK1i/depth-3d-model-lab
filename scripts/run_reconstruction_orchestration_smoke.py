#!/usr/bin/env python3
"""Deterministic non-UI Phase 8.7F1 orchestration foundation smoke."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction import (  # noqa: E402
    DeterministicMockStageExecutor,
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
    write_reconstruction_pipeline_json,
)

OUT = Path("/tmp/phase8_7f1_orchestration")


def provider(provider_id: str, role: ReconstructionProviderRole, *, priority: int = 100, available: bool = True):
    return ReconstructionProviderDescriptor(
        provider_id=provider_id,
        provider_version="phase8-7f1-mock-1",
        roles=(role,),
        execution=ReconstructionProviderExecution.DEVELOPMENT_MOCK,
        priority=priority,
        available=available,
        unavailable_reason="" if available else "deliberate_smoke_unavailable",
        adapter_version="f1-mock-adapter-1",
    )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    providers = [
        provider("mock-ingestion", ReconstructionProviderRole.INGESTION),
        provider("mock-pose", ReconstructionProviderRole.CAMERA_POSE),
        provider("mock-depth-preferred", ReconstructionProviderRole.DEPTH, priority=1, available=False),
        provider("mock-depth-fallback", ReconstructionProviderRole.DEPTH, priority=2),
        provider("mock-dense", ReconstructionProviderRole.DENSE_GEOMETRY),
        provider("mock-mesh", ReconstructionProviderRole.SURFACE_MESH),
        provider("mock-semantics", ReconstructionProviderRole.SEMANTICS),
        provider("mock-inspection", ReconstructionProviderRole.INSPECTION_QC),
        provider("mock-gaussian", ReconstructionProviderRole.GAUSSIAN_APPEARANCE),
        provider("mock-nerf", ReconstructionProviderRole.NERF_APPEARANCE),
    ]
    request = ReconstructionPipelineRequest(
        request_id="phase8-7f1-smoke",
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
        item.provider_id: DeterministicMockStageExecutor(item.provider_id)
        for item in providers
        if item.available
    }
    result = ReconstructionPipelineOrchestrator(
        plan=plan,
        providers=providers,
        executors=executors,
    ).run()

    depth = next(item for item in result.stages if item.stage is ReconstructionPipelineStage.DEPTH_EVIDENCE)
    gaussian = next(item for item in result.stages if item.stage is ReconstructionPipelineStage.GAUSSIAN_APPEARANCE)
    nerf = next(item for item in result.stages if item.stage is ReconstructionPipelineStage.NERF_APPEARANCE)

    assertions = {
        "fallback_recorded": depth.fallback_used and depth.provider_id == "mock-depth-fallback",
        "canonical_geometry_authoritative": result.canonical_geometry_authoritative,
        "gaussian_appearance_only": gaussian.artifact is not None and gaussian.artifact.authority.value == "appearance_only",
        "nerf_appearance_only": nerf.artifact is not None and nerf.artifact.authority.value == "appearance_only",
        "development_mock_used": result.development_mock_used,
        "no_production_claim": result.as_dict()["production_execution_claimed"] is False,
        "certified_metrology_false": result.certified_metrology is False,
        "ui_disabled": result.reconstruction_ui_enabled is False,
        "fallback_degradation_visible": result.status is ReconstructionPipelineRunStatus.DEGRADED,
    }
    if not all(assertions.values()):
        raise RuntimeError(f"F1 smoke assertions failed: {assertions}")

    plan_path = write_reconstruction_pipeline_json(plan, OUT / "plan.json")
    result_path = write_reconstruction_pipeline_json(result, OUT / "result.json")
    summary = {
        "phase": "8.7F1",
        "pass": True,
        "request_id": request.request_id,
        "status": result.status.value,
        "stage_count": len(result.stages),
        "provider_count": len(result.provider_provenance),
        "fallback_stage": depth.stage.value,
        "fallback_provider": depth.provider_id,
        "canonical_geometry_authoritative": result.canonical_geometry_authoritative,
        "gaussian_authority": gaussian.artifact.authority.value if gaussian.artifact else None,
        "nerf_authority": nerf.artifact.authority.value if nerf.artifact else None,
        "development_mock_used": result.development_mock_used,
        "production_execution_claimed": False,
        "certified_metrology": False,
        "reconstruction_ui_enabled": False,
        "future_phase_ownership": result.as_dict()["future_phase_ownership"],
        "plan": str(plan_path),
        "result": str(result_path),
    }
    summary_path = OUT / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("Phase 8.7F1 reconstruction orchestration foundation: PASS")
    print(f"stages={len(result.stages)} status={result.status.value} fallback={depth.provider_id}")
    print("canonical_geometry_authoritative=True appearance_authority=appearance_only")
    print("development_mock_used=True production_execution_claimed=False")
    print("certified_metrology=False reconstruction_ui_enabled=False")
    print("F2=DA3 F3=geometry+semantics+QC F4=Gaussian+NeRF F5=PersistentReconstructionScene")
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

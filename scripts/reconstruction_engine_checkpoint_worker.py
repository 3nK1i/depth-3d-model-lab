#!/usr/bin/env python3
"""G2 worker used to prove durable whole-engine checkpoint/resume across process restarts."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction import (
    GaussianAppearanceProviderExecutor,
    GaussianBackendKind,
    GaussianOptimizationProfile,
    GaussianSceneArtifact,
    GaussianWorkerProcessSpec,
    GaussianWorkerRequest,
    NerfAppearanceProviderExecutor,
    NerfBackendKind,
    NerfOptimizationProfile,
    NerfSceneArtifact,
    NerfWorkerProcessSpec,
    NerfWorkerRequest,
    ReconstructionArtifactKind,
    ReconstructionOperationControl,
    ReconstructionPath,
    ReconstructionPipelineGoal,
    ReconstructionPipelineOrchestrator,
    ReconstructionPipelineRequest,
    ReconstructionPipelineRunStatus,
    ReconstructionPipelineStage,
    ReconstructionQualityProfile,
    appearance_provider_descriptors,
    build_gaussian_seed,
    build_reconstruction_pipeline_plan,
    check_reconstruction_engine_resume_compatibility,
    create_reconstruction_engine_checkpoint,
    persistent_scene_from_pipeline_result,
    read_persistent_reconstruction_scene,
    read_reconstruction_engine_checkpoint,
    run_gaussian_external_worker,
    run_nerf_external_worker,
    verify_persistent_scene_file_references,
    write_gaussian_seed_npz,
    write_persistent_reconstruction_scene,
    write_reconstruction_engine_checkpoint,
    write_reconstruction_pipeline_json,
)
from tests.reconstruction_f3_helpers import make_f3_fixture


def _env_python(name: str) -> Path:
    current = Path(sys.executable).resolve()
    if current.parent.name == "bin" and len(current.parents) >= 3 and current.parents[2].name == "envs":
        return current.parents[2] / name / "bin/python"
    return Path.home() / f"anaconda3/envs/{name}/bin/python"


def _request_and_fixture(*, session_id: str, request_id: str):
    fixture = make_f3_fixture(session_id=session_id, request_id=request_id)
    providers = fixture.providers + appearance_provider_descriptors()
    request = ReconstructionPipelineRequest(
        request_id=request_id,
        session_id=session_id,
        path=ReconstructionPath.OFFLINE,
        quality_profile=ReconstructionQualityProfile.EFFICIENT,
        goals=(
            ReconstructionPipelineGoal.SEMANTIC_INSPECTION,
            ReconstructionPipelineGoal.GAUSSIAN_APPEARANCE,
            ReconstructionPipelineGoal.NERF_APPEARANCE,
        ),
        allow_optional_degradation=False,
    )
    plan = build_reconstruction_pipeline_plan(request, providers)
    return fixture, providers, request, plan


def _checkpoint_mode(args) -> int:
    fixture, providers, request, plan = _request_and_fixture(
        session_id=args.session_id, request_id=args.request_id
    )
    checkpoint_root = Path(args.checkpoint_root).resolve()
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    control = ReconstructionOperationControl()
    checkpoint_id = str(uuid4())
    writes = 0

    def hook(stages):
        nonlocal writes
        checkpoint = create_reconstruction_engine_checkpoint(
            plan=plan,
            providers=providers,
            stage_results=stages,
            checkpoint_id=checkpoint_id,
        )
        write_reconstruction_engine_checkpoint(checkpoint, checkpoint_root)
        writes += 1
        if stages[-1].stage is ReconstructionPipelineStage.DEPTH_EVIDENCE:
            control.request_cancel()

    result = ReconstructionPipelineOrchestrator(
        plan=plan, providers=providers, executors=fixture.executors
    ).run(control=control, stage_commit_hook=hook)
    checkpoint = read_reconstruction_engine_checkpoint(checkpoint_root)
    passed = (
        result.status is ReconstructionPipelineRunStatus.CANCELLED
        and checkpoint.completed_stage_count == 3
        and checkpoint.next_stage is ReconstructionPipelineStage.DENSE_GEOMETRY
        and checkpoint.checkpoint_id == checkpoint_id
        and writes == 3
    )
    payload = {
        "phase": "8.7G2",
        "mode": "checkpoint",
        "pass": passed,
        "pipeline_status": result.status.value,
        "checkpoint_id": checkpoint.checkpoint_id,
        "checkpoint_fingerprint_sha256": checkpoint.checkpoint_fingerprint_sha256,
        "completed_stage_count": checkpoint.completed_stage_count,
        "last_committed_stage": checkpoint.last_committed_stage.value,
        "next_stage": checkpoint.next_stage.value,
        "checkpoint_writes": writes,
        "payload_refs": list(checkpoint.payload_refs),
        "checkpoint_manifest": str((checkpoint_root / "reconstruction_engine_checkpoint.json").resolve()),
        "checkpoint_survives_interruption": True,
        "certified_metrology": False,
        "reconstruction_ui_enabled": False,
    }
    out = Path(args.workspace).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoint_result.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0 if passed else 1


class _ForbiddenCompletedStageExecutor:
    def execute(self, _context):
        raise RuntimeError("checkpointed_stage_was_reexecuted")


def _resume_mode(args) -> int:
    checkpoint = read_reconstruction_engine_checkpoint(Path(args.checkpoint_root).resolve())
    fixture, providers, request, plan = _request_and_fixture(
        session_id=checkpoint.session_id, request_id=checkpoint.request_id
    )
    compatibility = check_reconstruction_engine_resume_compatibility(
        checkpoint,
        plan=plan,
        providers=providers,
        available_payload_refs=fixture.store.payload_refs,
    )
    compatibility.require_compatible()

    output_root = Path(args.workspace).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    gsplat_python = Path(args.gsplat_python).expanduser().resolve()
    nerf_python = Path(args.nerf_python).expanduser().resolve()
    nerfstudio_root = Path(args.nerfstudio_root).expanduser().resolve()
    tcnn_root = Path(args.tiny_cuda_nn_root).expanduser().resolve()

    def gaussian_resolver(context, bundle, surface):
        seed = build_gaussian_seed(surface, max_gaussians=2048, minimum_vertex_quality=0.0)
        seed_path = write_gaussian_seed_npz(seed, (output_root / "gaussian-seed.npz").resolve())
        req = GaussianWorkerRequest.from_dataset(
            request_id=context.request.request_id,
            backend_kind=GaussianBackendKind.GSPLAT_EXTERNAL,
            profile=GaussianOptimizationProfile.EFFICIENT,
            dataset=bundle.dataset,
            dataset_manifest_path=bundle.manifest_path,
            seed_path=seed_path,
            output_dir=(output_root / "gaussian-worker").resolve(),
            maximum_input_frames=16,
        )
        spec = GaussianWorkerProcessSpec(
            python_executable=str(gsplat_python),
            worker_script=str((ROOT / "scripts/gaussian_workers/gsplat_worker.py").resolve()),
            timeout_s=args.backend_timeout_s,
            require_isolated_interpreter=True,
        )
        return run_gaussian_external_worker(spec, req)

    def nerf_resolver(context, bundle):
        req = NerfWorkerRequest.from_dataset(
            request_id=context.request.request_id,
            backend_kind=NerfBackendKind.NERFSTUDIO_NERFACTO_EXTERNAL,
            profile=NerfOptimizationProfile.EFFICIENT,
            dataset=bundle.dataset,
            dataset_manifest_path=bundle.manifest_path,
            output_dir=(output_root / "nerf-worker").resolve(),
            maximum_input_frames=16,
            qualification_training_steps=args.nerf_training_steps,
        )
        spec = NerfWorkerProcessSpec(
            python_executable=str(nerf_python),
            worker_script=str((ROOT / "scripts/nerf_workers/nerfstudio_worker.py").resolve()),
            timeout_s=args.backend_timeout_s,
            require_isolated_interpreter=True,
        )
        old_ns = os.environ.get("EIA_NERFSTUDIO_ROOT")
        old_tcnn = os.environ.get("EIA_TCNN_ROOT")
        os.environ["EIA_NERFSTUDIO_ROOT"] = str(nerfstudio_root)
        os.environ["EIA_TCNN_ROOT"] = str(tcnn_root)
        try:
            return run_nerf_external_worker(spec, req)
        finally:
            if old_ns is None:
                os.environ.pop("EIA_NERFSTUDIO_ROOT", None)
            else:
                os.environ["EIA_NERFSTUDIO_ROOT"] = old_ns
            if old_tcnn is None:
                os.environ.pop("EIA_TCNN_ROOT", None)
            else:
                os.environ["EIA_TCNN_ROOT"] = old_tcnn

    executors = dict(fixture.executors)
    # The first three stages are committed in the checkpoint. If resume is
    # implemented incorrectly this guard makes the run fail loudly.
    executors[fixture.providers[0].provider_id] = _ForbiddenCompletedStageExecutor()
    appearance = appearance_provider_descriptors()
    executors[appearance[0].provider_id] = GaussianAppearanceProviderExecutor(
        store=fixture.store, output_root=output_root, artifact_resolver=gaussian_resolver
    )
    executors[appearance[1].provider_id] = NerfAppearanceProviderExecutor(
        store=fixture.store, output_root=output_root, artifact_resolver=nerf_resolver
    )

    result = ReconstructionPipelineOrchestrator(
        plan=plan, providers=providers, executors=executors
    ).run(resume_stage_results=checkpoint.stage_results)
    write_reconstruction_pipeline_json(result, output_root / "pipeline_result.json")
    by_kind = {stage.artifact.kind: stage.artifact for stage in result.stages if stage.artifact}
    gaussian_art = by_kind.get(ReconstructionArtifactKind.GAUSSIAN_SCENE)
    nerf_art = by_kind.get(ReconstructionArtifactKind.NERF_SCENE)
    gaussian_scene = fixture.store.get(gaussian_art.payload_ref, GaussianSceneArtifact) if gaussian_art else None
    nerf_scene = fixture.store.get(nerf_art.payload_ref, NerfSceneArtifact) if nerf_art else None
    snapshots = {
        "gaussian_scene": gaussian_scene.as_dict() if gaussian_scene else {},
        "nerf_scene": nerf_scene.as_dict() if nerf_scene else {},
    }
    scene = persistent_scene_from_pipeline_result(result, payload_snapshots_by_kind=snapshots)
    manifest = write_persistent_reconstruction_scene(scene, output_root / "persistent-scene")
    reloaded = read_persistent_reconstruction_scene(manifest)
    refs = verify_persistent_scene_file_references(reloaded)
    passed = bool(
        result.status is ReconstructionPipelineRunStatus.COMPLETED
        and result.canonical_geometry_authoritative
        and gaussian_art
        and nerf_art
        and reloaded.state.value == "ready"
        and reloaded.canonical_geometry_authoritative
        and len(refs) >= 3
    )
    payload = {
        "phase": "8.7G2",
        "mode": "resume",
        "pass": passed,
        "compatibility": compatibility.as_dict(),
        "checkpoint_id": checkpoint.checkpoint_id,
        "checkpoint_fingerprint_sha256": checkpoint.checkpoint_fingerprint_sha256,
        "reused_stage_count": len(checkpoint.stage_results),
        "first_executed_stage": result.stages[len(checkpoint.stage_results)].stage.value,
        "pipeline_status": result.status.value,
        "canonical_geometry_authoritative": result.canonical_geometry_authoritative,
        "gaussian_authority": gaussian_art.authority.value if gaussian_art else "missing",
        "nerf_authority": nerf_art.authority.value if nerf_art else "missing",
        "persistent_scene_state": reloaded.state.value,
        "scene_id": reloaded.scene_id,
        "scene_fingerprint_sha256": reloaded.scene_fingerprint_sha256,
        "file_reference_count": len(refs),
        "completed_stage_reexecution_blocked": True,
        "certified_metrology": False,
        "reconstruction_ui_enabled": False,
    }
    (output_root / "resume_result.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0 if passed else 1


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("checkpoint", "resume"), required=True)
    p.add_argument("--workspace", required=True)
    p.add_argument("--checkpoint-root", required=True)
    p.add_argument("--request-id", default="phase8-7g2-real")
    p.add_argument("--session-id", default=str(uuid4()))
    p.add_argument("--gsplat-python", default=str(_env_python("eia-gaussian-gsplat")))
    p.add_argument("--nerf-python", default=str(_env_python("eia-nerf-nerfstudio")))
    p.add_argument("--nerfstudio-root", default=str(Path.home() / "Desktop/eia-nerf-backends/nerfstudio"))
    p.add_argument("--tiny-cuda-nn-root", default=str(Path.home() / "Desktop/eia-nerf-backends/tiny-cuda-nn"))
    p.add_argument("--nerf-training-steps", type=int, default=16)
    p.add_argument("--backend-timeout-s", type=float, default=1800.0)
    a = p.parse_args()
    return _checkpoint_mode(a) if a.mode == "checkpoint" else _resume_mode(a)


if __name__ == "__main__":
    raise SystemExit(main())

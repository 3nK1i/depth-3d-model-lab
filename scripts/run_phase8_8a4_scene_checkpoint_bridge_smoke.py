#!/usr/bin/env python3
"""Phase 8.8A4 application persistence/checkpoint bridge smoke.

This smoke intentionally starts no heavy/native backend.  It can optionally
consume the real Phase 8.8A3 ``offline_result.json`` to prove that sparse SfM
is retained as PARTIAL, then exercises the existing F5/G2 core contracts with
the core deterministic stage executor (not tests fixtures).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from uuid import uuid4

from PyQt6.QtCore import QCoreApplication

from app.reconstruction_persistence_controller import (
    ReconstructionPersistenceController,
    ReconstructionSceneAvailability,
)
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
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/phase8_8a4_scene_checkpoint_bridge"),
    )
    parser.add_argument(
        "--offline-result",
        type=Path,
        default=None,
        help="Optional real Phase 8.8A3 offline_result.json.",
    )
    return parser.parse_args()


def _provider(provider_id: str, role: ReconstructionProviderRole):
    return ReconstructionProviderDescriptor(
        provider_id=provider_id,
        provider_version="1",
        roles=(role,),
        execution=ReconstructionProviderExecution.DEVELOPMENT_MOCK,
        priority=10,
        adapter_version="phase8.8a4-smoke",
    )


def _fixture():
    providers = (
        _provider("ingestion", ReconstructionProviderRole.INGESTION),
        _provider("pose", ReconstructionProviderRole.CAMERA_POSE),
        _provider("depth", ReconstructionProviderRole.DEPTH),
        _provider("dense", ReconstructionProviderRole.DENSE_GEOMETRY),
        _provider("mesh", ReconstructionProviderRole.SURFACE_MESH),
        _provider("semantics", ReconstructionProviderRole.SEMANTICS),
        _provider("inspection", ReconstructionProviderRole.INSPECTION_QC),
    )
    request = ReconstructionPipelineRequest(
        request_id="phase8-8a4-smoke",
        session_id=str(uuid4()),
        path=ReconstructionPath.OFFLINE,
        quality_profile=ReconstructionQualityProfile.EFFICIENT,
        goals=(ReconstructionPipelineGoal.SEMANTIC_INSPECTION,),
    )
    plan = build_reconstruction_pipeline_plan(request, providers)
    return providers, request, plan


def _executors(providers):
    return {p.provider_id: DeterministicMockStageExecutor(p.provider_id) for p in providers}


def main() -> int:
    args = _args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _app = QCoreApplication.instance() or QCoreApplication(sys.argv[:1])
    bridge = ReconstructionPersistenceController()

    real_a3_checked = False
    real_a3_partial = None
    real_a3_session_id = ""
    if args.offline_result is not None:
        result_path = args.offline_result.expanduser().resolve()
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        bridge.accept_offline_result(payload, result_path.parent)
        real_a3_checked = True
        real_a3_partial = bridge.scene_availability is ReconstructionSceneAvailability.PARTIAL
        real_a3_session_id = str(payload.get("session_id", ""))
        if not real_a3_partial or bridge.scene is not None:
            raise RuntimeError("Real A3 sparse result was not preserved as partial-only.")

    providers, request, plan = _fixture()
    complete = ReconstructionPipelineOrchestrator(
        plan=plan,
        providers=providers,
        executors=_executors(providers),
    ).run()
    if complete.status is not ReconstructionPipelineRunStatus.COMPLETED:
        raise RuntimeError("Deterministic integrated pipeline did not complete.")
    scene = bridge.adopt_pipeline_result(complete, output_dir / "scene")
    scene_manifest = bridge.scene_manifest
    if scene_manifest is None:
        raise RuntimeError("Persistent scene manifest was not retained by application bridge.")
    loaded_scene = bridge.load_scene(scene_manifest, expected_session_id=request.session_id)

    control = ReconstructionOperationControl()
    committed = []

    def hook(stages):
        committed.append(stages)
        if stages[-1].stage is ReconstructionPipelineStage.DEPTH_EVIDENCE:
            control.request_cancel()

    cancelled = ReconstructionPipelineOrchestrator(
        plan=plan,
        providers=providers,
        executors=_executors(providers),
    ).run(control=control, stage_commit_hook=hook)
    if cancelled.status is not ReconstructionPipelineRunStatus.CANCELLED:
        raise RuntimeError("Checkpoint preparation run did not stop at the intended prefix.")
    checkpoint = bridge.create_checkpoint(
        plan=plan,
        providers=providers,
        stage_results=committed[-1],
        output_dir=output_dir / "checkpoint",
    )
    checkpoint_manifest = bridge.checkpoint_manifest
    if checkpoint_manifest is None:
        raise RuntimeError("Checkpoint manifest was not retained by application bridge.")
    bridge.load_checkpoint(checkpoint_manifest)
    preparation = bridge.prepare_resume(
        plan=plan,
        providers=providers,
        available_payload_refs=checkpoint.payload_refs,
    )
    resumed_executors = _executors(providers)
    resumed = ReconstructionPipelineOrchestrator(
        plan=plan,
        providers=providers,
        executors=resumed_executors,
    ).run(resume_stage_results=preparation.resume_stage_results)

    passed = bool(
        loaded_scene.scene_id == scene.scene_id
        and loaded_scene.canonical_geometry_authoritative
        and loaded_scene.certified_metrology is False
        and loaded_scene.reconstruction_ui_enabled is False
        and preparation.compatibility.compatible
        and preparation.reused_stage_count == 3
        and preparation.next_stage is ReconstructionPipelineStage.DENSE_GEOMETRY
        and resumed.status is ReconstructionPipelineRunStatus.COMPLETED
        and resumed_executors["ingestion"].calls == []
        and resumed_executors["pose"].calls == []
        and resumed_executors["depth"].calls == []
        and resumed_executors["dense"].calls == ["dense_geometry"]
        and (not real_a3_checked or bool(real_a3_partial))
    )
    summary = {
        "phase": "8.8A4",
        "pass": passed,
        "tests_fixture_used": False,
        "heavy_backends_started": False,
        "real_a3_result_checked": real_a3_checked,
        "real_a3_result_preserved_as_partial": real_a3_partial,
        "real_a3_session_id": real_a3_session_id,
        "persistent_scene": {
            "scene_id": scene.scene_id,
            "state": scene.state.value,
            "manifest": str(scene_manifest),
            "canonical_geometry_authoritative": scene.canonical_geometry_authoritative,
            "certified_metrology": scene.certified_metrology,
            "reconstruction_ui_enabled": scene.reconstruction_ui_enabled,
        },
        "checkpoint": {
            "checkpoint_id": checkpoint.checkpoint_id,
            "manifest": str(checkpoint_manifest),
            "reused_stage_count": preparation.reused_stage_count,
            "next_stage": preparation.next_stage.value,
            "compatibility": preparation.compatibility.as_dict(),
        },
        "resume": {
            "status": resumed.status.value,
            "ingestion_reexecuted": bool(resumed_executors["ingestion"].calls),
            "pose_reexecuted": bool(resumed_executors["pose"].calls),
            "depth_reexecuted": bool(resumed_executors["depth"].calls),
            "dense_calls": list(resumed_executors["dense"].calls),
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        "Phase 8.8A4 scene/checkpoint/resume bridge smoke: " + ("PASS" if passed else "FAIL")
    )
    print(f"scene_id={scene.scene_id} state={scene.state.value}")
    print(
        f"checkpoint_id={checkpoint.checkpoint_id} reused={preparation.reused_stage_count} "
        f"next_stage={preparation.next_stage.value}"
    )
    if real_a3_checked:
        print(
            f"real_a3_session_id={real_a3_session_id} "
            f"preserved_as_partial={real_a3_partial}"
        )
    print(f"summary={summary_path}")
    bridge.shutdown()
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

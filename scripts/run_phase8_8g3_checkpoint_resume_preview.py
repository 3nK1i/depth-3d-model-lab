#!/usr/bin/env python3
"""Interactive Phase 8.8G3 checkpoint/resume UX preview."""

from __future__ import annotations

from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyQt6.QtWidgets import QApplication

from app.application_controller import CameraEnumerator
from app.main_window import MainWindow
from app.reconstruction_persistence_controller import ReconstructionPersistenceController
from app.widgets.mode_tabs import AppMode
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
    create_reconstruction_engine_checkpoint,
    write_reconstruction_engine_checkpoint,
    build_reconstruction_pipeline_plan,
)

OUTPUT = Path("/tmp/phase8_8g3_checkpoint_resume_preview")
SESSION = "77777777-7777-4777-8777-777777777777"


def _provider(provider_id: str, role: ReconstructionProviderRole) -> ReconstructionProviderDescriptor:
    return ReconstructionProviderDescriptor(
        provider_id=provider_id,
        provider_version="1",
        roles=(role,),
        execution=ReconstructionProviderExecution.DEVELOPMENT_MOCK,
        priority=10,
        adapter_version="phase8.8g3-preview",
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
        request_id="phase8-8g3-preview",
        session_id=SESSION,
        path=ReconstructionPath.OFFLINE,
        quality_profile=ReconstructionQualityProfile.EFFICIENT,
        goals=(ReconstructionPipelineGoal.SEMANTIC_INSPECTION,),
    )
    plan = build_reconstruction_pipeline_plan(request, providers)
    executors = {item.provider_id: DeterministicMockStageExecutor(item.provider_id) for item in providers}
    full = ReconstructionPipelineOrchestrator(plan=plan, providers=providers, executors=executors).run()
    if full.status is not ReconstructionPipelineRunStatus.COMPLETED:
        raise RuntimeError("Could not build G3 preview scene fixture.")

    control = ReconstructionOperationControl()
    committed = []
    prefix_executors = {item.provider_id: DeterministicMockStageExecutor(item.provider_id) for item in providers}

    def hook(stages):
        committed.append(stages)
        if stages[-1].stage is ReconstructionPipelineStage.DEPTH_EVIDENCE:
            control.request_cancel()

    prefix = ReconstructionPipelineOrchestrator(
        plan=plan,
        providers=providers,
        executors=prefix_executors,
    ).run(control=control, stage_commit_hook=hook)
    if prefix.status is not ReconstructionPipelineRunStatus.CANCELLED:
        raise RuntimeError("Could not build G3 preview checkpoint fixture.")
    checkpoint = create_reconstruction_engine_checkpoint(
        plan=plan,
        providers=providers,
        stage_results=committed[-1],
    )
    return providers, plan, full, checkpoint


def main() -> int:
    shutil.rmtree(OUTPUT, ignore_errors=True)
    scene_dir = (OUTPUT / "scene").resolve()
    checkpoint_dir = (OUTPUT / "checkpoint").resolve()
    scene_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    providers, plan, full, checkpoint = _fixture()
    bridge = ReconstructionPersistenceController()
    bridge.adopt_pipeline_result(full, scene_dir)
    manifest = write_reconstruction_engine_checkpoint(checkpoint, checkpoint_dir)

    CameraEnumerator.start = lambda self: None  # type: ignore[method-assign]
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)
    controller = window._reconstruction_controller
    controller.load_persistent_scene(scene_dir)
    controller.load_engine_checkpoint(manifest)
    controller.set_engine_resume_context(
        plan=plan,
        providers=providers,
        available_payload_refs=checkpoint.payload_refs,
    )
    window.resize(1366, 768)
    window.show()
    exit_code = app.exec()
    bridge.shutdown()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

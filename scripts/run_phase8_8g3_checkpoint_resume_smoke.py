#!/usr/bin/env python3
"""Decisive Phase 8.8G3 checkpoint/load and resume-preparation smoke."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyQt6.QtWidgets import QApplication, QFileDialog

from app.application_controller import CameraEnumerator
from app.main_window import MainWindow
from app.reconstruction_checkpoint_resume import ReconstructionCheckpointResumeState
from app.reconstruction_session_model import ReconstructionSessionIdentityStatus
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

OUTPUT = Path("/tmp/phase8_8g3_checkpoint_resume")
SESSION = "77777777-7777-4777-8777-777777777777"


def _provider(provider_id: str, role: ReconstructionProviderRole) -> ReconstructionProviderDescriptor:
    return ReconstructionProviderDescriptor(
        provider_id=provider_id,
        provider_version="1",
        roles=(role,),
        execution=ReconstructionProviderExecution.DEVELOPMENT_MOCK,
        priority=10,
        adapter_version="phase8.8g3-smoke",
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
        request_id="phase8-8g3-smoke",
        session_id=SESSION,
        path=ReconstructionPath.OFFLINE,
        quality_profile=ReconstructionQualityProfile.EFFICIENT,
        goals=(ReconstructionPipelineGoal.SEMANTIC_INSPECTION,),
    )
    plan = build_reconstruction_pipeline_plan(request, providers)
    full_executors = {
        provider.provider_id: DeterministicMockStageExecutor(provider.provider_id)
        for provider in providers
    }
    full_result = ReconstructionPipelineOrchestrator(
        plan=plan,
        providers=providers,
        executors=full_executors,
    ).run()
    if full_result.status is not ReconstructionPipelineRunStatus.COMPLETED:
        raise RuntimeError(f"Unexpected full fixture status: {full_result.status.value}")

    prefix_executors = {
        provider.provider_id: DeterministicMockStageExecutor(provider.provider_id)
        for provider in providers
    }
    control = ReconstructionOperationControl()
    committed = []

    def hook(stages):
        committed.append(stages)
        if stages[-1].stage is ReconstructionPipelineStage.DEPTH_EVIDENCE:
            control.request_cancel()

    prefix_result = ReconstructionPipelineOrchestrator(
        plan=plan,
        providers=providers,
        executors=prefix_executors,
    ).run(control=control, stage_commit_hook=hook)
    if prefix_result.status is not ReconstructionPipelineRunStatus.CANCELLED:
        raise RuntimeError(f"Unexpected checkpoint fixture status: {prefix_result.status.value}")
    checkpoint = create_reconstruction_engine_checkpoint(
        plan=plan,
        providers=providers,
        stage_results=committed[-1],
    )
    return providers, plan, full_result, checkpoint


def main() -> int:
    shutil.rmtree(OUTPUT, ignore_errors=True)
    scene_dir = (OUTPUT / "scene").resolve()
    checkpoint_dir = (OUTPUT / "checkpoint").resolve()
    scene_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    providers, plan, full_result, checkpoint = _fixture()
    fixture_bridge = ReconstructionPersistenceController()
    fixture_bridge.adopt_pipeline_result(full_result, scene_dir)
    checkpoint_manifest = write_reconstruction_engine_checkpoint(checkpoint, checkpoint_dir)

    CameraEnumerator.start = lambda self: None  # type: ignore[method-assign]
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)
    page = window._reconstruction_page
    controller = window._reconstruction_controller
    controller.load_persistent_scene(scene_dir)
    window.resize(1366, 768)
    window.show()
    app.processEvents()

    errors: list[str] = []
    initial_load_only = bool(
        page.checkpoint_resume_presentation is not None
        and page.checkpoint_resume_presentation.state is ReconstructionCheckpointResumeState.LOAD_ONLY
        and page._session_checkpoint_load_button.isEnabled()
        and not page._session_resume_button.isEnabled()
    )

    original_open = QFileDialog.getOpenFileName
    try:
        QFileDialog.getOpenFileName = lambda *_args, **_kwargs: (str(checkpoint_manifest), "")
        page._session_checkpoint_load_button.click()
        app.processEvents()
        loaded_integrity_ok = bool(
            controller.engine_checkpoint is not None
            and controller.engine_checkpoint.checkpoint_id == checkpoint.checkpoint_id
            and controller.engine_checkpoint_manifest == checkpoint_manifest.resolve()
        )
        context_required = bool(
            page.checkpoint_resume_presentation is not None
            and page.checkpoint_resume_presentation.state
            is ReconstructionCheckpointResumeState.CONTEXT_REQUIRED
            and not page._session_resume_button.isEnabled()
        )

        compatibility = controller.set_engine_resume_context(
            plan=plan,
            providers=providers,
            available_payload_refs=checkpoint.payload_refs,
        )
        app.processEvents()
        compatible_ready = bool(
            compatibility is not None
            and compatibility.compatible
            and page.checkpoint_resume_presentation is not None
            and page.checkpoint_resume_presentation.state
            is ReconstructionCheckpointResumeState.COMPATIBLE
            and page._session_resume_button.isEnabled()
        )

        page._session_resume_button.click()
        app.processEvents()
        preparation = controller.engine_resume_preparation
        prepared_ok = bool(
            preparation is not None
            and preparation.checkpoint.checkpoint_id == checkpoint.checkpoint_id
            and preparation.resume_stage_results == checkpoint.stage_results
            and preparation.next_stage == checkpoint.next_stage
            and page.checkpoint_resume_presentation is not None
            and page.checkpoint_resume_presentation.state
            is ReconstructionCheckpointResumeState.PREPARED
        )

        incompatible = controller.set_engine_resume_context(
            plan=plan,
            providers=providers,
            available_payload_refs=checkpoint.payload_refs[:-1],
        )
        app.processEvents()
        incompatible_blocked = bool(
            incompatible is not None
            and not incompatible.compatible
            and page.checkpoint_resume_presentation is not None
            and page.checkpoint_resume_presentation.state
            is ReconstructionCheckpointResumeState.INCOMPATIBLE
            and not page._session_resume_button.isEnabled()
        )

        # Restore the compatible prepared view for final visual inspection.
        controller.set_engine_resume_context(
            plan=plan,
            providers=providers,
            available_payload_refs=checkpoint.payload_refs,
        )
        controller.prepare_loaded_engine_resume()
        app.processEvents()
    except Exception as exc:  # pragma: no cover - smoke diagnostics
        errors.append(f"{type(exc).__name__}: {exc}")
        loaded_integrity_ok = context_required = compatible_ready = prepared_ok = incompatible_blocked = False
    finally:
        QFileDialog.getOpenFileName = original_open

    screenshot = OUTPUT / "reconstruction_g3_checkpoint_resume.png"
    screenshot_ok = window.grab().save(str(screenshot))
    model = page.reconstruction_session_model
    final_preparation = controller.engine_resume_preparation
    summary = {
        "phase": "8.8G3",
        "pass": bool(
            initial_load_only
            and loaded_integrity_ok
            and context_required
            and compatible_ready
            and prepared_ok
            and incompatible_blocked
            and final_preparation is not None
            and model is not None
            and model.identity_status is ReconstructionSessionIdentityStatus.CONSISTENT
            and model.canonical_geometry_authoritative
            and not model.certified_metrology
            and not controller.is_active
            and screenshot_ok
            and not errors
        ),
        "session_id": model.session_id if model is not None else "",
        "initial_checkpoint_load_only": initial_load_only,
        "checkpoint_integrity_load_succeeded": loaded_integrity_ok,
        "exact_resume_context_required": context_required,
        "compatible_context_enables_prepare": compatible_ready,
        "resume_preparation_handoff_created": prepared_ok,
        "missing_payload_context_blocks_prepare": incompatible_blocked,
        "session_identity_consistent": bool(
            model is not None
            and model.identity_status is ReconstructionSessionIdentityStatus.CONSISTENT
        ),
        "next_stage": final_preparation.next_stage.value if final_preparation is not None else "",
        "reused_stage_count": final_preparation.reused_stage_count if final_preparation is not None else 0,
        "resumed_engine_executed_by_g3": False,
        "failure_recovery_executed_by_g3": False,
        "new_checkpoint_format_written_by_g3": False,
        "heavy_backend_executed_by_g3": False,
        "canonical_geometry_authoritative": bool(model and model.canonical_geometry_authoritative),
        "certified_metrology": False,
        "checkpoint_ui": page._checkpoint_value.text(),
        "resume_ui": page._session_resume_state_value.text(),
        "screenshot": str(screenshot),
        "errors": errors,
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    window.close()
    app.processEvents()
    fixture_bridge.shutdown()

    print("Phase 8.8G3 checkpoint/resume UX smoke: " + ("PASS" if summary["pass"] else "FAIL"))
    print(f"checkpoint={summary['checkpoint_ui']!r} resume={summary['resume_ui']!r}")
    print(f"next={summary['next_stage']!r} reused={summary['reused_stage_count']}")
    print(f"screenshot={screenshot}")
    print(f"summary={OUTPUT / 'summary.json'}")
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

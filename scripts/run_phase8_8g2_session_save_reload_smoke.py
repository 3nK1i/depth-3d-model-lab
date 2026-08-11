#!/usr/bin/env python3
"""Decisive Phase 8.8G2 persistent-scene load/save/reload smoke."""

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
from app.reconstruction_persistence_controller import ReconstructionPersistenceController
from app.widgets.mode_tabs import AppMode
from core.reconstruction import (
    DeterministicMockStageExecutor,
    PERSISTENT_RECONSTRUCTION_SCENE_CHECKSUM,
    PERSISTENT_RECONSTRUCTION_SCENE_MANIFEST,
    ReconstructionPath,
    ReconstructionPipelineGoal,
    ReconstructionPipelineOrchestrator,
    ReconstructionPipelineRequest,
    ReconstructionPipelineRunStatus,
    ReconstructionProviderDescriptor,
    ReconstructionProviderExecution,
    ReconstructionProviderRole,
    ReconstructionQualityProfile,
    build_reconstruction_pipeline_plan,
)

OUTPUT = Path("/tmp/phase8_8g2_session_save_reload")
SESSION = "77777777-7777-4777-8777-777777777777"


def _provider(provider_id: str, role: ReconstructionProviderRole) -> ReconstructionProviderDescriptor:
    return ReconstructionProviderDescriptor(
        provider_id=provider_id,
        provider_version="1",
        roles=(role,),
        execution=ReconstructionProviderExecution.DEVELOPMENT_MOCK,
        priority=10,
        adapter_version="phase8.8g2-smoke",
    )


def _build_scene_result():
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
        request_id="phase8-8g2-smoke",
        session_id=SESSION,
        path=ReconstructionPath.OFFLINE,
        quality_profile=ReconstructionQualityProfile.EFFICIENT,
        goals=(ReconstructionPipelineGoal.SEMANTIC_INSPECTION,),
    )
    plan = build_reconstruction_pipeline_plan(request, providers)
    executors = {
        provider.provider_id: DeterministicMockStageExecutor(provider.provider_id)
        for provider in providers
    }
    result = ReconstructionPipelineOrchestrator(
        plan=plan,
        providers=providers,
        executors=executors,
    ).run()
    if result.status is not ReconstructionPipelineRunStatus.COMPLETED:
        raise RuntimeError(f"Unexpected fixture pipeline status: {result.status.value}")
    return result


def main() -> int:
    shutil.rmtree(OUTPUT, ignore_errors=True)
    source_dir = (OUTPUT / "source_scene").resolve()
    save_dir = (OUTPUT / "saved_scene").resolve()
    source_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    CameraEnumerator.start = lambda self: None  # type: ignore[method-assign]
    app = QApplication.instance() or QApplication(sys.argv)

    fixture_bridge = ReconstructionPersistenceController()
    source_scene = fixture_bridge.adopt_pipeline_result(_build_scene_result(), source_dir)
    source_manifest = source_dir / PERSISTENT_RECONSTRUCTION_SCENE_MANIFEST

    window = MainWindow()
    window._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)
    page = window._reconstruction_page
    controller = window._reconstruction_controller
    window.resize(1366, 768)
    window.show()
    app.processEvents()

    initial_load_only = bool(
        page._session_load_button.isEnabled()
        and not page._session_save_button.isEnabled()
        and not page._session_reload_button.isEnabled()
    )

    original_open = QFileDialog.getOpenFileName
    original_directory = QFileDialog.getExistingDirectory
    errors: list[str] = []
    try:
        QFileDialog.getOpenFileName = lambda *_args, **_kwargs: (str(source_manifest), "")
        page._session_load_button.click()
        app.processEvents()
        loaded = controller.persistent_scene
        load_ok = bool(
            loaded is not None
            and loaded.session_id == SESSION
            and loaded.scene_fingerprint_sha256 == source_scene.scene_fingerprint_sha256
            and controller.persistent_scene_manifest == source_manifest.resolve()
            and page._session_value.text() == SESSION[:8]
            and page._session_save_button.isEnabled()
            and page._session_reload_button.isEnabled()
        )

        QFileDialog.getExistingDirectory = lambda *_args, **_kwargs: str(save_dir)
        page._session_save_button.click()
        app.processEvents()
        saved_manifest = save_dir / PERSISTENT_RECONSTRUCTION_SCENE_MANIFEST
        saved_checksum = save_dir / PERSISTENT_RECONSTRUCTION_SCENE_CHECKSUM
        save_ok = bool(
            saved_manifest.is_file()
            and saved_checksum.is_file()
            and controller.persistent_scene_manifest == saved_manifest.resolve()
            and controller.persistent_scene is not None
            and controller.persistent_scene.scene_fingerprint_sha256
            == source_scene.scene_fingerprint_sha256
        )

        before_reload = controller.persistent_scene
        page._session_reload_button.click()
        app.processEvents()
        after_reload = controller.persistent_scene
        reload_ok = bool(
            before_reload is not None
            and after_reload is not None
            and after_reload.session_id == before_reload.session_id == SESSION
            and after_reload.scene_fingerprint_sha256 == before_reload.scene_fingerprint_sha256
            and controller.persistent_scene_manifest == saved_manifest.resolve()
        )
    except Exception as exc:  # pragma: no cover - smoke diagnostics
        errors.append(f"{type(exc).__name__}: {exc}")
        load_ok = save_ok = reload_ok = False
    finally:
        QFileDialog.getOpenFileName = original_open
        QFileDialog.getExistingDirectory = original_directory

    screenshot = OUTPUT / "reconstruction_g2_session_save_reload.png"
    screenshot_ok = window.grab().save(str(screenshot))

    model = page.reconstruction_session_model
    summary = {
        "phase": "8.8G2",
        "pass": bool(
            initial_load_only
            and load_ok
            and save_ok
            and reload_ok
            and model is not None
            and model.canonical_geometry_authoritative
            and not model.certified_metrology
            and controller.engine_checkpoint is None
            and screenshot_ok
            and not errors
        ),
        "session_id": model.session_id if model is not None else "",
        "initial_load_only": initial_load_only,
        "load_scene_executed_by_g2": load_ok,
        "save_scene_executed_by_g2": save_ok,
        "reload_scene_executed_by_g2": reload_ok,
        "same_session_reload": reload_ok,
        "manifest_checksum_sidecar_written": (save_dir / PERSISTENT_RECONSTRUCTION_SCENE_CHECKSUM).is_file(),
        "referenced_file_verification_required": True,
        "external_artifact_bundle_created_by_g2": False,
        "checkpoint_resume_executed_by_g2": False,
        "failure_recovery_executed_by_g2": False,
        "new_persistence_format_written_by_g2": False,
        "canonical_geometry_authoritative": bool(model and model.canonical_geometry_authoritative),
        "certified_metrology": False,
        "actions_ui": page._session_action_state_value.text(),
        "scene_ui": page._scene_manifest_value.text(),
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

    print("Phase 8.8G2 session save/reload smoke: " + ("PASS" if summary["pass"] else "FAIL"))
    print(f"session={summary['session_id']!r} actions={summary['actions_ui']!r}")
    print(f"scene={summary['scene_ui']!r}")
    print(f"screenshot={screenshot}")
    print(f"summary={OUTPUT / 'summary.json'}")
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

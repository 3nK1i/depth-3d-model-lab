#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from app.application_controller import CameraEnumerator
from app.main_window import MainWindow
from app.offline_reconstruction_controller import OfflineReconstructionRunState
from app.reconstruction_checkpoint_resume import ReconstructionCheckpointResumeState
from app.reconstruction_failure_recovery import ReconstructionFailureRecoveryState
from app.reconstruction_session_model import ReconstructionSessionIdentityStatus
from app.widgets.mode_tabs import AppMode
from tests.reconstruction_h1_helpers import create_checkpoint_fixture, run_whole_pipeline, write_ready_image_sequence

OUT = Path("/tmp/phase8_8h1_whole_workflow")


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    CameraEnumerator.start = lambda self: None
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)
    window.resize(1366, 768)
    window.show()
    QTest.qWait(20)
    app.processEvents()

    page = window._reconstruction_page
    controller = window._reconstruction_controller
    errors: list[str] = []

    try:
        source = write_ready_image_sequence(OUT)
        controller.select_image_sequence(source)
        QTest.qWait(10)
        source_ready = bool(page._start_button.isEnabled() and page._source_value.text() == source.name)

        fixture, result = run_whole_pipeline()
        scene = controller.adopt_pipeline_result(result, (OUT / "scene").resolve())
        QTest.qWait(10)
        model = page.reconstruction_session_model
        scene_bound = bool(
            model is not None
            and model.session_id == scene.session_id
            and model.identity_status is ReconstructionSessionIdentityStatus.CONSISTENT
            and page._offline_semantics is not None
            and page.appearance_representation_presentation is not None
        )

        saved = controller.save_persistent_scene((OUT / "saved-scene").resolve())
        reloaded = controller.reload_persistent_scene()
        persistence_ok = bool(
            saved.is_file()
            and reloaded.session_id == scene.session_id
            and reloaded.scene_fingerprint_sha256 == scene.scene_fingerprint_sha256
        )

        _checkpoint, prefix = create_checkpoint_fixture(fixture)
        created = controller.create_engine_checkpoint(
            plan=fixture.plan,
            providers=fixture.providers,
            stage_results=prefix,
            output_dir=(OUT / "checkpoint").resolve(),
        )
        compatibility = controller.set_engine_resume_context(
            plan=fixture.plan,
            providers=fixture.providers,
            available_payload_refs=created.payload_refs,
        )
        QTest.qWait(10)
        resume_ready = bool(
            compatibility is not None
            and compatibility.compatible
            and page.checkpoint_resume_presentation is not None
            and page.checkpoint_resume_presentation.state is ReconstructionCheckpointResumeState.COMPATIBLE
            and page._session_resume_button.isEnabled()
            and controller.engine_resume_preparation is None
        )

        page.set_error("H1 contained integration failure")
        page.set_offline_state(OfflineReconstructionRunState.FAILED)
        QTest.qWait(10)
        recovery_safe = bool(
            page.failure_recovery_presentation is not None
            and page.failure_recovery_presentation.state is ReconstructionFailureRecoveryState.PRESERVED_EVIDENCE
            and not page._session_retry_button.isEnabled()
            and page._session_resume_button.isEnabled()
        )

        appearance = page.appearance_representation_presentation
        authority_ok = bool(
            scene.canonical_geometry_authoritative
            and not scene.certified_metrology
            and appearance is not None
            and all(not entry.canonical_geometry_authoritative for entry in appearance.entries)
            and all(not entry.certified_metrology for entry in appearance.entries)
            and page.reconstruction_inspection_qc_presentation is None
        )

        screenshot = OUT / "reconstruction_h1_whole_workflow.png"
        window.grab().save(str(screenshot))
        passed = bool(source_ready and scene_bound and persistence_ok and resume_ready and recovery_safe and authority_ok and not errors)
        payload = {
            "phase": "8.8H1",
            "pass": passed,
            "errors": errors,
            "source_workflow_ready": source_ready,
            "persistent_scene_bound": scene_bound,
            "save_reload_same_scene": persistence_ok,
            "checkpoint_prepare_resume_ready": resume_ready,
            "preserved_evidence_recovery_safe": recovery_safe,
            "authority_chain_preserved": authority_ok,
            "canonical_geometry_authoritative": scene.canonical_geometry_authoritative,
            "certified_metrology": False,
            "gaussian_authority": "appearance_only",
            "nerf_authority": "appearance_only",
            "heavy_backend_executed_by_h1": False,
            "checkpoint_resume_executed_by_h1": False,
            "new_persistence_format_written_by_h1": False,
            "new_recovery_engine_written_by_h1": False,
            "new_application_controller_written_by_h1": False,
            "inspection_qc_recomputed_by_h1": False,
            "screenshot": str(screenshot),
        }
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
        payload = {"phase": "8.8H1", "pass": False, "errors": errors}
    finally:
        window.close()
        app.processEvents()

    summary = OUT / "summary.json"
    summary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Phase 8.8H1 whole Reconstruction workflow smoke: {'PASS' if payload.get('pass') else 'FAIL'}")
    if payload.get("pass"):
        print("source=True scene=True persistence=True resume=True recovery=True authority=True")
    print(f"screenshot={payload.get('screenshot', '')}")
    print(f"summary={summary}")
    return 0 if payload.get("pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())

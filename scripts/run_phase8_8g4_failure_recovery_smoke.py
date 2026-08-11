#!/usr/bin/env python3
"""Dependency-light UI smoke for Phase 8.8G4 failure/recovery behavior."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyQt6.QtWidgets import QApplication

from app.application_controller import CameraEnumerator
from app.main_window import MainWindow
from app.widgets.mode_tabs import AppMode
from app.offline_reconstruction_controller import OfflineReconstructionRunState
from app.reconstruction_checkpoint_resume import ReconstructionCheckpointResumeState
from app.reconstruction_controller import ReconstructionSourceSelection, ReconstructionWorkflow
from app.reconstruction_failure_recovery import (
    ReconstructionFailureRecoveryState,
    build_reconstruction_failure_recovery_presentation,
)
from app.reconstruction_session_model import build_reconstruction_session_model
from core.reconstruction import LiveCaptureRuntimeState
from tests.reconstruction_g3_helpers import write_checkpoint_fixture

OUTPUT = Path("/tmp/phase8_8g4_failure_recovery").resolve()
SESSION = "77777777-7777-4777-8777-777777777777"


def _ready_selection(root: Path) -> ReconstructionSourceSelection:
    source = root / "sequence"
    source.mkdir(parents=True, exist_ok=True)
    for index in range(3):
        (source / f"frame-{index}.png").write_bytes(b"g4-smoke")
    return ReconstructionSourceSelection(
        workflow=ReconstructionWorkflow.IMAGE_SEQUENCE,
        ready=True,
        display_name="recovery-sequence",
        detail="3 supported images • recovery smoke",
        path=source,
        item_count=3,
    )


def _authority_preservation_probe() -> bool:
    scene = SimpleNamespace(
        scene_id="scene-g4",
        session_id=SESSION,
        state=SimpleNamespace(value="ready"),
        artifacts=(),
        provider_provenance=(),
        canonical_geometry_authoritative=True,
        certified_metrology=False,
        created_at_utc="",
    )
    source = SimpleNamespace(
        workflow=SimpleNamespace(value="image_sequence"),
        ready=True,
        display_name="generic",
        detail="generic",
        path=None,
    )
    model = build_reconstruction_session_model(
        source_selection=source,
        live_session_id="",
        offline_session_id=SESSION,
        live_state=LiveCaptureRuntimeState.IDLE,
        offline_state=SimpleNamespace(value="failed"),
        scene_availability=SimpleNamespace(value="ready"),
        scene=scene,
    )
    presentation = build_reconstruction_failure_recovery_presentation(
        model,
        live_state=LiveCaptureRuntimeState.IDLE,
        offline_state=SimpleNamespace(value="failed"),
        live_stats=None,
        source_ready=True,
        last_error="contained failure",
        checkpoint_resume=None,
        controller_active=False,
        controller_shutdown=False,
    )
    return bool(
        presentation.canonical_geometry_authoritative
        and not presentation.certified_metrology
        and presentation.state is ReconstructionFailureRecoveryState.PRESERVED_EVIDENCE
        and not presentation.can_retry_from_source
    )


def main() -> int:
    shutil.rmtree(OUTPUT, ignore_errors=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    CameraEnumerator.start = lambda self: None  # type: ignore[method-assign]
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)
    page = window._reconstruction_page
    controller = window._reconstruction_controller
    selection = _ready_selection(OUTPUT)
    controller._source_selection = selection
    controller.source_selection_changed.emit(selection)
    app.processEvents()

    errors: list[str] = []
    try:
        stable_initial = bool(
            page.failure_recovery_presentation is not None
            and page.failure_recovery_presentation.state is ReconstructionFailureRecoveryState.STABLE
            and not page._session_retry_button.isEnabled()
        )

        page.set_error("live worker timeout")
        page.set_live_state(LiveCaptureRuntimeState.RECOVERING)
        app.processEvents()
        auto_recovery_observed = bool(
            page.failure_recovery_presentation is not None
            and page.failure_recovery_presentation.state is ReconstructionFailureRecoveryState.AUTO_RECOVERING
            and page.failure_recovery_presentation.automatic_recovery
            and not page._session_retry_button.isEnabled()
        )

        page.set_live_state(LiveCaptureRuntimeState.IDLE)
        page.set_error("deterministic offline failure")
        page.set_offline_state(OfflineReconstructionRunState.FAILED)
        app.processEvents()
        retry_available = bool(
            page.failure_recovery_presentation is not None
            and page.failure_recovery_presentation.state is ReconstructionFailureRecoveryState.RETRY_AVAILABLE
            and page._session_retry_button.isEnabled()
        )
        retry_calls: list[str] = []
        original_start = page._start_runtime
        page._start_runtime = lambda: retry_calls.append("fresh-source-start")
        page._session_retry_button.click()
        app.processEvents()
        page._start_runtime = original_start
        retry_delegated = retry_calls == ["fresh-source-start"]

        providers, _request, plan, checkpoint, manifest = write_checkpoint_fixture(OUTPUT / "checkpoint")
        controller.load_engine_checkpoint(manifest)
        compatibility = controller.set_engine_resume_context(
            plan=plan,
            providers=providers,
            available_payload_refs=checkpoint.payload_refs,
        )
        page.set_error("offline worker failed with checkpoint retained")
        page.set_offline_state(OfflineReconstructionRunState.FAILED)
        app.processEvents()
        checkpoint_preserved = bool(
            compatibility is not None
            and compatibility.compatible
            and page.checkpoint_resume_presentation is not None
            and page.checkpoint_resume_presentation.state is ReconstructionCheckpointResumeState.COMPATIBLE
            and page.failure_recovery_presentation is not None
            and page.failure_recovery_presentation.state is ReconstructionFailureRecoveryState.PRESERVED_EVIDENCE
            and not page._session_retry_button.isEnabled()
        )

        # Return the final visual to the actionable fresh-retry state without
        # executing or discarding checkpoint evidence through G4. The screenshot
        # therefore uses a separate fresh page state after clearing the checkpoint
        # with the existing explicit controller API.
        controller.clear_engine_checkpoint()
        controller.clear_engine_resume_context()
        page.set_error("Example isolated worker failure • source ready for fresh retry")
        page.set_offline_state(OfflineReconstructionRunState.FAILED)
        app.processEvents()
        authority_preserved = _authority_preservation_probe()
    except Exception as exc:  # pragma: no cover - smoke diagnostics
        errors.append(f"{type(exc).__name__}: {exc}")
        stable_initial = auto_recovery_observed = retry_available = retry_delegated = False
        checkpoint_preserved = authority_preserved = False

    window.resize(1366, 768)
    window.show()
    app.processEvents()
    screenshot = OUTPUT / "reconstruction_g4_failure_recovery.png"
    screenshot_ok = window.grab().save(str(screenshot))

    summary = {
        "phase": "8.8G4",
        "pass": bool(
            stable_initial
            and auto_recovery_observed
            and retry_available
            and retry_delegated
            and checkpoint_preserved
            and authority_preserved
            and screenshot_ok
            and not errors
        ),
        "stable_initial": stable_initial,
        "bounded_live_auto_recovery_observed": auto_recovery_observed,
        "fresh_source_retry_exposed": retry_available,
        "retry_delegates_existing_start_path": retry_delegated,
        "retained_checkpoint_blocks_ambiguous_retry": checkpoint_preserved,
        "canonical_geometry_authority_preserved": authority_preserved,
        "certified_metrology": False,
        "manual_live_worker_restart_executed_by_g4": False,
        "checkpoint_resume_executed_by_g4": False,
        "heavy_backend_executed_by_g4": False,
        "new_recovery_persistence_format_written_by_g4": False,
        "failure_ui": page._session_recovery_state_value.text(),
        "retry_enabled": page._session_retry_button.isEnabled(),
        "screenshot": str(screenshot),
        "errors": errors,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    window.close()
    app.processEvents()
    print("Phase 8.8G4 failure/recovery UX smoke: " + ("PASS" if summary["pass"] else "FAIL"))
    print(f"recovery={summary['failure_ui']!r} retry_enabled={summary['retry_enabled']}")
    print(f"screenshot={screenshot}")
    print(f"summary={OUTPUT / 'summary.json'}")
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

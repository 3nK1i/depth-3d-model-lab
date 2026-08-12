#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
from threading import Event
from time import monotonic

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

import app.main_window as main_window_module
from app.application_controller import CameraEnumerator
from app.offline_reconstruction_controller import OfflineReconstructionRunState
from app.reconstruction_controller import ReconstructionController
from app.reconstruction_failure_recovery import ReconstructionFailureRecoveryState
from app.widgets.mode_tabs import AppMode
from core.capture.frame import SourceFrame
from core.reconstruction import (
    DeterministicMockLiveSlamBackend,
    LiveCaptureRuntimeState,
    LiveCaptureSlamRuntime,
    LiveSlamRecoveryPolicy,
)
from tests.reconstruction_h1_helpers import create_checkpoint_fixture, run_whole_pipeline
from tests.reconstruction_h3_helpers import offline_configuration, write_h3_worker, write_ready_sequence
from tests.reconstruction_live_helpers import LIVE_IMAGE_SIZE, make_live_request

OUT = Path("/tmp/phase8_8h3_failure_recovery")


def _live_failure_cleanup_check() -> bool:
    shutdown_entered = Event()
    allow_shutdown = Event()

    class SlowFailureBackend(DeterministicMockLiveSlamBackend):
        def track_live_frame(self, frame, context):
            raise RuntimeError("H3 smoke terminal live failure")

        def shutdown(self) -> None:
            shutdown_entered.set()
            allow_shutdown.wait(timeout=2.0)
            super().shutdown()

    runtime = LiveCaptureSlamRuntime(
        request=make_live_request(),
        backend_factory=lambda _spec: SlowFailureBackend(),
        recovery_policy=LiveSlamRecoveryPolicy(max_worker_restarts=0, restart_backoff_s=0.0),
    )
    runtime.start()
    if not runtime.wait_for_state(LiveCaptureRuntimeState.RUNNING, timeout_s=2.0):
        return False
    frame = SourceFrame(
        np.zeros((LIVE_IMAGE_SIZE.height, LIVE_IMAGE_SIZE.width, 3), dtype=np.uint8),
        frame_id=0,
        timestamp=monotonic(),
    )
    runtime.offer_source_frame(frame)
    if not shutdown_entered.wait(timeout=2.0):
        return False
    held = bool(
        runtime.state is LiveCaptureRuntimeState.STOPPING
        and runtime.is_active
        and not runtime.stats.cleanup_complete
    )
    allow_shutdown.set()
    terminal = runtime.wait_for_state(LiveCaptureRuntimeState.FAILED, timeout_s=2.0)
    return bool(
        held
        and terminal
        and runtime.stats.cleanup_complete
        and not runtime.is_active
    )


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    errors: list[str] = []
    payload: dict[str, object]

    live_cleanup = _live_failure_cleanup_check()

    CameraEnumerator.start = lambda self: None
    worker = write_h3_worker(OUT / "h3_smoke_worker.py")
    real_controller = ReconstructionController

    def controller_factory(application_controller, parent=None):
        return real_controller(application_controller, parent, offline_worker_script=worker)

    main_window_module.ReconstructionController = controller_factory
    app = QApplication.instance() or QApplication(sys.argv)
    window = main_window_module.MainWindow()
    window._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)
    window.resize(1366, 768)
    window.show()
    QTest.qWait(30)
    app.processEvents()

    controller = window._reconstruction_controller
    page = window._reconstruction_page

    try:
        source = write_ready_sequence(OUT)
        controller.select_image_sequence(source)

        cancelled_finished: list[dict] = []
        controller.offline_run_finished.connect(cancelled_finished.append)
        controller.start_offline(offline_configuration(OUT, "cancel-smoke"))
        deadline = 200
        while controller.offline_state is not OfflineReconstructionRunState.RUNNING and deadline > 0:
            QTest.qWait(10)
            deadline -= 1
        cancel_requested = controller.cancel_offline()
        deadline = 500
        while not cancelled_finished and deadline > 0:
            QTest.qWait(10)
            deadline -= 1
        cancellation_ok = bool(
            cancel_requested
            and cancelled_finished
            and controller.offline_state is OfflineReconstructionRunState.CANCELLED
            and not controller.is_active
            and page.failure_recovery_presentation is not None
            and page.failure_recovery_presentation.state
            is ReconstructionFailureRecoveryState.CANCELLED_RETRY_AVAILABLE
        )

        fixture, result = run_whole_pipeline()
        controller.adopt_pipeline_result(result, (OUT / "scene").resolve())
        _checkpoint, prefix = create_checkpoint_fixture(fixture)
        created = controller.create_engine_checkpoint(
            plan=fixture.plan,
            providers=fixture.providers,
            stage_results=prefix,
            output_dir=(OUT / "checkpoint").resolve(),
        )
        controller.set_engine_resume_context(
            plan=fixture.plan,
            providers=fixture.providers,
            available_payload_refs=created.payload_refs,
        )

        failed_finished: list[dict] = []
        controller.offline_run_finished.connect(failed_finished.append)
        controller.start_offline(
            offline_configuration(OUT, "fail-preserved-smoke"),
            session_id=fixture.request.session_id,
        )
        deadline = 500
        while not failed_finished and deadline > 0:
            QTest.qWait(10)
            deadline -= 1
        preserved_ok = bool(
            failed_finished
            and controller.offline_state is OfflineReconstructionRunState.FAILED
            and not controller.is_active
            and page.failure_recovery_presentation is not None
            and page.failure_recovery_presentation.state
            is ReconstructionFailureRecoveryState.PRESERVED_EVIDENCE
            and not page._session_retry_button.isEnabled()
            and page._session_resume_button.isEnabled()
        )

        if preserved_ok:
            page._session_resume_button.click()
            QTest.qWait(25)
        resume_prepared = bool(
            controller.engine_resume_preparation is not None
            and not controller.is_active
        )
        navigation_released = window._mode_tabs._buttons[AppMode.BENCHMARK].isEnabled()

        screenshot = OUT / "reconstruction_h3_failure_recovery.png"
        window.grab().save(str(screenshot))
        passed = bool(
            live_cleanup
            and cancellation_ok
            and preserved_ok
            and resume_prepared
            and navigation_released
            and not errors
        )
        payload = {
            "phase": "8.8H3",
            "pass": passed,
            "errors": errors,
            "live_failure_resource_held_until_cleanup": live_cleanup,
            "offline_cancellation_finalized": cancellation_ok,
            "retained_scene_checkpoint_preserved": preserved_ok,
            "checkpoint_resume_prepared": resume_prepared,
            "fresh_retry_separate_from_resume": True,
            "navigation_resources_released": navigation_released,
            "new_recovery_engine_added_by_h3": False,
            "heavy_backend_executed_by_h3": False,
            "appearance_runtime_executed_by_h3": False,
            "new_persistence_format_written_by_h3": False,
            "certified_metrology": False,
            "screenshot": str(screenshot),
        }
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
        payload = {"phase": "8.8H3", "pass": False, "errors": errors}
    finally:
        window.close()
        app.processEvents()

    summary = OUT / "summary.json"
    summary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Phase 8.8H3 failure/cancellation/recovery smoke: {'PASS' if payload.get('pass') else 'FAIL'}")
    if payload.get("pass"):
        print("live_cleanup=True cancel=True preserved=True resume=True resources=True")
    print(f"screenshot={payload.get('screenshot', '')}")
    print(f"summary={summary}")
    return 0 if payload.get("pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())

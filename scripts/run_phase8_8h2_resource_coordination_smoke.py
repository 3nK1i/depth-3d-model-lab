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
from app.reconstruction_controller import ReconstructionController
from app.widgets.mode_tabs import AppMode
from core.runtime.state_machine import ApplicationState

OUT = Path("/tmp/phase8_8h2_resource_coordination")


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

    errors: list[str] = []
    original_property = ReconstructionController.is_active
    active = {"value": True}
    ReconstructionController.is_active = property(lambda _self: active["value"])

    try:
        window._inference_controller._state_machine._state = ApplicationState.DISCONNECTED
        window._on_reconstruction_activity_changed()
        QTest.qWait(10)

        snapshot_calls: list[object] = []
        stable_calls: list[object] = []
        benchmark_calls: list[object] = []
        window._snapshot_controller.start = lambda **kwargs: snapshot_calls.append(kwargs)
        window._stable_video_controller.begin = lambda **kwargs: stable_calls.append(kwargs)
        window._benchmark_controller.start = lambda configuration: benchmark_calls.append(configuration)

        window._run_snapshot()
        window._start_stable_video()
        window._on_benchmark_run_requested(object())
        snapshot_block = not snapshot_calls and "Reconstruction is active" in window._snapshot_page._log.toPlainText()
        stable_block = not stable_calls and "Reconstruction is active" in window._stable_video_page._log.toPlainText()
        benchmark_block = not benchmark_calls and "Reconstruction is active" in window._benchmark_page._log.toPlainText()
        inspection_block = "Reconstruction is active" in window._inspection_jobs_conflict()

        export_failures: list[str] = []
        window._inspection_page.show_export_failure = export_failures.append
        window._export_inspection("all")
        export_block = bool(export_failures and "Reconstruction is active" in export_failures[-1])

        navigation_lock = bool(
            window._mode_tabs._buttons[AppMode.RECONSTRUCTION].isEnabled()
            and not window._mode_tabs._buttons[AppMode.BENCHMARK].isEnabled()
        )
        benchmark_safety = bool(
            not window._benchmark_page._runtime_safe
            and "Reconstruction is active" in window._benchmark_page._runtime_block_reason
        )
        stable_safety = bool(
            not window._stable_video_page._start.isEnabled()
            and "Reconstruction is active" in window._stable_video_page._status.text()
        )

        active["value"] = False
        window._on_reconstruction_activity_changed()
        QTest.qWait(10)
        idle_release = bool(
            window._reconstruction_conflict_for_application_job("Benchmark") == ""
            and window._mode_tabs._buttons[AppMode.BENCHMARK].isEnabled()
            and window._benchmark_page._runtime_safe
        )

        releases: list[bool] = []
        window._reconstruction_shared_job_conflict = lambda: None
        window._inference_controller.release_model_for_external_job = lambda: releases.append(True) or True
        window._inference_controller._state_machine._state = ApplicationState.RUNNING
        live_share = bool(window._reconstruction_live_start_guard() is None and releases == [])
        offline_busy = window._reconstruction_offline_start_guard()
        busy_blocks_release = bool(offline_busy and "Live 3D inference is busy" in offline_busy and releases == [])
        window._inference_controller._state_machine._state = ApplicationState.SOURCE_READY
        offline_release = bool(window._reconstruction_offline_start_guard() is None and releases == [True])

        screenshot = OUT / "reconstruction_h2_resource_coordination.png"
        window.grab().save(str(screenshot))
        passed = all((
            snapshot_block,
            stable_block,
            benchmark_block,
            inspection_block,
            export_block,
            navigation_lock,
            benchmark_safety,
            stable_safety,
            idle_release,
            live_share,
            busy_blocks_release,
            offline_release,
        )) and not errors
        payload = {
            "phase": "8.8H2",
            "pass": passed,
            "errors": errors,
            "reciprocal_snapshot_block": snapshot_block,
            "reciprocal_stable_video_block": stable_block,
            "reciprocal_benchmark_block": benchmark_block,
            "reciprocal_inspection_block": inspection_block,
            "reciprocal_export_block": export_block,
            "navigation_lock_consistent": navigation_lock,
            "benchmark_safety_consistent": benchmark_safety,
            "stable_video_safety_consistent": stable_safety,
            "idle_resource_release": idle_release,
            "live_inference_sharing_preserved": live_share,
            "offline_busy_inference_blocks_release": busy_blocks_release,
            "offline_model_release_preserved": offline_release,
            "new_resource_controller_added_by_h2": False,
            "vram_threshold_policy_added_by_h2": False,
            "heavy_backend_executed_by_h2": False,
            "appearance_runtime_executed_by_h2": False,
            "hardware_profile_claimed_by_h2": False,
            "screenshot": str(screenshot),
        }
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
        payload = {"phase": "8.8H2", "pass": False, "errors": errors}
    finally:
        ReconstructionController.is_active = original_property
        window.close()
        app.processEvents()

    summary = OUT / "summary.json"
    summary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Phase 8.8H2 application resource coordination smoke: {'PASS' if payload.get('pass') else 'FAIL'}")
    if payload.get("pass"):
        print("reciprocal=True navigation=True live_share=True offline_release=True isolation=True")
    print(f"screenshot={payload.get('screenshot', '')}")
    print(f"summary={summary}")
    return 0 if payload.get("pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Real B4 live status/telemetry UI smoke using Test Mode + isolated Stella."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from time import monotonic, sleep

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_OPENGL", "software")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("DEPTH3D_TEST_FAKE_VISPY", "1")

from PyQt6.QtWidgets import QApplication

from app.application_controller import CameraEnumerator
from app.live_slam_controller import LiveStellaLaunchSettings
from app.main_window import MainWindow
from app.widgets.mode_tabs import AppMode
from core.capture.base_source import SourceStatus
from core.capture.video_source import VideoSource
from core.reconstruction import CameraIntrinsicsPolicy, LiveCaptureRuntimeState
from scripts.run_live_capture_stella_smoke import _build_euroc_video
from scripts.run_live_stella_worker_smoke import _load_euroc_index


def _wait_until(app: QApplication, predicate, *, timeout_s: float) -> bool:
    deadline = monotonic() + max(0.0, float(timeout_s))
    while monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        sleep(0.005)
    app.processEvents()
    return bool(predicate())


def _require(path: Path, *, directory: bool = False) -> Path:
    resolved = path.expanduser().resolve()
    ok = resolved.is_dir() if directory else resolved.is_file()
    if not ok:
        raise RuntimeError(f"Required path is missing: {resolved}")
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stella-prefix", type=Path, default=Path.home() / "Desktop/eia-stella-vslam-probe")
    parser.add_argument("--worker", type=Path, default=Path.home() / "Desktop/eia-stella-live-worker/bin/eia_stella_live_worker")
    parser.add_argument("--vocab", type=Path, default=Path.home() / "Desktop/phase8_4_stella_probe_data/orb_vocab.fbow")
    parser.add_argument("--euroc", type=Path, default=Path.home() / "Desktop/phase8_4_stella_probe_data/MH_01_easy/mav0")
    parser.add_argument("--max-frames", type=int, default=500)
    parser.add_argument("--playback-speed", type=float, default=2.0)
    parser.add_argument("--stop-after-observations", type=int, default=80)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/phase8_8b4_live_status_ui"))
    args = parser.parse_args()

    stella_prefix = _require(args.stella_prefix, directory=True)
    worker = _require(args.worker)
    vocab = _require(args.vocab)
    euroc = _require(args.euroc, directory=True)
    records = _load_euroc_index(euroc, args.max_frames)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    CameraEnumerator.start = lambda self: None  # type: ignore[method-assign]
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)
    app.processEvents()
    page = window._reconstruction_page
    controller = window._reconstruction_controller
    application_controller = window._controller
    errors: list[str] = []
    observations: list[object] = []
    statuses: list[SourceStatus] = []
    controller.error_occurred.connect(errors.append)
    controller.live_observation_ready.connect(observations.append)
    application_controller.source_status_changed.connect(statuses.append)

    session_id: str | None = None
    running_stage = ""
    running_telemetry = ""
    running_elapsed = ""
    running_indeterminate = False
    running_camera_geometry = ""
    running_metric_scale = ""
    running_metrology = ""
    cleanup_ok = False
    try:
        with tempfile.TemporaryDirectory(prefix="depth3d-phase8-b4-live-status-") as temp_text:
            video_path = Path(temp_text) / "euroc_cam0.avi"
            _build_euroc_video(records, video_path)
            video = VideoSource(str(video_path))
            video.set_loop(False)
            video.set_speed(args.playback_speed)
            application_controller._replace_active_source(video)
            if not _wait_until(
                app,
                lambda: SourceStatus.CONNECTED in statuses and application_controller.latest_source_frame is not None,
                timeout_s=8.0,
            ):
                raise RuntimeError("Simulated-live source did not provide a frame.")

            selection = controller.select_live_source()
            if not selection.ready:
                raise RuntimeError(f"Live selection is not ready: {selection.detail}")
            camera = controller.build_estimated_live_camera()
            page._live_camera = camera
            page._live_camera_mode = "estimated"
            page._refresh_live_camera_presentation()
            session_id = controller.start_selected_live(
                camera=camera,
                settings=LiveStellaLaunchSettings(
                    install_prefix=stella_prefix,
                    worker_executable=worker,
                    vocabulary_path=vocab,
                    startup_timeout_s=20.0,
                    frame_timeout_s=5.0,
                    shutdown_timeout_s=5.0,
                    require_qualified_build=True,
                ),
                camera_intrinsics_policy=CameraIntrinsicsPolicy.ALLOW_ESTIMATED,
            )
            if not _wait_until(
                app,
                lambda: controller.live_state in {LiveCaptureRuntimeState.RUNNING, LiveCaptureRuntimeState.FAILED},
                timeout_s=25.0,
            ):
                raise RuntimeError("Live B4 status smoke did not reach RUNNING.")
            if controller.live_state is LiveCaptureRuntimeState.FAILED:
                raise RuntimeError(f"Live startup failed: {errors}")
            if not _wait_until(app, lambda: len(observations) >= args.stop_after_observations, timeout_s=40.0):
                raise RuntimeError(f"Insufficient observations: {len(observations)}; errors={errors}")
            page._refresh_runtime_telemetry()
            running_stage = page._stage_value.text()
            running_telemetry = page._telemetry_value.text()
            running_elapsed = page._elapsed_value.text()
            running_indeterminate = page._progress_bar.minimum() == 0 and page._progress_bar.maximum() == 0
            running_camera_geometry = page._camera_geometry_value.text()
            running_metric_scale = page._metric_scale_value.text()
            running_metrology = page._metrology_value.text()

            if not controller.request_stop_live():
                raise RuntimeError("Live B4 stop request was rejected.")
            if not _wait_until(
                app,
                lambda: controller.live_state in {LiveCaptureRuntimeState.STOPPED, LiveCaptureRuntimeState.FAILED},
                timeout_s=12.0,
            ):
                raise RuntimeError("Live B4 status smoke did not stop cleanly.")
            cleanup_ok = controller.stop_live(timeout_s=2.0)
            application_controller.disconnect_active_source()
            app.processEvents()

        passed = bool(
            session_id
            and not errors
            and cleanup_ok
            and running_stage == "Live tracking"
            and running_indeterminate
            and running_elapsed != "—"
            and "processed" in running_telemetry
            and "obs" in running_telemetry
            and "poses" in running_telemetry
            and page._stage_value.text() == "Stopped"
            and running_camera_geometry == "Estimated / test"
            and running_metric_scale == "Unavailable (test)"
            and running_metrology == "Not certified"
            and window._mode_tabs._buttons[AppMode.RECONSTRUCTION].isEnabled()
        )
        summary = {
            "phase": "8.8B4-live-status",
            "pass": passed,
            "session_id": session_id,
            "observations": len(observations),
            "running_stage": running_stage,
            "running_progress_indeterminate": running_indeterminate,
            "running_elapsed": running_elapsed,
            "running_telemetry": running_telemetry,
            "final_stage": page._stage_value.text(),
            "camera_geometry_ui": running_camera_geometry,
            "metric_scale_ui": running_metric_scale,
            "metrology_ui": running_metrology,
            "camera_intrinsics_source": "estimated",
            "calibration_status": "not_calibrated",
            "metric_scale_established": False,
            "certified_metrology": False,
            "reconstruction_navigation_enabled": window._mode_tabs._buttons[AppMode.RECONSTRUCTION].isEnabled(),
            "errors": errors,
        }
        summary_path = output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Phase 8.8B4 live status UI smoke: {'PASS' if passed else 'FAIL'}")
        print(f"session_id={session_id} observations={len(observations)} running_stage={running_stage!r}")
        print(f"telemetry={running_telemetry}")
        print(f"summary={summary_path}")
        return 0 if passed else 1
    finally:
        window.close()
        app.processEvents()


if __name__ == "__main__":
    raise SystemExit(main())

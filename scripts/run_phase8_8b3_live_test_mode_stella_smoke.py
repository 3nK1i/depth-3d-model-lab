#!/usr/bin/env python3
"""Real B3 live Test Mode smoke through estimated intrinsics + Stella.

Uses ApplicationController as the sole capture owner, a simulated-live EuRoC video,
the existing Phase 8.6 estimated-intrinsics contract, and the isolated Stella worker.
No camera calibration file is loaded.
"""

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

from PyQt6.QtWidgets import QApplication

from app.application_controller import ApplicationController
from app.live_slam_controller import LiveStellaLaunchSettings
from app.reconstruction_controller import ReconstructionController
from app.widgets.original_view import OriginalSourceView
from app.widgets.source_bar import SourceBar
from core.capture.base_source import SourceStatus
from core.capture.video_source import VideoSource
from core.reconstruction import (
    CameraIntrinsicsPolicy,
    LiveCaptureRuntimeState,
    ReconstructionQualityProfile,
)
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
    parser.add_argument(
        "--stella-prefix",
        type=Path,
        default=Path.home() / "Desktop/depth3d-stella-vslam-probe",
    )
    parser.add_argument(
        "--worker",
        type=Path,
        default=Path.home() / "Desktop/depth3d-stella-live-worker/bin/depth3d_stella_live_worker",
    )
    parser.add_argument(
        "--vocab",
        type=Path,
        default=Path.home() / "Desktop/phase8_4_stella_probe_data/orb_vocab.fbow",
    )
    parser.add_argument(
        "--euroc",
        type=Path,
        default=Path.home() / "Desktop/phase8_4_stella_probe_data/MH_01_easy/mav0",
    )
    parser.add_argument("--max-frames", type=int, default=500)
    parser.add_argument("--playback-speed", type=float, default=2.0)
    parser.add_argument("--stop-after-observations", type=int, default=100)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/phase8_8b3_live_test_mode_stella"),
    )
    args = parser.parse_args()

    stella_prefix = _require(args.stella_prefix, directory=True)
    worker = _require(args.worker)
    vocab = _require(args.vocab)
    euroc = _require(args.euroc, directory=True)
    records = _load_euroc_index(euroc, args.max_frames)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication([])
    source_bar = SourceBar()
    original_view = OriginalSourceView()
    application_controller = ApplicationController(source_bar, original_view)
    controller = ReconstructionController(application_controller)

    states: list[LiveCaptureRuntimeState] = []
    observations = []
    errors: list[str] = []
    statuses: list[SourceStatus] = []
    controller.live_state_changed.connect(states.append)
    controller.live_observation_ready.connect(observations.append)
    controller.error_occurred.connect(errors.append)
    application_controller.source_status_changed.connect(statuses.append)

    session_id: str | None = None
    camera = None
    cleanup_ok = False
    try:
        with tempfile.TemporaryDirectory(prefix="depth3d-phase8-b3-test-mode-") as temp_text:
            video_path = Path(temp_text) / "euroc_cam0.avi"
            _build_euroc_video(records, video_path)
            video = VideoSource(str(video_path))
            video.set_loop(False)
            video.set_speed(args.playback_speed)
            application_controller._replace_active_source(video)

            if not _wait_until(
                app,
                lambda: SourceStatus.CONNECTED in statuses
                and application_controller.latest_source_frame is not None,
                timeout_s=8.0,
            ):
                raise RuntimeError("Simulated-live source did not provide a frame.")

            selection = controller.select_live_source()
            if not selection.ready:
                raise RuntimeError(f"Live selection is not ready: {selection.detail}")
            camera = controller.build_estimated_live_camera()
            if camera.intrinsics.source.value != "estimated":
                raise RuntimeError("Test Mode camera was not marked estimated.")
            if camera.intrinsics.calibration_status.value != "not_calibrated":
                raise RuntimeError("Test Mode camera was incorrectly marked calibrated.")

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
                profile=ReconstructionQualityProfile.EFFICIENT,
                camera_intrinsics_policy=CameraIntrinsicsPolicy.ALLOW_ESTIMATED,
            )
            if not _wait_until(
                app,
                lambda: controller.live_state
                in {LiveCaptureRuntimeState.RUNNING, LiveCaptureRuntimeState.FAILED},
                timeout_s=25.0,
            ):
                raise RuntimeError(f"Test Mode did not reach RUNNING: {controller.live_state}")
            if controller.live_state is LiveCaptureRuntimeState.FAILED:
                raise RuntimeError(f"Test Mode startup failed: {errors}")
            if not _wait_until(
                app,
                lambda: len(observations) >= args.stop_after_observations,
                timeout_s=40.0,
            ):
                raise RuntimeError(
                    f"Insufficient Test Mode observations: {len(observations)}; errors={errors}"
                )
            if not controller.request_stop_live():
                raise RuntimeError("Test Mode stop request was rejected.")
            if not _wait_until(
                app,
                lambda: controller.live_state
                in {LiveCaptureRuntimeState.STOPPED, LiveCaptureRuntimeState.FAILED},
                timeout_s=12.0,
            ):
                raise RuntimeError("Test Mode did not stop cleanly.")
            cleanup_ok = controller.stop_live(timeout_s=2.0)
            application_controller.disconnect_active_source()
    finally:
        controller.shutdown()
        application_controller.shutdown()
        source_bar.close()
        original_view.close()
        app.processEvents()

    poses = [observation for observation in observations if observation.pose is not None]
    passed = bool(
        session_id
        and camera is not None
        and not errors
        and cleanup_ok
        and camera.intrinsics.source.value == "estimated"
        and camera.intrinsics.calibration_status.value == "not_calibrated"
        and len(observations) >= args.stop_after_observations
        and len(poses) >= max(30, int(len(observations) * 0.4))
        and LiveCaptureRuntimeState.RUNNING in states
        and LiveCaptureRuntimeState.STOPPED in states
    )
    summary = {
        "phase": "8.8B3-test-mode-correction",
        "pass": passed,
        "session_id": session_id,
        "camera_intrinsics_source": None if camera is None else camera.intrinsics.source.value,
        "calibration_status": None
        if camera is None
        else camera.intrinsics.calibration_status.value,
        "camera_intrinsics_policy": CameraIntrinsicsPolicy.ALLOW_ESTIMATED.value,
        "metric_scale_established": False,
        "certified_metrology": False,
        "observations": len(observations),
        "poses": len(poses),
        "states": [state.value for state in states],
        "errors": errors,
        "capture_owner": "ApplicationController",
        "worker_process_isolated": True,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if not passed:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 1
    print("Phase 8.8B3 live Test Mode Stella smoke: PASS")
    print(f"session_id={session_id} observations={len(observations)} poses={len(poses)}")
    print("intrinsics=estimated calibration_status=not_calibrated metric_scale=False")
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

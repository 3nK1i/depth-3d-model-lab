#!/usr/bin/env python3
"""Real Phase 8.8A2 application-controller -> live stella lifecycle smoke.

The smoke uses the production ``ApplicationController`` capture path and the
Phase 8.8 ``ReconstructionController``. A temporary calibrated EuRoC video is
opened through ``VideoSource`` exactly as the application's browse path does,
then the controller exercises start -> pause -> resume -> non-blocking stop
against the already-qualified isolated stella worker.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile
from time import monotonic, sleep

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# This is a non-visual hardware/backend validation. Keep Qt usable on machines
# where the application is launched from a terminal without an active display.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from app.application_controller import ApplicationController
from app.live_slam_controller import LiveStellaLaunchSettings
from app.reconstruction_controller import ReconstructionController
from app.widgets.original_view import OriginalSourceView
from app.widgets.source_bar import SourceBar
from core.capture.base_source import SourceStatus
from core.capture.video_source import VideoSource
from core.reconstruction import LiveCaptureRuntimeState, ReconstructionQualityProfile
from scripts.run_live_capture_stella_smoke import _build_euroc_video
from scripts.run_live_stella_worker_smoke import _euroc_camera, _load_euroc_index


def _wait_until(app: QApplication, predicate, *, timeout_s: float) -> bool:
    deadline = monotonic() + max(0.0, float(timeout_s))
    while monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        sleep(0.005)
    app.processEvents()
    return bool(predicate())


def _require_path(path: Path, *, label: str, directory: bool = False) -> Path:
    resolved = path.expanduser().resolve()
    ok = resolved.is_dir() if directory else resolved.is_file()
    if not ok:
        kind = "directory" if directory else "file"
        raise RuntimeError(f"Required {label} {kind} is missing: {resolved}")
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stella-prefix",
        type=Path,
        default=Path.home() / "Desktop" / "eia-stella-vslam-probe",
    )
    parser.add_argument(
        "--worker",
        type=Path,
        default=Path.home()
        / "Desktop"
        / "eia-stella-live-worker"
        / "bin"
        / "eia_stella_live_worker",
    )
    parser.add_argument(
        "--vocab",
        type=Path,
        default=Path.home() / "Desktop" / "phase8_4_stella_probe_data" / "orb_vocab.fbow",
    )
    parser.add_argument(
        "--euroc",
        type=Path,
        default=Path.home()
        / "Desktop"
        / "phase8_4_stella_probe_data"
        / "MH_01_easy"
        / "mav0",
    )
    parser.add_argument("--max-frames", type=int, default=500)
    parser.add_argument("--playback-speed", type=float, default=2.0)
    parser.add_argument("--pause-after-observations", type=int, default=30)
    parser.add_argument("--stop-after-observations", type=int, default=140)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/phase8_8a2_live_controller_stella"),
    )
    args = parser.parse_args()

    if args.max_frames < 300:
        parser.error("--max-frames must be at least 300")
    if args.playback_speed <= 0.0:
        parser.error("--playback-speed must be positive")
    if args.pause_after_observations < 10:
        parser.error("--pause-after-observations must be at least 10")
    if args.stop_after_observations <= args.pause_after_observations + 20:
        parser.error("--stop-after-observations must leave at least 20 observations after pause")

    stella_prefix = _require_path(args.stella_prefix, label="stella prefix", directory=True)
    worker = _require_path(args.worker, label="stella worker")
    vocab = _require_path(args.vocab, label="ORB vocabulary")
    euroc = _require_path(args.euroc, label="EuRoC dataset", directory=True)

    records = _load_euroc_index(euroc, args.max_frames)
    camera = _euroc_camera()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    app = QApplication.instance() or QApplication([])
    source_bar = SourceBar()
    original_view = OriginalSourceView()
    application_controller = ApplicationController(source_bar, original_view)
    reconstruction_controller = ReconstructionController(application_controller)

    states: list[LiveCaptureRuntimeState] = []
    observations = []
    errors: list[str] = []
    source_statuses: list[SourceStatus] = []
    captured = 0

    reconstruction_controller.live_state_changed.connect(states.append)
    reconstruction_controller.live_observation_ready.connect(observations.append)
    reconstruction_controller.error_occurred.connect(errors.append)
    application_controller.source_status_changed.connect(source_statuses.append)

    def on_capture(_frame) -> None:
        nonlocal captured
        captured += 1

    application_controller.frame_captured.connect(on_capture)

    pause_ok = False
    resume_ok = False
    stop_request_ok = False
    blocking_cleanup_ok = False
    paused_observation_count = 0
    paused_capture_count = 0
    session_id: str | None = None
    stats = None

    try:
        with tempfile.TemporaryDirectory(prefix="eia-phase8-8a2-video-") as temp_text:
            video_path = Path(temp_text) / "euroc_cam0.avi"
            _build_euroc_video(records, video_path)

            video = VideoSource(str(video_path))
            video.set_loop(False)
            video.set_speed(args.playback_speed)

            # This is the exact internal path used after the application's file
            # dialog returns a video path; it preserves ApplicationController as
            # the capture owner and does not create a second capture subsystem.
            application_controller._replace_active_source(video)
            if not _wait_until(
                app,
                lambda: SourceStatus.CONNECTED in source_statuses,
                timeout_s=5.0,
            ):
                raise RuntimeError(f"VideoSource did not connect: {source_statuses}")

            session_id = reconstruction_controller.start_live(
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
            )

            if not _wait_until(
                app,
                lambda: reconstruction_controller.live_state
                in {LiveCaptureRuntimeState.RUNNING, LiveCaptureRuntimeState.FAILED},
                timeout_s=25.0,
            ):
                raise RuntimeError(
                    f"ReconstructionController did not reach RUNNING: "
                    f"{reconstruction_controller.live_state}"
                )
            if reconstruction_controller.live_state is LiveCaptureRuntimeState.FAILED:
                raise RuntimeError(f"Live stella startup failed: {errors}")

            if not _wait_until(
                app,
                lambda: len(observations) >= args.pause_after_observations,
                timeout_s=35.0,
            ):
                raise RuntimeError(
                    "Insufficient live observations before pause: "
                    f"observations={len(observations)} captured={captured} errors={errors}"
                )

            pause_ok = reconstruction_controller.pause_live()
            paused_observation_count = len(observations)
            paused_capture_count = captured
            if (
                not pause_ok
                or reconstruction_controller.live_state is not LiveCaptureRuntimeState.PAUSED
            ):
                raise RuntimeError("Controller pause did not enter PAUSED state.")

            # Capture remains owned/running by ApplicationController while the
            # reconstruction mailbox is paused. This explicitly proves that A2
            # does not pause or replace the source subsystem.
            if not _wait_until(
                app,
                lambda: captured >= paused_capture_count + 5,
                timeout_s=3.0,
            ):
                raise RuntimeError("Capture did not continue while reconstruction was paused.")
            app.processEvents()
            sleep(0.05)
            app.processEvents()
            if len(observations) != paused_observation_count:
                raise RuntimeError("Live observations advanced while reconstruction was paused.")

            resume_ok = reconstruction_controller.resume_live()
            if (
                not resume_ok
                or reconstruction_controller.live_state is not LiveCaptureRuntimeState.RUNNING
            ):
                raise RuntimeError("Controller resume did not restore RUNNING state.")

            if not _wait_until(
                app,
                lambda: len(observations) >= args.stop_after_observations,
                timeout_s=35.0,
            ):
                raise RuntimeError(
                    "Insufficient observations after resume: "
                    f"observations={len(observations)} captured={captured} errors={errors}"
                )

            stop_request_ok = reconstruction_controller.request_stop_live()
            if not stop_request_ok:
                raise RuntimeError("Non-blocking controller stop request was rejected.")
            if not _wait_until(
                app,
                lambda: reconstruction_controller.live_state
                in {LiveCaptureRuntimeState.STOPPED, LiveCaptureRuntimeState.FAILED},
                timeout_s=12.0,
            ):
                raise RuntimeError(
                    f"Controller did not stop cleanly: {reconstruction_controller.live_state}"
                )
            if reconstruction_controller.live_state is LiveCaptureRuntimeState.FAILED:
                raise RuntimeError(f"Live controller failed during stop: {errors}")

            stats = reconstruction_controller.live_stats
            blocking_cleanup_ok = reconstruction_controller.stop_live(timeout_s=2.0)
            application_controller.disconnect_active_source()
            for _ in range(20):
                app.processEvents()
                sleep(0.005)
    finally:
        reconstruction_controller.shutdown()
        application_controller.shutdown()
        source_bar.close()
        original_view.close()
        app.processEvents()

    if stats is None:
        raise RuntimeError("Live runtime statistics were unavailable at validation end.")

    poses = [item for item in observations if item.pose is not None]
    latencies_ms = [
        item.quality.processing_latency_ns / 1_000_000.0 for item in observations
    ]
    state_values = [state.value for state in states]
    source_status_values = [status.value for status in source_statuses]

    passed = bool(
        not errors
        and session_id
        and reconstruction_controller.live_session_id == session_id
        and pause_ok
        and resume_ok
        and stop_request_ok
        and blocking_cleanup_ok
        and LiveCaptureRuntimeState.STARTING in states
        and LiveCaptureRuntimeState.RUNNING in states
        and LiveCaptureRuntimeState.PAUSED in states
        and LiveCaptureRuntimeState.STOPPING in states
        and LiveCaptureRuntimeState.STOPPED in states
        and captured > paused_capture_count
        and len(observations) >= args.stop_after_observations
        and len(poses) >= max(60, int(len(observations) * 0.55))
        and stats.cleanup_complete
    )

    summary = {
        "phase": "8.8A2",
        "pass": passed,
        "controller": "ReconstructionController",
        "live_bridge": "LiveSlamCaptureController",
        "capture_source": "VideoSource",
        "capture_owner": "ApplicationController",
        "controller_path_exercised": True,
        "session_id": session_id,
        "session_identity_preserved": reconstruction_controller.live_session_id == session_id,
        "states": state_values,
        "source_statuses": source_status_values,
        "pause_ok": pause_ok,
        "resume_ok": resume_ok,
        "nonblocking_stop_request_ok": stop_request_ok,
        "blocking_cleanup_ok": blocking_cleanup_ok,
        "capture_continued_while_reconstruction_paused": captured > paused_capture_count,
        "frames_captured": captured,
        "frames_offered": stats.offered_frames,
        "frames_processed": stats.processed_frames,
        "frames_replaced_latest_only": stats.replaced_frames,
        "frames_rejected": stats.rejected_frames,
        "observations_received": len(observations),
        "pose_observations": len(poses),
        "worker_restarts": stats.worker_restarts,
        "worker_restart_failures": stats.worker_restart_failures,
        "cleanup_complete": stats.cleanup_complete,
        "real_stella_executed": bool(observations),
        "worker_process_isolated": True,
        "capture_thread_blocking": False,
        "latest_frame_only": True,
        "pose_convention": "T_reconstruction_from_camera",
        "scale_status": "arbitrary",
        "unit_to_metre": None,
        "loop_closure_enabled": False,
        "global_optimization_feature_enabled": False,
        "median_tracking_latency_ms": statistics.median(latencies_ms) if latencies_ms else None,
        "mean_tracking_latency_ms": statistics.mean(latencies_ms) if latencies_ms else None,
        "errors": errors,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if not passed:
        raise RuntimeError(f"Phase 8.8A2 live controller stella smoke failed; see {summary_path}")

    print("\nPhase 8.8A2 ReconstructionController live stella smoke: PASS")
    print(f"session_id={session_id}")
    print(f"states={state_values}")
    print(
        f"captured={captured} processed={stats.processed_frames} "
        f"observations={len(observations)} poses={len(poses)}"
    )
    print("pause_resume=True nonblocking_stop=True cleanup_complete=True")
    print("capture_owner=ApplicationController worker_process_isolated=True")
    print("pose_convention=T_reconstruction_from_camera gauge=arbitrary unit_to_metre=None")
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Real Phase 8.4C recorded-video capture -> isolated stella_vslam smoke.

Creates a temporary calibrated EuRoC video, plays it through the production
VideoSource/QThread path, feeds SourceFrame objects into the non-blocking live
capture runtime, and validates real isolated stella observations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import tempfile
from time import monotonic, sleep

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyQt6.QtCore import QCoreApplication

from core.capture.base_source import SourceStatus
from core.capture.video_source import VideoSource
from core.reconstruction import (
    STELLA_LIVE_BACKEND_ID,
    ExternalStellaLiveSlamBackend,
    LiveCaptureRuntimeState,
    LiveCaptureSlamRuntime,
    LiveSlamRequest,
    ReconstructionQualityProfile,
    make_recorded_video_live_metadata,
    new_reconstruction_session_id,
)
from scripts.run_live_stella_worker_smoke import _euroc_camera, _load_euroc_index


def _build_euroc_video(records: list[tuple[int, Path]], output: Path) -> None:
    first = cv2.imread(str(records[0][1]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Could not decode first EuRoC frame: {records[0][1]}")
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"MJPG"),
        20.0,
        (width, height),
        True,
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not create temporary MJPEG/AVI validation video.")
    try:
        for _timestamp_ns, path in records:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != (height, width):
                raise RuntimeError(f"Invalid EuRoC validation frame: {path}")
            writer.write(image)
    finally:
        writer.release()


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
        default=Path.home() / "Desktop" / "eia-stella-live-worker" / "bin" / "eia_stella_live_worker",
    )
    parser.add_argument(
        "--vocab",
        type=Path,
        default=Path.home() / "Desktop" / "phase8_4_stella_probe_data" / "orb_vocab.fbow",
    )
    parser.add_argument(
        "--euroc",
        type=Path,
        default=Path.home() / "Desktop" / "phase8_4_stella_probe_data" / "MH_01_easy" / "mav0",
    )
    parser.add_argument("--max-frames", type=int, default=500)
    parser.add_argument("--playback-speed", type=float, default=1.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/phase8_4c_live_capture_stella"),
    )
    args = parser.parse_args()
    if args.max_frames < 100:
        parser.error("--max-frames must be at least 100")
    if args.playback_speed <= 0.0:
        parser.error("--playback-speed must be positive")

    records = _load_euroc_index(args.euroc.expanduser().resolve(), args.max_frames)
    camera = _euroc_camera()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="depth3d-phase8-4c-video-") as temp_text:
        video_path = Path(temp_text) / "euroc_cam0.avi"
        _build_euroc_video(records, video_path)
        source_metadata = make_recorded_video_live_metadata(
            video_path,
            expected_image_size=camera.image_size,
            frame_rate_hz=20.0,
        )
        session_id = new_reconstruction_session_id()
        request = LiveSlamRequest(
            session_id=session_id,
            source=source_metadata,
            camera=camera,
            profile=ReconstructionQualityProfile.EFFICIENT,
            backend_id=STELLA_LIVE_BACKEND_ID,
            simulated_live=True,
            enable_mapping=True,
            enable_loop_closure=False,
            enable_global_optimization=False,
            metadata={"validation": "phase8.4c-recorded-video-capture"},
        )

        observations = []
        errors: list[str] = []

        def backend_factory(ring_spec):
            return ExternalStellaLiveSlamBackend(
                install_prefix=args.stella_prefix,
                worker_executable=args.worker,
                vocabulary_path=args.vocab,
                frame_ring_spec=ring_spec,
            )

        runtime = LiveCaptureSlamRuntime(
            request=request,
            backend_factory=backend_factory,
            observation_callback=observations.append,
            error_callback=errors.append,
        )
        runtime.start()
        if not runtime.wait_for_state(LiveCaptureRuntimeState.RUNNING, timeout_s=20.0):
            raise RuntimeError(f"Live stella runtime did not start: {runtime.stats}")

        app = QCoreApplication.instance() or QCoreApplication([])
        video = VideoSource(str(video_path))
        video.set_loop(False)
        video.set_speed(args.playback_speed)
        captured = 0
        final_status = SourceStatus.IDLE

        def on_frame(frame) -> None:
            nonlocal captured
            captured += 1
            runtime.offer_source_frame(frame)
            if captured % 100 == 0:
                stats = runtime.stats
                print(
                    f"captured={captured} processed={stats.processed_frames} "
                    f"replaced={stats.replaced_frames} poses={sum(o.pose is not None for o in observations)}"
                )

        def on_status(status) -> None:
            nonlocal final_status
            final_status = status

        video.frame_ready.connect(on_frame)
        video.status_changed.connect(on_status)
        video.error_occurred.connect(errors.append)
        video.start()

        deadline = monotonic() + max(45.0, args.max_frames / 20.0 / args.playback_speed + 30.0)
        while monotonic() < deadline:
            app.processEvents()
            if final_status in {SourceStatus.STOPPED, SourceStatus.ERROR}:
                break
            sleep(0.005)
        video.stop()
        for _ in range(50):
            app.processEvents()
            sleep(0.005)

        # Give the newest pending frame a bounded chance to finish before stop.
        target_processed = runtime.stats.processed_frames
        settle_deadline = monotonic() + 1.0
        while monotonic() < settle_deadline:
            current = runtime.stats.processed_frames
            if current == target_processed:
                sleep(0.03)
                if runtime.stats.processed_frames == current:
                    break
            target_processed = current
            sleep(0.01)

        if not runtime.stop(timeout_s=8.0):
            raise RuntimeError("Live capture stella runtime did not stop cleanly.")

    stats = runtime.stats
    poses = [item for item in observations if item.pose is not None]
    keyframes = [item for item in observations if item.is_keyframe]
    latencies_ms = [item.quality.processing_latency_ns / 1_000_000.0 for item in observations]
    tracking_states: dict[str, int] = {}
    for observation in observations:
        key = observation.tracking_state.value
        tracking_states[key] = tracking_states.get(key, 0) + 1

    passed = bool(
        not errors
        and captured >= args.max_frames
        and stats.processed_frames >= 100
        and len(observations) == stats.observations
        and len(poses) >= max(50, int(stats.observations * 0.70))
        and runtime.state is LiveCaptureRuntimeState.STOPPED
    )
    summary = {
        "phase": "8.4C",
        "pass": passed,
        "capture_source": "VideoSource",
        "capture_owner": "existing_capture_layer",
        "frames_captured": captured,
        "frames_offered": stats.offered_frames,
        "frames_processed": stats.processed_frames,
        "frames_replaced_latest_only": stats.replaced_frames,
        "observations_received": len(observations),
        "pose_observations": len(poses),
        "keyframe_observations": len(keyframes),
        "tracking_states": tracking_states,
        "median_tracking_latency_ms": statistics.median(latencies_ms) if latencies_ms else None,
        "mean_tracking_latency_ms": statistics.mean(latencies_ms) if latencies_ms else None,
        "real_stella_executed": bool(observations),
        "worker_process_isolated": True,
        "capture_thread_blocking": False,
        "latest_frame_only": True,
        "pose_convention": "T_reconstruction_from_camera",
        "scale_status": "arbitrary",
        "unit_to_metre": None,
        "loop_closure_enabled": False,
        "global_optimization_feature_enabled": False,
        "errors": errors,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if not passed:
        raise RuntimeError(f"Phase 8.4C real capture smoke failed; see {summary_path}")
    print("\nPhase 8.4C real recorded-video capture stella smoke: PASS")
    print(
        f"captured={captured} processed={stats.processed_frames} replaced={stats.replaced_frames} "
        f"observations={len(observations)} poses={len(poses)} keyframes={len(keyframes)}"
    )
    print(f"tracking_states={tracking_states}")
    if latencies_ms:
        print(
            f"median_tracking_latency_ms={statistics.median(latencies_ms):.3f} "
            f"mean_tracking_latency_ms={statistics.mean(latencies_ms):.3f}"
        )
    print("pose_convention=T_reconstruction_from_camera")
    print("gauge=arbitrary unit_to_metre=None")
    print("capture_thread_blocking=False latest_frame_only=True")
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

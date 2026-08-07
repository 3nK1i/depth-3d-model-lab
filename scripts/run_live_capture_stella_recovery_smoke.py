#!/usr/bin/env python3
"""Real Phase 8.4D capture -> stella crash/restart hardening smoke.

A calibrated EuRoC sequence is played through the production VideoSource. The
first isolated stella worker is deliberately killed once; the hardened runtime
must restart only the worker, keep capture ownership unchanged, resume tracking,
and clean every process/socket/shared-memory resource on shutdown.
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import shared_memory
import os
from pathlib import Path
import signal
import statistics
import sys
import tempfile
from time import monotonic, sleep

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyQt6.QtCore import QCoreApplication

from core.capture.base_source import SourceStatus
from core.capture.video_source import VideoSource
from core.reconstruction import (
    ExternalStellaLiveSlamBackend,
    LiveCaptureRuntimeState,
    LiveCaptureSlamRuntime,
    LiveSlamRecoveryPolicy,
    LiveSlamRequest,
    ReconstructionQualityProfile,
    STELLA_LIVE_BACKEND_ID,
    make_recorded_video_live_metadata,
    new_reconstruction_session_id,
)
from scripts.run_live_capture_stella_smoke import _build_euroc_video
from scripts.run_live_stella_worker_smoke import _euroc_camera, _load_euroc_index


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
    parser.add_argument("--max-frames", type=int, default=800)
    parser.add_argument("--crash-after", type=int, default=250)
    parser.add_argument("--playback-speed", type=float, default=2.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/phase8_4d_live_capture_stella_recovery"),
    )
    args = parser.parse_args()
    if args.max_frames < 300:
        parser.error("--max-frames must be at least 300")
    if args.crash_after < 50 or args.crash_after >= args.max_frames - 50:
        parser.error("--crash-after must leave at least 50 frames before and after the crash")
    if args.playback_speed <= 0.0:
        parser.error("--playback-speed must be positive")

    records = _load_euroc_index(args.euroc.expanduser().resolve(), args.max_frames)
    camera = _euroc_camera()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sockets_before = {path.name for path in Path("/tmp").glob("eia-ls-*.sock")}

    with tempfile.TemporaryDirectory(prefix="eia-phase8-4d-video-") as temp_text:
        video_path = Path(temp_text) / "euroc_cam0.avi"
        _build_euroc_video(records, video_path)
        request = LiveSlamRequest(
            session_id=new_reconstruction_session_id(),
            source=make_recorded_video_live_metadata(
                video_path,
                expected_image_size=camera.image_size,
                frame_rate_hz=20.0,
            ),
            camera=camera,
            profile=ReconstructionQualityProfile.EFFICIENT,
            backend_id=STELLA_LIVE_BACKEND_ID,
            simulated_live=True,
            enable_mapping=True,
            enable_loop_closure=False,
            enable_global_optimization=False,
            metadata={"validation": "phase8.4d-real-worker-recovery"},
        )

        observations = []
        errors: list[str] = []
        states: list[LiveCaptureRuntimeState] = []
        backend_instances = []
        ring_names: list[str] = []
        crash_gate = {"done": False, "pid": None}

        class CrashOnceExternalStella(ExternalStellaLiveSlamBackend):
            def track_live_frame(self, frame, context):
                if not crash_gate["done"] and frame.sequence_index >= args.crash_after:
                    pid = self.worker_pid
                    if pid is None:
                        raise RuntimeError("real stella worker PID unavailable at crash gate")
                    crash_gate["done"] = True
                    crash_gate["pid"] = pid
                    os.kill(pid, signal.SIGKILL)
                    sleep(0.02)
                return super().track_live_frame(frame, context)

        def backend_factory(ring_spec):
            ring_names.append(ring_spec.shared_memory_name)
            backend = CrashOnceExternalStella(
                install_prefix=args.stella_prefix,
                worker_executable=args.worker,
                vocabulary_path=args.vocab,
                frame_ring_spec=ring_spec,
            )
            backend_instances.append(backend)
            return backend

        runtime = LiveCaptureSlamRuntime(
            request=request,
            backend_factory=backend_factory,
            recovery_policy=LiveSlamRecoveryPolicy(
                max_worker_restarts=2,
                restart_backoff_s=0.05,
                max_consecutive_stale_frames=3,
            ),
            observation_callback=observations.append,
            state_callback=states.append,
            error_callback=errors.append,
        )
        runtime.start()
        if not runtime.wait_for_state(LiveCaptureRuntimeState.RUNNING, timeout_s=20.0):
            raise RuntimeError(f"live stella runtime did not start: {runtime.stats}")

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
                    f"replaced={stats.replaced_frames} restarts={stats.worker_restarts} "
                    f"poses={sum(item.pose is not None for item in observations)}"
                )

        def on_status(status) -> None:
            nonlocal final_status
            final_status = status

        video.frame_ready.connect(on_frame)
        video.status_changed.connect(on_status)
        video.error_occurred.connect(errors.append)
        video.start()

        deadline = monotonic() + max(
            45.0,
            args.max_frames / 20.0 / args.playback_speed + 35.0,
        )
        while monotonic() < deadline:
            app.processEvents()
            if final_status in {SourceStatus.STOPPED, SourceStatus.ERROR}:
                break
            sleep(0.005)
        video.stop()
        for _ in range(40):
            app.processEvents()
            sleep(0.005)

        settle_deadline = monotonic() + 1.5
        last_processed = -1
        while monotonic() < settle_deadline:
            current = runtime.stats.processed_frames
            if current == last_processed:
                break
            last_processed = current
            sleep(0.05)

        if not runtime.stop(timeout_s=10.0):
            raise RuntimeError("hardened live capture runtime did not stop cleanly")

    stats = runtime.stats
    poses = [item for item in observations if item.pose is not None]
    latencies_ms = [item.quality.processing_latency_ns / 1_000_000.0 for item in observations]
    sockets_after = {path.name for path in Path("/tmp").glob("eia-ls-*.sock")}
    leaked_sockets = sorted(sockets_after - sockets_before)
    leaked_shared_memory: list[str] = []
    for name in sorted(set(ring_names)):
        try:
            memory = shared_memory.SharedMemory(name=name, create=False)
        except FileNotFoundError:
            continue
        else:
            memory.close()
            leaked_shared_memory.append(name)

    tracking_states: dict[str, int] = {}
    for observation in observations:
        key = observation.tracking_state.value
        tracking_states[key] = tracking_states.get(key, 0) + 1

    backend_resources_clean = all(not item.resources_active for item in backend_instances)
    passed = bool(
        not errors
        and crash_gate["done"]
        and captured >= args.max_frames
        and stats.worker_restarts == 1
        and stats.worker_restart_failures == 0
        and LiveCaptureRuntimeState.RECOVERING in states
        and stats.cleanup_complete
        and len(observations) >= max(150, int(stats.offered_frames * 0.60))
        and len(poses) >= max(120, int(len(observations) * 0.60))
        and runtime.state is LiveCaptureRuntimeState.STOPPED
        and backend_resources_clean
        and not leaked_sockets
        and not leaked_shared_memory
    )
    summary = {
        "phase": "8.4D",
        "pass": passed,
        "frames_captured": captured,
        "frames_offered": stats.offered_frames,
        "frames_processed": stats.processed_frames,
        "frames_replaced_latest_only": stats.replaced_frames,
        "observations_received": len(observations),
        "pose_observations": len(poses),
        "tracking_states": tracking_states,
        "worker_restarts": stats.worker_restarts,
        "worker_restart_failures": stats.worker_restart_failures,
        "deliberate_worker_crash_pid": crash_gate["pid"],
        "recovery_state_observed": LiveCaptureRuntimeState.RECOVERING in states,
        "cleanup_complete": stats.cleanup_complete,
        "backend_resources_clean": backend_resources_clean,
        "leaked_worker_sockets": leaked_sockets,
        "leaked_shared_memory": leaked_shared_memory,
        "median_tracking_latency_ms": statistics.median(latencies_ms) if latencies_ms else None,
        "mean_tracking_latency_ms": statistics.mean(latencies_ms) if latencies_ms else None,
        "real_stella_executed": bool(observations),
        "worker_process_isolated": True,
        "capture_source": "VideoSource",
        "capture_owner": "existing_capture_layer",
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
        raise RuntimeError(f"Phase 8.4D real worker recovery smoke failed; see {summary_path}")
    print("\nPhase 8.4D real live-capture stella recovery smoke: PASS")
    print(
        f"captured={captured} processed={stats.processed_frames} replaced={stats.replaced_frames} "
        f"observations={len(observations)} poses={len(poses)}"
    )
    print(
        f"worker_restarts={stats.worker_restarts} worker_restart_failures={stats.worker_restart_failures} "
        f"recovery_state_observed=True"
    )
    print("cleanup_complete=True leaked_worker_sockets=0 leaked_shared_memory=0")
    if latencies_ms:
        print(
            f"median_tracking_latency_ms={statistics.median(latencies_ms):.3f} "
            f"mean_tracking_latency_ms={statistics.mean(latencies_ms):.3f}"
        )
    print("pose_convention=T_reconstruction_from_camera")
    print("gauge=arbitrary unit_to_metre=None")
    print("loop_closure_enabled=False global_optimization_feature_enabled=False")
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

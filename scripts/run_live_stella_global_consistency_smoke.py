#!/usr/bin/env python3
"""Real Phase 8.5B native stella global-consistency smoke.

The production Python process still communicates with the isolated native
worker through POSIX shared memory + AF_UNIX JSON.  This smoke enables the
qualified stella loop/global-consistency features, replays a visually repeated
EuRoC segment, exercises the explicit native loop-closure control, validates
optimized keyframe snapshots, then forces a short tracking-loss/recovery cycle.

It intentionally does *not* apply optimized poses back to live tracking; that
belongs to Phase 8.5C.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import statistics
import sys
from time import monotonic

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.capture.frame import SourceFrame
from core.reconstruction import (
    BackendOperationContext,
    ExternalStellaLiveSlamBackend,
    InputSourceKind,
    InputSourceMetadata,
    LiveSharedMemoryFrameRing,
    LiveSlamRequest,
    LiveSourceFrameAdapter,
    LiveTrackingState,
    ReconstructionOperationControl,
    ReconstructionQualityProfile,
    ReconstructionStage,
    STELLA_LIVE_BACKEND_ID,
    TimestampOrigin,
    new_reconstruction_session_id,
)
from scripts.run_live_stella_worker_smoke import _euroc_camera, _load_euroc_index


def _submit_image(
    *,
    image: np.ndarray,
    frame_id: int,
    timestamp_s: float,
    adapter: LiveSourceFrameAdapter,
    backend: ExternalStellaLiveSlamBackend,
    context: BackendOperationContext,
):
    source_frame = SourceFrame(image=image, frame_id=frame_id, timestamp=timestamp_s)
    return backend.track_live_frame(adapter.publish(source_frame), context)


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
    parser.add_argument("--base-frames", type=int, default=700)
    parser.add_argument("--replay-frames", type=int, default=220)
    parser.add_argument("--blank-frames", type=int, default=25)
    parser.add_argument("--recovery-frames", type=int, default=160)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/phase8_5b_live_stella_global_consistency"),
    )
    args = parser.parse_args()
    if args.base_frames < 200:
        parser.error("--base-frames must be at least 200")
    if args.replay_frames < 80:
        parser.error("--replay-frames must be at least 80")
    if args.blank_frames < 5:
        parser.error("--blank-frames must be at least 5")
    if args.recovery_frames < 30:
        parser.error("--recovery-frames must be at least 30")

    mav0 = args.euroc.expanduser().resolve()
    load_count = max(args.base_frames, args.replay_frames, args.recovery_frames)
    records = _load_euroc_index(mav0, load_count)
    base_records = records[: args.base_frames]
    replay_records = records[: args.replay_frames]
    recovery_records = records[: args.recovery_frames]

    camera = _euroc_camera()
    session_id = new_reconstruction_session_id()
    source_fingerprint = sha256(str(mav0).encode("utf-8")).hexdigest()
    source = InputSourceMetadata(
        source_id=f"euroc-mh01-phase8-5b-{source_fingerprint[:12]}",
        kind=InputSourceKind.RECORDED_VIDEO,
        display_name="EuRoC MH_01 Phase 8.5B simulated-live loop stream",
        source_uri=f"euroc-phase8-5b://{mav0}",
        expected_image_size=camera.image_size,
        expected_frame_rate_hz=20.0,
        timestamp_origin=TimestampOrigin.CAPTURE_CLOCK,
        fingerprint=source_fingerprint,
        metadata={"dataset": "EuRoC MH_01_easy", "camera": "cam0", "phase": "8.5B"},
    )
    profile = ReconstructionQualityProfile.EFFICIENT
    request = LiveSlamRequest(
        session_id=session_id,
        source=source,
        camera=camera,
        profile=profile,
        backend_id=STELLA_LIVE_BACKEND_ID,
        simulated_live=True,
        enable_mapping=True,
        enable_loop_closure=True,
        enable_global_optimization=True,
        metadata={"validation": "phase8.5b-native-global-consistency"},
    )

    ring = LiveSharedMemoryFrameRing.create(
        slot_count=3,
        max_payload_bytes=camera.image_size.width * camera.image_size.height * 3,
    )
    adapter = LiveSourceFrameAdapter(
        session_id=session_id,
        source=source,
        camera=camera,
        ring=ring,
    )
    control = ReconstructionOperationControl()
    context = BackendOperationContext(
        session_id=session_id,
        stage=ReconstructionStage.LIVE_TRACKING,
        profile=profile,
        control=control,
        emit_event=lambda event: None,
    )
    backend = ExternalStellaLiveSlamBackend(
        install_prefix=args.stella_prefix,
        worker_executable=args.worker,
        vocabulary_path=args.vocab,
        frame_ring_spec=ring.spec,
        startup_timeout_s=20.0,
        frame_timeout_s=8.0,
        shutdown_timeout_s=10.0,
    )

    observations = []
    loop_results = []
    loop_ba_seen = False
    recovery_seen = False
    lost_seen = False
    blank_lost_seen = False
    post_blank_recovery_seen = False
    pre_blank_relocalization_count = None
    final_status = None
    pre_loop_status = None
    frame_counter = 0
    capture_origin_s = monotonic()

    def submit_path(path: Path):
        nonlocal frame_counter, recovery_seen, lost_seen
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not decode EuRoC image: {path}")
        if image.shape[:2] != (camera.image_size.height, camera.image_size.width):
            raise RuntimeError(f"Unexpected EuRoC image size: {image.shape}")
        timestamp_s = capture_origin_s + frame_counter * 0.05
        observation = _submit_image(
            image=image,
            frame_id=frame_counter,
            timestamp_s=timestamp_s,
            adapter=adapter,
            backend=backend,
            context=context,
        )
        frame_counter += 1
        observations.append(observation)
        lost_seen = lost_seen or observation.tracking_state is LiveTrackingState.LOST
        recovery_seen = recovery_seen or observation.tracking_state is LiveTrackingState.RECOVERING
        return observation

    try:
        backend.prepare(context)
        backend.start_live_session(request, context)

        for _, path in base_records:
            submit_path(path)
            if frame_counter % 100 == 0:
                print(f"base frames={frame_counter}", flush=True)

        # Return visually to the beginning of the sequence.  This gives the
        # native loop detector and the explicit test hook a strong repeated area.
        replay_start = frame_counter
        for _, path in replay_records:
            submit_path(path)
            if (frame_counter - replay_start) % 100 == 0:
                status = backend.get_native_global_consistency_status(context)
                loop_ba_seen = loop_ba_seen or status.loop_ba_running
                print(
                    f"replay={frame_counter - replay_start} keyframes={status.keyframe_count} "
                    f"revision={status.optimized_pose_revision}",
                    flush=True,
                )

        pre_loop_status = backend.get_native_global_consistency_status(context)
        loop_ba_seen = loop_ba_seen or pre_loop_status.loop_ba_running
        keyframe_ids = [pose.keyframe_id for pose in pre_loop_status.optimized_keyframe_poses]
        if len(keyframe_ids) < 4:
            raise RuntimeError("Real 8.5B smoke did not produce enough native keyframes.")

        # Try multiple well-separated early/recent pairs.  The public worker
        # control returns stella's own acceptance result; no Python-side fake
        # acceptance is permitted.
        early = keyframe_ids[: min(6, len(keyframe_ids))]
        recent = list(reversed(keyframe_ids[-min(12, len(keyframe_ids)) :]))
        accepted_result = None
        for from_id in recent:
            for to_id in early:
                if from_id == to_id:
                    continue
                result = backend.request_native_loop_closure(from_id, to_id, context)
                loop_results.append(result)
                if result.accepted:
                    accepted_result = result
                    break
            if accepted_result is not None:
                break

        # Give loop/global optimization a bounded observation window and poll
        # its real status.  Raw frame observations remain untouched.
        post_request_revision = pre_loop_status.optimized_pose_revision
        activity_seen = False
        for _, path in replay_records[:120]:
            submit_path(path)
            if frame_counter % 10 == 0:
                status = backend.get_native_global_consistency_status(context)
                loop_ba_seen = loop_ba_seen or status.loop_ba_running
                activity_seen = activity_seen or status.optimized_pose_revision > post_request_revision

        # Record relocalization telemetry immediately before the deliberate
        # loss interval.  Phase 8.5B acceptance requires a *new* native
        # relocalization caused by this blank->known-image transition rather
        # than accepting an unrelated recovery that happened earlier.
        pre_blank_status = backend.get_native_global_consistency_status(context)
        pre_blank_relocalization_count = pre_blank_status.relocalization_count

        # Induce a real tracking loss without changing capture ownership.
        blank = np.zeros((camera.image_size.height, camera.image_size.width, 3), dtype=np.uint8)
        for _ in range(args.blank_frames):
            timestamp_s = capture_origin_s + frame_counter * 0.05
            obs = _submit_image(
                image=blank,
                frame_id=frame_counter,
                timestamp_s=timestamp_s,
                adapter=adapter,
                backend=backend,
                context=context,
            )
            frame_counter += 1
            observations.append(obs)
            is_lost = obs.tracking_state is LiveTrackingState.LOST
            lost_seen = lost_seen or is_lost
            blank_lost_seen = blank_lost_seen or is_lost

        for _, path in recovery_records:
            observation = submit_path(path)
            if observation.tracking_state is LiveTrackingState.RECOVERING:
                post_blank_recovery_seen = True
                break

        final_status = backend.get_native_global_consistency_status(context)
        loop_ba_seen = loop_ba_seen or final_status.loop_ba_running
        activity_seen = activity_seen or (
            final_status.optimized_pose_revision > post_request_revision
        )
        backend.stop_live_session(context)
    finally:
        backend.shutdown()
        ring.shutdown()

    assert pre_loop_status is not None
    assert final_status is not None
    accepted_result = next((result for result in loop_results if result.accepted), None)
    pose_observations = sum(item.pose is not None for item in observations)
    tracking_states = {state.value: 0 for state in LiveTrackingState}
    for observation in observations:
        tracking_states[observation.tracking_state.value] += 1
    latencies_ms = [item.quality.processing_latency_ns / 1_000_000.0 for item in observations]
    optimized_pose_count = len(final_status.optimized_keyframe_poses)
    optimized_conventions_ok = all(
        pose.as_dict()["pose_convention"] == "T_reconstruction_from_camera"
        and pose.as_dict()["pose_layer"] == "optimized"
        for pose in final_status.optimized_keyframe_poses
    )
    raw_conventions_ok = all(
        item.pose is None
        or item.pose.as_dict()["pose_convention"] == "T_reconstruction_from_camera"
        for item in observations
    )
    arbitrary_gauge_ok = (
        final_status.gauge.unit_to_metre is None
        and all(item.pose is None or item.pose.gauge.unit_to_metre is None for item in observations)
    )
    manual_loop_accepted = accepted_result is not None
    native_optimization_activity = bool(
        loop_ba_seen
        or final_status.optimized_pose_revision > pre_loop_status.optimized_pose_revision
    )
    assert pre_blank_relocalization_count is not None
    relocalization_telemetry_ok = (
        blank_lost_seen
        and post_blank_recovery_seen
        and final_status.relocalization_count > pre_blank_relocalization_count
        and bool(final_status.last_relocalization_frame_id)
    )
    passed = bool(
        pose_observations >= 100
        and pre_loop_status.loop_detector_enabled
        and pre_loop_status.global_optimization_enabled
        and manual_loop_accepted
        and final_status.accepted_loop_closure_requests >= 1
        and native_optimization_activity
        and optimized_pose_count >= 4
        and optimized_conventions_ok
        and raw_conventions_ok
        and arbitrary_gauge_ok
        and relocalization_telemetry_ok
    )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "phase": "8.5B",
        "pass": passed,
        "real_stella_executed": True,
        "worker_process_isolated": True,
        "frames_submitted": len(observations),
        "pose_observations": pose_observations,
        "tracking_states": tracking_states,
        "loop_detector_enabled": final_status.loop_detector_enabled,
        "global_optimization_enabled": final_status.global_optimization_enabled,
        "manual_loop_request_accepted": manual_loop_accepted,
        "manual_loop_attempts": len(loop_results),
        "accepted_loop_closure_requests": final_status.accepted_loop_closure_requests,
        "loop_ba_running_observed": loop_ba_seen,
        "pre_loop_optimized_pose_revision": pre_loop_status.optimized_pose_revision,
        "final_optimized_pose_revision": final_status.optimized_pose_revision,
        "native_optimization_activity_observed": native_optimization_activity,
        "optimized_keyframe_pose_count": optimized_pose_count,
        "map_revision": final_status.map_revision,
        "landmark_count": final_status.landmark_count,
        "lost_observed": lost_seen,
        "recovering_observed": recovery_seen,
        "blank_interval_lost_observed": blank_lost_seen,
        "post_blank_recovery_observed": post_blank_recovery_seen,
        "pre_blank_relocalization_count": pre_blank_relocalization_count,
        "native_relocalization_count": final_status.relocalization_count,
        "last_relocalization_frame_id": final_status.last_relocalization_frame_id,
        "raw_pose_layer": "tracking",
        "optimized_pose_layer": "optimized",
        "pose_convention": "T_reconstruction_from_camera",
        "scale_status": "arbitrary",
        "unit_to_metre": None,
        "median_tracking_latency_ms": statistics.median(latencies_ms) if latencies_ms else None,
        "mean_tracking_latency_ms": statistics.fmean(latencies_ms) if latencies_ms else None,
        "environment_probe": None if backend.environment_probe is None else backend.environment_probe.as_dict(),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print()
    print(f"Phase 8.5B real native stella global-consistency smoke: {'PASS' if passed else 'FAIL'}")
    print(
        f"frames={len(observations)} poses={pose_observations} "
        f"keyframes={optimized_pose_count} landmarks={final_status.landmark_count}"
    )
    print(
        f"loop_enabled={final_status.loop_detector_enabled} "
        f"global_optimization_enabled={final_status.global_optimization_enabled} "
        f"manual_loop_accepted={manual_loop_accepted} attempts={len(loop_results)}"
    )
    print(
        f"optimized_pose_revision={pre_loop_status.optimized_pose_revision}"
        f"->{final_status.optimized_pose_revision} loop_ba_seen={loop_ba_seen}"
    )
    print(
        f"blank_lost_seen={blank_lost_seen} "
        f"post_blank_recovery_seen={post_blank_recovery_seen} "
        f"native_relocalizations={pre_blank_relocalization_count}"
        f"->{final_status.relocalization_count}"
    )
    print("raw_pose_layer=tracking optimized_pose_layer=optimized")
    print("pose_convention=T_reconstruction_from_camera")
    print("gauge=arbitrary unit_to_metre=None")
    print(f"summary={summary_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

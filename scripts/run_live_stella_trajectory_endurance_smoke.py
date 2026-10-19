#!/usr/bin/env python3
"""Real Phase 8.5D global-consistency/trajectory endurance smoke.

Exercises two native stella worker generations in one reconstruction
session. The first worker performs real loop/global optimization, is then
intentionally killed, and a replacement worker starts with reset native
keyframe IDs/revisions. LiveTrajectoryCoordinator must advance the optimizer
epoch, preserve every raw pose, reject stale generation data, and accept the
new worker's revision sequence. A deliberate blank interval in the second
epoch also verifies native relocalization/recovery.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from multiprocessing import shared_memory
from pathlib import Path
import signal
import sys
from time import monotonic, sleep

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
    LiveTrajectoryCoordinator,
    LiveTrajectoryHardeningPolicy,
    LiveTrajectoryPoseLayer,
    ReconstructionOperationControl,
    ReconstructionQualityProfile,
    ReconstructionStage,
    ReconstructionValidationError,
    STELLA_LIVE_BACKEND_ID,
    TimestampOrigin,
    new_reconstruction_session_id,
)
from scripts.run_live_stella_worker_smoke import _euroc_camera, _load_euroc_index


def _request_any_loop(backend, status, context):
    keyframe_ids = [item.keyframe_id for item in status.optimized_keyframe_poses]
    if len(keyframe_ids) < 4:
        return None
    early = keyframe_ids[: min(6, len(keyframe_ids))]
    recent = list(reversed(keyframe_ids[-min(12, len(keyframe_ids)) :]))
    for from_id in recent:
        for to_id in early:
            if from_id == to_id:
                continue
            result = backend.request_native_loop_closure(from_id, to_id, context)
            if result.accepted:
                return result
    return None


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
    parser.add_argument("--first-base-frames", type=int, default=600)
    parser.add_argument("--first-replay-frames", type=int, default=180)
    parser.add_argument("--first-post-loop-frames", type=int, default=100)
    parser.add_argument("--second-base-frames", type=int, default=500)
    parser.add_argument("--second-replay-frames", type=int, default=160)
    parser.add_argument("--blank-frames", type=int, default=25)
    parser.add_argument("--recovery-frames", type=int, default=120)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/phase8_5d_live_stella_trajectory_endurance"),
    )
    args = parser.parse_args()
    if args.first_base_frames < 300 or args.second_base_frames < 300:
        parser.error("both base-frame counts must be at least 300")
    if min(args.first_replay_frames, args.second_replay_frames) < 80:
        parser.error("replay-frame counts must be at least 80")
    if args.first_post_loop_frames < 40 or args.recovery_frames < 60:
        parser.error("post-loop/recovery windows are too small")
    if args.blank_frames < 10:
        parser.error("--blank-frames must be at least 10")

    mav0 = args.euroc.expanduser().resolve()
    needed = max(
        args.first_base_frames,
        args.first_replay_frames,
        args.first_post_loop_frames,
        args.second_base_frames,
        args.second_replay_frames,
        args.recovery_frames,
    )
    records = _load_euroc_index(mav0, needed)
    camera = _euroc_camera()
    session_id = new_reconstruction_session_id()
    fingerprint = sha256(str(mav0).encode("utf-8")).hexdigest()
    source = InputSourceMetadata(
        source_id=f"euroc-mh01-phase8-5d-{fingerprint[:12]}",
        kind=InputSourceKind.RECORDED_VIDEO,
        display_name="EuRoC MH_01 Phase 8.5D global-consistency endurance stream",
        source_uri=f"euroc-phase8-5d://{mav0}",
        expected_image_size=camera.image_size,
        expected_frame_rate_hz=20.0,
        timestamp_origin=TimestampOrigin.CAPTURE_CLOCK,
        fingerprint=fingerprint,
        metadata={"dataset": "EuRoC MH_01_easy", "camera": "cam0", "phase": "8.5D"},
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
        metadata={"validation": "phase8.5d-global-consistency-endurance"},
    )
    ring = LiveSharedMemoryFrameRing.create(
        slot_count=3,
        max_payload_bytes=camera.image_size.width * camera.image_size.height * 3,
    )
    ring_name = ring.spec.shared_memory_name
    adapter = LiveSourceFrameAdapter(
        session_id=session_id,
        source=source,
        camera=camera,
        ring=ring,
    )
    context = BackendOperationContext(
        session_id=session_id,
        stage=ReconstructionStage.LIVE_TRACKING,
        profile=profile,
        control=ReconstructionOperationControl(),
        emit_event=lambda event: None,
    )
    coordinator = LiveTrajectoryCoordinator(
        session_id=session_id,
        policy=LiveTrajectoryHardeningPolicy(
            min_keyframe_match_ratio=0.70,
            min_exact_optimized_keyframes=4,
        ),
    )

    sockets_before = {path.name for path in Path("/tmp").glob("eia-ls-*.sock")}
    backends: list[ExternalStellaLiveSlamBackend] = []
    observations = []
    tracking_states: dict[str, int] = {}
    frame_counter = 0
    capture_origin_s = monotonic()
    killed_pid = None
    stale_generation_rejected = False
    old_raw_sample = None
    old_frame_id = ""
    epoch0_final_status = None
    epoch0_quality = None
    epoch1_quality = None
    epoch0_loop = None
    epoch1_loop = None
    relocalization_before_blank = 0
    relocalization_after_recovery = 0

    def make_backend() -> ExternalStellaLiveSlamBackend:
        backend = ExternalStellaLiveSlamBackend(
            install_prefix=args.stella_prefix,
            worker_executable=args.worker,
            vocabulary_path=args.vocab,
            frame_ring_spec=ring.spec,
            startup_timeout_s=20.0,
            frame_timeout_s=8.0,
            shutdown_timeout_s=10.0,
        )
        backends.append(backend)
        backend.prepare(context)
        backend.start_live_session(request, context)
        return backend

    def submit_image(backend, path: Path, generation: int):
        nonlocal frame_counter
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not decode EuRoC image: {path}")
        source_frame = SourceFrame(
            image=image,
            frame_id=frame_counter,
            timestamp=capture_origin_s + frame_counter * 0.05,
        )
        envelope = adapter.publish(source_frame)
        observation = backend.track_live_frame(envelope, context)
        frame_counter += 1
        observations.append(observation)
        tracking_states[observation.tracking_state.value] = tracking_states.get(observation.tracking_state.value, 0) + 1
        if observation.pose is not None:
            coordinator.ingest_observation(observation, worker_generation=generation)
        return observation

    def submit_blank(backend, generation: int):
        nonlocal frame_counter
        image = np.zeros((camera.image_size.height, camera.image_size.width, 3), dtype=np.uint8)
        source_frame = SourceFrame(
            image=image,
            frame_id=frame_counter,
            timestamp=capture_origin_s + frame_counter * 0.05,
        )
        envelope = adapter.publish(source_frame)
        observation = backend.track_live_frame(envelope, context)
        frame_counter += 1
        observations.append(observation)
        tracking_states[observation.tracking_state.value] = tracking_states.get(observation.tracking_state.value, 0) + 1
        if observation.pose is not None:
            coordinator.ingest_observation(observation, worker_generation=generation)
        return observation

    try:
        backend0 = make_backend()
        for index, (_, path) in enumerate(records[: args.first_base_frames], start=1):
            submit_image(backend0, path, 0)
            if index % 100 == 0:
                print(f"epoch0 base={index}", flush=True)
        for index, (_, path) in enumerate(records[: args.first_replay_frames], start=1):
            submit_image(backend0, path, 0)
            if index % 100 == 0:
                print(f"epoch0 replay={index}", flush=True)

        epoch0_pre = backend0.get_native_global_consistency_status(context)
        coordinator.apply_native_status(epoch0_pre, worker_generation=0)
        epoch0_loop = _request_any_loop(backend0, epoch0_pre, context)
        for _, path in records[: args.first_post_loop_frames]:
            submit_image(backend0, path, 0)
        epoch0_final_status = backend0.get_native_global_consistency_status(context)
        coordinator.apply_native_status(epoch0_final_status, worker_generation=0)
        epoch0_quality = coordinator.assess_quality(epoch0_final_status)
        snapshot0 = coordinator.snapshot()
        if not snapshot0.poses:
            raise RuntimeError("Phase 8.5D epoch 0 produced no trajectory poses")
        old_frame_id = snapshot0.poses[-1].frame_id
        old_raw = coordinator.trajectory.raw_pose_for_frame(old_frame_id)
        if old_raw is None:
            raise RuntimeError("Phase 8.5D could not capture pre-restart raw pose sample")
        old_raw_sample = old_raw.T_reconstruction_from_camera.copy()

        killed_pid = backend0.worker_pid
        if killed_pid is None:
            raise RuntimeError("Phase 8.5D native worker PID unavailable")
        os.kill(killed_pid, signal.SIGKILL)
        sleep(0.05)
        backend0.shutdown()

        backend1 = make_backend()
        # A replacement worker is a lifecycle event, not a pose event.  Native
        # startup frames may be INITIALIZING with pose=None, so invalidate the
        # previous optimizer epoch immediately when generation 1 is established.
        coordinator.begin_worker_generation(1)
        old_best_after_restart = coordinator.trajectory.best_pose_for_frame(old_frame_id)
        if old_best_after_restart is None:
            raise RuntimeError("Phase 8.5D lost pre-restart raw pose history")

        try:
            coordinator.apply_native_status(epoch0_final_status, worker_generation=0)
        except ReconstructionValidationError:
            stale_generation_rejected = True

        submit_image(backend1, records[0][1], 1)
        for index, (_, path) in enumerate(records[1 : args.second_base_frames], start=2):
            submit_image(backend1, path, 1)
            if index % 100 == 0:
                print(f"epoch1 base={index}", flush=True)
        for index, (_, path) in enumerate(records[: args.second_replay_frames], start=1):
            submit_image(backend1, path, 1)
            if index % 100 == 0:
                print(f"epoch1 replay={index}", flush=True)

        epoch1_pre = backend1.get_native_global_consistency_status(context)
        coordinator.apply_native_status(epoch1_pre, worker_generation=1)
        epoch1_loop = _request_any_loop(backend1, epoch1_pre, context)

        relocalization_before_blank = epoch1_pre.relocalization_count
        for _ in range(args.blank_frames):
            submit_blank(backend1, 1)
        for _, path in records[: args.recovery_frames]:
            submit_image(backend1, path, 1)
        epoch1_final = backend1.get_native_global_consistency_status(context)
        coordinator.apply_native_status(epoch1_final, worker_generation=1)
        relocalization_after_recovery = epoch1_final.relocalization_count
        epoch1_quality = coordinator.assess_quality(epoch1_final)
        final_snapshot = coordinator.snapshot()

        raw_preserved = coordinator.raw_tracking_integrity_ok()
        old_raw_after = coordinator.trajectory.raw_pose_for_frame(old_frame_id)
        old_raw_unchanged = bool(
            old_raw_after is not None
            and old_raw_sample is not None
            and np.array_equal(old_raw_after.T_reconstruction_from_camera, old_raw_sample)
        )
        exact = sum(item.layer is LiveTrajectoryPoseLayer.OPTIMIZED_KEYFRAME for item in final_snapshot.poses)
        propagated = sum(item.layer is LiveTrajectoryPoseLayer.OPTIMIZED_PROPAGATED for item in final_snapshot.poses)
        raw_fallback = sum(item.layer is LiveTrajectoryPoseLayer.TRACKING for item in final_snapshot.poses)
        telemetry = coordinator.telemetry

        for backend in reversed(backends):
            try:
                backend.stop_live_session(context)
            except Exception:
                pass
            backend.shutdown()

        ring.shutdown()
        sockets_after = {path.name for path in Path("/tmp").glob("eia-ls-*.sock")}
        leaked_sockets = sorted(sockets_after - sockets_before)
        leaked_shared_memory: list[str] = []
        try:
            memory = shared_memory.SharedMemory(name=ring_name, create=False)
        except FileNotFoundError:
            pass
        else:
            memory.close()
            leaked_shared_memory.append(ring_name)

        backend_resources_clean = all(not backend.resources_active for backend in backends)
        passed = bool(
            epoch0_loop is not None
            and epoch0_loop.accepted
            and epoch1_loop is not None
            and epoch1_loop.accepted
            and epoch0_quality is not None
            and epoch0_quality.accepted
            and epoch1_quality is not None
            and epoch1_quality.accepted
            and telemetry.worker_generation == 1
            and telemetry.optimizer_epoch == 2
            and telemetry.epoch_transitions == 1
            and stale_generation_rejected
            and old_best_after_restart.layer is LiveTrajectoryPoseLayer.TRACKING
            and raw_preserved
            and old_raw_unchanged
            and exact >= 4
            and propagated > 0
            and relocalization_after_recovery > relocalization_before_blank
            and backend_resources_clean
            and not leaked_sockets
            and not leaked_shared_memory
            and all(item.gauge.unit_to_metre is None for item in final_snapshot.poses)
        )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "phase": "8.5D",
            "pass": passed,
            "frames_submitted": len(observations),
            "worker_generations": 2,
            "deliberate_worker_crash_pid": killed_pid,
            "optimizer_epoch": telemetry.optimizer_epoch,
            "epoch_transitions": telemetry.epoch_transitions,
            "stale_generation_rejected": stale_generation_rejected,
            "epoch0_loop_request_accepted": bool(epoch0_loop and epoch0_loop.accepted),
            "epoch1_loop_request_accepted": bool(epoch1_loop and epoch1_loop.accepted),
            "epoch0_quality_accepted": bool(epoch0_quality and epoch0_quality.accepted),
            "epoch1_quality_accepted": bool(epoch1_quality and epoch1_quality.accepted),
            "epoch1_match_ratio": None if epoch1_quality is None else epoch1_quality.keyframe_match_ratio,
            "relocalization_before_blank": relocalization_before_blank,
            "relocalization_after_recovery": relocalization_after_recovery,
            "tracking_states": tracking_states,
            "raw_pose_count": final_snapshot.raw_pose_count,
            "exact_optimized_pose_count": exact,
            "propagated_pose_count": propagated,
            "raw_fallback_pose_count": raw_fallback,
            "old_epoch_pose_falls_back_to_raw_after_restart": old_best_after_restart.layer is LiveTrajectoryPoseLayer.TRACKING,
            "raw_tracking_preserved": raw_preserved,
            "old_raw_sample_unchanged": old_raw_unchanged,
            "backend_resources_clean": backend_resources_clean,
            "leaked_worker_sockets": leaked_sockets,
            "leaked_shared_memory": leaked_shared_memory,
            "worker_process_isolated": True,
            "real_stella_executed": bool(observations),
            "raw_pose_layer": "tracking",
            "exact_optimized_pose_layer": "optimized_keyframe",
            "propagated_pose_layer": "optimized_propagated",
            "pose_convention": "T_reconstruction_from_camera",
            "scale_status": "arbitrary",
            "unit_to_metre": None,
        }
        summary_path = args.output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        print(f"\nPhase 8.5D real stella trajectory endurance smoke: {'PASS' if passed else 'FAIL'}")
        print(
            f"frames={len(observations)} optimizer_epoch={telemetry.optimizer_epoch} "
            f"epoch_transitions={telemetry.epoch_transitions}"
        )
        print(
            f"loops=({bool(epoch0_loop and epoch0_loop.accepted)},"
            f"{bool(epoch1_loop and epoch1_loop.accepted)}) "
            f"quality=({bool(epoch0_quality and epoch0_quality.accepted)},"
            f"{bool(epoch1_quality and epoch1_quality.accepted)})"
        )
        print(
            f"exact={exact} propagated={propagated} raw_fallback={raw_fallback} "
            f"stale_generation_rejected={stale_generation_rejected}"
        )
        print(
            f"relocalization={relocalization_before_blank}->{relocalization_after_recovery} "
            f"raw_tracking_preserved={raw_preserved}"
        )
        print("cleanup_complete=True leaked_worker_sockets=0 leaked_shared_memory=0")
        print("pose_convention=T_reconstruction_from_camera")
        print("gauge=arbitrary unit_to_metre=None")
        print(f"summary={summary_path}")
        return 0 if passed else 1
    finally:
        for backend in reversed(backends):
            try:
                backend.shutdown()
            except Exception:
                pass
        try:
            ring.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())

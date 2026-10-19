#!/usr/bin/env python3
"""Real Phase 8.5C stella optimized-trajectory propagation smoke.

Uses the already-qualified isolated Phase 8.5B native worker.  Raw stella
observations are stored unchanged while native optimized keyframe revisions are
applied to the separate LiveOptimizedTrajectory layer.  The test verifies exact
optimized keyframe poses, causal correction propagation to later non-keyframes,
and raw-pose immutability.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
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
    LiveOptimizedTrajectory,
    LiveSharedMemoryFrameRing,
    LiveSlamRequest,
    LiveSourceFrameAdapter,
    LiveTrajectoryPoseLayer,
    ReconstructionOperationControl,
    ReconstructionQualityProfile,
    ReconstructionStage,
    STELLA_LIVE_BACKEND_ID,
    TimestampOrigin,
    new_reconstruction_session_id,
)
from scripts.run_live_stella_worker_smoke import _euroc_camera, _load_euroc_index


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stella-prefix",
        type=Path,
        default=Path.home() / "Desktop" / "depth3d-stella-vslam-probe",
    )
    parser.add_argument(
        "--worker",
        type=Path,
        default=Path.home() / "Desktop" / "depth3d-stella-live-worker" / "bin" / "depth3d_stella_live_worker",
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
    parser.add_argument("--post-loop-frames", type=int, default=140)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/phase8_5c_live_stella_trajectory"),
    )
    args = parser.parse_args()
    if args.base_frames < 200:
        parser.error("--base-frames must be at least 200")
    if args.replay_frames < 80:
        parser.error("--replay-frames must be at least 80")
    if args.post_loop_frames < 40:
        parser.error("--post-loop-frames must be at least 40")

    mav0 = args.euroc.expanduser().resolve()
    records = _load_euroc_index(
        mav0,
        max(args.base_frames, args.replay_frames, args.post_loop_frames),
    )
    base_records = records[: args.base_frames]
    replay_records = records[: args.replay_frames]
    post_records = records[: args.post_loop_frames]

    camera = _euroc_camera()
    session_id = new_reconstruction_session_id()
    fingerprint = sha256(str(mav0).encode("utf-8")).hexdigest()
    source = InputSourceMetadata(
        source_id=f"euroc-mh01-phase8-5c-{fingerprint[:12]}",
        kind=InputSourceKind.RECORDED_VIDEO,
        display_name="EuRoC MH_01 Phase 8.5C simulated-live trajectory stream",
        source_uri=f"euroc-phase8-5c://{mav0}",
        expected_image_size=camera.image_size,
        expected_frame_rate_hz=20.0,
        timestamp_origin=TimestampOrigin.CAPTURE_CLOCK,
        fingerprint=fingerprint,
        metadata={"dataset": "EuRoC MH_01_easy", "camera": "cam0", "phase": "8.5C"},
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
        metadata={"validation": "phase8.5c-live-trajectory-propagation"},
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
    context = BackendOperationContext(
        session_id=session_id,
        stage=ReconstructionStage.LIVE_TRACKING,
        profile=profile,
        control=ReconstructionOperationControl(),
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
    trajectory = LiveOptimizedTrajectory(session_id=session_id)

    observations = []
    raw_pose_copies: dict[str, np.ndarray] = {}
    frame_counter = 0
    capture_origin_s = monotonic()
    pre_status = None
    pre_update = None
    final_status = None
    final_update = None
    accepted_result = None

    def submit(path: Path):
        nonlocal frame_counter
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not decode EuRoC image: {path}")
        if image.shape[:2] != (camera.image_size.height, camera.image_size.width):
            raise RuntimeError(f"Unexpected EuRoC image size: {image.shape}")
        source_frame = SourceFrame(
            image=image,
            frame_id=frame_counter,
            timestamp=capture_origin_s + frame_counter * 0.05,
        )
        envelope = adapter.publish(source_frame)
        observation = backend.track_live_frame(envelope, context)
        frame_counter += 1
        observations.append(observation)
        if observation.pose is not None:
            trajectory.ingest_observation(observation)
            raw_pose_copies.setdefault(
                observation.frame_id,
                observation.pose.T_reconstruction_from_camera.copy(),
            )
        return observation

    try:
        backend.prepare(context)
        backend.start_live_session(request, context)

        for _, path in base_records:
            submit(path)
            if frame_counter % 100 == 0:
                print(f"base frames={frame_counter}", flush=True)

        replay_start = frame_counter
        for _, path in replay_records:
            submit(path)
            if (frame_counter - replay_start) % 100 == 0:
                print(f"replay={frame_counter - replay_start}", flush=True)

        pre_status = backend.get_native_global_consistency_status(context)
        pre_update = trajectory.apply_native_status(pre_status)
        keyframe_ids = [item.keyframe_id for item in pre_status.optimized_keyframe_poses]
        if len(keyframe_ids) < 4:
            raise RuntimeError("Real 8.5C smoke did not produce enough native keyframes.")

        early = keyframe_ids[: min(6, len(keyframe_ids))]
        recent = list(reversed(keyframe_ids[-min(12, len(keyframe_ids)) :]))
        for from_id in recent:
            for to_id in early:
                if from_id == to_id:
                    continue
                result = backend.request_native_loop_closure(from_id, to_id, context)
                if result.accepted:
                    accepted_result = result
                    break
            if accepted_result is not None:
                break

        for index, (_, path) in enumerate(post_records, start=1):
            submit(path)
            if index % 20 == 0:
                candidate = backend.get_native_global_consistency_status(context)
                if candidate.optimized_pose_revision > pre_status.optimized_pose_revision:
                    final_status = candidate

        final_status = final_status or backend.get_native_global_consistency_status(context)
        final_update = trajectory.apply_native_status(final_status)
        snapshot = trajectory.snapshot()

        exact_poses = [
            item for item in snapshot.poses
            if item.layer is LiveTrajectoryPoseLayer.OPTIMIZED_KEYFRAME
        ]
        propagated_poses = [
            item for item in snapshot.poses
            if item.layer is LiveTrajectoryPoseLayer.OPTIMIZED_PROPAGATED
        ]
        raw_poses = [
            item for item in snapshot.poses
            if item.layer is LiveTrajectoryPoseLayer.TRACKING
        ]
        raw_preserved = True
        for frame_id, before in raw_pose_copies.items():
            current = trajectory.raw_pose_for_frame(frame_id)
            if current is None or not np.array_equal(current.T_reconstruction_from_camera, before):
                raw_preserved = False
                break

        matched_ratio = (
            0.0
            if final_status.keyframe_count <= 0
            else final_update.exact_optimized_keyframes / final_status.keyframe_count
        )
        nonzero_final_corrections = [
            item for item in final_update.corrections
            if item.translation_correction > 1e-8 or item.rotation_correction_deg > 1e-8
        ]

        passed = (
            accepted_result is not None
            and accepted_result.accepted
            and final_status.optimized_pose_revision > pre_status.optimized_pose_revision
            and final_update.optimization_revision == final_status.optimized_pose_revision
            and final_update.exact_optimized_keyframes >= 4
            and matched_ratio >= 0.70
            and bool(exact_poses)
            and bool(propagated_poses)
            and bool(nonzero_final_corrections)
            and raw_preserved
            and all(item.gauge.unit_to_metre is None for item in snapshot.poses)
        )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "phase": "8.5C",
            "pass": passed,
            "frames_submitted": len(observations),
            "raw_pose_count": snapshot.raw_pose_count,
            "native_keyframe_count": final_status.keyframe_count,
            "matched_optimized_keyframes": final_update.exact_optimized_keyframes,
            "matched_keyframe_ratio": matched_ratio,
            "unmatched_optimized_keyframe_ids": list(final_update.unmatched_optimized_keyframe_ids),
            "exact_optimized_pose_count": len(exact_poses),
            "propagated_pose_count": len(propagated_poses),
            "raw_fallback_pose_count": len(raw_poses),
            "pre_optimization_revision": pre_status.optimized_pose_revision,
            "final_optimization_revision": final_status.optimized_pose_revision,
            "final_revision_changed_corrections": len(final_update.corrections),
            "nonzero_final_corrections": len(nonzero_final_corrections),
            "manual_loop_request_accepted": bool(accepted_result and accepted_result.accepted),
            "raw_tracking_preserved": raw_preserved,
            "raw_pose_layer": "tracking",
            "exact_optimized_pose_layer": "optimized_keyframe",
            "propagated_pose_layer": "optimized_propagated",
            "propagation_policy": "latest_preceding_optimized_keyframe",
            "pose_convention": "T_reconstruction_from_camera",
            "scale_status": "arbitrary",
            "unit_to_metre": None,
            "real_stella_executed": bool(observations),
            "worker_process_isolated": True,
        }
        summary_path = args.output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        print(f"\nPhase 8.5C real stella trajectory propagation smoke: {'PASS' if passed else 'FAIL'}")
        print(
            f"frames={len(observations)} raw={snapshot.raw_pose_count} "
            f"native_keyframes={final_status.keyframe_count} matched={final_update.exact_optimized_keyframes}"
        )
        print(
            f"revision={pre_status.optimized_pose_revision}->{final_status.optimized_pose_revision} "
            f"changed_corrections={len(final_update.corrections)} "
            f"nonzero={len(nonzero_final_corrections)}"
        )
        print(
            f"exact_optimized={len(exact_poses)} propagated={len(propagated_poses)} "
            f"raw_fallback={len(raw_poses)} matched_ratio={matched_ratio:.3f}"
        )
        print(f"raw_tracking_preserved={raw_preserved}")
        print("pose_convention=T_reconstruction_from_camera")
        print("gauge=arbitrary unit_to_metre=None")
        print(f"summary={summary_path}")
        return 0 if passed else 1
    finally:
        try:
            backend.stop_live_session(context)
        except Exception:
            pass
        backend.shutdown()
        ring.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())

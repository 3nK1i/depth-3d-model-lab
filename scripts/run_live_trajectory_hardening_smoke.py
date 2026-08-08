#!/usr/bin/env python3
"""Dependency-light Phase 8.5D optimized-trajectory hardening smoke."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction import (
    LiveMapSnapshot,
    LiveSlamObservation,
    LiveSlamPose,
    LiveTrackingQuality,
    LiveTrackingState,
    LiveTrajectoryCoordinator,
    LiveTrajectoryHardeningPolicy,
    LiveTrajectoryPoseLayer,
    NativeGlobalConsistencyStatus,
    OptimizedKeyframePose,
    ReconstructionValidationError,
)

SESSION_ID = "00000000-0000-4000-8000-000000000064"


def _matrix(x: float) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 3] = x
    return matrix


def _observation(index: int, keyframe_id: str = "") -> LiveSlamObservation:
    frame_id = f"live-frame-{index}"
    pose = LiveSlamPose(
        pose_id=f"raw-pose-{index}",
        camera_id="camera-0",
        frame_id=frame_id,
        timestamp_ns=1_000_000_000 + index * 10_000_000,
        camera_frame_id="camera-optical",
        T_reconstruction_from_camera=_matrix(index * 0.1),
    )
    return LiveSlamObservation(
        session_id=SESSION_ID,
        frame_id=frame_id,
        sequence_index=index,
        tracking_state=LiveTrackingState.TRACKING,
        quality=LiveTrackingQuality(
            tracked_features=100,
            inlier_features=80,
            processing_latency_ns=1_000_000,
        ),
        pose=pose,
        is_keyframe=bool(keyframe_id),
        keyframe_id=keyframe_id,
        map_snapshot=LiveMapSnapshot(revision=index, keyframe_count=0, landmark_count=0),
    )


def _status(revision: int, rows: tuple[tuple[str, int], ...]) -> NativeGlobalConsistencyStatus:
    poses = tuple(
        OptimizedKeyframePose(
            keyframe_id=keyframe_id,
            frame_id=keyframe_id,
            timestamp_ns=1_000_000_000 + index * 10_000_000,
            source_pose_id=f"native-{revision}-{keyframe_id}",
            optimization_revision=revision,
            T_reconstruction_from_camera=_matrix(index * 0.1 - 0.02),
        )
        for keyframe_id, index in rows
    )
    return NativeGlobalConsistencyStatus(
        session_id=SESSION_ID,
        loop_detector_enabled=True,
        global_optimization_enabled=True,
        loop_ba_running=False,
        map_revision=revision,
        optimized_pose_revision=revision,
        keyframe_count=len(poses),
        landmark_count=100,
        accepted_loop_closure_requests=revision,
        relocalization_count=0,
        last_relocalization_frame_id="",
        optimized_keyframe_poses=poses,
    )


def main() -> int:
    coordinator = LiveTrajectoryCoordinator(
        session_id=SESSION_ID,
        policy=LiveTrajectoryHardeningPolicy(
            min_keyframe_match_ratio=0.75,
            min_exact_optimized_keyframes=2,
        ),
    )

    for index in range(6):
        coordinator.ingest_observation(
            _observation(index, f"stella-keyframe-{index // 3}" if index in {0, 3} else ""),
            worker_generation=0,
        )
    first_status = _status(3, (("stella-keyframe-0", 0), ("stella-keyframe-1", 3)))
    coordinator.apply_native_status(first_status, worker_generation=0)
    first_quality = coordinator.assess_quality(first_status)
    old_raw = coordinator.trajectory.raw_pose_for_frame("live-frame-5")
    old_best = coordinator.trajectory.best_pose_for_frame("live-frame-5")
    assert old_raw is not None and old_best is not None
    raw_matrix = old_raw.T_reconstruction_from_camera.copy()
    assert old_best.layer is LiveTrajectoryPoseLayer.OPTIMIZED_PROPAGATED

    for index in range(6, 12):
        local = index - 6
        coordinator.ingest_observation(
            _observation(index, f"stella-keyframe-{local // 3}" if local in {0, 3} else ""),
            worker_generation=1,
        )
    old_after_restart = coordinator.trajectory.best_pose_for_frame("live-frame-5")
    assert old_after_restart is not None
    assert old_after_restart.layer is LiveTrajectoryPoseLayer.TRACKING

    stale_generation_rejected = False
    try:
        coordinator.apply_native_status(first_status, worker_generation=0)
    except ReconstructionValidationError:
        stale_generation_rejected = True

    second_status = _status(1, (("stella-keyframe-0", 6), ("stella-keyframe-1", 9)))
    coordinator.apply_native_status(second_status, worker_generation=1)
    second_quality = coordinator.assess_quality(second_status)
    final = coordinator.snapshot()
    raw_after = coordinator.trajectory.raw_pose_for_frame("live-frame-5")
    raw_preserved = bool(
        raw_after is not None
        and np.array_equal(raw_after.T_reconstruction_from_camera, raw_matrix)
        and coordinator.raw_tracking_integrity_ok()
    )
    telemetry = coordinator.telemetry

    passed = bool(
        first_quality.accepted
        and second_quality.accepted
        and stale_generation_rejected
        and telemetry.worker_generation == 1
        and telemetry.optimizer_epoch == 2
        and telemetry.epoch_transitions == 1
        and telemetry.generation_regressions_rejected == 1
        and final.raw_pose_count == 12
        and final.optimized_keyframe_count == 2
        and raw_preserved
        and all(item.gauge.unit_to_metre is None for item in final.poses)
    )

    print(f"Phase 8.5D live trajectory hardening smoke: {'PASS' if passed else 'FAIL'}")
    print(
        f"worker_generation={telemetry.worker_generation} optimizer_epoch={telemetry.optimizer_epoch} "
        f"epoch_transitions={telemetry.epoch_transitions}"
    )
    print(
        f"raw_poses={final.raw_pose_count} optimized_keyframes={final.optimized_keyframe_count} "
        f"stale_generation_rejected={stale_generation_rejected}"
    )
    print(f"raw_tracking_preserved={raw_preserved}")
    print("pose_convention=T_reconstruction_from_camera")
    print("gauge=arbitrary unit_to_metre=None")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

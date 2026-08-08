#!/usr/bin/env python3
"""Dependency-light Phase 8.5C optimized-trajectory propagation smoke."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction import (
    LiveMapSnapshot,
    LiveOptimizedTrajectory,
    LiveSlamObservation,
    LiveSlamPose,
    LiveTrackingQuality,
    LiveTrackingState,
    LiveTrajectoryPoseLayer,
    NativeGlobalConsistencyStatus,
    OptimizedKeyframePose,
)

SESSION_ID = "00000000-0000-4000-8000-000000000053"


def matrix(x: float) -> np.ndarray:
    value = np.eye(4, dtype=np.float64)
    value[0, 3] = x
    return value


def observation(index: int, x: float, keyframe_id: str = "") -> LiveSlamObservation:
    frame_id = f"live-frame-{index}"
    return LiveSlamObservation(
        session_id=SESSION_ID,
        frame_id=frame_id,
        sequence_index=index,
        tracking_state=LiveTrackingState.TRACKING,
        quality=LiveTrackingQuality(
            tracked_features=100,
            inlier_features=85,
            processing_latency_ns=1_000_000,
        ),
        pose=LiveSlamPose(
            pose_id=f"tracking-pose-{index}",
            camera_id="camera-0",
            frame_id=frame_id,
            timestamp_ns=1_000_000_000 + index * 10_000_000,
            camera_frame_id="camera-optical",
            T_reconstruction_from_camera=matrix(x),
        ),
        is_keyframe=bool(keyframe_id),
        keyframe_id=keyframe_id,
        map_snapshot=LiveMapSnapshot(revision=2, keyframe_count=2, landmark_count=120),
    )


def main() -> int:
    trajectory = LiveOptimizedTrajectory(session_id=SESSION_ID)
    for item in (
        observation(0, 0.0, "stella-keyframe-0"),
        observation(1, 1.1),
        observation(2, 2.2),
        observation(3, 3.3, "stella-keyframe-3"),
        observation(4, 4.4),
    ):
        trajectory.ingest_observation(item)

    status = NativeGlobalConsistencyStatus(
        session_id=SESSION_ID,
        loop_detector_enabled=True,
        global_optimization_enabled=True,
        loop_ba_running=False,
        map_revision=9,
        optimized_pose_revision=4,
        keyframe_count=2,
        landmark_count=120,
        accepted_loop_closure_requests=1,
        relocalization_count=0,
        last_relocalization_frame_id="",
        optimized_keyframe_poses=(
            OptimizedKeyframePose(
                keyframe_id="stella-keyframe-0",
                frame_id="stella-keyframe-0",
                timestamp_ns=1_000_000_000,
                source_pose_id="native-optimized-0",
                optimization_revision=4,
                T_reconstruction_from_camera=matrix(0.0),
            ),
            OptimizedKeyframePose(
                keyframe_id="stella-keyframe-3",
                frame_id="stella-keyframe-3",
                timestamp_ns=1_030_000_000,
                source_pose_id="native-optimized-3",
                optimization_revision=4,
                T_reconstruction_from_camera=matrix(3.0),
            ),
        ),
    )
    raw_before = trajectory.raw_pose_for_frame("live-frame-4")
    update = trajectory.apply_native_status(status)
    exact = trajectory.best_pose_for_frame("live-frame-3")
    propagated = trajectory.best_pose_for_frame("live-frame-4")
    raw_after = trajectory.raw_pose_for_frame("live-frame-4")
    snapshot = trajectory.snapshot()

    passed = (
        raw_before is not None
        and raw_after is not None
        and exact is not None
        and propagated is not None
        and update.exact_optimized_keyframes == 2
        and exact.layer is LiveTrajectoryPoseLayer.OPTIMIZED_KEYFRAME
        and propagated.layer is LiveTrajectoryPoseLayer.OPTIMIZED_PROPAGATED
        and np.allclose(exact.T_reconstruction_from_camera, matrix(3.0))
        and np.allclose(propagated.T_reconstruction_from_camera, matrix(4.1))
        and np.array_equal(
            raw_before.T_reconstruction_from_camera,
            raw_after.T_reconstruction_from_camera,
        )
        and snapshot.optimization_revision == 4
        and snapshot.raw_pose_count == 5
        and snapshot.optimized_keyframe_count == 2
        and propagated.gauge.unit_to_metre is None
    )

    print(f"Phase 8.5C live optimized-trajectory propagation smoke: {'PASS' if passed else 'FAIL'}")
    print(
        f"raw_poses={snapshot.raw_pose_count} optimized_keyframes={snapshot.optimized_keyframe_count} "
        f"optimization_revision={snapshot.optimization_revision}"
    )
    print(
        f"exact_layer={exact.layer.value if exact else 'missing'} "
        f"propagated_layer={propagated.layer.value if propagated else 'missing'}"
    )
    print("raw_tracking_preserved=True")
    print("propagation_policy=latest_preceding_optimized_keyframe")
    print("pose_convention=T_reconstruction_from_camera")
    print("gauge=arbitrary unit_to_metre=None")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

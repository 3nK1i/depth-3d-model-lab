from __future__ import annotations

import numpy as np

from core.reconstruction.global_consistency import OptimizedKeyframePose
from core.reconstruction.live_contracts import (
    LiveMapSnapshot,
    LiveSlamObservation,
    LiveSlamPose,
    LiveTrackingQuality,
    LiveTrackingState,
)
from core.reconstruction.live_global_consistency import NativeGlobalConsistencyStatus
from core.reconstruction.live_trajectory import LiveOptimizedTrajectory, LiveTrajectoryPoseLayer

SESSION_ID = "00000000-0000-4000-8000-000000000052"


def _matrix(x: float) -> np.ndarray:
    value = np.eye(4, dtype=np.float64)
    value[0, 3] = x
    return value


def _observation(index: int, *, is_keyframe: bool) -> LiveSlamObservation:
    frame_id = f"live-frame-{index}"
    pose = LiveSlamPose(
        pose_id=f"pose-{index}",
        camera_id="camera-0",
        frame_id=frame_id,
        timestamp_ns=1_000_000_000 + index * 1_000_000,
        camera_frame_id="camera-optical",
        T_reconstruction_from_camera=_matrix(index * 0.01),
    )
    return LiveSlamObservation(
        session_id=SESSION_ID,
        frame_id=frame_id,
        sequence_index=index,
        tracking_state=LiveTrackingState.TRACKING,
        quality=LiveTrackingQuality(
            tracked_features=50,
            inlier_features=40,
            processing_latency_ns=100_000,
        ),
        pose=pose,
        is_keyframe=is_keyframe,
        keyframe_id=f"stella-keyframe-{index}" if is_keyframe else "",
        map_snapshot=LiveMapSnapshot(revision=index, keyframe_count=0, landmark_count=0),
    )


def _status(revision: int, keyframes: tuple[int, ...]) -> NativeGlobalConsistencyStatus:
    poses = tuple(
        OptimizedKeyframePose(
            keyframe_id=f"stella-keyframe-{index}",
            frame_id=f"stella-keyframe-{index}",
            timestamp_ns=1_000_000_000 + index * 1_000_000,
            source_pose_id=f"native-{index}",
            optimization_revision=revision,
            T_reconstruction_from_camera=_matrix(index * 0.01 - revision * 0.0001),
        )
        for index in keyframes
    )
    return NativeGlobalConsistencyStatus(
        session_id=SESSION_ID,
        loop_detector_enabled=True,
        global_optimization_enabled=True,
        loop_ba_running=False,
        map_revision=revision,
        optimized_pose_revision=revision,
        keyframe_count=len(poses),
        landmark_count=500,
        accepted_loop_closure_requests=revision,
        relocalization_count=0,
        last_relocalization_frame_id="",
        optimized_keyframe_poses=poses,
    )


def test_many_pose_revisions_remain_bounded_deterministic_and_raw_safe() -> None:
    trajectory = LiveOptimizedTrajectory(session_id=SESSION_ID)
    keyframes = tuple(range(0, 1000, 20))
    for index in range(1000):
        trajectory.ingest_observation(_observation(index, is_keyframe=index in keyframes))

    raw_tail = trajectory.raw_pose_for_frame("live-frame-999")
    assert raw_tail is not None
    raw_tail_matrix = raw_tail.T_reconstruction_from_camera.copy()

    for revision in range(1, 101):
        update = trajectory.apply_native_status(_status(revision, keyframes))
        assert update.optimization_revision == revision
        assert update.exact_optimized_keyframes == len(keyframes)
        snapshot = trajectory.snapshot()
        assert snapshot.optimization_revision == revision
        assert snapshot.raw_pose_count == 1000
        assert snapshot.optimized_keyframe_count == len(keyframes)
        assert len(snapshot.poses) == 1000
        assert snapshot.poses[-1].layer is LiveTrajectoryPoseLayer.OPTIMIZED_PROPAGATED

    raw_tail_after = trajectory.raw_pose_for_frame("live-frame-999")
    assert raw_tail_after is not None
    assert np.array_equal(raw_tail_after.T_reconstruction_from_camera, raw_tail_matrix)

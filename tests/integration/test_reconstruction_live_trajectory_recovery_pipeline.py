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
from core.reconstruction.live_trajectory import LiveTrajectoryPoseLayer
from core.reconstruction.live_trajectory_hardening import LiveTrajectoryCoordinator

SESSION_ID = "00000000-0000-4000-8000-000000000062"


def _matrix(x: float) -> np.ndarray:
    value = np.eye(4, dtype=np.float64)
    value[0, 3] = x
    return value


def _observation(index: int, *, keyframe_id: str = "") -> LiveSlamObservation:
    frame_id = f"live-frame-{index}"
    pose = LiveSlamPose(
        pose_id=f"raw-pose-{index}",
        camera_id="camera-0",
        frame_id=frame_id,
        timestamp_ns=1_000_000_000 + index * 5_000_000,
        camera_frame_id="camera-optical",
        T_reconstruction_from_camera=_matrix(index * 0.05),
    )
    return LiveSlamObservation(
        session_id=SESSION_ID,
        frame_id=frame_id,
        sequence_index=index,
        tracking_state=LiveTrackingState.TRACKING,
        quality=LiveTrackingQuality(
            tracked_features=80,
            inlier_features=60,
            processing_latency_ns=500_000,
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
            timestamp_ns=1_000_000_000 + index * 5_000_000,
            source_pose_id=f"native-{revision}-{keyframe_id}",
            optimization_revision=revision,
            T_reconstruction_from_camera=_matrix(index * 0.05 - 0.02),
        )
        for keyframe_id, index in rows
    )
    return NativeGlobalConsistencyStatus(
        session_id=SESSION_ID,
        loop_detector_enabled=True,
        global_optimization_enabled=True,
        loop_ba_running=False,
        map_revision=revision * 10,
        optimized_pose_revision=revision,
        keyframe_count=len(poses),
        landmark_count=300,
        accepted_loop_closure_requests=revision,
        relocalization_count=0,
        last_relocalization_frame_id="",
        optimized_keyframe_poses=poses,
    )


def test_worker_generation_restart_keeps_raw_history_and_prevents_stale_corrections() -> None:
    coordinator = LiveTrajectoryCoordinator(session_id=SESSION_ID)

    for index in range(8):
        coordinator.ingest_observation(
            _observation(
                index,
                keyframe_id=f"stella-keyframe-{index // 3}" if index in {0, 3, 6} else "",
            ),
            worker_generation=0,
        )
    first_status = _status(
        4,
        (
            ("stella-keyframe-0", 0),
            ("stella-keyframe-1", 3),
            ("stella-keyframe-2", 6),
        ),
    )
    coordinator.apply_native_status(first_status, worker_generation=0)
    old_best = coordinator.trajectory.best_pose_for_frame("live-frame-7")
    old_raw = coordinator.trajectory.raw_pose_for_frame("live-frame-7")
    assert old_best is not None and old_best.layer is LiveTrajectoryPoseLayer.OPTIMIZED_PROPAGATED
    assert old_raw is not None
    raw_matrix = old_raw.T_reconstruction_from_camera.copy()

    # Simulate the native worker being replaced. Its keyframe IDs and optimizer
    # revision are allowed to restart from zero/one in the new generation.
    for index in range(8, 16):
        local = index - 8
        coordinator.ingest_observation(
            _observation(
                index,
                keyframe_id=f"stella-keyframe-{local // 3}" if local in {0, 3, 6} else "",
            ),
            worker_generation=1,
        )

    after_restart_old_best = coordinator.trajectory.best_pose_for_frame("live-frame-7")
    assert after_restart_old_best is not None
    assert after_restart_old_best.layer is LiveTrajectoryPoseLayer.TRACKING

    second_status = _status(
        1,
        (
            ("stella-keyframe-0", 8),
            ("stella-keyframe-1", 11),
            ("stella-keyframe-2", 14),
        ),
    )
    second_update = coordinator.apply_native_status(second_status, worker_generation=1)
    final = coordinator.snapshot()

    assert second_update.optimizer_epoch == 2
    assert second_update.optimization_revision == 1
    assert final.optimizer_epoch == 2
    assert final.raw_pose_count == 16
    assert final.optimized_keyframe_count == 3
    assert coordinator.telemetry.epoch_transitions == 1
    assert coordinator.raw_tracking_integrity_ok()
    assert np.array_equal(
        coordinator.trajectory.raw_pose_for_frame("live-frame-7").T_reconstruction_from_camera,
        raw_matrix,
    )
    assert coordinator.trajectory.best_pose_for_frame("live-frame-15").layer is LiveTrajectoryPoseLayer.OPTIMIZED_PROPAGATED

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
from core.reconstruction.live_trajectory_hardening import LiveTrajectoryCoordinator

SESSION_ID = "00000000-0000-4000-8000-000000000063"


def _matrix(x: float) -> np.ndarray:
    value = np.eye(4, dtype=np.float64)
    value[0, 3] = x
    return value


def _observation(index: int, *, keyframe_id: str = "") -> LiveSlamObservation:
    frame_id = f"live-frame-{index}"
    pose = LiveSlamPose(
        pose_id=f"pose-{index}",
        camera_id="camera-0",
        frame_id=frame_id,
        timestamp_ns=2_000_000_000 + index * 1_000_000,
        camera_frame_id="camera-optical",
        T_reconstruction_from_camera=_matrix(index * 0.002),
    )
    return LiveSlamObservation(
        session_id=SESSION_ID,
        frame_id=frame_id,
        sequence_index=index,
        tracking_state=LiveTrackingState.TRACKING,
        quality=LiveTrackingQuality(
            tracked_features=60,
            inlier_features=45,
            processing_latency_ns=100_000,
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
            timestamp_ns=2_000_000_000 + index * 1_000_000,
            source_pose_id=f"native-{revision}-{keyframe_id}",
            optimization_revision=revision,
            T_reconstruction_from_camera=_matrix(index * 0.002 - revision * 0.00001),
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
        landmark_count=1000,
        accepted_loop_closure_requests=revision,
        relocalization_count=revision,
        last_relocalization_frame_id="",
        optimized_keyframe_poses=poses,
    )


def test_repeated_worker_epochs_and_revisions_preserve_all_raw_tracking() -> None:
    coordinator = LiveTrajectoryCoordinator(session_id=SESSION_ID)
    total_frames = 0
    generations = 6
    frames_per_generation = 180
    revisions_per_generation = 12

    raw_samples: dict[str, np.ndarray] = {}
    for generation in range(generations):
        start = total_frames
        keyframe_rows: list[tuple[str, int]] = []
        for local in range(frames_per_generation):
            index = start + local
            keyframe_id = ""
            if local % 30 == 0:
                keyframe_id = f"stella-keyframe-{local // 30}"
                keyframe_rows.append((keyframe_id, index))
            coordinator.ingest_observation(
                _observation(index, keyframe_id=keyframe_id),
                worker_generation=generation,
            )
            if local in {0, frames_per_generation - 1}:
                raw = coordinator.trajectory.raw_pose_for_frame(f"live-frame-{index}")
                assert raw is not None
                raw_samples[raw.frame_id] = raw.T_reconstruction_from_camera.copy()
        total_frames += frames_per_generation

        rows = tuple(keyframe_rows)
        for revision in range(1, revisions_per_generation + 1):
            update = coordinator.apply_native_status(
                _status(revision, rows),
                worker_generation=generation,
            )
            assert update.optimizer_epoch == generation + 1
            assert update.optimization_revision == revision

        assert coordinator.raw_tracking_integrity_ok()
        assert coordinator.snapshot().raw_pose_count == total_frames

    telemetry = coordinator.telemetry
    assert telemetry.worker_generation == generations - 1
    assert telemetry.optimizer_epoch == generations
    assert telemetry.epoch_transitions == generations - 1
    assert telemetry.statuses_applied == generations * revisions_per_generation
    assert telemetry.raw_pose_count == generations * frames_per_generation
    assert telemetry.raw_integrity_failures == 0

    for frame_id, before in raw_samples.items():
        after = coordinator.trajectory.raw_pose_for_frame(frame_id)
        assert after is not None
        assert np.array_equal(after.T_reconstruction_from_camera, before)

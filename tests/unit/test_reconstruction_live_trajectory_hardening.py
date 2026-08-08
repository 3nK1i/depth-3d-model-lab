from __future__ import annotations

import numpy as np
import pytest

from core.reconstruction.errors import ReconstructionValidationError
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
from core.reconstruction.live_trajectory_hardening import (
    LiveTrajectoryCoordinator,
    LiveTrajectoryHardeningPolicy,
)

SESSION_ID = "00000000-0000-4000-8000-000000000061"
CAMERA_ID = "camera-0"


def _matrix(x: float) -> np.ndarray:
    value = np.eye(4, dtype=np.float64)
    value[0, 3] = x
    return value


def _observation(
    index: int,
    *,
    x: float | None = None,
    keyframe_id: str = "",
    timestamp_ns: int | None = None,
) -> LiveSlamObservation:
    frame_id = f"live-frame-{index}"
    pose = LiveSlamPose(
        pose_id=f"raw-pose-{index}",
        camera_id=CAMERA_ID,
        frame_id=frame_id,
        timestamp_ns=timestamp_ns if timestamp_ns is not None else 1_000_000_000 + index * 10_000_000,
        camera_frame_id="camera-optical",
        T_reconstruction_from_camera=_matrix(index * 0.1 if x is None else x),
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


def _status(
    revision: int,
    *,
    keyframe_rows: tuple[tuple[str, int, float], ...],
    map_revision: int | None = None,
) -> NativeGlobalConsistencyStatus:
    poses = tuple(
        OptimizedKeyframePose(
            keyframe_id=keyframe_id,
            frame_id=keyframe_id,
            timestamp_ns=1_000_000_000 + frame_index * 10_000_000,
            source_pose_id=f"native-{revision}-{keyframe_id}",
            optimization_revision=revision,
            T_reconstruction_from_camera=_matrix(x),
        )
        for keyframe_id, frame_index, x in keyframe_rows
    )
    return NativeGlobalConsistencyStatus(
        session_id=SESSION_ID,
        loop_detector_enabled=True,
        global_optimization_enabled=True,
        loop_ba_running=False,
        map_revision=revision if map_revision is None else map_revision,
        optimized_pose_revision=revision,
        keyframe_count=len(poses),
        landmark_count=200,
        accepted_loop_closure_requests=revision,
        relocalization_count=0,
        last_relocalization_frame_id="",
        optimized_keyframe_poses=poses,
    )


def test_optimizer_epoch_reset_preserves_raw_and_allows_native_keyframe_id_reuse() -> None:
    trajectory = LiveOptimizedTrajectory(session_id=SESSION_ID)
    trajectory.ingest_observation(_observation(0, keyframe_id="stella-keyframe-0"))
    trajectory.ingest_observation(_observation(1))
    trajectory.apply_native_status(
        _status(3, keyframe_rows=(("stella-keyframe-0", 0, -0.1),))
    )

    old_raw = trajectory.raw_pose_for_frame("live-frame-0")
    old_best = trajectory.best_pose_for_frame("live-frame-1")
    assert old_raw is not None and old_best is not None
    assert old_best.layer is LiveTrajectoryPoseLayer.OPTIMIZED_PROPAGATED
    assert old_best.optimizer_epoch == 1

    transition = trajectory.begin_optimizer_epoch(reason="worker_generation:0->1")
    after_reset = trajectory.snapshot()

    assert transition.previous_optimizer_epoch == 1
    assert transition.optimizer_epoch == 2
    assert transition.cleared_optimized_keyframes == 1
    assert after_reset.raw_pose_count == 2
    assert after_reset.optimized_keyframe_count == 0
    assert after_reset.optimization_revision == 0
    assert trajectory.best_pose_for_frame("live-frame-1").layer is LiveTrajectoryPoseLayer.TRACKING
    assert np.array_equal(
        trajectory.raw_pose_for_frame("live-frame-0").T_reconstruction_from_camera,
        old_raw.T_reconstruction_from_camera,
    )

    trajectory.ingest_observation(
        _observation(2, keyframe_id="stella-keyframe-0")
    )
    update = trajectory.apply_native_status(
        _status(1, keyframe_rows=(("stella-keyframe-0", 2, 0.15),))
    )
    best = trajectory.best_pose_for_frame("live-frame-2")

    assert update.optimizer_epoch == 2
    assert update.optimization_revision == 1
    assert best is not None
    assert best.layer is LiveTrajectoryPoseLayer.OPTIMIZED_KEYFRAME
    assert best.optimizer_epoch == 2


def test_coordinator_generation_change_resets_only_optimizer_state() -> None:
    coordinator = LiveTrajectoryCoordinator(session_id=SESSION_ID)
    coordinator.ingest_observation(
        _observation(0, keyframe_id="stella-keyframe-0"),
        worker_generation=0,
    )
    coordinator.apply_native_status(
        _status(2, keyframe_rows=(("stella-keyframe-0", 0, -0.05),)),
        worker_generation=0,
    )
    before = coordinator.snapshot()
    assert before.optimizer_epoch == 1
    assert before.optimized_keyframe_count == 1

    coordinator.ingest_observation(_observation(1), worker_generation=1)
    after = coordinator.snapshot()
    telemetry = coordinator.telemetry

    assert after.optimizer_epoch == 2
    assert after.raw_pose_count == 2
    assert after.optimized_keyframe_count == 0
    assert telemetry.worker_generation == 1
    assert telemetry.epoch_transitions == 1
    assert coordinator.raw_tracking_integrity_ok()


def test_worker_generation_lifecycle_transition_does_not_wait_for_first_pose() -> None:
    coordinator = LiveTrajectoryCoordinator(session_id=SESSION_ID)
    coordinator.ingest_observation(
        _observation(0, keyframe_id="stella-keyframe-0"),
        worker_generation=0,
    )
    coordinator.apply_native_status(
        _status(2, keyframe_rows=(("stella-keyframe-0", 0, -0.05),)),
        worker_generation=0,
    )
    before = coordinator.trajectory.best_pose_for_frame("live-frame-0")
    assert before is not None
    assert before.layer is LiveTrajectoryPoseLayer.OPTIMIZED_KEYFRAME

    transition = coordinator.begin_worker_generation(1)
    after = coordinator.trajectory.best_pose_for_frame("live-frame-0")

    assert transition is not None
    assert transition.previous_optimizer_epoch == 1
    assert transition.optimizer_epoch == 2
    assert after is not None
    assert after.layer is LiveTrajectoryPoseLayer.TRACKING
    assert coordinator.telemetry.worker_generation == 1
    assert coordinator.telemetry.epoch_transitions == 1

    with pytest.raises(ReconstructionValidationError, match="worker generation"):
        coordinator.apply_native_status(
            _status(3, keyframe_rows=(("stella-keyframe-0", 0, -0.08),)),
            worker_generation=0,
        )
    assert coordinator.telemetry.generation_regressions_rejected == 1

def test_coordinator_rejects_stale_worker_generation() -> None:
    coordinator = LiveTrajectoryCoordinator(session_id=SESSION_ID)
    coordinator.ingest_observation(_observation(0), worker_generation=2)
    with pytest.raises(ReconstructionValidationError, match="worker generation"):
        coordinator.ingest_observation(_observation(1), worker_generation=1)
    assert coordinator.telemetry.generation_regressions_rejected == 1


def test_quality_assessment_requires_match_ratio_and_exact_pose_floor() -> None:
    policy = LiveTrajectoryHardeningPolicy(
        min_keyframe_match_ratio=0.75,
        min_exact_optimized_keyframes=3,
    )
    coordinator = LiveTrajectoryCoordinator(session_id=SESSION_ID, policy=policy)
    for index in range(5):
        coordinator.ingest_observation(
            _observation(index, keyframe_id=f"stella-keyframe-{index}"),
        )
    status = _status(
        1,
        keyframe_rows=tuple(
            (f"stella-keyframe-{index}", index, index * 0.1 - 0.01)
            for index in range(4)
        ),
    )
    coordinator.apply_native_status(status)
    accepted = coordinator.assess_quality(status)
    assert accepted.accepted
    assert accepted.keyframe_match_ratio == 1.0
    assert accepted.raw_tracking_integrity

    partial_poses = list(status.optimized_keyframe_poses[:2])
    for extra in range(3):
        partial_poses.append(
            OptimizedKeyframePose(
                keyframe_id=f"unmatched-{extra}",
                frame_id=f"unmatched-{extra}",
                timestamp_ns=9_000_000_000 + extra,
                source_pose_id=f"native-unmatched-{extra}",
                optimization_revision=1,
                T_reconstruction_from_camera=_matrix(10.0 + extra),
            )
        )
    partial = NativeGlobalConsistencyStatus(
        session_id=SESSION_ID,
        loop_detector_enabled=True,
        global_optimization_enabled=True,
        loop_ba_running=False,
        map_revision=1,
        optimized_pose_revision=1,
        keyframe_count=5,
        landmark_count=200,
        accepted_loop_closure_requests=1,
        relocalization_count=0,
        last_relocalization_frame_id="",
        optimized_keyframe_poses=tuple(partial_poses),
    )
    other = LiveTrajectoryCoordinator(session_id=SESSION_ID, policy=policy)
    for index in range(5):
        other.ingest_observation(
            _observation(index, keyframe_id=f"stella-keyframe-{index}"),
        )
    other.apply_native_status(partial)
    rejected = other.assess_quality(partial)
    assert not rejected.accepted
    assert "insufficient_exact_optimized_keyframes" in rejected.reasons
    assert "insufficient_keyframe_match_ratio" in rejected.reasons


def test_stale_native_revision_is_counted_without_mutating_raw_trajectory() -> None:
    coordinator = LiveTrajectoryCoordinator(session_id=SESSION_ID)
    coordinator.ingest_observation(_observation(0, keyframe_id="stella-keyframe-0"))
    status2 = _status(2, keyframe_rows=(("stella-keyframe-0", 0, -0.05),))
    coordinator.apply_native_status(status2)
    raw_before = coordinator.trajectory.raw_pose_for_frame("live-frame-0")
    with pytest.raises(ReconstructionValidationError, match="Stale"):
        coordinator.apply_native_status(
            _status(1, keyframe_rows=(("stella-keyframe-0", 0, -0.02),))
        )
    raw_after = coordinator.trajectory.raw_pose_for_frame("live-frame-0")
    assert coordinator.telemetry.status_rejections == 1
    assert raw_before is not None and raw_after is not None
    assert np.array_equal(
        raw_before.T_reconstruction_from_camera,
        raw_after.T_reconstruction_from_camera,
    )


def test_tracking_pose_has_no_optimizer_epoch_but_optimized_pose_does() -> None:
    coordinator = LiveTrajectoryCoordinator(session_id=SESSION_ID)
    coordinator.ingest_observation(_observation(0, keyframe_id="stella-keyframe-0"))
    raw = coordinator.trajectory.raw_pose_for_frame("live-frame-0")
    assert raw is not None
    assert raw.optimizer_epoch is None

    coordinator.apply_native_status(
        _status(1, keyframe_rows=(("stella-keyframe-0", 0, -0.01),))
    )
    optimized = coordinator.trajectory.best_pose_for_frame("live-frame-0")
    assert optimized is not None
    assert optimized.optimizer_epoch == 1
    assert optimized.layer is LiveTrajectoryPoseLayer.OPTIMIZED_KEYFRAME

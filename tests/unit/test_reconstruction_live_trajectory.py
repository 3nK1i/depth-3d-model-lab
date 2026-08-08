from __future__ import annotations

from dataclasses import replace

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
from core.reconstruction.live_trajectory import (
    LiveOptimizedTrajectory,
    LiveTrajectoryPoseLayer,
)

SESSION_ID = "00000000-0000-4000-8000-000000000051"
CAMERA_ID = "camera-0"


def _matrix(x: float) -> np.ndarray:
    value = np.eye(4, dtype=np.float64)
    value[0, 3] = x
    return value


def _observation(index: int, *, x: float, keyframe_id: str = "") -> LiveSlamObservation:
    frame_id = f"live-frame-{index}"
    pose = LiveSlamPose(
        pose_id=f"raw-pose-{index}",
        camera_id=CAMERA_ID,
        frame_id=frame_id,
        timestamp_ns=1_000_000_000 + index * 10_000_000,
        camera_frame_id="camera-optical",
        T_reconstruction_from_camera=_matrix(x),
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
        map_snapshot=LiveMapSnapshot(revision=1, keyframe_count=2, landmark_count=100),
    )


def _status(revision: int, *, kf0_x: float, kf3_x: float) -> NativeGlobalConsistencyStatus:
    poses = (
        OptimizedKeyframePose(
            keyframe_id="stella-keyframe-0",
            frame_id="stella-keyframe-0",
            timestamp_ns=1_000_000_000,
            source_pose_id="native-pose-0",
            optimization_revision=revision,
            T_reconstruction_from_camera=_matrix(kf0_x),
        ),
        OptimizedKeyframePose(
            keyframe_id="stella-keyframe-3",
            frame_id="stella-keyframe-3",
            timestamp_ns=1_030_000_000,
            source_pose_id="native-pose-3",
            optimization_revision=revision,
            T_reconstruction_from_camera=_matrix(kf3_x),
        ),
    )
    return NativeGlobalConsistencyStatus(
        session_id=SESSION_ID,
        loop_detector_enabled=True,
        global_optimization_enabled=True,
        loop_ba_running=False,
        map_revision=4 + revision,
        optimized_pose_revision=revision,
        keyframe_count=2,
        landmark_count=100,
        accepted_loop_closure_requests=1,
        relocalization_count=0,
        last_relocalization_frame_id="",
        optimized_keyframe_poses=poses,
    )


def _trajectory() -> LiveOptimizedTrajectory:
    trajectory = LiveOptimizedTrajectory(session_id=SESSION_ID)
    trajectory.ingest_observation(_observation(0, x=0.0, keyframe_id="stella-keyframe-0"))
    trajectory.ingest_observation(_observation(1, x=1.1))
    trajectory.ingest_observation(_observation(2, x=2.2))
    trajectory.ingest_observation(_observation(3, x=3.3, keyframe_id="stella-keyframe-3"))
    trajectory.ingest_observation(_observation(4, x=4.4))
    return trajectory


def test_keeps_raw_tracking_immutable_and_exposes_exact_optimized_keyframe() -> None:
    trajectory = _trajectory()
    raw_before = trajectory.raw_pose_for_frame("live-frame-3")
    update = trajectory.apply_native_status(_status(1, kf0_x=0.0, kf3_x=3.0))
    raw_after = trajectory.raw_pose_for_frame("live-frame-3")
    best = trajectory.best_pose_for_frame("live-frame-3")

    assert update.optimization_revision == 1
    assert update.exact_optimized_keyframes == 2
    assert raw_before is not None and raw_after is not None and best is not None
    assert raw_before.layer is LiveTrajectoryPoseLayer.TRACKING
    assert raw_after.layer is LiveTrajectoryPoseLayer.TRACKING
    assert np.allclose(raw_before.T_reconstruction_from_camera, _matrix(3.3))
    assert np.allclose(raw_after.T_reconstruction_from_camera, _matrix(3.3))
    assert best.layer is LiveTrajectoryPoseLayer.OPTIMIZED_KEYFRAME
    assert np.allclose(best.T_reconstruction_from_camera, _matrix(3.0))
    assert best.raw_pose_id == "raw-pose-3"
    assert best.native_optimized_source_pose_id == "native-pose-3"


def test_non_keyframe_uses_latest_causal_keyframe_correction() -> None:
    trajectory = _trajectory()
    trajectory.apply_native_status(_status(1, kf0_x=0.0, kf3_x=3.0))

    before_second_keyframe = trajectory.best_pose_for_frame("live-frame-2")
    after_second_keyframe = trajectory.best_pose_for_frame("live-frame-4")
    raw_only = trajectory.best_pose_for_frame("live-frame-4", allow_propagated=False)

    assert before_second_keyframe is not None
    assert before_second_keyframe.layer is LiveTrajectoryPoseLayer.OPTIMIZED_PROPAGATED
    assert np.allclose(before_second_keyframe.T_reconstruction_from_camera, _matrix(2.2))
    assert before_second_keyframe.source_keyframe_id == "stella-keyframe-0"

    assert after_second_keyframe is not None
    assert after_second_keyframe.layer is LiveTrajectoryPoseLayer.OPTIMIZED_PROPAGATED
    assert np.allclose(after_second_keyframe.T_reconstruction_from_camera, _matrix(4.1))
    assert after_second_keyframe.source_keyframe_id == "stella-keyframe-3"

    assert raw_only is not None
    assert raw_only.layer is LiveTrajectoryPoseLayer.TRACKING
    assert np.allclose(raw_only.T_reconstruction_from_camera, _matrix(4.4))


def test_revision_updates_are_deterministic_and_stale_revisions_are_rejected() -> None:
    trajectory = _trajectory()
    first = trajectory.apply_native_status(_status(2, kf0_x=0.0, kf3_x=3.1))
    repeated = trajectory.apply_native_status(_status(2, kf0_x=0.0, kf3_x=3.1))
    second = trajectory.apply_native_status(_status(3, kf0_x=0.0, kf3_x=3.0))

    assert [item.keyframe_id for item in first.corrections] == [
        "stella-keyframe-0",
        "stella-keyframe-3",
    ]
    assert repeated.corrections == ()
    assert not repeated.changed
    assert second.previous_optimization_revision == 2
    assert second.optimization_revision == 3
    assert [item.keyframe_id for item in second.corrections] == ["stella-keyframe-3"]

    with pytest.raises(ReconstructionValidationError, match="Stale"):
        trajectory.apply_native_status(_status(2, kf0_x=0.0, kf3_x=3.1))



def test_same_revision_may_add_new_keyframes_without_mutating_existing_poses() -> None:
    trajectory = _trajectory()
    full = _status(1, kf0_x=0.0, kf3_x=3.0)
    partial = replace(
        full,
        keyframe_count=1,
        optimized_keyframe_poses=(full.optimized_keyframe_poses[0],),
    )
    first = trajectory.apply_native_status(partial)
    extended = trajectory.apply_native_status(full)

    assert first.exact_optimized_keyframes == 1
    assert extended.optimization_revision == 1
    assert extended.previous_optimization_revision == 1
    assert extended.exact_optimized_keyframes == 2
    assert [item.keyframe_id for item in extended.corrections] == ["stella-keyframe-3"]
    assert extended.changed

def test_same_revision_with_different_native_payload_is_rejected() -> None:
    trajectory = _trajectory()
    trajectory.apply_native_status(_status(1, kf0_x=0.0, kf3_x=3.0))
    with pytest.raises(ReconstructionValidationError, match="without advancing"):
        trajectory.apply_native_status(_status(1, kf0_x=0.0, kf3_x=2.9))


def test_unmatched_native_keyframe_is_reported_without_overwriting_raw_path() -> None:
    trajectory = LiveOptimizedTrajectory(session_id=SESSION_ID)
    trajectory.ingest_observation(_observation(0, x=0.0, keyframe_id="stella-keyframe-0"))
    status = _status(1, kf0_x=0.0, kf3_x=3.0)
    update = trajectory.apply_native_status(status)
    snapshot = trajectory.snapshot()

    assert update.unmatched_optimized_keyframe_ids == ("stella-keyframe-3",)
    assert update.exact_optimized_keyframes == 1
    assert snapshot.raw_pose_count == 1
    assert snapshot.optimized_keyframe_count == 1
    assert snapshot.poses[0].layer is LiveTrajectoryPoseLayer.OPTIMIZED_KEYFRAME


def test_session_and_keyframe_timestamp_mismatch_are_rejected() -> None:
    trajectory = _trajectory()
    wrong_session = replace(_status(1, kf0_x=0.0, kf3_x=3.0), session_id="00000000-0000-4000-8000-000000000099")
    with pytest.raises(ReconstructionValidationError, match="session_id"):
        trajectory.apply_native_status(wrong_session)

    status = _status(1, kf0_x=0.0, kf3_x=3.0)
    bad_pose = replace(status.optimized_keyframe_poses[1], timestamp_ns=1_045_000_000)
    bad_status = replace(status, optimized_keyframe_poses=(status.optimized_keyframe_poses[0], bad_pose))
    with pytest.raises(ReconstructionValidationError, match="timestamp"):
        trajectory.apply_native_status(bad_status)

def test_native_timestamp_repairs_delayed_async_keyframe_id_association() -> None:
    trajectory = LiveOptimizedTrajectory(session_id=SESSION_ID)
    raw0 = _observation(0, x=0.0)
    delayed_hint = _observation(1, x=1.1, keyframe_id="stella-keyframe-0")
    trajectory.ingest_observation(raw0)
    trajectory.ingest_observation(delayed_hint)
    trajectory.ingest_observation(_observation(2, x=2.2))

    full_status = _status(1, kf0_x=-0.1, kf3_x=3.0)
    status = replace(
        full_status,
        keyframe_count=1,
        optimized_keyframe_poses=(full_status.optimized_keyframe_poses[0],),
    )
    raw0_before = trajectory.raw_pose_for_frame(raw0.frame_id)
    raw1_before = trajectory.raw_pose_for_frame(delayed_hint.frame_id)

    update = trajectory.apply_native_status(status)

    raw0_after = trajectory.raw_pose_for_frame(raw0.frame_id)
    raw1_after = trajectory.raw_pose_for_frame(delayed_hint.frame_id)
    best0 = trajectory.best_pose_for_frame(raw0.frame_id)
    best1_no_propagation = trajectory.best_pose_for_frame(
        delayed_hint.frame_id, allow_propagated=False
    )

    assert update.exact_optimized_keyframes == 1
    assert update.unmatched_optimized_keyframe_ids == ()
    assert raw0_before is not None and raw0_after is not None
    assert raw1_before is not None and raw1_after is not None
    assert np.array_equal(
        raw0_before.T_reconstruction_from_camera, raw0_after.T_reconstruction_from_camera
    )
    assert np.array_equal(
        raw1_before.T_reconstruction_from_camera, raw1_after.T_reconstruction_from_camera
    )
    assert best0 is not None
    assert best0.layer is LiveTrajectoryPoseLayer.OPTIMIZED_KEYFRAME
    assert best0.source_keyframe_id == "stella-keyframe-0"
    assert np.allclose(best0.T_reconstruction_from_camera, _matrix(-0.1))
    assert best1_no_propagation is not None
    assert best1_no_propagation.layer is LiveTrajectoryPoseLayer.TRACKING

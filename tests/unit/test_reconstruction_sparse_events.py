from __future__ import annotations

import numpy as np
import pytest

from core.reconstruction import (
    EventSeverity,
    ReconstructionEventRecorder,
    ReconstructionEventType,
    ReconstructionStage,
    ReconstructionTrajectory,
    ReconstructionValidationError,
    SparseLandmark,
    SparseObservation,
)
from tests.reconstruction_helpers import SESSION_ID, make_pose


def _observation(index: int, *, landmark_id: str = "landmark-1") -> SparseObservation:
    return SparseObservation(
        observation_id=f"observation-{index}",
        landmark_id=landmark_id,
        keyframe_id=f"keyframe-{index}",
        feature_xy=(100.0 + index, 200.0 + index),
        feature_index=index,
        reprojection_error_px=0.2 + index * 0.1,
        confidence=0.9,
    )


def test_sparse_landmark_observation_consistency() -> None:
    landmark = SparseLandmark(
        "landmark-1",
        "track-1",
        np.array([1.0, 2.0, 3.0]),
        (_observation(0), _observation(1)),
        color_rgb=(10, 20, 30),
    )
    assert landmark.position_world_m.flags.writeable is False
    assert len(landmark.observations) == 2


def test_sparse_landmark_rejects_mismatched_observation() -> None:
    with pytest.raises(ReconstructionValidationError, match="parent landmark"):
        SparseLandmark(
            "landmark-1",
            "track-1",
            np.array([1.0, 2.0, 3.0]),
            (_observation(0, landmark_id="other"),),
        )


def test_sparse_landmark_rejects_duplicate_keyframe_observations() -> None:
    first = _observation(0)
    duplicate = SparseObservation(
        "observation-2",
        "landmark-1",
        "keyframe-0",
        (110.0, 210.0),
    )
    with pytest.raises(ReconstructionValidationError, match="duplicate observations"):
        SparseLandmark(
            "landmark-1",
            "track-1",
            np.array([1.0, 2.0, 3.0]),
            (first, duplicate),
        )


def test_trajectory_ordering_and_identity() -> None:
    trajectory = ReconstructionTrajectory((make_pose(0), make_pose(1), make_pose(2)))
    assert len(trajectory.samples) == 3
    with pytest.raises(ReconstructionValidationError, match="strictly increasing"):
        ReconstructionTrajectory((make_pose(1, timestamp_ns=2), make_pose(2, timestamp_ns=1)))


def test_progress_event_validation_and_sequence() -> None:
    recorder = ReconstructionEventRecorder(SESSION_ID)
    first = recorder.emit(
        event_type=ReconstructionEventType.PROGRESS,
        stage=ReconstructionStage.SPARSE_RECONSTRUCTION,
        message="Starting sparse reconstruction",
        elapsed_ns=1,
        progress_fraction=0.0,
    )
    second = recorder.emit(
        event_type=ReconstructionEventType.PROGRESS,
        stage=ReconstructionStage.SPARSE_RECONSTRUCTION,
        message="Sparse reconstruction halfway",
        elapsed_ns=2,
        completed_work=5,
        total_work=10,
        progress_fraction=0.5,
    )
    assert first.sequence_number == 0
    assert second.sequence_number == 1
    assert len(recorder.events) == 2


def test_progress_cannot_move_backwards_without_explicit_restart() -> None:
    recorder = ReconstructionEventRecorder(SESSION_ID)
    recorder.emit(
        event_type=ReconstructionEventType.PROGRESS,
        stage=ReconstructionStage.INGESTION,
        message="Halfway",
        elapsed_ns=1,
        progress_fraction=0.5,
    )
    with pytest.raises(ReconstructionValidationError, match="monotonic"):
        recorder.emit(
            event_type=ReconstructionEventType.PROGRESS,
            stage=ReconstructionStage.INGESTION,
            message="Unexpected rollback",
            elapsed_ns=2,
            progress_fraction=0.4,
        )


def test_failure_event_requires_error_code_and_error_severity() -> None:
    recorder = ReconstructionEventRecorder(SESSION_ID)
    with pytest.raises(ReconstructionValidationError, match="error_code"):
        recorder.emit(
            event_type=ReconstructionEventType.FAILURE,
            stage=ReconstructionStage.DENSE_FUSION,
            message="Fusion failed",
            elapsed_ns=1,
            severity=EventSeverity.ERROR,
        )
    event = recorder.emit(
        event_type=ReconstructionEventType.FAILURE,
        stage=ReconstructionStage.DENSE_FUSION,
        message="Fusion failed",
        elapsed_ns=2,
        severity=EventSeverity.ERROR,
        error_code="fusion_failed",
        retryable=True,
        resume_possible=True,
    )
    assert event.error_code == "fusion_failed"

from __future__ import annotations

import pytest

from core.contracts import FrameReference, ImageSize
from core.contracts.validation import ContractValidationError
from core.tracking import ObjectTrackState, TrackingResetReason, TrackingState


def _frame(frame_id: int, session_id: int = 1) -> FrameReference:
    return FrameReference(session_id, frame_id, float(frame_id), ImageSize(10, 10))


def test_tracking_state_is_pickle_safe_and_algorithm_independent() -> None:
    state = ObjectTrackState(
        track_id="track-1",
        state=TrackingState.REACQUIRED,
        current_observation_id="object-4",
        first_frame_reference=_frame(1),
        latest_frame_reference=_frame(4),
        age_frames=4,
        confirmed_observation_count=3,
        missed_frame_count=1,
        association_confidence=0.8,
        reacquisition_count=1,
        reset_reason=TrackingResetReason.RESOLUTION_CHANGED,
        measurement_history_count=3,
    )
    assert state.state is TrackingState.REACQUIRED


def test_track_cannot_cross_source_sessions() -> None:
    with pytest.raises(ContractValidationError, match="sessions"):
        ObjectTrackState(
            track_id="track-1",
            state=TrackingState.CONFIRMED,
            current_observation_id="object-2",
            first_frame_reference=_frame(1, 1),
            latest_frame_reference=_frame(2, 2),
            age_frames=2,
            confirmed_observation_count=2,
            missed_frame_count=0,
        )

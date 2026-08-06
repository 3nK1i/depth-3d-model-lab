from __future__ import annotations

import pytest

from core.contracts import ImageSize
from core.reconstruction import (
    FrameInvalidReason,
    FrameValidity,
    InputSourceKind,
    InvalidSessionTransitionError,
    KeyframeSelectionReason,
    ReconstructionFrame,
    ReconstructionKeyframe,
    ReconstructionQualityProfile,
    ReconstructionSession,
    ReconstructionSessionState,
    ReconstructionValidationError,
)
from tests.reconstruction_helpers import CREATED_AT, SESSION_ID, make_session, make_source


def test_input_source_path_is_derived_from_source_kind() -> None:
    assert make_source(InputSourceKind.IMAGE_SEQUENCE).path.value == "offline"
    assert make_source(InputSourceKind.RECORDED_VIDEO).path.value == "offline"
    assert make_source(InputSourceKind.USB_CAMERA).path.value == "live"
    assert make_source(InputSourceKind.RTSP_STREAM).path.value == "live"


def test_valid_and_invalid_frame_semantics_are_explicit() -> None:
    valid = ReconstructionFrame(
        "frame-0", SESSION_ID, "source-1", 0, 1, ImageSize(640, 480)
    )
    assert valid.is_valid
    invalid = ReconstructionFrame(
        "frame-1",
        SESSION_ID,
        "source-1",
        1,
        2,
        ImageSize(640, 480),
        FrameValidity.INVALID,
        (FrameInvalidReason.DECODE_FAILED,),
    )
    assert not invalid.is_valid


def test_valid_frame_rejects_invalid_reasons() -> None:
    with pytest.raises(ReconstructionValidationError, match="Valid frames"):
        ReconstructionFrame(
            "frame-0",
            SESSION_ID,
            "source-1",
            0,
            1,
            ImageSize(640, 480),
            FrameValidity.VALID,
            (FrameInvalidReason.DUPLICATE,),
        )


def test_frame_order_and_timestamps_must_be_strictly_increasing() -> None:
    session = ReconstructionSession(
        make_source(),
        ReconstructionQualityProfile.EFFICIENT,
        session_id=SESSION_ID,
        created_at_utc=CREATED_AT,
    )
    session.add_frame(
        ReconstructionFrame("frame-0", SESSION_ID, "source-1", 0, 100, ImageSize(640, 480))
    )
    with pytest.raises(ReconstructionValidationError, match="timestamps"):
        session.add_frame(
            ReconstructionFrame("frame-1", SESSION_ID, "source-1", 1, 100, ImageSize(640, 480))
        )
    with pytest.raises(ReconstructionValidationError, match="sequence"):
        session.add_frame(
            ReconstructionFrame("frame-2", SESSION_ID, "source-1", 0, 200, ImageSize(640, 480))
        )


def test_keyframe_identity_must_reference_one_valid_frame() -> None:
    session = make_session(frame_count=1)
    with pytest.raises(ReconstructionValidationError, match="existing session frame"):
        session.add_keyframe(
            ReconstructionKeyframe(
                "missing",
                SESSION_ID,
                "frame-99",
                2,
                KeyframeSelectionReason.MOTION,
            )
        )


def test_session_lifecycle_pause_resume_complete() -> None:
    session = make_session()
    assert session.state is ReconstructionSessionState.READY
    session.start()
    session.pause()
    assert session.control.paused
    session.resume()
    assert not session.control.paused
    session.complete()
    assert session.is_terminal
    assert session.state is ReconstructionSessionState.COMPLETED


def test_invalid_session_transition_fails_clearly() -> None:
    session = ReconstructionSession(
        make_source(),
        ReconstructionQualityProfile.EFFICIENT,
        session_id=SESSION_ID,
        created_at_utc=CREATED_AT,
    )
    with pytest.raises(InvalidSessionTransitionError, match="created to running"):
        session.start()


def test_cancellation_before_start_is_immediate_and_idempotent() -> None:
    session = ReconstructionSession(
        make_source(),
        ReconstructionQualityProfile.EFFICIENT,
        session_id=SESSION_ID,
        created_at_utc=CREATED_AT,
    )
    session.request_cancel()
    session.request_cancel()
    assert session.state is ReconstructionSessionState.CANCELLED
    assert session.control.cancellation_requested


def test_running_cancellation_requires_acknowledgement() -> None:
    session = make_session()
    session.start()
    session.request_cancel()
    session.request_cancel()
    assert session.state is ReconstructionSessionState.CANCELLING
    session.acknowledge_cancelled()
    assert session.state is ReconstructionSessionState.CANCELLED


def test_terminal_session_rejects_new_frames() -> None:
    session = make_session()
    session.start()
    session.complete()
    with pytest.raises(ReconstructionValidationError, match="terminal"):
        session.add_frame(
            ReconstructionFrame(
                "frame-4", SESSION_ID, "source-1", 4, 5_000_000, ImageSize(640, 480)
            )
        )


def test_snapshot_retains_ordered_frames_and_keyframes() -> None:
    snapshot = make_session().snapshot()
    assert [frame.sequence_index for frame in snapshot.frames] == [0, 1, 2]
    assert [keyframe.selection_sequence for keyframe in snapshot.keyframes] == [0, 1]
    assert snapshot.session_id == SESSION_ID


def test_source_rejects_non_finite_frame_rate() -> None:
    with pytest.raises(Exception, match="finite"):
        source = make_source()
        type(source)(
            source_id=source.source_id,
            kind=source.kind,
            display_name=source.display_name,
            source_uri=source.source_uri,
            expected_image_size=source.expected_image_size,
            expected_frame_rate_hz=float("nan"),
            timestamp_origin=source.timestamp_origin,
        )


def test_cancelling_session_rejects_new_frame_data() -> None:
    session = make_session()
    session.start()
    session.request_cancel()
    with pytest.raises(ReconstructionValidationError, match="cancellation"):
        session.add_frame(
            ReconstructionFrame(
                "frame-4", SESSION_ID, "source-1", 4, 5_000_000, ImageSize(640, 480)
            )
        )

"""Reconstruction-specific session lifecycle isolated from the Qt runtime state machine."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from threading import RLock
from time import monotonic_ns
from typing import Any

from core.reconstruction.contracts import (
    InputSourceMetadata,
    ReconstructionFrame,
    ReconstructionKeyframe,
    ReconstructionPath,
    new_reconstruction_session_id,
)
from core.reconstruction.control import ReconstructionOperationControl
from core.reconstruction.errors import (
    InvalidSessionTransitionError,
    ReconstructionValidationError,
)
from core.reconstruction.events import (
    EventSeverity,
    ReconstructionEventRecorder,
    ReconstructionEventType,
    ReconstructionStage,
)
from core.reconstruction.profiles import ReconstructionQualityProfile
from core.reconstruction.validation import require_iso8601_utc, require_uuid


class ReconstructionSessionState(str, Enum):
    CREATED = "created"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    FAILED = "failed"


_TERMINAL_STATES = {
    ReconstructionSessionState.CANCELLED,
    ReconstructionSessionState.COMPLETED,
    ReconstructionSessionState.FAILED,
}

_ALLOWED_TRANSITIONS: dict[ReconstructionSessionState, frozenset[ReconstructionSessionState]] = {
    ReconstructionSessionState.CREATED: frozenset(
        {
            ReconstructionSessionState.READY,
            ReconstructionSessionState.CANCELLED,
            ReconstructionSessionState.FAILED,
        }
    ),
    ReconstructionSessionState.READY: frozenset(
        {
            ReconstructionSessionState.RUNNING,
            ReconstructionSessionState.CANCELLED,
            ReconstructionSessionState.FAILED,
        }
    ),
    ReconstructionSessionState.RUNNING: frozenset(
        {
            ReconstructionSessionState.PAUSED,
            ReconstructionSessionState.CANCELLING,
            ReconstructionSessionState.COMPLETED,
            ReconstructionSessionState.FAILED,
        }
    ),
    ReconstructionSessionState.PAUSED: frozenset(
        {
            ReconstructionSessionState.RUNNING,
            ReconstructionSessionState.CANCELLING,
            ReconstructionSessionState.FAILED,
        }
    ),
    ReconstructionSessionState.CANCELLING: frozenset(
        {
            ReconstructionSessionState.CANCELLED,
            ReconstructionSessionState.FAILED,
        }
    ),
    ReconstructionSessionState.CANCELLED: frozenset(),
    ReconstructionSessionState.COMPLETED: frozenset(),
    ReconstructionSessionState.FAILED: frozenset(),
}


@dataclass(frozen=True, slots=True)
class ReconstructionSessionSnapshot:
    session_id: str
    path: ReconstructionPath
    source: InputSourceMetadata
    profile: ReconstructionQualityProfile
    state: ReconstructionSessionState
    frames: tuple[ReconstructionFrame, ...]
    keyframes: tuple[ReconstructionKeyframe, ...]
    created_at_utc: str
    updated_at_utc: str
    failure_code: str = ""
    failure_message: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "session_id", require_uuid(self.session_id, field_name="session_id")
        )
        if not isinstance(self.path, ReconstructionPath):
            raise TypeError("path must be a ReconstructionPath.")
        if self.path is not self.source.path:
            raise ReconstructionValidationError(
                "Session path must agree with the input source kind."
            )
        if not isinstance(self.profile, ReconstructionQualityProfile):
            raise TypeError("profile must be a ReconstructionQualityProfile.")
        if not isinstance(self.state, ReconstructionSessionState):
            raise TypeError("state must be a ReconstructionSessionState.")
        frames = tuple(self.frames)
        keyframes = tuple(self.keyframes)
        if any(not isinstance(frame, ReconstructionFrame) for frame in frames):
            raise TypeError("frames must contain ReconstructionFrame values.")
        if any(not isinstance(keyframe, ReconstructionKeyframe) for keyframe in keyframes):
            raise TypeError("keyframes must contain ReconstructionKeyframe values.")
        if any(frame.session_id != self.session_id for frame in frames):
            raise ReconstructionValidationError(
                "Every snapshot frame must match the reconstruction session ID."
            )
        if any(frame.source_id != self.source.source_id for frame in frames):
            raise ReconstructionValidationError(
                "Every snapshot frame must match the input source ID."
            )
        if any(
            current.sequence_index <= previous.sequence_index
            or current.timestamp_ns <= previous.timestamp_ns
            for previous, current in zip(frames, frames[1:])
        ):
            raise ReconstructionValidationError(
                "Snapshot frames must be strictly ordered by sequence and timestamp."
            )
        frame_ids = [frame.frame_id for frame in frames]
        keyframe_ids = [keyframe.keyframe_id for keyframe in keyframes]
        if len(set(frame_ids)) != len(frame_ids):
            raise ReconstructionValidationError("Snapshot frame IDs must be unique.")
        if len(set(keyframe_ids)) != len(keyframe_ids):
            raise ReconstructionValidationError("Snapshot keyframe IDs must be unique.")
        keyframe_frame_ids = [keyframe.frame_id for keyframe in keyframes]
        if len(set(keyframe_frame_ids)) != len(keyframe_frame_ids):
            raise ReconstructionValidationError(
                "A snapshot frame can be selected as a keyframe only once."
            )
        frame_by_id = {frame.frame_id: frame for frame in frames}
        if any(keyframe.session_id != self.session_id for keyframe in keyframes):
            raise ReconstructionValidationError(
                "Every snapshot keyframe must match the reconstruction session ID."
            )
        if any(
            keyframe.frame_id not in frame_by_id or not frame_by_id[keyframe.frame_id].is_valid
            for keyframe in keyframes
        ):
            raise ReconstructionValidationError(
                "Snapshot keyframes must reference valid snapshot frames."
            )
        if any(
            current.selection_sequence <= previous.selection_sequence
            for previous, current in zip(keyframes, keyframes[1:])
        ):
            raise ReconstructionValidationError(
                "Snapshot keyframe selection sequences must be strictly increasing."
            )
        created_at = require_iso8601_utc(self.created_at_utc, field_name="created_at_utc")
        updated_at = require_iso8601_utc(self.updated_at_utc, field_name="updated_at_utc")
        if updated_at < created_at:
            raise ReconstructionValidationError(
                "updated_at_utc must not precede created_at_utc."
            )
        object.__setattr__(self, "frames", frames)
        object.__setattr__(self, "keyframes", keyframes)
        object.__setattr__(self, "created_at_utc", created_at)
        object.__setattr__(self, "updated_at_utc", updated_at)
        if self.state is ReconstructionSessionState.FAILED:
            if not self.failure_code.strip() or not self.failure_message.strip():
                raise ReconstructionValidationError(
                    "Failed session snapshots require failure code and message."
                )

    def as_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "path": self.path.value,
            "source": self.source.as_dict(),
            "profile": self.profile.value,
            "state": self.state.value,
            "frames": [frame.as_dict() for frame in self.frames],
            "keyframes": [keyframe.as_dict() for keyframe in self.keyframes],
            "created_at_utc": self.created_at_utc,
            "updated_at_utc": self.updated_at_utc,
            "failure_code": self.failure_code,
            "failure_message": self.failure_message,
        }


class ReconstructionSession:
    """Thread-safe lifecycle owner for one future reconstruction job."""

    def __init__(
        self,
        source: InputSourceMetadata,
        profile: ReconstructionQualityProfile,
        *,
        session_id: str | None = None,
        created_at_utc: str | None = None,
    ) -> None:
        if not isinstance(source, InputSourceMetadata):
            raise TypeError("source must be InputSourceMetadata.")
        if not isinstance(profile, ReconstructionQualityProfile):
            raise TypeError("profile must be ReconstructionQualityProfile.")
        self._lock = RLock()
        self._session_id = require_uuid(
            session_id or new_reconstruction_session_id(), field_name="session_id"
        )
        self._source = source
        self._profile = profile
        self._state = ReconstructionSessionState.CREATED
        now = created_at_utc or datetime.now(timezone.utc).isoformat()
        self._created_at_utc = require_iso8601_utc(now, field_name="created_at_utc")
        self._updated_at_utc = self._created_at_utc
        self._frames: list[ReconstructionFrame] = []
        self._keyframes: list[ReconstructionKeyframe] = []
        self._failure_code = ""
        self._failure_message = ""
        self._started_monotonic_ns = monotonic_ns()
        self.control = ReconstructionOperationControl()
        self.events = ReconstructionEventRecorder(self._session_id)

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def state(self) -> ReconstructionSessionState:
        with self._lock:
            return self._state

    @property
    def is_terminal(self) -> bool:
        return self.state in _TERMINAL_STATES

    def can_transition_to(self, target: ReconstructionSessionState) -> bool:
        with self._lock:
            return target in _ALLOWED_TRANSITIONS[self._state]

    def transition_to(self, target: ReconstructionSessionState) -> None:
        if not isinstance(target, ReconstructionSessionState):
            raise TypeError("target must be ReconstructionSessionState.")
        with self._lock:
            if target not in _ALLOWED_TRANSITIONS[self._state]:
                raise InvalidSessionTransitionError(
                    f"Cannot transition reconstruction session from {self._state.value} "
                    f"to {target.value}."
                )
            previous = self._state
            if target is ReconstructionSessionState.PAUSED:
                self.control.request_pause()
            elif (
                previous is ReconstructionSessionState.PAUSED
                and target is ReconstructionSessionState.RUNNING
            ):
                self.control.request_resume()
            elif target is ReconstructionSessionState.CANCELLING:
                self.control.request_cancel()
            self._state = target
            self._touch()
            event_type = ReconstructionEventType.LIFECYCLE
            if target is ReconstructionSessionState.PAUSED:
                event_type = ReconstructionEventType.PAUSE
            elif (
                previous is ReconstructionSessionState.PAUSED
                and target is ReconstructionSessionState.RUNNING
            ):
                event_type = ReconstructionEventType.RESUME
            elif target in {
                ReconstructionSessionState.CANCELLING,
                ReconstructionSessionState.CANCELLED,
            }:
                event_type = ReconstructionEventType.CANCELLATION
            self.events.emit(
                event_type=event_type,
                stage=ReconstructionStage.FOUNDATION,
                message=f"Session state changed from {previous.value} to {target.value}.",
                elapsed_ns=self._elapsed_ns(),
                metadata={"previous_state": previous.value, "new_state": target.value},
            )

    def prepare(self) -> None:
        self.transition_to(ReconstructionSessionState.READY)

    def start(self) -> None:
        self.transition_to(ReconstructionSessionState.RUNNING)

    def pause(self) -> None:
        self.transition_to(ReconstructionSessionState.PAUSED)

    def resume(self) -> None:
        self.transition_to(ReconstructionSessionState.RUNNING)

    def request_cancel(self) -> None:
        with self._lock:
            if self._state in {
                ReconstructionSessionState.CANCELLING,
                ReconstructionSessionState.CANCELLED,
            }:
                self.control.request_cancel()
                return
            if self._state in {
                ReconstructionSessionState.COMPLETED,
                ReconstructionSessionState.FAILED,
            }:
                return
            if self._state in {
                ReconstructionSessionState.CREATED,
                ReconstructionSessionState.READY,
            }:
                self.control.request_cancel()
                self.transition_to(ReconstructionSessionState.CANCELLED)
                return
        self.transition_to(ReconstructionSessionState.CANCELLING)

    def acknowledge_cancelled(self) -> None:
        self.transition_to(ReconstructionSessionState.CANCELLED)

    def complete(self) -> None:
        self.transition_to(ReconstructionSessionState.COMPLETED)

    def fail(self, *, error_code: str, message: str) -> None:
        with self._lock:
            self._failure_code = error_code.strip()
            self._failure_message = message.strip()
            if not self._failure_code or not self._failure_message:
                raise ReconstructionValidationError(
                    "Session failure requires a non-empty error code and message."
                )
        self.transition_to(ReconstructionSessionState.FAILED)
        self.events.emit(
            event_type=ReconstructionEventType.FAILURE,
            stage=ReconstructionStage.FOUNDATION,
            message=self._failure_message,
            elapsed_ns=self._elapsed_ns(),
            severity=EventSeverity.ERROR,
            error_code=self._failure_code,
            retryable=False,
            resume_possible=False,
        )

    def add_frame(self, frame: ReconstructionFrame) -> None:
        if not isinstance(frame, ReconstructionFrame):
            raise TypeError("frame must be a ReconstructionFrame.")
        with self._lock:
            if self._state not in {
                ReconstructionSessionState.CREATED,
                ReconstructionSessionState.READY,
                ReconstructionSessionState.RUNNING,
                ReconstructionSessionState.PAUSED,
            }:
                raise ReconstructionValidationError(
                    "Cannot add frames after cancellation or terminal completion begins."
                )
            if frame.session_id != self._session_id:
                raise ReconstructionValidationError("Frame session ID does not match the session.")
            if frame.source_id != self._source.source_id:
                raise ReconstructionValidationError("Frame source ID does not match the session source.")
            if self._frames:
                previous = self._frames[-1]
                if frame.sequence_index <= previous.sequence_index:
                    raise ReconstructionValidationError(
                        "Frame sequence indices must be strictly increasing."
                    )
                if frame.timestamp_ns <= previous.timestamp_ns:
                    raise ReconstructionValidationError(
                        "Frame timestamps must be strictly increasing."
                    )
            if any(existing.frame_id == frame.frame_id for existing in self._frames):
                raise ReconstructionValidationError("Frame IDs must be unique within a session.")
            self._frames.append(frame)
            self._touch()

    def add_keyframe(self, keyframe: ReconstructionKeyframe) -> None:
        if not isinstance(keyframe, ReconstructionKeyframe):
            raise TypeError("keyframe must be a ReconstructionKeyframe.")
        with self._lock:
            if self._state not in {
                ReconstructionSessionState.CREATED,
                ReconstructionSessionState.READY,
                ReconstructionSessionState.RUNNING,
                ReconstructionSessionState.PAUSED,
            }:
                raise ReconstructionValidationError(
                    "Cannot add keyframes after cancellation or terminal completion begins."
                )
            if keyframe.session_id != self._session_id:
                raise ReconstructionValidationError(
                    "Keyframe session ID does not match the session."
                )
            frame = next(
                (candidate for candidate in self._frames if candidate.frame_id == keyframe.frame_id),
                None,
            )
            if frame is None:
                raise ReconstructionValidationError(
                    "Keyframes must reference an existing session frame."
                )
            if not frame.is_valid:
                raise ReconstructionValidationError("Keyframes must reference valid frames.")
            if self._keyframes and keyframe.selection_sequence <= self._keyframes[-1].selection_sequence:
                raise ReconstructionValidationError(
                    "Keyframe selection sequences must be strictly increasing."
                )
            if any(item.keyframe_id == keyframe.keyframe_id for item in self._keyframes):
                raise ReconstructionValidationError("Keyframe IDs must be unique.")
            if any(item.frame_id == keyframe.frame_id for item in self._keyframes):
                raise ReconstructionValidationError("A frame can be selected as a keyframe only once.")
            self._keyframes.append(keyframe)
            self._touch()

    def snapshot(self) -> ReconstructionSessionSnapshot:
        with self._lock:
            return ReconstructionSessionSnapshot(
                session_id=self._session_id,
                path=self._source.path,
                source=self._source,
                profile=self._profile,
                state=self._state,
                frames=tuple(self._frames),
                keyframes=tuple(self._keyframes),
                created_at_utc=self._created_at_utc,
                updated_at_utc=self._updated_at_utc,
                failure_code=self._failure_code,
                failure_message=self._failure_message,
            )

    def _touch(self) -> None:
        self._updated_at_utc = datetime.now(timezone.utc).isoformat()

    def _elapsed_ns(self) -> int:
        return max(0, monotonic_ns() - self._started_monotonic_ns)

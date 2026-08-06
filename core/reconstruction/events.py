"""Structured reconstruction events suitable for CLI and future Qt bridges."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from threading import Lock
from typing import Any, Mapping, Sequence
from uuid import uuid4

from core.contracts.validation import (
    JsonValue,
    metadata_as_dict,
    normalize_metadata,
    require_nonempty,
)
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.validation import (
    require_iso8601_utc,
    require_nonnegative_int,
    require_probability,
    require_timestamp_ns,
    require_uuid,
)


class ReconstructionStage(str, Enum):
    FOUNDATION = "foundation"
    INGESTION = "ingestion"
    KEYFRAME_SELECTION = "keyframe_selection"
    SPARSE_RECONSTRUCTION = "sparse_reconstruction"
    LIVE_TRACKING = "live_tracking"
    LOOP_CLOSURE = "loop_closure"
    DENSE_FUSION = "dense_fusion"
    MESH_GENERATION = "mesh_generation"
    EXPORT = "export"


class ReconstructionEventType(str, Enum):
    STATUS = "status"
    PROGRESS = "progress"
    WARNING = "warning"
    FAILURE = "failure"
    LIFECYCLE = "lifecycle"
    CHECKPOINT = "checkpoint"
    CANCELLATION = "cancellation"
    PAUSE = "pause"
    RESUME = "resume"


class EventSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class ReconstructionEvent:
    event_id: str
    session_id: str
    sequence_number: int
    timestamp_utc: str
    elapsed_ns: int
    event_type: ReconstructionEventType
    stage: ReconstructionStage
    message: str
    severity: EventSeverity = EventSeverity.INFO
    progress_fraction: float | None = None
    completed_work: int | None = None
    total_work: int | None = None
    error_code: str = ""
    retryable: bool = False
    resume_possible: bool = False
    suggested_action: str = ""
    metadata: Mapping[str, JsonValue] | Sequence[tuple[str, JsonValue]] = field(
        default_factory=tuple
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", require_uuid(self.event_id, field_name="event_id"))
        object.__setattr__(
            self, "session_id", require_uuid(self.session_id, field_name="session_id")
        )
        object.__setattr__(
            self,
            "sequence_number",
            require_nonnegative_int(self.sequence_number, field_name="sequence_number"),
        )
        object.__setattr__(
            self,
            "timestamp_utc",
            require_iso8601_utc(self.timestamp_utc, field_name="timestamp_utc"),
        )
        object.__setattr__(self, "elapsed_ns", require_timestamp_ns(self.elapsed_ns, field_name="elapsed_ns"))
        if not isinstance(self.event_type, ReconstructionEventType):
            raise TypeError("event_type must be a ReconstructionEventType.")
        if not isinstance(self.stage, ReconstructionStage):
            raise TypeError("stage must be a ReconstructionStage.")
        if not isinstance(self.severity, EventSeverity):
            raise TypeError("severity must be an EventSeverity.")
        object.__setattr__(self, "message", require_nonempty(self.message, field_name="message"))
        if self.progress_fraction is not None:
            object.__setattr__(
                self,
                "progress_fraction",
                require_probability(self.progress_fraction, field_name="progress_fraction"),
            )
        if (self.completed_work is None) != (self.total_work is None):
            raise ReconstructionValidationError(
                "completed_work and total_work must either both be present or both be absent."
            )
        if self.completed_work is not None and self.total_work is not None:
            completed = require_nonnegative_int(self.completed_work, field_name="completed_work")
            total = require_nonnegative_int(self.total_work, field_name="total_work")
            if total <= 0 or completed > total:
                raise ReconstructionValidationError(
                    "Progress work counts require 0 <= completed_work <= total_work and total_work > 0."
                )
            object.__setattr__(self, "completed_work", completed)
            object.__setattr__(self, "total_work", total)
            derived = completed / total
            if self.progress_fraction is not None and abs(self.progress_fraction - derived) > 1e-9:
                raise ReconstructionValidationError(
                    "progress_fraction must agree with completed_work / total_work."
                )
        if self.event_type is ReconstructionEventType.FAILURE:
            error_code = str(self.error_code).strip()
            if not error_code:
                raise ReconstructionValidationError("error_code must not be empty.")
            object.__setattr__(self, "error_code", error_code)
            if self.severity is not EventSeverity.ERROR:
                raise ReconstructionValidationError("Failure events must use ERROR severity.")
        else:
            object.__setattr__(self, "error_code", str(self.error_code).strip())
        object.__setattr__(self, "suggested_action", str(self.suggested_action).strip())
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "session_id": self.session_id,
            "sequence_number": self.sequence_number,
            "timestamp_utc": self.timestamp_utc,
            "elapsed_ns": self.elapsed_ns,
            "event_type": self.event_type.value,
            "stage": self.stage.value,
            "message": self.message,
            "severity": self.severity.value,
            "progress_fraction": self.progress_fraction,
            "completed_work": self.completed_work,
            "total_work": self.total_work,
            "error_code": self.error_code,
            "retryable": self.retryable,
            "resume_possible": self.resume_possible,
            "suggested_action": self.suggested_action,
            "metadata": metadata_as_dict(self.metadata),  # type: ignore[arg-type]
        }


class ReconstructionEventRecorder:
    """Thread-safe event sequencer with monotonic per-stage progress validation."""

    def __init__(self, session_id: str) -> None:
        self._session_id = require_uuid(session_id, field_name="session_id")
        self._lock = Lock()
        self._events: list[ReconstructionEvent] = []
        self._last_progress: dict[ReconstructionStage, float] = {}

    def emit(
        self,
        *,
        event_type: ReconstructionEventType,
        stage: ReconstructionStage,
        message: str,
        elapsed_ns: int,
        timestamp_utc: str | None = None,
        severity: EventSeverity = EventSeverity.INFO,
        progress_fraction: float | None = None,
        completed_work: int | None = None,
        total_work: int | None = None,
        error_code: str = "",
        retryable: bool = False,
        resume_possible: bool = False,
        suggested_action: str = "",
        metadata: Mapping[str, JsonValue] | Sequence[tuple[str, JsonValue]] = (),
        allow_progress_restart: bool = False,
    ) -> ReconstructionEvent:
        with self._lock:
            sequence = len(self._events)
            event = ReconstructionEvent(
                event_id=str(uuid4()),
                session_id=self._session_id,
                sequence_number=sequence,
                timestamp_utc=timestamp_utc or datetime.now(timezone.utc).isoformat(),
                elapsed_ns=elapsed_ns,
                event_type=event_type,
                stage=stage,
                message=message,
                severity=severity,
                progress_fraction=progress_fraction,
                completed_work=completed_work,
                total_work=total_work,
                error_code=error_code,
                retryable=retryable,
                resume_possible=resume_possible,
                suggested_action=suggested_action,
                metadata=metadata,
            )
            if event.progress_fraction is not None:
                previous = self._last_progress.get(stage)
                if previous is not None and event.progress_fraction < previous and not allow_progress_restart:
                    raise ReconstructionValidationError(
                        "Progress must be monotonic within one stage unless restart is explicit."
                    )
                self._last_progress[stage] = event.progress_fraction
            self._events.append(event)
            return event

    @property
    def events(self) -> tuple[ReconstructionEvent, ...]:
        with self._lock:
            return tuple(self._events)

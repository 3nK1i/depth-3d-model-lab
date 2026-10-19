"""Versioned JSON-line protocol for the external Phase 8.3 PyCOLMAP worker."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from typing import Any, Mapping

from core.contracts.validation import require_nonempty, require_finite
from core.reconstruction.errors import ReconstructionValidationError, SfmWorkerProtocolError
from core.reconstruction.validation import require_nonnegative_int, require_probability

SFM_WORKER_SCHEMA_VERSION = 1
SFM_WORKER_EVENT_PREFIX = "DEPTH3D_SFM_EVENT "


class SfmWorkerEventType(str, Enum):
    STATUS = "status"
    PROGRESS = "progress"
    WARNING = "warning"


@dataclass(frozen=True, slots=True)
class SfmWorkerEvent:
    event_type: SfmWorkerEventType
    stage: str
    message: str
    progress_fraction: float | None = None
    completed_work: int | None = None
    total_work: int | None = None
    metadata: tuple[tuple[str, object], ...] = ()
    schema_version: int = SFM_WORKER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.event_type, SfmWorkerEventType):
            raise TypeError("event_type must be SfmWorkerEventType.")
        object.__setattr__(self, "stage", require_nonempty(self.stage, field_name="stage"))
        object.__setattr__(self, "message", require_nonempty(self.message, field_name="message"))
        if self.progress_fraction is not None:
            object.__setattr__(
                self,
                "progress_fraction",
                require_probability(self.progress_fraction, field_name="progress_fraction"),
            )
        if (self.completed_work is None) != (self.total_work is None):
            raise ReconstructionValidationError(
                "completed_work and total_work must both be present or both absent."
            )
        if self.completed_work is not None and self.total_work is not None:
            completed = require_nonnegative_int(self.completed_work, field_name="completed_work")
            total = require_nonnegative_int(self.total_work, field_name="total_work")
            if total <= 0 or completed > total:
                raise ReconstructionValidationError(
                    "Worker progress requires 0 <= completed_work <= total_work and total_work > 0."
                )
            object.__setattr__(self, "completed_work", completed)
            object.__setattr__(self, "total_work", total)
        metadata = tuple((str(key), value) for key, value in self.metadata)
        if len({key for key, _ in metadata}) != len(metadata):
            raise ReconstructionValidationError("Worker event metadata keys must be unique.")
        object.__setattr__(self, "metadata", metadata)
        if int(self.schema_version) != SFM_WORKER_SCHEMA_VERSION:
            raise ReconstructionValidationError(
                f"Unsupported SfM worker event schema: {self.schema_version}."
            )
        object.__setattr__(self, "schema_version", int(self.schema_version))

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "SfmWorkerEvent":
        try:
            metadata_raw = payload.get("metadata", {})
            if not isinstance(metadata_raw, Mapping):
                raise TypeError("metadata must be an object")
            return cls(
                event_type=SfmWorkerEventType(str(payload["event_type"])),
                stage=str(payload["stage"]),
                message=str(payload["message"]),
                progress_fraction=(
                    None
                    if payload.get("progress_fraction") is None
                    else require_finite(
                        payload["progress_fraction"], field_name="progress_fraction"
                    )
                ),
                completed_work=payload.get("completed_work"),
                total_work=payload.get("total_work"),
                metadata=tuple((str(key), value) for key, value in metadata_raw.items()),
                schema_version=int(payload.get("schema_version", 0)),
            )
        except (KeyError, TypeError, ValueError, ReconstructionValidationError) as exc:
            raise SfmWorkerProtocolError(f"Invalid SfM worker event payload: {exc}") from exc


def parse_worker_event_line(line: str) -> SfmWorkerEvent | None:
    """Parse one worker stdout line, ignoring ordinary COLMAP/log output."""

    text = str(line).rstrip("\r\n")
    if not text.startswith(SFM_WORKER_EVENT_PREFIX):
        return None
    raw = text[len(SFM_WORKER_EVENT_PREFIX) :]
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SfmWorkerProtocolError(f"Invalid SfM worker event JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise SfmWorkerProtocolError("SfM worker event JSON must be an object.")
    return SfmWorkerEvent.from_payload(payload)

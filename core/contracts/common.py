"""Shared immutable references and provenance for perception and measurement."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from core.contracts.validation import (
    JsonValue,
    MetadataItems,
    metadata_as_dict,
    normalize_metadata,
    require_finite,
    require_nonempty,
)


@dataclass(frozen=True, slots=True)
class ImageSize:
    """Image width and height in pixels."""

    width: int
    height: int

    def __post_init__(self) -> None:
        if isinstance(self.width, bool) or int(self.width) <= 0:
            raise ValueError("width must be a positive integer.")
        if isinstance(self.height, bool) or int(self.height) <= 0:
            raise ValueError("height must be a positive integer.")
        object.__setattr__(self, "width", int(self.width))
        object.__setattr__(self, "height", int(self.height))

    def as_dict(self) -> dict[str, int]:
        return {"width": self.width, "height": self.height}


@dataclass(frozen=True, slots=True)
class FrameReference:
    """Immutable cross-pipeline identity for one captured source frame."""

    session_id: int
    frame_id: int
    source_timestamp: float
    source_size: ImageSize

    def __post_init__(self) -> None:
        if isinstance(self.session_id, bool) or int(self.session_id) < 0:
            raise ValueError("session_id must be a non-negative integer.")
        if isinstance(self.frame_id, bool) or int(self.frame_id) < 0:
            raise ValueError("frame_id must be a non-negative integer.")
        object.__setattr__(self, "session_id", int(self.session_id))
        object.__setattr__(self, "frame_id", int(self.frame_id))
        object.__setattr__(
            self,
            "source_timestamp",
            require_finite(self.source_timestamp, field_name="source_timestamp"),
        )
        if not isinstance(self.source_size, ImageSize):
            raise TypeError("source_size must be an ImageSize.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "frame_id": self.frame_id,
            "source_timestamp": self.source_timestamp,
            "source_size": self.source_size.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class ModelProvenance:
    """Serializable model identity and runtime provenance without live resources."""

    model_id: str
    display_name: str
    task: str
    model_version: str = ""
    checkpoint: str = ""
    device: str = ""
    precision: str = ""
    runtime_config_digest: str = ""
    metadata: Mapping[str, JsonValue] | Sequence[tuple[str, JsonValue]] = field(
        default_factory=tuple
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", require_nonempty(self.model_id, field_name="model_id"))
        object.__setattr__(
            self, "display_name", require_nonempty(self.display_name, field_name="display_name")
        )
        object.__setattr__(self, "task", require_nonempty(self.task, field_name="task"))
        object.__setattr__(self, "model_version", str(self.model_version).strip())
        object.__setattr__(self, "checkpoint", str(self.checkpoint).strip())
        object.__setattr__(self, "device", str(self.device).strip())
        object.__setattr__(self, "precision", str(self.precision).strip())
        object.__setattr__(
            self, "runtime_config_digest", str(self.runtime_config_digest).strip()
        )
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata))

    @property
    def metadata_items(self) -> MetadataItems:
        return self.metadata  # type: ignore[return-value]

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "display_name": self.display_name,
            "task": self.task,
            "model_version": self.model_version,
            "checkpoint": self.checkpoint,
            "device": self.device,
            "precision": self.precision,
            "runtime_config_digest": self.runtime_config_digest,
            "metadata": metadata_as_dict(self.metadata_items),
        }


@dataclass(frozen=True, slots=True)
class ProcessingStep:
    """One explainable processing step recorded in result provenance."""

    step_id: str
    method: str
    input_ids: tuple[str, ...] = ()
    output_ids: tuple[str, ...] = ()
    duration_ms: float | None = None
    parameters: Mapping[str, JsonValue] | Sequence[tuple[str, JsonValue]] = field(
        default_factory=tuple
    )
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "step_id", require_nonempty(self.step_id, field_name="step_id"))
        object.__setattr__(self, "method", require_nonempty(self.method, field_name="method"))
        object.__setattr__(
            self,
            "input_ids",
            tuple(require_nonempty(value, field_name="input_id") for value in self.input_ids),
        )
        object.__setattr__(
            self,
            "output_ids",
            tuple(require_nonempty(value, field_name="output_id") for value in self.output_ids),
        )
        if self.duration_ms is not None:
            duration = require_finite(self.duration_ms, field_name="duration_ms")
            if duration < 0.0:
                raise ValueError("duration_ms must not be negative.")
            object.__setattr__(self, "duration_ms", duration)
        object.__setattr__(self, "parameters", normalize_metadata(self.parameters))
        object.__setattr__(self, "warnings", tuple(str(value).strip() for value in self.warnings))

    def as_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "method": self.method,
            "input_ids": list(self.input_ids),
            "output_ids": list(self.output_ids),
            "duration_ms": self.duration_ms,
            "parameters": metadata_as_dict(self.parameters),  # type: ignore[arg-type]
            "warnings": list(self.warnings),
        }

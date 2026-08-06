"""Serializable reconstruction provenance without live backend resources."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from core.contracts.validation import (
    JsonValue,
    metadata_as_dict,
    normalize_metadata,
    require_nonempty,
)
from core.reconstruction.poses import COORDINATE_CONVENTION_VERSION
from core.reconstruction.validation import require_iso8601_utc


@dataclass(frozen=True, slots=True)
class BackendProvenance:
    backend_id: str
    backend_version: str
    adapter_version: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "backend_id", require_nonempty(self.backend_id, field_name="backend_id")
        )
        object.__setattr__(
            self,
            "backend_version",
            require_nonempty(self.backend_version, field_name="backend_version"),
        )
        object.__setattr__(self, "adapter_version", str(self.adapter_version).strip())

    def as_dict(self) -> dict[str, str]:
        return {
            "backend_id": self.backend_id,
            "backend_version": self.backend_version,
            "adapter_version": self.adapter_version,
        }


@dataclass(frozen=True, slots=True)
class ReconstructionProvenance:
    application_version: str
    source_fingerprint: str
    hardware_label: str
    backends: tuple[BackendProvenance, ...] = ()
    coordinate_convention_version: int = COORDINATE_CONVENTION_VERSION
    created_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Mapping[str, JsonValue] | Sequence[tuple[str, JsonValue]] = field(
        default_factory=tuple
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "application_version",
            require_nonempty(self.application_version, field_name="application_version"),
        )
        object.__setattr__(self, "source_fingerprint", str(self.source_fingerprint).strip().lower())
        object.__setattr__(
            self, "hardware_label", require_nonempty(self.hardware_label, field_name="hardware_label")
        )
        backends = tuple(self.backends)
        if any(not isinstance(item, BackendProvenance) for item in backends):
            raise TypeError("backends must contain BackendProvenance values.")
        backend_ids = [item.backend_id for item in backends]
        if len(set(backend_ids)) != len(backend_ids):
            raise ValueError("Backend provenance IDs must be unique.")
        if int(self.coordinate_convention_version) <= 0:
            raise ValueError("coordinate_convention_version must be positive.")
        object.__setattr__(
            self, "coordinate_convention_version", int(self.coordinate_convention_version)
        )
        object.__setattr__(
            self,
            "created_at_utc",
            require_iso8601_utc(self.created_at_utc, field_name="created_at_utc"),
        )
        object.__setattr__(self, "backends", backends)
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "application_version": self.application_version,
            "source_fingerprint": self.source_fingerprint,
            "hardware_label": self.hardware_label,
            "backends": [item.as_dict() for item in self.backends],
            "coordinate_convention_version": self.coordinate_convention_version,
            "created_at_utc": self.created_at_utc,
            "metadata": metadata_as_dict(self.metadata),  # type: ignore[arg-type]
        }

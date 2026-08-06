"""Versioned checkpoint metadata and deterministic resume compatibility."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from core.contracts.validation import require_nonempty
from core.reconstruction.errors import (
    IncompatibleCheckpointError,
    ReconstructionValidationError,
)
from core.reconstruction.events import ReconstructionStage
from core.reconstruction.poses import COORDINATE_CONVENTION_VERSION
from core.reconstruction.profiles import ReconstructionQualityProfile
from core.reconstruction.session import ReconstructionSessionState
from core.reconstruction.validation import (
    require_iso8601_utc,
    require_safe_relative_path,
    require_sha256,
    require_uuid,
)


RECONSTRUCTION_SCHEMA_VERSION = 1
MINIMUM_READER_SCHEMA_VERSION = 1


class ResumeCompatibilityStatus(str, Enum):
    COMPATIBLE = "compatible"
    INCOMPATIBLE = "incompatible"


@dataclass(frozen=True, slots=True)
class CheckpointArtifact:
    role: str
    relative_path: str
    sha256: str = ""
    required_for_resume: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", require_nonempty(self.role, field_name="role"))
        object.__setattr__(
            self,
            "relative_path",
            require_safe_relative_path(self.relative_path, field_name="relative_path"),
        )
        object.__setattr__(self, "sha256", require_sha256(self.sha256))

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "required_for_resume": self.required_for_resume,
        }


@dataclass(frozen=True, slots=True)
class CheckpointMetadata:
    checkpoint_id: str
    session_id: str
    application_version: str
    session_state: ReconstructionSessionState
    completed_stage: ReconstructionStage
    profile: ReconstructionQualityProfile
    source_fingerprint: str
    backend_versions: tuple[tuple[str, str], ...] = ()
    artifacts: tuple[CheckpointArtifact, ...] = ()
    last_frame_id: str = ""
    last_keyframe_id: str = ""
    resume_allowed: bool = True
    schema_version: int = RECONSTRUCTION_SCHEMA_VERSION
    minimum_reader_version: int = MINIMUM_READER_SCHEMA_VERSION
    coordinate_convention_version: int = COORDINATE_CONVENTION_VERSION
    created_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "checkpoint_id", require_uuid(self.checkpoint_id, field_name="checkpoint_id")
        )
        object.__setattr__(
            self, "session_id", require_uuid(self.session_id, field_name="session_id")
        )
        object.__setattr__(
            self,
            "application_version",
            require_nonempty(self.application_version, field_name="application_version"),
        )
        if int(self.schema_version) != RECONSTRUCTION_SCHEMA_VERSION:
            raise ReconstructionValidationError(
                f"Unsupported reconstruction schema version: {self.schema_version}."
            )
        if int(self.minimum_reader_version) > RECONSTRUCTION_SCHEMA_VERSION:
            raise ReconstructionValidationError(
                "Checkpoint requires a newer reconstruction metadata reader."
            )
        if int(self.coordinate_convention_version) != COORDINATE_CONVENTION_VERSION:
            raise ReconstructionValidationError(
                "Unsupported reconstruction coordinate convention version."
            )
        if not isinstance(self.session_state, ReconstructionSessionState):
            raise TypeError("session_state must be a ReconstructionSessionState.")
        if not isinstance(self.completed_stage, ReconstructionStage):
            raise TypeError("completed_stage must be a ReconstructionStage.")
        if not isinstance(self.profile, ReconstructionQualityProfile):
            raise TypeError("profile must be a ReconstructionQualityProfile.")
        object.__setattr__(
            self,
            "source_fingerprint",
            require_nonempty(self.source_fingerprint, field_name="source_fingerprint").lower(),
        )
        backend_versions = tuple(
            (
                require_nonempty(backend_id, field_name="backend_id"),
                require_nonempty(version, field_name="backend_version"),
            )
            for backend_id, version in self.backend_versions
        )
        backend_ids = [backend_id for backend_id, _ in backend_versions]
        if len(set(backend_ids)) != len(backend_ids):
            raise ReconstructionValidationError("Checkpoint backend IDs must be unique.")
        artifacts = tuple(self.artifacts)
        if any(not isinstance(item, CheckpointArtifact) for item in artifacts):
            raise TypeError("artifacts must contain CheckpointArtifact values.")
        paths = [item.relative_path for item in artifacts]
        if len(set(paths)) != len(paths):
            raise ReconstructionValidationError("Checkpoint artifact paths must be unique.")
        if self.resume_allowed and self.session_state in {
            ReconstructionSessionState.COMPLETED,
            ReconstructionSessionState.CANCELLED,
        }:
            raise ReconstructionValidationError(
                "Completed and cancelled sessions cannot be marked resumable."
            )
        object.__setattr__(self, "backend_versions", backend_versions)
        object.__setattr__(self, "artifacts", artifacts)
        object.__setattr__(self, "last_frame_id", str(self.last_frame_id).strip())
        object.__setattr__(self, "last_keyframe_id", str(self.last_keyframe_id).strip())
        object.__setattr__(
            self,
            "created_at_utc",
            require_iso8601_utc(self.created_at_utc, field_name="created_at_utc"),
        )
        object.__setattr__(self, "schema_version", int(self.schema_version))
        object.__setattr__(self, "minimum_reader_version", int(self.minimum_reader_version))
        object.__setattr__(
            self, "coordinate_convention_version", int(self.coordinate_convention_version)
        )

    @classmethod
    def create(
        cls,
        *,
        session_id: str,
        application_version: str,
        session_state: ReconstructionSessionState,
        completed_stage: ReconstructionStage,
        profile: ReconstructionQualityProfile,
        source_fingerprint: str,
        **kwargs: Any,
    ) -> "CheckpointMetadata":
        return cls(
            checkpoint_id=str(uuid4()),
            session_id=session_id,
            application_version=application_version,
            session_state=session_state,
            completed_stage=completed_stage,
            profile=profile,
            source_fingerprint=source_fingerprint,
            **kwargs,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "session_id": self.session_id,
            "application_version": self.application_version,
            "session_state": self.session_state.value,
            "completed_stage": self.completed_stage.value,
            "profile": self.profile.value,
            "source_fingerprint": self.source_fingerprint,
            "backend_versions": [
                {"backend_id": backend_id, "backend_version": version}
                for backend_id, version in self.backend_versions
            ],
            "artifacts": [item.as_dict() for item in self.artifacts],
            "last_frame_id": self.last_frame_id,
            "last_keyframe_id": self.last_keyframe_id,
            "resume_allowed": self.resume_allowed,
            "schema_version": self.schema_version,
            "minimum_reader_version": self.minimum_reader_version,
            "coordinate_convention_version": self.coordinate_convention_version,
            "created_at_utc": self.created_at_utc,
        }


@dataclass(frozen=True, slots=True)
class ResumeCompatibilityResult:
    status: ResumeCompatibilityStatus
    reasons: tuple[str, ...] = ()

    @property
    def compatible(self) -> bool:
        return self.status is ResumeCompatibilityStatus.COMPATIBLE

    def require_compatible(self) -> None:
        if not self.compatible:
            raise IncompatibleCheckpointError("; ".join(self.reasons))


def check_resume_compatibility(
    checkpoint: CheckpointMetadata,
    *,
    session_id: str,
    source_fingerprint: str,
    available_backend_versions: dict[str, str],
    coordinate_convention_version: int = COORDINATE_CONVENTION_VERSION,
) -> ResumeCompatibilityResult:
    reasons: list[str] = []
    if not checkpoint.resume_allowed:
        reasons.append("Checkpoint is not marked resumable.")
    if checkpoint.session_id != require_uuid(session_id, field_name="session_id"):
        reasons.append("Session identity differs from the checkpoint.")
    if checkpoint.source_fingerprint != str(source_fingerprint).strip().lower():
        reasons.append("Input source fingerprint differs from the checkpoint.")
    if checkpoint.coordinate_convention_version != int(coordinate_convention_version):
        reasons.append("Coordinate convention version is incompatible.")
    for backend_id, expected_version in checkpoint.backend_versions:
        actual = available_backend_versions.get(backend_id)
        if actual is None:
            reasons.append(f"Required backend is unavailable: {backend_id}.")
        elif actual != expected_version:
            reasons.append(
                f"Backend version differs for {backend_id}: expected {expected_version}, got {actual}."
            )
    status = (
        ResumeCompatibilityStatus.COMPATIBLE
        if not reasons
        else ResumeCompatibilityStatus.INCOMPATIBLE
    )
    return ResumeCompatibilityResult(status, tuple(reasons))

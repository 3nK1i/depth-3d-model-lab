"""Backend-neutral reconstruction pipeline orchestration contracts for Phase 8.7F1.

This module deliberately contains no real backend integration.  It defines the
stage graph, provider resolution/fallback policy, generic artifact authority,
and deterministic orchestration state transitions that later F2-F5 steps bind
to the already-qualified reconstruction components.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from core.contracts.validation import (
    JsonValue,
    metadata_as_dict,
    normalize_metadata,
    require_nonempty,
)
from core.reconstruction.contracts import ReconstructionPath
from core.reconstruction.control import ReconstructionOperationControl
from core.reconstruction.errors import ReconstructionCancelledError, ReconstructionValidationError
from core.reconstruction.profiles import ReconstructionQualityProfile
from core.reconstruction.provenance import BackendProvenance
from core.reconstruction.validation import require_nonnegative_int, require_uuid

RECONSTRUCTION_ORCHESTRATION_SCHEMA_VERSION = 1


class ReconstructionPipelineGoal(str, Enum):
    """High-level outputs requested from the future integrated pipeline."""

    CANONICAL_GEOMETRY = "canonical_geometry"
    SEMANTIC_INSPECTION = "semantic_inspection"
    GAUSSIAN_APPEARANCE = "gaussian_appearance"
    NERF_APPEARANCE = "nerf_appearance"


class ReconstructionPipelineStage(str, Enum):
    SOURCE_INGESTION = "source_ingestion"
    CAMERA_POSE = "camera_pose"
    DEPTH_EVIDENCE = "depth_evidence"
    DENSE_GEOMETRY = "dense_geometry"
    SURFACE_MESH = "surface_mesh"
    SEMANTIC_GEOMETRY = "semantic_geometry"
    INSPECTION_QC = "inspection_qc"
    GAUSSIAN_APPEARANCE = "gaussian_appearance"
    NERF_APPEARANCE = "nerf_appearance"


class ReconstructionProviderRole(str, Enum):
    INGESTION = "ingestion"
    CAMERA_POSE = "camera_pose"
    DEPTH = "depth"
    DENSE_GEOMETRY = "dense_geometry"
    SURFACE_MESH = "surface_mesh"
    SEMANTICS = "semantics"
    INSPECTION_QC = "inspection_qc"
    GAUSSIAN_APPEARANCE = "gaussian_appearance"
    NERF_APPEARANCE = "nerf_appearance"


class ReconstructionProviderExecution(str, Enum):
    IN_PROCESS = "in_process"
    EXTERNAL_PROCESS = "external_process"
    DEVELOPMENT_MOCK = "development_mock"


class ReconstructionArtifactKind(str, Enum):
    INGESTION_RESULT = "ingestion_result"
    CAMERA_TRAJECTORY = "camera_trajectory"
    DEPTH_EVIDENCE = "depth_evidence"
    DENSE_GEOMETRY = "dense_geometry"
    SURFACE_MESH = "surface_mesh"
    SEMANTIC_GEOMETRY = "semantic_geometry"
    INSPECTION_REPORT = "inspection_report"
    GAUSSIAN_SCENE = "gaussian_scene"
    NERF_SCENE = "nerf_scene"


class ReconstructionArtifactAuthority(str, Enum):
    INPUT_EVIDENCE = "input_evidence"
    SUPPORTING_GEOMETRY = "supporting_geometry"
    CANONICAL_GEOMETRY = "canonical_geometry"
    SEMANTIC_CONTEXT = "semantic_context"
    INSPECTION_DECISION = "inspection_decision"
    APPEARANCE_ONLY = "appearance_only"


class ReconstructionStageRunState(str, Enum):
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ReconstructionPipelineRunStatus(str, Enum):
    COMPLETED = "completed"
    DEGRADED = "degraded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ReconstructionPipelineExecutionScope(str, Enum):
    NO_PROVIDER_EXECUTION = "no_provider_execution"
    REAL_PROVIDERS = "real_providers"
    DEVELOPMENT_MOCK = "development_mock"
    MIXED = "mixed"


@dataclass(frozen=True, slots=True)
class ReconstructionStageContract:
    stage: ReconstructionPipelineStage
    provider_role: ReconstructionProviderRole
    output_kind: ReconstructionArtifactKind
    output_authority: ReconstructionArtifactAuthority
    required_inputs: tuple[ReconstructionArtifactKind, ...] = ()
    optional_inputs: tuple[ReconstructionArtifactKind, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.stage, ReconstructionPipelineStage):
            raise TypeError("stage must be ReconstructionPipelineStage.")
        if not isinstance(self.provider_role, ReconstructionProviderRole):
            raise TypeError("provider_role must be ReconstructionProviderRole.")
        if not isinstance(self.output_kind, ReconstructionArtifactKind):
            raise TypeError("output_kind must be ReconstructionArtifactKind.")
        if not isinstance(self.output_authority, ReconstructionArtifactAuthority):
            raise TypeError("output_authority must be ReconstructionArtifactAuthority.")
        required = tuple(self.required_inputs)
        optional = tuple(self.optional_inputs)
        if any(not isinstance(value, ReconstructionArtifactKind) for value in required + optional):
            raise TypeError("stage inputs must contain ReconstructionArtifactKind values.")
        if len(set(required)) != len(required) or len(set(optional)) != len(optional):
            raise ReconstructionValidationError("Stage input kinds must be unique.")
        if set(required).intersection(optional):
            raise ReconstructionValidationError("Required and optional stage inputs must not overlap.")
        object.__setattr__(self, "required_inputs", required)
        object.__setattr__(self, "optional_inputs", optional)

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "provider_role": self.provider_role.value,
            "output_kind": self.output_kind.value,
            "output_authority": self.output_authority.value,
            "required_inputs": [item.value for item in self.required_inputs],
            "optional_inputs": [item.value for item in self.optional_inputs],
        }


STAGE_CONTRACTS: tuple[ReconstructionStageContract, ...] = (
    ReconstructionStageContract(
        ReconstructionPipelineStage.SOURCE_INGESTION,
        ReconstructionProviderRole.INGESTION,
        ReconstructionArtifactKind.INGESTION_RESULT,
        ReconstructionArtifactAuthority.INPUT_EVIDENCE,
    ),
    ReconstructionStageContract(
        ReconstructionPipelineStage.CAMERA_POSE,
        ReconstructionProviderRole.CAMERA_POSE,
        ReconstructionArtifactKind.CAMERA_TRAJECTORY,
        ReconstructionArtifactAuthority.SUPPORTING_GEOMETRY,
        required_inputs=(ReconstructionArtifactKind.INGESTION_RESULT,),
    ),
    ReconstructionStageContract(
        ReconstructionPipelineStage.DEPTH_EVIDENCE,
        ReconstructionProviderRole.DEPTH,
        ReconstructionArtifactKind.DEPTH_EVIDENCE,
        ReconstructionArtifactAuthority.SUPPORTING_GEOMETRY,
        required_inputs=(ReconstructionArtifactKind.INGESTION_RESULT,),
        optional_inputs=(ReconstructionArtifactKind.CAMERA_TRAJECTORY,),
    ),
    ReconstructionStageContract(
        ReconstructionPipelineStage.DENSE_GEOMETRY,
        ReconstructionProviderRole.DENSE_GEOMETRY,
        ReconstructionArtifactKind.DENSE_GEOMETRY,
        ReconstructionArtifactAuthority.CANONICAL_GEOMETRY,
        required_inputs=(
            ReconstructionArtifactKind.CAMERA_TRAJECTORY,
            ReconstructionArtifactKind.DEPTH_EVIDENCE,
        ),
    ),
    ReconstructionStageContract(
        ReconstructionPipelineStage.SURFACE_MESH,
        ReconstructionProviderRole.SURFACE_MESH,
        ReconstructionArtifactKind.SURFACE_MESH,
        ReconstructionArtifactAuthority.CANONICAL_GEOMETRY,
        required_inputs=(ReconstructionArtifactKind.DENSE_GEOMETRY,),
    ),
    ReconstructionStageContract(
        ReconstructionPipelineStage.SEMANTIC_GEOMETRY,
        ReconstructionProviderRole.SEMANTICS,
        ReconstructionArtifactKind.SEMANTIC_GEOMETRY,
        ReconstructionArtifactAuthority.SEMANTIC_CONTEXT,
        required_inputs=(ReconstructionArtifactKind.SURFACE_MESH,),
    ),
    ReconstructionStageContract(
        ReconstructionPipelineStage.INSPECTION_QC,
        ReconstructionProviderRole.INSPECTION_QC,
        ReconstructionArtifactKind.INSPECTION_REPORT,
        ReconstructionArtifactAuthority.INSPECTION_DECISION,
        required_inputs=(
            ReconstructionArtifactKind.SURFACE_MESH,
            ReconstructionArtifactKind.SEMANTIC_GEOMETRY,
        ),
    ),
    ReconstructionStageContract(
        ReconstructionPipelineStage.GAUSSIAN_APPEARANCE,
        ReconstructionProviderRole.GAUSSIAN_APPEARANCE,
        ReconstructionArtifactKind.GAUSSIAN_SCENE,
        ReconstructionArtifactAuthority.APPEARANCE_ONLY,
        required_inputs=(
            ReconstructionArtifactKind.INGESTION_RESULT,
            ReconstructionArtifactKind.CAMERA_TRAJECTORY,
            ReconstructionArtifactKind.SURFACE_MESH,
        ),
    ),
    ReconstructionStageContract(
        ReconstructionPipelineStage.NERF_APPEARANCE,
        ReconstructionProviderRole.NERF_APPEARANCE,
        ReconstructionArtifactKind.NERF_SCENE,
        ReconstructionArtifactAuthority.APPEARANCE_ONLY,
        required_inputs=(
            ReconstructionArtifactKind.INGESTION_RESULT,
            ReconstructionArtifactKind.CAMERA_TRAJECTORY,
        ),
    ),
)

_STAGE_BY_STAGE = {contract.stage: contract for contract in STAGE_CONTRACTS}
_STAGE_BY_OUTPUT = {contract.output_kind: contract for contract in STAGE_CONTRACTS}

if len(_STAGE_BY_STAGE) != len(STAGE_CONTRACTS) or len(_STAGE_BY_OUTPUT) != len(STAGE_CONTRACTS):
    raise RuntimeError("Reconstruction orchestration stage contracts must be one-to-one.")

_seen_outputs: set[ReconstructionArtifactKind] = set()
for _contract in STAGE_CONTRACTS:
    if not set(_contract.required_inputs).issubset(_seen_outputs):
        raise RuntimeError(
            f"Stage {_contract.stage.value!r} has a required input that is not produced earlier in the DAG."
        )
    if not set(_contract.optional_inputs).issubset(_seen_outputs):
        raise RuntimeError(
            f"Stage {_contract.stage.value!r} has an optional input that is not produced earlier in the DAG."
        )
    _seen_outputs.add(_contract.output_kind)


_GOAL_STAGE_REQUIREMENTS: dict[ReconstructionPipelineGoal, tuple[ReconstructionPipelineStage, ...]] = {
    ReconstructionPipelineGoal.CANONICAL_GEOMETRY: (
        ReconstructionPipelineStage.SOURCE_INGESTION,
        ReconstructionPipelineStage.CAMERA_POSE,
        ReconstructionPipelineStage.DEPTH_EVIDENCE,
        ReconstructionPipelineStage.DENSE_GEOMETRY,
        ReconstructionPipelineStage.SURFACE_MESH,
    ),
    ReconstructionPipelineGoal.SEMANTIC_INSPECTION: (
        ReconstructionPipelineStage.SOURCE_INGESTION,
        ReconstructionPipelineStage.CAMERA_POSE,
        ReconstructionPipelineStage.DEPTH_EVIDENCE,
        ReconstructionPipelineStage.DENSE_GEOMETRY,
        ReconstructionPipelineStage.SURFACE_MESH,
        ReconstructionPipelineStage.SEMANTIC_GEOMETRY,
        ReconstructionPipelineStage.INSPECTION_QC,
    ),
    ReconstructionPipelineGoal.GAUSSIAN_APPEARANCE: (
        ReconstructionPipelineStage.SOURCE_INGESTION,
        ReconstructionPipelineStage.CAMERA_POSE,
        ReconstructionPipelineStage.DEPTH_EVIDENCE,
        ReconstructionPipelineStage.DENSE_GEOMETRY,
        ReconstructionPipelineStage.SURFACE_MESH,
        ReconstructionPipelineStage.GAUSSIAN_APPEARANCE,
    ),
    ReconstructionPipelineGoal.NERF_APPEARANCE: (
        ReconstructionPipelineStage.SOURCE_INGESTION,
        ReconstructionPipelineStage.CAMERA_POSE,
        ReconstructionPipelineStage.NERF_APPEARANCE,
    ),
}

_OPTIONAL_NONAUTHORITATIVE_STAGES = {
    ReconstructionPipelineStage.GAUSSIAN_APPEARANCE,
    ReconstructionPipelineStage.NERF_APPEARANCE,
}


@dataclass(frozen=True, slots=True)
class ReconstructionPipelineRequest:
    request_id: str
    session_id: str
    path: ReconstructionPath
    quality_profile: ReconstructionQualityProfile
    goals: tuple[ReconstructionPipelineGoal, ...]
    allow_optional_degradation: bool = True
    metadata: Mapping[str, JsonValue] | Sequence[tuple[str, JsonValue]] = field(default_factory=tuple)
    schema_version: int = RECONSTRUCTION_ORCHESTRATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", require_nonempty(self.request_id, field_name="request_id"))
        object.__setattr__(self, "session_id", require_uuid(self.session_id, field_name="session_id"))
        if not isinstance(self.path, ReconstructionPath):
            raise TypeError("path must be ReconstructionPath.")
        if not isinstance(self.quality_profile, ReconstructionQualityProfile):
            raise TypeError("quality_profile must be ReconstructionQualityProfile.")
        goals = tuple(self.goals)
        if not goals or any(not isinstance(goal, ReconstructionPipelineGoal) for goal in goals):
            raise ReconstructionValidationError("goals must contain at least one ReconstructionPipelineGoal.")
        if len(set(goals)) != len(goals):
            raise ReconstructionValidationError("Pipeline goals must be unique.")
        if not isinstance(self.allow_optional_degradation, bool):
            raise TypeError("allow_optional_degradation must be a bool.")
        if int(self.schema_version) != RECONSTRUCTION_ORCHESTRATION_SCHEMA_VERSION:
            raise ReconstructionValidationError("Unsupported reconstruction orchestration schema version.")
        object.__setattr__(self, "goals", goals)
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata))
        object.__setattr__(self, "schema_version", RECONSTRUCTION_ORCHESTRATION_SCHEMA_VERSION)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "session_id": self.session_id,
            "path": self.path.value,
            "quality_profile": self.quality_profile.value,
            "goals": [goal.value for goal in self.goals],
            "allow_optional_degradation": self.allow_optional_degradation,
            "metadata": metadata_as_dict(self.metadata),  # type: ignore[arg-type]
        }


@dataclass(frozen=True, slots=True)
class ReconstructionProviderDescriptor:
    provider_id: str
    provider_version: str
    roles: tuple[ReconstructionProviderRole, ...]
    execution: ReconstructionProviderExecution
    priority: int = 100
    available: bool = True
    unavailable_reason: str = ""
    adapter_version: str = ""
    metadata: Mapping[str, JsonValue] | Sequence[tuple[str, JsonValue]] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider_id", require_nonempty(self.provider_id, field_name="provider_id"))
        object.__setattr__(
            self,
            "provider_version",
            require_nonempty(self.provider_version, field_name="provider_version"),
        )
        roles = tuple(self.roles)
        if not roles or any(not isinstance(role, ReconstructionProviderRole) for role in roles):
            raise ReconstructionValidationError("Provider roles must contain at least one provider role.")
        if len(set(roles)) != len(roles):
            raise ReconstructionValidationError("Provider roles must be unique.")
        if not isinstance(self.execution, ReconstructionProviderExecution):
            raise TypeError("execution must be ReconstructionProviderExecution.")
        object.__setattr__(self, "priority", require_nonnegative_int(self.priority, field_name="priority"))
        if not isinstance(self.available, bool):
            raise TypeError("available must be a bool.")
        reason = str(self.unavailable_reason).strip()
        if self.available and reason:
            raise ReconstructionValidationError("Available providers must not carry unavailable_reason.")
        if not self.available and not reason:
            raise ReconstructionValidationError("Unavailable providers require unavailable_reason.")
        object.__setattr__(self, "roles", roles)
        object.__setattr__(self, "unavailable_reason", reason)
        object.__setattr__(self, "adapter_version", str(self.adapter_version).strip())
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata))

    def supports(self, role: ReconstructionProviderRole) -> bool:
        return role in self.roles

    def as_backend_provenance(self) -> BackendProvenance:
        return BackendProvenance(
            backend_id=self.provider_id,
            backend_version=self.provider_version,
            adapter_version=self.adapter_version,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "provider_version": self.provider_version,
            "adapter_version": self.adapter_version,
            "roles": [role.value for role in self.roles],
            "execution": self.execution.value,
            "priority": self.priority,
            "available": self.available,
            "unavailable_reason": self.unavailable_reason,
            "metadata": metadata_as_dict(self.metadata),  # type: ignore[arg-type]
        }


@dataclass(frozen=True, slots=True)
class ReconstructionPlannedStage:
    contract: ReconstructionStageContract
    required: bool
    preferred_provider_id: str
    candidate_provider_ids: tuple[str, ...]
    unavailable_provider_ids: tuple[str, ...] = ()
    skip_reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.contract, ReconstructionStageContract):
            raise TypeError("contract must be ReconstructionStageContract.")
        if not isinstance(self.required, bool):
            raise TypeError("required must be a bool.")
        preferred = str(self.preferred_provider_id).strip()
        candidates = tuple(str(value).strip() for value in self.candidate_provider_ids)
        unavailable = tuple(str(value).strip() for value in self.unavailable_provider_ids)
        if any(not value for value in candidates + unavailable):
            raise ReconstructionValidationError("Provider IDs must not be empty.")
        if len(set(candidates)) != len(candidates) or len(set(unavailable)) != len(unavailable):
            raise ReconstructionValidationError("Planned provider IDs must be unique.")
        if set(candidates).intersection(unavailable):
            raise ReconstructionValidationError("Available and unavailable provider IDs must not overlap.")
        if preferred and preferred not in candidates and preferred not in unavailable:
            raise ReconstructionValidationError("preferred_provider_id must appear in the provider ordering.")
        reason = str(self.skip_reason).strip()
        if self.required and not candidates:
            raise ReconstructionValidationError("Required stages need at least one available provider candidate.")
        if not candidates and not reason:
            raise ReconstructionValidationError("Skipped stages require skip_reason.")
        object.__setattr__(self, "preferred_provider_id", preferred)
        object.__setattr__(self, "candidate_provider_ids", candidates)
        object.__setattr__(self, "unavailable_provider_ids", unavailable)
        object.__setattr__(self, "skip_reason", reason)

    @property
    def stage(self) -> ReconstructionPipelineStage:
        return self.contract.stage

    @property
    def fallback_planned(self) -> bool:
        return bool(
            self.preferred_provider_id
            and self.candidate_provider_ids
            and self.candidate_provider_ids[0] != self.preferred_provider_id
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "required": self.required,
            "preferred_provider_id": self.preferred_provider_id,
            "candidate_provider_ids": list(self.candidate_provider_ids),
            "unavailable_provider_ids": list(self.unavailable_provider_ids),
            "fallback_planned": self.fallback_planned,
            "skip_reason": self.skip_reason,
            "contract": self.contract.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class ReconstructionPipelinePlan:
    request: ReconstructionPipelineRequest
    stages: tuple[ReconstructionPlannedStage, ...]
    provider_ids: tuple[str, ...]
    schema_version: int = RECONSTRUCTION_ORCHESTRATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.request, ReconstructionPipelineRequest):
            raise TypeError("request must be ReconstructionPipelineRequest.")
        stages = tuple(self.stages)
        if not stages or any(not isinstance(stage, ReconstructionPlannedStage) for stage in stages):
            raise ReconstructionValidationError("Pipeline plan requires planned stages.")
        expected_order = [contract.stage for contract in STAGE_CONTRACTS]
        actual_positions = [expected_order.index(stage.stage) for stage in stages]
        if actual_positions != sorted(actual_positions) or len(set(actual_positions)) != len(actual_positions):
            raise ReconstructionValidationError("Pipeline stages must follow the canonical DAG order exactly once.")
        provider_ids = tuple(str(value).strip() for value in self.provider_ids)
        if len(set(provider_ids)) != len(provider_ids) or any(not value for value in provider_ids):
            raise ReconstructionValidationError("Pipeline provider_ids must be unique and non-empty.")
        referenced = {
            provider_id
            for stage in stages
            for provider_id in (stage.candidate_provider_ids + stage.unavailable_provider_ids)
        }
        if not referenced.issubset(set(provider_ids)):
            raise ReconstructionValidationError("Plan references providers outside provider_ids.")
        if int(self.schema_version) != RECONSTRUCTION_ORCHESTRATION_SCHEMA_VERSION:
            raise ReconstructionValidationError("Unsupported reconstruction orchestration schema version.")
        object.__setattr__(self, "stages", stages)
        object.__setattr__(self, "provider_ids", provider_ids)
        object.__setattr__(self, "schema_version", RECONSTRUCTION_ORCHESTRATION_SCHEMA_VERSION)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request": self.request.as_dict(),
            "provider_ids": list(self.provider_ids),
            "stages": [stage.as_dict() for stage in self.stages],
        }


@dataclass(frozen=True, slots=True)
class ReconstructionPipelineArtifact:
    artifact_id: str
    session_id: str
    stage: ReconstructionPipelineStage
    kind: ReconstructionArtifactKind
    authority: ReconstructionArtifactAuthority
    provider_id: str
    payload_ref: str = ""
    metadata: Mapping[str, JsonValue] | Sequence[tuple[str, JsonValue]] = field(default_factory=tuple)
    schema_version: int = RECONSTRUCTION_ORCHESTRATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_id", require_nonempty(self.artifact_id, field_name="artifact_id"))
        object.__setattr__(self, "session_id", require_uuid(self.session_id, field_name="session_id"))
        if not isinstance(self.stage, ReconstructionPipelineStage):
            raise TypeError("stage must be ReconstructionPipelineStage.")
        if not isinstance(self.kind, ReconstructionArtifactKind):
            raise TypeError("kind must be ReconstructionArtifactKind.")
        if not isinstance(self.authority, ReconstructionArtifactAuthority):
            raise TypeError("authority must be ReconstructionArtifactAuthority.")
        contract = _STAGE_BY_STAGE[self.stage]
        if self.kind is not contract.output_kind or self.authority is not contract.output_authority:
            raise ReconstructionValidationError("Artifact kind/authority must match its canonical stage contract.")
        if self.kind in {ReconstructionArtifactKind.GAUSSIAN_SCENE, ReconstructionArtifactKind.NERF_SCENE}:
            if self.authority is not ReconstructionArtifactAuthority.APPEARANCE_ONLY:
                raise ReconstructionValidationError("Neural appearance artifacts must remain appearance-only.")
        object.__setattr__(self, "provider_id", require_nonempty(self.provider_id, field_name="provider_id"))
        object.__setattr__(self, "payload_ref", str(self.payload_ref).strip())
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata))
        if int(self.schema_version) != RECONSTRUCTION_ORCHESTRATION_SCHEMA_VERSION:
            raise ReconstructionValidationError("Unsupported reconstruction orchestration schema version.")
        object.__setattr__(self, "schema_version", RECONSTRUCTION_ORCHESTRATION_SCHEMA_VERSION)

    @property
    def canonical_geometry_authoritative(self) -> bool:
        return self.authority is ReconstructionArtifactAuthority.CANONICAL_GEOMETRY

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "session_id": self.session_id,
            "stage": self.stage.value,
            "kind": self.kind.value,
            "authority": self.authority.value,
            "provider_id": self.provider_id,
            "payload_ref": self.payload_ref,
            "canonical_geometry_authoritative": self.canonical_geometry_authoritative,
            "certified_metrology": False,
            "metadata": metadata_as_dict(self.metadata),  # type: ignore[arg-type]
        }


@dataclass(frozen=True, slots=True)
class ReconstructionStageExecutionContext:
    request: ReconstructionPipelineRequest
    planned_stage: ReconstructionPlannedStage
    provider: ReconstructionProviderDescriptor
    required_inputs: tuple[ReconstructionPipelineArtifact, ...]
    optional_inputs: tuple[ReconstructionPipelineArtifact, ...]
    control: ReconstructionOperationControl | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.request, ReconstructionPipelineRequest):
            raise TypeError("request must be ReconstructionPipelineRequest.")
        if not isinstance(self.planned_stage, ReconstructionPlannedStage):
            raise TypeError("planned_stage must be ReconstructionPlannedStage.")
        if not isinstance(self.provider, ReconstructionProviderDescriptor):
            raise TypeError("provider must be ReconstructionProviderDescriptor.")
        if not self.provider.supports(self.planned_stage.contract.provider_role):
            raise ReconstructionValidationError("Provider does not support the planned stage role.")
        for name in ("required_inputs", "optional_inputs"):
            values = tuple(getattr(self, name))
            if any(not isinstance(value, ReconstructionPipelineArtifact) for value in values):
                raise TypeError(f"{name} must contain ReconstructionPipelineArtifact values.")
            object.__setattr__(self, name, values)
        if self.control is not None and not isinstance(self.control, ReconstructionOperationControl):
            raise TypeError("control must be ReconstructionOperationControl or None.")

    def checkpoint(self) -> None:
        """Cooperative whole-engine cancellation poll for long-running providers."""
        if self.control is not None:
            self.control.checkpoint()


class ReconstructionStageExecutor(Protocol):
    def execute(self, context: ReconstructionStageExecutionContext) -> ReconstructionPipelineArtifact:
        """Execute exactly one provider-bound orchestration stage."""


@dataclass(frozen=True, slots=True)
class ReconstructionStageRunResult:
    stage: ReconstructionPipelineStage
    state: ReconstructionStageRunState
    required: bool
    provider_attempts: tuple[str, ...]
    provider_id: str = ""
    artifact: ReconstructionPipelineArtifact | None = None
    fallback_used: bool = False
    reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.stage, ReconstructionPipelineStage):
            raise TypeError("stage must be ReconstructionPipelineStage.")
        if not isinstance(self.state, ReconstructionStageRunState):
            raise TypeError("state must be ReconstructionStageRunState.")
        if not isinstance(self.required, bool) or not isinstance(self.fallback_used, bool):
            raise TypeError("required and fallback_used must be bool values.")
        attempts = tuple(str(value).strip() for value in self.provider_attempts)
        if any(not value for value in attempts):
            raise ReconstructionValidationError("provider_attempts must not contain empty IDs.")
        provider_id = str(self.provider_id).strip()
        reason = str(self.reason).strip()
        if self.state is ReconstructionStageRunState.COMPLETED:
            if self.artifact is None or not provider_id:
                raise ReconstructionValidationError("Completed stages require provider_id and artifact.")
            if self.artifact.stage is not self.stage or self.artifact.provider_id != provider_id:
                raise ReconstructionValidationError("Completed stage artifact/provider identity mismatch.")
        elif self.artifact is not None:
            raise ReconstructionValidationError("Non-completed stages must not carry an artifact.")
        if self.state is not ReconstructionStageRunState.COMPLETED and not reason:
            raise ReconstructionValidationError("Non-completed stages require a reason.")
        object.__setattr__(self, "provider_attempts", attempts)
        object.__setattr__(self, "provider_id", provider_id)
        object.__setattr__(self, "reason", reason)

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "state": self.state.value,
            "required": self.required,
            "provider_attempts": list(self.provider_attempts),
            "provider_id": self.provider_id,
            "fallback_used": self.fallback_used,
            "reason": self.reason,
            "artifact": self.artifact.as_dict() if self.artifact else None,
        }


@dataclass(frozen=True, slots=True)
class ReconstructionProviderUse:
    provider_id: str
    provider_version: str
    execution: ReconstructionProviderExecution
    stages: tuple[ReconstructionPipelineStage, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider_id", require_nonempty(self.provider_id, field_name="provider_id"))
        object.__setattr__(
            self,
            "provider_version",
            require_nonempty(self.provider_version, field_name="provider_version"),
        )
        if not isinstance(self.execution, ReconstructionProviderExecution):
            raise TypeError("execution must be ReconstructionProviderExecution.")
        stages = tuple(self.stages)
        if not stages or any(not isinstance(stage, ReconstructionPipelineStage) for stage in stages):
            raise ReconstructionValidationError("Provider use requires at least one pipeline stage.")
        if len(set(stages)) != len(stages):
            raise ReconstructionValidationError("Provider-use stages must be unique.")
        object.__setattr__(self, "stages", stages)

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "provider_version": self.provider_version,
            "execution": self.execution.value,
            "stages": [stage.value for stage in self.stages],
        }


@dataclass(frozen=True, slots=True)
class ReconstructionPipelineRunResult:
    request_id: str
    session_id: str
    status: ReconstructionPipelineRunStatus
    execution_scope: ReconstructionPipelineExecutionScope
    stages: tuple[ReconstructionStageRunResult, ...]
    provider_provenance: tuple[ReconstructionProviderUse, ...]
    canonical_geometry_artifact_id: str = ""
    development_mock_used: bool = False
    certified_metrology: bool = False
    reconstruction_ui_enabled: bool = False
    schema_version: int = RECONSTRUCTION_ORCHESTRATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", require_nonempty(self.request_id, field_name="request_id"))
        object.__setattr__(self, "session_id", require_uuid(self.session_id, field_name="session_id"))
        if not isinstance(self.status, ReconstructionPipelineRunStatus):
            raise TypeError("status must be ReconstructionPipelineRunStatus.")
        if not isinstance(self.execution_scope, ReconstructionPipelineExecutionScope):
            raise TypeError("execution_scope must be ReconstructionPipelineExecutionScope.")
        stages = tuple(self.stages)
        provenance = tuple(self.provider_provenance)
        if any(not isinstance(value, ReconstructionStageRunResult) for value in stages):
            raise TypeError("stages must contain ReconstructionStageRunResult values.")
        if any(not isinstance(value, ReconstructionProviderUse) for value in provenance):
            raise TypeError("provider_provenance must contain ReconstructionProviderUse values.")
        if len({item.provider_id for item in provenance}) != len(provenance):
            raise ReconstructionValidationError("Provider provenance IDs must be unique.")
        if bool(self.certified_metrology):
            raise ReconstructionValidationError("Phase 8.7F1 orchestration must not claim certified metrology.")
        if bool(self.reconstruction_ui_enabled):
            raise ReconstructionValidationError("Reconstruction UI remains disabled until Phase 8.8.")
        mock_used = any(item.execution is ReconstructionProviderExecution.DEVELOPMENT_MOCK for item in provenance)
        if bool(self.development_mock_used) != mock_used:
            raise ReconstructionValidationError("development_mock_used must agree with provider provenance.")
        canonical_id = str(self.canonical_geometry_artifact_id).strip()
        canonical_artifacts = [
            stage.artifact
            for stage in stages
            if stage.artifact is not None and stage.artifact.canonical_geometry_authoritative
        ]
        if canonical_artifacts:
            expected = canonical_artifacts[-1].artifact_id
            if canonical_id != expected:
                raise ReconstructionValidationError("canonical_geometry_artifact_id must point to the latest canonical geometry artifact.")
        elif canonical_id:
            raise ReconstructionValidationError("canonical_geometry_artifact_id cannot exist without canonical geometry.")
        if int(self.schema_version) != RECONSTRUCTION_ORCHESTRATION_SCHEMA_VERSION:
            raise ReconstructionValidationError("Unsupported reconstruction orchestration schema version.")
        object.__setattr__(self, "stages", stages)
        object.__setattr__(self, "provider_provenance", provenance)
        object.__setattr__(self, "canonical_geometry_artifact_id", canonical_id)
        object.__setattr__(self, "development_mock_used", mock_used)
        object.__setattr__(self, "certified_metrology", False)
        object.__setattr__(self, "reconstruction_ui_enabled", False)
        object.__setattr__(self, "schema_version", RECONSTRUCTION_ORCHESTRATION_SCHEMA_VERSION)

    @property
    def canonical_geometry_authoritative(self) -> bool:
        return bool(self.canonical_geometry_artifact_id)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "session_id": self.session_id,
            "status": self.status.value,
            "execution_scope": self.execution_scope.value,
            "stages": [stage.as_dict() for stage in self.stages],
            "provider_provenance": [item.as_dict() for item in self.provider_provenance],
            "canonical_geometry_artifact_id": self.canonical_geometry_artifact_id,
            "canonical_geometry_authoritative": self.canonical_geometry_authoritative,
            "development_mock_used": self.development_mock_used,
            "production_execution_claimed": (
                self.execution_scope is ReconstructionPipelineExecutionScope.REAL_PROVIDERS
                and bool(self.provider_provenance)
            ),
            "certified_metrology": self.certified_metrology,
            "reconstruction_ui_enabled": self.reconstruction_ui_enabled,
            "future_phase_ownership": {
                "depth_anything_3_provider": "8.7F2",
                "canonical_geometry_semantics_qc_binding": "8.7F3",
                "gaussian_nerf_appearance_binding": "8.7F4",
                "persistent_reconstruction_scene": "8.7F5",
                "whole_engine_hardening": "8.7G",
                "ui": "8.8",
            },
        }


def reconstruction_stage_contract(stage: ReconstructionPipelineStage) -> ReconstructionStageContract:
    if not isinstance(stage, ReconstructionPipelineStage):
        raise TypeError("stage must be ReconstructionPipelineStage.")
    return _STAGE_BY_STAGE[stage]


def required_pipeline_stages(goals: Sequence[ReconstructionPipelineGoal]) -> tuple[ReconstructionPipelineStage, ...]:
    normalized = tuple(goals)
    if not normalized or any(not isinstance(goal, ReconstructionPipelineGoal) for goal in normalized):
        raise ReconstructionValidationError("goals must contain ReconstructionPipelineGoal values.")
    included: set[ReconstructionPipelineStage] = set()
    for goal in normalized:
        included.update(_GOAL_STAGE_REQUIREMENTS[goal])
    return tuple(contract.stage for contract in STAGE_CONTRACTS if contract.stage in included)


def build_reconstruction_pipeline_plan(
    request: ReconstructionPipelineRequest,
    providers: Sequence[ReconstructionProviderDescriptor],
) -> ReconstructionPipelinePlan:
    """Resolve one deterministic provider/fallback plan without executing any backend."""

    if not isinstance(request, ReconstructionPipelineRequest):
        raise TypeError("request must be ReconstructionPipelineRequest.")
    provider_values = tuple(providers)
    if not provider_values or any(not isinstance(value, ReconstructionProviderDescriptor) for value in provider_values):
        raise ReconstructionValidationError("providers must contain at least one provider descriptor.")
    if len({provider.provider_id for provider in provider_values}) != len(provider_values):
        raise ReconstructionValidationError("Provider IDs must be globally unique.")

    selected_stages = set(required_pipeline_stages(request.goals))
    planned: list[ReconstructionPlannedStage] = []
    for contract in STAGE_CONTRACTS:
        if contract.stage not in selected_stages:
            continue
        stage_providers = sorted(
            (provider for provider in provider_values if provider.supports(contract.provider_role)),
            key=lambda provider: (provider.priority, provider.provider_id),
        )
        preferred = stage_providers[0].provider_id if stage_providers else ""
        candidates = tuple(provider.provider_id for provider in stage_providers if provider.available)
        unavailable = tuple(provider.provider_id for provider in stage_providers if not provider.available)
        required = contract.stage not in _OPTIONAL_NONAUTHORITATIVE_STAGES or not request.allow_optional_degradation
        if required and not candidates:
            raise ReconstructionValidationError(
                f"Required stage {contract.stage.value!r} has no available provider."
            )
        skip_reason = ""
        if not candidates:
            skip_reason = "optional_provider_unavailable"
        planned.append(
            ReconstructionPlannedStage(
                contract=contract,
                required=required,
                preferred_provider_id=preferred,
                candidate_provider_ids=candidates,
                unavailable_provider_ids=unavailable,
                skip_reason=skip_reason,
            )
        )
    return ReconstructionPipelinePlan(
        request=request,
        stages=tuple(planned),
        provider_ids=tuple(sorted(provider.provider_id for provider in provider_values)),
    )


def _execution_scope(providers: Sequence[ReconstructionProviderUse]) -> ReconstructionPipelineExecutionScope:
    if not providers:
        return ReconstructionPipelineExecutionScope.NO_PROVIDER_EXECUTION
    executions = {provider.execution for provider in providers}
    if executions == {ReconstructionProviderExecution.DEVELOPMENT_MOCK}:
        return ReconstructionPipelineExecutionScope.DEVELOPMENT_MOCK
    if ReconstructionProviderExecution.DEVELOPMENT_MOCK in executions:
        return ReconstructionPipelineExecutionScope.MIXED
    return ReconstructionPipelineExecutionScope.REAL_PROVIDERS


def _aggregate_provider_uses(
    stage_results: Sequence[ReconstructionStageRunResult],
    providers_by_id: Mapping[str, ReconstructionProviderDescriptor],
) -> tuple[ReconstructionProviderUse, ...]:
    stages_by_provider: dict[str, list[ReconstructionPipelineStage]] = {}
    for result in stage_results:
        if result.state is ReconstructionStageRunState.COMPLETED:
            stages_by_provider.setdefault(result.provider_id, []).append(result.stage)
    uses: list[ReconstructionProviderUse] = []
    for provider_id in sorted(stages_by_provider):
        provider = providers_by_id[provider_id]
        uses.append(
            ReconstructionProviderUse(
                provider_id=provider.provider_id,
                provider_version=provider.provider_version,
                execution=provider.execution,
                stages=tuple(stages_by_provider[provider_id]),
            )
        )
    return tuple(uses)


def aggregate_pipeline_backend_provenance(
    result: ReconstructionPipelineRunResult,
    providers: Sequence[ReconstructionProviderDescriptor],
) -> tuple[BackendProvenance, ...]:
    """Return stable backend provenance only for providers that produced artifacts."""

    if not isinstance(result, ReconstructionPipelineRunResult):
        raise TypeError("result must be ReconstructionPipelineRunResult.")
    providers_by_id = {provider.provider_id: provider for provider in providers}
    used_ids = [item.provider_id for item in result.provider_provenance]
    missing = [provider_id for provider_id in used_ids if provider_id not in providers_by_id]
    if missing:
        raise ReconstructionValidationError(f"Provider provenance descriptors missing: {missing!r}.")
    return tuple(providers_by_id[provider_id].as_backend_provenance() for provider_id in used_ids)


class ReconstructionPipelineOrchestrator:
    """Dependency-light sequential DAG runner with deterministic provider fallback."""

    def __init__(
        self,
        *,
        plan: ReconstructionPipelinePlan,
        providers: Sequence[ReconstructionProviderDescriptor],
        executors: Mapping[str, ReconstructionStageExecutor],
    ) -> None:
        if not isinstance(plan, ReconstructionPipelinePlan):
            raise TypeError("plan must be ReconstructionPipelinePlan.")
        provider_values = tuple(providers)
        if len({provider.provider_id for provider in provider_values}) != len(provider_values):
            raise ReconstructionValidationError("Provider IDs must be globally unique.")
        providers_by_id = {provider.provider_id: provider for provider in provider_values}
        if set(plan.provider_ids) != set(providers_by_id):
            raise ReconstructionValidationError("Plan/provider descriptor identity mismatch.")
        self._plan = plan
        self._providers = providers_by_id
        self._executors = dict(executors)

    @property
    def plan(self) -> ReconstructionPipelinePlan:
        return self._plan

    def run(self, *, control: ReconstructionOperationControl | None = None) -> ReconstructionPipelineRunResult:
        control = control or ReconstructionOperationControl()
        artifacts_by_kind: dict[ReconstructionArtifactKind, ReconstructionPipelineArtifact] = {}
        stage_results: list[ReconstructionStageRunResult] = []
        degraded = False

        for planned in self._plan.stages:
            try:
                control.checkpoint()
            except ReconstructionCancelledError:
                stage_results.append(
                    ReconstructionStageRunResult(
                        stage=planned.stage,
                        state=ReconstructionStageRunState.CANCELLED,
                        required=planned.required,
                        provider_attempts=(),
                        reason="cancellation_requested",
                    )
                )
                return self._finalize(stage_results, ReconstructionPipelineRunStatus.CANCELLED)

            if not planned.candidate_provider_ids:
                degraded = True
                stage_results.append(
                    ReconstructionStageRunResult(
                        stage=planned.stage,
                        state=ReconstructionStageRunState.SKIPPED,
                        required=planned.required,
                        provider_attempts=(),
                        reason=planned.skip_reason,
                    )
                )
                continue

            required_inputs: list[ReconstructionPipelineArtifact] = []
            missing_required: list[ReconstructionArtifactKind] = []
            for kind in planned.contract.required_inputs:
                artifact = artifacts_by_kind.get(kind)
                if artifact is None:
                    missing_required.append(kind)
                else:
                    required_inputs.append(artifact)
            if missing_required:
                reason = "missing_required_inputs:" + ",".join(kind.value for kind in missing_required)
                state = ReconstructionStageRunState.FAILED if planned.required else ReconstructionStageRunState.SKIPPED
                stage_results.append(
                    ReconstructionStageRunResult(
                        stage=planned.stage,
                        state=state,
                        required=planned.required,
                        provider_attempts=(),
                        reason=reason,
                    )
                )
                if planned.required:
                    return self._finalize(stage_results, ReconstructionPipelineRunStatus.FAILED)
                degraded = True
                continue

            optional_inputs = tuple(
                artifact
                for kind in planned.contract.optional_inputs
                if (artifact := artifacts_by_kind.get(kind)) is not None
            )
            attempts: list[str] = []
            errors: list[str] = []
            completed_artifact: ReconstructionPipelineArtifact | None = None
            completed_provider_id = ""
            for candidate_index, provider_id in enumerate(planned.candidate_provider_ids):
                attempts.append(provider_id)
                provider = self._providers[provider_id]
                executor = self._executors.get(provider_id)
                if executor is None:
                    errors.append(f"{provider_id}:executor_missing")
                    continue
                context = ReconstructionStageExecutionContext(
                    request=self._plan.request,
                    planned_stage=planned,
                    provider=provider,
                    required_inputs=tuple(required_inputs),
                    optional_inputs=optional_inputs,
                    control=control,
                )
                try:
                    artifact = executor.execute(context)
                    # Providers may either poll context.checkpoint() while working,
                    # or cancellation may arrive during an uncooperative call. In
                    # both cases the engine must not accept a post-cancel artifact.
                    control.checkpoint()
                except ReconstructionCancelledError:
                    stage_results.append(
                        ReconstructionStageRunResult(
                            stage=planned.stage,
                            state=ReconstructionStageRunState.CANCELLED,
                            required=planned.required,
                            provider_attempts=tuple(attempts),
                            reason="provider_observed_cancellation",
                        )
                    )
                    return self._finalize(stage_results, ReconstructionPipelineRunStatus.CANCELLED)
                except Exception as exc:  # provider boundary: capture failure and try declared fallback
                    errors.append(f"{provider_id}:{type(exc).__name__}:{exc}")
                    continue
                if not isinstance(artifact, ReconstructionPipelineArtifact):
                    errors.append(f"{provider_id}:invalid_artifact_type")
                    continue
                if artifact.session_id != self._plan.request.session_id:
                    errors.append(f"{provider_id}:session_mismatch")
                    continue
                if artifact.stage is not planned.stage or artifact.provider_id != provider_id:
                    errors.append(f"{provider_id}:artifact_identity_mismatch")
                    continue
                completed_artifact = artifact
                completed_provider_id = provider_id
                if candidate_index > 0 or planned.fallback_planned:
                    degraded = True
                break

            if completed_artifact is None:
                reason = "provider_attempts_failed:" + "|".join(errors or ["no_executor_succeeded"])
                state = ReconstructionStageRunState.FAILED if planned.required else ReconstructionStageRunState.SKIPPED
                stage_results.append(
                    ReconstructionStageRunResult(
                        stage=planned.stage,
                        state=state,
                        required=planned.required,
                        provider_attempts=tuple(attempts),
                        reason=reason,
                    )
                )
                if planned.required:
                    return self._finalize(stage_results, ReconstructionPipelineRunStatus.FAILED)
                degraded = True
                continue

            artifacts_by_kind[completed_artifact.kind] = completed_artifact
            fallback_used = (
                len(attempts) > 1
                or planned.fallback_planned
                or (planned.preferred_provider_id and completed_provider_id != planned.preferred_provider_id)
            )
            stage_results.append(
                ReconstructionStageRunResult(
                    stage=planned.stage,
                    state=ReconstructionStageRunState.COMPLETED,
                    required=planned.required,
                    provider_attempts=tuple(attempts),
                    provider_id=completed_provider_id,
                    artifact=completed_artifact,
                    fallback_used=fallback_used,
                    reason="",
                )
            )

        status = ReconstructionPipelineRunStatus.DEGRADED if degraded else ReconstructionPipelineRunStatus.COMPLETED
        return self._finalize(stage_results, status)

    def _finalize(
        self,
        stage_results: Sequence[ReconstructionStageRunResult],
        status: ReconstructionPipelineRunStatus,
    ) -> ReconstructionPipelineRunResult:
        provider_uses = _aggregate_provider_uses(stage_results, self._providers)
        canonical = [
            result.artifact
            for result in stage_results
            if result.artifact is not None and result.artifact.canonical_geometry_authoritative
        ]
        return ReconstructionPipelineRunResult(
            request_id=self._plan.request.request_id,
            session_id=self._plan.request.session_id,
            status=status,
            execution_scope=_execution_scope(provider_uses),
            stages=tuple(stage_results),
            provider_provenance=provider_uses,
            canonical_geometry_artifact_id=canonical[-1].artifact_id if canonical else "",
            development_mock_used=any(
                item.execution is ReconstructionProviderExecution.DEVELOPMENT_MOCK for item in provider_uses
            ),
            certified_metrology=False,
            reconstruction_ui_enabled=False,
        )


def write_reconstruction_pipeline_json(value: object, path: str | Path) -> Path:
    if not hasattr(value, "as_dict"):
        raise TypeError("value must expose as_dict().")
    target = Path(path)
    if not target.is_absolute():
        raise ReconstructionValidationError("Pipeline JSON path must be absolute.")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = value.as_dict()  # type: ignore[attr-defined]
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target

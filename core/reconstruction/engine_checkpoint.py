"""Whole-engine pipeline checkpoint and deterministic resume contracts for Phase 8.7G2.

G2 checkpoints persist the neutral F1-F5 orchestration prefix, not opaque backend
process memory.  A resume is allowed only when the exact request/plan/provider
identity still matches and every completed artifact payload reference can be
rehydrated by the caller.  This keeps checkpointing backend-neutral and avoids
silently re-running already completed pipeline stages.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Collection, Mapping, Sequence
from uuid import uuid4

from core.reconstruction.errors import IncompatibleCheckpointError, ReconstructionValidationError
from core.reconstruction.orchestration import (
    ReconstructionArtifactAuthority,
    ReconstructionArtifactKind,
    ReconstructionPipelineArtifact,
    ReconstructionPipelinePlan,
    ReconstructionPipelineStage,
    ReconstructionProviderDescriptor,
    ReconstructionStageRunResult,
    ReconstructionStageRunState,
)
from core.reconstruction.validation import require_iso8601_utc, require_sha256, require_uuid

ENGINE_CHECKPOINT_SCHEMA_VERSION = 1
ENGINE_CHECKPOINT_PHASE = "8.7G2"
ENGINE_CHECKPOINT_MANIFEST = "reconstruction_engine_checkpoint.json"
ENGINE_CHECKPOINT_CHECKSUM = "reconstruction_engine_checkpoint.json.sha256"


class ReconstructionResumeCompatibilityStatus(str, Enum):
    COMPATIBLE = "compatible"
    INCOMPATIBLE = "incompatible"


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha256_payload(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _request_fingerprint(plan: ReconstructionPipelinePlan) -> str:
    return _sha256_payload(plan.request.as_dict())


def _plan_fingerprint(plan: ReconstructionPipelinePlan) -> str:
    return _sha256_payload(plan.as_dict())


def _provider_versions(providers: Sequence[ReconstructionProviderDescriptor]) -> tuple[tuple[str, str, str, str], ...]:
    values = tuple(
        sorted(
            (
                provider.provider_id,
                provider.provider_version,
                provider.adapter_version,
                provider.execution.value,
            )
            for provider in providers
        )
    )
    if len({value[0] for value in values}) != len(values):
        raise ReconstructionValidationError("Checkpoint providers must have unique IDs.")
    return values


def _validate_resume_prefix(plan: ReconstructionPipelinePlan, stages: Sequence[ReconstructionStageRunResult]) -> tuple[ReconstructionStageRunResult, ...]:
    values = tuple(stages)
    if not values:
        raise ReconstructionValidationError("A resumable whole-engine checkpoint requires at least one committed stage.")
    if len(values) >= len(plan.stages):
        raise ReconstructionValidationError("A G2 resumable checkpoint must stop before the complete pipeline finishes.")
    for index, result in enumerate(values):
        if not isinstance(result, ReconstructionStageRunResult):
            raise TypeError("checkpoint stages must contain ReconstructionStageRunResult values.")
        planned = plan.stages[index]
        if result.stage is not planned.stage:
            raise ReconstructionValidationError("Checkpoint stages must be an exact canonical plan prefix.")
        if result.state not in {ReconstructionStageRunState.COMPLETED, ReconstructionStageRunState.SKIPPED}:
            raise ReconstructionValidationError("Checkpoint prefix may contain only completed/skipped stages.")
        if result.required and result.state is ReconstructionStageRunState.SKIPPED:
            raise ReconstructionValidationError("Required stages cannot be skipped in a resumable checkpoint.")
        if result.state is ReconstructionStageRunState.COMPLETED:
            if result.artifact is None:
                raise ReconstructionValidationError("Completed checkpoint stages require artifacts.")
            if result.artifact.session_id != plan.request.session_id:
                raise ReconstructionValidationError("Checkpoint artifact session identity mismatch.")
            if result.provider_id not in planned.candidate_provider_ids:
                raise ReconstructionValidationError("Checkpoint provider is not valid for the current plan stage.")
    return values


@dataclass(frozen=True, slots=True)
class ReconstructionEngineCheckpoint:
    checkpoint_id: str
    request_id: str
    session_id: str
    request_fingerprint_sha256: str
    plan_fingerprint_sha256: str
    provider_versions: tuple[tuple[str, str, str, str], ...]
    stage_results: tuple[ReconstructionStageRunResult, ...]
    payload_refs: tuple[str, ...]
    next_stage: ReconstructionPipelineStage
    resume_allowed: bool = True
    created_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    schema_version: int = ENGINE_CHECKPOINT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "checkpoint_id", require_uuid(self.checkpoint_id, field_name="checkpoint_id"))
        request_id = str(self.request_id).strip()
        if not request_id:
            raise ReconstructionValidationError("request_id must be non-empty.")
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(self, "session_id", require_uuid(self.session_id, field_name="session_id"))
        object.__setattr__(self, "request_fingerprint_sha256", require_sha256(self.request_fingerprint_sha256))
        object.__setattr__(self, "plan_fingerprint_sha256", require_sha256(self.plan_fingerprint_sha256))
        providers = tuple(tuple(str(part).strip() for part in item) for item in self.provider_versions)
        if any(len(item) != 4 or any(not part for part in item) for item in providers):
            raise ReconstructionValidationError("provider_versions entries require id/version/adapter/execution.")
        if len({item[0] for item in providers}) != len(providers):
            raise ReconstructionValidationError("Checkpoint provider IDs must be unique.")
        stages = tuple(self.stage_results)
        if not stages or any(not isinstance(item, ReconstructionStageRunResult) for item in stages):
            raise ReconstructionValidationError("Checkpoint must contain committed stage results.")
        refs = tuple(str(value).strip() for value in self.payload_refs)
        if any(not value for value in refs) or len(set(refs)) != len(refs):
            raise ReconstructionValidationError("Checkpoint payload_refs must be unique and non-empty.")
        expected_refs = tuple(
            sorted(
                {
                    result.artifact.payload_ref
                    for result in stages
                    if result.artifact is not None and result.artifact.payload_ref
                }
            )
        )
        if refs != expected_refs:
            raise ReconstructionValidationError("payload_refs must exactly match committed artifact payload references.")
        if not isinstance(self.next_stage, ReconstructionPipelineStage):
            raise TypeError("next_stage must be ReconstructionPipelineStage.")
        if not isinstance(self.resume_allowed, bool):
            raise TypeError("resume_allowed must be bool.")
        if not self.resume_allowed:
            raise ReconstructionValidationError("G2 checkpoint objects represent resumable checkpoints only.")
        if int(self.schema_version) != ENGINE_CHECKPOINT_SCHEMA_VERSION:
            raise ReconstructionValidationError("Unsupported whole-engine checkpoint schema version.")
        object.__setattr__(self, "provider_versions", providers)
        object.__setattr__(self, "stage_results", stages)
        object.__setattr__(self, "payload_refs", refs)
        object.__setattr__(self, "created_at_utc", require_iso8601_utc(self.created_at_utc, field_name="created_at_utc"))
        object.__setattr__(self, "schema_version", ENGINE_CHECKPOINT_SCHEMA_VERSION)

    @property
    def completed_stage_count(self) -> int:
        return len(self.stage_results)

    @property
    def last_committed_stage(self) -> ReconstructionPipelineStage:
        return self.stage_results[-1].stage

    @property
    def checkpoint_fingerprint_sha256(self) -> str:
        payload = {
            "checkpoint_id": self.checkpoint_id,
            "request_id": self.request_id,
            "session_id": self.session_id,
            "request_fingerprint_sha256": self.request_fingerprint_sha256,
            "plan_fingerprint_sha256": self.plan_fingerprint_sha256,
            "provider_versions": [list(item) for item in self.provider_versions],
            "stage_results": [item.as_dict() for item in self.stage_results],
            "payload_refs": list(self.payload_refs),
            "next_stage": self.next_stage.value,
            "resume_allowed": self.resume_allowed,
            "schema_version": self.schema_version,
        }
        return _sha256_payload(payload)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "phase": ENGINE_CHECKPOINT_PHASE,
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_fingerprint_sha256": self.checkpoint_fingerprint_sha256,
            "request_id": self.request_id,
            "session_id": self.session_id,
            "request_fingerprint_sha256": self.request_fingerprint_sha256,
            "plan_fingerprint_sha256": self.plan_fingerprint_sha256,
            "provider_versions": [
                {
                    "provider_id": provider_id,
                    "provider_version": provider_version,
                    "adapter_version": adapter_version,
                    "execution": execution,
                }
                for provider_id, provider_version, adapter_version, execution in self.provider_versions
            ],
            "stage_results": [item.as_dict() for item in self.stage_results],
            "completed_stage_count": self.completed_stage_count,
            "last_committed_stage": self.last_committed_stage.value,
            "next_stage": self.next_stage.value,
            "payload_refs": list(self.payload_refs),
            "resume_allowed": self.resume_allowed,
            "created_at_utc": self.created_at_utc,
            "certified_metrology": False,
            "reconstruction_ui_enabled": False,
            "endurance_hardening_deferred_to": "8.7G3",
            "full_regression_deferred_to": "8.7G4",
        }


@dataclass(frozen=True, slots=True)
class ReconstructionResumeCompatibilityResult:
    status: ReconstructionResumeCompatibilityStatus
    reasons: tuple[str, ...]
    reused_stage_count: int
    next_stage: ReconstructionPipelineStage | None

    @property
    def compatible(self) -> bool:
        return self.status is ReconstructionResumeCompatibilityStatus.COMPATIBLE

    def require_compatible(self) -> None:
        if not self.compatible:
            raise IncompatibleCheckpointError("; ".join(self.reasons))

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "compatible": self.compatible,
            "reasons": list(self.reasons),
            "reused_stage_count": int(self.reused_stage_count),
            "next_stage": self.next_stage.value if self.next_stage is not None else None,
        }


def create_reconstruction_engine_checkpoint(
    *,
    plan: ReconstructionPipelinePlan,
    providers: Sequence[ReconstructionProviderDescriptor],
    stage_results: Sequence[ReconstructionStageRunResult],
    checkpoint_id: str | None = None,
) -> ReconstructionEngineCheckpoint:
    if not isinstance(plan, ReconstructionPipelinePlan):
        raise TypeError("plan must be ReconstructionPipelinePlan.")
    prefix = _validate_resume_prefix(plan, stage_results)
    next_stage = plan.stages[len(prefix)].stage
    refs = tuple(
        sorted(
            {
                result.artifact.payload_ref
                for result in prefix
                if result.artifact is not None and result.artifact.payload_ref
            }
        )
    )
    return ReconstructionEngineCheckpoint(
        checkpoint_id=str(uuid4()) if checkpoint_id is None else require_uuid(checkpoint_id, field_name="checkpoint_id"),
        request_id=plan.request.request_id,
        session_id=plan.request.session_id,
        request_fingerprint_sha256=_request_fingerprint(plan),
        plan_fingerprint_sha256=_plan_fingerprint(plan),
        provider_versions=_provider_versions(providers),
        stage_results=prefix,
        payload_refs=refs,
        next_stage=next_stage,
    )


def check_reconstruction_engine_resume_compatibility(
    checkpoint: ReconstructionEngineCheckpoint,
    *,
    plan: ReconstructionPipelinePlan,
    providers: Sequence[ReconstructionProviderDescriptor],
    available_payload_refs: Collection[str],
) -> ReconstructionResumeCompatibilityResult:
    if not isinstance(checkpoint, ReconstructionEngineCheckpoint):
        raise TypeError("checkpoint must be ReconstructionEngineCheckpoint.")
    if not isinstance(plan, ReconstructionPipelinePlan):
        raise TypeError("plan must be ReconstructionPipelinePlan.")
    reasons: list[str] = []
    if not checkpoint.resume_allowed:
        reasons.append("Checkpoint is not marked resumable.")
    if checkpoint.request_id != plan.request.request_id:
        reasons.append("Request identity differs from checkpoint.")
    if checkpoint.session_id != plan.request.session_id:
        reasons.append("Session identity differs from checkpoint.")
    if checkpoint.request_fingerprint_sha256 != _request_fingerprint(plan):
        reasons.append("Pipeline request fingerprint differs from checkpoint.")
    if checkpoint.plan_fingerprint_sha256 != _plan_fingerprint(plan):
        reasons.append("Pipeline plan fingerprint differs from checkpoint.")
    if checkpoint.provider_versions != _provider_versions(providers):
        reasons.append("Provider versions/execution identity differ from checkpoint.")
    try:
        _validate_resume_prefix(plan, checkpoint.stage_results)
    except (TypeError, ReconstructionValidationError) as exc:
        reasons.append(f"Checkpoint stage prefix is incompatible: {exc}")
    if len(checkpoint.stage_results) < len(plan.stages):
        expected_next = plan.stages[len(checkpoint.stage_results)].stage
        if checkpoint.next_stage is not expected_next:
            reasons.append("Checkpoint next_stage does not match the current pipeline plan.")
    available = {str(value).strip() for value in available_payload_refs if str(value).strip()}
    missing_refs = [value for value in checkpoint.payload_refs if value not in available]
    if missing_refs:
        reasons.append("Checkpoint payload references are not rehydrated: " + ",".join(missing_refs))
    status = (
        ReconstructionResumeCompatibilityStatus.COMPATIBLE
        if not reasons
        else ReconstructionResumeCompatibilityStatus.INCOMPATIBLE
    )
    return ReconstructionResumeCompatibilityResult(
        status=status,
        reasons=tuple(reasons),
        reused_stage_count=len(checkpoint.stage_results),
        next_stage=checkpoint.next_stage if not reasons else None,
    )


def _artifact_from_mapping(raw: Mapping[str, Any]) -> ReconstructionPipelineArtifact:
    return ReconstructionPipelineArtifact(
        artifact_id=str(raw["artifact_id"]),
        session_id=str(raw["session_id"]),
        stage=ReconstructionPipelineStage(str(raw["stage"])),
        kind=ReconstructionArtifactKind(str(raw["kind"])),
        authority=ReconstructionArtifactAuthority(str(raw["authority"])),
        provider_id=str(raw["provider_id"]),
        payload_ref=str(raw.get("payload_ref", "")),
        metadata=raw.get("metadata", {}),
        schema_version=int(raw.get("schema_version", 0)),
    )


def _stage_result_from_mapping(raw: Mapping[str, Any]) -> ReconstructionStageRunResult:
    artifact_raw = raw.get("artifact")
    artifact = _artifact_from_mapping(artifact_raw) if isinstance(artifact_raw, Mapping) else None
    return ReconstructionStageRunResult(
        stage=ReconstructionPipelineStage(str(raw["stage"])),
        state=ReconstructionStageRunState(str(raw["state"])),
        required=bool(raw["required"]),
        provider_attempts=tuple(str(value) for value in raw.get("provider_attempts", [])),
        provider_id=str(raw.get("provider_id", "")),
        artifact=artifact,
        fallback_used=bool(raw.get("fallback_used", False)),
        reason=str(raw.get("reason", "")),
    )


def _checkpoint_from_mapping(raw: Mapping[str, Any]) -> ReconstructionEngineCheckpoint:
    if int(raw.get("schema_version", 0)) != ENGINE_CHECKPOINT_SCHEMA_VERSION:
        raise ReconstructionValidationError("Unsupported whole-engine checkpoint schema version.")
    if str(raw.get("phase", "")) != ENGINE_CHECKPOINT_PHASE:
        raise ReconstructionValidationError("Checkpoint phase marker is incompatible.")
    providers = tuple(
        (
            str(item["provider_id"]),
            str(item["provider_version"]),
            str(item.get("adapter_version", "")),
            str(item["execution"]),
        )
        for item in raw.get("provider_versions", [])
        if isinstance(item, Mapping)
    )
    stages = tuple(
        _stage_result_from_mapping(item)
        for item in raw.get("stage_results", [])
        if isinstance(item, Mapping)
    )
    checkpoint = ReconstructionEngineCheckpoint(
        checkpoint_id=str(raw["checkpoint_id"]),
        request_id=str(raw["request_id"]),
        session_id=str(raw["session_id"]),
        request_fingerprint_sha256=str(raw["request_fingerprint_sha256"]),
        plan_fingerprint_sha256=str(raw["plan_fingerprint_sha256"]),
        provider_versions=providers,
        stage_results=stages,
        payload_refs=tuple(str(value) for value in raw.get("payload_refs", [])),
        next_stage=ReconstructionPipelineStage(str(raw["next_stage"])),
        resume_allowed=bool(raw.get("resume_allowed", False)),
        created_at_utc=str(raw["created_at_utc"]),
        schema_version=int(raw["schema_version"]),
    )
    supplied_fingerprint = require_sha256(str(raw.get("checkpoint_fingerprint_sha256", "")))
    if checkpoint.checkpoint_fingerprint_sha256 != supplied_fingerprint:
        raise ReconstructionValidationError("Whole-engine checkpoint semantic fingerprint mismatch.")
    return checkpoint


def write_reconstruction_engine_checkpoint(
    checkpoint: ReconstructionEngineCheckpoint,
    output_dir: str | Path,
) -> Path:
    if not isinstance(checkpoint, ReconstructionEngineCheckpoint):
        raise TypeError("checkpoint must be ReconstructionEngineCheckpoint.")
    root = Path(output_dir).expanduser()
    if not root.is_absolute():
        raise ReconstructionValidationError("Checkpoint output_dir must be absolute.")
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink():
        raise ReconstructionValidationError("Refusing symlinked checkpoint output directory.")
    manifest = root / ENGINE_CHECKPOINT_MANIFEST
    checksum = root / ENGINE_CHECKPOINT_CHECKSUM
    payload = json.dumps(checkpoint.as_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n"
    encoded = payload.encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    tmp_manifest = root / f".{ENGINE_CHECKPOINT_MANIFEST}.tmp-{os.getpid()}"
    tmp_checksum = root / f".{ENGINE_CHECKPOINT_CHECKSUM}.tmp-{os.getpid()}"
    with tmp_manifest.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    with tmp_checksum.open("w", encoding="utf-8") as handle:
        handle.write(f"{digest}  {ENGINE_CHECKPOINT_MANIFEST}\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_manifest, manifest)
    os.replace(tmp_checksum, checksum)
    return manifest


def read_reconstruction_engine_checkpoint(path: str | Path) -> ReconstructionEngineCheckpoint:
    target = Path(path).expanduser()
    if target.is_dir():
        target = target / ENGINE_CHECKPOINT_MANIFEST
    if not target.is_absolute():
        raise ReconstructionValidationError("Checkpoint path must be absolute.")
    target = target.resolve()
    if not target.is_file():
        raise ReconstructionValidationError(f"Checkpoint manifest does not exist: {target}")
    checksum = target.parent / ENGINE_CHECKPOINT_CHECKSUM
    if not checksum.is_file():
        raise ReconstructionValidationError("Checkpoint checksum sidecar is missing.")
    parts = checksum.read_text(encoding="utf-8").strip().split()
    if len(parts) < 2 or parts[1] != ENGINE_CHECKPOINT_MANIFEST:
        raise ReconstructionValidationError("Checkpoint checksum sidecar is malformed.")
    expected = require_sha256(parts[0])
    encoded = target.read_bytes()
    if hashlib.sha256(encoded).hexdigest() != expected:
        raise ReconstructionValidationError("Checkpoint manifest SHA-256 verification failed.")
    try:
        raw = json.loads(encoded)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ReconstructionValidationError("Checkpoint manifest must be valid JSON.") from exc
    if not isinstance(raw, Mapping):
        raise ReconstructionValidationError("Checkpoint manifest root must be an object.")
    return _checkpoint_from_mapping(raw)

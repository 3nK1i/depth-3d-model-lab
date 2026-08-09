"""Persistent reconstruction-scene manifest for Phase 8.7F5.

This module closes the Phase 8.7F integration layer without introducing a UI
or whole-engine checkpoint/resume semantics.  It turns a completed/degraded
pipeline result into a durable, reloadable scene manifest with stable identity,
provider provenance, authority metadata, fallback/degradation evidence and
bounded file-integrity references for file-backed appearance artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.contracts.validation import (
    JsonValue,
    metadata_as_dict,
    normalize_metadata,
    require_nonempty,
)
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.orchestration import (
    ReconstructionArtifactAuthority,
    ReconstructionArtifactKind,
    ReconstructionPipelineArtifact,
    ReconstructionPipelineRunResult,
    ReconstructionPipelineRunStatus,
    ReconstructionPipelineStage,
    ReconstructionProviderExecution,
    ReconstructionProviderUse,
    ReconstructionStageRunState,
    reconstruction_stage_contract,
)
from core.reconstruction.validation import require_iso8601_utc, require_sha256, require_uuid

PERSISTENT_RECONSTRUCTION_SCENE_SCHEMA_VERSION = 1
PERSISTENT_RECONSTRUCTION_SCENE_MANIFEST = "persistent_reconstruction_scene.json"
PERSISTENT_RECONSTRUCTION_SCENE_CHECKSUM = "persistent_reconstruction_scene.json.sha256"
PERSISTENT_SCENE_PHASE = "8.7F5"


class PersistentReconstructionSceneState(str, Enum):
    READY = "ready"
    DEGRADED = "degraded"


@dataclass(frozen=True, slots=True)
class PersistentSceneArtifactRecord:
    artifact_id: str
    stage: ReconstructionPipelineStage
    kind: ReconstructionArtifactKind
    authority: ReconstructionArtifactAuthority
    provider_id: str
    payload_ref: str = ""
    metadata: Mapping[str, JsonValue] | Sequence[tuple[str, JsonValue]] = field(default_factory=tuple)
    payload_snapshot: Mapping[str, JsonValue] | Sequence[tuple[str, JsonValue]] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_id", require_nonempty(self.artifact_id, field_name="artifact_id"))
        if not isinstance(self.stage, ReconstructionPipelineStage):
            raise TypeError("stage must be ReconstructionPipelineStage.")
        if not isinstance(self.kind, ReconstructionArtifactKind):
            raise TypeError("kind must be ReconstructionArtifactKind.")
        if not isinstance(self.authority, ReconstructionArtifactAuthority):
            raise TypeError("authority must be ReconstructionArtifactAuthority.")
        contract = reconstruction_stage_contract(self.stage)
        if contract.output_kind is not self.kind or contract.output_authority is not self.authority:
            raise ReconstructionValidationError("Persistent scene artifact must preserve the canonical stage authority contract.")
        if self.kind in {ReconstructionArtifactKind.GAUSSIAN_SCENE, ReconstructionArtifactKind.NERF_SCENE}:
            if self.authority is not ReconstructionArtifactAuthority.APPEARANCE_ONLY:
                raise ReconstructionValidationError("Gaussian/NeRF persistent artifacts must remain appearance-only.")
        object.__setattr__(self, "provider_id", require_nonempty(self.provider_id, field_name="provider_id"))
        object.__setattr__(self, "payload_ref", str(self.payload_ref).strip())
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata))
        object.__setattr__(self, "payload_snapshot", normalize_metadata(self.payload_snapshot))

    @property
    def canonical_geometry_authoritative(self) -> bool:
        return self.authority is ReconstructionArtifactAuthority.CANONICAL_GEOMETRY

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "stage": self.stage.value,
            "kind": self.kind.value,
            "authority": self.authority.value,
            "provider_id": self.provider_id,
            "payload_ref": self.payload_ref,
            "canonical_geometry_authoritative": self.canonical_geometry_authoritative,
            "certified_metrology": False,
            "metadata": metadata_as_dict(self.metadata),  # type: ignore[arg-type]
            "payload_snapshot": metadata_as_dict(self.payload_snapshot),  # type: ignore[arg-type]
        }


@dataclass(frozen=True, slots=True)
class PersistentSceneFallbackEvidence:
    stage: ReconstructionPipelineStage
    state: ReconstructionStageRunState
    provider_attempts: tuple[str, ...]
    fallback_used: bool
    reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.stage, ReconstructionPipelineStage):
            raise TypeError("stage must be ReconstructionPipelineStage.")
        if not isinstance(self.state, ReconstructionStageRunState):
            raise TypeError("state must be ReconstructionStageRunState.")
        attempts = tuple(require_nonempty(value, field_name="provider_attempt") for value in self.provider_attempts)
        object.__setattr__(self, "provider_attempts", attempts)
        object.__setattr__(self, "fallback_used", bool(self.fallback_used))
        object.__setattr__(self, "reason", str(self.reason).strip())
        if self.state is ReconstructionStageRunState.COMPLETED and not self.fallback_used:
            raise ReconstructionValidationError("Completed fallback evidence must represent an actual fallback.")
        if self.state is not ReconstructionStageRunState.COMPLETED and not self.reason:
            raise ReconstructionValidationError("Skipped/failed/cancelled fallback evidence requires a reason.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "state": self.state.value,
            "provider_attempts": list(self.provider_attempts),
            "fallback_used": self.fallback_used,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class PersistentSceneFileReference:
    artifact_id: str
    role: str
    path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_id", require_nonempty(self.artifact_id, field_name="artifact_id"))
        object.__setattr__(self, "role", require_nonempty(self.role, field_name="role"))
        path = str(Path(self.path).resolve())
        object.__setattr__(self, "path", path)
        size = int(self.size_bytes)
        if size < 0:
            raise ReconstructionValidationError("size_bytes must be non-negative.")
        object.__setattr__(self, "size_bytes", size)
        object.__setattr__(self, "sha256", require_sha256(self.sha256, field_name="sha256", allow_empty=False))

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "role": self.role,
            "path": self.path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint_payload(
    *,
    request_id: str,
    session_id: str,
    state: PersistentReconstructionSceneState,
    pipeline_status: ReconstructionPipelineRunStatus,
    artifacts: Sequence[PersistentSceneArtifactRecord],
    provider_provenance: Sequence[ReconstructionProviderUse],
    fallback_evidence: Sequence[PersistentSceneFallbackEvidence],
    file_references: Sequence[PersistentSceneFileReference],
    canonical_geometry_artifact_id: str,
) -> dict[str, Any]:
    return {
        "schema_version": PERSISTENT_RECONSTRUCTION_SCENE_SCHEMA_VERSION,
        "phase": PERSISTENT_SCENE_PHASE,
        "request_id": request_id,
        "session_id": session_id,
        "state": state.value,
        "pipeline_status": pipeline_status.value,
        "artifacts": [item.as_dict() for item in artifacts],
        "provider_provenance": [item.as_dict() for item in provider_provenance],
        "fallback_evidence": [item.as_dict() for item in fallback_evidence],
        "file_references": [item.as_dict() for item in file_references],
        "canonical_geometry_artifact_id": canonical_geometry_artifact_id,
        "canonical_geometry_authoritative": True,
        "certified_metrology": False,
        "reconstruction_ui_enabled": False,
    }


@dataclass(frozen=True, slots=True)
class PersistentReconstructionScene:
    scene_id: str
    request_id: str
    session_id: str
    state: PersistentReconstructionSceneState
    pipeline_status: ReconstructionPipelineRunStatus
    artifacts: tuple[PersistentSceneArtifactRecord, ...]
    provider_provenance: tuple[ReconstructionProviderUse, ...]
    fallback_evidence: tuple[PersistentSceneFallbackEvidence, ...]
    file_references: tuple[PersistentSceneFileReference, ...]
    canonical_geometry_artifact_id: str
    scene_fingerprint_sha256: str
    created_at_utc: str
    certified_metrology: bool = False
    reconstruction_ui_enabled: bool = False
    schema_version: int = PERSISTENT_RECONSTRUCTION_SCENE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "scene_id", require_nonempty(self.scene_id, field_name="scene_id"))
        object.__setattr__(self, "request_id", require_nonempty(self.request_id, field_name="request_id"))
        object.__setattr__(self, "session_id", require_uuid(self.session_id, field_name="session_id"))
        if not isinstance(self.state, PersistentReconstructionSceneState):
            raise TypeError("state must be PersistentReconstructionSceneState.")
        if not isinstance(self.pipeline_status, ReconstructionPipelineRunStatus):
            raise TypeError("pipeline_status must be ReconstructionPipelineRunStatus.")
        if self.pipeline_status not in {ReconstructionPipelineRunStatus.COMPLETED, ReconstructionPipelineRunStatus.DEGRADED}:
            raise ReconstructionValidationError("Persistent scenes require a completed or degraded pipeline result.")
        expected_state = (
            PersistentReconstructionSceneState.DEGRADED
            if self.pipeline_status is ReconstructionPipelineRunStatus.DEGRADED
            else PersistentReconstructionSceneState.READY
        )
        if self.state is not expected_state:
            raise ReconstructionValidationError("Persistent scene lifecycle state must agree with pipeline status.")
        artifacts = tuple(self.artifacts)
        provenance = tuple(self.provider_provenance)
        fallback = tuple(self.fallback_evidence)
        files = tuple(self.file_references)
        if not artifacts or any(not isinstance(item, PersistentSceneArtifactRecord) for item in artifacts):
            raise ReconstructionValidationError("Persistent scene requires artifact records.")
        if any(not isinstance(item, ReconstructionProviderUse) for item in provenance):
            raise TypeError("provider_provenance must contain ReconstructionProviderUse values.")
        if any(not isinstance(item, PersistentSceneFallbackEvidence) for item in fallback):
            raise TypeError("fallback_evidence must contain PersistentSceneFallbackEvidence values.")
        if any(not isinstance(item, PersistentSceneFileReference) for item in files):
            raise TypeError("file_references must contain PersistentSceneFileReference values.")
        if len({item.artifact_id for item in artifacts}) != len(artifacts):
            raise ReconstructionValidationError("Persistent artifact IDs must be unique.")
        if len({item.provider_id for item in provenance}) != len(provenance):
            raise ReconstructionValidationError("Persistent provider provenance IDs must be unique.")
        stage_order = list(ReconstructionPipelineStage)
        positions = [stage_order.index(item.stage) for item in artifacts]
        if positions != sorted(positions) or len(set(positions)) != len(positions):
            raise ReconstructionValidationError("Persistent artifacts must follow the canonical stage order exactly once.")
        by_stage = {item.stage: item for item in artifacts}
        required_core = {
            ReconstructionPipelineStage.SOURCE_INGESTION,
            ReconstructionPipelineStage.CAMERA_POSE,
            ReconstructionPipelineStage.DEPTH_EVIDENCE,
            ReconstructionPipelineStage.DENSE_GEOMETRY,
            ReconstructionPipelineStage.SURFACE_MESH,
            ReconstructionPipelineStage.SEMANTIC_GEOMETRY,
            ReconstructionPipelineStage.INSPECTION_QC,
        }
        if not required_core.issubset(by_stage):
            missing = sorted(stage.value for stage in required_core.difference(by_stage))
            raise ReconstructionValidationError(f"Persistent integrated scene is missing required F3 stages: {missing!r}.")
        canonical_id = require_nonempty(self.canonical_geometry_artifact_id, field_name="canonical_geometry_artifact_id")
        canonical = [item for item in artifacts if item.canonical_geometry_authoritative]
        if not canonical or canonical[-1].artifact_id != canonical_id:
            raise ReconstructionValidationError("Persistent scene canonical geometry ID must point to the latest canonical artifact.")
        for kind in (ReconstructionArtifactKind.GAUSSIAN_SCENE, ReconstructionArtifactKind.NERF_SCENE):
            item = next((artifact for artifact in artifacts if artifact.kind is kind), None)
            if item is not None and item.authority is not ReconstructionArtifactAuthority.APPEARANCE_ONLY:
                raise ReconstructionValidationError("Neural appearance persistence cannot change authority.")
        if bool(self.certified_metrology):
            raise ReconstructionValidationError("Phase 8.7F5 persistence cannot certify metrology.")
        if bool(self.reconstruction_ui_enabled):
            raise ReconstructionValidationError("Reconstruction UI remains disabled until Phase 8.8.")
        fingerprint = require_sha256(self.scene_fingerprint_sha256, field_name="scene_fingerprint_sha256", allow_empty=False)
        expected_scene_id = f"prs-{fingerprint[:24]}"
        if self.scene_id != expected_scene_id:
            raise ReconstructionValidationError("scene_id must be derived from scene_fingerprint_sha256.")
        created = require_iso8601_utc(self.created_at_utc, field_name="created_at_utc")
        if int(self.schema_version) != PERSISTENT_RECONSTRUCTION_SCENE_SCHEMA_VERSION:
            raise ReconstructionValidationError("Unsupported persistent reconstruction scene schema version.")
        expected_fingerprint = hashlib.sha256(
            _canonical_json(
                _fingerprint_payload(
                    request_id=self.request_id,
                    session_id=self.session_id,
                    state=self.state,
                    pipeline_status=self.pipeline_status,
                    artifacts=artifacts,
                    provider_provenance=provenance,
                    fallback_evidence=fallback,
                    file_references=files,
                    canonical_geometry_artifact_id=canonical_id,
                )
            )
        ).hexdigest()
        if fingerprint != expected_fingerprint:
            raise ReconstructionValidationError("Persistent scene fingerprint does not match its normalized content.")
        object.__setattr__(self, "artifacts", artifacts)
        object.__setattr__(self, "provider_provenance", provenance)
        object.__setattr__(self, "fallback_evidence", fallback)
        object.__setattr__(self, "file_references", files)
        object.__setattr__(self, "canonical_geometry_artifact_id", canonical_id)
        object.__setattr__(self, "scene_fingerprint_sha256", fingerprint)
        object.__setattr__(self, "created_at_utc", created)
        object.__setattr__(self, "certified_metrology", False)
        object.__setattr__(self, "reconstruction_ui_enabled", False)
        object.__setattr__(self, "schema_version", PERSISTENT_RECONSTRUCTION_SCENE_SCHEMA_VERSION)

    @property
    def canonical_geometry_authoritative(self) -> bool:
        return True

    @property
    def gaussian_artifact(self) -> PersistentSceneArtifactRecord | None:
        return next((item for item in self.artifacts if item.kind is ReconstructionArtifactKind.GAUSSIAN_SCENE), None)

    @property
    def nerf_artifact(self) -> PersistentSceneArtifactRecord | None:
        return next((item for item in self.artifacts if item.kind is ReconstructionArtifactKind.NERF_SCENE), None)

    def as_dict(self) -> dict[str, Any]:
        payload = _fingerprint_payload(
            request_id=self.request_id,
            session_id=self.session_id,
            state=self.state,
            pipeline_status=self.pipeline_status,
            artifacts=self.artifacts,
            provider_provenance=self.provider_provenance,
            fallback_evidence=self.fallback_evidence,
            file_references=self.file_references,
            canonical_geometry_artifact_id=self.canonical_geometry_artifact_id,
        )
        payload.update(
            {
                "scene_id": self.scene_id,
                "scene_fingerprint_sha256": self.scene_fingerprint_sha256,
                "created_at_utc": self.created_at_utc,
                "appearance": {
                    "gaussian_artifact_id": self.gaussian_artifact.artifact_id if self.gaussian_artifact else "",
                    "nerf_artifact_id": self.nerf_artifact.artifact_id if self.nerf_artifact else "",
                    "authority": "appearance_only",
                },
                "future_phase_ownership": {
                    "whole_engine_hardening": "8.7G",
                    "ui": "8.8",
                    "production_release": "8.9",
                },
            }
        )
        return payload


def _artifact_record_from_mapping(
    artifact: Mapping[str, Any],
    *,
    payload_snapshot: Mapping[str, Any] | None = None,
) -> PersistentSceneArtifactRecord:
    return PersistentSceneArtifactRecord(
        artifact_id=str(artifact["artifact_id"]),
        stage=ReconstructionPipelineStage(str(artifact["stage"])),
        kind=ReconstructionArtifactKind(str(artifact["kind"])),
        authority=ReconstructionArtifactAuthority(str(artifact["authority"])),
        provider_id=str(artifact["provider_id"]),
        payload_ref=str(artifact.get("payload_ref", "")),
        metadata=artifact.get("metadata", {}),
        payload_snapshot=payload_snapshot or {},
    )


def _provider_use_from_mapping(payload: Mapping[str, Any]) -> ReconstructionProviderUse:
    return ReconstructionProviderUse(
        provider_id=str(payload["provider_id"]),
        provider_version=str(payload["provider_version"]),
        execution=ReconstructionProviderExecution(str(payload["execution"])),
        stages=tuple(ReconstructionPipelineStage(str(value)) for value in payload.get("stages", [])),
    )


def _file_references_from_artifacts(
    artifacts: Sequence[PersistentSceneArtifactRecord],
) -> tuple[PersistentSceneFileReference, ...]:
    references: list[PersistentSceneFileReference] = []
    seen: set[tuple[str, str]] = set()
    for artifact in artifacts:
        snapshot = metadata_as_dict(artifact.payload_snapshot)  # type: ignore[arg-type]
        for key, value in snapshot.items():
            if not isinstance(value, str) or not value.strip():
                continue
            if not (key.endswith("_path") or key in {"scene_path", "config_path", "checkpoint_path"}):
                continue
            path = Path(value).expanduser()
            if not path.is_absolute() or not path.is_file():
                continue
            resolved = path.resolve()
            marker = (artifact.artifact_id, str(resolved))
            if marker in seen:
                continue
            seen.add(marker)
            references.append(
                PersistentSceneFileReference(
                    artifact_id=artifact.artifact_id,
                    role=key,
                    path=str(resolved),
                    size_bytes=resolved.stat().st_size,
                    sha256=_sha256_file(resolved),
                )
            )
    references.sort(key=lambda item: (item.artifact_id, item.role, item.path))
    return tuple(references)


def persistent_scene_from_pipeline_payload(
    payload: Mapping[str, Any],
    *,
    payload_snapshots_by_kind: Mapping[str, Mapping[str, Any]] | None = None,
    created_at_utc: str | None = None,
) -> PersistentReconstructionScene:
    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a pipeline-result mapping.")
    status = ReconstructionPipelineRunStatus(str(payload["status"]))
    if status not in {ReconstructionPipelineRunStatus.COMPLETED, ReconstructionPipelineRunStatus.DEGRADED}:
        raise ReconstructionValidationError("Only completed/degraded pipeline results can become persistent scenes.")
    if not bool(payload.get("canonical_geometry_authoritative", False)):
        raise ReconstructionValidationError("Persistent scene requires canonical geometry authority.")
    if bool(payload.get("certified_metrology", False)):
        raise ReconstructionValidationError("F5 cannot persist a pipeline that unexpectedly claims certified metrology.")
    snapshots = payload_snapshots_by_kind or {}
    artifacts: list[PersistentSceneArtifactRecord] = []
    fallback: list[PersistentSceneFallbackEvidence] = []
    for stage_payload in payload.get("stages", []):
        if not isinstance(stage_payload, Mapping):
            raise ReconstructionValidationError("Pipeline stage payloads must be objects.")
        state = ReconstructionStageRunState(str(stage_payload["state"]))
        artifact_payload = stage_payload.get("artifact")
        if state is ReconstructionStageRunState.COMPLETED:
            if not isinstance(artifact_payload, Mapping):
                raise ReconstructionValidationError("Completed pipeline stage lacks artifact payload.")
            kind = str(artifact_payload["kind"])
            artifacts.append(
                _artifact_record_from_mapping(
                    artifact_payload,
                    payload_snapshot=snapshots.get(kind),
                )
            )
        if bool(stage_payload.get("fallback_used", False)) or state is not ReconstructionStageRunState.COMPLETED:
            fallback.append(
                PersistentSceneFallbackEvidence(
                    stage=ReconstructionPipelineStage(str(stage_payload["stage"])),
                    state=state,
                    provider_attempts=tuple(str(value) for value in stage_payload.get("provider_attempts", [])),
                    fallback_used=bool(stage_payload.get("fallback_used", False)),
                    reason=str(stage_payload.get("reason", "")),
                )
            )
    provider_provenance = tuple(
        _provider_use_from_mapping(item)
        for item in payload.get("provider_provenance", [])
        if isinstance(item, Mapping)
    )
    state = (
        PersistentReconstructionSceneState.DEGRADED
        if status is ReconstructionPipelineRunStatus.DEGRADED
        else PersistentReconstructionSceneState.READY
    )
    files = _file_references_from_artifacts(artifacts)
    fingerprint_payload = _fingerprint_payload(
        request_id=str(payload["request_id"]),
        session_id=str(payload["session_id"]),
        state=state,
        pipeline_status=status,
        artifacts=artifacts,
        provider_provenance=provider_provenance,
        fallback_evidence=fallback,
        file_references=files,
        canonical_geometry_artifact_id=str(payload["canonical_geometry_artifact_id"]),
    )
    fingerprint = hashlib.sha256(_canonical_json(fingerprint_payload)).hexdigest()
    return PersistentReconstructionScene(
        scene_id=f"prs-{fingerprint[:24]}",
        request_id=str(payload["request_id"]),
        session_id=str(payload["session_id"]),
        state=state,
        pipeline_status=status,
        artifacts=tuple(artifacts),
        provider_provenance=provider_provenance,
        fallback_evidence=tuple(fallback),
        file_references=files,
        canonical_geometry_artifact_id=str(payload["canonical_geometry_artifact_id"]),
        scene_fingerprint_sha256=fingerprint,
        created_at_utc=created_at_utc or datetime.now(timezone.utc).isoformat(),
    )


def persistent_scene_from_pipeline_result(
    result: ReconstructionPipelineRunResult,
    *,
    payload_snapshots_by_kind: Mapping[str, Mapping[str, Any]] | None = None,
    created_at_utc: str | None = None,
) -> PersistentReconstructionScene:
    if not isinstance(result, ReconstructionPipelineRunResult):
        raise TypeError("result must be ReconstructionPipelineRunResult.")
    return persistent_scene_from_pipeline_payload(
        result.as_dict(),
        payload_snapshots_by_kind=payload_snapshots_by_kind,
        created_at_utc=created_at_utc,
    )


def write_persistent_reconstruction_scene(
    scene: PersistentReconstructionScene,
    output_dir: str | Path,
) -> Path:
    if not isinstance(scene, PersistentReconstructionScene):
        raise TypeError("scene must be PersistentReconstructionScene.")
    root = Path(output_dir).expanduser()
    if not root.is_absolute():
        raise ReconstructionValidationError("Persistent scene output_dir must be absolute.")
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / PERSISTENT_RECONSTRUCTION_SCENE_MANIFEST
    payload = json.dumps(scene.as_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n"
    manifest.write_text(payload, encoding="utf-8")
    manifest_sha = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    (root / PERSISTENT_RECONSTRUCTION_SCENE_CHECKSUM).write_text(
        f"{manifest_sha}  {PERSISTENT_RECONSTRUCTION_SCENE_MANIFEST}\n",
        encoding="utf-8",
    )
    return manifest


def _parse_scene(payload: Mapping[str, Any]) -> PersistentReconstructionScene:
    artifacts = tuple(
        _artifact_record_from_mapping(
            item,
            payload_snapshot=item.get("payload_snapshot", {}),
        )
        for item in payload.get("artifacts", [])
        if isinstance(item, Mapping)
    )
    provenance = tuple(
        _provider_use_from_mapping(item)
        for item in payload.get("provider_provenance", [])
        if isinstance(item, Mapping)
    )
    fallback = tuple(
        PersistentSceneFallbackEvidence(
            stage=ReconstructionPipelineStage(str(item["stage"])),
            state=ReconstructionStageRunState(str(item["state"])),
            provider_attempts=tuple(str(value) for value in item.get("provider_attempts", [])),
            fallback_used=bool(item.get("fallback_used", False)),
            reason=str(item.get("reason", "")),
        )
        for item in payload.get("fallback_evidence", [])
        if isinstance(item, Mapping)
    )
    files = tuple(
        PersistentSceneFileReference(
            artifact_id=str(item["artifact_id"]),
            role=str(item["role"]),
            path=str(item["path"]),
            size_bytes=int(item["size_bytes"]),
            sha256=str(item["sha256"]),
        )
        for item in payload.get("file_references", [])
        if isinstance(item, Mapping)
    )
    return PersistentReconstructionScene(
        scene_id=str(payload["scene_id"]),
        request_id=str(payload["request_id"]),
        session_id=str(payload["session_id"]),
        state=PersistentReconstructionSceneState(str(payload["state"])),
        pipeline_status=ReconstructionPipelineRunStatus(str(payload["pipeline_status"])),
        artifacts=artifacts,
        provider_provenance=provenance,
        fallback_evidence=fallback,
        file_references=files,
        canonical_geometry_artifact_id=str(payload["canonical_geometry_artifact_id"]),
        scene_fingerprint_sha256=str(payload["scene_fingerprint_sha256"]),
        created_at_utc=str(payload["created_at_utc"]),
        certified_metrology=bool(payload.get("certified_metrology", False)),
        reconstruction_ui_enabled=bool(payload.get("reconstruction_ui_enabled", False)),
        schema_version=int(payload.get("schema_version", 0)),
    )


def read_persistent_reconstruction_scene(path: str | Path) -> PersistentReconstructionScene:
    target = Path(path).expanduser()
    if target.is_dir():
        target = target / PERSISTENT_RECONSTRUCTION_SCENE_MANIFEST
    if not target.is_absolute():
        raise ReconstructionValidationError("Persistent scene path must be absolute.")
    target = target.resolve()
    if not target.is_file():
        raise ReconstructionValidationError(f"Persistent scene manifest does not exist: {target}")
    checksum = target.parent / PERSISTENT_RECONSTRUCTION_SCENE_CHECKSUM
    if not checksum.is_file():
        raise ReconstructionValidationError("Persistent scene checksum sidecar is missing.")
    parts = checksum.read_text(encoding="utf-8").strip().split()
    if len(parts) < 2 or parts[1] != PERSISTENT_RECONSTRUCTION_SCENE_MANIFEST:
        raise ReconstructionValidationError("Persistent scene checksum sidecar is malformed.")
    expected_manifest_sha = require_sha256(parts[0], field_name="manifest_sha256", allow_empty=False)
    actual_manifest_sha = _sha256_file(target)
    if actual_manifest_sha != expected_manifest_sha:
        raise ReconstructionValidationError("Persistent scene manifest checksum mismatch.")
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ReconstructionValidationError("Persistent scene manifest is not valid JSON.") from exc
    if not isinstance(raw, Mapping):
        raise ReconstructionValidationError("Persistent scene manifest root must be an object.")
    return _parse_scene(raw)


def verify_persistent_scene_file_references(scene: PersistentReconstructionScene) -> tuple[PersistentSceneFileReference, ...]:
    if not isinstance(scene, PersistentReconstructionScene):
        raise TypeError("scene must be PersistentReconstructionScene.")
    for item in scene.file_references:
        path = Path(item.path)
        if not path.is_file():
            raise ReconstructionValidationError(f"Persistent scene referenced file is missing: {path}")
        if path.stat().st_size != item.size_bytes:
            raise ReconstructionValidationError(f"Persistent scene referenced file size changed: {path}")
        if _sha256_file(path) != item.sha256:
            raise ReconstructionValidationError(f"Persistent scene referenced file checksum changed: {path}")
    return scene.file_references

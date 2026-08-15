"""Phase 9.4D persistent Gaussian appearance representation.

This module attaches one already-optimized Gaussian appearance artifact to the
existing persistent reconstruction-scene contract.  It does not introduce a
new scene schema, does not change canonical geometry authority, and never
starts a Gaussian backend.  The output is a new immutable persistent-scene
revision whose canonical artifact lineage is unchanged and whose Gaussian
artifact remains appearance-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import json
from pathlib import Path
import shutil
from typing import Any, Mapping

from core.contracts.validation import metadata_as_dict
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.gaussian_splatting import (
    GaussianOptimizationState,
    GaussianSceneArtifact,
    GaussianSceneRole,
)
from core.reconstruction.orchestration import (
    ReconstructionArtifactAuthority,
    ReconstructionArtifactKind,
    ReconstructionPipelineStage,
    ReconstructionProviderExecution,
    ReconstructionProviderUse,
    ReconstructionStageRunState,
)
from core.reconstruction.persistent_scene import (
    PersistentReconstructionScene,
    PersistentSceneArtifactRecord,
    PersistentSceneFileReference,
    persistent_scene_from_pipeline_payload,
    read_persistent_reconstruction_scene,
    verify_persistent_scene_file_references,
    write_persistent_reconstruction_scene,
)
from core.reconstruction.pipeline_appearance import PIPELINE_GAUSSIAN_PROVIDER_ID

PHASE9_4D_GAUSSIAN_PERSISTENCE_SCHEMA_VERSION = 1
PHASE9_4D_GAUSSIAN_PERSISTENCE_PHASE = "9.4D"


class GaussianPersistenceState(str, Enum):
    PERSISTED = "persisted"
    RELOADED = "reloaded"


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_verified(source: str | Path, target: Path, *, label: str) -> Path:
    src = Path(source).expanduser().resolve()
    if not src.is_file() or src.stat().st_size <= 0:
        raise ReconstructionValidationError(f"{label} is missing or empty: {src}")
    target = target.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, target)
    if target.stat().st_size != src.stat().st_size or _sha256_file(target) != _sha256_file(src):
        raise ReconstructionValidationError(f"{label} copy failed integrity verification.")
    return target


def _scene_snapshot(scene: PersistentReconstructionScene, kind: ReconstructionArtifactKind) -> dict[str, Any]:
    artifact = next((item for item in scene.artifacts if item.kind is kind), None)
    if artifact is None:
        raise ReconstructionValidationError(f"Persistent scene lacks required {kind.value} evidence.")
    return metadata_as_dict(artifact.payload_snapshot)  # type: ignore[arg-type]


def _scene_reconstruction_frame_id(scene: PersistentReconstructionScene) -> str:
    for kind in (
        ReconstructionArtifactKind.CAMERA_TRAJECTORY,
        ReconstructionArtifactKind.INGESTION_RESULT,
        ReconstructionArtifactKind.SURFACE_MESH,
    ):
        artifact = next((item for item in scene.artifacts if item.kind is kind), None)
        if artifact is None:
            continue
        snapshot = metadata_as_dict(artifact.payload_snapshot)  # type: ignore[arg-type]
        value = str(snapshot.get("reconstruction_frame_id", "")).strip()
        if value:
            return value
    raise ReconstructionValidationError("Persistent scene does not expose a reconstruction frame identity.")


def _validate_scene_artifact_compatibility(
    scene: PersistentReconstructionScene,
    artifact: GaussianSceneArtifact,
    *,
    expected_dataset_fingerprint_sha256: str = "",
) -> None:
    if scene.gaussian_artifact is not None:
        raise ReconstructionValidationError(
            "Phase 9.4D refuses to overwrite an existing persisted Gaussian appearance artifact."
        )
    verify_persistent_scene_file_references(scene)
    if artifact.optimization_state is not GaussianOptimizationState.OPTIMIZED:
        raise ReconstructionValidationError("Phase 9.4D requires an optimized Gaussian artifact.")
    if artifact.scene_role is not GaussianSceneRole.APPEARANCE_ONLY or artifact.canonical_geometry_authoritative:
        raise ReconstructionValidationError("Persisted Gaussian representation must remain appearance-only.")
    if artifact.certified_metrology:
        raise ReconstructionValidationError("Gaussian persistence cannot certify metrology.")
    if not artifact.render_capable:
        raise ReconstructionValidationError("Persistent Gaussian representation must be render-capable.")
    expected = str(expected_dataset_fingerprint_sha256).strip().lower()
    if expected and artifact.dataset_fingerprint_sha256 != expected:
        raise ReconstructionValidationError("Gaussian dataset fingerprint differs from the qualified input bridge.")
    frame_id = _scene_reconstruction_frame_id(scene)
    if artifact.reconstruction_frame_id != frame_id:
        raise ReconstructionValidationError("Gaussian reconstruction frame differs from the canonical scene frame.")
    surface = _scene_snapshot(scene, ReconstructionArtifactKind.SURFACE_MESH)
    metric_status = str(surface.get("metric_geometry_status", "")).strip()
    if metric_status and artifact.metric_geometry_status.value != metric_status:
        raise ReconstructionValidationError("Gaussian scale state differs from the canonical surface scale state.")
    if artifact.unit_to_metre != surface.get("unit_to_metre"):
        raise ReconstructionValidationError("Gaussian unit_to_metre differs from canonical surface evidence.")


def _provider_use_for_artifact(artifact: GaussianSceneArtifact) -> ReconstructionProviderUse:
    return ReconstructionProviderUse(
        provider_id=PIPELINE_GAUSSIAN_PROVIDER_ID,
        provider_version=artifact.backend_version,
        execution=ReconstructionProviderExecution.EXTERNAL_PROCESS,
        stages=(ReconstructionPipelineStage.GAUSSIAN_APPEARANCE,),
    )


def _pipeline_payload_with_gaussian(
    scene: PersistentReconstructionScene,
    gaussian_record: PersistentSceneArtifactRecord,
    provider_use: ReconstructionProviderUse,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    artifacts = {item.stage: item for item in scene.artifacts}
    artifacts[ReconstructionPipelineStage.GAUSSIAN_APPEARANCE] = gaussian_record
    fallback = {item.stage: item for item in scene.fallback_evidence}
    stages: list[dict[str, Any]] = []
    for stage in ReconstructionPipelineStage:
        artifact = artifacts.get(stage)
        evidence = fallback.get(stage)
        if artifact is None and evidence is None:
            continue
        if artifact is not None:
            attempts = evidence.provider_attempts if evidence is not None else (artifact.provider_id,)
            stages.append(
                {
                    "stage": stage.value,
                    "state": ReconstructionStageRunState.COMPLETED.value,
                    "required": True,
                    "provider_attempts": list(attempts),
                    "provider_id": artifact.provider_id,
                    "fallback_used": bool(evidence.fallback_used) if evidence is not None else False,
                    "reason": evidence.reason if evidence is not None else "",
                    "artifact": artifact.as_dict(),
                }
            )
        else:
            assert evidence is not None
            stages.append(
                {
                    "stage": stage.value,
                    "state": evidence.state.value,
                    "required": True,
                    "provider_attempts": list(evidence.provider_attempts),
                    "provider_id": "",
                    "fallback_used": evidence.fallback_used,
                    "reason": evidence.reason,
                    "artifact": None,
                }
            )
    provenance = list(scene.provider_provenance)
    if any(item.provider_id == provider_use.provider_id for item in provenance):
        raise ReconstructionValidationError("Persistent scene already contains Gaussian provider provenance.")
    provenance.append(provider_use)
    snapshots: dict[str, dict[str, Any]] = {
        item.kind.value: metadata_as_dict(item.payload_snapshot)  # type: ignore[arg-type]
        for item in scene.artifacts
    }
    snapshots[ReconstructionArtifactKind.GAUSSIAN_SCENE.value] = metadata_as_dict(
        gaussian_record.payload_snapshot  # type: ignore[arg-type]
    )
    payload = {
        "request_id": scene.request_id,
        "session_id": scene.session_id,
        "status": scene.pipeline_status.value,
        "canonical_geometry_artifact_id": scene.canonical_geometry_artifact_id,
        "canonical_geometry_authoritative": True,
        "certified_metrology": False,
        "stages": stages,
        "provider_provenance": [item.as_dict() for item in provenance],
    }
    return payload, snapshots


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return path


@dataclass(frozen=True, slots=True)
class GaussianPersistentRepresentationResult:
    state: GaussianPersistenceState
    parent_scene_id: str
    parent_scene_fingerprint_sha256: str
    scene: PersistentReconstructionScene
    scene_manifest_path: str
    artifact: GaussianSceneArtifact
    gaussian_file_references: tuple[PersistentSceneFileReference, ...]
    restart_reload_verified: bool
    source_scene_unchanged: bool = True
    canonical_geometry_authoritative: bool = True
    gaussian_canonical_geometry_authoritative: bool = False
    metric_scale_authoritative: bool = False
    measurement_authoritative: bool = False
    qc_authoritative: bool = False
    certified_metrology: bool = False
    backend_started: bool = False
    schema_version: int = PHASE9_4D_GAUSSIAN_PERSISTENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.scene.gaussian_artifact is None:
            raise ReconstructionValidationError("Persistent result does not contain a Gaussian artifact.")
        if self.scene.canonical_geometry_artifact_id == "":
            raise ReconstructionValidationError("Persistent result lost canonical geometry identity.")
        if self.artifact.canonical_geometry_authoritative or self.artifact.certified_metrology:
            raise ReconstructionValidationError("Gaussian persistence result changed authority.")
        object.__setattr__(self, "source_scene_unchanged", True)
        object.__setattr__(self, "canonical_geometry_authoritative", True)
        object.__setattr__(self, "gaussian_canonical_geometry_authoritative", False)
        object.__setattr__(self, "metric_scale_authoritative", False)
        object.__setattr__(self, "measurement_authoritative", False)
        object.__setattr__(self, "qc_authoritative", False)
        object.__setattr__(self, "certified_metrology", False)
        object.__setattr__(self, "backend_started", False)
        object.__setattr__(self, "schema_version", PHASE9_4D_GAUSSIAN_PERSISTENCE_SCHEMA_VERSION)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "phase": PHASE9_4D_GAUSSIAN_PERSISTENCE_PHASE,
            "state": self.state.value,
            "parent_scene_id": self.parent_scene_id,
            "parent_scene_fingerprint_sha256": self.parent_scene_fingerprint_sha256,
            "persistent_scene_id": self.scene.scene_id,
            "persistent_scene_fingerprint_sha256": self.scene.scene_fingerprint_sha256,
            "scene_manifest_path": self.scene_manifest_path,
            "gaussian_artifact_id": self.scene.gaussian_artifact.artifact_id if self.scene.gaussian_artifact else "",
            "gaussian_scene_path": self.artifact.scene_path,
            "gaussian_backend_id": self.artifact.backend_id,
            "gaussian_backend_version": self.artifact.backend_version,
            "gaussian_count": self.artifact.gaussian_count,
            "training_steps": self.artifact.training_steps,
            "dataset_fingerprint_sha256": self.artifact.dataset_fingerprint_sha256,
            "gaussian_file_reference_count": len(self.gaussian_file_references),
            "gaussian_file_references": [item.as_dict() for item in self.gaussian_file_references],
            "restart_reload_verified": self.restart_reload_verified,
            "source_scene_unchanged": self.source_scene_unchanged,
            "canonical_geometry_authoritative": self.canonical_geometry_authoritative,
            "gaussian_scene_role": self.artifact.scene_role.value,
            "gaussian_canonical_geometry_authoritative": self.gaussian_canonical_geometry_authoritative,
            "metric_scale_authoritative": self.metric_scale_authoritative,
            "measurement_authoritative": self.measurement_authoritative,
            "qc_authoritative": self.qc_authoritative,
            "certified_metrology": self.certified_metrology,
            "metric_geometry_status": self.artifact.metric_geometry_status.value,
            "unit_to_metre": self.artifact.unit_to_metre,
            "backend_started": self.backend_started,
        }


def persist_gaussian_representation(
    *,
    base_scene: PersistentReconstructionScene,
    gaussian_artifact: GaussianSceneArtifact,
    preview_path: str | Path,
    qualification_metrics_path: str | Path,
    source_artifact_manifest_path: str | Path,
    output_dir: str | Path,
    source_optimization_summary_path: str | Path = "",
    expected_dataset_fingerprint_sha256: str = "",
) -> GaussianPersistentRepresentationResult:
    """Create a durable persistent-scene revision containing Gaussian appearance."""

    if not isinstance(base_scene, PersistentReconstructionScene):
        raise TypeError("base_scene must be PersistentReconstructionScene.")
    if not isinstance(gaussian_artifact, GaussianSceneArtifact):
        raise TypeError("gaussian_artifact must be GaussianSceneArtifact.")
    _validate_scene_artifact_compatibility(
        base_scene,
        gaussian_artifact,
        expected_dataset_fingerprint_sha256=expected_dataset_fingerprint_sha256,
    )
    root = Path(output_dir).expanduser()
    if not root.is_absolute():
        raise ReconstructionValidationError("Phase 9.4D output_dir must be absolute.")
    root = root.resolve()
    gaussian_dir = (root / "appearance" / "gaussian").resolve()
    gaussian_dir.mkdir(parents=True, exist_ok=True)

    source_manifest = Path(source_artifact_manifest_path).expanduser().resolve()
    if not source_manifest.is_file():
        raise ReconstructionValidationError(f"Gaussian source artifact manifest is missing: {source_manifest}")
    try:
        source_manifest_payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError("Gaussian source artifact manifest is invalid JSON.") from exc
    if not isinstance(source_manifest_payload, Mapping):
        raise ReconstructionValidationError("Gaussian source artifact manifest must be a JSON object.")
    source_manifest_artifact = GaussianSceneArtifact.from_payload(source_manifest_payload)
    if source_manifest_artifact != gaussian_artifact:
        raise ReconstructionValidationError("Gaussian source artifact manifest does not match qualified 9.4C evidence.")
    metrics_source = Path(qualification_metrics_path).expanduser().resolve()
    if not metrics_source.is_file():
        raise ReconstructionValidationError(f"Gaussian qualification metrics are missing: {metrics_source}")
    try:
        metrics_payload = json.loads(metrics_source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError("Gaussian qualification metrics are invalid JSON.") from exc
    if not isinstance(metrics_payload, Mapping):
        raise ReconstructionValidationError("Gaussian qualification metrics must be a JSON object.")
    if metrics_payload.get("real_cuda_execution") is not True or metrics_payload.get("optimization_completed") is not True:
        raise ReconstructionValidationError("Gaussian persistence requires completed real CUDA optimization evidence.")
    if metrics_payload.get("render_written") is not True:
        raise ReconstructionValidationError("Gaussian persistence requires a qualified preview render.")

    durable_scene_path = _copy_verified(gaussian_artifact.scene_path, gaussian_dir / "scene.pt", label="Gaussian scene")
    durable_preview_path = _copy_verified(preview_path, gaussian_dir / "preview.png", label="Gaussian preview")
    source_metrics_copy = _copy_verified(
        metrics_source,
        gaussian_dir / "source_qualification_metrics.json",
        label="Gaussian qualification metrics",
    )
    durable_metrics_payload = dict(metrics_payload)
    durable_metrics_payload["preview_path"] = str(durable_preview_path)
    durable_metrics_payload["source_qualification_metrics_sha256"] = _sha256_file(source_metrics_copy)
    durable_metrics_path = _write_json(gaussian_dir / "qualification_metrics.json", durable_metrics_payload)

    durable_artifact = GaussianSceneArtifact(
        request_id=gaussian_artifact.request_id,
        backend_id=gaussian_artifact.backend_id,
        backend_version=gaussian_artifact.backend_version,
        backend_provenance=gaussian_artifact.backend_provenance,
        optimization_state=gaussian_artifact.optimization_state,
        scene_role=gaussian_artifact.scene_role,
        artifact_dir=str(gaussian_dir),
        scene_path=str(durable_scene_path),
        dataset_fingerprint_sha256=gaussian_artifact.dataset_fingerprint_sha256,
        input_frame_count=gaussian_artifact.input_frame_count,
        gaussian_count=gaussian_artifact.gaussian_count,
        training_steps=gaussian_artifact.training_steps,
        training_duration_s=gaussian_artifact.training_duration_s,
        render_capable=gaussian_artifact.render_capable,
        device=gaussian_artifact.device,
        pose_convention=gaussian_artifact.pose_convention,
        reconstruction_frame_id=gaussian_artifact.reconstruction_frame_id,
        metric_geometry_status=gaussian_artifact.metric_geometry_status,
        unit_to_metre=gaussian_artifact.unit_to_metre,
        relative_scale_uncertainty=gaussian_artifact.relative_scale_uncertainty,
        peak_vram_mib=gaussian_artifact.peak_vram_mib,
    )
    durable_manifest_path = _write_json(gaussian_dir / "artifact_manifest.json", durable_artifact.as_dict())
    source_manifest_copy = _copy_verified(
        source_manifest,
        gaussian_dir / "source_artifact_manifest.json",
        label="Gaussian source artifact manifest",
    )
    optimization_summary_copy: Path | None = None
    if str(source_optimization_summary_path).strip():
        optimization_summary_copy = _copy_verified(
            source_optimization_summary_path,
            gaussian_dir / "source_optimization_summary.json",
            label="Phase 9.4C optimization summary",
        )

    snapshot = durable_artifact.as_dict()
    snapshot.update(
        {
            "preview_path": str(durable_preview_path),
            "artifact_manifest_path": str(durable_manifest_path),
            "qualification_metrics_path": str(durable_metrics_path),
            "source_qualification_metrics_path": str(source_metrics_copy),
            "source_artifact_manifest_path": str(source_manifest_copy),
            "source_optimization_summary_path": str(optimization_summary_copy) if optimization_summary_copy else "",
            "parent_scene_id": base_scene.scene_id,
            "parent_scene_fingerprint_sha256": base_scene.scene_fingerprint_sha256,
            "canonical_geometry_artifact_id": base_scene.canonical_geometry_artifact_id,
            "canonical_geometry_authoritative": False,
            "metric_scale_authoritative": False,
            "measurement_authoritative": False,
            "qc_authoritative": False,
            "certified_metrology": False,
        }
    )
    record = PersistentSceneArtifactRecord(
        artifact_id=f"{base_scene.request_id}:gaussian_appearance:{PIPELINE_GAUSSIAN_PROVIDER_ID}",
        stage=ReconstructionPipelineStage.GAUSSIAN_APPEARANCE,
        kind=ReconstructionArtifactKind.GAUSSIAN_SCENE,
        authority=ReconstructionArtifactAuthority.APPEARANCE_ONLY,
        provider_id=PIPELINE_GAUSSIAN_PROVIDER_ID,
        payload_ref=str(durable_manifest_path),
        metadata={
            "backend_id": durable_artifact.backend_id,
            "backend_version": durable_artifact.backend_version,
            "dataset_fingerprint_sha256": durable_artifact.dataset_fingerprint_sha256,
            "render_capable": durable_artifact.render_capable,
            "scene_role": durable_artifact.scene_role.value,
            "parent_scene_id": base_scene.scene_id,
            "parent_scene_fingerprint_sha256": base_scene.scene_fingerprint_sha256,
            "canonical_surface_artifact_id": base_scene.canonical_geometry_artifact_id,
            "canonical_geometry_authoritative": False,
            "certified_metrology": False,
            "phase": PHASE9_4D_GAUSSIAN_PERSISTENCE_PHASE,
        },
        payload_snapshot=snapshot,
    )
    provider = _provider_use_for_artifact(durable_artifact)
    payload, snapshots = _pipeline_payload_with_gaussian(base_scene, record, provider)
    persisted = persistent_scene_from_pipeline_payload(payload, payload_snapshots_by_kind=snapshots)
    if persisted.canonical_geometry_artifact_id != base_scene.canonical_geometry_artifact_id:
        raise ReconstructionValidationError("Gaussian persistence changed canonical geometry artifact identity.")
    base_canonical = next(item for item in base_scene.artifacts if item.artifact_id == base_scene.canonical_geometry_artifact_id)
    new_canonical = next(item for item in persisted.artifacts if item.artifact_id == persisted.canonical_geometry_artifact_id)
    if base_canonical != new_canonical:
        raise ReconstructionValidationError("Gaussian persistence modified canonical geometry evidence.")
    manifest = write_persistent_reconstruction_scene(persisted, root)
    loaded = read_persistent_reconstruction_scene(manifest)
    verify_persistent_scene_file_references(loaded)
    gaussian_refs = tuple(item for item in loaded.file_references if item.artifact_id == record.artifact_id)
    if len(gaussian_refs) < 5:
        raise ReconstructionValidationError("Persistent Gaussian representation lacks complete file-integrity evidence.")
    durable_manifest_payload = json.loads(durable_manifest_path.read_text(encoding="utf-8"))
    durable_loaded_artifact = GaussianSceneArtifact.from_payload(durable_manifest_payload)
    if durable_loaded_artifact != durable_artifact:
        raise ReconstructionValidationError("Durable Gaussian artifact manifest does not round-trip exactly.")
    return GaussianPersistentRepresentationResult(
        state=GaussianPersistenceState.PERSISTED,
        parent_scene_id=base_scene.scene_id,
        parent_scene_fingerprint_sha256=base_scene.scene_fingerprint_sha256,
        scene=loaded,
        scene_manifest_path=str(manifest.resolve()),
        artifact=durable_loaded_artifact,
        gaussian_file_references=gaussian_refs,
        restart_reload_verified=True,
    )


def reload_persistent_gaussian_representation(
    scene_manifest_path: str | Path,
) -> GaussianPersistentRepresentationResult:
    """Reload a persisted Gaussian scene in a fresh process without backend execution."""

    manifest = Path(scene_manifest_path).expanduser().resolve()
    scene = read_persistent_reconstruction_scene(manifest)
    verify_persistent_scene_file_references(scene)
    record = scene.gaussian_artifact
    if record is None:
        raise ReconstructionValidationError("Persistent scene does not contain Gaussian appearance.")
    snapshot = metadata_as_dict(record.payload_snapshot)  # type: ignore[arg-type]
    artifact_manifest = Path(str(snapshot.get("artifact_manifest_path", ""))).expanduser().resolve()
    if not artifact_manifest.is_file():
        raise ReconstructionValidationError("Persisted Gaussian artifact manifest is unavailable.")
    payload = json.loads(artifact_manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ReconstructionValidationError("Persisted Gaussian artifact manifest must be a JSON object.")
    artifact = GaussianSceneArtifact.from_payload(payload)
    _validate_scene_artifact_compatibility_for_reload(scene, artifact)
    refs = tuple(item for item in scene.file_references if item.artifact_id == record.artifact_id)
    if not refs:
        raise ReconstructionValidationError("Persistent Gaussian appearance has no integrity references.")
    parent_scene_id = str(snapshot.get("parent_scene_id", "")).strip()
    parent_fingerprint = str(snapshot.get("parent_scene_fingerprint_sha256", "")).strip()
    if not parent_scene_id or len(parent_fingerprint) != 64:
        raise ReconstructionValidationError("Persistent Gaussian lineage metadata is incomplete.")
    return GaussianPersistentRepresentationResult(
        state=GaussianPersistenceState.RELOADED,
        parent_scene_id=parent_scene_id,
        parent_scene_fingerprint_sha256=parent_fingerprint,
        scene=scene,
        scene_manifest_path=str(manifest),
        artifact=artifact,
        gaussian_file_references=refs,
        restart_reload_verified=True,
    )


def _validate_scene_artifact_compatibility_for_reload(
    scene: PersistentReconstructionScene,
    artifact: GaussianSceneArtifact,
) -> None:
    if scene.gaussian_artifact is None:
        raise ReconstructionValidationError("Reloaded scene lost Gaussian appearance artifact.")
    if artifact.scene_role is not GaussianSceneRole.APPEARANCE_ONLY or artifact.canonical_geometry_authoritative:
        raise ReconstructionValidationError("Reloaded Gaussian representation changed authority.")
    if artifact.certified_metrology:
        raise ReconstructionValidationError("Reloaded Gaussian representation claims certified metrology.")
    if artifact.reconstruction_frame_id != _scene_reconstruction_frame_id(scene):
        raise ReconstructionValidationError("Reloaded Gaussian reconstruction frame differs from canonical scene.")
    surface = _scene_snapshot(scene, ReconstructionArtifactKind.SURFACE_MESH)
    if str(surface.get("metric_geometry_status", "")).strip() != artifact.metric_geometry_status.value:
        raise ReconstructionValidationError("Reloaded Gaussian scale state differs from canonical surface.")
    if artifact.unit_to_metre != surface.get("unit_to_metre"):
        raise ReconstructionValidationError("Reloaded Gaussian unit_to_metre differs from canonical surface.")


__all__ = [
    "PHASE9_4D_GAUSSIAN_PERSISTENCE_SCHEMA_VERSION",
    "PHASE9_4D_GAUSSIAN_PERSISTENCE_PHASE",
    "GaussianPersistenceState",
    "GaussianPersistentRepresentationResult",
    "persist_gaussian_representation",
    "reload_persistent_gaussian_representation",
]

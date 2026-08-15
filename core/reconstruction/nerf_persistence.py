"""Phase 9.5D persistent NeRF appearance representation.

This module attaches one already-trained/render-capable NeRF appearance artifact
from Phase 9.5C to the existing ``PersistentReconstructionScene`` schema.  It
never starts Nerfstudio/tiny-cuda-nn, never changes canonical geometry authority,
and creates a new immutable persistent-scene revision with integrity-bound NeRF
checkpoint/config/preview/runtime evidence and a self-owned copy of the input
NeRF dataset.

The persisted training config is copied byte-for-byte as provenance.  Because a
Nerfstudio config can contain source-time absolute paths, ``runtime_load_spec``
records the durable checkpoint/dataset paths that a later isolated interactive
renderer must explicitly rebind during staging.  Phase 9.5D itself performs no
backend load or rendering.
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
from core.reconstruction.nerf import NerfSceneArtifact, NerfSceneRole, NerfTrainingState
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
from core.reconstruction.pipeline_appearance import PIPELINE_NERF_PROVIDER_ID

PHASE9_5D_NERF_PERSISTENCE_SCHEMA_VERSION = 1
PHASE9_5D_NERF_PERSISTENCE_PHASE = "9.5D"


class NerfPersistenceState(str, Enum):
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


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return path


def _load_json(path: str | Path, *, label: str) -> dict[str, Any]:
    target = Path(path).expanduser().resolve()
    if not target.is_file() or target.stat().st_size <= 0:
        raise ReconstructionValidationError(f"{label} is missing or empty: {target}")
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError(f"{label} is invalid JSON.") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"{label} must be a JSON object.")
    return payload


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
    artifact: NerfSceneArtifact,
    *,
    expected_dataset_fingerprint_sha256: str = "",
) -> None:
    if scene.nerf_artifact is not None:
        raise ReconstructionValidationError(
            "Phase 9.5D refuses to overwrite an existing persisted NeRF appearance artifact."
        )
    verify_persistent_scene_file_references(scene)
    if artifact.training_state is not NerfTrainingState.TRAINED:
        raise ReconstructionValidationError("Phase 9.5D requires a trained NeRF artifact.")
    if artifact.scene_role is not NerfSceneRole.APPEARANCE_ONLY or artifact.canonical_geometry_authoritative:
        raise ReconstructionValidationError("Persisted NeRF representation must remain appearance-only.")
    if artifact.certified_metrology:
        raise ReconstructionValidationError("NeRF persistence cannot certify metrology.")
    if not artifact.render_capable:
        raise ReconstructionValidationError("Persistent NeRF representation must be render-capable.")
    expected = str(expected_dataset_fingerprint_sha256).strip().lower()
    if expected and artifact.dataset_fingerprint_sha256 != expected:
        raise ReconstructionValidationError("NeRF dataset fingerprint differs from the qualified input bridge.")
    if artifact.reconstruction_frame_id != _scene_reconstruction_frame_id(scene):
        raise ReconstructionValidationError("NeRF reconstruction frame differs from the canonical scene frame.")
    surface = _scene_snapshot(scene, ReconstructionArtifactKind.SURFACE_MESH)
    metric_status = str(surface.get("metric_geometry_status", "")).strip()
    if metric_status and artifact.metric_geometry_status.value != metric_status:
        raise ReconstructionValidationError("NeRF scale state differs from the canonical surface scale state.")
    if artifact.unit_to_metre != surface.get("unit_to_metre"):
        raise ReconstructionValidationError("NeRF unit_to_metre differs from canonical surface evidence.")


def _provider_use_for_artifact(artifact: NerfSceneArtifact) -> ReconstructionProviderUse:
    return ReconstructionProviderUse(
        provider_id=PIPELINE_NERF_PROVIDER_ID,
        provider_version=artifact.backend_version,
        execution=ReconstructionProviderExecution.EXTERNAL_PROCESS,
        stages=(ReconstructionPipelineStage.NERF_APPEARANCE,),
    )


def _pipeline_payload_with_nerf(
    scene: PersistentReconstructionScene,
    nerf_record: PersistentSceneArtifactRecord,
    provider_use: ReconstructionProviderUse,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    artifacts = {item.stage: item for item in scene.artifacts}
    artifacts[ReconstructionPipelineStage.NERF_APPEARANCE] = nerf_record
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
        raise ReconstructionValidationError("Persistent scene already contains NeRF provider provenance.")
    provenance.append(provider_use)
    snapshots: dict[str, dict[str, Any]] = {
        item.kind.value: metadata_as_dict(item.payload_snapshot)  # type: ignore[arg-type]
        for item in scene.artifacts
    }
    snapshots[ReconstructionArtifactKind.NERF_SCENE.value] = metadata_as_dict(
        nerf_record.payload_snapshot  # type: ignore[arg-type]
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


def _copy_dataset_tree(source_root: Path, target_root: Path) -> Path:
    if not source_root.is_dir():
        raise ReconstructionValidationError(f"NeRF source dataset directory is missing: {source_root}")
    if target_root.exists():
        shutil.rmtree(target_root)
    shutil.copytree(source_root, target_root)
    files = sorted(path for path in target_root.rglob("*") if path.is_file())
    if not files:
        raise ReconstructionValidationError("Persisted NeRF dataset is empty.")
    return target_root


def _dataset_integrity_payload(dataset_root: Path) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for path in sorted(item for item in dataset_root.rglob("*") if item.is_file()):
        entries.append(
            {
                "relative_path": path.relative_to(dataset_root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    if not entries:
        raise ReconstructionValidationError("Persisted NeRF dataset has no files to integrity-bind.")
    return {"schema_version": 1, "file_count": len(entries), "files": entries}


def _verify_dataset_integrity(dataset_root: Path, integrity_manifest: Path) -> None:
    payload = _load_json(integrity_manifest, label="NeRF dataset integrity manifest")
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise ReconstructionValidationError("NeRF dataset integrity manifest contains no files.")
    for entry in files:
        if not isinstance(entry, Mapping):
            raise ReconstructionValidationError("NeRF dataset integrity entry is invalid.")
        relative = str(entry.get("relative_path", "")).strip()
        if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ReconstructionValidationError("NeRF dataset integrity path is unsafe.")
        path = (dataset_root / relative).resolve()
        try:
            path.relative_to(dataset_root.resolve())
        except ValueError as exc:
            raise ReconstructionValidationError("NeRF dataset integrity path escapes dataset root.") from exc
        if not path.is_file():
            raise ReconstructionValidationError(f"Persisted NeRF dataset file is missing: {path}")
        if path.stat().st_size != int(entry.get("size_bytes", -1)):
            raise ReconstructionValidationError(f"Persisted NeRF dataset file size changed: {path}")
        if _sha256_file(path) != str(entry.get("sha256", "")):
            raise ReconstructionValidationError(f"Persisted NeRF dataset file checksum changed: {path}")


@dataclass(frozen=True, slots=True)
class NerfPersistentRepresentationResult:
    state: NerfPersistenceState
    parent_scene_id: str
    parent_scene_fingerprint_sha256: str
    scene: PersistentReconstructionScene
    scene_manifest_path: str
    artifact: NerfSceneArtifact
    nerf_file_references: tuple[PersistentSceneFileReference, ...]
    dataset_root: str
    dataset_integrity_manifest_path: str
    runtime_load_spec_path: str
    restart_reload_verified: bool
    source_scene_unchanged: bool = True
    canonical_geometry_authoritative: bool = True
    nerf_canonical_geometry_authoritative: bool = False
    metric_scale_authoritative: bool = False
    measurement_authoritative: bool = False
    qc_authoritative: bool = False
    certified_metrology: bool = False
    backend_started: bool = False
    rendering_started: bool = False
    schema_version: int = PHASE9_5D_NERF_PERSISTENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.scene.nerf_artifact is None:
            raise ReconstructionValidationError("Persistent result does not contain a NeRF artifact.")
        if not self.scene.canonical_geometry_artifact_id:
            raise ReconstructionValidationError("Persistent result lost canonical geometry identity.")
        if self.artifact.canonical_geometry_authoritative or self.artifact.certified_metrology:
            raise ReconstructionValidationError("NeRF persistence result changed authority.")
        object.__setattr__(self, "source_scene_unchanged", True)
        object.__setattr__(self, "canonical_geometry_authoritative", True)
        object.__setattr__(self, "nerf_canonical_geometry_authoritative", False)
        object.__setattr__(self, "metric_scale_authoritative", False)
        object.__setattr__(self, "measurement_authoritative", False)
        object.__setattr__(self, "qc_authoritative", False)
        object.__setattr__(self, "certified_metrology", False)
        object.__setattr__(self, "backend_started", False)
        object.__setattr__(self, "rendering_started", False)
        object.__setattr__(self, "schema_version", PHASE9_5D_NERF_PERSISTENCE_SCHEMA_VERSION)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "phase": PHASE9_5D_NERF_PERSISTENCE_PHASE,
            "state": self.state.value,
            "parent_scene_id": self.parent_scene_id,
            "parent_scene_fingerprint_sha256": self.parent_scene_fingerprint_sha256,
            "persistent_scene_id": self.scene.scene_id,
            "persistent_scene_fingerprint_sha256": self.scene.scene_fingerprint_sha256,
            "scene_manifest_path": self.scene_manifest_path,
            "nerf_artifact_id": self.scene.nerf_artifact.artifact_id if self.scene.nerf_artifact else "",
            "nerf_artifact_dir": self.artifact.artifact_dir,
            "nerf_config_path": self.artifact.config_path,
            "nerf_checkpoint_path": self.artifact.checkpoint_path,
            "nerf_backend_id": self.artifact.backend_id,
            "nerf_backend_version": self.artifact.backend_version,
            "nerf_backend_revision": self.artifact.backend_revision,
            "method_name": self.artifact.method_name,
            "training_steps": self.artifact.training_steps,
            "dataset_fingerprint_sha256": self.artifact.dataset_fingerprint_sha256,
            "dataset_root": self.dataset_root,
            "dataset_integrity_manifest_path": self.dataset_integrity_manifest_path,
            "runtime_load_spec_path": self.runtime_load_spec_path,
            "nerf_file_reference_count": len(self.nerf_file_references),
            "nerf_file_references": [item.as_dict() for item in self.nerf_file_references],
            "restart_reload_verified": self.restart_reload_verified,
            "source_scene_unchanged": self.source_scene_unchanged,
            "canonical_geometry_authoritative": self.canonical_geometry_authoritative,
            "nerf_scene_role": self.artifact.scene_role.value,
            "nerf_canonical_geometry_authoritative": self.nerf_canonical_geometry_authoritative,
            "metric_scale_authoritative": self.metric_scale_authoritative,
            "measurement_authoritative": self.measurement_authoritative,
            "qc_authoritative": self.qc_authoritative,
            "certified_metrology": self.certified_metrology,
            "metric_geometry_status": self.artifact.metric_geometry_status.value,
            "unit_to_metre": self.artifact.unit_to_metre,
            "backend_started": self.backend_started,
            "rendering_started": self.rendering_started,
        }


def persist_nerf_representation(
    *,
    base_scene: PersistentReconstructionScene,
    nerf_artifact: NerfSceneArtifact,
    preview_path: str | Path,
    runtime_evidence_path: str | Path,
    source_optimization_manifest_path: str | Path,
    output_dir: str | Path,
    source_optimization_summary_path: str | Path = "",
    source_worker_console_path: str | Path = "",
    expected_dataset_fingerprint_sha256: str = "",
) -> NerfPersistentRepresentationResult:
    """Create a durable persistent-scene revision containing NeRF appearance."""

    if not isinstance(base_scene, PersistentReconstructionScene):
        raise TypeError("base_scene must be PersistentReconstructionScene.")
    if not isinstance(nerf_artifact, NerfSceneArtifact):
        raise TypeError("nerf_artifact must be NerfSceneArtifact.")
    _validate_scene_artifact_compatibility(
        base_scene,
        nerf_artifact,
        expected_dataset_fingerprint_sha256=expected_dataset_fingerprint_sha256,
    )

    root = Path(output_dir).expanduser()
    if not root.is_absolute():
        raise ReconstructionValidationError("Phase 9.5D output_dir must be absolute.")
    root = root.resolve()
    nerf_dir = (root / "appearance" / "nerf").resolve()
    nerf_dir.mkdir(parents=True, exist_ok=True)

    source_artifact_root = Path(nerf_artifact.artifact_dir).expanduser().resolve()
    if not source_artifact_root.is_dir():
        raise ReconstructionValidationError(f"NeRF source artifact directory is missing: {source_artifact_root}")
    source_dataset_root = (source_artifact_root / "dataset").resolve()
    dataset_root = _copy_dataset_tree(source_dataset_root, (nerf_dir / "dataset").resolve())
    transforms_path = (dataset_root / "transforms.json").resolve()
    if not transforms_path.is_file():
        raise ReconstructionValidationError("Persisted NeRF dataset lacks transforms.json.")
    dataset_integrity_path = _write_json(
        nerf_dir / "dataset_integrity.json",
        _dataset_integrity_payload(dataset_root),
    )
    _verify_dataset_integrity(dataset_root, dataset_integrity_path)

    durable_config = _copy_verified(nerf_artifact.config_path, nerf_dir / "config.yml", label="Nerfstudio config")
    durable_checkpoint = _copy_verified(
        nerf_artifact.checkpoint_path,
        nerf_dir / "checkpoint" / Path(nerf_artifact.checkpoint_path).name,
        label="Nerfstudio checkpoint",
    )
    durable_preview = _copy_verified(preview_path, nerf_dir / "preview.png", label="NeRF preview")
    source_runtime_copy = _copy_verified(
        runtime_evidence_path,
        nerf_dir / "source_runtime_evidence.json",
        label="NeRF runtime evidence",
    )
    runtime_payload = _load_json(source_runtime_copy, label="NeRF source runtime evidence")
    if runtime_payload.get("training_completed_without_oom") is not True:
        raise ReconstructionValidationError("NeRF persistence requires OOM-free training evidence.")
    runtime_payload.update(
        {
            "preview_path": str(durable_preview),
            "config_path": str(durable_config),
            "checkpoint_path": str(durable_checkpoint),
            "persisted_dataset_root": str(dataset_root),
            "source_runtime_evidence_sha256": _sha256_file(source_runtime_copy),
            "phase9_5d_backend_started": False,
            "phase9_5d_rendering_started": False,
        }
    )
    durable_runtime_evidence = _write_json(nerf_dir / "e2_runtime_evidence.json", runtime_payload)

    optimization_manifest_source = Path(source_optimization_manifest_path).expanduser().resolve()
    optimization_manifest = _load_json(optimization_manifest_source, label="Phase 9.5C optimization manifest")
    if optimization_manifest.get("phase") != "9.5C":
        raise ReconstructionValidationError("NeRF persistence requires a Phase 9.5C optimization manifest.")
    if optimization_manifest.get("dataset_fingerprint_sha256") != nerf_artifact.dataset_fingerprint_sha256:
        raise ReconstructionValidationError("NeRF optimization manifest dataset fingerprint mismatch.")
    for field, source_path in (
        ("config_sha256", Path(nerf_artifact.config_path)),
        ("checkpoint_sha256", Path(nerf_artifact.checkpoint_path)),
        ("preview_sha256", Path(preview_path)),
        ("runtime_evidence_sha256", Path(runtime_evidence_path)),
    ):
        expected_hash = str(optimization_manifest.get(field, ""))
        if expected_hash != _sha256_file(source_path.expanduser().resolve()):
            raise ReconstructionValidationError(f"NeRF optimization manifest {field} mismatch.")
    if optimization_manifest.get("render_capable") is not True or not str(nerf_artifact.device).lower().startswith("cuda:"):
        raise ReconstructionValidationError("NeRF persistence requires real CUDA render-capable 9.5C evidence.")
    if any(bool(optimization_manifest.get(key, False)) for key in (
        "canonical_geometry_authoritative", "metric_scale_authoritative", "measurement_authoritative",
        "qc_authoritative", "certified_metrology",
    )):
        raise ReconstructionValidationError("NeRF optimization manifest claims forbidden authority.")
    source_optimization_manifest_copy = _copy_verified(
        optimization_manifest_source,
        nerf_dir / "source_optimization_manifest.json",
        label="Phase 9.5C optimization manifest",
    )

    source_summary_copy: Path | None = None
    if str(source_optimization_summary_path).strip():
        source_summary_copy = _copy_verified(
            source_optimization_summary_path,
            nerf_dir / "source_optimization_summary.json",
            label="Phase 9.5C optimization summary",
        )
    source_console_copy: Path | None = None
    if str(source_worker_console_path).strip():
        source_console_copy = _copy_verified(
            source_worker_console_path,
            nerf_dir / "source_worker_console.log",
            label="Phase 9.5C worker console",
        )

    durable_artifact = NerfSceneArtifact(
        request_id=nerf_artifact.request_id,
        backend_id=nerf_artifact.backend_id,
        backend_version=nerf_artifact.backend_version,
        backend_revision=nerf_artifact.backend_revision,
        backend_provenance=nerf_artifact.backend_provenance,
        method_name=nerf_artifact.method_name,
        training_state=nerf_artifact.training_state,
        scene_role=nerf_artifact.scene_role,
        artifact_dir=str(nerf_dir),
        config_path=str(durable_config),
        checkpoint_path=str(durable_checkpoint),
        dataset_fingerprint_sha256=nerf_artifact.dataset_fingerprint_sha256,
        input_frame_count=nerf_artifact.input_frame_count,
        training_steps=nerf_artifact.training_steps,
        training_duration_s=nerf_artifact.training_duration_s,
        render_capable=nerf_artifact.render_capable,
        device=nerf_artifact.device,
        pose_convention=nerf_artifact.pose_convention,
        reconstruction_frame_id=nerf_artifact.reconstruction_frame_id,
        metric_geometry_status=nerf_artifact.metric_geometry_status,
        unit_to_metre=nerf_artifact.unit_to_metre,
        relative_scale_uncertainty=nerf_artifact.relative_scale_uncertainty,
        canonical_geometry_authoritative=False,
        certified_metrology=False,
    )
    durable_manifest_path = _write_json(nerf_dir / "artifact_manifest.json", durable_artifact.as_dict())
    runtime_load_spec = {
        "schema_version": 1,
        "phase": PHASE9_5D_NERF_PERSISTENCE_PHASE,
        "backend_kind": "nerfstudio_nerfacto_external",
        "method_name": durable_artifact.method_name,
        "immutable_training_config_path": str(durable_config),
        "checkpoint_path": str(durable_checkpoint),
        "dataset_root": str(dataset_root),
        "transforms_path": str(transforms_path),
        "dataset_integrity_manifest_path": str(dataset_integrity_path),
        "dataset_fingerprint_sha256": durable_artifact.dataset_fingerprint_sha256,
        "source_artifact_dir": str(source_artifact_root),
        "load_strategy": "isolated_staged_rebind_from_immutable_training_config",
        "backend_start_performed": False,
        "rendering_performed": False,
        "scene_role": "appearance_only",
        "canonical_geometry_authoritative": False,
        "metric_scale_authoritative": False,
        "measurement_authoritative": False,
        "qc_authoritative": False,
        "certified_metrology": False,
        "interactive_renderer_deferred_to": "9.5E",
    }
    runtime_load_spec_path = _write_json(nerf_dir / "runtime_load_spec.json", runtime_load_spec)

    snapshot = durable_artifact.as_dict()
    snapshot.update(
        {
            "preview_path": str(durable_preview),
            "artifact_manifest_path": str(durable_manifest_path),
            "runtime_evidence_path": str(durable_runtime_evidence),
            "source_runtime_evidence_path": str(source_runtime_copy),
            "runtime_load_spec_path": str(runtime_load_spec_path),
            "dataset_integrity_manifest_path": str(dataset_integrity_path),
            "dataset_transforms_path": str(transforms_path),
            "source_optimization_manifest_path": str(source_optimization_manifest_copy),
            "source_optimization_summary_path": str(source_summary_copy) if source_summary_copy else "",
            "source_worker_console_path": str(source_console_copy) if source_console_copy else "",
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
        artifact_id=f"{base_scene.request_id}:nerf_appearance:{PIPELINE_NERF_PROVIDER_ID}",
        stage=ReconstructionPipelineStage.NERF_APPEARANCE,
        kind=ReconstructionArtifactKind.NERF_SCENE,
        authority=ReconstructionArtifactAuthority.APPEARANCE_ONLY,
        provider_id=PIPELINE_NERF_PROVIDER_ID,
        payload_ref=str(durable_manifest_path),
        metadata={
            "backend_id": durable_artifact.backend_id,
            "backend_version": durable_artifact.backend_version,
            "backend_revision": durable_artifact.backend_revision,
            "method_name": durable_artifact.method_name,
            "dataset_fingerprint_sha256": durable_artifact.dataset_fingerprint_sha256,
            "render_capable": durable_artifact.render_capable,
            "scene_role": durable_artifact.scene_role.value,
            "parent_scene_id": base_scene.scene_id,
            "parent_scene_fingerprint_sha256": base_scene.scene_fingerprint_sha256,
            "canonical_surface_artifact_id": base_scene.canonical_geometry_artifact_id,
            "canonical_geometry_authoritative": False,
            "certified_metrology": False,
            "phase": PHASE9_5D_NERF_PERSISTENCE_PHASE,
        },
        payload_snapshot=snapshot,
    )
    provider = _provider_use_for_artifact(durable_artifact)
    payload, snapshots = _pipeline_payload_with_nerf(base_scene, record, provider)
    persisted = persistent_scene_from_pipeline_payload(payload, payload_snapshots_by_kind=snapshots)
    if persisted.canonical_geometry_artifact_id != base_scene.canonical_geometry_artifact_id:
        raise ReconstructionValidationError("NeRF persistence changed canonical geometry artifact identity.")
    base_canonical = next(item for item in base_scene.artifacts if item.artifact_id == base_scene.canonical_geometry_artifact_id)
    new_canonical = next(item for item in persisted.artifacts if item.artifact_id == persisted.canonical_geometry_artifact_id)
    if base_canonical != new_canonical:
        raise ReconstructionValidationError("NeRF persistence modified canonical geometry evidence.")

    manifest = write_persistent_reconstruction_scene(persisted, root)
    loaded = read_persistent_reconstruction_scene(manifest)
    verify_persistent_scene_file_references(loaded)
    _verify_dataset_integrity(dataset_root, dataset_integrity_path)
    nerf_refs = tuple(item for item in loaded.file_references if item.artifact_id == record.artifact_id)
    if len(nerf_refs) < 8:
        raise ReconstructionValidationError("Persistent NeRF representation lacks complete file-integrity evidence.")
    durable_loaded_artifact = NerfSceneArtifact.from_payload(_load_json(durable_manifest_path, label="NeRF artifact manifest"))
    if durable_loaded_artifact != durable_artifact:
        raise ReconstructionValidationError("Durable NeRF artifact manifest does not round-trip exactly.")
    return NerfPersistentRepresentationResult(
        state=NerfPersistenceState.PERSISTED,
        parent_scene_id=base_scene.scene_id,
        parent_scene_fingerprint_sha256=base_scene.scene_fingerprint_sha256,
        scene=loaded,
        scene_manifest_path=str(manifest.resolve()),
        artifact=durable_loaded_artifact,
        nerf_file_references=nerf_refs,
        dataset_root=str(dataset_root),
        dataset_integrity_manifest_path=str(dataset_integrity_path),
        runtime_load_spec_path=str(runtime_load_spec_path),
        restart_reload_verified=True,
    )


def reload_persistent_nerf_representation(scene_manifest_path: str | Path) -> NerfPersistentRepresentationResult:
    """Reload persisted NeRF evidence without starting Nerfstudio/tiny-cuda-nn."""

    manifest = Path(scene_manifest_path).expanduser().resolve()
    scene = read_persistent_reconstruction_scene(manifest)
    verify_persistent_scene_file_references(scene)
    record = scene.nerf_artifact
    if record is None:
        raise ReconstructionValidationError("Persistent scene does not contain NeRF appearance.")
    snapshot = metadata_as_dict(record.payload_snapshot)  # type: ignore[arg-type]
    artifact_manifest = Path(str(snapshot.get("artifact_manifest_path", ""))).expanduser().resolve()
    artifact = NerfSceneArtifact.from_payload(_load_json(artifact_manifest, label="Persisted NeRF artifact manifest"))
    _validate_scene_artifact_compatibility_for_reload(scene, artifact)
    refs = tuple(item for item in scene.file_references if item.artifact_id == record.artifact_id)
    if not refs:
        raise ReconstructionValidationError("Persistent NeRF appearance has no integrity references.")
    parent_scene_id = str(snapshot.get("parent_scene_id", "")).strip()
    parent_fingerprint = str(snapshot.get("parent_scene_fingerprint_sha256", "")).strip()
    if not parent_scene_id or len(parent_fingerprint) != 64:
        raise ReconstructionValidationError("Persistent NeRF lineage metadata is incomplete.")
    dataset_root = Path(str(snapshot.get("dataset_transforms_path", ""))).expanduser().resolve().parent
    dataset_integrity = Path(str(snapshot.get("dataset_integrity_manifest_path", ""))).expanduser().resolve()
    runtime_load_spec = Path(str(snapshot.get("runtime_load_spec_path", ""))).expanduser().resolve()
    _verify_dataset_integrity(dataset_root, dataset_integrity)
    load_spec = _load_json(runtime_load_spec, label="NeRF runtime load spec")
    if load_spec.get("scene_role") != "appearance_only" or load_spec.get("backend_start_performed") is not False:
        raise ReconstructionValidationError("Persisted NeRF runtime load spec changed authority/execution state.")
    if Path(str(load_spec.get("checkpoint_path", ""))).resolve() != Path(artifact.checkpoint_path).resolve():
        raise ReconstructionValidationError("Persisted NeRF runtime load spec checkpoint mismatch.")
    if Path(str(load_spec.get("immutable_training_config_path", ""))).resolve() != Path(artifact.config_path).resolve():
        raise ReconstructionValidationError("Persisted NeRF runtime load spec config mismatch.")
    return NerfPersistentRepresentationResult(
        state=NerfPersistenceState.RELOADED,
        parent_scene_id=parent_scene_id,
        parent_scene_fingerprint_sha256=parent_fingerprint,
        scene=scene,
        scene_manifest_path=str(manifest),
        artifact=artifact,
        nerf_file_references=refs,
        dataset_root=str(dataset_root),
        dataset_integrity_manifest_path=str(dataset_integrity),
        runtime_load_spec_path=str(runtime_load_spec),
        restart_reload_verified=True,
    )


def _validate_scene_artifact_compatibility_for_reload(
    scene: PersistentReconstructionScene,
    artifact: NerfSceneArtifact,
) -> None:
    if scene.nerf_artifact is None:
        raise ReconstructionValidationError("Reloaded scene lost NeRF appearance artifact.")
    if artifact.scene_role is not NerfSceneRole.APPEARANCE_ONLY or artifact.canonical_geometry_authoritative:
        raise ReconstructionValidationError("Reloaded NeRF representation changed authority.")
    if artifact.certified_metrology:
        raise ReconstructionValidationError("Reloaded NeRF representation claims certified metrology.")
    if artifact.training_state is not NerfTrainingState.TRAINED or not artifact.render_capable:
        raise ReconstructionValidationError("Reloaded NeRF representation is not trained/render-capable.")
    if artifact.reconstruction_frame_id != _scene_reconstruction_frame_id(scene):
        raise ReconstructionValidationError("Reloaded NeRF reconstruction frame differs from canonical scene.")
    surface = _scene_snapshot(scene, ReconstructionArtifactKind.SURFACE_MESH)
    if str(surface.get("metric_geometry_status", "")).strip() != artifact.metric_geometry_status.value:
        raise ReconstructionValidationError("Reloaded NeRF scale state differs from canonical surface.")
    if artifact.unit_to_metre != surface.get("unit_to_metre"):
        raise ReconstructionValidationError("Reloaded NeRF unit_to_metre differs from canonical surface.")


__all__ = [
    "PHASE9_5D_NERF_PERSISTENCE_SCHEMA_VERSION",
    "PHASE9_5D_NERF_PERSISTENCE_PHASE",
    "NerfPersistenceState",
    "NerfPersistentRepresentationResult",
    "persist_nerf_representation",
    "reload_persistent_nerf_representation",
]

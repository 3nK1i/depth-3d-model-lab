"""Phase 9.2H durable Auto-Segment workspace persistence.

The persisted workspace is semantic review state only.  It stores the exact BGR
source frame, dense masks, class/instance/review metadata, and provenance needed
to resume human review after restart.  It never grants geometry, metric,
measurement, QC, or canonical-scene authority.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping
from uuid import uuid4

import cv2
import numpy as np

from core.contracts import FrameReference, ImageSize, ModelProvenance, PixelCoordinateSpace, PixelSpaceRole
from core.contracts.validation import ContractValidationError, require_nonempty
from core.models.metadata import ConfidenceKind
from core.perception.result import ApprovalStatus, DenseMaskPayload, InstanceMask, MaskSource
from core.perception.reviewed_analysis import source_image_sha256
from core.perception.semantic_project import SegmentationProject
from core.perception.semantic_workspace import (
    SemanticLabelSource,
    SemanticReviewStatus,
    SemanticWorkspace,
    SemanticWorkspaceObject,
)

SEMANTIC_WORKSPACE_PERSISTENCE_SCHEMA_VERSION = 1
_WORKSPACE_DIR_SUFFIX = ".semantic-workspace"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _project_key(project_id: str) -> str:
    return sha256(require_nonempty(project_id, field_name="project_id").encode("utf-8")).hexdigest()


def _model_provenance_from_dict(payload: Mapping[str, Any]) -> ModelProvenance:
    return ModelProvenance(
        model_id=str(payload.get("model_id", "persisted-semantic-mask")),
        display_name=str(payload.get("display_name", "Persisted Semantic Mask")),
        task=str(payload.get("task", "segmentation")),
        model_version=str(payload.get("model_version", "")),
        checkpoint=str(payload.get("checkpoint", "")),
        device=str(payload.get("device", "")),
        precision=str(payload.get("precision", "")),
        runtime_config_digest=str(payload.get("runtime_config_digest", "")),
        metadata=payload.get("metadata", {}),
    )


@dataclass(frozen=True, slots=True)
class PersistedSemanticWorkspace:
    project_id: str
    source_label: str
    source_image_sha256: str
    image_bgr: np.ndarray
    workspace: SemanticWorkspace
    saved_at_utc: str

    def __post_init__(self) -> None:
        image = np.ascontiguousarray(self.image_bgr, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ContractValidationError("Persisted semantic source must be an HxWx3 BGR image.")
        if source_image_sha256(image) != str(self.source_image_sha256):
            raise ContractValidationError("Persisted semantic source digest does not match image bytes.")
        object.__setattr__(self, "image_bgr", image)
        object.__setattr__(self, "project_id", require_nonempty(self.project_id, field_name="project_id"))
        object.__setattr__(self, "source_label", str(self.source_label).strip() or "Persisted image")
        object.__setattr__(self, "saved_at_utc", require_nonempty(self.saved_at_utc, field_name="saved_at_utc"))

    @property
    def is_geometry_authority(self) -> bool:
        return False

    @property
    def is_metric_authority(self) -> bool:
        return False

    @property
    def is_measurement_authority(self) -> bool:
        return False


class SegmentationWorkspaceStore:
    """Atomic latest-workspace store keyed by segmentation project ID."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()

    def path_for_project(self, project_id: str) -> Path:
        return self.root / f"{_project_key(project_id)}{_WORKSPACE_DIR_SUFFIX}"

    def contains(self, project_id: str) -> bool:
        root = self.path_for_project(project_id)
        return (root / "manifest.json").is_file() and (root / "source.png").is_file() and (root / "masks.npz").is_file()

    def clear(self, project_id: str) -> None:
        target = self.path_for_project(project_id)
        if target.exists():
            shutil.rmtree(target)

    def save(
        self,
        project: SegmentationProject,
        workspace: SemanticWorkspace,
        image_bgr: np.ndarray,
        *,
        source_label: str,
    ) -> Path:
        if not isinstance(project, SegmentationProject):
            raise TypeError("project must be SegmentationProject.")
        if not isinstance(workspace, SemanticWorkspace):
            raise TypeError("workspace must be SemanticWorkspace.")
        image = np.ascontiguousarray(image_bgr, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ContractValidationError("Semantic workspace source must be an HxWx3 BGR image.")
        if (workspace.frame_reference.source_size.width, workspace.frame_reference.source_size.height) != (image.shape[1], image.shape[0]):
            raise ContractValidationError("Workspace frame size does not match persisted source image.")

        project_classes = {item.class_id: item for item in project.class_catalog.classes}
        for obj in workspace.objects:
            if obj.class_id is not None:
                definition = project_classes.get(obj.class_id)
                if definition is None or definition.class_name != obj.class_name:
                    raise ContractValidationError("Workspace contains a class that is not active in its project.")
            if obj.proposal_mask is None or not isinstance(obj.proposal_mask.payload, DenseMaskPayload):
                raise ContractValidationError("Phase 9.2H persistence currently requires proposal-backed dense masks.")

        target = self.path_for_project(project.project_id)
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.root / f".{target.name}.{uuid4().hex}.tmp"
        temporary.mkdir(parents=True, exist_ok=False)
        try:
            ok, encoded = cv2.imencode(".png", image)
            if not ok:
                raise ContractValidationError("Could not encode semantic workspace source image.")
            (temporary / "source.png").write_bytes(encoded.tobytes())

            arrays: dict[str, np.ndarray] = {}
            object_records: list[dict[str, Any]] = []
            for index, obj in enumerate(workspace.objects):
                assert obj.proposal_mask is not None
                mask = obj.proposal_mask
                key = f"mask_{index:04d}"
                arrays[key] = np.asarray(mask.payload.data, dtype=bool)
                object_records.append({
                    "object_id": obj.object_id,
                    "class_id": obj.class_id,
                    "class_name": obj.class_name,
                    "instance_name": obj.instance_name,
                    "label_source": obj.label_source.value,
                    "review_status": obj.review_status.value,
                    "label_confidence": obj.label_confidence,
                    "notes": obj.notes,
                    "merged_from_object_ids": list(obj.merged_from_object_ids),
                    "merged_from_mask_ids": list(obj.merged_from_mask_ids),
                    "mask": {
                        "array_key": key,
                        "mask_id": mask.mask_id,
                        "source": mask.source.value,
                        "revision": mask.revision,
                        "parent_mask_id": mask.parent_mask_id,
                        "approval_status": mask.approval_status.value,
                        "approved_by": mask.approved_by,
                        "approved_timestamp": mask.approved_timestamp,
                        "confidence": mask.confidence,
                        "confidence_kind": mask.confidence_kind.value,
                        "foreground_valid": mask.foreground_valid,
                        "model_provenance": mask.model_provenance.as_dict(),
                    },
                })
            np.savez_compressed(temporary / "masks.npz", **arrays)
            manifest = {
                "schema_version": SEMANTIC_WORKSPACE_PERSISTENCE_SCHEMA_VERSION,
                "project_id": project.project_id,
                "source_label": str(source_label).strip() or "Image",
                "source_image_sha256": source_image_sha256(image),
                "saved_at_utc": _utc_now_iso(),
                "workspace_id": workspace.workspace_id,
                "workspace_revision": workspace.revision,
                "frame_reference": workspace.frame_reference.as_dict(),
                "objects": object_records,
                "authority": {
                    "geometry": False,
                    "metric": False,
                    "measurement": False,
                },
            }
            (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            if target.exists():
                backup = self.root / f".{target.name}.{uuid4().hex}.old"
                os.replace(target, backup)
                try:
                    os.replace(temporary, target)
                except Exception:
                    os.replace(backup, target)
                    raise
                shutil.rmtree(backup, ignore_errors=True)
            else:
                os.replace(temporary, target)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary, ignore_errors=True)
        return target

    def load(self, project: SegmentationProject) -> PersistedSemanticWorkspace:
        if not isinstance(project, SegmentationProject):
            raise TypeError("project must be SegmentationProject.")
        root = self.path_for_project(project.project_id)
        try:
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            image = cv2.imread(str(root / "source.png"), cv2.IMREAD_COLOR)
            arrays = np.load(root / "masks.npz", allow_pickle=False)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise ContractValidationError("Could not load persisted semantic workspace.") from exc
        if image is None:
            raise ContractValidationError("Persisted semantic workspace source image is unreadable.")
        if int(manifest.get("schema_version", -1)) != SEMANTIC_WORKSPACE_PERSISTENCE_SCHEMA_VERSION:
            raise ContractValidationError("Unsupported persisted semantic workspace schema version.")
        if str(manifest.get("project_id", "")) != project.project_id:
            raise ContractValidationError("Persisted semantic workspace belongs to another project.")
        digest = source_image_sha256(image)
        if digest != str(manifest.get("source_image_sha256", "")):
            raise ContractValidationError("Persisted semantic workspace source image failed digest verification.")

        frame_payload = manifest.get("frame_reference", {})
        size_payload = frame_payload.get("source_size", {})
        frame = FrameReference(
            int(frame_payload.get("session_id", 0)),
            int(frame_payload.get("frame_id", 0)),
            float(frame_payload.get("source_timestamp", 0.0)),
            ImageSize(int(size_payload.get("width", image.shape[1])), int(size_payload.get("height", image.shape[0]))),
        )
        if (frame.source_size.width, frame.source_size.height) != (image.shape[1], image.shape[0]):
            raise ContractValidationError("Persisted workspace frame size does not match source image.")
        space = PixelCoordinateSpace("rgb", frame, frame.source_size, PixelSpaceRole.RGB)

        objects: list[SemanticWorkspaceObject] = []
        active_classes = {item.class_id: item for item in project.class_catalog.classes}
        for record in manifest.get("objects", []):
            mask_payload = record.get("mask", {})
            key = str(mask_payload.get("array_key", ""))
            if key not in arrays:
                raise ContractValidationError("Persisted semantic workspace mask array is missing.")
            data = np.asarray(arrays[key], dtype=bool)
            provenance_payload = mask_payload.get("model_provenance", {})
            if not isinstance(provenance_payload, Mapping):
                raise ContractValidationError("Persisted mask provenance must be an object.")
            mask = InstanceMask(
                mask_id=str(mask_payload.get("mask_id", "")),
                frame_reference=frame,
                pixel_space=space,
                payload=DenseMaskPayload(data),
                source=MaskSource(str(mask_payload.get("source", MaskSource.NATIVE_INSTANCE.value))),
                model_provenance=_model_provenance_from_dict(provenance_payload),
                revision=int(mask_payload.get("revision", 0)),
                parent_mask_id=str(mask_payload.get("parent_mask_id", "")),
                approval_status=ApprovalStatus(str(mask_payload.get("approval_status", ApprovalStatus.NOT_REQUIRED.value))),
                approved_by=str(mask_payload.get("approved_by", "")),
                approved_timestamp=mask_payload.get("approved_timestamp"),
                confidence=mask_payload.get("confidence"),
                confidence_kind=ConfidenceKind(str(mask_payload.get("confidence_kind", ConfidenceKind.NONE.value))),
                foreground_valid=bool(mask_payload.get("foreground_valid", bool(data.any()))),
            )
            class_id = record.get("class_id")
            if class_id is not None:
                class_id = int(class_id)
                definition = active_classes.get(class_id)
                if definition is None or definition.class_name != str(record.get("class_name", "")):
                    raise ContractValidationError("Persisted workspace references a class that is no longer active.")
            objects.append(SemanticWorkspaceObject(
                object_id=str(record.get("object_id", "")),
                proposal_mask=mask,
                class_id=class_id,
                class_name=str(record.get("class_name", "")),
                instance_name=str(record.get("instance_name", "")),
                label_source=SemanticLabelSource(str(record.get("label_source", SemanticLabelSource.UNASSIGNED.value))),
                review_status=SemanticReviewStatus(str(record.get("review_status", SemanticReviewStatus.PENDING.value))),
                label_confidence=record.get("label_confidence"),
                notes=str(record.get("notes", "")),
                merged_from_object_ids=tuple(str(v) for v in record.get("merged_from_object_ids", ())),
                merged_from_mask_ids=tuple(str(v) for v in record.get("merged_from_mask_ids", ())),
            ))
        workspace = SemanticWorkspace(
            workspace_id=str(manifest.get("workspace_id", "")),
            frame_reference=frame,
            class_catalog=project.class_catalog,
            objects=tuple(objects),
            revision=int(manifest.get("workspace_revision", 0)),
        )
        return PersistedSemanticWorkspace(
            project_id=project.project_id,
            source_label=str(manifest.get("source_label", "Persisted image")),
            source_image_sha256=digest,
            image_bgr=image,
            workspace=workspace,
            saved_at_utc=str(manifest.get("saved_at_utc", "")),
        )

"""Transient bridge from reviewed Phase 9.2 semantics into Measure & Inspect.

This module is intentionally *not* workspace persistence.  It snapshots only the
accepted semantic objects for one exact source image so an isolated measurement
job can reuse reviewed masks instead of rerunning a perception model.  Phase
9.2H owns durable workspace/project persistence.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from core.contracts import FrameReference, ModelProvenance, PixelCoordinateSpace
from core.contracts.validation import ContractValidationError, readonly_array, require_nonempty
from core.models.metadata import ConfidenceKind
from core.perception.result import (
    ApprovalStatus,
    DenseMaskPayload,
    InstanceMask,
    MaskSource,
    ObjectObservation,
    PerceptionFrameResult,
)
from core.perception.semantic_workspace import SemanticWorkspace


REVIEWED_ANALYSIS_SCHEMA_VERSION = 1
REVIEWED_ANALYSIS_METADATA = "reviewed_semantics.json"
REVIEWED_ANALYSIS_ARRAYS = "reviewed_semantic_masks.npz"


def source_image_sha256(image_bgr: np.ndarray) -> str:
    """Deterministic identity for the exact BGR frame reviewed by the operator."""

    image = np.ascontiguousarray(image_bgr, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ContractValidationError("Reviewed semantic source must be an HxWx3 BGR image.")
    digest = sha256()
    digest.update(str(tuple(int(value) for value in image.shape)).encode("ascii"))
    digest.update(b"\0uint8\0")
    digest.update(image.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ReviewedSemanticObject:
    object_id: str
    class_id: int
    class_name: str
    instance_name: str
    mask_id: str
    mask_source: MaskSource
    mask_revision: int
    mask: np.ndarray
    merged_from_object_ids: tuple[str, ...] = ()
    merged_from_mask_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "object_id", require_nonempty(self.object_id, field_name="object_id"))
        if isinstance(self.class_id, bool) or int(self.class_id) < 0:
            raise ContractValidationError("Reviewed class_id must be a non-negative integer.")
        object.__setattr__(self, "class_id", int(self.class_id))
        object.__setattr__(self, "class_name", require_nonempty(self.class_name, field_name="class_name"))
        object.__setattr__(self, "instance_name", str(self.instance_name).strip())
        object.__setattr__(self, "mask_id", require_nonempty(self.mask_id, field_name="mask_id"))
        if not isinstance(self.mask_source, MaskSource):
            raise TypeError("mask_source must be MaskSource.")
        if isinstance(self.mask_revision, bool) or int(self.mask_revision) < 0:
            raise ContractValidationError("mask_revision must be a non-negative integer.")
        object.__setattr__(self, "mask_revision", int(self.mask_revision))
        mask = readonly_array(self.mask, field_name="reviewed mask", dtype=np.bool_, ndim=2)
        if not bool(mask.any()):
            raise ContractValidationError("Reviewed masks must contain foreground pixels.")
        object.__setattr__(self, "mask", mask)
        objects = tuple(require_nonempty(value, field_name="merged_from_object_id") for value in self.merged_from_object_ids)
        masks = tuple(require_nonempty(value, field_name="merged_from_mask_id") for value in self.merged_from_mask_ids)
        if bool(objects) != bool(masks) or (objects and len(objects) != len(masks)):
            raise ContractValidationError("Reviewed merge provenance must contain matched object/mask IDs.")
        object.__setattr__(self, "merged_from_object_ids", objects)
        object.__setattr__(self, "merged_from_mask_ids", masks)


@dataclass(frozen=True, slots=True)
class ReviewedSemanticSelection:
    workspace_id: str
    workspace_revision: int
    source_image_sha256: str
    image_width: int
    image_height: int
    objects: tuple[ReviewedSemanticObject, ...]
    schema_version: int = REVIEWED_ANALYSIS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "workspace_id", require_nonempty(self.workspace_id, field_name="workspace_id"))
        if isinstance(self.workspace_revision, bool) or int(self.workspace_revision) < 0:
            raise ContractValidationError("workspace_revision must be a non-negative integer.")
        object.__setattr__(self, "workspace_revision", int(self.workspace_revision))
        digest = str(self.source_image_sha256).strip().lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ContractValidationError("source_image_sha256 must be one lowercase SHA-256 digest.")
        object.__setattr__(self, "source_image_sha256", digest)
        if int(self.image_width) <= 0 or int(self.image_height) <= 0:
            raise ContractValidationError("Reviewed source image dimensions must be positive.")
        object.__setattr__(self, "image_width", int(self.image_width))
        object.__setattr__(self, "image_height", int(self.image_height))
        if int(self.schema_version) != REVIEWED_ANALYSIS_SCHEMA_VERSION:
            raise ContractValidationError("Unsupported reviewed semantic analysis schema version.")
        object.__setattr__(self, "schema_version", REVIEWED_ANALYSIS_SCHEMA_VERSION)
        objects = tuple(self.objects)
        if not objects:
            raise ContractValidationError("Reviewed semantic analysis requires at least one accepted object.")
        ids = [item.object_id for item in objects]
        if len(ids) != len(set(ids)):
            raise ContractValidationError("Reviewed semantic object IDs must be unique.")
        expected = (self.image_height, self.image_width)
        for item in objects:
            if tuple(item.mask.shape) != expected:
                raise ContractValidationError(
                    f"Reviewed mask {item.object_id!r} has shape {item.mask.shape}, expected {expected}."
                )
        object.__setattr__(self, "objects", objects)

    @property
    def is_geometry_authority(self) -> bool:
        return False

    @property
    def is_metric_authority(self) -> bool:
        return False

    @property
    def is_measurement_authority(self) -> bool:
        return False


def reviewed_selection_from_workspace(
    workspace: SemanticWorkspace,
    image_bgr: np.ndarray,
) -> ReviewedSemanticSelection:
    """Snapshot accepted reviewed masks for the exact image used by the workspace."""

    if not isinstance(workspace, SemanticWorkspace):
        raise TypeError("workspace must be SemanticWorkspace.")
    image = np.ascontiguousarray(image_bgr, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ContractValidationError("Reviewed semantic source must be an HxWx3 BGR image.")
    height, width = image.shape[:2]
    if workspace.frame_reference.source_size.width != width or workspace.frame_reference.source_size.height != height:
        raise ContractValidationError("Semantic workspace frame dimensions do not match its source image.")

    output: list[ReviewedSemanticObject] = []
    for item in workspace.accepted_objects:
        if item.class_id is None:
            raise ContractValidationError("Accepted semantic objects must remain class-assigned.")
        mask = item.proposal_mask if item.proposal_mask is not None else (
            item.observation.selected_mask if item.observation is not None else None
        )
        if mask is None or not isinstance(mask.payload, DenseMaskPayload):
            raise ContractValidationError("Reviewed Measure & Inspect integration requires dense accepted masks.")
        output.append(
            ReviewedSemanticObject(
                object_id=item.object_id,
                class_id=item.class_id,
                class_name=item.class_name,
                instance_name=item.instance_name,
                mask_id=mask.mask_id,
                mask_source=mask.source,
                mask_revision=mask.revision,
                mask=np.asarray(mask.payload.data, dtype=np.bool_),
                merged_from_object_ids=item.merged_from_object_ids,
                merged_from_mask_ids=item.merged_from_mask_ids,
            )
        )
    return ReviewedSemanticSelection(
        workspace_id=workspace.workspace_id,
        workspace_revision=workspace.revision,
        source_image_sha256=source_image_sha256(image),
        image_width=width,
        image_height=height,
        objects=tuple(output),
    )


def save_reviewed_semantic_selection(
    selection: ReviewedSemanticSelection,
    directory: str | Path,
) -> tuple[Path, Path]:
    """Write one transient, transparent reviewed-mask handoff for an isolated job."""

    if not isinstance(selection, ReviewedSemanticSelection):
        raise TypeError("selection must be ReviewedSemanticSelection.")
    root = Path(directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    arrays_path = root / REVIEWED_ANALYSIS_ARRAYS
    metadata_path = root / REVIEWED_ANALYSIS_METADATA
    arrays: dict[str, np.ndarray] = {}
    records: list[dict[str, Any]] = []
    for index, item in enumerate(selection.objects):
        key = f"mask_{index:03d}"
        arrays[key] = np.asarray(item.mask, dtype=np.bool_)
        records.append({
            "object_id": item.object_id,
            "class_id": item.class_id,
            "class_name": item.class_name,
            "instance_name": item.instance_name,
            "mask_id": item.mask_id,
            "mask_source": item.mask_source.value,
            "mask_revision": item.mask_revision,
            "mask_key": key,
            "merged_from_object_ids": list(item.merged_from_object_ids),
            "merged_from_mask_ids": list(item.merged_from_mask_ids),
        })
    np.savez_compressed(arrays_path, **arrays)
    metadata = {
        "schema_version": selection.schema_version,
        "workspace_id": selection.workspace_id,
        "workspace_revision": selection.workspace_revision,
        "source_image_sha256": selection.source_image_sha256,
        "image_size": {"width": selection.image_width, "height": selection.image_height},
        "arrays_file": arrays_path.name,
        "objects": records,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata_path, arrays_path


def load_reviewed_semantic_selection(path: str | Path) -> ReviewedSemanticSelection:
    metadata_path = Path(path).expanduser().resolve()
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ContractValidationError("Reviewed semantic metadata root must be an object.")
    arrays_name = str(payload.get("arrays_file", REVIEWED_ANALYSIS_ARRAYS)).strip()
    arrays_path = (metadata_path.parent / arrays_name).resolve()
    if arrays_path.parent != metadata_path.parent or not arrays_path.is_file():
        raise ContractValidationError("Reviewed semantic mask archive is missing or outside its handoff directory.")
    objects_payload = payload.get("objects", ())
    if not isinstance(objects_payload, list):
        raise ContractValidationError("Reviewed semantic objects must be a JSON array.")
    objects: list[ReviewedSemanticObject] = []
    with np.load(arrays_path, allow_pickle=False) as archive:
        for item in objects_payload:
            if not isinstance(item, Mapping):
                raise ContractValidationError("Reviewed semantic object records must be JSON objects.")
            key = str(item.get("mask_key", "")).strip()
            if not key or key not in archive:
                raise ContractValidationError("Reviewed semantic mask archive is incomplete.")
            try:
                objects.append(
                    ReviewedSemanticObject(
                        object_id=str(item["object_id"]),
                        class_id=int(item["class_id"]),
                        class_name=str(item["class_name"]),
                        instance_name=str(item.get("instance_name", "")),
                        mask_id=str(item["mask_id"]),
                        mask_source=MaskSource(str(item["mask_source"])),
                        mask_revision=int(item.get("mask_revision", 0)),
                        mask=np.asarray(archive[key], dtype=np.bool_),
                        merged_from_object_ids=tuple(str(value) for value in item.get("merged_from_object_ids", ())),
                        merged_from_mask_ids=tuple(str(value) for value in item.get("merged_from_mask_ids", ())),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ContractValidationError("Invalid reviewed semantic object record.") from exc
    image_size = payload.get("image_size", {})
    if not isinstance(image_size, Mapping):
        raise ContractValidationError("Reviewed semantic image_size must be an object.")
    return ReviewedSemanticSelection(
        workspace_id=str(payload["workspace_id"]),
        workspace_revision=int(payload.get("workspace_revision", 0)),
        source_image_sha256=str(payload["source_image_sha256"]),
        image_width=int(image_size["width"]),
        image_height=int(image_size["height"]),
        objects=tuple(objects),
        schema_version=int(payload.get("schema_version", 0)),
    )


def perception_from_reviewed_selection(
    selection: ReviewedSemanticSelection,
    frame_reference: FrameReference,
    pixel_space: PixelCoordinateSpace,
    *,
    masks_by_object_id: Mapping[str, np.ndarray] | None = None,
    transformed: bool = False,
) -> tuple[PerceptionFrameResult, dict[str, dict[str, Any]]]:
    """Normalize reviewed objects into the existing downstream perception contract."""

    if pixel_space.size.width != selection.image_width or pixel_space.size.height != selection.image_height:
        raise ContractValidationError("Reviewed semantic selection dimensions do not match analysis RGB space.")
    provenance = ModelProvenance(
        model_id="reviewed-auto-segment",
        display_name="Reviewed Auto Segment",
        task="instance_segmentation",
        metadata=(
            ("workspace_id", selection.workspace_id),
            ("workspace_revision", str(selection.workspace_revision)),
            ("operator_reviewed", "true"),
        ),
    )
    observations: list[ObjectObservation] = []
    metadata: dict[str, dict[str, Any]] = {}
    for item in selection.objects:
        dense = np.asarray(
            masks_by_object_id[item.object_id] if masks_by_object_id is not None else item.mask,
            dtype=np.bool_,
        )
        if dense.shape != (selection.image_height, selection.image_width) or not bool(dense.any()):
            raise ContractValidationError(f"Reviewed analysis mask is invalid for {item.object_id!r}.")
        mask_id = f"{item.mask_id}:analysis-rectified" if transformed else item.mask_id
        mask_provenance = ModelProvenance(
            model_id="reviewed-auto-segment",
            display_name="Reviewed Auto Segment",
            task="reviewed_mask",
            metadata=(("source_mask_id", item.mask_id), ("workspace_id", selection.workspace_id)),
        )
        mask = InstanceMask(
            mask_id=mask_id,
            frame_reference=frame_reference,
            pixel_space=pixel_space,
            payload=DenseMaskPayload(dense),
            source=item.mask_source,
            confidence_kind=ConfidenceKind.NONE,
            confidence=None,
            model_provenance=mask_provenance,
            revision=item.mask_revision,
            approval_status=ApprovalStatus.NOT_REQUIRED,
            foreground_valid=True,
        )
        warnings = ["operator_reviewed_semantic_mask"]
        if transformed:
            warnings.append("reviewed_mask_rectified_for_calibrated_analysis")
        if item.merged_from_mask_ids:
            warnings.append("reviewed_semantic_merge")
        observations.append(
            ObjectObservation(
                observation_id=item.object_id,
                frame_reference=frame_reference,
                class_id=item.class_id,
                class_name=item.class_name,
                masks=(mask,),
                warnings=tuple(warnings),
            )
        )
        metadata[item.object_id] = {
            "class_id": item.class_id,
            "class_name": item.class_name,
            "instance_name": item.instance_name,
            "detection_confidence": None,
            "detection_box_xyxy": None,
            "track_id": "",
            "semantic_source": "reviewed_auto_segment",
            "semantic_review_status": "accepted",
            "workspace_id": selection.workspace_id,
            "workspace_revision": selection.workspace_revision,
            "merged_from_object_ids": list(item.merged_from_object_ids),
            "merged_from_mask_ids": list(item.merged_from_mask_ids),
        }
    return (
        PerceptionFrameResult(
            frame_reference=frame_reference,
            observations=tuple(observations),
            model_provenance=(provenance,),
        ),
        metadata,
    )

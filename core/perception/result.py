"""Per-frame detection, mask and object-observation contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeAlias

import numpy as np

from core.contracts.common import FrameReference, ModelProvenance
from core.contracts.spatial import PixelCoordinateSpace
from core.contracts.validation import (
    ContractValidationError,
    readonly_array,
    require_finite,
    require_nonempty,
    require_probability,
)
from core.models.metadata import ConfidenceKind
from core.perception.messages import SegmentationPrompt


class MaskRepresentation(str, Enum):
    DENSE_BOOLEAN = "dense_boolean"
    RLE = "rle"
    POLYGON = "polygon"
    SHARED_MEMORY_REFERENCE = "shared_memory_reference"


class MaskSource(str, Enum):
    NATIVE_INSTANCE = "native_instance_segmentation"
    PROMPTED_REFINEMENT = "prompted_refinement"
    DETECTION_BOX_FALLBACK = "detection_box_fallback"
    MANUAL_CORRECTION = "manual_correction"


class ApprovalStatus(str, Enum):
    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class PerceptionResultStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


class UnsupportedMaskRepresentationError(ContractValidationError):
    """Raised when runtime normalization is not implemented for a schema-ready payload."""


@dataclass(frozen=True, slots=True)
class BoundingBox2D:
    """Normalized clipped box using [x_min, y_min, x_max, y_max)."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float
    pixel_space_id: str
    was_clipped: bool = False
    raw_xyxy: tuple[float, float, float, float] | None = None

    def __post_init__(self) -> None:
        values = tuple(
            require_finite(value, field_name=name)
            for value, name in zip(
                (self.x_min, self.y_min, self.x_max, self.y_max),
                ("x_min", "y_min", "x_max", "y_max"),
                strict=True,
            )
        )
        if values[2] <= values[0] or values[3] <= values[1]:
            raise ContractValidationError("BoundingBox2D must have positive area.")
        object.__setattr__(self, "x_min", values[0])
        object.__setattr__(self, "y_min", values[1])
        object.__setattr__(self, "x_max", values[2])
        object.__setattr__(self, "y_max", values[3])
        object.__setattr__(
            self, "pixel_space_id", require_nonempty(self.pixel_space_id, field_name="pixel_space_id")
        )
        if self.raw_xyxy is not None:
            raw = tuple(float(value) for value in self.raw_xyxy)
            if len(raw) != 4 or not np.isfinite(np.asarray(raw)).all():
                raise ContractValidationError("raw_xyxy must contain four finite coordinates.")
            object.__setattr__(self, "raw_xyxy", raw)

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return (self.x_min, self.y_min, self.x_max, self.y_max)


def normalize_detection_box(
    raw_xyxy: tuple[float, float, float, float],
    pixel_space: PixelCoordinateSpace,
) -> BoundingBox2D:
    """Clip a finite raw detector box and preserve its original coordinates."""

    if len(raw_xyxy) != 4:
        raise ContractValidationError("raw_xyxy must contain four coordinates.")
    raw = tuple(float(value) for value in raw_xyxy)
    if not np.isfinite(np.asarray(raw)).all():
        raise ContractValidationError("raw detector coordinates must be finite.")
    x0, y0, x1, y1 = raw
    clipped = (
        min(max(x0, 0.0), float(pixel_space.size.width)),
        min(max(y0, 0.0), float(pixel_space.size.height)),
        min(max(x1, 0.0), float(pixel_space.size.width)),
        min(max(y1, 0.0), float(pixel_space.size.height)),
    )
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        raise ContractValidationError("Detector box is empty or invalid after clipping.")
    return BoundingBox2D(
        *clipped,
        pixel_space_id=pixel_space.space_id,
        was_clipped=not np.allclose(np.asarray(raw), np.asarray(clipped)),
        raw_xyxy=raw,
    )


@dataclass(frozen=True, slots=True)
class Detection2D:
    detection_id: str
    frame_reference: FrameReference
    class_id: int
    class_name: str
    confidence: float
    box: BoundingBox2D
    model_provenance: ModelProvenance

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "detection_id", require_nonempty(self.detection_id, field_name="detection_id")
        )
        if isinstance(self.class_id, bool) or int(self.class_id) < 0:
            raise ContractValidationError("class_id must be a non-negative integer.")
        object.__setattr__(self, "class_id", int(self.class_id))
        object.__setattr__(
            self, "class_name", require_nonempty(self.class_name, field_name="class_name")
        )
        object.__setattr__(
            self, "confidence", require_probability(self.confidence, field_name="confidence")
        )
        if not isinstance(self.box, BoundingBox2D):
            raise TypeError("box must be a BoundingBox2D.")


@dataclass(frozen=True, slots=True)
class DenseMaskPayload:
    data: np.ndarray
    representation: MaskRepresentation = field(
        default=MaskRepresentation.DENSE_BOOLEAN, init=False
    )

    def __post_init__(self) -> None:
        array = readonly_array(
            self.data,
            field_name="dense mask",
            dtype=np.bool_,
            ndim=2,
        )
        object.__setattr__(self, "data", array)

    def __reduce__(self):
        # Reconstruct through __init__ after multiprocessing/pickle so the
        # public immutable-array invariant is restored in the receiver.
        return (type(self), (self.data,))

    @property
    def height(self) -> int:
        return int(self.data.shape[0])

    @property
    def width(self) -> int:
        return int(self.data.shape[1])


@dataclass(frozen=True, slots=True)
class RLEMaskPayload:
    width: int
    height: int
    counts: tuple[int, ...] | str
    representation: MaskRepresentation = field(default=MaskRepresentation.RLE, init=False)

    def __post_init__(self) -> None:
        if int(self.width) <= 0 or int(self.height) <= 0:
            raise ContractValidationError("RLE mask dimensions must be positive.")
        if isinstance(self.counts, str):
            if not self.counts:
                raise ContractValidationError("RLE string counts must not be empty.")
        else:
            counts = tuple(int(value) for value in self.counts)
            if not counts or any(value < 0 for value in counts):
                raise ContractValidationError("RLE counts must be non-negative and non-empty.")
            object.__setattr__(self, "counts", counts)
        object.__setattr__(self, "width", int(self.width))
        object.__setattr__(self, "height", int(self.height))


@dataclass(frozen=True, slots=True)
class PolygonMaskPayload:
    width: int
    height: int
    polygons_xy: tuple[tuple[float, ...], ...]
    representation: MaskRepresentation = field(default=MaskRepresentation.POLYGON, init=False)

    def __post_init__(self) -> None:
        if int(self.width) <= 0 or int(self.height) <= 0:
            raise ContractValidationError("Polygon mask dimensions must be positive.")
        polygons = tuple(tuple(float(value) for value in polygon) for polygon in self.polygons_xy)
        if not polygons:
            raise ContractValidationError("At least one polygon is required.")
        for polygon in polygons:
            if len(polygon) < 6 or len(polygon) % 2:
                raise ContractValidationError(
                    "Each polygon must contain at least three x/y coordinate pairs."
                )
            if not np.isfinite(np.asarray(polygon)).all():
                raise ContractValidationError("Polygon coordinates must be finite.")
        object.__setattr__(self, "width", int(self.width))
        object.__setattr__(self, "height", int(self.height))
        object.__setattr__(self, "polygons_xy", polygons)


@dataclass(frozen=True, slots=True)
class SharedMemoryMaskPayload:
    width: int
    height: int
    name: str
    dtype: str = "bool"
    offset_bytes: int = 0
    nbytes: int | None = None
    representation: MaskRepresentation = field(
        default=MaskRepresentation.SHARED_MEMORY_REFERENCE, init=False
    )

    def __post_init__(self) -> None:
        if int(self.width) <= 0 or int(self.height) <= 0:
            raise ContractValidationError("Shared-memory mask dimensions must be positive.")
        if int(self.offset_bytes) < 0:
            raise ContractValidationError("offset_bytes must not be negative.")
        if self.nbytes is not None and int(self.nbytes) <= 0:
            raise ContractValidationError("nbytes must be positive when provided.")
        object.__setattr__(self, "width", int(self.width))
        object.__setattr__(self, "height", int(self.height))
        object.__setattr__(self, "name", require_nonempty(self.name, field_name="name"))
        object.__setattr__(self, "dtype", require_nonempty(self.dtype, field_name="dtype"))
        object.__setattr__(self, "offset_bytes", int(self.offset_bytes))
        if self.nbytes is not None:
            object.__setattr__(self, "nbytes", int(self.nbytes))


MaskPayload: TypeAlias = (
    DenseMaskPayload | RLEMaskPayload | PolygonMaskPayload | SharedMemoryMaskPayload
)


def normalize_mask_payload(payload: MaskPayload, pixel_space: PixelCoordinateSpace) -> np.ndarray:
    """Normalize a dense payload; other representations are schema-ready only in Step 1."""

    if payload.width != pixel_space.size.width or payload.height != pixel_space.size.height:
        raise ContractValidationError(
            "Mask payload dimensions do not match the declared pixel coordinate space."
        )
    if not isinstance(payload, DenseMaskPayload):
        raise UnsupportedMaskRepresentationError(
            f"Runtime normalization for {payload.representation.value!r} is deferred."
        )
    return payload.data


@dataclass(frozen=True, slots=True)
class InstanceMask:
    mask_id: str
    frame_reference: FrameReference
    pixel_space: PixelCoordinateSpace
    payload: MaskPayload
    source: MaskSource
    model_provenance: ModelProvenance
    revision: int = 0
    parent_mask_id: str = ""
    approval_status: ApprovalStatus = ApprovalStatus.NOT_REQUIRED
    approved_by: str = ""
    approved_timestamp: float | None = None
    confidence: float | None = None
    confidence_kind: ConfidenceKind = ConfidenceKind.NONE
    foreground_valid: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "mask_id", require_nonempty(self.mask_id, field_name="mask_id"))
        if self.pixel_space.frame_reference != self.frame_reference:
            raise ContractValidationError("Mask pixel space does not match frame_reference.")
        if self.payload.width != self.pixel_space.size.width or self.payload.height != self.pixel_space.size.height:
            raise ContractValidationError(
                "Mask payload dimensions do not match the declared pixel coordinate space."
            )
        if isinstance(self.revision, bool) or int(self.revision) < 0:
            raise ContractValidationError("revision must be a non-negative integer.")
        object.__setattr__(self, "revision", int(self.revision))
        parent = str(self.parent_mask_id).strip()
        if parent == self.mask_id:
            raise ContractValidationError("A mask cannot be its own parent.")
        object.__setattr__(self, "parent_mask_id", parent)
        approved_by = str(self.approved_by).strip()
        object.__setattr__(self, "approved_by", approved_by)
        if self.approved_timestamp is not None:
            object.__setattr__(
                self,
                "approved_timestamp",
                require_finite(self.approved_timestamp, field_name="approved_timestamp"),
            )

        if self.source is MaskSource.MANUAL_CORRECTION:
            if not parent:
                raise ContractValidationError("Manual mask revisions require parent_mask_id.")
            if self.revision <= 0:
                raise ContractValidationError("Manual mask revisions must have revision > 0.")
            if self.approval_status is ApprovalStatus.NOT_REQUIRED:
                raise ContractValidationError(
                    "Manual mask revisions require pending, approved or rejected status."
                )
            if self.approval_status is ApprovalStatus.APPROVED:
                if not approved_by or self.approved_timestamp is None:
                    raise ContractValidationError(
                        "Approved manual masks require approver and approval timestamp."
                    )
        else:
            if self.approval_status is not ApprovalStatus.NOT_REQUIRED:
                raise ContractValidationError(
                    "Automatic masks must use approval_status=NOT_REQUIRED."
                )
            if approved_by or self.approved_timestamp is not None:
                raise ContractValidationError(
                    "Automatic masks must not contain manual approval provenance."
                )

        if self.confidence_kind is ConfidenceKind.NONE:
            if self.confidence is not None:
                raise ContractValidationError(
                    "confidence must be None when confidence_kind is NONE."
                )
        elif self.confidence is None:
            raise ContractValidationError(
                "confidence is required when confidence_kind is not NONE."
            )
        elif self.confidence_kind is ConfidenceKind.CONFIDENCE:
            object.__setattr__(
                self, "confidence", require_probability(self.confidence, field_name="confidence")
            )
        else:
            uncertainty = require_finite(self.confidence, field_name="uncertainty")
            if uncertainty < 0.0:
                raise ContractValidationError("Mask uncertainty must not be negative.")
            object.__setattr__(self, "confidence", uncertainty)

        if isinstance(self.payload, DenseMaskPayload):
            derived_foreground = bool(np.any(self.payload.data))
            if bool(self.foreground_valid) != derived_foreground:
                raise ContractValidationError(
                    "foreground_valid disagrees with the dense mask payload."
                )
        object.__setattr__(self, "foreground_valid", bool(self.foreground_valid))

    @property
    def representation(self) -> MaskRepresentation:
        return self.payload.representation

    @property
    def is_approved_manual(self) -> bool:
        return (
            self.source is MaskSource.MANUAL_CORRECTION
            and self.approval_status is ApprovalStatus.APPROVED
            and self.foreground_valid
        )


def select_authoritative_mask(masks: tuple[InstanceMask, ...]) -> InstanceMask | None:
    """Apply approved manual authority before all automatic mask sources."""

    if not masks:
        return None
    ids = [mask.mask_id for mask in masks]
    if len(set(ids)) != len(ids):
        raise ContractValidationError("Mask IDs must be unique within an observation.")
    by_id = {mask.mask_id: mask for mask in masks}
    first = masks[0]
    for mask in masks:
        if mask.frame_reference != first.frame_reference:
            raise ContractValidationError("Observation masks must refer to the same frame.")
        if mask.pixel_space.space_id != first.pixel_space.space_id:
            raise ContractValidationError("Observation masks must use the same pixel space.")
        if mask.parent_mask_id and mask.parent_mask_id in by_id:
            parent = by_id[mask.parent_mask_id]
            if mask.revision <= parent.revision:
                raise ContractValidationError(
                    "A child mask revision must be greater than its parent revision."
                )

    approved_manual = [mask for mask in masks if mask.is_approved_manual]
    if approved_manual:
        highest = max(mask.revision for mask in approved_manual)
        winners = [mask for mask in approved_manual if mask.revision == highest]
        if len(winners) != 1:
            raise ContractValidationError(
                "Approved manual masks must have one authoritative highest revision."
            )
        return winners[0]

    priority = {
        MaskSource.NATIVE_INSTANCE: 0,
        MaskSource.PROMPTED_REFINEMENT: 1,
        MaskSource.DETECTION_BOX_FALLBACK: 2,
    }
    automatic = [
        mask
        for mask in masks
        if mask.source in priority and mask.foreground_valid
    ]
    if not automatic:
        return None
    return min(
        automatic,
        key=lambda mask: (
            priority[mask.source],
            -(mask.confidence if mask.confidence is not None else -1.0),
            -mask.revision,
            mask.mask_id,
        ),
    )


@dataclass(frozen=True, slots=True)
class PromptedMaskRefinement:
    refinement_id: str
    prompt: SegmentationPrompt
    output_mask: InstanceMask
    input_detection_id: str = ""
    input_mask_id: str = ""
    fallback_reason: str = ""
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "refinement_id", require_nonempty(self.refinement_id, field_name="refinement_id")
        )
        if self.output_mask.source is not MaskSource.PROMPTED_REFINEMENT:
            raise ContractValidationError(
                "PromptedMaskRefinement output must use PROMPTED_REFINEMENT source."
            )
        if not self.input_detection_id and not self.input_mask_id:
            raise ContractValidationError(
                "Prompted refinement requires an input detection or mask ID."
            )
        object.__setattr__(self, "input_detection_id", str(self.input_detection_id).strip())
        object.__setattr__(self, "input_mask_id", str(self.input_mask_id).strip())
        object.__setattr__(self, "fallback_reason", str(self.fallback_reason).strip())
        object.__setattr__(self, "warnings", tuple(str(value).strip() for value in self.warnings))


@dataclass(frozen=True, slots=True)
class ObjectObservation:
    observation_id: str
    frame_reference: FrameReference
    class_id: int
    class_name: str
    detection: Detection2D | None = None
    masks: tuple[InstanceMask, ...] = ()
    selected_mask_id: str = ""
    track_id: str = ""
    border_contacts: tuple[str, ...] = ()
    estimated_occlusion_fraction: float | None = None
    warnings: tuple[str, ...] = ()
    refinements: tuple[PromptedMaskRefinement, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "observation_id", require_nonempty(self.observation_id, field_name="observation_id")
        )
        if isinstance(self.class_id, bool) or int(self.class_id) < 0:
            raise ContractValidationError("class_id must be a non-negative integer.")
        object.__setattr__(self, "class_id", int(self.class_id))
        object.__setattr__(
            self, "class_name", require_nonempty(self.class_name, field_name="class_name")
        )
        if self.detection is not None and self.detection.frame_reference != self.frame_reference:
            raise ContractValidationError("Detection and observation refer to different frames.")
        for mask in self.masks:
            if mask.frame_reference != self.frame_reference:
                raise ContractValidationError("Mask and observation refer to different frames.")
        refinement_ids = [value.refinement_id for value in self.refinements]
        if len(refinement_ids) != len(set(refinement_ids)):
            raise ContractValidationError("Refinement IDs must be unique within an observation.")
        mask_ids = {mask.mask_id for mask in self.masks}
        for refinement in self.refinements:
            if refinement.output_mask.frame_reference != self.frame_reference:
                raise ContractValidationError("Refinement and observation refer to different frames.")
            if refinement.output_mask.mask_id not in mask_ids:
                raise ContractValidationError("Every refinement output mask must be present in masks.")
            if self.detection is not None and refinement.input_detection_id:
                if refinement.input_detection_id != self.detection.detection_id:
                    raise ContractValidationError("Refinement input detection does not match observation detection.")
        selected = select_authoritative_mask(tuple(self.masks))
        expected_id = selected.mask_id if selected is not None else ""
        if self.selected_mask_id and self.selected_mask_id != expected_id:
            raise ContractValidationError(
                "selected_mask_id disagrees with authoritative manual/automatic mask precedence."
            )
        object.__setattr__(self, "selected_mask_id", expected_id)
        object.__setattr__(self, "track_id", str(self.track_id).strip())
        object.__setattr__(
            self, "border_contacts", tuple(str(value).strip() for value in self.border_contacts)
        )
        if self.estimated_occlusion_fraction is not None:
            object.__setattr__(
                self,
                "estimated_occlusion_fraction",
                require_probability(
                    self.estimated_occlusion_fraction,
                    field_name="estimated_occlusion_fraction",
                ),
            )
        object.__setattr__(self, "warnings", tuple(str(value).strip() for value in self.warnings))

    @property
    def selected_mask(self) -> InstanceMask | None:
        return next(
            (mask for mask in self.masks if mask.mask_id == self.selected_mask_id),
            None,
        )


@dataclass(frozen=True, slots=True)
class PerceptionFrameResult:
    frame_reference: FrameReference
    observations: tuple[ObjectObservation, ...]
    model_provenance: tuple[ModelProvenance, ...]
    status: PerceptionResultStatus = PerceptionResultStatus.SUCCESS
    processing_ms: float = 0.0
    warnings: tuple[str, ...] = ()
    failure: str = ""

    def __post_init__(self) -> None:
        observation_ids = [value.observation_id for value in self.observations]
        if len(set(observation_ids)) != len(observation_ids):
            raise ContractValidationError("Observation IDs must be unique within a frame result.")
        for observation in self.observations:
            if observation.frame_reference != self.frame_reference:
                raise ContractValidationError(
                    "Every observation must match the frame result reference."
                )
        processing_ms = require_finite(self.processing_ms, field_name="processing_ms")
        if processing_ms < 0.0:
            raise ContractValidationError("processing_ms must not be negative.")
        if self.status is PerceptionResultStatus.FAILED and not str(self.failure).strip():
            raise ContractValidationError("FAILED perception results require failure details.")
        object.__setattr__(self, "processing_ms", processing_ms)
        object.__setattr__(self, "warnings", tuple(str(value).strip() for value in self.warnings))
        object.__setattr__(self, "failure", str(self.failure).strip())

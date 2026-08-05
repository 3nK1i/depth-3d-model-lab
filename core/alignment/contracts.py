"""Dependency-light contracts for RGB, mask, depth, confidence and intrinsics alignment."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

import numpy as np

from core.contracts.common import FrameReference, ModelProvenance, ProcessingStep
from core.contracts.spatial import CameraIntrinsics, ImageTransform2D, PixelCoordinateSpace, PixelSpaceRole
from core.contracts.validation import ContractValidationError, readonly_array, require_finite, require_nonempty
from core.models.metadata import ConfidenceKind, DepthScale
from core.perception.result import (
    ApprovalStatus,
    BoundingBox2D,
    MaskSource,
    PerceptionFrameResult,
)


class AlignmentTargetGrid(str, Enum):
    """Authoritative output grid used by the alignment pipeline."""

    NATIVE_DEPTH = "native_depth"


class AlignmentStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


class ResamplingMethod(str, Enum):
    AUTO = "auto"
    NEAREST = "nearest"
    LINEAR = "linear"
    AREA = "area"


@dataclass(frozen=True, slots=True)
class AlignmentPolicy:
    """Step 3 alignment policy.

    Depth is never resampled in the default pipeline. RGB and masks are moved
    into the native depth grid using explicit pixel-space transforms.
    """

    target_grid: AlignmentTargetGrid = AlignmentTargetGrid.NATIVE_DEPTH
    allow_implicit_resize: bool = False
    rgb_interpolation: ResamplingMethod = ResamplingMethod.AUTO
    mask_interpolation: ResamplingMethod = ResamplingMethod.NEAREST
    confidence_interpolation: ResamplingMethod = ResamplingMethod.LINEAR
    preserve_overlapping_masks: bool = True
    warn_on_nonuniform_resize: bool = True

    def __post_init__(self) -> None:
        if self.target_grid is not AlignmentTargetGrid.NATIVE_DEPTH:
            raise ContractValidationError("Step 3 supports only the native depth target grid.")
        if self.mask_interpolation is not ResamplingMethod.NEAREST:
            raise ContractValidationError("Boolean masks must use nearest-neighbour resampling.")
        if not self.preserve_overlapping_masks:
            raise ContractValidationError("Step 3 must preserve overlapping instance masks.")


@dataclass(frozen=True, slots=True)
class DepthAlignmentInput:
    """Normalized depth-side input for one captured frame."""

    frame_reference: FrameReference
    depth_space: PixelCoordinateSpace
    depth: np.ndarray
    valid_mask: np.ndarray | None
    confidence: np.ndarray | None
    confidence_kind: ConfidenceKind
    intrinsics: CameraIntrinsics
    depth_scale: DepthScale
    depth_model_provenance: ModelProvenance
    point_stride: int = 1

    def __post_init__(self) -> None:
        if self.depth_space.frame_reference != self.frame_reference:
            raise ContractValidationError("Depth pixel space does not match frame_reference.")
        if self.depth_space.role is not PixelSpaceRole.DEPTH:
            raise ContractValidationError("depth_space must use PixelSpaceRole.DEPTH.")

        depth = readonly_array(self.depth, field_name="depth", ndim=2)
        if not np.issubdtype(depth.dtype, np.number):
            raise ContractValidationError("depth must be numeric.")
        expected_shape = (self.depth_space.size.height, self.depth_space.size.width)
        if depth.shape != expected_shape:
            raise ContractValidationError(
                f"depth shape {depth.shape} does not match depth space {expected_shape}."
            )

        valid_mask = None
        if self.valid_mask is not None:
            valid_mask = readonly_array(
                self.valid_mask,
                field_name="valid_mask",
                dtype=np.bool_,
                shape=expected_shape,
            )

        confidence = None
        if self.confidence_kind is ConfidenceKind.NONE:
            if self.confidence is not None:
                raise ContractValidationError(
                    "confidence must be None when confidence_kind is NONE."
                )
        else:
            if self.confidence is None:
                raise ContractValidationError(
                    "confidence is required when confidence_kind is not NONE."
                )
            confidence = readonly_array(
                self.confidence,
                field_name="confidence",
                shape=expected_shape,
            )
            if not np.issubdtype(confidence.dtype, np.number):
                raise ContractValidationError("confidence must be numeric.")
            finite = np.isfinite(confidence)
            values = confidence[finite]
            if self.confidence_kind is ConfidenceKind.CONFIDENCE:
                if values.size and (np.any(values < 0.0) or np.any(values > 1.0)):
                    raise ContractValidationError(
                        "Finite confidence values must remain within [0, 1]."
                    )
            elif values.size and np.any(values < 0.0):
                raise ContractValidationError(
                    "Finite uncertainty values must not be negative."
                )

        if self.intrinsics.pixel_space_id != self.depth_space.space_id:
            raise ContractValidationError(
                "Camera intrinsics must reference the native depth pixel space."
            )
        if not isinstance(self.depth_scale, DepthScale):
            raise TypeError("depth_scale must be a DepthScale.")
        if not isinstance(self.confidence_kind, ConfidenceKind):
            raise TypeError("confidence_kind must be a ConfidenceKind.")
        if isinstance(self.point_stride, bool) or int(self.point_stride) <= 0:
            raise ContractValidationError("point_stride must be a positive integer.")

        object.__setattr__(self, "depth", depth)
        object.__setattr__(self, "valid_mask", valid_mask)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "point_stride", int(self.point_stride))

    def __reduce__(self):
        return (
            type(self),
            (
                self.frame_reference,
                self.depth_space,
                self.depth,
                self.valid_mask,
                self.confidence,
                self.confidence_kind,
                self.intrinsics,
                self.depth_scale,
                self.depth_model_provenance,
                self.point_stride,
            ),
        )


@dataclass(frozen=True, slots=True)
class AlignmentRequest:
    """One authoritative request joining matching depth and perception results."""

    frame_reference: FrameReference
    rgb: np.ndarray
    rgb_space: PixelCoordinateSpace
    depth_input: DepthAlignmentInput
    perception_result: PerceptionFrameResult
    transforms: tuple[ImageTransform2D, ...] = ()
    policy: AlignmentPolicy = field(default_factory=AlignmentPolicy)

    def __post_init__(self) -> None:
        rgb = readonly_array(self.rgb, field_name="rgb", dtype=np.uint8, ndim=3)
        expected = (self.rgb_space.size.height, self.rgb_space.size.width, 3)
        if rgb.shape != expected:
            raise ContractValidationError(
                f"rgb shape {rgb.shape} does not match RGB pixel space {expected}."
            )
        if self.rgb_space.role is not PixelSpaceRole.RGB:
            raise ContractValidationError("rgb_space must use PixelSpaceRole.RGB.")
        if self.rgb_space.frame_reference != self.frame_reference:
            raise ContractValidationError("RGB pixel space does not match frame_reference.")
        if self.depth_input.frame_reference != self.frame_reference:
            raise ContractValidationError("Depth input does not match frame_reference.")
        if self.perception_result.frame_reference != self.frame_reference:
            raise ContractValidationError("Perception result does not match frame_reference.")
        if not isinstance(self.policy, AlignmentPolicy):
            raise TypeError("policy must be an AlignmentPolicy.")

        transforms = tuple(self.transforms)
        pairs: set[tuple[str, str]] = set()
        for transform in transforms:
            if transform.source_space.frame_reference != self.frame_reference:
                raise ContractValidationError("Transform source space refers to another frame.")
            if transform.destination_space.frame_reference != self.frame_reference:
                raise ContractValidationError("Transform destination space refers to another frame.")
            pair = (transform.source_space.space_id, transform.destination_space.space_id)
            if pair in pairs:
                raise ContractValidationError(
                    f"Duplicate transform declared for {pair[0]!r} -> {pair[1]!r}."
                )
            pairs.add(pair)

        object.__setattr__(self, "rgb", rgb)
        object.__setattr__(self, "transforms", transforms)

    def __reduce__(self):
        return (
            type(self),
            (
                self.frame_reference,
                self.rgb,
                self.rgb_space,
                self.depth_input,
                self.perception_result,
                self.transforms,
                self.policy,
            ),
        )


@dataclass(frozen=True, slots=True)
class AlignedObjectObservation:
    """One perception observation represented on the native depth grid."""

    observation_id: str
    class_id: int
    class_name: str
    target_space: PixelCoordinateSpace
    source_detection_id: str = ""
    aligned_detection_box: BoundingBox2D | None = None
    source_mask_id: str = ""
    source_mask_space_id: str = ""
    aligned_mask: np.ndarray | None = None
    source_foreground_pixels: int = 0
    aligned_foreground_pixels: int = 0
    foreground_retention_ratio: float = 0.0
    border_contacts: tuple[str, ...] = ()
    mask_source: MaskSource | None = None
    mask_revision: int = 0
    approval_status: ApprovalStatus = ApprovalStatus.NOT_REQUIRED
    mask_model_provenance: ModelProvenance | None = None
    detection_model_provenance: ModelProvenance | None = None
    processing_steps: tuple[ProcessingStep, ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observation_id",
            require_nonempty(self.observation_id, field_name="observation_id"),
        )
        if isinstance(self.class_id, bool) or int(self.class_id) < 0:
            raise ContractValidationError("class_id must be a non-negative integer.")
        object.__setattr__(self, "class_id", int(self.class_id))
        object.__setattr__(
            self, "class_name", require_nonempty(self.class_name, field_name="class_name")
        )
        if self.target_space.role is not PixelSpaceRole.DEPTH:
            raise ContractValidationError("Aligned observations must target the depth pixel space.")

        mask = None
        expected_shape = (self.target_space.size.height, self.target_space.size.width)
        if self.aligned_mask is not None:
            mask = readonly_array(
                self.aligned_mask,
                field_name="aligned_mask",
                dtype=np.bool_,
                shape=expected_shape,
            )
            derived_count = int(np.count_nonzero(mask))
            if int(self.aligned_foreground_pixels) != derived_count:
                raise ContractValidationError(
                    "aligned_foreground_pixels disagrees with aligned_mask."
                )
        elif int(self.aligned_foreground_pixels) != 0:
            raise ContractValidationError(
                "aligned_foreground_pixels must be zero when aligned_mask is unavailable."
            )

        source_count = int(self.source_foreground_pixels)
        aligned_count = int(self.aligned_foreground_pixels)
        if source_count < 0 or aligned_count < 0:
            raise ContractValidationError("Foreground pixel counts must not be negative.")
        ratio = require_finite(
            self.foreground_retention_ratio, field_name="foreground_retention_ratio"
        )
        if ratio < 0.0 or ratio > 1.0:
            raise ContractValidationError("foreground_retention_ratio must be within [0, 1].")
        if self.aligned_detection_box is not None:
            if self.aligned_detection_box.pixel_space_id != self.target_space.space_id:
                raise ContractValidationError(
                    "Aligned detection box must reference the target depth pixel space."
                )
        if isinstance(self.mask_revision, bool) or int(self.mask_revision) < 0:
            raise ContractValidationError("mask_revision must be a non-negative integer.")

        object.__setattr__(self, "source_detection_id", str(self.source_detection_id).strip())
        object.__setattr__(self, "source_mask_id", str(self.source_mask_id).strip())
        object.__setattr__(self, "source_mask_space_id", str(self.source_mask_space_id).strip())
        object.__setattr__(self, "aligned_mask", mask)
        object.__setattr__(self, "source_foreground_pixels", source_count)
        object.__setattr__(self, "aligned_foreground_pixels", aligned_count)
        object.__setattr__(self, "foreground_retention_ratio", ratio)
        object.__setattr__(self, "mask_revision", int(self.mask_revision))
        object.__setattr__(
            self, "border_contacts", tuple(str(value).strip() for value in self.border_contacts)
        )
        object.__setattr__(self, "processing_steps", tuple(self.processing_steps))
        object.__setattr__(self, "warnings", tuple(str(value).strip() for value in self.warnings))

    def __reduce__(self):
        return (
            type(self),
            (
                self.observation_id,
                self.class_id,
                self.class_name,
                self.target_space,
                self.source_detection_id,
                self.aligned_detection_box,
                self.source_mask_id,
                self.source_mask_space_id,
                self.aligned_mask,
                self.source_foreground_pixels,
                self.aligned_foreground_pixels,
                self.foreground_retention_ratio,
                self.border_contacts,
                self.mask_source,
                self.mask_revision,
                self.approval_status,
                self.mask_model_provenance,
                self.detection_model_provenance,
                self.processing_steps,
                self.warnings,
            ),
        )


@dataclass(frozen=True, slots=True)
class AlignedFrameBundle:
    """Authoritative Step 3 output on the native depth pixel grid."""

    frame_reference: FrameReference
    target_depth_space: PixelCoordinateSpace
    aligned_rgb: np.ndarray
    depth: np.ndarray
    depth_valid_mask: np.ndarray
    rgb_support_mask: np.ndarray
    confidence: np.ndarray | None
    confidence_valid_mask: np.ndarray
    confidence_kind: ConfidenceKind
    intrinsics: CameraIntrinsics
    depth_scale: DepthScale
    point_stride: int
    depth_model_provenance: ModelProvenance
    perception_model_provenance: tuple[ModelProvenance, ...]
    aligned_observations: tuple[AlignedObjectObservation, ...]
    transforms: tuple[ImageTransform2D, ...]
    status: AlignmentStatus = AlignmentStatus.SUCCESS
    warnings: tuple[str, ...] = ()
    processing_steps: tuple[ProcessingStep, ...] = ()

    def __post_init__(self) -> None:
        if self.target_depth_space.frame_reference != self.frame_reference:
            raise ContractValidationError("Target depth space does not match frame_reference.")
        if self.target_depth_space.role is not PixelSpaceRole.DEPTH:
            raise ContractValidationError("target_depth_space must use PixelSpaceRole.DEPTH.")
        shape = (self.target_depth_space.size.height, self.target_depth_space.size.width)
        rgb = readonly_array(
            self.aligned_rgb,
            field_name="aligned_rgb",
            dtype=np.uint8,
            shape=(shape[0], shape[1], 3),
        )
        depth = readonly_array(self.depth, field_name="depth", shape=shape)
        depth_valid = readonly_array(
            self.depth_valid_mask,
            field_name="depth_valid_mask",
            dtype=np.bool_,
            shape=shape,
        )
        rgb_support = readonly_array(
            self.rgb_support_mask,
            field_name="rgb_support_mask",
            dtype=np.bool_,
            shape=shape,
        )
        confidence = None
        if self.confidence is not None:
            confidence = readonly_array(self.confidence, field_name="confidence", shape=shape)
        confidence_valid = readonly_array(
            self.confidence_valid_mask,
            field_name="confidence_valid_mask",
            dtype=np.bool_,
            shape=shape,
        )
        if self.confidence_kind is ConfidenceKind.NONE and confidence is not None:
            raise ContractValidationError(
                "confidence must be None when confidence_kind is NONE."
            )
        if self.confidence_kind is not ConfidenceKind.NONE and confidence is None:
            raise ContractValidationError(
                "confidence is required when confidence_kind is not NONE."
            )
        if self.intrinsics.pixel_space_id != self.target_depth_space.space_id:
            raise ContractValidationError("Intrinsics must reference the target depth space.")
        if isinstance(self.point_stride, bool) or int(self.point_stride) <= 0:
            raise ContractValidationError("point_stride must be a positive integer.")

        observation_ids = [value.observation_id for value in self.aligned_observations]
        if len(observation_ids) != len(set(observation_ids)):
            raise ContractValidationError("Aligned observation IDs must be unique.")
        for observation in self.aligned_observations:
            if observation.target_space != self.target_depth_space:
                raise ContractValidationError(
                    "Every aligned observation must use target_depth_space."
                )

        object.__setattr__(self, "aligned_rgb", rgb)
        object.__setattr__(self, "depth", depth)
        object.__setattr__(self, "depth_valid_mask", depth_valid)
        object.__setattr__(self, "rgb_support_mask", rgb_support)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "confidence_valid_mask", confidence_valid)
        object.__setattr__(self, "point_stride", int(self.point_stride))
        object.__setattr__(self, "perception_model_provenance", tuple(self.perception_model_provenance))
        object.__setattr__(self, "aligned_observations", tuple(self.aligned_observations))
        object.__setattr__(self, "transforms", tuple(self.transforms))
        object.__setattr__(self, "warnings", tuple(str(value).strip() for value in self.warnings))
        object.__setattr__(self, "processing_steps", tuple(self.processing_steps))

    def __reduce__(self):
        return (
            type(self),
            (
                self.frame_reference,
                self.target_depth_space,
                self.aligned_rgb,
                self.depth,
                self.depth_valid_mask,
                self.rgb_support_mask,
                self.confidence,
                self.confidence_valid_mask,
                self.confidence_kind,
                self.intrinsics,
                self.depth_scale,
                self.point_stride,
                self.depth_model_provenance,
                self.perception_model_provenance,
                self.aligned_observations,
                self.transforms,
                self.status,
                self.warnings,
                self.processing_steps,
            ),
        )


def unique_model_provenance(values: Iterable[ModelProvenance]) -> tuple[ModelProvenance, ...]:
    """Return deterministic provenance records without duplicate model IDs/configs."""

    result: list[ModelProvenance] = []
    seen: set[tuple[str, str, str]] = set()
    for value in values:
        key = (value.model_id, value.checkpoint, value.runtime_config_digest)
        if key not in seen:
            seen.add(key)
            result.append(value)
    return tuple(result)

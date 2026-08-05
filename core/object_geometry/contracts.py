"""Immutable contracts for object-specific 3D point extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from core.contracts.common import FrameReference, ProcessingStep
from core.contracts.spatial import CoordinateFrame
from core.contracts.validation import ContractValidationError, require_nonempty
from core.measurement.contracts import ObjectPointCloud, RejectionReason


class ObjectPointExtractionStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    REJECTED = "rejected"


class PointCoordinateSemantics(str, Enum):
    METRIC_METRES = "metric_metres"
    RELATIVE_MODEL_UNITS = "relative_model_units"
    SYNTHETIC_UNITS = "synthetic_units"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ExtractionStageCounts:
    selected_mask_pixels: int = 0
    largest_component_pixels: int = 0
    eroded_mask_pixels: int = 0
    depth_valid_pixels: int = 0
    confidence_valid_pixels: int = 0
    depth_range_pixels: int = 0
    discontinuity_retained_pixels: int = 0
    raw_projected_points: int = 0
    outlier_retained_points: int = 0
    plane_retained_points: int = 0
    cluster_retained_points: int = 0
    output_points: int = 0

    def __post_init__(self) -> None:
        values = tuple(int(getattr(self, name)) for name in self.__dataclass_fields__)
        if any(value < 0 for value in values):
            raise ContractValidationError("Extraction stage counts must not be negative.")
        for name, value in zip(self.__dataclass_fields__, values, strict=True):
            object.__setattr__(self, name, value)
        chain = (
            self.selected_mask_pixels,
            self.largest_component_pixels,
            self.eroded_mask_pixels,
            self.depth_valid_pixels,
            self.confidence_valid_pixels,
            self.depth_range_pixels,
            self.discontinuity_retained_pixels,
            self.raw_projected_points,
            self.outlier_retained_points,
            self.plane_retained_points,
            self.cluster_retained_points,
            self.output_points,
        )
        if any(later > earlier for earlier, later in zip(chain, chain[1:])):
            raise ContractValidationError(
                "Extraction stage counts must be monotonically non-increasing."
            )

    def as_dict(self) -> dict[str, int]:
        return {name: int(getattr(self, name)) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class ObjectPointExtractionResult:
    observation_id: str
    frame_reference: FrameReference
    status: ObjectPointExtractionStatus
    coordinate_frame: CoordinateFrame
    coordinate_semantics: PointCoordinateSemantics
    raw_cloud: ObjectPointCloud | None = None
    filtered_cloud: ObjectPointCloud | None = None
    stage_counts: ExtractionStageCounts = field(default_factory=ExtractionStageCounts)
    rejection_reasons: tuple[RejectionReason, ...] = ()
    processing_steps: tuple[ProcessingStep, ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "observation_id", require_nonempty(self.observation_id, field_name="observation_id")
        )
        if self.coordinate_frame.source_frame != self.frame_reference:
            raise ContractValidationError("coordinate_frame must reference frame_reference.")
        raw_cloud = self._normalize_cloud(self.raw_cloud) if self.raw_cloud is not None else None
        filtered_cloud = self._normalize_cloud(self.filtered_cloud) if self.filtered_cloud is not None else None
        if raw_cloud is not None:
            self._validate_cloud(raw_cloud, "raw_cloud")
        if filtered_cloud is not None:
            self._validate_cloud(filtered_cloud, "filtered_cloud")
        object.__setattr__(self, "raw_cloud", raw_cloud)
        object.__setattr__(self, "filtered_cloud", filtered_cloud)
        reasons = tuple(dict.fromkeys(self.rejection_reasons))
        warnings = tuple(value for value in (str(v).strip() for v in self.warnings) if value)
        steps = tuple(self.processing_steps)
        if self.status is ObjectPointExtractionStatus.REJECTED:
            if filtered_cloud is not None:
                raise ContractValidationError("Rejected extraction results cannot expose filtered_cloud.")
            if not reasons:
                raise ContractValidationError("Rejected extraction results require rejection reasons.")
        else:
            if filtered_cloud is None:
                raise ContractValidationError("Successful or partial extraction requires filtered_cloud.")
            if reasons:
                raise ContractValidationError(
                    "Successful or partial extraction cannot contain authoritative rejection reasons."
                )
        if raw_cloud is None and self.stage_counts.raw_projected_points:
            raise ContractValidationError("raw_cloud is required when raw_projected_points is non-zero.")
        if raw_cloud is not None and raw_cloud.retained_point_count != self.stage_counts.raw_projected_points:
            raise ContractValidationError("raw_cloud count disagrees with stage_counts.")
        if filtered_cloud is not None and filtered_cloud.retained_point_count != self.stage_counts.output_points:
            raise ContractValidationError("filtered_cloud count disagrees with stage_counts.")
        object.__setattr__(self, "rejection_reasons", reasons)
        object.__setattr__(self, "processing_steps", steps)
        object.__setattr__(self, "warnings", warnings)

    @staticmethod
    def _normalize_cloud(cloud: ObjectPointCloud) -> ObjectPointCloud:
        return ObjectPointCloud(
            observation_id=cloud.observation_id,
            frame_reference=cloud.frame_reference,
            coordinate_frame_id=cloud.coordinate_frame_id,
            points_xyz_m=cloud.points_xyz_m,
            source_pixel_space_id=cloud.source_pixel_space_id,
            source_pixel_space_size=cloud.source_pixel_space_size,
            source_pixel_indices_xy=cloud.source_pixel_indices_xy,
            colors_rgb=cloud.colors_rgb,
            point_weights=cloud.point_weights,
            raw_point_count=cloud.raw_point_count,
            retained_point_count=cloud.retained_point_count,
            mask_id=cloud.mask_id,
            depth_model_provenance=cloud.depth_model_provenance,
            depth_scale=cloud.depth_scale,
            intrinsics_source=cloud.intrinsics_source,
            scale_recovery_status=cloud.scale_recovery_status,
            processing_steps=cloud.processing_steps,
        )

    def __reduce__(self):
        return (
            type(self),
            (
                self.observation_id, self.frame_reference, self.status,
                self.coordinate_frame, self.coordinate_semantics, self.raw_cloud,
                self.filtered_cloud, self.stage_counts, self.rejection_reasons,
                self.processing_steps, self.warnings,
            ),
        )

    def _validate_cloud(self, cloud: ObjectPointCloud, field_name: str) -> None:
        if cloud.observation_id != self.observation_id:
            raise ContractValidationError(f"{field_name} observation_id mismatch.")
        if cloud.frame_reference != self.frame_reference:
            raise ContractValidationError(f"{field_name} frame_reference mismatch.")
        if cloud.coordinate_frame_id != self.coordinate_frame.frame_id:
            raise ContractValidationError(f"{field_name} coordinate_frame_id mismatch.")


@dataclass(frozen=True, slots=True)
class ObjectPointExtractionBatchResult:
    frame_reference: FrameReference
    coordinate_frame: CoordinateFrame
    object_results: tuple[ObjectPointExtractionResult, ...]
    status: ObjectPointExtractionStatus
    processing_steps: tuple[ProcessingStep, ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.coordinate_frame.source_frame != self.frame_reference:
            raise ContractValidationError("Batch coordinate_frame must reference frame_reference.")
        results = tuple(self.object_results)
        ids = [result.observation_id for result in results]
        if len(ids) != len(set(ids)):
            raise ContractValidationError("Extraction batch observation IDs must be unique.")
        for result in results:
            if result.frame_reference != self.frame_reference:
                raise ContractValidationError("Extraction result frame_reference mismatch.")
            if result.coordinate_frame != self.coordinate_frame:
                raise ContractValidationError("Extraction result coordinate_frame mismatch.")
        expected = _derive_batch_status(results)
        if self.status is not expected:
            raise ContractValidationError(
                f"Batch status {self.status.value!r} disagrees with object results; expected {expected.value!r}."
            )
        object.__setattr__(self, "object_results", results)
        object.__setattr__(self, "processing_steps", tuple(self.processing_steps))
        object.__setattr__(
            self, "warnings", tuple(value for value in (str(v).strip() for v in self.warnings) if value)
        )

    def __reduce__(self):
        return (
            type(self),
            (
                self.frame_reference, self.coordinate_frame, self.object_results,
                self.status, self.processing_steps, self.warnings,
            ),
        )



def _derive_batch_status(
    results: tuple[ObjectPointExtractionResult, ...],
) -> ObjectPointExtractionStatus:
    if not results or all(result.status is ObjectPointExtractionStatus.REJECTED for result in results):
        return ObjectPointExtractionStatus.REJECTED
    if all(result.status is ObjectPointExtractionStatus.SUCCESS for result in results):
        return ObjectPointExtractionStatus.SUCCESS
    return ObjectPointExtractionStatus.PARTIAL


def derive_batch_status(
    results: tuple[ObjectPointExtractionResult, ...],
) -> ObjectPointExtractionStatus:
    """Public deterministic batch-status helper."""

    return _derive_batch_status(tuple(results))

"""Immutable object geometry, measurement and uncertainty contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np

from core.contracts.common import (
    FrameReference,
    ImageSize,
    ModelProvenance,
    ProcessingStep,
)
from core.contracts.spatial import CalibrationStatus
from core.contracts.validation import (
    ContractValidationError,
    JsonValue,
    metadata_as_dict,
    normalize_metadata,
    readonly_array,
    require_finite,
    require_nonempty,
    require_probability,
)
from core.models.metadata import DepthScale, IntrinsicsSource
from core.tracking.contracts import TrackingState


class BoundingBoxKind(str, Enum):
    AXIS_ALIGNED = "axis_aligned"
    CAMERA_ALIGNED = "camera_aligned"
    ORIENTED = "oriented"


class ShapeKind(str, Enum):
    GENERIC_BOX = "generic_box"
    CUBOID = "cuboid"
    CYLINDER = "cylinder"
    PLANE = "plane"
    CONVEX_HULL = "convex_hull"
    IRREGULAR = "irregular"


class MeasurementKind(str, Enum):
    WIDTH = "width"
    HEIGHT = "height"
    DEPTH = "depth"
    CAMERA_DISTANCE = "camera_distance"
    CENTRE_X = "centre_x"
    CENTRE_Y = "centre_y"
    CENTRE_Z = "centre_z"
    ORIENTATION_ROLL = "orientation_roll"
    ORIENTATION_PITCH = "orientation_pitch"
    ORIENTATION_YAW = "orientation_yaw"
    VOLUME = "volume"


class MeasurementStatus(str, Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    REJECTED = "rejected"


class ReliabilityLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    REJECTED = "rejected"
    UNAVAILABLE = "unavailable"


class UncertaintyBasis(str, Enum):
    HEURISTIC_PERTURBATION = "heuristic_perturbation"
    TEMPORAL_OBSERVATION = "temporal_observation"
    MODEL_PROVIDED = "model_provided"
    EMPIRICALLY_CALIBRATED = "empirically_calibrated"
    UNKNOWN = "unknown"

    @property
    def supports_coverage_probability(self) -> bool:
        return self is UncertaintyBasis.EMPIRICALLY_CALIBRATED


class UncertaintyComponentKind(str, Enum):
    DEPTH = "depth_uncertainty"
    MASK_BOUNDARY = "mask_boundary_uncertainty"
    INTRINSICS = "intrinsics_uncertainty"
    DEPTH_DISCONTINUITY = "depth_discontinuity_sensitivity"
    POINT_CLOUD_OUTLIERS = "point_cloud_outlier_sensitivity"
    FITTING = "fitting_sensitivity"
    TEMPORAL = "temporal_variation"
    SCALE = "scale_uncertainty"
    OCCLUSION = "occlusion"
    BORDER_CONTACT = "border_contact"
    OTHER = "other"


class ScaleRecoveryStatus(str, Enum):
    NOT_REQUIRED = "not_required"
    NOT_ATTEMPTED = "not_attempted"
    SUCCEEDED = "succeeded"
    FAILED = "failed"



class RejectionReason(str, Enum):
    RELATIVE_DEPTH_WITHOUT_SCALE = "relative_depth_without_valid_scale"
    MISSING_OR_INVALID_INTRINSICS = "missing_or_invalid_intrinsics"
    TOO_FEW_VALID_POINTS = "too_few_valid_object_points"
    BORDER_CONTACT = "object_touches_image_border"
    EXCESSIVE_OCCLUSION = "excessive_occlusion"
    UNSTABLE_SEGMENTATION = "unstable_segmentation"
    INVALID_DEPTH_SCALE = "invalid_depth_scale"
    EXCESSIVE_UNCERTAINTY = "excessive_uncertainty"
    REFLECTIVE_OR_TRANSPARENT = "reflective_or_transparent_geometry_unreliable"
    FITTING_FAILED = "shape_or_box_fitting_failed"
    INVALID_COORDINATE_FRAME = "invalid_coordinate_frame_relationship"
    MASK_UNAVAILABLE = "validated_mask_unavailable"
    OTHER = "other"


class LowerBoundPolicy(str, Enum):
    REJECT_NEGATIVE = "reject_negative"
    CLIP_TO_ZERO = "clip_to_zero"
    ALLOW_NEGATIVE = "allow_negative"


_NONNEGATIVE_MEASUREMENTS = {
    MeasurementKind.WIDTH,
    MeasurementKind.HEIGHT,
    MeasurementKind.DEPTH,
    MeasurementKind.CAMERA_DISTANCE,
    MeasurementKind.VOLUME,
}

_EXPECTED_UNITS = {
    MeasurementKind.WIDTH: "m",
    MeasurementKind.HEIGHT: "m",
    MeasurementKind.DEPTH: "m",
    MeasurementKind.CAMERA_DISTANCE: "m",
    MeasurementKind.CENTRE_X: "m",
    MeasurementKind.CENTRE_Y: "m",
    MeasurementKind.CENTRE_Z: "m",
    MeasurementKind.ORIENTATION_ROLL: "rad",
    MeasurementKind.ORIENTATION_PITCH: "rad",
    MeasurementKind.ORIENTATION_YAW: "rad",
    MeasurementKind.VOLUME: "m3",
}


@dataclass(frozen=True, slots=True)
class ObjectPointCloud:
    observation_id: str
    frame_reference: FrameReference
    coordinate_frame_id: str
    points_xyz_m: np.ndarray
    source_pixel_space_id: str
    source_pixel_space_size: ImageSize
    source_pixel_indices_xy: np.ndarray | None = None
    colors_rgb: np.ndarray | None = None
    point_weights: np.ndarray | None = None
    raw_point_count: int = 0
    retained_point_count: int = 0
    mask_id: str = ""
    depth_model_provenance: ModelProvenance | None = None
    depth_scale: DepthScale = DepthScale.UNKNOWN
    intrinsics_source: IntrinsicsSource = IntrinsicsSource.NONE
    scale_recovery_status: ScaleRecoveryStatus = ScaleRecoveryStatus.NOT_ATTEMPTED
    processing_steps: tuple[ProcessingStep, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "observation_id", require_nonempty(self.observation_id, field_name="observation_id")
        )
        object.__setattr__(
            self,
            "coordinate_frame_id",
            require_nonempty(self.coordinate_frame_id, field_name="coordinate_frame_id"),
        )
        object.__setattr__(
            self,
            "source_pixel_space_id",
            require_nonempty(self.source_pixel_space_id, field_name="source_pixel_space_id"),
        )
        if not isinstance(self.source_pixel_space_size, ImageSize):
            raise TypeError("source_pixel_space_size must be an ImageSize.")
        points = readonly_array(
            self.points_xyz_m,
            field_name="points_xyz_m",
            dtype=np.float64,
            ndim=2,
            finite=True,
        )
        if points.shape[1:] != (3,):
            raise ContractValidationError("points_xyz_m must have shape N×3.")
        count = int(points.shape[0])
        if count <= 0:
            raise ContractValidationError("ObjectPointCloud must contain at least one point.")
        object.__setattr__(self, "points_xyz_m", points)

        if self.source_pixel_indices_xy is not None:
            indices = readonly_array(
                self.source_pixel_indices_xy,
                field_name="source_pixel_indices_xy",
                dtype=np.int64,
                shape=(count, 2),
            )
            if (
                np.any(indices[:, 0] < 0)
                or np.any(indices[:, 0] >= self.source_pixel_space_size.width)
                or np.any(indices[:, 1] < 0)
                or np.any(indices[:, 1] >= self.source_pixel_space_size.height)
            ):
                raise ContractValidationError(
                    "source_pixel_indices_xy are outside source_pixel_space_id dimensions."
                )
            object.__setattr__(self, "source_pixel_indices_xy", indices)
        if self.colors_rgb is not None:
            colors = readonly_array(
                self.colors_rgb,
                field_name="colors_rgb",
                dtype=np.uint8,
                shape=(count, 3),
            )
            object.__setattr__(self, "colors_rgb", colors)
        if self.point_weights is not None:
            weights = readonly_array(
                self.point_weights,
                field_name="point_weights",
                dtype=np.float64,
                shape=(count,),
                finite=True,
            )
            if np.any(weights < 0.0):
                raise ContractValidationError("point_weights must not be negative.")
            object.__setattr__(self, "point_weights", weights)

        raw = int(self.raw_point_count or count)
        retained = int(self.retained_point_count or count)
        if raw < retained or retained != count:
            raise ContractValidationError(
                "raw_point_count must be >= retained_point_count, and retained count must match N."
            )
        object.__setattr__(self, "raw_point_count", raw)
        object.__setattr__(self, "retained_point_count", retained)
        object.__setattr__(self, "mask_id", str(self.mask_id).strip())
        if self.depth_scale is DepthScale.RELATIVE and self.scale_recovery_status is not ScaleRecoveryStatus.SUCCEEDED:
            # Relative points are allowed, but the contract makes non-metric readiness explicit.
            pass

    @property
    def metric_ready(self) -> bool:
        return self.depth_scale is DepthScale.METRIC or self.scale_recovery_status is ScaleRecoveryStatus.SUCCEEDED


@dataclass(frozen=True, slots=True)
class BoundingBox3D:
    box_id: str
    observation_id: str
    coordinate_frame_id: str
    kind: BoundingBoxKind
    centre_xyz_m: np.ndarray
    extents_xyz_m: np.ndarray
    orientation_matrix: np.ndarray
    fit_method: str
    support_point_count: int
    fit_residual_m: float | None = None
    percentile_policy: str = ""
    processing_steps: tuple[ProcessingStep, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "box_id", require_nonempty(self.box_id, field_name="box_id"))
        object.__setattr__(
            self, "observation_id", require_nonempty(self.observation_id, field_name="observation_id")
        )
        object.__setattr__(
            self,
            "coordinate_frame_id",
            require_nonempty(self.coordinate_frame_id, field_name="coordinate_frame_id"),
        )
        centre = readonly_array(
            self.centre_xyz_m,
            field_name="centre_xyz_m",
            dtype=np.float64,
            shape=(3,),
            finite=True,
        )
        extents = readonly_array(
            self.extents_xyz_m,
            field_name="extents_xyz_m",
            dtype=np.float64,
            shape=(3,),
            finite=True,
        )
        if np.any(extents <= 0.0):
            raise ContractValidationError("3D box extents must be positive.")
        orientation = readonly_array(
            self.orientation_matrix,
            field_name="orientation_matrix",
            dtype=np.float64,
            shape=(3, 3),
            finite=True,
        )
        if not np.allclose(orientation.T @ orientation, np.eye(3), atol=1e-6):
            raise ContractValidationError("3D box orientation must be orthonormal.")
        if not np.isclose(np.linalg.det(orientation), 1.0, atol=1e-6):
            raise ContractValidationError("3D box orientation must have determinant +1.")
        if isinstance(self.support_point_count, bool) or int(self.support_point_count) <= 0:
            raise ContractValidationError("support_point_count must be positive.")
        if self.fit_residual_m is not None:
            residual = require_finite(self.fit_residual_m, field_name="fit_residual_m")
            if residual < 0.0:
                raise ContractValidationError("fit_residual_m must not be negative.")
            object.__setattr__(self, "fit_residual_m", residual)
        object.__setattr__(self, "centre_xyz_m", centre)
        object.__setattr__(self, "extents_xyz_m", extents)
        object.__setattr__(self, "orientation_matrix", orientation)
        object.__setattr__(self, "fit_method", require_nonempty(self.fit_method, field_name="fit_method"))
        object.__setattr__(self, "support_point_count", int(self.support_point_count))
        object.__setattr__(self, "percentile_policy", str(self.percentile_policy).strip())


@dataclass(frozen=True, slots=True)
class ShapeModel:
    shape_model_id: str
    observation_id: str
    coordinate_frame_id: str
    kind: ShapeKind
    fit_method: str
    fit_status: str
    support_point_count: int
    parameters: Mapping[str, JsonValue] | Sequence[tuple[str, JsonValue]] = field(
        default_factory=tuple
    )
    fit_residual_m: float | None = None
    processing_steps: tuple[ProcessingStep, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "shape_model_id",
            require_nonempty(self.shape_model_id, field_name="shape_model_id"),
        )
        object.__setattr__(
            self, "observation_id", require_nonempty(self.observation_id, field_name="observation_id")
        )
        object.__setattr__(
            self,
            "coordinate_frame_id",
            require_nonempty(self.coordinate_frame_id, field_name="coordinate_frame_id"),
        )
        object.__setattr__(self, "fit_method", require_nonempty(self.fit_method, field_name="fit_method"))
        object.__setattr__(self, "fit_status", require_nonempty(self.fit_status, field_name="fit_status"))
        if isinstance(self.support_point_count, bool) or int(self.support_point_count) < 0:
            raise ContractValidationError("support_point_count must be non-negative.")
        object.__setattr__(self, "support_point_count", int(self.support_point_count))
        object.__setattr__(self, "parameters", normalize_metadata(self.parameters))
        if self.fit_residual_m is not None:
            residual = require_finite(self.fit_residual_m, field_name="fit_residual_m")
            if residual < 0.0:
                raise ContractValidationError("fit_residual_m must not be negative.")
            object.__setattr__(self, "fit_residual_m", residual)


@dataclass(frozen=True, slots=True)
class ScaleRecovery:
    status: ScaleRecoveryStatus
    method: str = ""
    scale_factor: float | None = None
    supporting_observation_ids: tuple[str, ...] = ()
    reliability: ReliabilityLevel = ReliabilityLevel.UNAVAILABLE
    failure_reason: str = ""

    def __post_init__(self) -> None:
        method = str(self.method).strip()
        failure = str(self.failure_reason).strip()
        if self.status is ScaleRecoveryStatus.SUCCEEDED:
            if not method or self.scale_factor is None:
                raise ContractValidationError(
                    "Successful scale recovery requires method and scale_factor."
                )
            factor = require_finite(self.scale_factor, field_name="scale_factor")
            if factor <= 0.0:
                raise ContractValidationError("scale_factor must be positive.")
            object.__setattr__(self, "scale_factor", factor)
            if self.reliability in {ReliabilityLevel.UNAVAILABLE, ReliabilityLevel.REJECTED}:
                raise ContractValidationError(
                    "Successful scale recovery requires an available reliability level."
                )
        else:
            if self.scale_factor is not None:
                raise ContractValidationError(
                    "scale_factor is only valid when scale recovery succeeded."
                )
        if self.status is ScaleRecoveryStatus.FAILED and not failure:
            raise ContractValidationError("Failed scale recovery requires failure_reason.")
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "failure_reason", failure)
        object.__setattr__(
            self,
            "supporting_observation_ids",
            tuple(
                require_nonempty(value, field_name="supporting_observation_id")
                for value in self.supporting_observation_ids
            ),
        )


@dataclass(frozen=True, slots=True)
class UncertaintyComponent:
    kind: UncertaintyComponentKind
    contribution_si: float
    method: str
    sample_count: int = 0
    lower_contribution_si: float | None = None
    upper_contribution_si: float | None = None
    note: str = ""

    def __post_init__(self) -> None:
        contribution = require_finite(self.contribution_si, field_name="contribution_si")
        if contribution < 0.0:
            raise ContractValidationError("contribution_si must not be negative.")
        object.__setattr__(self, "contribution_si", contribution)
        object.__setattr__(self, "method", require_nonempty(self.method, field_name="method"))
        if isinstance(self.sample_count, bool) or int(self.sample_count) < 0:
            raise ContractValidationError("sample_count must be non-negative.")
        object.__setattr__(self, "sample_count", int(self.sample_count))
        for field_name in ("lower_contribution_si", "upper_contribution_si"):
            value = getattr(self, field_name)
            if value is not None:
                number = require_finite(value, field_name=field_name)
                if number < 0.0:
                    raise ContractValidationError(f"{field_name} must not be negative.")
                object.__setattr__(self, field_name, number)
        object.__setattr__(self, "note", str(self.note).strip())


@dataclass(frozen=True, slots=True)
class UncertaintyEstimate:
    basis: UncertaintyBasis
    expanded_uncertainty_si: float
    standard_uncertainty_si: float | None = None
    coverage_factor: float | None = None
    coverage_probability: float | None = None
    components: tuple[UncertaintyComponent, ...] = ()
    perturbation_sample_count: int = 0
    temporal_sample_count: int = 0
    note: str = ""

    def __post_init__(self) -> None:
        expanded = require_finite(
            self.expanded_uncertainty_si, field_name="expanded_uncertainty_si"
        )
        if expanded < 0.0:
            raise ContractValidationError("expanded_uncertainty_si must not be negative.")
        object.__setattr__(self, "expanded_uncertainty_si", expanded)
        if self.standard_uncertainty_si is not None:
            standard = require_finite(
                self.standard_uncertainty_si, field_name="standard_uncertainty_si"
            )
            if standard < 0.0:
                raise ContractValidationError("standard_uncertainty_si must not be negative.")
            object.__setattr__(self, "standard_uncertainty_si", standard)
        if self.coverage_factor is not None:
            factor = require_finite(self.coverage_factor, field_name="coverage_factor")
            if factor <= 0.0:
                raise ContractValidationError("coverage_factor must be positive.")
            if not self.basis.supports_coverage_probability:
                raise ContractValidationError(
                    "coverage_factor requires an empirically calibrated uncertainty basis."
                )
            object.__setattr__(self, "coverage_factor", factor)
        if self.coverage_probability is not None:
            if not self.basis.supports_coverage_probability:
                raise ContractValidationError(
                    "coverage_probability requires an empirically calibrated uncertainty basis."
                )
            object.__setattr__(
                self,
                "coverage_probability",
                require_probability(
                    self.coverage_probability, field_name="coverage_probability"
                ),
            )
        for field_name in ("perturbation_sample_count", "temporal_sample_count"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or int(value) < 0:
                raise ContractValidationError(f"{field_name} must be non-negative.")
            object.__setattr__(self, field_name, int(value))
        object.__setattr__(self, "note", str(self.note).strip())

    @property
    def display_qualification(self) -> str:
        if self.basis is UncertaintyBasis.EMPIRICALLY_CALIBRATED:
            return "Empirically Calibrated Uncertainty"
        return "Estimated Uncertainty — Not Calibrated"


@dataclass(frozen=True, slots=True)
class ReliabilityAssessment:
    level: ReliabilityLevel
    score: float | None = None
    passed_checks: tuple[str, ...] = ()
    failed_checks: tuple[str, ...] = ()
    explanations: tuple[str, ...] = ()
    policy_id: str = ""

    def __post_init__(self) -> None:
        if self.score is not None:
            object.__setattr__(self, "score", require_probability(self.score, field_name="score"))
        object.__setattr__(
            self, "passed_checks", tuple(str(value).strip() for value in self.passed_checks)
        )
        object.__setattr__(
            self, "failed_checks", tuple(str(value).strip() for value in self.failed_checks)
        )
        object.__setattr__(
            self, "explanations", tuple(str(value).strip() for value in self.explanations)
        )
        object.__setattr__(self, "policy_id", str(self.policy_id).strip())


@dataclass(frozen=True, slots=True)
class MeasurementEstimate:
    """Per-kind eligibility is authoritative; object warnings never reject all kinds."""

    measurement_id: str
    observation_id: str
    kind: MeasurementKind
    status: MeasurementStatus
    method: str
    unit_si: str
    reliability: ReliabilityAssessment
    depth_scale: DepthScale
    intrinsics_source: IntrinsicsSource
    calibration_status: CalibrationStatus
    scale_recovery: ScaleRecovery
    value_si: float | None = None
    uncertainty: UncertaintyEstimate | None = None
    lower_bound_si: float | None = None
    upper_bound_si: float | None = None
    lower_bound_policy: LowerBoundPolicy = LowerBoundPolicy.REJECT_NEGATIVE
    valid_observation_count: int = 0
    model_provenance: tuple[ModelProvenance, ...] = ()
    processing_steps: tuple[ProcessingStep, ...] = ()
    rejection_reasons: tuple[RejectionReason, ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "measurement_id",
            require_nonempty(self.measurement_id, field_name="measurement_id"),
        )
        object.__setattr__(
            self, "observation_id", require_nonempty(self.observation_id, field_name="observation_id")
        )
        object.__setattr__(self, "method", require_nonempty(self.method, field_name="method"))
        expected_unit = _EXPECTED_UNITS[self.kind]
        if self.unit_si != expected_unit:
            raise ContractValidationError(
                f"{self.kind.value} measurements must use SI unit {expected_unit!r}."
            )
        if isinstance(self.valid_observation_count, bool) or int(self.valid_observation_count) < 0:
            raise ContractValidationError("valid_observation_count must be non-negative.")
        object.__setattr__(self, "valid_observation_count", int(self.valid_observation_count))
        object.__setattr__(self, "warnings", tuple(str(value).strip() for value in self.warnings))
        if self.intrinsics_source in {
            IntrinsicsSource.MODEL_PREDICTED,
            IntrinsicsSource.ESTIMATED,
        } and self.calibration_status is not CalibrationStatus.NOT_CALIBRATED:
            raise ContractValidationError(
                "Model-predicted and estimated intrinsics must remain Not Calibrated."
            )
        if self.calibration_status is CalibrationStatus.CALIBRATED and self.intrinsics_source is not IntrinsicsSource.CALIBRATED:
            raise ContractValidationError(
                "Calibrated measurement status requires calibrated intrinsics provenance."
            )

        if self.status is MeasurementStatus.AVAILABLE:
            if self.value_si is None or self.uncertainty is None:
                raise ContractValidationError(
                    "Available measurements require value_si and uncertainty."
                )
            if not self.model_provenance:
                raise ContractValidationError(
                    "Available measurements require model provenance."
                )
            if self.rejection_reasons:
                raise ContractValidationError(
                    "Available measurements must not contain rejection reasons."
                )
            value = require_finite(self.value_si, field_name="value_si")
            if self.kind in _NONNEGATIVE_MEASUREMENTS and value < 0.0:
                raise ContractValidationError(
                    f"{self.kind.value} value_si must not be negative."
                )
            if self.depth_scale is DepthScale.RELATIVE and self.scale_recovery.status is not ScaleRecoveryStatus.SUCCEEDED:
                raise ContractValidationError(
                    "Relative-depth metric measurements require successful scale recovery."
                )
            if self.depth_scale in {DepthScale.SYNTHETIC, DepthScale.UNKNOWN}:
                raise ContractValidationError(
                    "Synthetic or unknown depth scale cannot produce available metric measurements."
                )
            raw_lower = value - self.uncertainty.expanded_uncertainty_si
            upper = value + self.uncertainty.expanded_uncertainty_si
            lower = raw_lower
            if self.kind in _NONNEGATIVE_MEASUREMENTS and raw_lower < 0.0:
                if self.lower_bound_policy is LowerBoundPolicy.REJECT_NEGATIVE:
                    raise ContractValidationError(
                        "Derived lower bound is negative for a physically bounded quantity."
                    )
                if self.lower_bound_policy is LowerBoundPolicy.CLIP_TO_ZERO:
                    lower = 0.0
                else:
                    raise ContractValidationError(
                        "ALLOW_NEGATIVE is invalid for physically bounded measurements."
                    )
            if self.lower_bound_si is not None and not np.isclose(
                float(self.lower_bound_si), lower, rtol=1e-9, atol=1e-12
            ):
                raise ContractValidationError(
                    "lower_bound_si is inconsistent with value_si and expanded_uncertainty_si."
                )
            if self.upper_bound_si is not None and not np.isclose(
                float(self.upper_bound_si), upper, rtol=1e-9, atol=1e-12
            ):
                raise ContractValidationError(
                    "upper_bound_si is inconsistent with value_si and expanded_uncertainty_si."
                )
            object.__setattr__(self, "value_si", value)
            object.__setattr__(self, "lower_bound_si", lower)
            object.__setattr__(self, "upper_bound_si", upper)
        else:
            if self.value_si is not None or self.uncertainty is not None:
                raise ContractValidationError(
                    "Unavailable or rejected measurements must not expose numeric values."
                )
            if self.lower_bound_si is not None or self.upper_bound_si is not None:
                raise ContractValidationError(
                    "Unavailable or rejected measurements must not expose bounds."
                )
            if not self.rejection_reasons:
                raise ContractValidationError(
                    "Unavailable or rejected measurements require per-kind rejection reasons."
                )
            if self.status is MeasurementStatus.REJECTED and self.reliability.level is not ReliabilityLevel.REJECTED:
                raise ContractValidationError(
                    "Rejected measurements require ReliabilityLevel.REJECTED."
                )


@dataclass(frozen=True, slots=True)
class ObjectMeasurementResult:
    observation_id: str
    frame_reference: FrameReference
    estimates: tuple[MeasurementEstimate, ...]
    boxes: tuple[BoundingBox3D, ...] = ()
    shape_models: tuple[ShapeModel, ...] = ()
    tracking_state: TrackingState = TrackingState.UNTRACKED
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "observation_id", require_nonempty(self.observation_id, field_name="observation_id")
        )
        ids = [estimate.measurement_id for estimate in self.estimates]
        if len(set(ids)) != len(ids):
            raise ContractValidationError("Measurement IDs must be unique per object result.")
        kinds = [estimate.kind for estimate in self.estimates]
        if len(set(kinds)) != len(kinds):
            raise ContractValidationError(
                "Only one authoritative estimate per measurement kind is allowed."
            )
        for estimate in self.estimates:
            if estimate.observation_id != self.observation_id:
                raise ContractValidationError(
                    "Measurement estimate belongs to a different observation."
                )
        object.__setattr__(self, "warnings", tuple(str(value).strip() for value in self.warnings))


def shape_parameters_as_dict(shape: ShapeModel) -> dict[str, Any]:
    return metadata_as_dict(shape.parameters)  # type: ignore[arg-type]

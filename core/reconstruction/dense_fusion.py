"""Dependency-light dense multi-view fusion in reconstruction coordinates.

Phase 8.6C turns aligned depth keyframes plus camera-to-reconstruction poses
into a bounded, deterministic voxel-fused point cloud.  It deliberately keeps
three concerns separate:

* camera geometry: rays / intrinsics (Phase 8.6A),
* reconstruction gauge: arbitrary or metric (Phase 8.6B),
* depth-to-pose-gauge alignment: this module.

The fusion kernel never infers metric scale implicitly.  Depth values must
already be expressed in reconstruction units, be converted from metres by an
explicit metric scale profile, or carry an explicit/sparse-correspondence scale
factor into reconstruction units.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Any, Iterable

import cv2
import numpy as np

from core.contracts.validation import require_nonempty
from core.reconstruction.camera import CameraDistortionModel, ReconstructionCamera
from core.reconstruction.camera_geometry import CameraGeometryProfile, CameraGeometryPurpose
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.live_trajectory import LiveTrajectoryPose, LiveTrajectoryPoseLayer
from core.reconstruction.scale_reference import (
    MetricGeometryStatus,
    ScaleReferenceProfile,
    assess_reconstruction_geometry,
    reconstruction_points_to_metres,
)
from core.reconstruction.sfm_contracts import SfmScaleStatus, SfmWorldAlignmentStatus
from core.reconstruction.validation import require_nonnegative_int

DENSE_FUSION_SCHEMA_VERSION = 1
DENSE_FUSION_POSE_CONVENTION = "T_reconstruction_from_camera"


class DenseSceneProfile(str, Enum):
    """Product intent, separate from hardware quality/performance profile."""

    OBJECT_INSPECTION = "object_inspection"
    SPATIAL_DIGITAL_TWIN = "spatial_digital_twin"


class DenseDepthUnit(str, Enum):
    """Units carried by one input depth map before alignment."""

    RECONSTRUCTION_UNITS = "reconstruction_units"
    METRES = "metres"
    SOURCE_MODEL_UNITS = "source_model_units"


class DenseDepthAlignmentMode(str, Enum):
    RECONSTRUCTION_UNITS = "reconstruction_units"
    METRIC_SCALE_REFERENCE = "metric_scale_reference"
    EXPLICIT_SCALE_FACTOR = "explicit_scale_factor"
    SPARSE_CORRESPONDENCE_SCALE = "sparse_correspondence_scale"


@dataclass(frozen=True, slots=True)
class DenseDepthAlignment:
    mode: DenseDepthAlignmentMode
    scale_to_reconstruction_units: float
    provenance: str
    relative_scale_uncertainty: float | None = None
    scale_reference_profile_id: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.mode, DenseDepthAlignmentMode):
            raise TypeError("mode must be DenseDepthAlignmentMode.")
        factor = float(self.scale_to_reconstruction_units)
        if not isfinite(factor) or factor <= 0.0:
            raise ReconstructionValidationError(
                "scale_to_reconstruction_units must be finite and positive."
            )
        provenance = require_nonempty(self.provenance, field_name="provenance")
        uncertainty = self.relative_scale_uncertainty
        if uncertainty is not None:
            uncertainty = float(uncertainty)
            if not isfinite(uncertainty) or uncertainty < 0.0 or uncertainty >= 1.0:
                raise ReconstructionValidationError(
                    "relative_scale_uncertainty must be finite and in [0, 1)."
                )
        scale_profile_id = str(self.scale_reference_profile_id).strip()
        if self.mode is DenseDepthAlignmentMode.RECONSTRUCTION_UNITS:
            if not np.isclose(factor, 1.0, rtol=0.0, atol=1e-12):
                raise ReconstructionValidationError(
                    "RECONSTRUCTION_UNITS alignment requires scale factor 1."
                )
            if scale_profile_id:
                raise ReconstructionValidationError(
                    "RECONSTRUCTION_UNITS alignment must not bind a metric scale profile."
                )
        elif self.mode is DenseDepthAlignmentMode.METRIC_SCALE_REFERENCE:
            if not scale_profile_id:
                raise ReconstructionValidationError(
                    "Metric depth alignment requires scale_reference_profile_id."
                )
        else:
            if scale_profile_id:
                raise ReconstructionValidationError(
                    "Explicit/sparse alignment must not impersonate a scale-reference profile."
                )
            if uncertainty is None:
                raise ReconstructionValidationError(
                    "Explicit/sparse depth alignment requires relative_scale_uncertainty."
                )
        object.__setattr__(self, "scale_to_reconstruction_units", factor)
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "relative_scale_uncertainty", uncertainty)
        object.__setattr__(self, "scale_reference_profile_id", scale_profile_id)

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "scale_to_reconstruction_units": self.scale_to_reconstruction_units,
            "relative_scale_uncertainty": self.relative_scale_uncertainty,
            "scale_reference_profile_id": self.scale_reference_profile_id,
            "provenance": self.provenance,
        }


def reconstruction_unit_depth_alignment(*, provenance: str) -> DenseDepthAlignment:
    return DenseDepthAlignment(
        mode=DenseDepthAlignmentMode.RECONSTRUCTION_UNITS,
        scale_to_reconstruction_units=1.0,
        provenance=provenance,
    )


def metric_depth_alignment(
    scale_profile: ScaleReferenceProfile,
    *,
    provenance: str,
) -> DenseDepthAlignment:
    if not isinstance(scale_profile, ScaleReferenceProfile):
        raise TypeError("scale_profile must be ScaleReferenceProfile.")
    if not scale_profile.is_metric or scale_profile.gauge.unit_to_metre is None:
        raise ReconstructionValidationError(
            "Metric depth cannot be aligned to an arbitrary reconstruction gauge without "
            "an explicit/sparse depth-to-reconstruction scale."
        )
    return DenseDepthAlignment(
        mode=DenseDepthAlignmentMode.METRIC_SCALE_REFERENCE,
        scale_to_reconstruction_units=1.0 / float(scale_profile.gauge.unit_to_metre),
        relative_scale_uncertainty=scale_profile.gauge.relative_scale_uncertainty,
        scale_reference_profile_id=scale_profile.profile_id,
        provenance=provenance,
    )


def explicit_depth_scale_alignment(
    *,
    scale_to_reconstruction_units: float,
    relative_scale_uncertainty: float,
    provenance: str,
) -> DenseDepthAlignment:
    return DenseDepthAlignment(
        mode=DenseDepthAlignmentMode.EXPLICIT_SCALE_FACTOR,
        scale_to_reconstruction_units=scale_to_reconstruction_units,
        relative_scale_uncertainty=relative_scale_uncertainty,
        provenance=provenance,
    )


def estimate_depth_alignment_from_correspondences(
    depth_values_source_units: object,
    camera_z_reconstruction_units: object,
    *,
    provenance: str,
    minimum_correspondences: int = 8,
    maximum_relative_residual: float = 0.35,
) -> DenseDepthAlignment:
    """Robustly estimate one multiplicative depth-to-reconstruction scale.

    Correspondences normally come from sparse SfM/SLAM landmarks projected into
    a keyframe: ``depth_source * scale ~= landmark_z_in_camera``.  The estimator
    is deliberately scale-only; no offset is allowed for projective depth.
    """

    if isinstance(minimum_correspondences, bool) or int(minimum_correspondences) < 3:
        raise ReconstructionValidationError("minimum_correspondences must be at least 3.")
    maximum_relative_residual = float(maximum_relative_residual)
    if not isfinite(maximum_relative_residual) or not 0.0 < maximum_relative_residual < 1.0:
        raise ReconstructionValidationError(
            "maximum_relative_residual must be finite and in (0, 1)."
        )
    source = np.asarray(depth_values_source_units, dtype=np.float64).reshape(-1)
    target = np.asarray(camera_z_reconstruction_units, dtype=np.float64).reshape(-1)
    if source.shape != target.shape:
        raise ReconstructionValidationError("Depth/reference correspondence arrays must match.")
    valid = np.isfinite(source) & np.isfinite(target) & (source > 0.0) & (target > 0.0)
    source = source[valid]
    target = target[valid]
    if source.size < int(minimum_correspondences):
        raise ReconstructionValidationError(
            "Insufficient valid sparse/depth correspondences for scale alignment."
        )
    ratios = target / source
    initial = float(np.median(ratios))
    residual = np.abs(target - source * initial) / np.maximum(target, 1e-12)
    median_residual = float(np.median(residual))
    mad_residual = float(np.median(np.abs(residual - median_residual)))
    robust_limit = min(
        maximum_relative_residual,
        max(0.03, median_residual + 3.5 * 1.4826 * mad_residual),
    )
    inliers = residual <= robust_limit
    if int(np.count_nonzero(inliers)) < int(minimum_correspondences):
        raise ReconstructionValidationError(
            "Sparse/depth scale alignment rejected too many correspondences."
        )
    factor = float(np.median(target[inliers] / source[inliers]))
    relative_residual = np.abs(target[inliers] - source[inliers] * factor) / np.maximum(
        target[inliers], 1e-12
    )
    uncertainty = float(np.quantile(relative_residual, 0.90))
    if uncertainty > maximum_relative_residual:
        raise ReconstructionValidationError(
            "Sparse/depth scale alignment residual exceeds the accepted bound."
        )
    return DenseDepthAlignment(
        mode=DenseDepthAlignmentMode.SPARSE_CORRESPONDENCE_SCALE,
        scale_to_reconstruction_units=factor,
        relative_scale_uncertainty=min(uncertainty, 0.999999),
        provenance=(
            f"{require_nonempty(provenance, field_name='provenance')};"
            f"inliers={int(np.count_nonzero(inliers))}/{source.size}"
        ),
    )


@dataclass(frozen=True, slots=True)
class DenseFusionPolicy:
    scene_profile: DenseSceneProfile
    voxel_size_reconstruction_units: float
    pixel_stride: int = 2
    minimum_view_observations: int = 1
    minimum_confidence: float = 0.0
    minimum_depth_reconstruction_units: float | None = None
    maximum_depth_reconstruction_units: float | None = None
    max_voxels: int = 500_000

    def __post_init__(self) -> None:
        if not isinstance(self.scene_profile, DenseSceneProfile):
            raise TypeError("scene_profile must be DenseSceneProfile.")
        voxel = float(self.voxel_size_reconstruction_units)
        if not isfinite(voxel) or voxel <= 0.0:
            raise ReconstructionValidationError(
                "voxel_size_reconstruction_units must be finite and positive."
            )
        if isinstance(self.pixel_stride, bool) or int(self.pixel_stride) <= 0:
            raise ReconstructionValidationError("pixel_stride must be a positive integer.")
        if (
            isinstance(self.minimum_view_observations, bool)
            or int(self.minimum_view_observations) <= 0
        ):
            raise ReconstructionValidationError(
                "minimum_view_observations must be a positive integer."
            )
        confidence = float(self.minimum_confidence)
        if not isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ReconstructionValidationError("minimum_confidence must be in [0, 1].")
        minimum_depth = self.minimum_depth_reconstruction_units
        maximum_depth = self.maximum_depth_reconstruction_units
        if minimum_depth is not None:
            minimum_depth = float(minimum_depth)
            if not isfinite(minimum_depth) or minimum_depth <= 0.0:
                raise ReconstructionValidationError(
                    "minimum_depth_reconstruction_units must be positive."
                )
        if maximum_depth is not None:
            maximum_depth = float(maximum_depth)
            if not isfinite(maximum_depth) or maximum_depth <= 0.0:
                raise ReconstructionValidationError(
                    "maximum_depth_reconstruction_units must be positive."
                )
        if minimum_depth is not None and maximum_depth is not None and maximum_depth <= minimum_depth:
            raise ReconstructionValidationError(
                "maximum depth must be greater than minimum depth."
            )
        if isinstance(self.max_voxels, bool) or int(self.max_voxels) <= 0:
            raise ReconstructionValidationError("max_voxels must be a positive integer.")
        object.__setattr__(self, "voxel_size_reconstruction_units", voxel)
        object.__setattr__(self, "pixel_stride", int(self.pixel_stride))
        object.__setattr__(self, "minimum_view_observations", int(self.minimum_view_observations))
        object.__setattr__(self, "minimum_confidence", confidence)
        object.__setattr__(self, "minimum_depth_reconstruction_units", minimum_depth)
        object.__setattr__(self, "maximum_depth_reconstruction_units", maximum_depth)
        object.__setattr__(self, "max_voxels", int(self.max_voxels))

    def as_dict(self) -> dict[str, Any]:
        return {
            "scene_profile": self.scene_profile.value,
            "voxel_size_reconstruction_units": self.voxel_size_reconstruction_units,
            "pixel_stride": self.pixel_stride,
            "minimum_view_observations": self.minimum_view_observations,
            "minimum_confidence": self.minimum_confidence,
            "minimum_depth_reconstruction_units": self.minimum_depth_reconstruction_units,
            "maximum_depth_reconstruction_units": self.maximum_depth_reconstruction_units,
            "max_voxels": self.max_voxels,
        }


@dataclass(frozen=True, slots=True)
class DenseFusionFrame:
    frame_id: str
    timestamp_ns: int
    rgb: np.ndarray
    depth: np.ndarray
    T_reconstruction_from_camera: np.ndarray
    depth_unit: DenseDepthUnit
    depth_alignment: DenseDepthAlignment
    confidence: np.ndarray | None = None
    valid_mask: np.ndarray | None = None
    pose_layer: str = "tracking"

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_id", require_nonempty(self.frame_id, field_name="frame_id"))
        object.__setattr__(
            self,
            "timestamp_ns",
            require_nonnegative_int(self.timestamp_ns, field_name="timestamp_ns"),
        )
        rgb = np.asarray(self.rgb)
        depth = np.asarray(self.depth, dtype=np.float32)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ReconstructionValidationError("rgb must be HxWx3 uint8.")
        if depth.ndim != 2 or depth.shape != rgb.shape[:2]:
            raise ReconstructionValidationError("depth must be HxW and match rgb dimensions.")
        if not isinstance(self.depth_unit, DenseDepthUnit):
            raise TypeError("depth_unit must be DenseDepthUnit.")
        if not isinstance(self.depth_alignment, DenseDepthAlignment):
            raise TypeError("depth_alignment must be DenseDepthAlignment.")
        if self.depth_unit is DenseDepthUnit.RECONSTRUCTION_UNITS:
            if self.depth_alignment.mode is not DenseDepthAlignmentMode.RECONSTRUCTION_UNITS:
                raise ReconstructionValidationError(
                    "Reconstruction-unit depth requires RECONSTRUCTION_UNITS alignment."
                )
        elif self.depth_unit is DenseDepthUnit.METRES:
            if self.depth_alignment.mode not in {
                DenseDepthAlignmentMode.METRIC_SCALE_REFERENCE,
                DenseDepthAlignmentMode.EXPLICIT_SCALE_FACTOR,
                DenseDepthAlignmentMode.SPARSE_CORRESPONDENCE_SCALE,
            }:
                raise ReconstructionValidationError(
                    "Metric depth must be explicitly aligned into reconstruction units."
                )
        else:
            if self.depth_alignment.mode not in {
                DenseDepthAlignmentMode.EXPLICIT_SCALE_FACTOR,
                DenseDepthAlignmentMode.SPARSE_CORRESPONDENCE_SCALE,
            }:
                raise ReconstructionValidationError(
                    "Source-model depth requires an explicit or sparse-correspondence scale."
                )

        matrix = _validate_rigid_matrix(
            self.T_reconstruction_from_camera,
            field_name="T_reconstruction_from_camera",
        )
        confidence = self.confidence
        if confidence is not None:
            confidence = np.asarray(confidence, dtype=np.float32)
            if confidence.shape != depth.shape:
                raise ReconstructionValidationError("confidence must match depth shape.")
            finite_confidence = np.isfinite(confidence)
            if finite_confidence.any() and (
                float(np.min(confidence[finite_confidence])) < 0.0
                or float(np.max(confidence[finite_confidence])) > 1.0
            ):
                raise ReconstructionValidationError("confidence values must lie in [0, 1].")
            confidence = _readonly_array(confidence, dtype=np.float32)
        valid_mask = self.valid_mask
        if valid_mask is not None:
            valid_mask = np.asarray(valid_mask, dtype=bool)
            if valid_mask.shape != depth.shape:
                raise ReconstructionValidationError("valid_mask must match depth shape.")
            valid_mask = _readonly_array(valid_mask, dtype=bool)
        pose_layer = require_nonempty(self.pose_layer, field_name="pose_layer")
        object.__setattr__(self, "rgb", _readonly_array(rgb, dtype=np.uint8))
        object.__setattr__(self, "depth", _readonly_array(depth, dtype=np.float32))
        object.__setattr__(self, "T_reconstruction_from_camera", matrix)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "valid_mask", valid_mask)
        object.__setattr__(self, "pose_layer", pose_layer)

    @classmethod
    def from_live_trajectory_pose(
        cls,
        *,
        pose: LiveTrajectoryPose,
        rgb: np.ndarray,
        depth: np.ndarray,
        depth_unit: DenseDepthUnit,
        depth_alignment: DenseDepthAlignment,
        confidence: np.ndarray | None = None,
        valid_mask: np.ndarray | None = None,
    ) -> "DenseFusionFrame":
        if not isinstance(pose, LiveTrajectoryPose):
            raise TypeError("pose must be LiveTrajectoryPose.")
        return cls(
            frame_id=pose.frame_id,
            timestamp_ns=pose.timestamp_ns,
            rgb=rgb,
            depth=depth,
            T_reconstruction_from_camera=pose.T_reconstruction_from_camera,
            depth_unit=depth_unit,
            depth_alignment=depth_alignment,
            confidence=confidence,
            valid_mask=valid_mask,
            pose_layer=pose.layer.value,
        )


@dataclass(frozen=True, slots=True)
class DenseFusionStatistics:
    frames_integrated: int
    sampled_pixels: int
    valid_depth_samples: int
    frame_voxel_observations: int
    retained_voxels: int
    multi_view_voxels: int
    dropped_new_voxels_capacity: int

    def __post_init__(self) -> None:
        for name in (
            "frames_integrated",
            "sampled_pixels",
            "valid_depth_samples",
            "frame_voxel_observations",
            "retained_voxels",
            "multi_view_voxels",
            "dropped_new_voxels_capacity",
        ):
            object.__setattr__(
                self,
                name,
                require_nonnegative_int(getattr(self, name), field_name=name),
            )

    def as_dict(self) -> dict[str, int]:
        return {
            "frames_integrated": self.frames_integrated,
            "sampled_pixels": self.sampled_pixels,
            "valid_depth_samples": self.valid_depth_samples,
            "frame_voxel_observations": self.frame_voxel_observations,
            "retained_voxels": self.retained_voxels,
            "multi_view_voxels": self.multi_view_voxels,
            "dropped_new_voxels_capacity": self.dropped_new_voxels_capacity,
        }


@dataclass(frozen=True, slots=True)
class DensePointCloud:
    points_reconstruction: np.ndarray
    colors_rgb: np.ndarray
    view_observation_counts: np.ndarray
    mean_confidence: np.ndarray
    voxel_size_reconstruction_units: float

    def __post_init__(self) -> None:
        points = np.asarray(self.points_reconstruction, dtype=np.float64)
        colors = np.asarray(self.colors_rgb, dtype=np.uint8)
        counts = np.asarray(self.view_observation_counts, dtype=np.uint32)
        confidence = np.asarray(self.mean_confidence, dtype=np.float32)
        if points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all():
            raise ReconstructionValidationError("points_reconstruction must be finite Nx3.")
        count = len(points)
        if colors.shape != (count, 3):
            raise ReconstructionValidationError("colors_rgb must have shape Nx3.")
        if counts.shape != (count,) or np.any(counts == 0):
            raise ReconstructionValidationError(
                "view_observation_counts must contain one positive value per point."
            )
        if confidence.shape != (count,) or (
            count and (not np.isfinite(confidence).all() or np.any(confidence < 0.0) or np.any(confidence > 1.0))
        ):
            raise ReconstructionValidationError("mean_confidence must be finite within [0, 1].")
        voxel = float(self.voxel_size_reconstruction_units)
        if not isfinite(voxel) or voxel <= 0.0:
            raise ReconstructionValidationError("voxel size must be finite and positive.")
        object.__setattr__(self, "points_reconstruction", _readonly_array(points, dtype=np.float64))
        object.__setattr__(self, "colors_rgb", _readonly_array(colors, dtype=np.uint8))
        object.__setattr__(self, "view_observation_counts", _readonly_array(counts, dtype=np.uint32))
        object.__setattr__(self, "mean_confidence", _readonly_array(confidence, dtype=np.float32))
        object.__setattr__(self, "voxel_size_reconstruction_units", voxel)

    def points_metres(self, scale_profile: ScaleReferenceProfile) -> np.ndarray:
        return reconstruction_points_to_metres(self.points_reconstruction, scale_profile)

    def as_dict(self) -> dict[str, Any]:
        return {
            "point_count": int(len(self.points_reconstruction)),
            "voxel_size_reconstruction_units": self.voxel_size_reconstruction_units,
            "minimum_view_observations": (
                int(np.min(self.view_observation_counts)) if len(self.view_observation_counts) else None
            ),
            "maximum_view_observations": (
                int(np.max(self.view_observation_counts)) if len(self.view_observation_counts) else None
            ),
        }


@dataclass(frozen=True, slots=True)
class DenseFusionResult:
    session_id: str
    scene_profile: DenseSceneProfile
    camera_profile_id: str
    scale_profile_id: str
    metric_geometry_status: MetricGeometryStatus
    pose_convention: str
    reconstruction_frame_id: str
    scale_status: SfmScaleStatus
    world_alignment_status: SfmWorldAlignmentStatus
    unit_to_metre: float | None
    certified_metrology: bool
    source_frame_ids: tuple[str, ...]
    source_pose_layers: tuple[str, ...]
    point_cloud: DensePointCloud
    statistics: DenseFusionStatistics
    schema_version: int = DENSE_FUSION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_id", require_nonempty(self.session_id, field_name="session_id"))
        if not isinstance(self.scene_profile, DenseSceneProfile):
            raise TypeError("scene_profile must be DenseSceneProfile.")
        object.__setattr__(
            self, "camera_profile_id", require_nonempty(self.camera_profile_id, field_name="camera_profile_id")
        )
        object.__setattr__(
            self, "scale_profile_id", require_nonempty(self.scale_profile_id, field_name="scale_profile_id")
        )
        if not isinstance(self.metric_geometry_status, MetricGeometryStatus):
            raise TypeError("metric_geometry_status must be MetricGeometryStatus.")
        if self.pose_convention != DENSE_FUSION_POSE_CONVENTION:
            raise ReconstructionValidationError(
                f"pose_convention must be {DENSE_FUSION_POSE_CONVENTION}."
            )
        object.__setattr__(
            self,
            "reconstruction_frame_id",
            require_nonempty(self.reconstruction_frame_id, field_name="reconstruction_frame_id"),
        )
        if not isinstance(self.scale_status, SfmScaleStatus):
            raise TypeError("scale_status must be SfmScaleStatus.")
        if not isinstance(self.world_alignment_status, SfmWorldAlignmentStatus):
            raise TypeError("world_alignment_status must be SfmWorldAlignmentStatus.")
        if self.certified_metrology:
            raise ReconstructionValidationError(
                "Phase 8.6C dense fusion must not claim certified metrology."
            )
        frame_ids = tuple(str(value).strip() for value in self.source_frame_ids)
        pose_layers = tuple(str(value).strip() for value in self.source_pose_layers)
        if not frame_ids or any(not value for value in frame_ids):
            raise ReconstructionValidationError("Dense fusion requires source frame IDs.")
        if len(set(frame_ids)) != len(frame_ids):
            raise ReconstructionValidationError("Dense fusion source frame IDs must be unique.")
        if len(pose_layers) != len(frame_ids) or any(not value for value in pose_layers):
            raise ReconstructionValidationError("source_pose_layers must align with source_frame_ids.")
        if not isinstance(self.point_cloud, DensePointCloud):
            raise TypeError("point_cloud must be DensePointCloud.")
        if not isinstance(self.statistics, DenseFusionStatistics):
            raise TypeError("statistics must be DenseFusionStatistics.")
        if int(self.schema_version) != DENSE_FUSION_SCHEMA_VERSION:
            raise ReconstructionValidationError("Unsupported dense-fusion schema version.")
        object.__setattr__(self, "source_frame_ids", frame_ids)
        object.__setattr__(self, "source_pose_layers", pose_layers)
        object.__setattr__(self, "certified_metrology", False)
        object.__setattr__(self, "schema_version", DENSE_FUSION_SCHEMA_VERSION)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "scene_profile": self.scene_profile.value,
            "camera_profile_id": self.camera_profile_id,
            "scale_profile_id": self.scale_profile_id,
            "metric_geometry_status": self.metric_geometry_status.value,
            "pose_convention": self.pose_convention,
            "reconstruction_frame_id": self.reconstruction_frame_id,
            "scale_status": self.scale_status.value,
            "world_alignment_status": self.world_alignment_status.value,
            "unit_to_metre": self.unit_to_metre,
            "certified_metrology": self.certified_metrology,
            "source_frame_ids": list(self.source_frame_ids),
            "source_pose_layers": list(self.source_pose_layers),
            "point_cloud": self.point_cloud.as_dict(),
            "statistics": self.statistics.as_dict(),
        }


@dataclass(slots=True)
class _VoxelAccumulator:
    weight_sum: float = 0.0
    position_sum: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    color_sum: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    confidence_sum: float = 0.0
    view_count: int = 0


class DenseVoxelFusion:
    """Bounded deterministic weighted voxel fusion accumulator."""

    def __init__(
        self,
        *,
        session_id: str,
        camera_profile: CameraGeometryProfile,
        scale_profile: ScaleReferenceProfile,
        policy: DenseFusionPolicy,
        reconstruction_frame_id: str = "reconstruction",
    ) -> None:
        self.session_id = require_nonempty(session_id, field_name="session_id")
        if not isinstance(camera_profile, CameraGeometryProfile):
            raise TypeError("camera_profile must be CameraGeometryProfile.")
        if not isinstance(scale_profile, ScaleReferenceProfile):
            raise TypeError("scale_profile must be ScaleReferenceProfile.")
        if not isinstance(policy, DenseFusionPolicy):
            raise TypeError("policy must be DenseFusionPolicy.")
        assessment = camera_profile.assess(CameraGeometryPurpose.RELATIVE_RECONSTRUCTION)
        if not assessment.allowed:
            raise ReconstructionValidationError(
                "Camera geometry is not allowed for relative dense reconstruction."
            )
        self.camera_profile = camera_profile
        self.scale_profile = scale_profile
        self.policy = policy
        self.reconstruction_frame_id = require_nonempty(
            reconstruction_frame_id, field_name="reconstruction_frame_id"
        )
        self._voxels: dict[tuple[int, int, int], _VoxelAccumulator] = {}
        self._frames: list[str] = []
        self._pose_layers: list[str] = []
        self._timestamps: list[int] = []
        self._sampled_pixels = 0
        self._valid_depth_samples = 0
        self._frame_voxel_observations = 0
        self._dropped_new_voxels_capacity = 0

    def integrate(self, frame: DenseFusionFrame) -> int:
        if not isinstance(frame, DenseFusionFrame):
            raise TypeError("frame must be DenseFusionFrame.")
        if frame.frame_id in self._frames:
            raise ReconstructionValidationError("Dense fusion frame IDs must be unique.")
        if self._timestamps and frame.timestamp_ns <= self._timestamps[-1]:
            raise ReconstructionValidationError(
                "Dense fusion frames must be integrated in strictly increasing timestamp order."
            )
        expected_shape = (
            self.camera_profile.camera.image_size.height,
            self.camera_profile.camera.image_size.width,
        )
        if frame.depth.shape != expected_shape:
            raise ReconstructionValidationError(
                "Dense fusion frame dimensions must match the camera geometry profile."
            )
        self._validate_alignment(frame)

        points, colors, confidences, sampled_count = _frame_points_in_reconstruction(
            frame,
            self.camera_profile.camera,
            self.policy,
        )
        self._sampled_pixels += sampled_count
        self._valid_depth_samples += len(points)
        if len(points):
            voxel_indices = np.floor(
                points / self.policy.voxel_size_reconstruction_units
            ).astype(np.int64)
            unique_voxels, inverse = np.unique(voxel_indices, axis=0, return_inverse=True)
            per_frame_count = len(unique_voxels)
            self._frame_voxel_observations += per_frame_count
            weights = np.clip(confidences.astype(np.float64), 1e-6, 1.0)
            weight_sums = np.bincount(inverse, weights=weights, minlength=per_frame_count)
            position_sums = np.column_stack(
                [
                    np.bincount(inverse, weights=points[:, axis] * weights, minlength=per_frame_count)
                    for axis in range(3)
                ]
            )
            color_sums = np.column_stack(
                [
                    np.bincount(inverse, weights=colors[:, axis] * weights, minlength=per_frame_count)
                    for axis in range(3)
                ]
            )
            confidence_sums = np.bincount(
                inverse, weights=confidences.astype(np.float64), minlength=per_frame_count
            )
            pixel_counts = np.bincount(inverse, minlength=per_frame_count)

            for index, voxel in enumerate(unique_voxels):
                key = (int(voxel[0]), int(voxel[1]), int(voxel[2]))
                accumulator = self._voxels.get(key)
                if accumulator is None:
                    if len(self._voxels) >= self.policy.max_voxels:
                        self._dropped_new_voxels_capacity += 1
                        continue
                    accumulator = _VoxelAccumulator()
                    self._voxels[key] = accumulator
                accumulator.weight_sum += float(weight_sums[index])
                accumulator.position_sum += position_sums[index]
                accumulator.color_sum += color_sums[index]
                accumulator.confidence_sum += float(confidence_sums[index] / max(pixel_counts[index], 1))
                accumulator.view_count += 1

        self._frames.append(frame.frame_id)
        self._pose_layers.append(frame.pose_layer)
        self._timestamps.append(frame.timestamp_ns)
        return len(points)

    def _validate_alignment(self, frame: DenseFusionFrame) -> None:
        alignment = frame.depth_alignment
        if alignment.mode is DenseDepthAlignmentMode.METRIC_SCALE_REFERENCE:
            if frame.depth_unit is not DenseDepthUnit.METRES:
                raise ReconstructionValidationError(
                    "METRIC_SCALE_REFERENCE alignment is valid only for metre-valued depth."
                )
            if alignment.scale_reference_profile_id != self.scale_profile.profile_id:
                raise ReconstructionValidationError(
                    "Metric depth alignment references a different scale profile."
                )
            if not self.scale_profile.is_metric or self.scale_profile.gauge.unit_to_metre is None:
                raise ReconstructionValidationError(
                    "Metric depth alignment requires a metric reconstruction gauge."
                )
            expected = 1.0 / float(self.scale_profile.gauge.unit_to_metre)
            if not np.isclose(
                alignment.scale_to_reconstruction_units, expected, rtol=1e-10, atol=1e-12
            ):
                raise ReconstructionValidationError(
                    "Metric depth alignment factor does not match the active scale profile."
                )

    def finalize(self) -> DenseFusionResult:
        if not self._frames:
            raise ReconstructionValidationError("Dense fusion requires at least one frame.")
        selected = [
            (key, value)
            for key, value in sorted(self._voxels.items())
            if value.view_count >= self.policy.minimum_view_observations
        ]
        if not selected:
            raise ReconstructionValidationError(
                "Dense fusion produced no voxels satisfying the minimum view-observation policy."
            )
        points = np.vstack([value.position_sum / value.weight_sum for _, value in selected])
        colors = np.vstack(
            [np.clip(np.rint(value.color_sum / value.weight_sum), 0, 255) for _, value in selected]
        ).astype(np.uint8)
        counts = np.asarray([value.view_count for _, value in selected], dtype=np.uint32)
        confidence = np.asarray(
            [value.confidence_sum / value.view_count for _, value in selected], dtype=np.float32
        )
        cloud = DensePointCloud(
            points_reconstruction=points,
            colors_rgb=colors,
            view_observation_counts=counts,
            mean_confidence=confidence,
            voxel_size_reconstruction_units=self.policy.voxel_size_reconstruction_units,
        )
        assessment = assess_reconstruction_geometry(self.camera_profile, self.scale_profile)
        statistics = DenseFusionStatistics(
            frames_integrated=len(self._frames),
            sampled_pixels=self._sampled_pixels,
            valid_depth_samples=self._valid_depth_samples,
            frame_voxel_observations=self._frame_voxel_observations,
            retained_voxels=len(selected),
            multi_view_voxels=int(np.count_nonzero(counts >= 2)),
            dropped_new_voxels_capacity=self._dropped_new_voxels_capacity,
        )
        return DenseFusionResult(
            session_id=self.session_id,
            scene_profile=self.policy.scene_profile,
            camera_profile_id=self.camera_profile.profile_id,
            scale_profile_id=self.scale_profile.profile_id,
            metric_geometry_status=assessment.metric_status,
            pose_convention=DENSE_FUSION_POSE_CONVENTION,
            reconstruction_frame_id=self.reconstruction_frame_id,
            scale_status=self.scale_profile.gauge.scale_status,
            world_alignment_status=self.scale_profile.gauge.alignment_status,
            unit_to_metre=self.scale_profile.gauge.unit_to_metre,
            certified_metrology=False,
            source_frame_ids=tuple(self._frames),
            source_pose_layers=tuple(self._pose_layers),
            point_cloud=cloud,
            statistics=statistics,
        )


def fuse_dense_frames(
    frames: Iterable[DenseFusionFrame],
    *,
    session_id: str,
    camera_profile: CameraGeometryProfile,
    scale_profile: ScaleReferenceProfile,
    policy: DenseFusionPolicy,
    reconstruction_frame_id: str = "reconstruction",
) -> DenseFusionResult:
    fusion = DenseVoxelFusion(
        session_id=session_id,
        camera_profile=camera_profile,
        scale_profile=scale_profile,
        policy=policy,
        reconstruction_frame_id=reconstruction_frame_id,
    )
    for frame in frames:
        fusion.integrate(frame)
    return fusion.finalize()


def _frame_points_in_reconstruction(
    frame: DenseFusionFrame,
    camera: ReconstructionCamera,
    policy: DenseFusionPolicy,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    stride = policy.pixel_stride
    depth = frame.depth[::stride, ::stride].astype(np.float64, copy=False)
    rgb = frame.rgb[::stride, ::stride]
    height, width = depth.shape
    original_u = np.arange(0, frame.depth.shape[1], stride, dtype=np.float64)[:width]
    original_v = np.arange(0, frame.depth.shape[0], stride, dtype=np.float64)[:height]
    grid_u, grid_v = np.meshgrid(original_u, original_v)
    sampled_count = int(depth.size)

    valid = np.isfinite(depth) & (depth > 0.0)
    if frame.valid_mask is not None:
        valid &= frame.valid_mask[::stride, ::stride]
    if frame.confidence is not None:
        confidence_grid = frame.confidence[::stride, ::stride].astype(np.float64, copy=False)
        valid &= np.isfinite(confidence_grid) & (confidence_grid >= policy.minimum_confidence)
    else:
        confidence_grid = np.ones_like(depth, dtype=np.float64)
        if policy.minimum_confidence > 0.0:
            # Unknown confidence is neutral rather than silently treated as a high-confidence claim.
            confidence_grid.fill(policy.minimum_confidence)

    aligned_depth = depth * frame.depth_alignment.scale_to_reconstruction_units
    if policy.minimum_depth_reconstruction_units is not None:
        valid &= aligned_depth >= policy.minimum_depth_reconstruction_units
    if policy.maximum_depth_reconstruction_units is not None:
        valid &= aligned_depth <= policy.maximum_depth_reconstruction_units
    if not np.any(valid):
        empty3 = np.empty((0, 3), dtype=np.float64)
        return empty3, np.empty((0, 3), dtype=np.uint8), np.empty((0,), dtype=np.float32), sampled_count

    u = grid_u[valid]
    v = grid_v[valid]
    z = aligned_depth[valid]
    normalized_xy = _undistorted_normalized_coordinates(camera, u, v)
    points_camera = np.column_stack(
        (normalized_xy[:, 0] * z, normalized_xy[:, 1] * z, z)
    )
    rotation = frame.T_reconstruction_from_camera[:3, :3]
    translation = frame.T_reconstruction_from_camera[:3, 3]
    points_reconstruction = points_camera @ rotation.T + translation
    colors = rgb[valid]
    confidences = np.clip(confidence_grid[valid], 0.0, 1.0).astype(np.float32)
    finite_points = np.isfinite(points_reconstruction).all(axis=1)
    return (
        np.ascontiguousarray(points_reconstruction[finite_points], dtype=np.float64),
        np.ascontiguousarray(colors[finite_points], dtype=np.uint8),
        np.ascontiguousarray(confidences[finite_points], dtype=np.float32),
        sampled_count,
    )


def _undistorted_normalized_coordinates(
    camera: ReconstructionCamera,
    u: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    matrix = np.asarray(camera.intrinsics.matrix, dtype=np.float64)
    if camera.distortion_model is CameraDistortionModel.NONE:
        return np.column_stack(
            ((u - matrix[0, 2]) / matrix[0, 0], (v - matrix[1, 2]) / matrix[1, 1])
        )
    pixels = np.column_stack((u, v)).reshape(-1, 1, 2).astype(np.float64)
    coefficients = np.asarray(camera.distortion_coefficients, dtype=np.float64)
    if camera.distortion_model is CameraDistortionModel.BROWN_CONRADY:
        normalized = cv2.undistortPoints(pixels, matrix, coefficients)
    elif camera.distortion_model is CameraDistortionModel.OPENCV_FISHEYE:
        normalized = cv2.fisheye.undistortPoints(pixels, matrix, coefficients)
    else:
        raise ReconstructionValidationError(
            "Backend-specific lens distortion is not supported by the Phase 8.6C NumPy fusion kernel."
        )
    return np.ascontiguousarray(normalized.reshape(-1, 2), dtype=np.float64)


def _validate_rigid_matrix(value: object, *, field_name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ReconstructionValidationError(f"{field_name} must be a finite homogeneous 4x4 matrix.")
    if not np.allclose(matrix[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-8):
        raise ReconstructionValidationError(f"{field_name} final row must be [0, 0, 0, 1].")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ReconstructionValidationError(f"{field_name} rotation must be orthonormal.")
    if not np.isclose(float(np.linalg.det(rotation)), 1.0, atol=1e-6):
        raise ReconstructionValidationError(f"{field_name} rotation determinant must be +1.")
    return _readonly_array(matrix, dtype=np.float64)


def _readonly_array(value: object, *, dtype: Any) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype).copy()
    array.setflags(write=False)
    return array

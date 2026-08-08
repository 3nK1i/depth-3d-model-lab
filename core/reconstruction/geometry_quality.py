"""Dense reconstruction quality and uncertainty assessment.

Phase 8.6D evaluates Phase 8.6C dense geometry without changing its coordinate
semantics.  The quality layer combines evidence already carried by the fused
point cloud (view support and confidence) with camera/pose/scale provenance and
optional multi-view consistency evidence.

Important safety properties:

* arbitrary reconstruction remains arbitrary and never gains metric units;
* provisional metric reconstruction remains provisional;
* local geometric uncertainty and metric scale uncertainty are reported
  separately instead of being collapsed into a false metrology guarantee;
* unavailable evidence is treated conservatively rather than assumed perfect;
* ``certified_metrology`` is always false in Phase 8.6D.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite, sqrt
from typing import Any

import numpy as np

from core.contracts.validation import require_nonempty
from core.reconstruction.camera_geometry import CameraGeometryMode, CameraGeometryProfile
from core.reconstruction.dense_fusion import DenseFusionResult, DenseSceneProfile
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.scale_reference import MetricGeometryStatus, ScaleReferenceProfile

DENSE_GEOMETRY_QUALITY_SCHEMA_VERSION = 1


class GeometryQualityLevel(str, Enum):
    """Human-readable geometry-quality band.

    ``UNOBSERVED`` is reserved for a future expected-domain/mesh visibility
    analysis.  A fused point represents an observed location and therefore
    receives HIGH/MEDIUM/LOW only in Phase 8.6D.
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNOBSERVED = "unobserved"


class GeometryQualityCode(int, Enum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2


@dataclass(frozen=True, slots=True)
class DenseGeometryQualityPolicy:
    scene_profile: DenseSceneProfile
    target_view_observations: int
    minimum_view_observations: int
    minimum_confidence: float
    desired_view_angle_span_degrees: float
    maximum_depth_relative_residual: float
    high_score_threshold: float
    medium_score_threshold: float
    minimum_input_depth_coverage: float

    def __post_init__(self) -> None:
        if not isinstance(self.scene_profile, DenseSceneProfile):
            raise TypeError("scene_profile must be DenseSceneProfile.")
        for name in ("target_view_observations", "minimum_view_observations"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) <= 0:
                raise ReconstructionValidationError(f"{name} must be a positive integer.")
            object.__setattr__(self, name, int(value))
        if self.minimum_view_observations > self.target_view_observations:
            raise ReconstructionValidationError(
                "minimum_view_observations must not exceed target_view_observations."
            )
        for name in (
            "minimum_confidence",
            "high_score_threshold",
            "medium_score_threshold",
            "minimum_input_depth_coverage",
        ):
            value = float(getattr(self, name))
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise ReconstructionValidationError(f"{name} must be finite in [0, 1].")
            object.__setattr__(self, name, value)
        if self.high_score_threshold <= self.medium_score_threshold:
            raise ReconstructionValidationError(
                "high_score_threshold must be greater than medium_score_threshold."
            )
        angle = float(self.desired_view_angle_span_degrees)
        if not isfinite(angle) or not 0.0 < angle <= 180.0:
            raise ReconstructionValidationError(
                "desired_view_angle_span_degrees must be finite in (0, 180]."
            )
        residual = float(self.maximum_depth_relative_residual)
        if not isfinite(residual) or not 0.0 < residual < 1.0:
            raise ReconstructionValidationError(
                "maximum_depth_relative_residual must be finite in (0, 1)."
            )
        object.__setattr__(self, "desired_view_angle_span_degrees", angle)
        object.__setattr__(self, "maximum_depth_relative_residual", residual)

    def as_dict(self) -> dict[str, Any]:
        return {
            "scene_profile": self.scene_profile.value,
            "target_view_observations": self.target_view_observations,
            "minimum_view_observations": self.minimum_view_observations,
            "minimum_confidence": self.minimum_confidence,
            "desired_view_angle_span_degrees": self.desired_view_angle_span_degrees,
            "maximum_depth_relative_residual": self.maximum_depth_relative_residual,
            "high_score_threshold": self.high_score_threshold,
            "medium_score_threshold": self.medium_score_threshold,
            "minimum_input_depth_coverage": self.minimum_input_depth_coverage,
        }


def default_dense_geometry_quality_policy(
    scene_profile: DenseSceneProfile,
) -> DenseGeometryQualityPolicy:
    if not isinstance(scene_profile, DenseSceneProfile):
        raise TypeError("scene_profile must be DenseSceneProfile.")
    if scene_profile is DenseSceneProfile.OBJECT_INSPECTION:
        return DenseGeometryQualityPolicy(
            scene_profile=scene_profile,
            target_view_observations=4,
            minimum_view_observations=2,
            minimum_confidence=0.65,
            desired_view_angle_span_degrees=20.0,
            maximum_depth_relative_residual=0.04,
            high_score_threshold=0.82,
            medium_score_threshold=0.64,
            minimum_input_depth_coverage=0.65,
        )
    return DenseGeometryQualityPolicy(
        scene_profile=scene_profile,
        target_view_observations=3,
        minimum_view_observations=2,
        minimum_confidence=0.45,
        desired_view_angle_span_degrees=12.0,
        maximum_depth_relative_residual=0.08,
        high_score_threshold=0.78,
        medium_score_threshold=0.60,
        minimum_input_depth_coverage=0.40,
    )


def _readonly_array(value: object, *, dtype: np.dtype[Any]) -> np.ndarray:
    array = np.asarray(value, dtype=dtype).copy()
    array.setflags(write=False)
    return array


def _optional_point_signal(
    value: object | None,
    *,
    point_count: int,
    field_name: str,
    minimum: float,
    maximum: float | None,
) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (point_count,):
        raise ReconstructionValidationError(
            f"{field_name} must contain one value per fused point."
        )
    if not np.isfinite(array).all() or np.any(array < minimum):
        raise ReconstructionValidationError(
            f"{field_name} must be finite and >= {minimum}."
        )
    if maximum is not None and np.any(array > maximum):
        raise ReconstructionValidationError(
            f"{field_name} must be <= {maximum}."
        )
    return _readonly_array(array, dtype=np.float32)


@dataclass(frozen=True, slots=True)
class DenseGeometryEvidence:
    """Optional per-point evidence unavailable from the Phase 8.6C cloud alone.

    ``depth_relative_residual`` is dimensionless (for example, normalized
    cross-view depth disagreement). ``view_angle_span_degrees`` records the
    angular diversity of supporting views. ``visibility_ratio`` is the fraction
    of expected contributing visibility checks that considered the point
    visible.  Missing signals remain missing; they are never imputed as perfect.
    """

    point_count: int
    depth_relative_residual: np.ndarray | None = None
    view_angle_span_degrees: np.ndarray | None = None
    visibility_ratio: np.ndarray | None = None
    provenance: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.point_count, bool) or int(self.point_count) <= 0:
            raise ReconstructionValidationError("point_count must be a positive integer.")
        point_count = int(self.point_count)
        depth = _optional_point_signal(
            self.depth_relative_residual,
            point_count=point_count,
            field_name="depth_relative_residual",
            minimum=0.0,
            maximum=None,
        )
        angle = _optional_point_signal(
            self.view_angle_span_degrees,
            point_count=point_count,
            field_name="view_angle_span_degrees",
            minimum=0.0,
            maximum=180.0,
        )
        visibility = _optional_point_signal(
            self.visibility_ratio,
            point_count=point_count,
            field_name="visibility_ratio",
            minimum=0.0,
            maximum=1.0,
        )
        object.__setattr__(self, "point_count", point_count)
        object.__setattr__(self, "depth_relative_residual", depth)
        object.__setattr__(self, "view_angle_span_degrees", angle)
        object.__setattr__(self, "visibility_ratio", visibility)
        object.__setattr__(self, "provenance", str(self.provenance).strip())

    @property
    def available_signal_count(self) -> int:
        return sum(
            signal is not None
            for signal in (
                self.depth_relative_residual,
                self.view_angle_span_degrees,
                self.visibility_ratio,
            )
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "point_count": self.point_count,
            "depth_consistency_available": self.depth_relative_residual is not None,
            "view_diversity_available": self.view_angle_span_degrees is not None,
            "visibility_available": self.visibility_ratio is not None,
            "available_signal_count": self.available_signal_count,
            "provenance": self.provenance,
        }


@dataclass(frozen=True, slots=True)
class DenseGeometryPointQuality:
    quality_scores: np.ndarray
    quality_codes: np.ndarray
    accepted_mask: np.ndarray
    local_uncertainty_reconstruction_units: np.ndarray

    def __post_init__(self) -> None:
        scores = np.asarray(self.quality_scores, dtype=np.float32)
        codes = np.asarray(self.quality_codes, dtype=np.uint8)
        accepted = np.asarray(self.accepted_mask, dtype=bool)
        uncertainty = np.asarray(self.local_uncertainty_reconstruction_units, dtype=np.float32)
        if scores.ndim != 1 or not len(scores):
            raise ReconstructionValidationError("Point quality requires at least one score.")
        shape = scores.shape
        if codes.shape != shape or accepted.shape != shape or uncertainty.shape != shape:
            raise ReconstructionValidationError("Point-quality arrays must have matching shapes.")
        if not np.isfinite(scores).all() or np.any(scores < 0.0) or np.any(scores > 1.0):
            raise ReconstructionValidationError("quality_scores must be finite in [0, 1].")
        if np.any(codes > int(GeometryQualityCode.HIGH)):
            raise ReconstructionValidationError("quality_codes contain an unsupported code.")
        if not np.isfinite(uncertainty).all() or np.any(uncertainty <= 0.0):
            raise ReconstructionValidationError(
                "local_uncertainty_reconstruction_units must be finite and positive."
            )
        object.__setattr__(self, "quality_scores", _readonly_array(scores, dtype=np.float32))
        object.__setattr__(self, "quality_codes", _readonly_array(codes, dtype=np.uint8))
        object.__setattr__(self, "accepted_mask", _readonly_array(accepted, dtype=bool))
        object.__setattr__(
            self,
            "local_uncertainty_reconstruction_units",
            _readonly_array(uncertainty, dtype=np.float32),
        )

    @property
    def point_count(self) -> int:
        return int(len(self.quality_scores))

    def local_uncertainty_metres(self, scale_profile: ScaleReferenceProfile) -> np.ndarray:
        if not isinstance(scale_profile, ScaleReferenceProfile):
            raise TypeError("scale_profile must be ScaleReferenceProfile.")
        if not scale_profile.is_metric or scale_profile.gauge.unit_to_metre is None:
            raise ReconstructionValidationError(
                "Local metric uncertainty requires an explicit metric scale reference."
            )
        result = self.local_uncertainty_reconstruction_units.astype(np.float64) * float(
            scale_profile.gauge.unit_to_metre
        )
        result.setflags(write=False)
        return result


@dataclass(frozen=True, slots=True)
class DenseGeometryQualitySummary:
    global_level: GeometryQualityLevel
    global_score: float
    high_quality_ratio: float
    medium_quality_ratio: float
    low_quality_ratio: float
    accepted_ratio: float
    input_depth_coverage_ratio: float
    unobserved_input_sample_ratio: float
    multi_view_ratio: float
    mean_view_observations: float
    median_view_observations: float
    mean_confidence: float
    pose_reliability_score: float
    camera_reliability_score: float
    evidence_completeness_ratio: float
    capacity_drop_ratio: float
    median_local_uncertainty_reconstruction_units: float
    p95_local_uncertainty_reconstruction_units: float
    median_local_uncertainty_m: float | None
    p95_local_uncertainty_m: float | None
    relative_scale_uncertainty: float | None

    def __post_init__(self) -> None:
        if not isinstance(self.global_level, GeometryQualityLevel):
            raise TypeError("global_level must be GeometryQualityLevel.")
        bounded = (
            "global_score",
            "high_quality_ratio",
            "medium_quality_ratio",
            "low_quality_ratio",
            "accepted_ratio",
            "input_depth_coverage_ratio",
            "unobserved_input_sample_ratio",
            "multi_view_ratio",
            "mean_confidence",
            "pose_reliability_score",
            "camera_reliability_score",
            "evidence_completeness_ratio",
            "capacity_drop_ratio",
        )
        for name in bounded:
            value = float(getattr(self, name))
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise ReconstructionValidationError(f"{name} must be finite in [0, 1].")
            object.__setattr__(self, name, value)
        for name in (
            "mean_view_observations",
            "median_view_observations",
            "median_local_uncertainty_reconstruction_units",
            "p95_local_uncertainty_reconstruction_units",
        ):
            value = float(getattr(self, name))
            if not isfinite(value) or value < 0.0:
                raise ReconstructionValidationError(f"{name} must be finite and non-negative.")
            object.__setattr__(self, name, value)
        for name in ("median_local_uncertainty_m", "p95_local_uncertainty_m"):
            value = getattr(self, name)
            if value is not None:
                value = float(value)
                if not isfinite(value) or value < 0.0:
                    raise ReconstructionValidationError(f"{name} must be finite and non-negative.")
                object.__setattr__(self, name, value)
        scale_uncertainty = self.relative_scale_uncertainty
        if scale_uncertainty is not None:
            scale_uncertainty = float(scale_uncertainty)
            if not isfinite(scale_uncertainty) or scale_uncertainty < 0.0:
                raise ReconstructionValidationError(
                    "relative_scale_uncertainty must be finite and non-negative."
                )
            object.__setattr__(self, "relative_scale_uncertainty", scale_uncertainty)

    def as_dict(self) -> dict[str, Any]:
        return {
            "global_level": self.global_level.value,
            "global_score": self.global_score,
            "high_quality_ratio": self.high_quality_ratio,
            "medium_quality_ratio": self.medium_quality_ratio,
            "low_quality_ratio": self.low_quality_ratio,
            "accepted_ratio": self.accepted_ratio,
            "input_depth_coverage_ratio": self.input_depth_coverage_ratio,
            "unobserved_input_sample_ratio": self.unobserved_input_sample_ratio,
            "multi_view_ratio": self.multi_view_ratio,
            "mean_view_observations": self.mean_view_observations,
            "median_view_observations": self.median_view_observations,
            "mean_confidence": self.mean_confidence,
            "pose_reliability_score": self.pose_reliability_score,
            "camera_reliability_score": self.camera_reliability_score,
            "evidence_completeness_ratio": self.evidence_completeness_ratio,
            "capacity_drop_ratio": self.capacity_drop_ratio,
            "median_local_uncertainty_reconstruction_units": self.median_local_uncertainty_reconstruction_units,
            "p95_local_uncertainty_reconstruction_units": self.p95_local_uncertainty_reconstruction_units,
            "median_local_uncertainty_m": self.median_local_uncertainty_m,
            "p95_local_uncertainty_m": self.p95_local_uncertainty_m,
            "relative_scale_uncertainty": self.relative_scale_uncertainty,
        }


@dataclass(frozen=True, slots=True)
class DenseGeometryQualityReport:
    session_id: str
    scene_profile: DenseSceneProfile
    camera_profile_id: str
    scale_profile_id: str
    metric_geometry_status: MetricGeometryStatus
    pose_convention: str
    reconstruction_frame_id: str
    certified_metrology: bool
    policy: DenseGeometryQualityPolicy
    evidence: DenseGeometryEvidence
    point_quality: DenseGeometryPointQuality
    summary: DenseGeometryQualitySummary
    quality_limiters: tuple[str, ...]
    unobserved_semantics: str = "sampled_input_depth_only"
    schema_version: int = DENSE_GEOMETRY_QUALITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_id", require_nonempty(self.session_id, field_name="session_id"))
        if not isinstance(self.scene_profile, DenseSceneProfile):
            raise TypeError("scene_profile must be DenseSceneProfile.")
        object.__setattr__(
            self,
            "camera_profile_id",
            require_nonempty(self.camera_profile_id, field_name="camera_profile_id"),
        )
        object.__setattr__(
            self,
            "scale_profile_id",
            require_nonempty(self.scale_profile_id, field_name="scale_profile_id"),
        )
        if not isinstance(self.metric_geometry_status, MetricGeometryStatus):
            raise TypeError("metric_geometry_status must be MetricGeometryStatus.")
        object.__setattr__(
            self,
            "pose_convention",
            require_nonempty(self.pose_convention, field_name="pose_convention"),
        )
        object.__setattr__(
            self,
            "reconstruction_frame_id",
            require_nonempty(self.reconstruction_frame_id, field_name="reconstruction_frame_id"),
        )
        if bool(self.certified_metrology):
            raise ReconstructionValidationError(
                "Phase 8.6D quality reports must not claim certified metrology."
            )
        if not isinstance(self.policy, DenseGeometryQualityPolicy):
            raise TypeError("policy must be DenseGeometryQualityPolicy.")
        if not isinstance(self.evidence, DenseGeometryEvidence):
            raise TypeError("evidence must be DenseGeometryEvidence.")
        if not isinstance(self.point_quality, DenseGeometryPointQuality):
            raise TypeError("point_quality must be DenseGeometryPointQuality.")
        if not isinstance(self.summary, DenseGeometryQualitySummary):
            raise TypeError("summary must be DenseGeometryQualitySummary.")
        if self.evidence.point_count != self.point_quality.point_count:
            raise ReconstructionValidationError("Evidence and point-quality counts must match.")
        limiters = tuple(str(value).strip() for value in self.quality_limiters if str(value).strip())
        object.__setattr__(self, "quality_limiters", tuple(dict.fromkeys(limiters)))
        object.__setattr__(self, "certified_metrology", False)
        object.__setattr__(self, "unobserved_semantics", "sampled_input_depth_only")
        if int(self.schema_version) != DENSE_GEOMETRY_QUALITY_SCHEMA_VERSION:
            raise ReconstructionValidationError("Unsupported dense geometry quality schema version.")
        object.__setattr__(self, "schema_version", DENSE_GEOMETRY_QUALITY_SCHEMA_VERSION)

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
            "certified_metrology": self.certified_metrology,
            "policy": self.policy.as_dict(),
            "evidence": self.evidence.as_dict(),
            "summary": self.summary.as_dict(),
            "quality_limiters": list(self.quality_limiters),
            "unobserved_semantics": self.unobserved_semantics,
        }


def _camera_reliability(camera_profile: CameraGeometryProfile) -> float:
    if camera_profile.mode is CameraGeometryMode.CALIBRATED:
        return 1.0
    uncertainty = camera_profile.focal_length_relative_uncertainty
    if uncertainty is None:
        score = 0.55
    else:
        score = float(np.clip(0.85 - 1.25 * float(uncertainty), 0.45, 0.85))
    if camera_profile.distortion_assumed_zero:
        score -= 0.05
    return float(np.clip(score, 0.35, 0.85))


def _pose_layer_reliability(layer: str) -> float:
    normalized = str(layer).strip().lower()
    if normalized == "optimized_keyframe":
        return 1.0
    if normalized == "optimized_propagated":
        return 0.90
    if normalized == "tracking":
        return 0.70
    return 0.55


def _global_level(
    *,
    score: float,
    high_ratio: float,
    accepted_ratio: float,
    input_depth_coverage_ratio: float,
    policy: DenseGeometryQualityPolicy,
) -> GeometryQualityLevel:
    if input_depth_coverage_ratio <= 0.0:
        return GeometryQualityLevel.UNOBSERVED
    if (
        score >= policy.high_score_threshold
        and high_ratio >= 0.50
        and accepted_ratio >= 0.75
        and input_depth_coverage_ratio >= policy.minimum_input_depth_coverage
    ):
        return GeometryQualityLevel.HIGH
    if (
        score >= policy.medium_score_threshold
        and accepted_ratio >= 0.50
        and input_depth_coverage_ratio >= policy.minimum_input_depth_coverage * 0.75
    ):
        return GeometryQualityLevel.MEDIUM
    return GeometryQualityLevel.LOW


def assess_dense_geometry_quality(
    result: DenseFusionResult,
    *,
    camera_profile: CameraGeometryProfile,
    scale_profile: ScaleReferenceProfile,
    policy: DenseGeometryQualityPolicy | None = None,
    evidence: DenseGeometryEvidence | None = None,
) -> DenseGeometryQualityReport:
    """Assess dense geometry and attach conservative local uncertainty.

    The report deliberately separates *local geometry uncertainty* from metric
    scale uncertainty.  Multiplying local uncertainty by ``unit_to_metre`` is
    allowed when a metric gauge exists, but scale uncertainty remains a separate
    relative term because combining it into an absolute point-coordinate error
    would depend on the chosen reference origin and measurement task.
    """

    if not isinstance(result, DenseFusionResult):
        raise TypeError("result must be DenseFusionResult.")
    if not isinstance(camera_profile, CameraGeometryProfile):
        raise TypeError("camera_profile must be CameraGeometryProfile.")
    if not isinstance(scale_profile, ScaleReferenceProfile):
        raise TypeError("scale_profile must be ScaleReferenceProfile.")
    if result.camera_profile_id != camera_profile.profile_id:
        raise ReconstructionValidationError("Dense result camera profile ID does not match.")
    if result.scale_profile_id != scale_profile.profile_id:
        raise ReconstructionValidationError("Dense result scale profile ID does not match.")
    if result.metric_geometry_status is MetricGeometryStatus.ARBITRARY and scale_profile.is_metric:
        raise ReconstructionValidationError("Dense result and scale profile metric semantics disagree.")
    if result.metric_geometry_status is not MetricGeometryStatus.ARBITRARY and not scale_profile.is_metric:
        raise ReconstructionValidationError("Metric dense result requires a metric scale profile.")

    policy = policy or default_dense_geometry_quality_policy(result.scene_profile)
    if not isinstance(policy, DenseGeometryQualityPolicy):
        raise TypeError("policy must be DenseGeometryQualityPolicy.")
    if policy.scene_profile is not result.scene_profile:
        raise ReconstructionValidationError("Quality policy scene profile must match dense result.")

    cloud = result.point_cloud
    point_count = len(cloud.points_reconstruction)
    evidence = evidence or DenseGeometryEvidence(point_count=point_count)
    if not isinstance(evidence, DenseGeometryEvidence):
        raise TypeError("evidence must be DenseGeometryEvidence.")
    if evidence.point_count != point_count:
        raise ReconstructionValidationError("Evidence point_count must match dense point cloud.")

    views = cloud.view_observation_counts.astype(np.float64)
    confidence = cloud.mean_confidence.astype(np.float64)
    view_score = np.clip(views / float(policy.target_view_observations), 0.0, 1.0)
    confidence_score = np.clip(confidence, 0.0, 1.0)

    pose_scores = [_pose_layer_reliability(layer) for layer in result.source_pose_layers]
    pose_reliability = float(np.mean(pose_scores)) if pose_scores else 0.0
    camera_reliability = _camera_reliability(camera_profile)

    if evidence.depth_relative_residual is None:
        depth_score = np.full(point_count, 0.5, dtype=np.float64)
    else:
        depth_score = np.clip(
            1.0
            - evidence.depth_relative_residual.astype(np.float64)
            / policy.maximum_depth_relative_residual,
            0.0,
            1.0,
        )
    if evidence.view_angle_span_degrees is None:
        angle_score = np.full(point_count, 0.5, dtype=np.float64)
    else:
        angle_score = np.clip(
            evidence.view_angle_span_degrees.astype(np.float64)
            / policy.desired_view_angle_span_degrees,
            0.0,
            1.0,
        )
    if evidence.visibility_ratio is None:
        visibility_score = np.full(point_count, 0.5, dtype=np.float64)
    else:
        visibility_score = evidence.visibility_ratio.astype(np.float64)

    scores = (
        0.20 * view_score
        + 0.15 * confidence_score
        + 0.15 * pose_reliability
        + 0.15 * camera_reliability
        + 0.15 * depth_score
        + 0.10 * angle_score
        + 0.10 * visibility_score
    )
    scores = np.clip(scores, 0.0, 1.0)

    accepted = (
        (views >= policy.minimum_view_observations)
        & (confidence >= policy.minimum_confidence)
        & (scores >= policy.medium_score_threshold)
    )
    codes = np.full(point_count, int(GeometryQualityCode.LOW), dtype=np.uint8)
    codes[accepted] = int(GeometryQualityCode.MEDIUM)
    codes[accepted & (scores >= policy.high_score_threshold)] = int(GeometryQualityCode.HIGH)

    base = float(cloud.voxel_size_reconstruction_units) * 0.5
    support_factor = np.sqrt(
        float(policy.target_view_observations)
        / np.clip(views, 1.0, float(policy.target_view_observations))
    )
    confidence_factor = 1.0 + 1.5 * (1.0 - confidence_score)
    pose_factor = 1.0 + 0.75 * (1.0 - pose_reliability)
    camera_factor = 1.0 + (1.0 - camera_reliability)
    depth_factor = 1.0 + 1.5 * (1.0 - depth_score)
    angle_factor = 1.0 + 0.5 * (1.0 - angle_score)
    visibility_factor = 1.0 + 0.5 * (1.0 - visibility_score)
    uncertainty = (
        base
        * support_factor
        * confidence_factor
        * pose_factor
        * camera_factor
        * depth_factor
        * angle_factor
        * visibility_factor
    )

    point_quality = DenseGeometryPointQuality(
        quality_scores=scores.astype(np.float32),
        quality_codes=codes,
        accepted_mask=accepted,
        local_uncertainty_reconstruction_units=uncertainty.astype(np.float32),
    )

    stats = result.statistics
    sampled = max(stats.sampled_pixels, 1)
    coverage = float(np.clip(stats.valid_depth_samples / sampled, 0.0, 1.0))
    unobserved_samples = 1.0 - coverage
    multi_view_ratio = float(
        np.clip(stats.multi_view_voxels / max(stats.retained_voxels, 1), 0.0, 1.0)
    )
    capacity_denominator = stats.retained_voxels + stats.dropped_new_voxels_capacity
    capacity_drop_ratio = (
        float(stats.dropped_new_voxels_capacity / capacity_denominator)
        if capacity_denominator
        else 0.0
    )

    high_ratio = float(np.count_nonzero(codes == int(GeometryQualityCode.HIGH)) / point_count)
    medium_ratio = float(np.count_nonzero(codes == int(GeometryQualityCode.MEDIUM)) / point_count)
    low_ratio = float(np.count_nonzero(codes == int(GeometryQualityCode.LOW)) / point_count)
    accepted_ratio = float(np.count_nonzero(accepted) / point_count)
    global_score = float(np.mean(scores))
    global_score *= 0.90 + 0.10 * coverage
    global_score *= 1.0 - 0.10 * capacity_drop_ratio
    global_score = float(np.clip(global_score, 0.0, 1.0))
    level = _global_level(
        score=global_score,
        high_ratio=high_ratio,
        accepted_ratio=accepted_ratio,
        input_depth_coverage_ratio=coverage,
        policy=policy,
    )

    median_uncertainty = float(np.median(uncertainty))
    p95_uncertainty = float(np.quantile(uncertainty, 0.95))
    median_m: float | None = None
    p95_m: float | None = None
    if scale_profile.is_metric and scale_profile.gauge.unit_to_metre is not None:
        factor = float(scale_profile.gauge.unit_to_metre)
        median_m = median_uncertainty * factor
        p95_m = p95_uncertainty * factor

    summary = DenseGeometryQualitySummary(
        global_level=level,
        global_score=global_score,
        high_quality_ratio=high_ratio,
        medium_quality_ratio=medium_ratio,
        low_quality_ratio=low_ratio,
        accepted_ratio=accepted_ratio,
        input_depth_coverage_ratio=coverage,
        unobserved_input_sample_ratio=unobserved_samples,
        multi_view_ratio=multi_view_ratio,
        mean_view_observations=float(np.mean(views)),
        median_view_observations=float(np.median(views)),
        mean_confidence=float(np.mean(confidence)),
        pose_reliability_score=pose_reliability,
        camera_reliability_score=camera_reliability,
        evidence_completeness_ratio=evidence.available_signal_count / 3.0,
        capacity_drop_ratio=capacity_drop_ratio,
        median_local_uncertainty_reconstruction_units=median_uncertainty,
        p95_local_uncertainty_reconstruction_units=p95_uncertainty,
        median_local_uncertainty_m=median_m,
        p95_local_uncertainty_m=p95_m,
        relative_scale_uncertainty=scale_profile.gauge.relative_scale_uncertainty,
    )

    limiters: list[str] = []
    if camera_profile.mode is CameraGeometryMode.ESTIMATED:
        limiters.append("estimated_camera_intrinsics")
    if result.metric_geometry_status is MetricGeometryStatus.ARBITRARY:
        limiters.append("arbitrary_scale_no_metric_claim")
    elif result.metric_geometry_status is MetricGeometryStatus.PROVISIONAL_METRIC:
        limiters.append("provisional_metric_geometry")
    if evidence.depth_relative_residual is None:
        limiters.append("depth_consistency_evidence_unavailable")
    if evidence.view_angle_span_degrees is None:
        limiters.append("view_diversity_evidence_unavailable")
    if evidence.visibility_ratio is None:
        limiters.append("visibility_evidence_unavailable")
    if coverage < policy.minimum_input_depth_coverage:
        limiters.append("insufficient_input_depth_coverage")
    if capacity_drop_ratio > 0.0:
        limiters.append("fusion_capacity_saturation")
    if scale_profile.gauge.relative_scale_uncertainty is not None:
        limiters.append("metric_scale_uncertainty_present")

    return DenseGeometryQualityReport(
        session_id=result.session_id,
        scene_profile=result.scene_profile,
        camera_profile_id=result.camera_profile_id,
        scale_profile_id=result.scale_profile_id,
        metric_geometry_status=result.metric_geometry_status,
        pose_convention=result.pose_convention,
        reconstruction_frame_id=result.reconstruction_frame_id,
        certified_metrology=False,
        policy=policy,
        evidence=evidence,
        point_quality=point_quality,
        summary=summary,
        quality_limiters=tuple(limiters),
    )

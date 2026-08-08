"""Reference registration and geometry inspection/QC for Phase 8.7C.

Phase 8.7C consumes persistent semantic 3D geometry from Phase 8.7B and a
trusted semantic reference built from a golden reconstruction, a CAD-derived
feature model, or another explicit reference source.  The layer is designed
for safe connector/object inspection rather than certified metrology:

* registration is explicit and provenance-preserving,
* arbitrary-scale comparisons stay relative,
* metric values are emitted only when both candidate and reference carry
  compatible metric scale semantics,
* weak geometry yields ``INSUFFICIENT_EVIDENCE`` rather than a false PASS,
* connector-specific labels (missing/bent/misaligned pin) are derived from
  generic geometric deviations, and
* ``certified_metrology`` is always false in Phase 8.7C.

The implementation intentionally operates on semantic feature geometry and
quality-aware mesh points.  General CAD file parsing and free-form mesh ICP
are outside this bounded step; a CAD backend can populate the same
``ReferenceAssembly`` contract later without changing inspection semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import acos, degrees, isfinite
from pathlib import Path
import json
from typing import Any, Iterable, Mapping

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from core.contracts.validation import require_nonempty
from core.reconstruction.dense_fusion import DenseSceneProfile
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.scale_reference import MetricGeometryStatus
from core.reconstruction.semantic_geometry import (
    SEMANTIC_GEOMETRY_POSE_CONVENTION,
    Semantic3DObject,
    SemanticGeometryResult,
    SemanticGeometryRole,
)
from core.reconstruction.surface_mesh import SurfaceMeshResult

INSPECTION_QC_SCHEMA_VERSION = 1
INSPECTION_QC_POSE_CONVENTION = SEMANTIC_GEOMETRY_POSE_CONVENTION


class ReferenceGeometrySource(str, Enum):
    GOLDEN_RECONSTRUCTION = "golden_reconstruction"
    CAD_FEATURE_MODEL = "cad_feature_model"
    MANUAL_REFERENCE = "manual_reference"


class RegistrationMethod(str, Enum):
    ORIENTED_SEMANTIC_ANCHOR = "oriented_semantic_anchor"
    MULTI_ANCHOR_SIMILARITY = "multi_anchor_similarity"


class RegistrationStatus(str, Enum):
    SUCCESS = "success"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    DEGENERATE = "degenerate"
    RESIDUAL_TOO_HIGH = "residual_too_high"


class InspectionDecision(str, Enum):
    PASS = "pass"
    WARN = "warn"
    REJECT = "reject"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class InspectionFindingKind(str, Enum):
    WITHIN_TOLERANCE = "within_tolerance"
    MISSING_FEATURE = "missing_feature"
    EXTRA_FEATURE = "extra_feature"
    POSITION_DEVIATION = "position_deviation"
    ORIENTATION_DEVIATION = "orientation_deviation"
    EXTENT_DEVIATION = "extent_deviation"
    LOW_GEOMETRY_QUALITY = "low_geometry_quality"
    HIGH_GEOMETRY_UNCERTAINTY = "high_geometry_uncertainty"
    REGISTRATION_INSUFFICIENT_EVIDENCE = "registration_insufficient_evidence"


@dataclass(frozen=True, slots=True)
class FeatureTolerance:
    """Engineering tolerance in reference reconstruction units plus angles.

    The contract deliberately keeps positional thresholds in the reference
    coordinate system.  When a reference is metric, reports additionally
    expose the corresponding metre values.  This avoids silently treating an
    arbitrary reconstruction unit as a physical unit.
    """

    position_warn: float
    position_reject: float
    orientation_warn_degrees: float = 5.0
    orientation_reject_degrees: float = 12.0
    extent_relative_warn: float = 0.08
    extent_relative_reject: float = 0.15
    minimum_geometry_quality: float = 0.60
    maximum_uncertainty_fraction: float = 0.30

    def __post_init__(self) -> None:
        _positive_ordered(self.position_warn, self.position_reject, "position")
        _positive_ordered(
            self.orientation_warn_degrees,
            self.orientation_reject_degrees,
            "orientation",
        )
        _positive_ordered(
            self.extent_relative_warn,
            self.extent_relative_reject,
            "extent_relative",
        )
        quality = float(self.minimum_geometry_quality)
        if not isfinite(quality) or not 0.0 <= quality <= 1.0:
            raise ReconstructionValidationError(
                "minimum_geometry_quality must be finite in [0, 1]."
            )
        uncertainty = float(self.maximum_uncertainty_fraction)
        if not isfinite(uncertainty) or uncertainty <= 0.0:
            raise ReconstructionValidationError(
                "maximum_uncertainty_fraction must be finite and positive."
            )
        object.__setattr__(self, "position_warn", float(self.position_warn))
        object.__setattr__(self, "position_reject", float(self.position_reject))
        object.__setattr__(
            self, "orientation_warn_degrees", float(self.orientation_warn_degrees)
        )
        object.__setattr__(
            self, "orientation_reject_degrees", float(self.orientation_reject_degrees)
        )
        object.__setattr__(self, "extent_relative_warn", float(self.extent_relative_warn))
        object.__setattr__(self, "extent_relative_reject", float(self.extent_relative_reject))
        object.__setattr__(self, "minimum_geometry_quality", quality)
        object.__setattr__(self, "maximum_uncertainty_fraction", uncertainty)

    def as_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class RegistrationInspectionPolicy:
    maximum_reference_features: int = 4_096
    maximum_candidate_objects: int = 4_096
    maximum_anchor_sample_points: int = 512
    minimum_registration_anchor_quality: float = 0.60
    maximum_registration_rms_fraction: float = 0.12
    maximum_auto_association_distance_fraction: float = 0.60
    extra_feature_decision: InspectionDecision = InspectionDecision.WARN

    def __post_init__(self) -> None:
        for name in (
            "maximum_reference_features",
            "maximum_candidate_objects",
            "maximum_anchor_sample_points",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) < 1:
                raise ReconstructionValidationError(f"{name} must be >= 1.")
            object.__setattr__(self, name, int(value))
        quality = float(self.minimum_registration_anchor_quality)
        if not isfinite(quality) or not 0.0 <= quality <= 1.0:
            raise ReconstructionValidationError(
                "minimum_registration_anchor_quality must be finite in [0, 1]."
            )
        for name in (
            "maximum_registration_rms_fraction",
            "maximum_auto_association_distance_fraction",
        ):
            value = float(getattr(self, name))
            if not isfinite(value) or value <= 0.0:
                raise ReconstructionValidationError(f"{name} must be finite and positive.")
            object.__setattr__(self, name, value)
        if not isinstance(self.extra_feature_decision, InspectionDecision):
            raise TypeError("extra_feature_decision must be InspectionDecision.")
        if self.extra_feature_decision not in {InspectionDecision.WARN, InspectionDecision.REJECT}:
            raise ReconstructionValidationError(
                "extra_feature_decision must be WARN or REJECT."
            )
        object.__setattr__(self, "minimum_registration_anchor_quality", quality)

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            result[name] = value.value if isinstance(value, Enum) else value
        return result


@dataclass(frozen=True, slots=True)
class ReferenceFeature:
    feature_id: str
    association_key: str
    class_id: int
    class_name: str
    semantic_role: SemanticGeometryRole
    centroid_reference: np.ndarray
    principal_axes_reference: np.ndarray
    oriented_extents_reference: np.ndarray
    orientation_reliable: bool
    confidence: float
    mean_geometry_quality: float
    median_uncertainty_reference_units: float
    required: bool
    registration_anchor: bool
    tolerance: FeatureTolerance
    registration_points_reference: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "feature_id", require_nonempty(self.feature_id, field_name="feature_id"))
        object.__setattr__(
            self,
            "association_key",
            require_nonempty(self.association_key, field_name="association_key"),
        )
        if isinstance(self.class_id, bool) or int(self.class_id) < 0:
            raise ReconstructionValidationError("class_id must be a non-negative integer.")
        object.__setattr__(self, "class_id", int(self.class_id))
        object.__setattr__(self, "class_name", require_nonempty(self.class_name, field_name="class_name"))
        if not isinstance(self.semantic_role, SemanticGeometryRole):
            raise TypeError("semantic_role must be SemanticGeometryRole.")
        centroid = _vector3(self.centroid_reference, "centroid_reference")
        axes = _axes(self.principal_axes_reference, "principal_axes_reference")
        extents = _nonnegative_vector3(self.oriented_extents_reference, "oriented_extents_reference")
        if float(np.max(extents)) <= 0.0:
            raise ReconstructionValidationError("Reference feature must have non-zero extent.")
        confidence = _probability(self.confidence, "confidence")
        quality = _probability(self.mean_geometry_quality, "mean_geometry_quality")
        uncertainty = float(self.median_uncertainty_reference_units)
        if not isfinite(uncertainty) or uncertainty <= 0.0:
            raise ReconstructionValidationError(
                "median_uncertainty_reference_units must be finite and positive."
            )
        if not isinstance(self.tolerance, FeatureTolerance):
            raise TypeError("tolerance must be FeatureTolerance.")
        points = np.asarray(self.registration_points_reference, dtype=np.float64)
        if points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all():
            raise ReconstructionValidationError("registration_points_reference must be finite Nx3.")
        if bool(self.registration_anchor) and len(points) < 3:
            raise ReconstructionValidationError(
                "Registration anchors require at least three sampled points."
            )
        if not bool(self.registration_anchor) and len(points) != 0:
            raise ReconstructionValidationError(
                "Non-anchor reference features must not carry registration point samples."
            )
        object.__setattr__(self, "centroid_reference", _readonly(centroid, np.float64))
        object.__setattr__(self, "principal_axes_reference", _readonly(axes, np.float64))
        object.__setattr__(self, "oriented_extents_reference", _readonly(extents, np.float64))
        object.__setattr__(self, "orientation_reliable", bool(self.orientation_reliable))
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "mean_geometry_quality", quality)
        object.__setattr__(self, "median_uncertainty_reference_units", uncertainty)
        object.__setattr__(self, "required", bool(self.required))
        object.__setattr__(self, "registration_anchor", bool(self.registration_anchor))
        object.__setattr__(self, "registration_points_reference", _readonly(points, np.float64))

    @property
    def characteristic_extent(self) -> float:
        return float(np.max(self.oriented_extents_reference))

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "association_key": self.association_key,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "semantic_role": self.semantic_role.value,
            "centroid_reference": self.centroid_reference.tolist(),
            "principal_axes_reference": self.principal_axes_reference.tolist(),
            "oriented_extents_reference": self.oriented_extents_reference.tolist(),
            "orientation_reliable": self.orientation_reliable,
            "confidence": self.confidence,
            "mean_geometry_quality": self.mean_geometry_quality,
            "median_uncertainty_reference_units": self.median_uncertainty_reference_units,
            "required": self.required,
            "registration_anchor": self.registration_anchor,
            "tolerance": self.tolerance.as_dict(),
            "registration_point_count": int(len(self.registration_points_reference)),
        }


@dataclass(frozen=True, slots=True)
class ReferenceAssembly:
    reference_id: str
    source: ReferenceGeometrySource
    scene_profile: DenseSceneProfile
    reference_frame_id: str
    metric_geometry_status: MetricGeometryStatus
    unit_to_metre: float | None
    relative_scale_uncertainty: float | None
    features: tuple[ReferenceFeature, ...]
    certified_metrology: bool = False
    schema_version: int = INSPECTION_QC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "reference_id", require_nonempty(self.reference_id, field_name="reference_id"))
        if not isinstance(self.source, ReferenceGeometrySource):
            raise TypeError("source must be ReferenceGeometrySource.")
        if not isinstance(self.scene_profile, DenseSceneProfile):
            raise TypeError("scene_profile must be DenseSceneProfile.")
        object.__setattr__(
            self,
            "reference_frame_id",
            require_nonempty(self.reference_frame_id, field_name="reference_frame_id"),
        )
        if not isinstance(self.metric_geometry_status, MetricGeometryStatus):
            raise TypeError("metric_geometry_status must be MetricGeometryStatus.")
        unit = self.unit_to_metre
        rel = self.relative_scale_uncertainty
        if self.metric_geometry_status is MetricGeometryStatus.ARBITRARY:
            if unit is not None or rel is not None:
                raise ReconstructionValidationError(
                    "Arbitrary reference geometry cannot expose metric scale values."
                )
        else:
            if unit is None:
                raise ReconstructionValidationError("Metric reference geometry requires unit_to_metre.")
            unit = float(unit)
            if not isfinite(unit) or unit <= 0.0:
                raise ReconstructionValidationError("unit_to_metre must be finite and positive.")
            if rel is not None:
                rel = float(rel)
                if not isfinite(rel) or rel < 0.0:
                    raise ReconstructionValidationError(
                        "relative_scale_uncertainty must be finite and non-negative."
                    )
        features = tuple(self.features)
        if not features:
            raise ReconstructionValidationError("Reference assembly must contain at least one feature.")
        if len({feature.feature_id for feature in features}) != len(features):
            raise ReconstructionValidationError("Reference feature IDs must be unique.")
        if len({feature.association_key for feature in features}) != len(features):
            raise ReconstructionValidationError("Reference association keys must be unique.")
        if not any(feature.registration_anchor for feature in features):
            raise ReconstructionValidationError("Reference assembly requires a registration anchor.")
        if bool(self.certified_metrology):
            raise ReconstructionValidationError("Phase 8.7C cannot claim certified metrology.")
        object.__setattr__(self, "unit_to_metre", unit)
        object.__setattr__(self, "relative_scale_uncertainty", rel)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "certified_metrology", False)
        object.__setattr__(self, "schema_version", INSPECTION_QC_SCHEMA_VERSION)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "reference_id": self.reference_id,
            "source": self.source.value,
            "scene_profile": self.scene_profile.value,
            "reference_frame_id": self.reference_frame_id,
            "metric_geometry_status": self.metric_geometry_status.value,
            "unit_to_metre": self.unit_to_metre,
            "relative_scale_uncertainty": self.relative_scale_uncertainty,
            "certified_metrology": False,
            "features": [feature.as_dict() for feature in self.features],
        }


@dataclass(frozen=True, slots=True)
class RegistrationResult:
    status: RegistrationStatus
    method: RegistrationMethod | None
    scale_candidate_to_reference: float | None
    T_reference_from_scaled_candidate: np.ndarray | None
    anchor_feature_ids: tuple[str, ...]
    rms_residual_reference_units: float | None
    maximum_residual_reference_units: float | None
    rms_residual_m: float | None
    maximum_residual_m: float | None
    normalized_rms_fraction: float | None
    confidence: float
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.status, RegistrationStatus):
            raise TypeError("status must be RegistrationStatus.")
        if self.method is not None and not isinstance(self.method, RegistrationMethod):
            raise TypeError("method must be RegistrationMethod or None.")
        successful = self.status is RegistrationStatus.SUCCESS
        if successful:
            if self.method is None or self.scale_candidate_to_reference is None or self.T_reference_from_scaled_candidate is None:
                raise ReconstructionValidationError("Successful registration requires method, scale, and transform.")
            scale = float(self.scale_candidate_to_reference)
            if not isfinite(scale) or scale <= 0.0:
                raise ReconstructionValidationError("Registration scale must be finite and positive.")
            matrix = np.asarray(self.T_reference_from_scaled_candidate, dtype=np.float64)
            if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
                raise ReconstructionValidationError("T_reference_from_scaled_candidate must be finite 4x4.")
            if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
                raise ReconstructionValidationError("Registration transform must be homogeneous rigid.")
            rotation = matrix[:3, :3]
            if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or np.linalg.det(rotation) < 0.999:
                raise ReconstructionValidationError("Registration rotation must be right-handed orthonormal.")
            if not self.anchor_feature_ids:
                raise ReconstructionValidationError("Successful registration requires anchors.")
            for field_name in (
                "rms_residual_reference_units",
                "maximum_residual_reference_units",
                "normalized_rms_fraction",
            ):
                value = getattr(self, field_name)
                if value is None or not isfinite(float(value)) or float(value) < 0.0:
                    raise ReconstructionValidationError(f"{field_name} must be finite and non-negative.")
            object.__setattr__(self, "scale_candidate_to_reference", scale)
            object.__setattr__(self, "T_reference_from_scaled_candidate", _readonly(matrix, np.float64))
        else:
            if self.T_reference_from_scaled_candidate is not None:
                raise ReconstructionValidationError("Failed registration cannot expose a transform.")
            object.__setattr__(self, "scale_candidate_to_reference", None)
        object.__setattr__(self, "anchor_feature_ids", tuple(self.anchor_feature_ids))
        object.__setattr__(self, "confidence", _probability(self.confidence, "confidence"))
        object.__setattr__(self, "reason", str(self.reason).strip())

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "method": None if self.method is None else self.method.value,
            "scale_candidate_to_reference": self.scale_candidate_to_reference,
            "T_reference_from_scaled_candidate": (
                None
                if self.T_reference_from_scaled_candidate is None
                else self.T_reference_from_scaled_candidate.tolist()
            ),
            "anchor_feature_ids": list(self.anchor_feature_ids),
            "rms_residual_reference_units": self.rms_residual_reference_units,
            "maximum_residual_reference_units": self.maximum_residual_reference_units,
            "rms_residual_m": self.rms_residual_m,
            "maximum_residual_m": self.maximum_residual_m,
            "normalized_rms_fraction": self.normalized_rms_fraction,
            "confidence": self.confidence,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class InspectionFeatureResult:
    reference_feature_id: str
    candidate_object_id: str
    semantic_role: SemanticGeometryRole
    decision: InspectionDecision
    findings: tuple[InspectionFindingKind, ...]
    connector_defects: tuple[str, ...]
    position_error_reference_units: float | None
    position_error_m: float | None
    orientation_error_degrees: float | None
    extent_relative_error: float | None
    geometry_quality: float | None
    uncertainty_reference_units: float | None
    uncertainty_m: float | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reference_feature_id",
            require_nonempty(self.reference_feature_id, field_name="reference_feature_id"),
        )
        object.__setattr__(self, "candidate_object_id", str(self.candidate_object_id).strip())
        if not isinstance(self.semantic_role, SemanticGeometryRole):
            raise TypeError("semantic_role must be SemanticGeometryRole.")
        if not isinstance(self.decision, InspectionDecision):
            raise TypeError("decision must be InspectionDecision.")
        findings = tuple(self.findings)
        if not findings or any(not isinstance(value, InspectionFindingKind) for value in findings):
            raise ReconstructionValidationError("Inspection feature results require finding kinds.")
        object.__setattr__(self, "findings", findings)
        object.__setattr__(self, "connector_defects", tuple(dict.fromkeys(self.connector_defects)))
        for field_name in (
            "position_error_reference_units",
            "position_error_m",
            "orientation_error_degrees",
            "extent_relative_error",
            "uncertainty_reference_units",
            "uncertainty_m",
        ):
            value = getattr(self, field_name)
            if value is not None:
                value = float(value)
                if not isfinite(value) or value < 0.0:
                    raise ReconstructionValidationError(f"{field_name} must be finite and non-negative.")
                object.__setattr__(self, field_name, value)
        if self.geometry_quality is not None:
            object.__setattr__(self, "geometry_quality", _probability(self.geometry_quality, "geometry_quality"))

    def as_dict(self) -> dict[str, Any]:
        return {
            "reference_feature_id": self.reference_feature_id,
            "candidate_object_id": self.candidate_object_id,
            "semantic_role": self.semantic_role.value,
            "decision": self.decision.value,
            "findings": [value.value for value in self.findings],
            "connector_defects": list(self.connector_defects),
            "position_error_reference_units": self.position_error_reference_units,
            "position_error_m": self.position_error_m,
            "orientation_error_degrees": self.orientation_error_degrees,
            "extent_relative_error": self.extent_relative_error,
            "geometry_quality": self.geometry_quality,
            "uncertainty_reference_units": self.uncertainty_reference_units,
            "uncertainty_m": self.uncertainty_m,
        }


@dataclass(frozen=True, slots=True)
class InspectionStatistics:
    reference_feature_count: int
    matched_feature_count: int
    missing_required_count: int
    extra_candidate_count: int
    pass_count: int
    warn_count: int
    reject_count: int
    insufficient_evidence_count: int

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) < 0:
                raise ReconstructionValidationError(f"{name} must be a non-negative integer.")
            object.__setattr__(self, name, int(value))

    def as_dict(self) -> dict[str, int]:
        return {name: int(getattr(self, name)) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class InspectionReport:
    session_id: str
    reference_id: str
    scene_profile: DenseSceneProfile
    pose_convention: str
    candidate_frame_id: str
    reference_frame_id: str
    candidate_metric_geometry_status: MetricGeometryStatus
    reference_metric_geometry_status: MetricGeometryStatus
    comparison_metric_available: bool
    reference_unit_to_metre: float | None
    relative_scale_uncertainty: float | None
    registration: RegistrationResult
    feature_results: tuple[InspectionFeatureResult, ...]
    extra_candidate_object_ids: tuple[str, ...]
    decision: InspectionDecision
    statistics: InspectionStatistics
    certified_metrology: bool = False
    schema_version: int = INSPECTION_QC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_id", require_nonempty(self.session_id, field_name="session_id"))
        object.__setattr__(self, "reference_id", require_nonempty(self.reference_id, field_name="reference_id"))
        if not isinstance(self.scene_profile, DenseSceneProfile):
            raise TypeError("scene_profile must be DenseSceneProfile.")
        object.__setattr__(self, "pose_convention", require_nonempty(self.pose_convention, field_name="pose_convention"))
        if self.pose_convention != INSPECTION_QC_POSE_CONVENTION:
            raise ReconstructionValidationError("Unsupported inspection pose convention.")
        object.__setattr__(self, "candidate_frame_id", require_nonempty(self.candidate_frame_id, field_name="candidate_frame_id"))
        object.__setattr__(self, "reference_frame_id", require_nonempty(self.reference_frame_id, field_name="reference_frame_id"))
        if not isinstance(self.candidate_metric_geometry_status, MetricGeometryStatus):
            raise TypeError("candidate_metric_geometry_status must be MetricGeometryStatus.")
        if not isinstance(self.reference_metric_geometry_status, MetricGeometryStatus):
            raise TypeError("reference_metric_geometry_status must be MetricGeometryStatus.")
        if not isinstance(self.registration, RegistrationResult):
            raise TypeError("registration must be RegistrationResult.")
        if not isinstance(self.decision, InspectionDecision):
            raise TypeError("decision must be InspectionDecision.")
        if not isinstance(self.statistics, InspectionStatistics):
            raise TypeError("statistics must be InspectionStatistics.")
        if bool(self.certified_metrology):
            raise ReconstructionValidationError("Phase 8.7C cannot claim certified metrology.")
        unit = self.reference_unit_to_metre
        if bool(self.comparison_metric_available):
            if unit is None:
                raise ReconstructionValidationError("Metric comparison requires reference_unit_to_metre.")
            unit = float(unit)
            if not isfinite(unit) or unit <= 0.0:
                raise ReconstructionValidationError("reference_unit_to_metre must be finite and positive.")
        else:
            if unit is not None:
                raise ReconstructionValidationError(
                    "Relative-only comparison cannot expose reference_unit_to_metre."
                )
        rel = self.relative_scale_uncertainty
        if rel is not None:
            rel = float(rel)
            if not isfinite(rel) or rel < 0.0:
                raise ReconstructionValidationError(
                    "relative_scale_uncertainty must be finite and non-negative."
                )
        object.__setattr__(self, "comparison_metric_available", bool(self.comparison_metric_available))
        object.__setattr__(self, "reference_unit_to_metre", unit)
        object.__setattr__(self, "relative_scale_uncertainty", rel)
        object.__setattr__(self, "feature_results", tuple(self.feature_results))
        object.__setattr__(
            self,
            "extra_candidate_object_ids",
            tuple(dict.fromkeys(str(value) for value in self.extra_candidate_object_ids)),
        )
        object.__setattr__(self, "certified_metrology", False)
        object.__setattr__(self, "schema_version", INSPECTION_QC_SCHEMA_VERSION)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "reference_id": self.reference_id,
            "scene_profile": self.scene_profile.value,
            "pose_convention": self.pose_convention,
            "candidate_frame_id": self.candidate_frame_id,
            "reference_frame_id": self.reference_frame_id,
            "candidate_metric_geometry_status": self.candidate_metric_geometry_status.value,
            "reference_metric_geometry_status": self.reference_metric_geometry_status.value,
            "comparison_metric_available": self.comparison_metric_available,
            "reference_unit_to_metre": self.reference_unit_to_metre,
            "relative_scale_uncertainty": self.relative_scale_uncertainty,
            "registration": self.registration.as_dict(),
            "feature_results": [value.as_dict() for value in self.feature_results],
            "extra_candidate_object_ids": list(self.extra_candidate_object_ids),
            "decision": self.decision.value,
            "statistics": self.statistics.as_dict(),
            "certified_metrology": False,
            "inspection_qc_enabled": True,
        }


def build_reference_assembly(
    surface: SurfaceMeshResult,
    semantic: SemanticGeometryResult,
    *,
    reference_id: str,
    source: ReferenceGeometrySource = ReferenceGeometrySource.GOLDEN_RECONSTRUCTION,
    default_tolerance: FeatureTolerance,
    feature_id_by_object_id: Mapping[str, str] | None = None,
    association_key_by_object_id: Mapping[str, str] | None = None,
    registration_anchor_feature_ids: Iterable[str] | None = None,
    optional_feature_ids: Iterable[str] = (),
    tolerance_by_feature_id: Mapping[str, FeatureTolerance] | None = None,
    maximum_anchor_sample_points: int = 512,
) -> ReferenceAssembly:
    """Build a bounded semantic reference from a trusted golden surface/result."""

    _validate_surface_semantic_pair(surface, semantic)
    if not isinstance(default_tolerance, FeatureTolerance):
        raise TypeError("default_tolerance must be FeatureTolerance.")
    if not isinstance(source, ReferenceGeometrySource):
        raise TypeError("source must be ReferenceGeometrySource.")
    max_points = int(maximum_anchor_sample_points)
    if max_points < 3:
        raise ReconstructionValidationError("maximum_anchor_sample_points must be >= 3.")
    feature_id_map = dict(feature_id_by_object_id or {})
    association_map = dict(association_key_by_object_id or {})
    tolerance_map = dict(tolerance_by_feature_id or {})
    object_features: list[tuple[Semantic3DObject, str, str]] = []
    for obj in semantic.objects:
        feature_id = str(feature_id_map.get(obj.object_id, _default_feature_key(obj))).strip()
        association_key = str(association_map.get(obj.object_id, feature_id)).strip()
        if not feature_id or not association_key:
            raise ReconstructionValidationError("Reference feature IDs and association keys must not be empty.")
        object_features.append((obj, feature_id, association_key))
    if len({feature_id for _, feature_id, _ in object_features}) != len(object_features):
        raise ReconstructionValidationError("Resolved reference feature IDs must be unique.")
    if len({key for _, _, key in object_features}) != len(object_features):
        raise ReconstructionValidationError("Resolved reference association keys must be unique.")

    if registration_anchor_feature_ids is None:
        preferred = [
            feature_id
            for obj, feature_id, _ in object_features
            if obj.semantic_role
            in {
                SemanticGeometryRole.CONNECTOR_BODY,
                SemanticGeometryRole.CONNECTOR_HOUSING,
                SemanticGeometryRole.REFERENCE_SURFACE,
            }
            and obj.orientation_reliable
        ]
        if preferred:
            anchor_ids = {preferred[0]}
        else:
            anchor_ids = {feature_id for _, feature_id, _ in object_features[:3]}
    else:
        anchor_ids = {str(value).strip() for value in registration_anchor_feature_ids}
    known_ids = {feature_id for _, feature_id, _ in object_features}
    if not anchor_ids or not anchor_ids.issubset(known_ids):
        raise ReconstructionValidationError("Registration anchor IDs must reference existing features.")
    optional_ids = {str(value).strip() for value in optional_feature_ids}
    if not optional_ids.issubset(known_ids):
        raise ReconstructionValidationError("Optional feature IDs must reference existing features.")

    features: list[ReferenceFeature] = []
    for obj, feature_id, association_key in object_features:
        points = surface.mesh.vertices_reconstruction[obj.vertex_indices.astype(np.int64)]
        descriptor = _descriptor_from_points(points)
        is_anchor = feature_id in anchor_ids
        samples = _deterministic_subsample(points, max_points) if is_anchor else np.empty((0, 3), np.float64)
        features.append(
            ReferenceFeature(
                feature_id=feature_id,
                association_key=association_key,
                class_id=obj.class_id,
                class_name=obj.class_name,
                semantic_role=obj.semantic_role,
                centroid_reference=descriptor.centroid,
                principal_axes_reference=descriptor.axes,
                oriented_extents_reference=descriptor.extents,
                orientation_reliable=descriptor.orientation_reliable and obj.orientation_reliable,
                confidence=obj.confidence,
                mean_geometry_quality=obj.mean_vertex_quality,
                median_uncertainty_reference_units=obj.median_vertex_uncertainty_reconstruction_units,
                required=feature_id not in optional_ids,
                registration_anchor=is_anchor,
                tolerance=tolerance_map.get(feature_id, default_tolerance),
                registration_points_reference=samples,
            )
        )
    return ReferenceAssembly(
        reference_id=reference_id,
        source=source,
        scene_profile=semantic.scene_profile,
        reference_frame_id=semantic.reconstruction_frame_id,
        metric_geometry_status=semantic.metric_geometry_status,
        unit_to_metre=semantic.unit_to_metre,
        relative_scale_uncertainty=semantic.relative_scale_uncertainty,
        features=tuple(features),
        certified_metrology=False,
    )


def inspect_semantic_geometry(
    reference: ReferenceAssembly,
    candidate_surface: SurfaceMeshResult,
    candidate_semantic: SemanticGeometryResult,
    *,
    policy: RegistrationInspectionPolicy | None = None,
    candidate_key_by_object_id: Mapping[str, str] | None = None,
) -> InspectionReport:
    """Register candidate semantics to a reference and produce safe QC decisions."""

    if not isinstance(reference, ReferenceAssembly):
        raise TypeError("reference must be ReferenceAssembly.")
    _validate_surface_semantic_pair(candidate_surface, candidate_semantic)
    if candidate_semantic.scene_profile is not reference.scene_profile:
        raise ReconstructionValidationError("Candidate and reference scene profiles must match.")
    resolved_policy = policy or RegistrationInspectionPolicy()
    if not isinstance(resolved_policy, RegistrationInspectionPolicy):
        raise TypeError("policy must be RegistrationInspectionPolicy.")
    if len(reference.features) > resolved_policy.maximum_reference_features:
        raise ReconstructionValidationError("Reference feature capacity exceeded.")
    if len(candidate_semantic.objects) > resolved_policy.maximum_candidate_objects:
        raise ReconstructionValidationError("Candidate object capacity exceeded.")

    comparison_metric_available, candidate_scale_to_reference_units = _scale_compatibility(
        reference, candidate_semantic
    )
    candidate_key_map = _candidate_key_map(candidate_semantic.objects, candidate_key_by_object_id)
    descriptors = {
        obj.object_id: _descriptor_from_points(
            candidate_surface.mesh.vertices_reconstruction[obj.vertex_indices.astype(np.int64)]
        )
        for obj in candidate_semantic.objects
    }

    registration = _register(
        reference,
        candidate_semantic,
        candidate_surface,
        descriptors,
        candidate_key_map,
        resolved_policy,
        candidate_scale_to_reference_units,
        comparison_metric_available,
    )
    if registration.status is not RegistrationStatus.SUCCESS:
        feature_results = tuple(
            InspectionFeatureResult(
                reference_feature_id=feature.feature_id,
                candidate_object_id="",
                semantic_role=feature.semantic_role,
                decision=InspectionDecision.INSUFFICIENT_EVIDENCE,
                findings=(InspectionFindingKind.REGISTRATION_INSUFFICIENT_EVIDENCE,),
                connector_defects=(),
                position_error_reference_units=None,
                position_error_m=None,
                orientation_error_degrees=None,
                extent_relative_error=None,
                geometry_quality=None,
                uncertainty_reference_units=None,
                uncertainty_m=None,
            )
            for feature in reference.features
        )
        stats = InspectionStatistics(
            reference_feature_count=len(reference.features),
            matched_feature_count=0,
            missing_required_count=0,
            extra_candidate_count=len(candidate_semantic.objects),
            pass_count=0,
            warn_count=0,
            reject_count=0,
            insufficient_evidence_count=len(reference.features),
        )
        return InspectionReport(
            session_id=candidate_semantic.session_id,
            reference_id=reference.reference_id,
            scene_profile=candidate_semantic.scene_profile,
            pose_convention=candidate_semantic.pose_convention,
            candidate_frame_id=candidate_semantic.reconstruction_frame_id,
            reference_frame_id=reference.reference_frame_id,
            candidate_metric_geometry_status=candidate_semantic.metric_geometry_status,
            reference_metric_geometry_status=reference.metric_geometry_status,
            comparison_metric_available=comparison_metric_available,
            reference_unit_to_metre=reference.unit_to_metre if comparison_metric_available else None,
            relative_scale_uncertainty=_combined_scale_uncertainty(reference, candidate_semantic),
            registration=registration,
            feature_results=feature_results,
            extra_candidate_object_ids=tuple(obj.object_id for obj in candidate_semantic.objects),
            decision=InspectionDecision.INSUFFICIENT_EVIDENCE,
            statistics=stats,
            certified_metrology=False,
        )

    matches, extras = _match_features(
        reference,
        candidate_semantic,
        descriptors,
        candidate_key_map,
        registration,
        resolved_policy,
    )
    by_object_id = {obj.object_id: obj for obj in candidate_semantic.objects}
    results: list[InspectionFeatureResult] = []
    missing_required = 0
    for feature in reference.features:
        candidate_id = matches.get(feature.feature_id)
        if candidate_id is None:
            decision = InspectionDecision.REJECT if feature.required else InspectionDecision.WARN
            if feature.required:
                missing_required += 1
            defects = (
                ("missing_pin",)
                if feature.semantic_role is SemanticGeometryRole.CONNECTOR_PIN
                else ()
            )
            results.append(
                InspectionFeatureResult(
                    reference_feature_id=feature.feature_id,
                    candidate_object_id="",
                    semantic_role=feature.semantic_role,
                    decision=decision,
                    findings=(InspectionFindingKind.MISSING_FEATURE,),
                    connector_defects=defects,
                    position_error_reference_units=None,
                    position_error_m=None,
                    orientation_error_degrees=None,
                    extent_relative_error=None,
                    geometry_quality=None,
                    uncertainty_reference_units=None,
                    uncertainty_m=None,
                )
            )
            continue
        obj = by_object_id[candidate_id]
        descriptor = descriptors[candidate_id]
        results.append(
            _inspect_feature(
                feature,
                obj,
                descriptor,
                registration,
                reference,
                comparison_metric_available,
            )
        )

    decisions = [result.decision for result in results]
    if extras:
        decisions.extend([resolved_policy.extra_feature_decision] * len(extras))
    overall = _aggregate_decision(decisions)
    stats = InspectionStatistics(
        reference_feature_count=len(reference.features),
        matched_feature_count=sum(1 for result in results if result.candidate_object_id),
        missing_required_count=missing_required,
        extra_candidate_count=len(extras),
        pass_count=sum(result.decision is InspectionDecision.PASS for result in results),
        warn_count=sum(result.decision is InspectionDecision.WARN for result in results)
        + (len(extras) if resolved_policy.extra_feature_decision is InspectionDecision.WARN else 0),
        reject_count=sum(result.decision is InspectionDecision.REJECT for result in results)
        + (len(extras) if resolved_policy.extra_feature_decision is InspectionDecision.REJECT else 0),
        insufficient_evidence_count=sum(
            result.decision is InspectionDecision.INSUFFICIENT_EVIDENCE for result in results
        ),
    )
    return InspectionReport(
        session_id=candidate_semantic.session_id,
        reference_id=reference.reference_id,
        scene_profile=candidate_semantic.scene_profile,
        pose_convention=candidate_semantic.pose_convention,
        candidate_frame_id=candidate_semantic.reconstruction_frame_id,
        reference_frame_id=reference.reference_frame_id,
        candidate_metric_geometry_status=candidate_semantic.metric_geometry_status,
        reference_metric_geometry_status=reference.metric_geometry_status,
        comparison_metric_available=comparison_metric_available,
        reference_unit_to_metre=reference.unit_to_metre if comparison_metric_available else None,
        relative_scale_uncertainty=_combined_scale_uncertainty(reference, candidate_semantic),
        registration=registration,
        feature_results=tuple(results),
        extra_candidate_object_ids=tuple(extras),
        decision=overall,
        statistics=stats,
        certified_metrology=False,
    )


def write_reference_assembly_json(reference: ReferenceAssembly, path: str | Path) -> Path:
    if not isinstance(reference, ReferenceAssembly):
        raise TypeError("reference must be ReferenceAssembly.")
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(reference.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def write_inspection_report_json(report: InspectionReport, path: str | Path) -> Path:
    if not isinstance(report, InspectionReport):
        raise TypeError("report must be InspectionReport.")
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


@dataclass(frozen=True, slots=True)
class _Descriptor:
    centroid: np.ndarray
    axes: np.ndarray
    extents: np.ndarray
    orientation_reliable: bool



def _descriptor_from_points(points: np.ndarray) -> _Descriptor:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1:] != (3,) or len(values) < 1 or not np.isfinite(values).all():
        raise ReconstructionValidationError("Semantic feature points must be finite Nx3.")
    centroid = np.mean(values, axis=0)
    if len(values) < 3:
        axes = np.eye(3, dtype=np.float64)
        extents = np.ptp(values, axis=0)
        reliable = False
    else:
        centered = values - centroid
        covariance = centered.T @ centered / max(len(values) - 1, 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 0.0)
        axes = eigenvectors[:, order]
        for column in range(3):
            pivot = int(np.argmax(np.abs(axes[:, column])))
            if axes[pivot, column] < 0.0:
                axes[:, column] *= -1.0
        if np.linalg.det(axes) < 0.0:
            axes[:, 2] *= -1.0
        local = centered @ axes
        extents = np.ptp(local, axis=0)
        epsilon = max(float(eigenvalues[0]) * 1e-9, 1e-15)
        first_distinct = eigenvalues[0] / max(eigenvalues[1], epsilon) >= 1.08
        second_distinct = eigenvalues[1] / max(eigenvalues[2], epsilon) >= 1.25
        reliable = bool(eigenvalues[0] > epsilon and first_distinct and second_distinct)
    return _Descriptor(
        centroid=_readonly(centroid, np.float64),
        axes=_readonly(axes, np.float64),
        extents=_readonly(np.maximum(extents, 0.0), np.float64),
        orientation_reliable=reliable,
    )


def _register(
    reference: ReferenceAssembly,
    candidate_semantic: SemanticGeometryResult,
    candidate_surface: SurfaceMeshResult,
    descriptors: Mapping[str, _Descriptor],
    candidate_key_map: Mapping[str, str],
    policy: RegistrationInspectionPolicy,
    candidate_scale_to_reference_units: float | None,
    metric_comparison: bool,
) -> RegistrationResult:
    feature_by_key = {feature.association_key: feature for feature in reference.features}
    candidate_by_key = {
        key: obj for obj in candidate_semantic.objects if (key := candidate_key_map.get(obj.object_id))
    }
    anchors = [feature for feature in reference.features if feature.registration_anchor]
    direct: list[tuple[ReferenceFeature, Semantic3DObject]] = []
    for feature in anchors:
        obj = candidate_by_key.get(feature.association_key)
        if obj is not None and _semantic_compatible(feature, obj):
            direct.append((feature, obj))
    if not direct:
        return _failed_registration(
            RegistrationStatus.INSUFFICIENT_EVIDENCE,
            "No registration anchor correspondence is available.",
        )
    if any(obj.mean_vertex_quality < policy.minimum_registration_anchor_quality for _, obj in direct):
        return _failed_registration(
            RegistrationStatus.INSUFFICIENT_EVIDENCE,
            "Registration anchor geometry quality is below policy threshold.",
        )

    oriented = [pair for pair in direct if pair[0].orientation_reliable and descriptors[pair[1].object_id].orientation_reliable]
    if oriented:
        feature, obj = oriented[0]
        descriptor = descriptors[obj.object_id]
        candidate_points = candidate_surface.mesh.vertices_reconstruction[obj.vertex_indices.astype(np.int64)]
        try:
            scale = _registration_scale(feature, descriptor, candidate_scale_to_reference_units)
        except ReconstructionValidationError as exc:
            return _failed_registration(RegistrationStatus.DEGENERATE, str(exc))
        rotation, translation, rms, maximum = _oriented_anchor_alignment(
            feature,
            candidate_points,
            descriptor,
            scale,
        )
        method = RegistrationMethod.ORIENTED_SEMANTIC_ANCHOR
        anchor_ids = (feature.feature_id,)
        characteristic = max(feature.characteristic_extent, 1e-12)
    elif len(direct) >= 3:
        ref_points = np.asarray([feature.centroid_reference for feature, _ in direct], dtype=np.float64)
        cand_points = np.asarray([descriptors[obj.object_id].centroid for _, obj in direct], dtype=np.float64)
        try:
            scale, rotation, translation = _similarity_from_correspondences(
                cand_points,
                ref_points,
                candidate_scale_to_reference_units,
            )
        except ReconstructionValidationError as exc:
            return _failed_registration(RegistrationStatus.DEGENERATE, str(exc))
        transformed = _apply_similarity(cand_points, scale, rotation, translation)
        residuals = np.linalg.norm(transformed - ref_points, axis=1)
        rms = float(np.sqrt(np.mean(np.square(residuals))))
        maximum = float(np.max(residuals))
        method = RegistrationMethod.MULTI_ANCHOR_SIMILARITY
        anchor_ids = tuple(feature.feature_id for feature, _ in direct)
        characteristic = max(
            float(np.median([feature.characteristic_extent for feature, _ in direct])),
            1e-12,
        )
    else:
        return _failed_registration(
            RegistrationStatus.INSUFFICIENT_EVIDENCE,
            "Registration requires one reliable oriented anchor or at least three anchor centroids.",
        )

    normalized = rms / characteristic
    if not isfinite(normalized) or normalized > policy.maximum_registration_rms_fraction:
        return RegistrationResult(
            status=RegistrationStatus.RESIDUAL_TOO_HIGH,
            method=None,
            scale_candidate_to_reference=None,
            T_reference_from_scaled_candidate=None,
            anchor_feature_ids=anchor_ids,
            rms_residual_reference_units=rms,
            maximum_residual_reference_units=maximum,
            rms_residual_m=(rms * reference.unit_to_metre if metric_comparison else None),
            maximum_residual_m=(maximum * reference.unit_to_metre if metric_comparison else None),
            normalized_rms_fraction=normalized,
            confidence=float(np.clip(1.0 - normalized, 0.0, 1.0)),
            reason="Registration residual exceeds policy threshold.",
        )
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    confidence = float(np.clip(1.0 - normalized / max(policy.maximum_registration_rms_fraction, 1e-12), 0.0, 1.0))
    return RegistrationResult(
        status=RegistrationStatus.SUCCESS,
        method=method,
        scale_candidate_to_reference=scale,
        T_reference_from_scaled_candidate=matrix,
        anchor_feature_ids=anchor_ids,
        rms_residual_reference_units=rms,
        maximum_residual_reference_units=maximum,
        rms_residual_m=(rms * reference.unit_to_metre if metric_comparison else None),
        maximum_residual_m=(maximum * reference.unit_to_metre if metric_comparison else None),
        normalized_rms_fraction=normalized,
        confidence=confidence,
        reason="",
    )


def _oriented_anchor_alignment(
    feature: ReferenceFeature,
    candidate_points: np.ndarray,
    candidate_descriptor: _Descriptor,
    scale: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    reference_points = feature.registration_points_reference
    if len(reference_points) < 3 or len(candidate_points) < 3:
        raise ReconstructionValidationError("Oriented anchor registration requires point support.")
    candidate_sample = _deterministic_subsample(candidate_points, len(reference_points))
    reference_axes = feature.principal_axes_reference
    candidate_axes = candidate_descriptor.axes
    sign_sets = (
        np.diag([1.0, 1.0, 1.0]),
        np.diag([1.0, -1.0, -1.0]),
        np.diag([-1.0, 1.0, -1.0]),
        np.diag([-1.0, -1.0, 1.0]),
    )
    best: tuple[float, np.ndarray, np.ndarray, float] | None = None
    ref_tree = cKDTree(reference_points)
    for signs in sign_sets:
        rotation = reference_axes @ signs @ candidate_axes.T
        if np.linalg.det(rotation) < 0.999:
            continue
        translation = feature.centroid_reference - rotation @ (scale * candidate_descriptor.centroid)
        transformed = _apply_similarity(candidate_sample, scale, rotation, translation)
        distances, _ = ref_tree.query(transformed, k=1, workers=1)
        rms = float(np.sqrt(np.mean(np.square(distances))))
        maximum = float(np.max(distances))
        candidate = (rms, rotation, translation, maximum)
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is None:
        raise ReconstructionValidationError("No right-handed oriented-anchor alignment was found.")
    return best[1], best[2], best[0], best[3]


def _registration_scale(
    feature: ReferenceFeature,
    candidate_descriptor: _Descriptor,
    metric_scale: float | None,
) -> float:
    if metric_scale is not None:
        return float(metric_scale)
    ref = feature.oriented_extents_reference
    cand = candidate_descriptor.extents
    valid = (ref > 1e-9) & (cand > 1e-9)
    ratios = ref[valid] / cand[valid]
    if len(ratios) < 1:
        raise ReconstructionValidationError("Cannot estimate relative registration scale from anchor extent.")
    scale = float(np.median(ratios))
    spread = float(np.median(np.abs(ratios - scale)) / max(scale, 1e-12))
    if not isfinite(scale) or scale <= 0.0 or spread > 0.35:
        raise ReconstructionValidationError("Anchor extent does not support a stable relative scale estimate.")
    return scale


def _similarity_from_correspondences(
    candidate: np.ndarray,
    reference: np.ndarray,
    fixed_scale: float | None,
) -> tuple[float, np.ndarray, np.ndarray]:
    if candidate.shape != reference.shape or candidate.ndim != 2 or candidate.shape[1:] != (3,) or len(candidate) < 3:
        raise ReconstructionValidationError("Similarity registration requires matching Nx3 anchors.")
    cand_mean = np.mean(candidate, axis=0)
    ref_mean = np.mean(reference, axis=0)
    cand_centered = candidate - cand_mean
    ref_centered = reference - ref_mean
    if np.linalg.matrix_rank(cand_centered) < 2 or np.linalg.matrix_rank(ref_centered) < 2:
        raise ReconstructionValidationError("Anchor centroids are geometrically degenerate.")
    if fixed_scale is None:
        variance = float(np.mean(np.sum(cand_centered * cand_centered, axis=1)))
        if variance <= 1e-15:
            raise ReconstructionValidationError("Candidate anchor scale is degenerate.")
        covariance = (ref_centered.T @ cand_centered) / len(candidate)
        u, singular, vt = np.linalg.svd(covariance)
        signs = np.ones(3)
        if np.linalg.det(u @ vt) < 0.0:
            signs[-1] = -1.0
        rotation = u @ np.diag(signs) @ vt
        scale = float(np.sum(singular * signs) / variance)
    else:
        scale = float(fixed_scale)
        covariance = ref_centered.T @ (candidate * scale - cand_mean * scale)
        u, _, vt = np.linalg.svd(covariance)
        signs = np.ones(3)
        if np.linalg.det(u @ vt) < 0.0:
            signs[-1] = -1.0
        rotation = u @ np.diag(signs) @ vt
    if not isfinite(scale) or scale <= 0.0:
        raise ReconstructionValidationError("Similarity registration produced an invalid scale.")
    translation = ref_mean - rotation @ (scale * cand_mean)
    return scale, rotation, translation


def _match_features(
    reference: ReferenceAssembly,
    candidate_semantic: SemanticGeometryResult,
    descriptors: Mapping[str, _Descriptor],
    candidate_key_map: Mapping[str, str],
    registration: RegistrationResult,
    policy: RegistrationInspectionPolicy,
) -> tuple[dict[str, str], tuple[str, ...]]:
    if registration.status is not RegistrationStatus.SUCCESS:
        return {}, tuple(obj.object_id for obj in candidate_semantic.objects)
    reference_by_key = {feature.association_key: feature for feature in reference.features}
    candidate_by_key = {
        key: obj for obj in candidate_semantic.objects if (key := candidate_key_map.get(obj.object_id))
    }
    matches: dict[str, str] = {}
    used: set[str] = set()
    for key, feature in reference_by_key.items():
        obj = candidate_by_key.get(key)
        if obj is not None and _semantic_compatible(feature, obj):
            matches[feature.feature_id] = obj.object_id
            used.add(obj.object_id)

    unmatched_features = [feature for feature in reference.features if feature.feature_id not in matches]
    unmatched_objects = [obj for obj in candidate_semantic.objects if obj.object_id not in used]
    for role in SemanticGeometryRole:
        role_features = [feature for feature in unmatched_features if feature.semantic_role is role]
        role_objects = [obj for obj in unmatched_objects if obj.semantic_role is role]
        if not role_features or not role_objects:
            continue
        cost = np.full((len(role_features), len(role_objects)), 1e9, dtype=np.float64)
        for i, feature in enumerate(role_features):
            max_distance = policy.maximum_auto_association_distance_fraction * max(
                feature.characteristic_extent, 1e-12
            )
            for j, obj in enumerate(role_objects):
                if not _semantic_compatible(feature, obj):
                    continue
                transformed = _transform_centroid(
                    descriptors[obj.object_id].centroid,
                    registration,
                )
                distance = float(np.linalg.norm(transformed - feature.centroid_reference))
                if distance <= max_distance:
                    cost[i, j] = distance
        rows, cols = linear_sum_assignment(cost)
        for i, j in zip(rows, cols, strict=True):
            if cost[i, j] >= 1e8:
                continue
            feature = role_features[int(i)]
            obj = role_objects[int(j)]
            matches[feature.feature_id] = obj.object_id
            used.add(obj.object_id)
    extras = tuple(obj.object_id for obj in candidate_semantic.objects if obj.object_id not in used)
    return matches, extras


def _inspect_feature(
    feature: ReferenceFeature,
    obj: Semantic3DObject,
    descriptor: _Descriptor,
    registration: RegistrationResult,
    reference: ReferenceAssembly,
    metric_comparison: bool,
) -> InspectionFeatureResult:
    transformed_centroid = _transform_centroid(descriptor.centroid, registration)
    position_error = float(np.linalg.norm(transformed_centroid - feature.centroid_reference))
    rotation = registration.T_reference_from_scaled_candidate[:3, :3]  # type: ignore[index]
    transformed_axes = rotation @ descriptor.axes
    orientation_error = _axis_angle_degrees(
        feature.principal_axes_reference[:, 0],
        transformed_axes[:, 0],
    ) if feature.orientation_reliable and descriptor.orientation_reliable else None
    candidate_extents = descriptor.extents * float(registration.scale_candidate_to_reference)
    valid = feature.oriented_extents_reference > 1e-9
    if np.any(valid):
        extent_relative = float(
            np.max(
                np.abs(candidate_extents[valid] - feature.oriented_extents_reference[valid])
                / feature.oriented_extents_reference[valid]
            )
        )
    else:
        extent_relative = 0.0
    characteristic = max(feature.characteristic_extent, 1e-12)
    uncertainty_reference = (
        obj.median_vertex_uncertainty_reconstruction_units
        * float(registration.scale_candidate_to_reference)
    )
    uncertainty_fraction = uncertainty_reference / characteristic

    findings: list[InspectionFindingKind] = []
    decision = InspectionDecision.PASS
    if obj.mean_vertex_quality < feature.tolerance.minimum_geometry_quality:
        findings.append(InspectionFindingKind.LOW_GEOMETRY_QUALITY)
        decision = InspectionDecision.INSUFFICIENT_EVIDENCE
    if uncertainty_fraction > feature.tolerance.maximum_uncertainty_fraction:
        findings.append(InspectionFindingKind.HIGH_GEOMETRY_UNCERTAINTY)
        decision = InspectionDecision.INSUFFICIENT_EVIDENCE

    if decision is not InspectionDecision.INSUFFICIENT_EVIDENCE:
        if position_error >= feature.tolerance.position_reject:
            findings.append(InspectionFindingKind.POSITION_DEVIATION)
            decision = InspectionDecision.REJECT
        elif position_error >= feature.tolerance.position_warn:
            findings.append(InspectionFindingKind.POSITION_DEVIATION)
            decision = InspectionDecision.WARN
        if orientation_error is not None:
            if orientation_error >= feature.tolerance.orientation_reject_degrees:
                findings.append(InspectionFindingKind.ORIENTATION_DEVIATION)
                decision = InspectionDecision.REJECT
            elif orientation_error >= feature.tolerance.orientation_warn_degrees:
                findings.append(InspectionFindingKind.ORIENTATION_DEVIATION)
                if decision is InspectionDecision.PASS:
                    decision = InspectionDecision.WARN
        if extent_relative >= feature.tolerance.extent_relative_reject:
            findings.append(InspectionFindingKind.EXTENT_DEVIATION)
            decision = InspectionDecision.REJECT
        elif extent_relative >= feature.tolerance.extent_relative_warn:
            findings.append(InspectionFindingKind.EXTENT_DEVIATION)
            if decision is InspectionDecision.PASS:
                decision = InspectionDecision.WARN
    if not findings:
        findings.append(InspectionFindingKind.WITHIN_TOLERANCE)

    defects: list[str] = []
    if feature.semantic_role is SemanticGeometryRole.CONNECTOR_PIN:
        if InspectionFindingKind.POSITION_DEVIATION in findings:
            defects.append("misaligned_pin")
        if InspectionFindingKind.ORIENTATION_DEVIATION in findings:
            defects.append("bent_pin")
        if InspectionFindingKind.EXTENT_DEVIATION in findings:
            defects.append("pin_geometry_deviation")

    factor = reference.unit_to_metre if metric_comparison else None
    return InspectionFeatureResult(
        reference_feature_id=feature.feature_id,
        candidate_object_id=obj.object_id,
        semantic_role=feature.semantic_role,
        decision=decision,
        findings=tuple(findings),
        connector_defects=tuple(defects),
        position_error_reference_units=position_error,
        position_error_m=(position_error * factor if factor is not None else None),
        orientation_error_degrees=orientation_error,
        extent_relative_error=extent_relative,
        geometry_quality=obj.mean_vertex_quality,
        uncertainty_reference_units=uncertainty_reference,
        uncertainty_m=(uncertainty_reference * factor if factor is not None else None),
    )


def _scale_compatibility(
    reference: ReferenceAssembly,
    candidate: SemanticGeometryResult,
) -> tuple[bool, float | None]:
    ref_metric = reference.metric_geometry_status is not MetricGeometryStatus.ARBITRARY
    cand_metric = candidate.metric_geometry_status is not MetricGeometryStatus.ARBITRARY
    if ref_metric != cand_metric:
        raise ReconstructionValidationError(
            "Mixed arbitrary/metric inspection requires an explicit Phase 8.6B scale alignment before 8.7C."
        )
    if not ref_metric:
        return False, None
    if reference.unit_to_metre is None or candidate.unit_to_metre is None:
        raise ReconstructionValidationError("Metric inspection requires explicit unit_to_metre values.")
    return True, float(candidate.unit_to_metre) / float(reference.unit_to_metre)


def _combined_scale_uncertainty(
    reference: ReferenceAssembly,
    candidate: SemanticGeometryResult,
) -> float | None:
    values = [
        value
        for value in (reference.relative_scale_uncertainty, candidate.relative_scale_uncertainty)
        if value is not None
    ]
    if not values:
        return None
    return float(np.sqrt(np.sum(np.square(values))))


def _candidate_key_map(
    objects: tuple[Semantic3DObject, ...],
    explicit: Mapping[str, str] | None,
) -> dict[str, str]:
    overrides = dict(explicit or {})
    result: dict[str, str] = {}
    seen: set[str] = set()
    for obj in objects:
        key = str(overrides.get(obj.object_id, _default_feature_key(obj))).strip()
        if not key:
            continue
        if key in seen:
            raise ReconstructionValidationError(
                f"Candidate association key {key!r} resolves to multiple objects."
            )
        seen.add(key)
        result[obj.object_id] = key
    return result


def _default_feature_key(obj: Semantic3DObject) -> str:
    if len(obj.track_ids) == 1:
        return obj.track_ids[0]
    return obj.object_id


def _semantic_compatible(feature: ReferenceFeature, obj: Semantic3DObject) -> bool:
    return (
        feature.class_id == obj.class_id
        and feature.class_name == obj.class_name
        and feature.semantic_role is obj.semantic_role
    )


def _transform_centroid(point: np.ndarray, registration: RegistrationResult) -> np.ndarray:
    scale = float(registration.scale_candidate_to_reference)
    matrix = registration.T_reference_from_scaled_candidate
    assert matrix is not None
    return matrix[:3, :3] @ (scale * np.asarray(point, dtype=np.float64)) + matrix[:3, 3]


def _apply_similarity(
    points: np.ndarray,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    return (scale * np.asarray(points, dtype=np.float64)) @ rotation.T + translation


def _axis_angle_degrees(reference_axis: np.ndarray, candidate_axis: np.ndarray) -> float:
    a = np.asarray(reference_axis, dtype=np.float64).copy()
    b = np.asarray(candidate_axis, dtype=np.float64).copy()
    a /= max(float(np.linalg.norm(a)), 1e-15)
    b /= max(float(np.linalg.norm(b)), 1e-15)
    cosine = float(np.clip(abs(np.dot(a, b)), -1.0, 1.0))
    return float(degrees(acos(cosine)))


def _aggregate_decision(decisions: Iterable[InspectionDecision]) -> InspectionDecision:
    values = tuple(decisions)
    if any(value is InspectionDecision.REJECT for value in values):
        return InspectionDecision.REJECT
    if any(value is InspectionDecision.INSUFFICIENT_EVIDENCE for value in values):
        return InspectionDecision.INSUFFICIENT_EVIDENCE
    if any(value is InspectionDecision.WARN for value in values):
        return InspectionDecision.WARN
    return InspectionDecision.PASS


def _failed_registration(status: RegistrationStatus, reason: str) -> RegistrationResult:
    return RegistrationResult(
        status=status,
        method=None,
        scale_candidate_to_reference=None,
        T_reference_from_scaled_candidate=None,
        anchor_feature_ids=(),
        rms_residual_reference_units=None,
        maximum_residual_reference_units=None,
        rms_residual_m=None,
        maximum_residual_m=None,
        normalized_rms_fraction=None,
        confidence=0.0,
        reason=reason,
    )


def _validate_surface_semantic_pair(
    surface: SurfaceMeshResult,
    semantic: SemanticGeometryResult,
) -> None:
    if not isinstance(surface, SurfaceMeshResult):
        raise TypeError("surface must be SurfaceMeshResult.")
    if not isinstance(semantic, SemanticGeometryResult):
        raise TypeError("semantic must be SemanticGeometryResult.")
    if surface.session_id != semantic.session_id:
        raise ReconstructionValidationError("Surface and semantic result belong to different sessions.")
    if surface.scene_profile is not semantic.scene_profile:
        raise ReconstructionValidationError("Surface and semantic result scene profiles differ.")
    if surface.reconstruction_frame_id != semantic.reconstruction_frame_id:
        raise ReconstructionValidationError("Surface and semantic reconstruction frames differ.")
    if surface.metric_geometry_status is not semantic.metric_geometry_status:
        raise ReconstructionValidationError("Surface and semantic metric statuses differ.")
    if surface.unit_to_metre != semantic.unit_to_metre:
        raise ReconstructionValidationError("Surface and semantic unit_to_metre differ.")


def _deterministic_subsample(points: np.ndarray, maximum: int) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if len(values) <= maximum:
        return np.ascontiguousarray(values)
    positions = np.linspace(0, len(values) - 1, num=maximum, dtype=np.int64)
    return np.ascontiguousarray(values[positions])


def _positive_ordered(warn: float, reject: float, prefix: str) -> None:
    warn_value = float(warn)
    reject_value = float(reject)
    if not isfinite(warn_value) or not isfinite(reject_value) or warn_value <= 0.0 or reject_value <= warn_value:
        raise ReconstructionValidationError(
            f"{prefix} warn/reject thresholds must be finite, positive, and reject > warn."
        )


def _probability(value: object, field_name: str) -> float:
    result = float(value)
    if not isfinite(result) or not 0.0 <= result <= 1.0:
        raise ReconstructionValidationError(f"{field_name} must be finite in [0, 1].")
    return result


def _vector3(value: object, field_name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ReconstructionValidationError(f"{field_name} must be a finite 3-vector.")
    return array


def _nonnegative_vector3(value: object, field_name: str) -> np.ndarray:
    array = _vector3(value, field_name)
    if np.any(array < 0.0):
        raise ReconstructionValidationError(f"{field_name} must be non-negative.")
    return array


def _axes(value: object, field_name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3, 3) or not np.isfinite(array).all():
        raise ReconstructionValidationError(f"{field_name} must be finite 3x3.")
    if not np.allclose(array.T @ array, np.eye(3), atol=1e-5) or np.linalg.det(array) < 0.99:
        raise ReconstructionValidationError(f"{field_name} must be right-handed orthonormal.")
    return array


def _readonly(value: object, dtype: np.dtype[Any]) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    array.setflags(write=False)
    return array

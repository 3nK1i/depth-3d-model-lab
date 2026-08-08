"""Persistent semantic 3D geometry for Phase 8.7B.

Phase 8.7B projects authoritative Phase 7.5 perception masks onto the trusted
Phase 8.7A surface mesh and associates repeated observations into persistent
semantic 3D objects.  The layer is intentionally geometry-first and
provenance-preserving:

* one 2D mask is never promoted directly to a persistent object without an
  explicit association stage,
* track IDs are authoritative only within a class/role and geometry-overlap
  association is cross-view only,
* mesh vertices and faces are referenced, never deformed or duplicated,
* arbitrary/provisional/metric scale semantics are inherited without
  promotion, and
* no inspection/QC or certified-metrology decision is made here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from pathlib import Path
from typing import Any, Iterable, Mapping

import cv2
import numpy as np

from core.contracts.validation import require_nonempty
from core.perception.result import (
    ObjectObservation,
    UnsupportedMaskRepresentationError,
    normalize_mask_payload,
)
from core.reconstruction.camera import CameraDistortionModel
from core.reconstruction.camera_geometry import CameraGeometryProfile
from core.reconstruction.dense_fusion import DenseSceneProfile
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.scale_reference import MetricGeometryStatus, ScaleReferenceProfile
from core.reconstruction.sfm_coordinates import validate_rigid_matrix
from core.reconstruction.surface_mesh import SurfaceMeshResult

SEMANTIC_GEOMETRY_SCHEMA_VERSION = 1
SEMANTIC_GEOMETRY_POSE_CONVENTION = "T_reconstruction_from_camera"


class SemanticGeometryRole(str, Enum):
    """Explicit semantic role; never inferred from a class name implicitly."""

    GENERIC_OBJECT = "generic_object"
    DIGITAL_TWIN_ENTITY = "digital_twin_entity"
    CONNECTOR_BODY = "connector_body"
    CONNECTOR_PIN = "connector_pin"
    CONNECTOR_PIN_ROW = "connector_pin_row"
    CONNECTOR_SLOT = "connector_slot"
    CONNECTOR_HOUSING = "connector_housing"
    REFERENCE_SURFACE = "reference_surface"


class SemanticProjectionRejection(str, Enum):
    NONE = "none"
    NO_AUTHORITATIVE_MASK = "no_authoritative_mask"
    FRAME_NOT_IN_SURFACE_PROVENANCE = "frame_not_in_surface_provenance"
    CAMERA_IMAGE_SIZE_MISMATCH = "camera_image_size_mismatch"
    MASK_PIXEL_SPACE_MISMATCH = "mask_pixel_space_mismatch"
    UNSUPPORTED_MASK_REPRESENTATION = "unsupported_mask_representation"
    NO_VISIBLE_VERTICES = "no_visible_vertices"
    INSUFFICIENT_VERTEX_SUPPORT = "insufficient_vertex_support"


@dataclass(frozen=True, slots=True)
class SemanticAssociationPolicy:
    """Bounded controls for 2D-to-3D projection and cross-view association."""

    minimum_projected_vertices: int = 6
    minimum_shared_vertices: int = 4
    minimum_cross_view_vertex_iou: float = 0.12
    face_minimum_assigned_vertices: int = 3
    maximum_observations: int = 10_000
    maximum_objects: int = 4_096
    minimum_camera_depth: float = 1e-6

    def __post_init__(self) -> None:
        for name, minimum in (
            ("minimum_projected_vertices", 1),
            ("minimum_shared_vertices", 1),
            ("face_minimum_assigned_vertices", 1),
            ("maximum_observations", 1),
            ("maximum_objects", 1),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) < minimum:
                raise ReconstructionValidationError(f"{name} must be >= {minimum}.")
            object.__setattr__(self, name, int(value))
        if self.face_minimum_assigned_vertices > 3:
            raise ReconstructionValidationError("face_minimum_assigned_vertices must be <= 3.")
        iou = float(self.minimum_cross_view_vertex_iou)
        if not isfinite(iou) or not 0.0 <= iou <= 1.0:
            raise ReconstructionValidationError(
                "minimum_cross_view_vertex_iou must be finite in [0, 1]."
            )
        depth = float(self.minimum_camera_depth)
        if not isfinite(depth) or depth <= 0.0:
            raise ReconstructionValidationError("minimum_camera_depth must be finite and positive.")
        object.__setattr__(self, "minimum_cross_view_vertex_iou", iou)
        object.__setattr__(self, "minimum_camera_depth", depth)

    def as_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class SemanticFrameObservation:
    """One perception observation with the camera pose used for mesh projection."""

    frame_id: str
    observation: ObjectObservation
    camera_profile: CameraGeometryProfile
    T_reconstruction_from_camera: np.ndarray
    pose_layer: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_id", require_nonempty(self.frame_id, field_name="frame_id"))
        if not isinstance(self.observation, ObjectObservation):
            raise TypeError("observation must be ObjectObservation.")
        if not isinstance(self.camera_profile, CameraGeometryProfile):
            raise TypeError("camera_profile must be CameraGeometryProfile.")
        object.__setattr__(
            self,
            "T_reconstruction_from_camera",
            validate_rigid_matrix(
                self.T_reconstruction_from_camera,
                field_name="T_reconstruction_from_camera",
            ),
        )
        object.__setattr__(
            self, "pose_layer", require_nonempty(self.pose_layer, field_name="pose_layer")
        )


@dataclass(frozen=True, slots=True)
class MeshProjection:
    """Projected mesh vertices for one camera pose."""

    pixel_xy: np.ndarray
    visible_vertex_indices: np.ndarray

    def __post_init__(self) -> None:
        pixel_xy = np.asarray(self.pixel_xy, dtype=np.float64)
        indices = np.asarray(self.visible_vertex_indices, dtype=np.uint32)
        if pixel_xy.ndim != 2 or pixel_xy.shape[1:] != (2,):
            raise ReconstructionValidationError("pixel_xy must be Nx2.")
        if indices.shape != (len(pixel_xy),):
            raise ReconstructionValidationError(
                "visible_vertex_indices must contain one index per projected pixel."
            )
        if not np.isfinite(pixel_xy).all():
            raise ReconstructionValidationError("pixel_xy must be finite.")
        object.__setattr__(self, "pixel_xy", _readonly(pixel_xy, np.float64))
        object.__setattr__(self, "visible_vertex_indices", _readonly(indices, np.uint32))


@dataclass(frozen=True, slots=True)
class SemanticObservationProjection:
    observation_id: str
    frame_id: str
    class_id: int
    class_name: str
    semantic_role: SemanticGeometryRole
    track_id: str
    mask_id: str
    camera_profile_id: str
    pose_layer: str
    accepted: bool
    rejection: SemanticProjectionRejection
    visible_vertex_count: int
    vertex_indices: np.ndarray
    face_indices: np.ndarray
    source_confidence: float
    mean_vertex_quality: float
    median_vertex_uncertainty_reconstruction_units: float | None
    projection_confidence: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "observation_id", require_nonempty(self.observation_id, field_name="observation_id")
        )
        object.__setattr__(self, "frame_id", require_nonempty(self.frame_id, field_name="frame_id"))
        if isinstance(self.class_id, bool) or int(self.class_id) < 0:
            raise ReconstructionValidationError("class_id must be a non-negative integer.")
        object.__setattr__(self, "class_id", int(self.class_id))
        object.__setattr__(
            self, "class_name", require_nonempty(self.class_name, field_name="class_name")
        )
        if not isinstance(self.semantic_role, SemanticGeometryRole):
            raise TypeError("semantic_role must be SemanticGeometryRole.")
        object.__setattr__(self, "track_id", str(self.track_id).strip())
        object.__setattr__(self, "mask_id", str(self.mask_id).strip())
        object.__setattr__(
            self,
            "camera_profile_id",
            require_nonempty(self.camera_profile_id, field_name="camera_profile_id"),
        )
        object.__setattr__(
            self, "pose_layer", require_nonempty(self.pose_layer, field_name="pose_layer")
        )
        if not isinstance(self.rejection, SemanticProjectionRejection):
            raise TypeError("rejection must be SemanticProjectionRejection.")
        if isinstance(self.visible_vertex_count, bool) or int(self.visible_vertex_count) < 0:
            raise ReconstructionValidationError("visible_vertex_count must be non-negative.")
        object.__setattr__(self, "visible_vertex_count", int(self.visible_vertex_count))
        vertices = np.asarray(self.vertex_indices, dtype=np.uint32)
        faces = np.asarray(self.face_indices, dtype=np.uint32)
        if vertices.ndim != 1 or faces.ndim != 1:
            raise ReconstructionValidationError("vertex_indices and face_indices must be 1D.")
        if len(vertices) != len(np.unique(vertices)) or len(faces) != len(np.unique(faces)):
            raise ReconstructionValidationError("Projection indices must be unique.")
        if bool(self.accepted):
            if self.rejection is not SemanticProjectionRejection.NONE:
                raise ReconstructionValidationError("Accepted projections cannot contain a rejection.")
            if len(vertices) == 0 or not self.mask_id:
                raise ReconstructionValidationError("Accepted projections require mask-backed vertices.")
        else:
            if self.rejection is SemanticProjectionRejection.NONE:
                raise ReconstructionValidationError("Rejected projections require a rejection reason.")
            if len(vertices) or len(faces):
                raise ReconstructionValidationError("Rejected projections cannot expose mesh assignments.")
        source_confidence = _probability(self.source_confidence, "source_confidence")
        mean_quality = _probability(self.mean_vertex_quality, "mean_vertex_quality")
        projection_confidence = _probability(self.projection_confidence, "projection_confidence")
        uncertainty = self.median_vertex_uncertainty_reconstruction_units
        if uncertainty is not None:
            uncertainty = float(uncertainty)
            if not isfinite(uncertainty) or uncertainty <= 0.0:
                raise ReconstructionValidationError(
                    "median_vertex_uncertainty_reconstruction_units must be finite and positive."
                )
        object.__setattr__(self, "vertex_indices", _readonly(vertices, np.uint32))
        object.__setattr__(self, "face_indices", _readonly(faces, np.uint32))
        object.__setattr__(self, "source_confidence", source_confidence)
        object.__setattr__(self, "mean_vertex_quality", mean_quality)
        object.__setattr__(self, "projection_confidence", projection_confidence)
        object.__setattr__(self, "median_vertex_uncertainty_reconstruction_units", uncertainty)

    def as_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "frame_id": self.frame_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "semantic_role": self.semantic_role.value,
            "track_id": self.track_id,
            "mask_id": self.mask_id,
            "camera_profile_id": self.camera_profile_id,
            "pose_layer": self.pose_layer,
            "accepted": self.accepted,
            "rejection": self.rejection.value,
            "visible_vertex_count": self.visible_vertex_count,
            "assigned_vertex_count": int(len(self.vertex_indices)),
            "assigned_face_count": int(len(self.face_indices)),
            "source_confidence": self.source_confidence,
            "mean_vertex_quality": self.mean_vertex_quality,
            "median_vertex_uncertainty_reconstruction_units": self.median_vertex_uncertainty_reconstruction_units,
            "projection_confidence": self.projection_confidence,
        }


@dataclass(frozen=True, slots=True)
class Semantic3DObject:
    object_id: str
    class_id: int
    class_name: str
    semantic_role: SemanticGeometryRole
    confidence: float
    vertex_indices: np.ndarray
    face_indices: np.ndarray
    centroid_reconstruction: np.ndarray
    bounds_min_reconstruction: np.ndarray
    bounds_max_reconstruction: np.ndarray
    principal_axes_reconstruction: np.ndarray
    orientation_reliable: bool
    observation_ids: tuple[str, ...]
    frame_ids: tuple[str, ...]
    mask_ids: tuple[str, ...]
    track_ids: tuple[str, ...]
    camera_profile_ids: tuple[str, ...]
    pose_layers: tuple[str, ...]
    mean_vertex_quality: float
    median_vertex_uncertainty_reconstruction_units: float
    maximum_vertex_uncertainty_reconstruction_units: float
    scale_profile_id: str
    metric_geometry_status: MetricGeometryStatus
    unit_to_metre: float | None
    relative_scale_uncertainty: float | None
    certified_metrology: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "object_id", require_nonempty(self.object_id, field_name="object_id"))
        if isinstance(self.class_id, bool) or int(self.class_id) < 0:
            raise ReconstructionValidationError("class_id must be a non-negative integer.")
        object.__setattr__(self, "class_id", int(self.class_id))
        object.__setattr__(
            self, "class_name", require_nonempty(self.class_name, field_name="class_name")
        )
        if not isinstance(self.semantic_role, SemanticGeometryRole):
            raise TypeError("semantic_role must be SemanticGeometryRole.")
        object.__setattr__(self, "confidence", _probability(self.confidence, "confidence"))
        vertices = np.asarray(self.vertex_indices, dtype=np.uint32)
        faces = np.asarray(self.face_indices, dtype=np.uint32)
        if vertices.ndim != 1 or len(vertices) < 1 or len(vertices) != len(np.unique(vertices)):
            raise ReconstructionValidationError("Semantic objects require unique 1D vertex indices.")
        if faces.ndim != 1 or len(faces) != len(np.unique(faces)):
            raise ReconstructionValidationError("face_indices must be unique and 1D.")
        centroid = _vector3(self.centroid_reconstruction, "centroid_reconstruction")
        bounds_min = _vector3(self.bounds_min_reconstruction, "bounds_min_reconstruction")
        bounds_max = _vector3(self.bounds_max_reconstruction, "bounds_max_reconstruction")
        if np.any(bounds_max < bounds_min):
            raise ReconstructionValidationError("Object bounds are invalid.")
        if np.any(centroid < bounds_min - 1e-9) or np.any(centroid > bounds_max + 1e-9):
            raise ReconstructionValidationError("Object centroid must lie inside object bounds.")
        axes = np.asarray(self.principal_axes_reconstruction, dtype=np.float64)
        if axes.shape != (3, 3) or not np.isfinite(axes).all():
            raise ReconstructionValidationError("principal_axes_reconstruction must be finite 3x3.")
        if not np.allclose(axes.T @ axes, np.eye(3), atol=1e-5) or np.linalg.det(axes) < 0.99:
            raise ReconstructionValidationError("principal_axes_reconstruction must be right-handed orthonormal.")
        if bool(self.certified_metrology):
            raise ReconstructionValidationError("Phase 8.7B cannot claim certified metrology.")
        object.__setattr__(self, "vertex_indices", _readonly(vertices, np.uint32))
        object.__setattr__(self, "face_indices", _readonly(faces, np.uint32))
        object.__setattr__(self, "centroid_reconstruction", _readonly(centroid, np.float64))
        object.__setattr__(self, "bounds_min_reconstruction", _readonly(bounds_min, np.float64))
        object.__setattr__(self, "bounds_max_reconstruction", _readonly(bounds_max, np.float64))
        object.__setattr__(self, "principal_axes_reconstruction", _readonly(axes, np.float64))
        object.__setattr__(self, "orientation_reliable", bool(self.orientation_reliable))
        for field_name in (
            "observation_ids", "frame_ids", "mask_ids", "track_ids", "camera_profile_ids", "pose_layers"
        ):
            values = tuple(dict.fromkeys(str(value).strip() for value in getattr(self, field_name) if str(value).strip()))
            if field_name in {"observation_ids", "frame_ids", "mask_ids", "camera_profile_ids", "pose_layers"} and not values:
                raise ReconstructionValidationError(f"{field_name} must not be empty.")
            object.__setattr__(self, field_name, values)
        object.__setattr__(
            self, "mean_vertex_quality", _probability(self.mean_vertex_quality, "mean_vertex_quality")
        )
        for field_name in (
            "median_vertex_uncertainty_reconstruction_units",
            "maximum_vertex_uncertainty_reconstruction_units",
        ):
            value = float(getattr(self, field_name))
            if not isfinite(value) or value <= 0.0:
                raise ReconstructionValidationError(f"{field_name} must be finite and positive.")
            object.__setattr__(self, field_name, value)
        object.__setattr__(
            self, "scale_profile_id", require_nonempty(self.scale_profile_id, field_name="scale_profile_id")
        )
        if not isinstance(self.metric_geometry_status, MetricGeometryStatus):
            raise TypeError("metric_geometry_status must be MetricGeometryStatus.")
        unit = self.unit_to_metre
        if self.metric_geometry_status is MetricGeometryStatus.ARBITRARY:
            if unit is not None or self.relative_scale_uncertainty is not None:
                raise ReconstructionValidationError(
                    "Arbitrary semantic geometry cannot expose metric scale values."
                )
        else:
            if unit is None:
                raise ReconstructionValidationError("Metric semantic geometry requires unit_to_metre.")
            unit = float(unit)
            if not isfinite(unit) or unit <= 0.0:
                raise ReconstructionValidationError("unit_to_metre must be finite and positive.")
        rel = self.relative_scale_uncertainty
        if rel is not None:
            rel = float(rel)
            if not isfinite(rel) or rel < 0.0:
                raise ReconstructionValidationError(
                    "relative_scale_uncertainty must be finite and non-negative."
                )
        object.__setattr__(self, "unit_to_metre", unit)
        object.__setattr__(self, "relative_scale_uncertainty", rel)
        object.__setattr__(self, "certified_metrology", False)

    def centroid_metres(self, scale_profile: ScaleReferenceProfile) -> np.ndarray:
        self._validate_metric_scale(scale_profile)
        value = self.centroid_reconstruction * float(scale_profile.gauge.unit_to_metre)
        value.setflags(write=False)
        return value

    def bounds_metres(self, scale_profile: ScaleReferenceProfile) -> tuple[np.ndarray, np.ndarray]:
        self._validate_metric_scale(scale_profile)
        factor = float(scale_profile.gauge.unit_to_metre)
        lower = np.asarray(self.bounds_min_reconstruction * factor, dtype=np.float64)
        upper = np.asarray(self.bounds_max_reconstruction * factor, dtype=np.float64)
        lower.setflags(write=False)
        upper.setflags(write=False)
        return lower, upper

    def _validate_metric_scale(self, scale_profile: ScaleReferenceProfile) -> None:
        if not isinstance(scale_profile, ScaleReferenceProfile):
            raise TypeError("scale_profile must be ScaleReferenceProfile.")
        if scale_profile.profile_id != self.scale_profile_id:
            raise ReconstructionValidationError("Scale profile does not match semantic object provenance.")
        if self.metric_geometry_status is MetricGeometryStatus.ARBITRARY:
            raise ReconstructionValidationError("Arbitrary semantic geometry cannot be converted to metres.")
        if not scale_profile.is_metric or scale_profile.gauge.unit_to_metre is None:
            raise ReconstructionValidationError("Metric semantic geometry requires a metric scale profile.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "semantic_role": self.semantic_role.value,
            "confidence": self.confidence,
            "vertex_count": int(len(self.vertex_indices)),
            "face_count": int(len(self.face_indices)),
            "centroid_reconstruction": self.centroid_reconstruction.tolist(),
            "bounds_min_reconstruction": self.bounds_min_reconstruction.tolist(),
            "bounds_max_reconstruction": self.bounds_max_reconstruction.tolist(),
            "principal_axes_reconstruction": self.principal_axes_reconstruction.tolist(),
            "orientation_reliable": self.orientation_reliable,
            "observation_ids": list(self.observation_ids),
            "frame_ids": list(self.frame_ids),
            "mask_ids": list(self.mask_ids),
            "track_ids": list(self.track_ids),
            "camera_profile_ids": list(self.camera_profile_ids),
            "pose_layers": list(self.pose_layers),
            "mean_vertex_quality": self.mean_vertex_quality,
            "median_vertex_uncertainty_reconstruction_units": self.median_vertex_uncertainty_reconstruction_units,
            "maximum_vertex_uncertainty_reconstruction_units": self.maximum_vertex_uncertainty_reconstruction_units,
            "scale_profile_id": self.scale_profile_id,
            "metric_geometry_status": self.metric_geometry_status.value,
            "unit_to_metre": self.unit_to_metre,
            "relative_scale_uncertainty": self.relative_scale_uncertainty,
            "certified_metrology": False,
        }


@dataclass(frozen=True, slots=True)
class SemanticGeometryStatistics:
    observations_total: int
    observations_accepted: int
    observations_rejected: int
    persistent_object_count: int
    assigned_vertex_count: int
    ambiguous_vertex_count: int
    unassigned_vertex_count: int
    assigned_face_count: int

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) < 0:
                raise ReconstructionValidationError(f"{name} must be a non-negative integer.")
            object.__setattr__(self, name, int(value))
        if self.observations_accepted + self.observations_rejected != self.observations_total:
            raise ReconstructionValidationError("Observation statistics do not sum to total.")

    def as_dict(self) -> dict[str, int]:
        return {name: int(getattr(self, name)) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class SemanticGeometryResult:
    session_id: str
    scene_profile: DenseSceneProfile
    pose_convention: str
    reconstruction_frame_id: str
    scale_profile_id: str
    metric_geometry_status: MetricGeometryStatus
    unit_to_metre: float | None
    relative_scale_uncertainty: float | None
    certified_metrology: bool
    source_surface_schema_version: int
    policy: SemanticAssociationPolicy
    projections: tuple[SemanticObservationProjection, ...]
    objects: tuple[Semantic3DObject, ...]
    statistics: SemanticGeometryStatistics
    schema_version: int = SEMANTIC_GEOMETRY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_id", require_nonempty(self.session_id, field_name="session_id"))
        if not isinstance(self.scene_profile, DenseSceneProfile):
            raise TypeError("scene_profile must be DenseSceneProfile.")
        object.__setattr__(
            self, "pose_convention", require_nonempty(self.pose_convention, field_name="pose_convention")
        )
        if self.pose_convention != SEMANTIC_GEOMETRY_POSE_CONVENTION:
            raise ReconstructionValidationError("Unsupported semantic geometry pose convention.")
        object.__setattr__(
            self,
            "reconstruction_frame_id",
            require_nonempty(self.reconstruction_frame_id, field_name="reconstruction_frame_id"),
        )
        object.__setattr__(
            self, "scale_profile_id", require_nonempty(self.scale_profile_id, field_name="scale_profile_id")
        )
        if not isinstance(self.metric_geometry_status, MetricGeometryStatus):
            raise TypeError("metric_geometry_status must be MetricGeometryStatus.")
        if bool(self.certified_metrology):
            raise ReconstructionValidationError("Phase 8.7B cannot claim certified metrology.")
        unit = self.unit_to_metre
        rel = self.relative_scale_uncertainty
        if self.metric_geometry_status is MetricGeometryStatus.ARBITRARY:
            if unit is not None or rel is not None:
                raise ReconstructionValidationError("Arbitrary semantic result cannot expose metric scale.")
        else:
            if unit is None:
                raise ReconstructionValidationError("Metric semantic result requires unit_to_metre.")
            unit = float(unit)
            if not isfinite(unit) or unit <= 0.0:
                raise ReconstructionValidationError("unit_to_metre must be finite and positive.")
            if rel is not None:
                rel = float(rel)
                if not isfinite(rel) or rel < 0.0:
                    raise ReconstructionValidationError(
                        "relative_scale_uncertainty must be finite and non-negative."
                    )
        if not isinstance(self.policy, SemanticAssociationPolicy):
            raise TypeError("policy must be SemanticAssociationPolicy.")
        projections = tuple(self.projections)
        objects = tuple(self.objects)
        if len({p.observation_id for p in projections}) != len(projections):
            raise ReconstructionValidationError("Projection observation IDs must be unique.")
        if len({obj.object_id for obj in objects}) != len(objects):
            raise ReconstructionValidationError("Semantic object IDs must be unique.")
        if not isinstance(self.statistics, SemanticGeometryStatistics):
            raise TypeError("statistics must be SemanticGeometryStatistics.")
        if self.statistics.observations_total != len(projections):
            raise ReconstructionValidationError("Projection count disagrees with statistics.")
        if self.statistics.persistent_object_count != len(objects):
            raise ReconstructionValidationError("Object count disagrees with statistics.")
        if int(self.source_surface_schema_version) <= 0:
            raise ReconstructionValidationError("source_surface_schema_version must be positive.")
        if int(self.schema_version) != SEMANTIC_GEOMETRY_SCHEMA_VERSION:
            raise ReconstructionValidationError("Unsupported semantic geometry schema version.")
        object.__setattr__(self, "unit_to_metre", unit)
        object.__setattr__(self, "relative_scale_uncertainty", rel)
        object.__setattr__(self, "certified_metrology", False)
        object.__setattr__(self, "projections", projections)
        object.__setattr__(self, "objects", objects)
        object.__setattr__(self, "source_surface_schema_version", int(self.source_surface_schema_version))
        object.__setattr__(self, "schema_version", SEMANTIC_GEOMETRY_SCHEMA_VERSION)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "scene_profile": self.scene_profile.value,
            "pose_convention": self.pose_convention,
            "reconstruction_frame_id": self.reconstruction_frame_id,
            "scale_profile_id": self.scale_profile_id,
            "metric_geometry_status": self.metric_geometry_status.value,
            "unit_to_metre": self.unit_to_metre,
            "relative_scale_uncertainty": self.relative_scale_uncertainty,
            "certified_metrology": False,
            "source_surface_schema_version": self.source_surface_schema_version,
            "policy": self.policy.as_dict(),
            "statistics": self.statistics.as_dict(),
            "objects": [obj.as_dict() for obj in self.objects],
            "projections": [projection.as_dict() for projection in self.projections],
        }


def project_surface_mesh_vertices(
    surface: SurfaceMeshResult,
    camera_profile: CameraGeometryProfile,
    T_reconstruction_from_camera: np.ndarray,
    *,
    minimum_camera_depth: float = 1e-6,
) -> MeshProjection:
    """Project visible mesh vertices into one camera image.

    The projection uses OpenCV's calibrated pinhole/fisheye kernels for the
    same distortion models already supported by Phase 8.6C.  Backend-specific
    distortion remains intentionally unsupported.
    """

    if not isinstance(surface, SurfaceMeshResult):
        raise TypeError("surface must be SurfaceMeshResult.")
    if not isinstance(camera_profile, CameraGeometryProfile):
        raise TypeError("camera_profile must be CameraGeometryProfile.")
    matrix = validate_rigid_matrix(
        T_reconstruction_from_camera,
        field_name="T_reconstruction_from_camera",
    )
    minimum_depth = float(minimum_camera_depth)
    if not isfinite(minimum_depth) or minimum_depth <= 0.0:
        raise ReconstructionValidationError("minimum_camera_depth must be finite and positive.")
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    vertices = surface.mesh.vertices_reconstruction
    points_camera = (vertices - translation) @ rotation
    visible = np.isfinite(points_camera).all(axis=1) & (points_camera[:, 2] > minimum_depth)
    visible_indices = np.flatnonzero(visible).astype(np.uint32)
    if len(visible_indices) == 0:
        return MeshProjection(np.empty((0, 2), np.float64), visible_indices)

    points = np.ascontiguousarray(points_camera[visible], dtype=np.float64)
    camera = camera_profile.camera
    intrinsic = np.asarray(camera.intrinsics.matrix, dtype=np.float64)
    distortion = np.asarray(camera.distortion_coefficients, dtype=np.float64)
    zero = np.zeros((3, 1), dtype=np.float64)
    if camera.distortion_model is CameraDistortionModel.BACKEND_SPECIFIC:
        raise ReconstructionValidationError(
            "Backend-specific distortion is unsupported for semantic mesh projection."
        )
    if camera.distortion_model is CameraDistortionModel.OPENCV_FISHEYE:
        projected, _ = cv2.fisheye.projectPoints(
            points.reshape(-1, 1, 3), zero, zero, intrinsic, distortion
        )
    else:
        projected, _ = cv2.projectPoints(
            points.reshape(-1, 1, 3), zero, zero, intrinsic, distortion
        )
    pixels = np.ascontiguousarray(projected.reshape(-1, 2), dtype=np.float64)
    width = camera.image_size.width
    height = camera.image_size.height
    inside = (
        np.isfinite(pixels).all(axis=1)
        & (pixels[:, 0] >= 0.0)
        & (pixels[:, 0] < float(width))
        & (pixels[:, 1] >= 0.0)
        & (pixels[:, 1] < float(height))
    )
    return MeshProjection(pixels[inside], visible_indices[inside])


def build_semantic_3d_geometry(
    surface: SurfaceMeshResult,
    observations: Iterable[SemanticFrameObservation],
    *,
    policy: SemanticAssociationPolicy | None = None,
    role_by_class_name: Mapping[str, SemanticGeometryRole | str] | None = None,
) -> SemanticGeometryResult:
    """Project 2D observations onto a surface and build persistent 3D objects."""

    if not isinstance(surface, SurfaceMeshResult):
        raise TypeError("surface must be SurfaceMeshResult.")
    resolved_policy = policy or SemanticAssociationPolicy()
    if not isinstance(resolved_policy, SemanticAssociationPolicy):
        raise TypeError("policy must be SemanticAssociationPolicy.")
    frames = tuple(observations)
    if len(frames) > resolved_policy.maximum_observations:
        raise ReconstructionValidationError(
            f"Semantic observation capacity exceeded ({len(frames)} > {resolved_policy.maximum_observations})."
        )
    if len({item.observation.observation_id for item in frames}) != len(frames):
        raise ReconstructionValidationError("Semantic observation IDs must be unique.")
    _validate_track_identity_consistency(frames, role_by_class_name, surface.scene_profile)

    projections = tuple(
        _project_observation(surface, item, resolved_policy, role_by_class_name)
        for item in sorted(frames, key=lambda value: (value.frame_id, value.observation.observation_id))
    )
    accepted = [projection for projection in projections if projection.accepted]
    clusters = _associate_projections(accepted, resolved_policy)
    if len(clusters) > resolved_policy.maximum_objects:
        raise ReconstructionValidationError(
            f"Semantic object capacity exceeded ({len(clusters)} > {resolved_policy.maximum_objects})."
        )
    objects = tuple(
        _object_from_cluster(surface, cluster, index)
        for index, cluster in enumerate(clusters)
    )

    vertex_membership = np.zeros(surface.mesh.vertex_count, dtype=np.uint16)
    face_union: set[int] = set()
    for obj in objects:
        vertex_membership[obj.vertex_indices.astype(np.int64)] += 1
        face_union.update(int(value) for value in obj.face_indices)
    assigned_vertex_count = int(np.count_nonzero(vertex_membership))
    ambiguous_vertex_count = int(np.count_nonzero(vertex_membership > 1))
    stats = SemanticGeometryStatistics(
        observations_total=len(projections),
        observations_accepted=len(accepted),
        observations_rejected=len(projections) - len(accepted),
        persistent_object_count=len(objects),
        assigned_vertex_count=assigned_vertex_count,
        ambiguous_vertex_count=ambiguous_vertex_count,
        unassigned_vertex_count=surface.mesh.vertex_count - assigned_vertex_count,
        assigned_face_count=len(face_union),
    )
    return SemanticGeometryResult(
        session_id=surface.session_id,
        scene_profile=surface.scene_profile,
        pose_convention=surface.pose_convention,
        reconstruction_frame_id=surface.reconstruction_frame_id,
        scale_profile_id=surface.scale_profile_id,
        metric_geometry_status=surface.metric_geometry_status,
        unit_to_metre=surface.unit_to_metre,
        relative_scale_uncertainty=surface.relative_scale_uncertainty,
        certified_metrology=False,
        source_surface_schema_version=surface.schema_version,
        policy=resolved_policy,
        projections=projections,
        objects=objects,
        statistics=stats,
    )


def write_semantic_groups_obj(
    surface: SurfaceMeshResult,
    semantic: SemanticGeometryResult,
    path: str | Path,
) -> Path:
    """Write a lightweight grouped OBJ for non-UI semantic validation."""

    if semantic.session_id != surface.session_id:
        raise ReconstructionValidationError("Semantic result and surface belong to different sessions.")
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    vertices = surface.mesh.vertices_reconstruction
    lines = ["# Phase 8.7B semantic grouped mesh", f"# session {surface.session_id}"]
    for vertex in vertices:
        lines.append(f"v {vertex[0]:.9g} {vertex[1]:.9g} {vertex[2]:.9g}")
    for obj in semantic.objects:
        lines.append(f"g {obj.object_id}")
        lines.append(f"# class={obj.class_name} role={obj.semantic_role.value} confidence={obj.confidence:.6f}")
        for face_index in obj.face_indices:
            face = surface.mesh.faces[int(face_index)] + 1
            lines.append(f"f {int(face[0])} {int(face[1])} {int(face[2])}")
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


def _project_observation(
    surface: SurfaceMeshResult,
    item: SemanticFrameObservation,
    policy: SemanticAssociationPolicy,
    role_by_class_name: Mapping[str, SemanticGeometryRole | str] | None,
) -> SemanticObservationProjection:
    observation = item.observation
    role = _resolve_role(observation.class_name, role_by_class_name, surface.scene_profile)
    base = dict(
        observation_id=observation.observation_id,
        frame_id=item.frame_id,
        class_id=observation.class_id,
        class_name=observation.class_name,
        semantic_role=role,
        track_id=observation.track_id,
        camera_profile_id=item.camera_profile.profile_id,
        pose_layer=item.pose_layer,
    )
    if item.frame_id not in surface.source_frame_ids:
        return _rejected_projection(base, SemanticProjectionRejection.FRAME_NOT_IN_SURFACE_PROVENANCE)
    camera = item.camera_profile.camera
    if (
        observation.frame_reference.source_size.width != camera.image_size.width
        or observation.frame_reference.source_size.height != camera.image_size.height
    ):
        return _rejected_projection(base, SemanticProjectionRejection.CAMERA_IMAGE_SIZE_MISMATCH)
    mask = observation.selected_mask
    if mask is None:
        return _rejected_projection(base, SemanticProjectionRejection.NO_AUTHORITATIVE_MASK)
    base["mask_id"] = mask.mask_id
    if (
        mask.pixel_space.size.width != camera.image_size.width
        or mask.pixel_space.size.height != camera.image_size.height
        or mask.pixel_space.space_id != camera.intrinsics.pixel_space_id
    ):
        return _rejected_projection(base, SemanticProjectionRejection.MASK_PIXEL_SPACE_MISMATCH)
    try:
        dense_mask = normalize_mask_payload(mask.payload, mask.pixel_space)
    except UnsupportedMaskRepresentationError:
        return _rejected_projection(
            base, SemanticProjectionRejection.UNSUPPORTED_MASK_REPRESENTATION
        )
    projection = project_surface_mesh_vertices(
        surface,
        item.camera_profile,
        item.T_reconstruction_from_camera,
        minimum_camera_depth=policy.minimum_camera_depth,
    )
    if len(projection.visible_vertex_indices) == 0:
        return _rejected_projection(
            base,
            SemanticProjectionRejection.NO_VISIBLE_VERTICES,
            visible_vertex_count=0,
        )
    x = np.rint(projection.pixel_xy[:, 0]).astype(np.int64)
    y = np.rint(projection.pixel_xy[:, 1]).astype(np.int64)
    x = np.clip(x, 0, camera.image_size.width - 1)
    y = np.clip(y, 0, camera.image_size.height - 1)
    inside_mask = dense_mask[y, x]
    vertex_indices = np.unique(projection.visible_vertex_indices[inside_mask]).astype(np.uint32)
    if len(vertex_indices) < policy.minimum_projected_vertices:
        return _rejected_projection(
            base,
            SemanticProjectionRejection.INSUFFICIENT_VERTEX_SUPPORT,
            visible_vertex_count=len(projection.visible_vertex_indices),
        )
    faces = surface.mesh.faces.astype(np.int64, copy=False)
    membership = np.zeros(surface.mesh.vertex_count, dtype=np.uint8)
    membership[vertex_indices.astype(np.int64)] = 1
    face_support = np.sum(membership[faces], axis=1)
    face_indices = np.flatnonzero(face_support >= policy.face_minimum_assigned_vertices).astype(np.uint32)
    quality = surface.mesh.vertex_quality_scores[vertex_indices.astype(np.int64)]
    uncertainty = surface.mesh.vertex_uncertainty_reconstruction_units[vertex_indices.astype(np.int64)]
    source_confidence = _observation_confidence(observation)
    support_score = min(1.0, len(vertex_indices) / max(policy.minimum_projected_vertices * 4.0, 1.0))
    mean_quality = float(np.mean(quality))
    projection_confidence = float(
        np.clip(0.45 * source_confidence + 0.40 * mean_quality + 0.15 * support_score, 0.0, 1.0)
    )
    return SemanticObservationProjection(
        **base,
        accepted=True,
        rejection=SemanticProjectionRejection.NONE,
        visible_vertex_count=len(projection.visible_vertex_indices),
        vertex_indices=vertex_indices,
        face_indices=face_indices,
        source_confidence=source_confidence,
        mean_vertex_quality=mean_quality,
        median_vertex_uncertainty_reconstruction_units=float(np.median(uncertainty)),
        projection_confidence=projection_confidence,
    )


def _rejected_projection(
    base: dict[str, Any],
    rejection: SemanticProjectionRejection,
    *,
    visible_vertex_count: int = 0,
) -> SemanticObservationProjection:
    values = dict(base)
    values["mask_id"] = str(values.get("mask_id", ""))
    return SemanticObservationProjection(
        **values,
        accepted=False,
        rejection=rejection,
        visible_vertex_count=visible_vertex_count,
        vertex_indices=np.empty((0,), dtype=np.uint32),
        face_indices=np.empty((0,), dtype=np.uint32),
        source_confidence=0.0,
        mean_vertex_quality=0.0,
        median_vertex_uncertainty_reconstruction_units=None,
        projection_confidence=0.0,
    )


def _associate_projections(
    projections: list[SemanticObservationProjection],
    policy: SemanticAssociationPolicy,
) -> list[list[SemanticObservationProjection]]:
    clusters: list[list[SemanticObservationProjection]] = []
    for projection in projections:
        candidates: list[tuple[float, int]] = []
        for index, cluster in enumerate(clusters):
            head = cluster[0]
            if head.class_id != projection.class_id or head.semantic_role is not projection.semantic_role:
                continue
            if any(existing.frame_id == projection.frame_id for existing in cluster):
                # Geometry overlap from the same frame is not cross-view evidence.
                if projection.track_id and projection.track_id in {value.track_id for value in cluster if value.track_id}:
                    raise ReconstructionValidationError(
                        "One semantic track ID cannot identify two observations in the same frame."
                    )
                continue
            cluster_track_ids = {value.track_id for value in cluster if value.track_id}
            if projection.track_id and projection.track_id in cluster_track_ids:
                candidates.append((2.0, index))
                continue
            if projection.track_id and cluster_track_ids and projection.track_id not in cluster_track_ids:
                # Distinct explicit tracker IDs are kept separate; do not overrule them with geometry.
                continue
            union = np.unique(
                np.concatenate([value.vertex_indices for value in cluster])
            ).astype(np.uint32)
            shared = np.intersect1d(union, projection.vertex_indices, assume_unique=True)
            if len(shared) < policy.minimum_shared_vertices:
                continue
            denom = len(union) + len(projection.vertex_indices) - len(shared)
            iou = float(len(shared) / denom) if denom else 0.0
            if iou >= policy.minimum_cross_view_vertex_iou:
                candidates.append((iou, index))
        if candidates:
            _, chosen = max(candidates, key=lambda item: (item[0], -item[1]))
            clusters[chosen].append(projection)
        else:
            clusters.append([projection])
    return clusters


def _object_from_cluster(
    surface: SurfaceMeshResult,
    cluster: list[SemanticObservationProjection],
    ordinal: int,
) -> Semantic3DObject:
    ordered = sorted(cluster, key=lambda value: (value.frame_id, value.observation_id))
    vertices_idx = np.unique(np.concatenate([value.vertex_indices for value in ordered])).astype(np.uint32)
    faces_idx = np.unique(np.concatenate([value.face_indices for value in ordered])).astype(np.uint32)
    points = surface.mesh.vertices_reconstruction[vertices_idx.astype(np.int64)]
    qualities = surface.mesh.vertex_quality_scores[vertices_idx.astype(np.int64)]
    uncertainties = surface.mesh.vertex_uncertainty_reconstruction_units[vertices_idx.astype(np.int64)]
    centroid = np.mean(points, axis=0)
    bounds_min = np.min(points, axis=0)
    bounds_max = np.max(points, axis=0)
    axes, reliable = _principal_axes(points)
    weights = np.asarray([max(len(value.vertex_indices), 1) for value in ordered], dtype=np.float64)
    confidences = np.asarray([value.projection_confidence for value in ordered], dtype=np.float64)
    confidence = float(np.average(confidences, weights=weights))
    head = ordered[0]
    object_id = f"semantic-{head.class_id:04d}-{ordinal:04d}"
    return Semantic3DObject(
        object_id=object_id,
        class_id=head.class_id,
        class_name=head.class_name,
        semantic_role=head.semantic_role,
        confidence=confidence,
        vertex_indices=vertices_idx,
        face_indices=faces_idx,
        centroid_reconstruction=centroid,
        bounds_min_reconstruction=bounds_min,
        bounds_max_reconstruction=bounds_max,
        principal_axes_reconstruction=axes,
        orientation_reliable=reliable,
        observation_ids=tuple(value.observation_id for value in ordered),
        frame_ids=tuple(value.frame_id for value in ordered),
        mask_ids=tuple(value.mask_id for value in ordered),
        track_ids=tuple(value.track_id for value in ordered if value.track_id),
        camera_profile_ids=tuple(value.camera_profile_id for value in ordered),
        pose_layers=tuple(value.pose_layer for value in ordered),
        mean_vertex_quality=float(np.mean(qualities)),
        median_vertex_uncertainty_reconstruction_units=float(np.median(uncertainties)),
        maximum_vertex_uncertainty_reconstruction_units=float(np.max(uncertainties)),
        scale_profile_id=surface.scale_profile_id,
        metric_geometry_status=surface.metric_geometry_status,
        unit_to_metre=surface.unit_to_metre,
        relative_scale_uncertainty=surface.relative_scale_uncertainty,
        certified_metrology=False,
    )


def _validate_track_identity_consistency(
    frames: tuple[SemanticFrameObservation, ...],
    role_by_class_name: Mapping[str, SemanticGeometryRole | str] | None,
    scene_profile: DenseSceneProfile,
) -> None:
    seen: dict[str, tuple[int, SemanticGeometryRole]] = {}
    for item in frames:
        track_id = item.observation.track_id.strip()
        if not track_id:
            continue
        role = _resolve_role(item.observation.class_name, role_by_class_name, scene_profile)
        identity = (item.observation.class_id, role)
        previous = seen.get(track_id)
        if previous is not None and previous != identity:
            raise ReconstructionValidationError(
                f"Track ID {track_id!r} changes semantic class/role within one result."
            )
        seen[track_id] = identity


def _resolve_role(
    class_name: str,
    role_by_class_name: Mapping[str, SemanticGeometryRole | str] | None,
    scene_profile: DenseSceneProfile,
) -> SemanticGeometryRole:
    if role_by_class_name and class_name in role_by_class_name:
        value = role_by_class_name[class_name]
        return value if isinstance(value, SemanticGeometryRole) else SemanticGeometryRole(str(value))
    return (
        SemanticGeometryRole.DIGITAL_TWIN_ENTITY
        if scene_profile is DenseSceneProfile.SPATIAL_DIGITAL_TWIN
        else SemanticGeometryRole.GENERIC_OBJECT
    )


def _observation_confidence(observation: ObjectObservation) -> float:
    values: list[float] = []
    if observation.detection is not None:
        values.append(float(observation.detection.confidence))
    mask = observation.selected_mask
    if mask is not None and mask.confidence is not None and mask.confidence_kind.value == "confidence":
        values.append(float(mask.confidence))
    if not values:
        return 0.5
    return float(np.clip(np.mean(values), 0.0, 1.0))


def _principal_axes(points: np.ndarray) -> tuple[np.ndarray, bool]:
    if len(points) < 3:
        return np.eye(3, dtype=np.float64), False
    centered = np.asarray(points, dtype=np.float64) - np.mean(points, axis=0)
    covariance = centered.T @ centered / max(len(points) - 1, 1)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values = np.maximum(values[order], 0.0)
    axes = vectors[:, order]
    for column in range(3):
        vector = axes[:, column]
        pivot = int(np.argmax(np.abs(vector)))
        if vector[pivot] < 0.0:
            axes[:, column] *= -1.0
    if np.linalg.det(axes) < 0.0:
        axes[:, 2] *= -1.0
    epsilon = max(float(values[0]) * 1e-9, 1e-15)
    first_distinct = values[0] / max(values[1], epsilon) >= 1.08
    second_distinct = values[1] / max(values[2], epsilon) >= 1.25
    reliable = bool(values[0] > epsilon and first_distinct and second_distinct)
    return np.asarray(axes, dtype=np.float64), reliable


def _readonly(value: object, dtype: np.dtype[Any]) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    array.setflags(write=False)
    return array


def _vector3(value: object, field_name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ReconstructionValidationError(f"{field_name} must be a finite 3-vector.")
    return array


def _probability(value: object, field_name: str) -> float:
    result = float(value)
    if not isfinite(result) or not 0.0 <= result <= 1.0:
        raise ReconstructionValidationError(f"{field_name} must be finite in [0, 1].")
    return result

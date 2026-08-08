"""Quality-aware surface mesh representation for Phase 8.7A.

Phase 8.7A converts the trusted dense geometry produced by Phase 8.6C/8.6D
into a bounded triangle-mesh artifact while preserving reconstruction gauge,
quality, uncertainty, and provenance semantics.

The implementation is deliberately dependency-light.  It uses SciPy's cKDTree
(already part of the project dependency set) for deterministic local
neighbourhoods and constructs triangles in local tangent planes.  It does not
silently smooth vertex positions, invent surfaces across large gaps, promote
arbitrary geometry to metric geometry, or claim certified metrology.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import cos, isfinite, radians
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from core.contracts.validation import require_nonempty
from core.reconstruction.dense_fusion import DenseFusionResult, DenseSceneProfile
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.geometry_quality import (
    DenseGeometryQualityReport,
    GeometryQualityCode,
)
from core.reconstruction.scale_reference import (
    MetricGeometryStatus,
    ScaleReferenceProfile,
    reconstruction_points_to_metres,
)
from core.reconstruction.sfm_contracts import SfmScaleStatus, SfmWorldAlignmentStatus

SURFACE_MESH_SCHEMA_VERSION = 1
SURFACE_MESH_POSE_CONVENTION = "T_reconstruction_from_camera"


class SurfaceMeshingMethod(str, Enum):
    """Surface reconstruction method used by this Phase 8.7A artifact."""

    LOCAL_TANGENT_FAN = "local_tangent_fan"


@dataclass(frozen=True, slots=True)
class SurfaceMeshingPolicy:
    """Bounded, product-aware controls for local surface triangulation."""

    scene_profile: DenseSceneProfile
    minimum_vertex_quality_score: float
    neighbour_count: int
    neighbourhood_radius_multiplier: float
    maximum_edge_length_multiplier: float
    maximum_fan_angle_degrees: float
    maximum_normal_deviation_degrees: float
    minimum_triangle_area_multiplier: float
    minimum_component_faces: int
    max_vertices: int
    max_faces: int
    method: SurfaceMeshingMethod = SurfaceMeshingMethod.LOCAL_TANGENT_FAN

    def __post_init__(self) -> None:
        if not isinstance(self.scene_profile, DenseSceneProfile):
            raise TypeError("scene_profile must be DenseSceneProfile.")
        if not isinstance(self.method, SurfaceMeshingMethod):
            raise TypeError("method must be SurfaceMeshingMethod.")
        quality = float(self.minimum_vertex_quality_score)
        if not isfinite(quality) or not 0.0 <= quality <= 1.0:
            raise ReconstructionValidationError(
                "minimum_vertex_quality_score must be finite in [0, 1]."
            )
        for name, minimum in (
            ("neighbour_count", 4),
            ("minimum_component_faces", 1),
            ("max_vertices", 4),
            ("max_faces", 1),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) < minimum:
                raise ReconstructionValidationError(f"{name} must be >= {minimum}.")
            object.__setattr__(self, name, int(value))
        for name in (
            "neighbourhood_radius_multiplier",
            "maximum_edge_length_multiplier",
            "minimum_triangle_area_multiplier",
        ):
            value = float(getattr(self, name))
            if not isfinite(value) or value <= 0.0:
                raise ReconstructionValidationError(f"{name} must be finite and positive.")
            object.__setattr__(self, name, value)
        fan_angle = float(self.maximum_fan_angle_degrees)
        if not isfinite(fan_angle) or not 0.0 < fan_angle <= 180.0:
            raise ReconstructionValidationError(
                "maximum_fan_angle_degrees must be finite in (0, 180]."
            )
        normal_angle = float(self.maximum_normal_deviation_degrees)
        if not isfinite(normal_angle) or not 0.0 < normal_angle <= 90.0:
            raise ReconstructionValidationError(
                "maximum_normal_deviation_degrees must be finite in (0, 90]."
            )
        object.__setattr__(self, "minimum_vertex_quality_score", quality)
        object.__setattr__(self, "maximum_fan_angle_degrees", fan_angle)
        object.__setattr__(self, "maximum_normal_deviation_degrees", normal_angle)

    def as_dict(self) -> dict[str, Any]:
        return {
            "scene_profile": self.scene_profile.value,
            "method": self.method.value,
            "minimum_vertex_quality_score": self.minimum_vertex_quality_score,
            "neighbour_count": self.neighbour_count,
            "neighbourhood_radius_multiplier": self.neighbourhood_radius_multiplier,
            "maximum_edge_length_multiplier": self.maximum_edge_length_multiplier,
            "maximum_fan_angle_degrees": self.maximum_fan_angle_degrees,
            "maximum_normal_deviation_degrees": self.maximum_normal_deviation_degrees,
            "minimum_triangle_area_multiplier": self.minimum_triangle_area_multiplier,
            "minimum_component_faces": self.minimum_component_faces,
            "max_vertices": self.max_vertices,
            "max_faces": self.max_faces,
        }


def default_surface_meshing_policy(scene_profile: DenseSceneProfile) -> SurfaceMeshingPolicy:
    if not isinstance(scene_profile, DenseSceneProfile):
        raise TypeError("scene_profile must be DenseSceneProfile.")
    if scene_profile is DenseSceneProfile.OBJECT_INSPECTION:
        return SurfaceMeshingPolicy(
            scene_profile=scene_profile,
            minimum_vertex_quality_score=0.64,
            neighbour_count=12,
            neighbourhood_radius_multiplier=3.0,
            maximum_edge_length_multiplier=2.6,
            maximum_fan_angle_degrees=105.0,
            maximum_normal_deviation_degrees=70.0,
            minimum_triangle_area_multiplier=0.025,
            minimum_component_faces=8,
            max_vertices=30_000,
            max_faces=120_000,
        )
    return SurfaceMeshingPolicy(
        scene_profile=scene_profile,
        minimum_vertex_quality_score=0.58,
        neighbour_count=10,
        neighbourhood_radius_multiplier=3.4,
        maximum_edge_length_multiplier=3.0,
        maximum_fan_angle_degrees=115.0,
        maximum_normal_deviation_degrees=78.0,
        minimum_triangle_area_multiplier=0.02,
        minimum_component_faces=6,
        max_vertices=40_000,
        max_faces=140_000,
    )


def _readonly_array(value: object, *, dtype: np.dtype[Any]) -> np.ndarray:
    array = np.asarray(value, dtype=dtype).copy()
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class SurfaceMesh:
    """Triangle mesh in reconstruction coordinates with quality provenance."""

    vertices_reconstruction: np.ndarray
    faces: np.ndarray
    colors_rgb: np.ndarray
    normals_reconstruction: np.ndarray
    vertex_quality_scores: np.ndarray
    vertex_quality_codes: np.ndarray
    vertex_uncertainty_reconstruction_units: np.ndarray
    source_point_indices: np.ndarray
    face_component_ids: np.ndarray

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices_reconstruction, dtype=np.float64)
        faces = np.asarray(self.faces, dtype=np.uint32)
        colors = np.asarray(self.colors_rgb, dtype=np.uint8)
        normals = np.asarray(self.normals_reconstruction, dtype=np.float32)
        scores = np.asarray(self.vertex_quality_scores, dtype=np.float32)
        codes = np.asarray(self.vertex_quality_codes, dtype=np.uint8)
        uncertainty = np.asarray(self.vertex_uncertainty_reconstruction_units, dtype=np.float32)
        source_indices = np.asarray(self.source_point_indices, dtype=np.uint32)
        component_ids = np.asarray(self.face_component_ids, dtype=np.uint32)

        if vertices.ndim != 2 or vertices.shape[1:] != (3,) or len(vertices) < 3:
            raise ReconstructionValidationError("vertices_reconstruction must contain at least 3 Nx3 vertices.")
        if not np.isfinite(vertices).all():
            raise ReconstructionValidationError("vertices_reconstruction must be finite.")
        vertex_count = len(vertices)
        if faces.ndim != 2 or faces.shape[1:] != (3,) or len(faces) < 1:
            raise ReconstructionValidationError("faces must contain at least one Mx3 triangle.")
        if np.any(faces >= vertex_count):
            raise ReconstructionValidationError("faces reference vertices outside the mesh.")
        if np.any(faces[:, 0] == faces[:, 1]) or np.any(faces[:, 0] == faces[:, 2]) or np.any(
            faces[:, 1] == faces[:, 2]
        ):
            raise ReconstructionValidationError("faces must not contain repeated vertices.")
        if colors.shape != (vertex_count, 3):
            raise ReconstructionValidationError("colors_rgb must have shape Nx3.")
        if normals.shape != (vertex_count, 3) or not np.isfinite(normals).all():
            raise ReconstructionValidationError("normals_reconstruction must be finite Nx3.")
        normal_lengths = np.linalg.norm(normals.astype(np.float64), axis=1)
        if np.any(normal_lengths < 0.9) or np.any(normal_lengths > 1.1):
            raise ReconstructionValidationError("normals_reconstruction must be unit length.")
        if scores.shape != (vertex_count,) or not np.isfinite(scores).all() or np.any(scores < 0.0) or np.any(scores > 1.0):
            raise ReconstructionValidationError("vertex_quality_scores must be finite in [0, 1].")
        if codes.shape != (vertex_count,) or np.any(codes > int(GeometryQualityCode.HIGH)):
            raise ReconstructionValidationError("vertex_quality_codes contain unsupported values.")
        if uncertainty.shape != (vertex_count,) or not np.isfinite(uncertainty).all() or np.any(uncertainty <= 0.0):
            raise ReconstructionValidationError(
                "vertex_uncertainty_reconstruction_units must be finite and positive."
            )
        if source_indices.shape != (vertex_count,) or len(np.unique(source_indices)) != vertex_count:
            raise ReconstructionValidationError("source_point_indices must contain one unique source index per vertex.")
        if component_ids.shape != (len(faces),):
            raise ReconstructionValidationError("face_component_ids must contain one value per face.")

        object.__setattr__(self, "vertices_reconstruction", _readonly_array(vertices, dtype=np.float64))
        object.__setattr__(self, "faces", _readonly_array(faces, dtype=np.uint32))
        object.__setattr__(self, "colors_rgb", _readonly_array(colors, dtype=np.uint8))
        object.__setattr__(self, "normals_reconstruction", _readonly_array(normals, dtype=np.float32))
        object.__setattr__(self, "vertex_quality_scores", _readonly_array(scores, dtype=np.float32))
        object.__setattr__(self, "vertex_quality_codes", _readonly_array(codes, dtype=np.uint8))
        object.__setattr__(
            self,
            "vertex_uncertainty_reconstruction_units",
            _readonly_array(uncertainty, dtype=np.float32),
        )
        object.__setattr__(self, "source_point_indices", _readonly_array(source_indices, dtype=np.uint32))
        object.__setattr__(self, "face_component_ids", _readonly_array(component_ids, dtype=np.uint32))

    @property
    def vertex_count(self) -> int:
        return int(len(self.vertices_reconstruction))

    @property
    def face_count(self) -> int:
        return int(len(self.faces))

    @property
    def component_count(self) -> int:
        return int(len(np.unique(self.face_component_ids)))

    def vertices_metres(self, scale_profile: ScaleReferenceProfile) -> np.ndarray:
        return reconstruction_points_to_metres(self.vertices_reconstruction, scale_profile)

    def uncertainty_metres(self, scale_profile: ScaleReferenceProfile) -> np.ndarray:
        if not isinstance(scale_profile, ScaleReferenceProfile):
            raise TypeError("scale_profile must be ScaleReferenceProfile.")
        if not scale_profile.is_metric or scale_profile.gauge.unit_to_metre is None:
            raise ReconstructionValidationError(
                "Metric mesh uncertainty requires an explicit metric scale reference."
            )
        values = self.vertex_uncertainty_reconstruction_units.astype(np.float64) * float(
            scale_profile.gauge.unit_to_metre
        )
        values.setflags(write=False)
        return values

    def as_dict(self) -> dict[str, Any]:
        edges = _unique_edges(self.faces)
        return {
            "vertex_count": self.vertex_count,
            "face_count": self.face_count,
            "component_count": self.component_count,
            "edge_count": int(len(edges)),
            "minimum_vertex_quality": float(np.min(self.vertex_quality_scores)),
            "mean_vertex_quality": float(np.mean(self.vertex_quality_scores)),
            "median_vertex_uncertainty_reconstruction_units": float(
                np.median(self.vertex_uncertainty_reconstruction_units)
            ),
        }


@dataclass(frozen=True, slots=True)
class SurfaceMeshStatistics:
    input_points: int
    quality_accepted_points: int
    selected_points_before_cleanup: int
    retained_vertices: int
    candidate_faces: int
    retained_faces: int
    rejected_gap_faces: int
    rejected_normal_faces: int
    rejected_degenerate_faces: int
    rejected_duplicate_faces: int
    dropped_faces_capacity: int
    components_before_cleanup: int
    components_retained: int
    removed_small_component_faces: int
    removed_unreferenced_vertices: int

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) < 0:
                raise ReconstructionValidationError(f"{name} must be a non-negative integer.")
            object.__setattr__(self, name, int(value))

    def as_dict(self) -> dict[str, int]:
        return {name: int(getattr(self, name)) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class SurfaceMeshResult:
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
    relative_scale_uncertainty: float | None
    certified_metrology: bool
    source_frame_ids: tuple[str, ...]
    source_pose_layers: tuple[str, ...]
    source_quality_schema_version: int
    quality_limiters: tuple[str, ...]
    policy: SurfaceMeshingPolicy
    mesh: SurfaceMesh
    statistics: SurfaceMeshStatistics
    schema_version: int = SURFACE_MESH_SCHEMA_VERSION

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
        if self.pose_convention != SURFACE_MESH_POSE_CONVENTION:
            raise ReconstructionValidationError("Unsupported surface-mesh pose convention.")
        if not isinstance(self.scale_status, SfmScaleStatus):
            raise TypeError("scale_status must be SfmScaleStatus.")
        if not isinstance(self.world_alignment_status, SfmWorldAlignmentStatus):
            raise TypeError("world_alignment_status must be SfmWorldAlignmentStatus.")
        if bool(self.certified_metrology):
            raise ReconstructionValidationError(
                "Phase 8.7A surface meshes must not claim certified metrology."
            )
        unit_to_metre = self.unit_to_metre
        if self.metric_geometry_status is MetricGeometryStatus.ARBITRARY:
            if unit_to_metre is not None:
                raise ReconstructionValidationError("Arbitrary surface mesh must not expose unit_to_metre.")
        else:
            if unit_to_metre is None:
                raise ReconstructionValidationError("Metric surface mesh requires unit_to_metre.")
            unit_to_metre = float(unit_to_metre)
            if not isfinite(unit_to_metre) or unit_to_metre <= 0.0:
                raise ReconstructionValidationError("unit_to_metre must be finite and positive.")
        scale_uncertainty = self.relative_scale_uncertainty
        if scale_uncertainty is not None:
            scale_uncertainty = float(scale_uncertainty)
            if not isfinite(scale_uncertainty) or scale_uncertainty < 0.0:
                raise ReconstructionValidationError(
                    "relative_scale_uncertainty must be finite and non-negative."
                )
        if self.metric_geometry_status is MetricGeometryStatus.ARBITRARY and scale_uncertainty is not None:
            raise ReconstructionValidationError(
                "Arbitrary surface mesh must not expose metric scale uncertainty."
            )
        if not isinstance(self.policy, SurfaceMeshingPolicy):
            raise TypeError("policy must be SurfaceMeshingPolicy.")
        if self.policy.scene_profile is not self.scene_profile:
            raise ReconstructionValidationError("Surface policy scene profile must match result.")
        if not isinstance(self.mesh, SurfaceMesh):
            raise TypeError("mesh must be SurfaceMesh.")
        if not isinstance(self.statistics, SurfaceMeshStatistics):
            raise TypeError("statistics must be SurfaceMeshStatistics.")
        if self.statistics.retained_vertices != self.mesh.vertex_count:
            raise ReconstructionValidationError("retained_vertices must match mesh vertex count.")
        if self.statistics.retained_faces != self.mesh.face_count:
            raise ReconstructionValidationError("retained_faces must match mesh face count.")
        frames = tuple(require_nonempty(v, field_name="source_frame_id") for v in self.source_frame_ids)
        layers = tuple(require_nonempty(v, field_name="source_pose_layer") for v in self.source_pose_layers)
        if not frames:
            raise ReconstructionValidationError("Surface mesh requires source_frame_ids.")
        if not layers:
            raise ReconstructionValidationError("Surface mesh requires source_pose_layers.")
        limiters = tuple(str(value).strip() for value in self.quality_limiters if str(value).strip())
        object.__setattr__(self, "source_frame_ids", frames)
        object.__setattr__(self, "source_pose_layers", layers)
        object.__setattr__(self, "quality_limiters", tuple(dict.fromkeys(limiters)))
        object.__setattr__(self, "unit_to_metre", unit_to_metre)
        object.__setattr__(self, "relative_scale_uncertainty", scale_uncertainty)
        object.__setattr__(self, "certified_metrology", False)
        if int(self.source_quality_schema_version) <= 0:
            raise ReconstructionValidationError("source_quality_schema_version must be positive.")
        if int(self.schema_version) != SURFACE_MESH_SCHEMA_VERSION:
            raise ReconstructionValidationError("Unsupported surface mesh schema version.")
        object.__setattr__(self, "source_quality_schema_version", int(self.source_quality_schema_version))
        object.__setattr__(self, "schema_version", SURFACE_MESH_SCHEMA_VERSION)

    def vertices_metres(self, scale_profile: ScaleReferenceProfile) -> np.ndarray:
        _validate_mesh_scale_profile(self, scale_profile)
        return self.mesh.vertices_metres(scale_profile)

    def uncertainty_metres(self, scale_profile: ScaleReferenceProfile) -> np.ndarray:
        _validate_mesh_scale_profile(self, scale_profile)
        return self.mesh.uncertainty_metres(scale_profile)

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
            "relative_scale_uncertainty": self.relative_scale_uncertainty,
            "certified_metrology": self.certified_metrology,
            "source_frame_ids": list(self.source_frame_ids),
            "source_pose_layers": list(self.source_pose_layers),
            "source_quality_schema_version": self.source_quality_schema_version,
            "quality_limiters": list(self.quality_limiters),
            "policy": self.policy.as_dict(),
            "mesh": self.mesh.as_dict(),
            "statistics": self.statistics.as_dict(),
        }


def _validate_mesh_scale_profile(result: SurfaceMeshResult, scale_profile: ScaleReferenceProfile) -> None:
    if not isinstance(scale_profile, ScaleReferenceProfile):
        raise TypeError("scale_profile must be ScaleReferenceProfile.")
    if scale_profile.profile_id != result.scale_profile_id:
        raise ReconstructionValidationError("Scale profile ID does not match surface mesh result.")
    if result.metric_geometry_status is MetricGeometryStatus.ARBITRARY:
        raise ReconstructionValidationError("Arbitrary surface mesh cannot be converted to metres.")
    if not scale_profile.is_metric or scale_profile.gauge.unit_to_metre is None:
        raise ReconstructionValidationError("Metric surface mesh requires an explicit metric scale profile.")
    if not np.isclose(float(scale_profile.gauge.unit_to_metre), float(result.unit_to_metre), rtol=0.0, atol=1e-15):
        raise ReconstructionValidationError("Scale profile unit_to_metre does not match surface mesh result.")


def _deterministic_subsample(indices: np.ndarray, maximum: int) -> np.ndarray:
    if len(indices) <= maximum:
        return indices
    positions = np.linspace(0, len(indices) - 1, num=maximum, dtype=np.int64)
    return indices[positions]


def _normal_sign_deterministic(normal: np.ndarray) -> np.ndarray:
    normal = np.asarray(normal, dtype=np.float64)
    largest = int(np.argmax(np.abs(normal)))
    if normal[largest] < 0.0:
        normal = -normal
    return normal


def _estimate_normals_and_neighbours(
    points: np.ndarray,
    *,
    neighbour_count: int,
    radius: float,
) -> tuple[np.ndarray, list[np.ndarray]]:
    tree = cKDTree(points)
    k = min(len(points), neighbour_count + 1)
    distances, indices = tree.query(points, k=k, distance_upper_bound=radius, workers=1)
    if k == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    normals = np.zeros((len(points), 3), dtype=np.float64)
    neighbours: list[np.ndarray] = []
    for i in range(len(points)):
        row_idx = np.asarray(indices[i]).reshape(-1)
        row_dist = np.asarray(distances[i]).reshape(-1)
        valid = (row_idx < len(points)) & np.isfinite(row_dist) & (row_idx != i)
        pairs = sorted(
            ((float(d), int(j)) for d, j in zip(row_dist[valid], row_idx[valid], strict=True)),
            key=lambda pair: (pair[0], pair[1]),
        )
        neigh = np.asarray([j for _, j in pairs[:neighbour_count]], dtype=np.int64)
        neighbours.append(neigh)
        if len(neigh) < 3:
            normals[i] = np.array([0.0, 0.0, 1.0])
            continue
        offsets = points[neigh] - points[i]
        covariance = offsets.T @ offsets / max(1, len(offsets))
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        normal = eigenvectors[:, int(np.argmin(eigenvalues))]
        length = float(np.linalg.norm(normal))
        if not isfinite(length) or length <= 1e-12:
            normal = np.array([0.0, 0.0, 1.0])
        else:
            normal = _normal_sign_deterministic(normal / length)
        normals[i] = normal

    # Propagate sign consistency over the local-neighbour graph.  The absolute
    # orientation remains arbitrary, but adjacent normals no longer flip.
    visited = np.zeros(len(points), dtype=bool)
    for seed in range(len(points)):
        if visited[seed]:
            continue
        visited[seed] = True
        stack = [seed]
        while stack:
            current = stack.pop()
            for neighbour in neighbours[current]:
                j = int(neighbour)
                if visited[j]:
                    continue
                if float(np.dot(normals[current], normals[j])) < 0.0:
                    normals[j] *= -1.0
                visited[j] = True
                stack.append(j)
    lengths = np.linalg.norm(normals, axis=1)
    normals /= np.maximum(lengths[:, None], 1e-12)
    return normals, neighbours


def _tangent_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axes = np.eye(3, dtype=np.float64)
    reference = axes[int(np.argmin(np.abs(axes @ normal)))]
    tangent1 = np.cross(normal, reference)
    tangent1 /= max(float(np.linalg.norm(tangent1)), 1e-12)
    tangent2 = np.cross(normal, tangent1)
    tangent2 /= max(float(np.linalg.norm(tangent2)), 1e-12)
    return tangent1, tangent2


def _triangle_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return 0.5 * float(np.linalg.norm(np.cross(b - a, c - a)))


def _build_candidate_faces(
    points: np.ndarray,
    normals: np.ndarray,
    neighbours: list[np.ndarray],
    *,
    voxel_size: float,
    policy: SurfaceMeshingPolicy,
) -> tuple[np.ndarray, dict[str, int]]:
    max_edge = voxel_size * policy.maximum_edge_length_multiplier
    minimum_area = voxel_size * voxel_size * policy.minimum_triangle_area_multiplier
    normal_cos = cos(radians(policy.maximum_normal_deviation_degrees))
    face_keys: set[tuple[int, int, int]] = set()
    oriented_faces: list[tuple[int, int, int]] = []
    rejected_gap = 0
    rejected_normal = 0
    rejected_degenerate = 0
    rejected_duplicate = 0
    dropped_capacity = 0
    candidate_faces = 0

    for i, neigh in enumerate(neighbours):
        if len(neigh) < 2:
            continue
        normal = normals[i]
        tangent1, tangent2 = _tangent_basis(normal)
        offsets = points[neigh] - points[i]
        x = offsets @ tangent1
        y = offsets @ tangent2
        angles = np.arctan2(y, x)
        order = np.lexsort((neigh, angles))
        ordered = neigh[order]
        ordered_angles = angles[order]
        count = len(ordered)
        for position in range(count):
            j = int(ordered[position])
            k = int(ordered[(position + 1) % count])
            if j == k or j == i or k == i:
                continue
            angle_a = float(ordered_angles[position])
            angle_b = float(ordered_angles[(position + 1) % count])
            gap = angle_b - angle_a
            if position == count - 1:
                gap += 2.0 * np.pi
            if np.degrees(gap) > policy.maximum_fan_angle_degrees:
                rejected_gap += 1
                continue
            candidate_faces += 1
            a, b, c = points[i], points[j], points[k]
            if (
                np.linalg.norm(a - b) > max_edge
                or np.linalg.norm(a - c) > max_edge
                or np.linalg.norm(b - c) > max_edge
            ):
                rejected_gap += 1
                continue
            if (
                abs(float(np.dot(normals[i], normals[j]))) < normal_cos
                or abs(float(np.dot(normals[i], normals[k]))) < normal_cos
            ):
                rejected_normal += 1
                continue
            area = _triangle_area(a, b, c)
            if not isfinite(area) or area < minimum_area:
                rejected_degenerate += 1
                continue
            key = tuple(sorted((i, j, k)))
            if key in face_keys:
                rejected_duplicate += 1
                continue
            if len(oriented_faces) >= policy.max_faces:
                dropped_capacity += 1
                continue
            face_normal = np.cross(b - a, c - a)
            if float(np.dot(face_normal, normal)) < 0.0:
                face = (i, k, j)
            else:
                face = (i, j, k)
            face_keys.add(key)
            oriented_faces.append(face)

    faces = np.asarray(oriented_faces, dtype=np.uint32)
    if faces.size == 0:
        faces = np.empty((0, 3), dtype=np.uint32)
    return faces, {
        "candidate_faces": candidate_faces,
        "rejected_gap_faces": rejected_gap,
        "rejected_normal_faces": rejected_normal,
        "rejected_degenerate_faces": rejected_degenerate,
        "rejected_duplicate_faces": rejected_duplicate,
        "dropped_faces_capacity": dropped_capacity,
    }


def _face_components(faces: np.ndarray, vertex_count: int) -> tuple[np.ndarray, int, np.ndarray]:
    parent = np.arange(vertex_count, dtype=np.int64)
    rank = np.zeros(vertex_count, dtype=np.uint8)

    def find(value: int) -> int:
        root = value
        while parent[root] != root:
            root = int(parent[root])
        while parent[value] != value:
            nxt = int(parent[value])
            parent[value] = root
            value = nxt
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    for a, b, c in faces.astype(np.int64, copy=False):
        union(int(a), int(b))
        union(int(a), int(c))
    roots = np.asarray([find(int(face[0])) for face in faces], dtype=np.int64)
    unique_roots, inverse = np.unique(roots, return_inverse=True)
    counts = np.bincount(inverse, minlength=len(unique_roots)).astype(np.int64)
    return inverse.astype(np.uint32), int(len(unique_roots)), counts


def _cleanup_components(
    faces: np.ndarray,
    *,
    vertex_count: int,
    minimum_component_faces: int,
) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    if len(faces) == 0:
        return faces, np.empty((0,), np.uint32), 0, 0, 0
    component_ids, component_count, counts = _face_components(faces, vertex_count)
    keep_components = np.flatnonzero(counts >= minimum_component_faces)
    if len(keep_components) == 0:
        keep_components = np.asarray([int(np.argmax(counts))], dtype=np.int64)
    keep_mask = np.isin(component_ids, keep_components)
    removed_faces = int(np.count_nonzero(~keep_mask))
    retained_faces = faces[keep_mask]
    retained_old_ids = component_ids[keep_mask]
    remap_component = {int(old): new for new, old in enumerate(sorted(set(map(int, retained_old_ids))))}
    retained_component_ids = np.asarray(
        [remap_component[int(old)] for old in retained_old_ids], dtype=np.uint32
    )
    return (
        retained_faces,
        retained_component_ids,
        component_count,
        len(remap_component),
        removed_faces,
    )


def _compact_mesh_vertices(
    points: np.ndarray,
    colors: np.ndarray,
    normals: np.ndarray,
    quality_scores: np.ndarray,
    quality_codes: np.ndarray,
    uncertainty: np.ndarray,
    source_indices: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    used = np.unique(faces.reshape(-1)).astype(np.int64)
    mapping = np.full(len(points), -1, dtype=np.int64)
    mapping[used] = np.arange(len(used), dtype=np.int64)
    compact_faces = mapping[faces.astype(np.int64)].astype(np.uint32)
    removed = int(len(points) - len(used))
    return (
        points[used],
        colors[used],
        normals[used],
        quality_scores[used],
        quality_codes[used],
        uncertainty[used],
        source_indices[used],
        compact_faces,
        removed,
    )


def _unique_edges(faces: np.ndarray) -> np.ndarray:
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])).astype(np.uint32)
    edges.sort(axis=1)
    return np.unique(edges, axis=0)


def generate_quality_aware_surface_mesh(
    dense_result: DenseFusionResult,
    quality_report: DenseGeometryQualityReport,
    *,
    policy: SurfaceMeshingPolicy | None = None,
) -> SurfaceMeshResult:
    """Generate a bounded triangle mesh from quality-accepted dense points.

    Vertex positions are copied from the dense cloud; no geometric smoothing or
    scale modification is performed.  Low/rejected quality points are excluded,
    local holes are not bridged beyond the edge/radius policy, and small
    disconnected components are removed conservatively.
    """

    if not isinstance(dense_result, DenseFusionResult):
        raise TypeError("dense_result must be DenseFusionResult.")
    if not isinstance(quality_report, DenseGeometryQualityReport):
        raise TypeError("quality_report must be DenseGeometryQualityReport.")
    if dense_result.session_id != quality_report.session_id:
        raise ReconstructionValidationError("Dense result and quality report session IDs differ.")
    if dense_result.scene_profile is not quality_report.scene_profile:
        raise ReconstructionValidationError("Dense result and quality report scene profiles differ.")
    for name in ("camera_profile_id", "scale_profile_id", "pose_convention", "reconstruction_frame_id"):
        if getattr(dense_result, name) != getattr(quality_report, name):
            raise ReconstructionValidationError(f"Dense result and quality report {name} differ.")
    if dense_result.metric_geometry_status is not quality_report.metric_geometry_status:
        raise ReconstructionValidationError("Dense result and quality report metric semantics differ.")
    if dense_result.certified_metrology or quality_report.certified_metrology:
        raise ReconstructionValidationError("Phase 8.7A refuses certified-metrology input claims.")
    if dense_result.pose_convention != SURFACE_MESH_POSE_CONVENTION:
        raise ReconstructionValidationError("Unsupported dense-result pose convention for meshing.")

    policy = policy or default_surface_meshing_policy(dense_result.scene_profile)
    if not isinstance(policy, SurfaceMeshingPolicy):
        raise TypeError("policy must be SurfaceMeshingPolicy.")
    if policy.scene_profile is not dense_result.scene_profile:
        raise ReconstructionValidationError("Meshing policy scene profile must match dense result.")

    cloud = dense_result.point_cloud
    point_quality = quality_report.point_quality
    if point_quality.point_count != len(cloud.points_reconstruction):
        raise ReconstructionValidationError("Point-quality count must match dense point cloud.")

    quality_mask = point_quality.accepted_mask & (
        point_quality.quality_scores >= policy.minimum_vertex_quality_score
    )
    accepted_indices = np.flatnonzero(quality_mask).astype(np.int64)
    if len(accepted_indices) < 4:
        raise ReconstructionValidationError(
            "Insufficient quality-accepted points for surface reconstruction."
        )
    selected_indices = _deterministic_subsample(accepted_indices, policy.max_vertices)
    points = cloud.points_reconstruction[selected_indices].astype(np.float64, copy=True)
    colors = cloud.colors_rgb[selected_indices].astype(np.uint8, copy=True)
    scores = point_quality.quality_scores[selected_indices].astype(np.float32, copy=True)
    codes = point_quality.quality_codes[selected_indices].astype(np.uint8, copy=True)
    uncertainty = point_quality.local_uncertainty_reconstruction_units[selected_indices].astype(
        np.float32, copy=True
    )
    voxel_size = float(cloud.voxel_size_reconstruction_units)
    radius = voxel_size * policy.neighbourhood_radius_multiplier
    normals, neighbours = _estimate_normals_and_neighbours(
        points,
        neighbour_count=policy.neighbour_count,
        radius=radius,
    )
    faces, counters = _build_candidate_faces(
        points,
        normals,
        neighbours,
        voxel_size=voxel_size,
        policy=policy,
    )
    if len(faces) == 0:
        raise ReconstructionValidationError(
            "Surface reconstruction produced no triangles under the current quality/gap policy."
        )
    faces, component_ids, components_before, components_retained, removed_component_faces = _cleanup_components(
        faces,
        vertex_count=len(points),
        minimum_component_faces=policy.minimum_component_faces,
    )
    if len(faces) == 0:
        raise ReconstructionValidationError("Surface cleanup removed every triangle.")

    (
        points,
        colors,
        normals,
        scores,
        codes,
        uncertainty,
        selected_indices,
        faces,
        removed_unreferenced,
    ) = _compact_mesh_vertices(
        points,
        colors,
        normals,
        scores,
        codes,
        uncertainty,
        selected_indices,
        faces,
    )

    mesh = SurfaceMesh(
        vertices_reconstruction=points,
        faces=faces,
        colors_rgb=colors,
        normals_reconstruction=normals,
        vertex_quality_scores=scores,
        vertex_quality_codes=codes,
        vertex_uncertainty_reconstruction_units=uncertainty,
        source_point_indices=selected_indices.astype(np.uint32),
        face_component_ids=component_ids,
    )
    statistics = SurfaceMeshStatistics(
        input_points=len(cloud.points_reconstruction),
        quality_accepted_points=len(accepted_indices),
        selected_points_before_cleanup=len(_deterministic_subsample(accepted_indices, policy.max_vertices)),
        retained_vertices=mesh.vertex_count,
        candidate_faces=counters["candidate_faces"],
        retained_faces=mesh.face_count,
        rejected_gap_faces=counters["rejected_gap_faces"],
        rejected_normal_faces=counters["rejected_normal_faces"],
        rejected_degenerate_faces=counters["rejected_degenerate_faces"],
        rejected_duplicate_faces=counters["rejected_duplicate_faces"],
        dropped_faces_capacity=counters["dropped_faces_capacity"],
        components_before_cleanup=components_before,
        components_retained=components_retained,
        removed_small_component_faces=removed_component_faces,
        removed_unreferenced_vertices=removed_unreferenced,
    )
    return SurfaceMeshResult(
        session_id=dense_result.session_id,
        scene_profile=dense_result.scene_profile,
        camera_profile_id=dense_result.camera_profile_id,
        scale_profile_id=dense_result.scale_profile_id,
        metric_geometry_status=dense_result.metric_geometry_status,
        pose_convention=dense_result.pose_convention,
        reconstruction_frame_id=dense_result.reconstruction_frame_id,
        scale_status=dense_result.scale_status,
        world_alignment_status=dense_result.world_alignment_status,
        unit_to_metre=dense_result.unit_to_metre,
        relative_scale_uncertainty=quality_report.summary.relative_scale_uncertainty,
        certified_metrology=False,
        source_frame_ids=dense_result.source_frame_ids,
        source_pose_layers=dense_result.source_pose_layers,
        source_quality_schema_version=quality_report.schema_version,
        quality_limiters=quality_report.quality_limiters,
        policy=policy,
        mesh=mesh,
        statistics=statistics,
    )


def write_surface_mesh_obj(result: SurfaceMeshResult, path: str | Path) -> Path:
    """Validation/export helper for the Phase 8.7A artifact.

    Productized export remains a Phase 8.9 concern.  This helper intentionally
    writes a simple OBJ containing geometry, RGB vertex values, normals, and
    triangular faces so target-machine validation can inspect the mesh without
    enabling the Reconstruction UI.
    """

    if not isinstance(result, SurfaceMeshResult):
        raise TypeError("result must be SurfaceMeshResult.")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    mesh = result.mesh
    lines = [
        "# EIA TECH Phase 8.7A validation mesh",
        f"# metric_geometry_status={result.metric_geometry_status.value}",
        f"# unit_to_metre={result.unit_to_metre}",
        "# certified_metrology=false",
    ]
    for point, color in zip(mesh.vertices_reconstruction, mesh.colors_rgb, strict=True):
        rgb = color.astype(np.float64) / 255.0
        lines.append(
            f"v {point[0]:.12g} {point[1]:.12g} {point[2]:.12g} "
            f"{rgb[0]:.8f} {rgb[1]:.8f} {rgb[2]:.8f}"
        )
    for normal in mesh.normals_reconstruction:
        lines.append(f"vn {normal[0]:.9g} {normal[1]:.9g} {normal[2]:.9g}")
    for a, b, c in mesh.faces.astype(np.int64, copy=False):
        ai, bi, ci = int(a) + 1, int(b) + 1, int(c) + 1
        lines.append(f"f {ai}//{ai} {bi}//{bi} {ci}//{ci}")
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination

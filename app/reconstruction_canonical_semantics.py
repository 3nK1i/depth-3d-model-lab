"""Persistent semantic-object binding for the canonical Reconstruction viewer.

Phase 9.3D consumes only semantic 3D evidence that is already persisted by the
reconstruction pipeline.  It binds that evidence to the integrity-verified
canonical viewer mesh without rebuilding semantics, rerunning perception, or
promoting semantic membership to canonical/metric/measurement/QC authority.

Legacy detailed semantic snapshots that predate persisted vertex/face indices
remain valid semantic context.  They can still provide labels, centroids and
support bounds, but no canonical-surface highlight is fabricated for them.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping, Sequence

import numpy as np

from app.reconstruction_canonical_geometry import CanonicalViewerMesh
from app.reconstruction_visualization import reconstruction_points_to_viewer
from core.contracts.validation import metadata_as_dict
from core.reconstruction.orchestration import (
    ReconstructionArtifactAuthority,
    ReconstructionArtifactKind,
)
from core.reconstruction.persistent_scene import (
    PersistentReconstructionScene,
    PersistentSceneArtifactRecord,
)

MAX_CANONICAL_SEMANTIC_OBJECTS = 256


@dataclass(frozen=True, slots=True)
class CanonicalSemanticObjectBinding:
    """One persisted semantic object bound to the loaded canonical surface."""

    object_id: str
    class_id: int
    class_name: str
    semantic_role: str
    confidence: float
    centroid_viewer: np.ndarray
    vertex_indices: np.ndarray
    face_indices: np.ndarray
    surface_vertices_viewer: np.ndarray
    surface_faces: np.ndarray
    observation_ids: tuple[str, ...]
    frame_ids: tuple[str, ...]
    mask_ids: tuple[str, ...]
    track_ids: tuple[str, ...]
    camera_profile_ids: tuple[str, ...]
    pose_layers: tuple[str, ...]
    certified_metrology: bool = False

    def __post_init__(self) -> None:
        object_id = str(self.object_id).strip()
        class_name = str(self.class_name).strip()
        role = str(self.semantic_role).strip()
        if not object_id or not class_name or not role:
            raise ValueError("Canonical semantic object identity, class, and role are required.")
        if isinstance(self.class_id, bool) or int(self.class_id) < 0:
            raise ValueError("Canonical semantic class_id must be a non-negative integer.")
        confidence = float(self.confidence)
        if not isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("Canonical semantic confidence must be finite in [0, 1].")
        if bool(self.certified_metrology):
            raise ValueError("Phase 9.3D semantic presentation cannot claim certified metrology.")

        centroid = np.asarray(self.centroid_viewer, dtype=np.float32)
        vertices = np.asarray(self.vertex_indices, dtype=np.uint32)
        faces = np.asarray(self.face_indices, dtype=np.uint32)
        surface_vertices = np.asarray(self.surface_vertices_viewer, dtype=np.float32)
        surface_faces = np.asarray(self.surface_faces, dtype=np.uint32)
        if centroid.shape != (3,) or not np.isfinite(centroid).all():
            raise ValueError("centroid_viewer must be a finite 3-vector.")
        if vertices.ndim != 1 or faces.ndim != 1:
            raise ValueError("Semantic vertex_indices and face_indices must be 1D.")
        if len(vertices) != len(np.unique(vertices)) or len(faces) != len(np.unique(faces)):
            raise ValueError("Semantic support indices must be unique.")
        if surface_vertices.size == 0:
            surface_vertices = np.zeros((0, 3), dtype=np.float32)
        if surface_faces.size == 0:
            surface_faces = np.zeros((0, 3), dtype=np.uint32)
        if surface_vertices.ndim != 2 or surface_vertices.shape[1:] != (3,):
            raise ValueError("surface_vertices_viewer must have shape (N, 3).")
        if surface_faces.ndim != 2 or surface_faces.shape[1:] != (3,):
            raise ValueError("surface_faces must have shape (M, 3).")
        if surface_faces.shape[0] and not surface_vertices.shape[0]:
            raise ValueError("Semantic surface faces require surface vertices.")
        if surface_faces.shape[0] and np.any(surface_faces >= surface_vertices.shape[0]):
            raise ValueError("Semantic surface faces reference missing local vertices.")
        if not np.isfinite(surface_vertices).all():
            raise ValueError("Semantic surface vertices must be finite.")

        for name, value in (
            ("centroid_viewer", centroid),
            ("vertex_indices", vertices),
            ("face_indices", faces),
            ("surface_vertices_viewer", surface_vertices),
            ("surface_faces", surface_faces),
        ):
            array = np.ascontiguousarray(value)
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        object.__setattr__(self, "object_id", object_id)
        object.__setattr__(self, "class_id", int(self.class_id))
        object.__setattr__(self, "class_name", class_name)
        object.__setattr__(self, "semantic_role", role)
        object.__setattr__(self, "confidence", confidence)
        for name in (
            "observation_ids",
            "frame_ids",
            "mask_ids",
            "track_ids",
            "camera_profile_ids",
            "pose_layers",
        ):
            object.__setattr__(self, name, _identity_tuple(getattr(self, name)))
        object.__setattr__(self, "certified_metrology", False)

    @property
    def surface_membership_available(self) -> bool:
        return bool(self.vertex_indices.size or self.face_indices.size)

    @property
    def surface_highlight_available(self) -> bool:
        return bool(self.surface_faces.shape[0])

    @property
    def label_text(self) -> str:
        return self.class_name

    @property
    def provenance_text(self) -> str:
        parts = [
            f"{len(self.observation_ids)} observation(s)",
            f"{len(self.frame_ids)} frame(s)",
            f"{len(self.mask_ids)} mask(s)",
        ]
        if self.track_ids:
            parts.append(f"{len(self.track_ids)} track(s)")
        if self.camera_profile_ids:
            parts.append(f"{len(self.camera_profile_ids)} camera profile(s)")
        if self.pose_layers:
            parts.append("pose " + ", ".join(self.pose_layers[:3]))
        return " • ".join(parts)


@dataclass(frozen=True, slots=True)
class CanonicalSemanticScenePresentation:
    """Bounded 9.3D semantic context for one persistent canonical scene."""

    scene_id: str
    session_id: str
    semantic_artifact_id: str
    semantic_provider_id: str
    canonical_artifact_id: str
    reconstruction_frame_id: str
    metric_geometry_status: str
    source_surface_schema_version: int | None
    total_object_count: int
    objects: tuple[CanonicalSemanticObjectBinding, ...]
    detailed_snapshot: bool
    certified_metrology: bool = False

    def __post_init__(self) -> None:
        for name in (
            "scene_id",
            "session_id",
            "semantic_artifact_id",
            "semantic_provider_id",
            "canonical_artifact_id",
        ):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must be non-empty.")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "reconstruction_frame_id", str(self.reconstruction_frame_id).strip())
        status = str(self.metric_geometry_status).strip().lower() or "unknown"
        if status not in {"arbitrary", "provisional_metric", "validated_metric", "unknown"}:
            raise ValueError(f"Unsupported semantic metric geometry status: {status!r}")
        object.__setattr__(self, "metric_geometry_status", status)
        source_schema = self.source_surface_schema_version
        if source_schema is not None:
            if isinstance(source_schema, bool) or int(source_schema) <= 0:
                raise ValueError("source_surface_schema_version must be positive when present.")
            source_schema = int(source_schema)
        object.__setattr__(self, "source_surface_schema_version", source_schema)
        total = _nonnegative_int(self.total_object_count, "total_object_count")
        objects = tuple(self.objects)
        if len(objects) > MAX_CANONICAL_SEMANTIC_OBJECTS:
            raise ValueError("Canonical semantic presentation exceeds the bounded object limit.")
        if total < len(objects):
            raise ValueError("total_object_count cannot be smaller than displayed objects.")
        if len({item.object_id for item in objects}) != len(objects):
            raise ValueError("Canonical semantic object IDs must be unique.")
        if bool(self.certified_metrology):
            raise ValueError("Phase 9.3D semantic presentation cannot claim certified metrology.")
        object.__setattr__(self, "total_object_count", total)
        object.__setattr__(self, "objects", objects)
        object.__setattr__(self, "detailed_snapshot", bool(self.detailed_snapshot))
        object.__setattr__(self, "certified_metrology", False)

    @property
    def surface_bound_object_count(self) -> int:
        return sum(item.surface_membership_available for item in self.objects)

    @property
    def surface_highlight_object_count(self) -> int:
        return sum(item.surface_highlight_available for item in self.objects)

    @property
    def object_ids(self) -> tuple[str, ...]:
        return tuple(item.object_id for item in self.objects)

    @property
    def binding_text(self) -> str:
        if not self.objects:
            return "No detailed semantic objects persisted"
        if not self.surface_bound_object_count:
            return "3D semantic objects persisted • exact canonical surface membership unavailable"
        return (
            f"{self.surface_bound_object_count}/{len(self.objects)} objects surface-bound • "
            f"{self.surface_highlight_object_count} highlightable"
        )

    @property
    def authority_text(self) -> str:
        return "Semantic membership only • canonical geometry authority unchanged"


def _semantic_artifact(scene: PersistentReconstructionScene) -> PersistentSceneArtifactRecord | None:
    artifact = next(
        (item for item in scene.artifacts if item.kind is ReconstructionArtifactKind.SEMANTIC_GEOMETRY),
        None,
    )
    if artifact is None:
        return None
    if artifact.authority is not ReconstructionArtifactAuthority.SEMANTIC_CONTEXT:
        raise ValueError("Persistent semantic artifact lost semantic_context authority.")
    if artifact.canonical_geometry_authoritative:
        raise ValueError("Persistent semantic artifact cannot be canonical geometry authority.")
    return artifact


def load_persistent_canonical_semantics(
    scene: PersistentReconstructionScene,
    canonical_mesh: CanonicalViewerMesh,
    *,
    max_objects: int = MAX_CANONICAL_SEMANTIC_OBJECTS,
) -> CanonicalSemanticScenePresentation | None:
    """Bind an existing persisted semantic snapshot to its canonical viewer mesh.

    No 2D mask is projected here and no semantic inference is executed.  Exact
    surface highlighting is enabled only when the persisted object already
    contains vertex/face membership that validates against the canonical mesh.
    """

    if not isinstance(scene, PersistentReconstructionScene):
        raise TypeError("scene must be PersistentReconstructionScene.")
    if not isinstance(canonical_mesh, CanonicalViewerMesh):
        raise TypeError("canonical_mesh must be CanonicalViewerMesh.")
    if canonical_mesh.scene_id != scene.scene_id:
        raise ValueError("Canonical viewer mesh scene identity does not match the persistent scene.")
    if canonical_mesh.artifact_id != scene.canonical_geometry_artifact_id:
        raise ValueError("Canonical viewer mesh is not bound to the scene's canonical artifact.")
    max_objects = int(max_objects)
    if max_objects < 1 or max_objects > MAX_CANONICAL_SEMANTIC_OBJECTS:
        raise ValueError(
            f"max_objects must be within [1, {MAX_CANONICAL_SEMANTIC_OBJECTS}]."
        )

    artifact = _semantic_artifact(scene)
    if artifact is None:
        return None
    snapshot = metadata_as_dict(artifact.payload_snapshot)  # type: ignore[arg-type]
    if not snapshot or snapshot.get("objects") is None:
        return None
    if bool(snapshot.get("certified_metrology", False)):
        raise ValueError("Persisted semantic geometry cannot claim certified metrology.")
    if bool(snapshot.get("canonical_geometry_authoritative", False)):
        raise ValueError("Persisted semantic geometry cannot claim canonical geometry authority.")

    session_id = str(snapshot.get("session_id", "")).strip()
    if not session_id:
        raise ValueError("Detailed persistent semantic snapshot is missing session_id.")
    if session_id != scene.session_id:
        raise ValueError("Persistent semantic snapshot session identity does not match the scene.")

    raw_objects = snapshot.get("objects", ())
    if isinstance(raw_objects, Mapping):
        if raw_objects:
            raise ValueError("Persistent semantic snapshot objects must be a sequence.")
        raw_objects = ()
    if not isinstance(raw_objects, Sequence) or isinstance(raw_objects, (str, bytes, bytearray)):
        raise ValueError("Persistent semantic snapshot objects must be a sequence.")

    statistics = snapshot.get("statistics", {})
    if not isinstance(statistics, Mapping):
        statistics = {}
    total_object_count = _coerce_nonnegative(
        statistics.get("persistent_object_count"), fallback=len(raw_objects)
    )
    objects = tuple(
        _bind_object(raw, canonical_mesh)
        for raw in list(raw_objects)[:max_objects]
    )
    return CanonicalSemanticScenePresentation(
        scene_id=scene.scene_id,
        session_id=session_id,
        semantic_artifact_id=artifact.artifact_id,
        semantic_provider_id=artifact.provider_id,
        canonical_artifact_id=canonical_mesh.artifact_id,
        reconstruction_frame_id=str(snapshot.get("reconstruction_frame_id", "")),
        metric_geometry_status=str(snapshot.get("metric_geometry_status", "unknown")),
        source_surface_schema_version=_optional_positive_int(
            snapshot.get("source_surface_schema_version")
        ),
        total_object_count=max(total_object_count, len(raw_objects), len(objects)),
        objects=objects,
        detailed_snapshot=True,
        certified_metrology=False,
    )


def _bind_object(raw_value: Any, canonical_mesh: CanonicalViewerMesh) -> CanonicalSemanticObjectBinding:
    if not isinstance(raw_value, Mapping):
        raise ValueError("Each persisted semantic object must be a mapping.")
    raw = dict(raw_value)
    if bool(raw.get("certified_metrology", False)):
        raise ValueError("Persisted semantic object cannot claim certified metrology.")
    if bool(raw.get("canonical_geometry_authoritative", False)):
        raise ValueError("Persisted semantic object cannot claim canonical geometry authority.")

    has_vertex_indices = "vertex_indices" in raw
    has_face_indices = "face_indices" in raw
    if has_vertex_indices != has_face_indices:
        raise ValueError("Semantic surface membership must persist vertex_indices and face_indices together.")

    if has_vertex_indices:
        vertex_indices = _index_array(raw.get("vertex_indices"), "vertex_indices")
        face_indices = _index_array(raw.get("face_indices"), "face_indices", allow_empty=True)
        if vertex_indices.size == 0:
            raise ValueError("Persisted semantic vertex_indices cannot be empty when membership is present.")
        if np.any(vertex_indices >= canonical_mesh.vertex_count):
            raise ValueError("Persisted semantic vertex_indices exceed the canonical mesh vertex count.")
        if face_indices.size and np.any(face_indices >= canonical_mesh.face_count):
            raise ValueError("Persisted semantic face_indices exceed the canonical mesh face count.")
        expected_vertices = _optional_count(raw.get("vertex_count"))
        expected_faces = _optional_count(raw.get("face_count"))
        if expected_vertices is not None and expected_vertices != int(vertex_indices.size):
            raise ValueError("Persisted semantic vertex_count disagrees with vertex_indices.")
        if expected_faces is not None and expected_faces != int(face_indices.size):
            raise ValueError("Persisted semantic face_count disagrees with face_indices.")
    else:
        # Legacy Phase 8 snapshots: valid 3D semantic context, but exact surface
        # membership was not serialized.  Never infer it from bounds/centroids.
        vertex_indices = np.zeros((0,), dtype=np.uint32)
        face_indices = np.zeros((0,), dtype=np.uint32)

    local_vertices, local_faces = _compact_surface_membership(
        canonical_mesh,
        face_indices,
    )
    centroid = _vector3(raw.get("centroid_reconstruction"), "centroid_reconstruction")
    centroid_viewer = reconstruction_points_to_viewer(centroid.reshape(1, 3))[0].astype(
        np.float32, copy=False
    )
    return CanonicalSemanticObjectBinding(
        object_id=str(raw.get("object_id", "")),
        class_id=_nonnegative_int(raw.get("class_id", 0), "class_id"),
        class_name=str(raw.get("class_name", "")),
        semantic_role=str(raw.get("semantic_role", "")),
        confidence=float(raw.get("confidence", 0.0)),
        centroid_viewer=centroid_viewer,
        vertex_indices=vertex_indices,
        face_indices=face_indices,
        surface_vertices_viewer=local_vertices,
        surface_faces=local_faces,
        observation_ids=_sequence_of_strings(raw.get("observation_ids", ())),
        frame_ids=_sequence_of_strings(raw.get("frame_ids", ())),
        mask_ids=_sequence_of_strings(raw.get("mask_ids", ())),
        track_ids=_sequence_of_strings(raw.get("track_ids", ())),
        camera_profile_ids=_sequence_of_strings(raw.get("camera_profile_ids", ())),
        pose_layers=_sequence_of_strings(raw.get("pose_layers", ())),
        certified_metrology=False,
    )


def _compact_surface_membership(
    canonical_mesh: CanonicalViewerMesh,
    face_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if face_indices.size == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.uint32),
        )
    global_faces = canonical_mesh.faces[face_indices.astype(np.int64, copy=False)]
    used_vertices, inverse = np.unique(global_faces.reshape(-1), return_inverse=True)
    local_vertices = canonical_mesh.vertices_viewer[used_vertices.astype(np.int64, copy=False)]
    local_faces = inverse.reshape((-1, 3)).astype(np.uint32, copy=False)
    return (
        np.ascontiguousarray(local_vertices, dtype=np.float32),
        np.ascontiguousarray(local_faces, dtype=np.uint32),
    )


def _index_array(value: Any, field_name: str, *, allow_empty: bool = False) -> np.ndarray:
    if isinstance(value, Mapping) and not value:
        value = ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{field_name} must be a sequence of non-negative integers.")
    values: list[int] = []
    for item in value:
        if isinstance(item, bool):
            raise ValueError(f"{field_name} cannot contain booleans.")
        try:
            numeric = int(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} must contain integers.") from exc
        if numeric < 0:
            raise ValueError(f"{field_name} cannot contain negative indices.")
        if isinstance(item, float) and not float(item).is_integer():
            raise ValueError(f"{field_name} cannot contain fractional indices.")
        values.append(numeric)
    if not values and not allow_empty:
        raise ValueError(f"{field_name} must not be empty.")
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must contain unique indices.")
    return np.asarray(values, dtype=np.uint32)


def _sequence_of_strings(value: Any) -> tuple[str, ...]:
    if isinstance(value, Mapping) and not value:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    return _identity_tuple(value)


def _identity_tuple(value: Sequence[Any]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))


def _vector3(value: Any, field_name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{field_name} must be a finite 3-vector.")
    result = np.ascontiguousarray(array)
    result.setflags(write=False)
    return result


def _nonnegative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a non-negative integer.")
    numeric = int(value)
    if numeric < 0:
        raise ValueError(f"{field_name} must be a non-negative integer.")
    return numeric


def _optional_count(value: Any) -> int | None:
    if value is None:
        return None
    return _nonnegative_int(value, "count")


def _coerce_nonnegative(value: Any, *, fallback: int) -> int:
    try:
        return _nonnegative_int(value, "count")
    except (TypeError, ValueError):
        return max(0, int(fallback))


def _optional_positive_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("source_surface_schema_version must be positive when present.")
    numeric = int(value)
    if numeric <= 0:
        raise ValueError("source_surface_schema_version must be positive when present.")
    return numeric


__all__ = [
    "MAX_CANONICAL_SEMANTIC_OBJECTS",
    "CanonicalSemanticObjectBinding",
    "CanonicalSemanticScenePresentation",
    "load_persistent_canonical_semantics",
]

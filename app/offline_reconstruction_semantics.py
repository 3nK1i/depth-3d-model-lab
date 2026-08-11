"""Presentation-only semantic/object adapter for Phase 8.8D5.

The reconstruction engine already owns semantic geometry and its authority
contracts.  D5 does not rebuild semantics, run perception, make QC decisions,
or infer measurements.  This module converts an existing Phase 8.7B semantic
payload (or a persisted semantic artifact snapshot) into a bounded,
UI-friendly representation for object selection and viewport highlighting.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping, Sequence

import numpy as np

from app.reconstruction_visualization import reconstruction_points_to_viewer

MAX_SEMANTIC_OBJECT_PRESENTATIONS = 256


@dataclass(frozen=True, slots=True)
class OfflineSemanticObjectPresentation:
    object_id: str
    class_name: str
    semantic_role: str
    confidence: float
    vertex_count: int
    face_count: int
    observation_count: int
    mean_vertex_quality: float
    median_vertex_uncertainty_reconstruction_units: float | None
    maximum_vertex_uncertainty_reconstruction_units: float | None
    centroid_reconstruction: np.ndarray
    bounds_min_reconstruction: np.ndarray
    bounds_max_reconstruction: np.ndarray
    orientation_reliable: bool
    metric_geometry_status: str
    certified_metrology: bool = False

    def __post_init__(self) -> None:
        object_id = str(self.object_id).strip()
        class_name = str(self.class_name).strip()
        role = str(self.semantic_role).strip()
        if not object_id or not class_name or not role:
            raise ValueError("Semantic object identity, class, and role must be non-empty.")
        confidence = _probability(self.confidence, "confidence")
        mean_quality = _probability(self.mean_vertex_quality, "mean_vertex_quality")
        vertex_count = _nonnegative_int(self.vertex_count, "vertex_count")
        face_count = _nonnegative_int(self.face_count, "face_count")
        observation_count = _nonnegative_int(self.observation_count, "observation_count")
        centroid = _vector3(self.centroid_reconstruction, "centroid_reconstruction")
        lower = _vector3(self.bounds_min_reconstruction, "bounds_min_reconstruction")
        upper = _vector3(self.bounds_max_reconstruction, "bounds_max_reconstruction")
        if np.any(upper < lower):
            raise ValueError("Semantic object bounds are invalid.")
        if np.any(centroid < lower - 1e-8) or np.any(centroid > upper + 1e-8):
            raise ValueError("Semantic object centroid must lie inside its bounds.")
        status = str(self.metric_geometry_status).strip().lower()
        if status not in {"arbitrary", "provisional_metric", "validated_metric"}:
            raise ValueError(f"Unsupported semantic metric geometry status: {status!r}")
        if bool(self.certified_metrology):
            raise ValueError("D5 semantic presentation cannot claim certified metrology.")
        object.__setattr__(self, "object_id", object_id)
        object.__setattr__(self, "class_name", class_name)
        object.__setattr__(self, "semantic_role", role)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "mean_vertex_quality", mean_quality)
        object.__setattr__(self, "vertex_count", vertex_count)
        object.__setattr__(self, "face_count", face_count)
        object.__setattr__(self, "observation_count", observation_count)
        object.__setattr__(self, "centroid_reconstruction", centroid)
        object.__setattr__(self, "bounds_min_reconstruction", lower)
        object.__setattr__(self, "bounds_max_reconstruction", upper)
        object.__setattr__(self, "orientation_reliable", bool(self.orientation_reliable))
        object.__setattr__(self, "metric_geometry_status", status)
        object.__setattr__(self, "certified_metrology", False)
        for field_name in (
            "median_vertex_uncertainty_reconstruction_units",
            "maximum_vertex_uncertainty_reconstruction_units",
        ):
            value = getattr(self, field_name)
            if value is None:
                continue
            numeric = float(value)
            if not isfinite(numeric) or numeric < 0.0:
                raise ValueError(f"{field_name} must be finite and non-negative when present.")
            object.__setattr__(self, field_name, numeric)

    @property
    def display_name(self) -> str:
        role = self.semantic_role.replace("_", " ")
        return f"{self.class_name} • {role}"

    @property
    def role_text(self) -> str:
        return self.semantic_role.replace("_", " ").title()

    @property
    def support_text(self) -> str:
        return (
            f"{self.observation_count} obs • {self.vertex_count:,} vertices • "
            f"{self.face_count:,} faces"
        )

    @property
    def quality_text(self) -> str:
        return (
            f"Semantic {self.confidence * 100.0:.1f}% • "
            f"vertex quality {self.mean_vertex_quality * 100.0:.1f}%"
        )

    @property
    def scale_text(self) -> str:
        if self.metric_geometry_status == "arbitrary":
            return "Arbitrary reconstruction gauge"
        if self.metric_geometry_status == "provisional_metric":
            return "Provisional metric geometry • not certified"
        return "Validated metric geometry • metrology not certified"

    @property
    def authority_text(self) -> str:
        return "Semantic context • not inspection/QC decision"


@dataclass(frozen=True, slots=True)
class OfflineSemanticPresentation:
    session_id: str
    source: str
    metric_geometry_status: str
    total_object_count: int
    objects: tuple[OfflineSemanticObjectPresentation, ...]
    observations_total: int
    observations_accepted: int
    detailed_snapshot: bool
    unavailable_reason: str = ""
    certified_metrology: bool = False

    def __post_init__(self) -> None:
        session_id = str(self.session_id).strip()
        source = str(self.source).strip()
        status = str(self.metric_geometry_status).strip().lower()
        if not session_id:
            raise ValueError("Semantic presentation session_id must be non-empty.")
        if not source:
            raise ValueError("Semantic presentation source must be non-empty.")
        if status not in {"arbitrary", "provisional_metric", "validated_metric", "unknown"}:
            raise ValueError(f"Unsupported semantic metric geometry status: {status!r}")
        total = _nonnegative_int(self.total_object_count, "total_object_count")
        observations_total = _nonnegative_int(self.observations_total, "observations_total")
        accepted = _nonnegative_int(self.observations_accepted, "observations_accepted")
        if accepted > observations_total:
            raise ValueError("observations_accepted cannot exceed observations_total.")
        objects = tuple(self.objects)
        if any(not isinstance(item, OfflineSemanticObjectPresentation) for item in objects):
            raise TypeError("objects must contain OfflineSemanticObjectPresentation values.")
        if len(objects) > MAX_SEMANTIC_OBJECT_PRESENTATIONS:
            raise ValueError("Semantic presentation exceeds the bounded UI object limit.")
        if total < len(objects):
            raise ValueError("total_object_count cannot be smaller than the displayed object count.")
        if len({item.object_id for item in objects}) != len(objects):
            raise ValueError("Displayed semantic object IDs must remain unique.")
        if bool(self.certified_metrology):
            raise ValueError("D5 semantic presentation cannot claim certified metrology.")
        object.__setattr__(self, "session_id", session_id)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "metric_geometry_status", status)
        object.__setattr__(self, "total_object_count", total)
        object.__setattr__(self, "objects", objects)
        object.__setattr__(self, "observations_total", observations_total)
        object.__setattr__(self, "observations_accepted", accepted)
        object.__setattr__(self, "detailed_snapshot", bool(self.detailed_snapshot))
        object.__setattr__(self, "unavailable_reason", str(self.unavailable_reason).strip())
        object.__setattr__(self, "certified_metrology", False)

    @property
    def display_object_count(self) -> int:
        return len(self.objects)

    @property
    def has_objects(self) -> bool:
        return bool(self.objects)

    @property
    def structure_text(self) -> str:
        if self.detailed_snapshot:
            return (
                f"{self.display_object_count}/{self.total_object_count} objects shown • "
                f"{self.observations_accepted}/{self.observations_total} observations accepted"
            )
        if self.total_object_count:
            return f"{self.total_object_count} semantic objects • detailed snapshot unavailable"
        return "Semantic stage present • detailed snapshot unavailable"

    @property
    def authority_text(self) -> str:
        return "Semantic context only • QC/measurement authority unchanged"


def semantic_presentation_from_payload(
    payload: Mapping[str, Any] | Any,
    *,
    source: str = "semantic_geometry",
    max_objects: int = MAX_SEMANTIC_OBJECT_PRESENTATIONS,
) -> OfflineSemanticPresentation:
    """Normalize an existing semantic result without recomputing semantics."""

    if not isinstance(payload, Mapping):
        as_dict = getattr(payload, "as_dict", None)
        if not callable(as_dict):
            raise TypeError("Semantic payload must be a mapping or expose as_dict().")
        payload = as_dict()
    raw = dict(payload)
    if bool(raw.get("certified_metrology", False)):
        raise ValueError("Semantic payload cannot claim certified metrology in D5.")
    session_id = str(raw.get("session_id", "")).strip()
    status = str(raw.get("metric_geometry_status", "unknown")).strip().lower() or "unknown"
    statistics = raw.get("statistics", {})
    if not isinstance(statistics, Mapping):
        statistics = {}
    raw_objects = raw.get("objects", ())
    if isinstance(raw_objects, Mapping):
        # Known metadata empty-container ambiguity: an empty normalized list may
        # round-trip as an empty mapping.  Non-empty mappings remain invalid.
        if raw_objects:
            raise ValueError("Semantic payload objects must be a sequence.")
        raw_objects = ()
    if not isinstance(raw_objects, Sequence) or isinstance(raw_objects, (str, bytes, bytearray)):
        raise ValueError("Semantic payload objects must be a sequence.")
    max_objects = int(max_objects)
    if max_objects < 1:
        raise ValueError("max_objects must be >= 1.")
    objects = tuple(_object_from_mapping(item) for item in list(raw_objects)[:max_objects])
    total = _coerce_count(
        statistics.get("persistent_object_count"),
        fallback=len(raw_objects),
    )
    observations_total = _coerce_count(statistics.get("observations_total"), fallback=0)
    observations_accepted = _coerce_count(
        statistics.get("observations_accepted"),
        fallback=observations_total,
    )
    return OfflineSemanticPresentation(
        session_id=session_id,
        source=source,
        metric_geometry_status=status,
        total_object_count=max(total, len(raw_objects), len(objects)),
        objects=objects,
        observations_total=observations_total,
        observations_accepted=observations_accepted,
        detailed_snapshot=True,
        certified_metrology=False,
    )


def semantic_presentation_from_persistent_scene(
    scene: Any,
    *,
    max_objects: int = MAX_SEMANTIC_OBJECT_PRESENTATIONS,
) -> OfflineSemanticPresentation | None:
    """Extract semantic context from a persistent scene when a snapshot exists.

    F5 persistence may contain only semantic artifact metadata.  D5 never
    invents per-object details in that case; it presents an explicit summary
    and waits for a detailed snapshot rather than fabricating geometry.
    """

    if scene is None:
        return None
    session_id = str(getattr(scene, "session_id", "")).strip()
    if not session_id:
        raise ValueError("Persistent scene does not expose a semantic session identity.")
    artifacts = tuple(getattr(scene, "artifacts", ()) or ())
    semantic_artifact = None
    for artifact in artifacts:
        kind = getattr(getattr(artifact, "kind", None), "value", None)
        if kind is None:
            kind = str(getattr(artifact, "kind", "")).strip().lower()
        if kind == "semantic_geometry":
            semantic_artifact = artifact
            break
    if semantic_artifact is None:
        return None
    artifact_dict = semantic_artifact.as_dict() if callable(getattr(semantic_artifact, "as_dict", None)) else {}
    snapshot = artifact_dict.get("payload_snapshot", {}) if isinstance(artifact_dict, Mapping) else {}
    if isinstance(snapshot, Mapping) and snapshot and snapshot.get("objects") is not None:
        presentation = semantic_presentation_from_payload(
            snapshot,
            source="persistent_scene.semantic_geometry",
            max_objects=max_objects,
        )
        if presentation.session_id != session_id:
            raise ValueError("Persistent semantic snapshot session identity does not match the scene.")
        return presentation

    metadata = artifact_dict.get("metadata", {}) if isinstance(artifact_dict, Mapping) else {}
    if not isinstance(metadata, Mapping):
        metadata = {}
    total = _coerce_count(metadata.get("object_count"), fallback=0)
    observations_total = _coerce_count(metadata.get("observations_total"), fallback=0)
    observations_accepted = _coerce_count(
        metadata.get("observations_accepted"), fallback=observations_total
    )
    status = str(metadata.get("metric_geometry_status", "unknown")).strip().lower() or "unknown"
    return OfflineSemanticPresentation(
        session_id=session_id,
        source="persistent_scene.semantic_geometry",
        metric_geometry_status=status,
        total_object_count=total,
        objects=(),
        observations_total=observations_total,
        observations_accepted=observations_accepted,
        detailed_snapshot=False,
        unavailable_reason="Detailed semantic payload snapshot was not persisted.",
        certified_metrology=False,
    )


def semantic_bounds_segments_in_viewer(
    bounds_min_reconstruction: np.ndarray,
    bounds_max_reconstruction: np.ndarray,
) -> np.ndarray:
    """Return 12 semantic-context box edges in viewer coordinates.

    These lines visualize semantic support bounds only.  They are not object
    dimensions and do not establish metric or measurement authority.
    """

    lower = _vector3(bounds_min_reconstruction, "bounds_min_reconstruction")
    upper = _vector3(bounds_max_reconstruction, "bounds_max_reconstruction")
    if np.any(upper < lower):
        raise ValueError("Semantic bounds are invalid.")
    corners = np.asarray(
        [
            [lower[0], lower[1], lower[2]],
            [upper[0], lower[1], lower[2]],
            [upper[0], upper[1], lower[2]],
            [lower[0], upper[1], lower[2]],
            [lower[0], lower[1], upper[2]],
            [upper[0], lower[1], upper[2]],
            [upper[0], upper[1], upper[2]],
            [lower[0], upper[1], upper[2]],
        ],
        dtype=np.float64,
    )
    viewer = reconstruction_points_to_viewer(corners).astype(np.float32, copy=False)
    edges = (
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    )
    return np.asarray([viewer[index] for edge in edges for index in edge], dtype=np.float32)


def semantic_centroid_in_viewer(centroid_reconstruction: np.ndarray) -> np.ndarray:
    value = _vector3(centroid_reconstruction, "centroid_reconstruction")
    return reconstruction_points_to_viewer(value.reshape(1, 3))[0].astype(np.float32, copy=False)


def _object_from_mapping(value: Any) -> OfflineSemanticObjectPresentation:
    if not isinstance(value, Mapping):
        raise ValueError("Each semantic object must be a mapping.")
    raw = dict(value)
    observation_ids = raw.get("observation_ids", ())
    if isinstance(observation_ids, Mapping) and not observation_ids:
        observation_ids = ()
    if not isinstance(observation_ids, Sequence) or isinstance(observation_ids, (str, bytes, bytearray)):
        observation_ids = ()
    return OfflineSemanticObjectPresentation(
        object_id=str(raw.get("object_id", "")),
        class_name=str(raw.get("class_name", "")),
        semantic_role=str(raw.get("semantic_role", "")),
        confidence=float(raw.get("confidence", 0.0)),
        vertex_count=_coerce_count(raw.get("vertex_count"), fallback=0),
        face_count=_coerce_count(raw.get("face_count"), fallback=0),
        observation_count=len(observation_ids),
        mean_vertex_quality=float(raw.get("mean_vertex_quality", 0.0)),
        median_vertex_uncertainty_reconstruction_units=_optional_float(
            raw.get("median_vertex_uncertainty_reconstruction_units")
        ),
        maximum_vertex_uncertainty_reconstruction_units=_optional_float(
            raw.get("maximum_vertex_uncertainty_reconstruction_units")
        ),
        centroid_reconstruction=np.asarray(raw.get("centroid_reconstruction", ()), dtype=np.float64),
        bounds_min_reconstruction=np.asarray(raw.get("bounds_min_reconstruction", ()), dtype=np.float64),
        bounds_max_reconstruction=np.asarray(raw.get("bounds_max_reconstruction", ()), dtype=np.float64),
        orientation_reliable=bool(raw.get("orientation_reliable", False)),
        metric_geometry_status=str(raw.get("metric_geometry_status", "unknown")),
        certified_metrology=bool(raw.get("certified_metrology", False)),
    )


def _vector3(value: Any, field_name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{field_name} must be a finite 3-vector.")
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


def _probability(value: Any, field_name: str) -> float:
    numeric = float(value)
    if not isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{field_name} must be finite in [0, 1].")
    return numeric


def _nonnegative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a non-negative integer.")
    numeric = int(value)
    if numeric < 0:
        raise ValueError(f"{field_name} must be a non-negative integer.")
    return numeric


def _coerce_count(value: Any, *, fallback: int) -> int:
    try:
        return _nonnegative_int(value, "count")
    except (TypeError, ValueError):
        return max(0, int(fallback))


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if not isfinite(numeric):
        raise ValueError("Semantic uncertainty values must be finite when present.")
    return numeric


__all__ = [
    "MAX_SEMANTIC_OBJECT_PRESENTATIONS",
    "OfflineSemanticObjectPresentation",
    "OfflineSemanticPresentation",
    "semantic_bounds_segments_in_viewer",
    "semantic_centroid_in_viewer",
    "semantic_presentation_from_payload",
    "semantic_presentation_from_persistent_scene",
]

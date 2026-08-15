"""Uncertainty-aware Phase 9.3J measurement -> canonical registration.

The original Phase 7.5 measurement remains the measurement authority.  This
module only decides whether already-computed camera-frame measurement geometry
can be placed truthfully in a persistent canonical reconstruction scene.

Canonical dimension geometry is emitted only when all required evidence is
explicit: exact source-frame identity, a matching persisted camera pose,
qualified metric scene scale, and quantified pose/registration uncertainty.
Missing evidence produces an unavailable/provisional state rather than an
invented transform, metric scale, or uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite, sqrt
from typing import Mapping, Sequence

import numpy as np

from core.reconstruction.errors import ReconstructionValidationError


class CanonicalMeasurementRegistrationState(str, Enum):
    UNAVAILABLE = "unavailable"
    PROVISIONAL = "provisional"
    QUALIFIED = "qualified"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class CanonicalMeasurementRegistrationEvidence:
    """Explicit camera-frame -> canonical reconstruction registration evidence."""

    evidence_id: str
    scene_id: str
    canonical_object_id: str
    source_observation_id: str
    source_frame_sha256: str
    canonical_camera_id: str
    measurement_coordinate_frame_id: str
    reconstruction_frame_id: str
    state: CanonicalMeasurementRegistrationState
    T_reconstruction_from_measurement_camera: np.ndarray | None = None
    unit_to_metre: float | None = None
    relative_scale_uncertainty: float | None = None
    translation_uncertainty_m: float | None = None
    rotation_uncertainty_rad: float | None = None
    provenance: str = ""
    reasons: tuple[str, ...] = ()
    certified_metrology: bool = False

    def __post_init__(self) -> None:
        for name in (
            "evidence_id",
            "scene_id",
            "canonical_object_id",
            "source_observation_id",
        ):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ReconstructionValidationError(f"{name} must be non-empty.")
            object.__setattr__(self, name, value)
        for name in (
            "source_frame_sha256",
            "canonical_camera_id",
            "measurement_coordinate_frame_id",
            "reconstruction_frame_id",
            "provenance",
        ):
            object.__setattr__(self, name, str(getattr(self, name)).strip())
        digest = self.source_frame_sha256
        if digest and (len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest.lower())):
            raise ReconstructionValidationError("source_frame_sha256 must be hexadecimal SHA-256 when present.")
        object.__setattr__(self, "source_frame_sha256", digest.lower())
        if not isinstance(self.state, CanonicalMeasurementRegistrationState):
            raise TypeError("state must be CanonicalMeasurementRegistrationState.")
        if bool(self.certified_metrology):
            raise ReconstructionValidationError("Phase 9.3J registration cannot certify metrology.")
        reasons = tuple(str(value).strip() for value in self.reasons if str(value).strip())
        object.__setattr__(self, "reasons", reasons)

        matrix = self.T_reconstruction_from_measurement_camera
        if matrix is not None:
            matrix = np.asarray(matrix, dtype=np.float64)
            if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
                raise ReconstructionValidationError("Registration transform must be a finite 4x4 matrix.")
            if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8):
                raise ReconstructionValidationError("Registration transform must be homogeneous.")
            rotation = matrix[:3, :3]
            if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
                raise ReconstructionValidationError("Registration rotation must be orthonormal.")
            if not np.isclose(float(np.linalg.det(rotation)), 1.0, atol=1e-5):
                raise ReconstructionValidationError("Registration rotation must have determinant +1.")
            matrix = np.ascontiguousarray(matrix)
            matrix.setflags(write=False)
            object.__setattr__(self, "T_reconstruction_from_measurement_camera", matrix)

        unit = _optional_positive(self.unit_to_metre, "unit_to_metre")
        scale_u = _optional_relative(self.relative_scale_uncertainty, "relative_scale_uncertainty")
        translation_u = _optional_nonnegative(self.translation_uncertainty_m, "translation_uncertainty_m")
        rotation_u = _optional_nonnegative(self.rotation_uncertainty_rad, "rotation_uncertainty_rad")
        object.__setattr__(self, "unit_to_metre", unit)
        object.__setattr__(self, "relative_scale_uncertainty", scale_u)
        object.__setattr__(self, "translation_uncertainty_m", translation_u)
        object.__setattr__(self, "rotation_uncertainty_rad", rotation_u)
        object.__setattr__(self, "certified_metrology", False)

        if self.state is CanonicalMeasurementRegistrationState.QUALIFIED:
            missing = []
            if matrix is None:
                missing.append("transform")
            if not self.source_frame_sha256:
                missing.append("source_frame_sha256")
            if not self.canonical_camera_id:
                missing.append("canonical_camera_id")
            if not self.measurement_coordinate_frame_id:
                missing.append("measurement_coordinate_frame_id")
            if not self.reconstruction_frame_id:
                missing.append("reconstruction_frame_id")
            if unit is None:
                missing.append("unit_to_metre")
            if scale_u is None:
                missing.append("relative_scale_uncertainty")
            if translation_u is None:
                missing.append("translation_uncertainty_m")
            if rotation_u is None:
                missing.append("rotation_uncertainty_rad")
            if missing:
                raise ReconstructionValidationError(
                    "Qualified canonical measurement registration is missing: " + ", ".join(missing)
                )
            if reasons:
                raise ReconstructionValidationError("Qualified registration cannot carry rejection reasons.")
        elif self.state in {
            CanonicalMeasurementRegistrationState.UNAVAILABLE,
            CanonicalMeasurementRegistrationState.REJECTED,
        }:
            if not reasons:
                raise ReconstructionValidationError(
                    "Unavailable/rejected registration requires an explicit reason."
                )

    @property
    def canonical_dimension_geometry_allowed(self) -> bool:
        return self.state is CanonicalMeasurementRegistrationState.QUALIFIED


@dataclass(frozen=True, slots=True, eq=False)
class CanonicalRegisteredDimension:
    kind: str
    value_si: float
    source_expanded_uncertainty_si: float
    scale_uncertainty_si: float
    registered_dimension_uncertainty_si: float
    endpoint_placement_uncertainty_si: float
    start_reconstruction: np.ndarray
    end_reconstruction: np.ndarray

    def __post_init__(self) -> None:
        kind = str(self.kind).strip().lower()
        if kind not in {"width", "height", "depth"}:
            raise ReconstructionValidationError("Canonical dimension kind must be width/height/depth.")
        object.__setattr__(self, "kind", kind)
        for name in (
            "value_si",
            "source_expanded_uncertainty_si",
            "scale_uncertainty_si",
            "registered_dimension_uncertainty_si",
            "endpoint_placement_uncertainty_si",
        ):
            value = float(getattr(self, name))
            if not isfinite(value) or value < 0.0:
                raise ReconstructionValidationError(f"{name} must be finite and non-negative.")
            object.__setattr__(self, name, value)
        for name in ("start_reconstruction", "end_reconstruction"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (3,) or not np.isfinite(value).all():
                raise ReconstructionValidationError(f"{name} must be a finite 3-vector.")
            value = np.ascontiguousarray(value)
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    @property
    def label(self) -> str:
        return self.kind.title()

    @property
    def value_text(self) -> str:
        return (
            f"{self.label}: {self.value_si * 1000.0:.1f} mm ± "
            f"{self.registered_dimension_uncertainty_si * 1000.0:.1f} mm"
        )


@dataclass(frozen=True, slots=True)
class CanonicalMeasurementRegistrationResult:
    evidence: CanonicalMeasurementRegistrationEvidence
    dimensions: tuple[CanonicalRegisteredDimension, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.evidence, CanonicalMeasurementRegistrationEvidence):
            raise TypeError("evidence must be CanonicalMeasurementRegistrationEvidence.")
        dimensions = tuple(self.dimensions)
        if dimensions and not self.evidence.canonical_dimension_geometry_allowed:
            raise ReconstructionValidationError(
                "Canonical dimension geometry requires qualified registration evidence."
            )
        if any(not isinstance(value, CanonicalRegisteredDimension) for value in dimensions):
            raise TypeError("dimensions must contain CanonicalRegisteredDimension values.")
        object.__setattr__(self, "dimensions", dimensions)


def build_registered_dimensions(
    evidence: CanonicalMeasurementRegistrationEvidence,
    *,
    box_payload: Mapping[str, object],
    measurement_entries: Sequence[Mapping[str, object]],
) -> CanonicalMeasurementRegistrationResult:
    """Transform source-authored 3D measurement axes into canonical coordinates.

    The original numeric measurement remains authoritative.  The emitted line
    endpoints are presentation geometry only and exist only for qualified
    registration evidence.
    """

    if not isinstance(evidence, CanonicalMeasurementRegistrationEvidence):
        raise TypeError("evidence must be CanonicalMeasurementRegistrationEvidence.")
    if not evidence.canonical_dimension_geometry_allowed:
        return CanonicalMeasurementRegistrationResult(evidence=evidence, dimensions=())
    coord = str(box_payload.get("coordinate_frame_id", "")).strip()
    if coord != evidence.measurement_coordinate_frame_id:
        raise ReconstructionValidationError(
            "Measurement box coordinate frame does not match registration evidence."
        )
    box_fit_method = str(box_payload.get("fit_method", "")).strip()
    if not box_fit_method:
        raise ReconstructionValidationError("Qualified measurement box requires explicit fit_method.")
    centre = _vector3(box_payload.get("centre_xyz_m"), "centre_xyz_m")
    extents = _vector3(box_payload.get("extents_xyz_m"), "extents_xyz_m")
    if np.any(extents <= 0.0):
        raise ReconstructionValidationError("Measurement box extents must be positive.")
    orientation = np.asarray(box_payload.get("orientation_matrix"), dtype=np.float64)
    if orientation.shape != (3, 3) or not np.isfinite(orientation).all():
        raise ReconstructionValidationError("Measurement box orientation_matrix must be finite 3x3.")
    if not np.allclose(orientation.T @ orientation, np.eye(3), atol=1e-5):
        raise ReconstructionValidationError("Measurement box orientation must be orthonormal.")
    kind = str(box_payload.get("kind", "")).strip().lower()
    axis_indices = _semantic_axis_indices(orientation, kind)

    by_kind: dict[str, Mapping[str, object]] = {}
    for raw in measurement_entries:
        name = str(raw.get("kind", "")).strip().lower()
        if name in {"width", "height", "depth"}:
            by_kind[name] = raw

    transform = evidence.T_reconstruction_from_measurement_camera
    assert transform is not None
    unit_to_metre = float(evidence.unit_to_metre)
    scale_rel = float(evidence.relative_scale_uncertainty)
    translation_u = float(evidence.translation_uncertainty_m)
    rotation_u = float(evidence.rotation_uncertainty_rad)

    output: list[CanonicalRegisteredDimension] = []
    for semantic_index, name in enumerate(("width", "height", "depth")):
        raw = by_kind.get(name)
        if raw is None or str(raw.get("status", "")).strip().lower() != "available":
            continue
        if str(raw.get("method", "")).strip() != box_fit_method:
            raise ReconstructionValidationError(
                f"{name} measurement method does not match the persisted 3D box fit method."
            )
        value = _finite_nonnegative(raw.get("value_si"), f"{name}.value_si")
        source_u = _finite_nonnegative(
            raw.get("expanded_uncertainty_si"), f"{name}.expanded_uncertainty_si"
        )
        unit = str(raw.get("unit_si", "")).strip()
        if unit != "m":
            continue
        axis_index = axis_indices[semantic_index]
        axis = orientation[:, axis_index]
        half = float(extents[axis_index]) * 0.5
        start_m = centre - axis * half
        end_m = centre + axis * half
        start = _camera_metres_to_reconstruction(start_m, transform, unit_to_metre)
        end = _camera_metres_to_reconstruction(end_m, transform, unit_to_metre)

        scale_u = abs(value) * scale_rel
        registered_u = sqrt(source_u * source_u + scale_u * scale_u)
        endpoint_u = sqrt(
            translation_u * translation_u
            + (0.5 * abs(value) * rotation_u) ** 2
            + (0.5 * scale_u) ** 2
        )
        output.append(
            CanonicalRegisteredDimension(
                kind=name,
                value_si=value,
                source_expanded_uncertainty_si=source_u,
                scale_uncertainty_si=scale_u,
                registered_dimension_uncertainty_si=registered_u,
                endpoint_placement_uncertainty_si=endpoint_u,
                start_reconstruction=start,
                end_reconstruction=end,
            )
        )
    return CanonicalMeasurementRegistrationResult(evidence=evidence, dimensions=tuple(output))


def _camera_metres_to_reconstruction(point_m: np.ndarray, transform: np.ndarray, unit_to_metre: float) -> np.ndarray:
    point_units = np.asarray(point_m, dtype=np.float64) / unit_to_metre
    homogeneous = np.concatenate([point_units, np.ones((1,), dtype=np.float64)])
    return np.asarray((transform @ homogeneous)[:3], dtype=np.float64)


def _semantic_axis_indices(orientation: np.ndarray, kind: str) -> tuple[int, int, int]:
    if kind != "oriented":
        return (0, 1, 2)
    from itertools import permutations

    best = max(
        permutations(range(3)),
        key=lambda p: sum(abs(float(orientation[camera_axis, p[camera_axis]])) for camera_axis in range(3)),
    )
    return int(best[0]), int(best[1]), int(best[2])


def _vector3(value: object, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ReconstructionValidationError(f"{name} must be a finite 3-vector.")
    return array


def _finite_nonnegative(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ReconstructionValidationError(f"{name} must be numeric.") from exc
    if not isfinite(result) or result < 0.0:
        raise ReconstructionValidationError(f"{name} must be finite and non-negative.")
    return result


def _optional_positive(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not isfinite(result) or result <= 0.0:
        raise ReconstructionValidationError(f"{name} must be finite and positive.")
    return result


def _optional_nonnegative(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not isfinite(result) or result < 0.0:
        raise ReconstructionValidationError(f"{name} must be finite and non-negative.")
    return result


def _optional_relative(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not isfinite(result) or not 0.0 <= result < 1.0:
        raise ReconstructionValidationError(f"{name} must be finite in [0,1).")
    return result


__all__ = [
    "CanonicalMeasurementRegistrationEvidence",
    "CanonicalMeasurementRegistrationResult",
    "CanonicalMeasurementRegistrationState",
    "CanonicalRegisteredDimension",
    "build_registered_dimensions",
]

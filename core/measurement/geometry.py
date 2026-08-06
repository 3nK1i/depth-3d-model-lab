"""Dimension, centre, distance and orientation derivation from Step 5 boxes."""
from __future__ import annotations
from itertools import permutations
import math
import numpy as np
from core.measurement.contracts import BoundingBox3D, BoundingBoxKind, MeasurementKind
from core.object_geometry.fitting_contracts import ObjectGeometryFitResult, OrientationStability

_DIMENSION_KINDS = (MeasurementKind.WIDTH, MeasurementKind.HEIGHT, MeasurementKind.DEPTH)

def fitted_boxes(fit: ObjectGeometryFitResult) -> tuple[BoundingBox3D, ...]:
    return tuple(b for b in (fit.axis_aligned_box, fit.camera_aligned_box, fit.oriented_box) if b is not None)

def box_by_id(fit: ObjectGeometryFitResult, box_id: str) -> BoundingBox3D | None:
    return next((b for b in fitted_boxes(fit) if b.box_id == box_id), None)

def select_measurement_box(
    fit: ObjectGeometryFitResult,
    *,
    prefer_stable_oriented: bool = True,
    forced_kind: BoundingBoxKind | None = None,
) -> tuple[BoundingBox3D | None, str]:
    if forced_kind is not None:
        by_kind = {
            BoundingBoxKind.AXIS_ALIGNED: fit.axis_aligned_box,
            BoundingBoxKind.CAMERA_ALIGNED: fit.camera_aligned_box,
            BoundingBoxKind.ORIENTED: fit.oriented_box,
        }
        selected = by_kind[forced_kind]
        return (selected, f"operator-selected {forced_kind.value} box") if selected is not None else (None, f"operator-selected {forced_kind.value} box unavailable")
    stable = fit.object_frame is not None and fit.object_frame.orientation_stability is OrientationStability.STABLE
    if prefer_stable_oriented and stable and fit.oriented_box is not None:
        return fit.oriented_box, "stable oriented box"
    if fit.camera_aligned_box is not None:
        return fit.camera_aligned_box, "camera-aligned fallback"
    if fit.axis_aligned_box is not None:
        return fit.axis_aligned_box, "axis-aligned fallback"
    if fit.oriented_box is not None:
        return fit.oriented_box, "available oriented box"
    return None, "no fitted box"

def _camera_axis_assignment(orientation: np.ndarray) -> tuple[int, int, int]:
    R = np.asarray(orientation, dtype=np.float64)
    best = max(permutations(range(3)), key=lambda p: sum(abs(R[camera_axis, p[camera_axis]]) for camera_axis in range(3)))
    return int(best[0]), int(best[1]), int(best[2])

def semantic_extents(box: BoundingBox3D) -> dict[MeasurementKind, float]:
    ext = np.asarray(box.extents_xyz_m, dtype=np.float64)
    if box.kind is BoundingBoxKind.ORIENTED:
        ix, iy, iz = _camera_axis_assignment(box.orientation_matrix)
        ext = ext[[ix, iy, iz]]
    return {MeasurementKind.WIDTH: float(ext[0]), MeasurementKind.HEIGHT: float(ext[1]), MeasurementKind.DEPTH: float(ext[2])}

def rotation_matrix_to_rpy(matrix: np.ndarray) -> tuple[float, float, float]:
    """Return intrinsic XYZ roll/pitch/yaw in radians using a stable ZYX decomposition."""
    R = np.asarray(matrix, dtype=np.float64)
    sy = math.hypot(float(R[0, 0]), float(R[1, 0]))
    if sy > 1e-9:
        roll = math.atan2(float(R[2, 1]), float(R[2, 2]))
        pitch = math.atan2(-float(R[2, 0]), sy)
        yaw = math.atan2(float(R[1, 0]), float(R[0, 0]))
    else:
        roll = math.atan2(-float(R[1, 2]), float(R[1, 1]))
        pitch = math.atan2(-float(R[2, 0]), sy)
        yaw = 0.0
    return roll, pitch, yaw

def values_from_box(box: BoundingBox3D) -> dict[MeasurementKind, float]:
    values = semantic_extents(box)
    centre = np.asarray(box.centre_xyz_m, dtype=np.float64)
    values.update({
        MeasurementKind.CAMERA_DISTANCE: float(np.linalg.norm(centre)),
        MeasurementKind.CENTRE_X: float(centre[0]),
        MeasurementKind.CENTRE_Y: float(centre[1]),
        MeasurementKind.CENTRE_Z: float(centre[2]),
        MeasurementKind.VOLUME: float(values[MeasurementKind.WIDTH] * values[MeasurementKind.HEIGHT] * values[MeasurementKind.DEPTH]),
    })
    roll, pitch, yaw = rotation_matrix_to_rpy(box.orientation_matrix)
    values.update({MeasurementKind.ORIENTATION_ROLL: roll, MeasurementKind.ORIENTATION_PITCH: pitch, MeasurementKind.ORIENTATION_YAW: yaw})
    return values

def unit_for_kind(kind: MeasurementKind) -> str:
    if kind in {MeasurementKind.ORIENTATION_ROLL, MeasurementKind.ORIENTATION_PITCH, MeasurementKind.ORIENTATION_YAW}:
        return "rad"
    if kind is MeasurementKind.VOLUME:
        return "m3"
    return "m"

def is_orientation_kind(kind: MeasurementKind) -> bool:
    return kind in {MeasurementKind.ORIENTATION_ROLL, MeasurementKind.ORIENTATION_PITCH, MeasurementKind.ORIENTATION_YAW}

def is_scale_dependent(kind: MeasurementKind) -> bool:
    return not is_orientation_kind(kind)


def scale_factor_for_kind(kind: MeasurementKind, factor: float) -> float:
    if is_orientation_kind(kind):
        return 1.0
    if kind is MeasurementKind.VOLUME:
        return float(factor) ** 3
    return float(factor)

def apply_scale_to_values(values: dict[MeasurementKind, float], factor: float) -> dict[MeasurementKind, float]:
    return {kind: value * scale_factor_for_kind(kind, factor) for kind, value in values.items()}

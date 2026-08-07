"""Coordinate conversion helpers for external offline SfM adapters.

COLMAP exposes camera-from-world poses. Phase 8.3 stores the inverse transform,
``T_reconstruction_from_camera``. The reconstruction frame has arbitrary gauge
until a later metric alignment step.
"""

from __future__ import annotations

import numpy as np

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.poses import quaternion_wxyz_to_rotation_matrix

COLMAP_CAMERA_AXES = "x_right_y_down_z_forward"
COLMAP_TEXT_QUATERNION_ORDER = "wxyz"
SFM_RECONSTRUCTION_FRAME = "arbitrary_gauge_reconstruction"


def validate_rigid_matrix(value: object, *, field_name: str = "transform") -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ReconstructionValidationError(f"{field_name} must be a finite 4x4 matrix.")
    if not np.allclose(matrix[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-7):
        raise ReconstructionValidationError(
            f"{field_name} must have homogeneous final row [0, 0, 0, 1]."
        )
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ReconstructionValidationError(f"{field_name} rotation must be orthonormal.")
    if not np.isclose(float(np.linalg.det(rotation)), 1.0, atol=1e-6):
        raise ReconstructionValidationError(
            f"{field_name} rotation must have positive unit determinant."
        )
    matrix = np.ascontiguousarray(matrix).copy()
    matrix.setflags(write=False)
    return matrix


def invert_rigid_matrix(value: object) -> np.ndarray:
    matrix = validate_rigid_matrix(value)
    inverse = np.eye(4, dtype=np.float64)
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -(rotation.T @ translation)
    inverse.setflags(write=False)
    return inverse


def colmap_camera_from_reconstruction_to_internal_pose(value: object) -> np.ndarray:
    """Convert COLMAP camera-from-reconstruction to internal camera-to-map pose."""

    return invert_rigid_matrix(value)


def internal_pose_to_colmap_camera_from_reconstruction(value: object) -> np.ndarray:
    """Convert internal ``T_reconstruction_from_camera`` back to COLMAP form."""

    return invert_rigid_matrix(value)


def colmap_qvec_tvec_to_camera_from_reconstruction(
    quaternion_wxyz: object,
    translation: object,
) -> np.ndarray:
    """Build a COLMAP text-format camera-from-reconstruction transform.

    ``quaternion_wxyz`` follows COLMAP's text-model order. PyCOLMAP object
    adapters should preferably convert through a rotation matrix at the adapter
    boundary instead of relying on object-specific quaternion storage order.
    """

    rotation = quaternion_wxyz_to_rotation_matrix(quaternion_wxyz)
    t = np.asarray(translation, dtype=np.float64)
    if t.shape != (3,) or not np.isfinite(t).all():
        raise ReconstructionValidationError("COLMAP translation must be a finite 3-vector.")
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = t
    return validate_rigid_matrix(matrix, field_name="T_camera_from_reconstruction")


def colmap_camera_center_reconstruction(value: object) -> np.ndarray:
    """Return the camera centre in arbitrary reconstruction coordinates."""

    pose = colmap_camera_from_reconstruction_to_internal_pose(value)
    center = pose[:3, 3].copy()
    center.setflags(write=False)
    return center

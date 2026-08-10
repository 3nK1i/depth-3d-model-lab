"""Dependency-light geometry helpers for the Phase 8.8C reconstruction viewer.

The reconstruction core keeps its canonical camera convention unchanged:
``T_reconstruction_from_camera`` maps X-right, Y-down, Z-forward camera
coordinates into the arbitrary-gauge reconstruction frame.  The existing
application 3D viewer presents a Z-up display frame.  This module owns only the
presentation transform between those two spaces; it never mutates reconstruction
poses or claims metric scale.
"""

from __future__ import annotations

import numpy as np

# Same display convention used by the existing live point-cloud viewport:
# viewer X = reconstruction X, viewer Y = reconstruction Z,
# viewer Z = -reconstruction Y.  This keeps Z visually "up".
_RECONSTRUCTION_TO_VIEWER = np.asarray(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)


def reconstruction_point_to_viewer(point: np.ndarray) -> np.ndarray:
    """Return one reconstruction-frame point in the application's Z-up viewer frame."""

    value = np.asarray(point, dtype=np.float64)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError("point must be a finite 3-vector.")
    return _RECONSTRUCTION_TO_VIEWER @ value


def reconstruction_points_to_viewer(points: np.ndarray) -> np.ndarray:
    """Convert an ``(N, 3)`` reconstruction-frame array to viewer coordinates."""

    value = np.asarray(points, dtype=np.float64)
    if value.ndim != 2 or value.shape[1:] != (3,) or not np.isfinite(value).all():
        raise ValueError("points must be a finite (N, 3) array.")
    return value @ _RECONSTRUCTION_TO_VIEWER.T


def camera_position_in_viewer(T_reconstruction_from_camera: np.ndarray) -> np.ndarray:
    """Return the camera centre in viewer coordinates without changing pose authority."""

    pose = _validate_pose(T_reconstruction_from_camera)
    return reconstruction_point_to_viewer(pose[:3, 3])


def camera_frustum_segments_in_viewer(
    T_reconstruction_from_camera: np.ndarray,
    *,
    intrinsics_matrix: np.ndarray | None = None,
    image_size: tuple[int, int] | None = None,
    depth: float = 0.35,
) -> np.ndarray:
    """Build line-segment vertices for a small current-camera frustum.

    The frustum is a presentation visual only.  If calibrated/estimated
    intrinsics are supplied, their pinhole rays define the symbolic frustum.
    Otherwise a conservative 70-degree, 16:9 symbolic frustum is used.  The
    output shape is ``(16, 3)``: eight independent line segments suitable for a
    VisPy ``Line(connect='segments')`` visual.
    """

    pose = _validate_pose(T_reconstruction_from_camera)
    depth = float(depth)
    if not np.isfinite(depth) or depth <= 0.0:
        raise ValueError("depth must be a positive finite value.")

    corners = _camera_plane_corners(
        intrinsics_matrix=intrinsics_matrix,
        image_size=image_size,
        depth=depth,
    )
    origin = np.zeros(3, dtype=np.float64)
    camera_points = np.vstack([origin, corners])
    rotation = pose[:3, :3]
    translation = pose[:3, 3]
    reconstruction_points = camera_points @ rotation.T + translation
    viewer_points = reconstruction_points_to_viewer(reconstruction_points)

    # origin -> four corners, then the four perimeter edges.
    segment_indices = (
        (0, 1), (0, 2), (0, 3), (0, 4),
        (1, 2), (2, 3), (3, 4), (4, 1),
    )
    return np.asarray(
        [viewer_points[index] for pair in segment_indices for index in pair],
        dtype=np.float32,
    )


def _camera_plane_corners(
    *,
    intrinsics_matrix: np.ndarray | None,
    image_size: tuple[int, int] | None,
    depth: float,
) -> np.ndarray:
    if intrinsics_matrix is not None or image_size is not None:
        if intrinsics_matrix is None or image_size is None:
            raise ValueError("intrinsics_matrix and image_size must be supplied together.")
        K = np.asarray(intrinsics_matrix, dtype=np.float64)
        if K.shape != (3, 3) or not np.isfinite(K).all():
            raise ValueError("intrinsics_matrix must be a finite 3x3 array.")
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        width, height = int(image_size[0]), int(image_size[1])
        if fx <= 0.0 or fy <= 0.0 or width <= 0 or height <= 0:
            raise ValueError("intrinsics and image_size must describe a valid pinhole camera.")
        pixels = np.asarray(
            [
                [0.0, 0.0],
                [float(width - 1), 0.0],
                [float(width - 1), float(height - 1)],
                [0.0, float(height - 1)],
            ],
            dtype=np.float64,
        )
        x = (pixels[:, 0] - cx) / fx * depth
        y = (pixels[:, 1] - cy) / fy * depth
        return np.column_stack([x, y, np.full(4, depth, dtype=np.float64)])

    # Symbolic fallback only; no calibration/metric authority is implied.
    half_width = depth * np.tan(np.deg2rad(70.0) * 0.5)
    half_height = half_width / (16.0 / 9.0)
    return np.asarray(
        [
            [-half_width, -half_height, depth],
            [half_width, -half_height, depth],
            [half_width, half_height, depth],
            [-half_width, half_height, depth],
        ],
        dtype=np.float64,
    )


def _validate_pose(value: np.ndarray) -> np.ndarray:
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("T_reconstruction_from_camera must be a finite 4x4 matrix.")
    if not np.allclose(pose[3], np.asarray([0.0, 0.0, 0.0, 1.0]), atol=1e-8):
        raise ValueError("T_reconstruction_from_camera must be homogeneous.")
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError("T_reconstruction_from_camera rotation must be orthonormal.")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError("T_reconstruction_from_camera rotation must have determinant +1.")
    return pose

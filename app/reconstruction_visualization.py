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


def orientation_gizmo_projection(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """Project viewer X/Y/Z basis vectors into 2-D gizmo screen directions.

    The result has shape ``(3, 2)`` in X/Y/Z order.  Coordinates are normalized
    screen deltas where +x points right and +y points down (Qt convention).
    This is display-only orientation feedback for the turntable camera; it does
    not alter or reinterpret reconstruction geometry.
    """

    azimuth = np.deg2rad(float(azimuth_deg))
    elevation = np.deg2rad(float(elevation_deg))
    if not np.isfinite([azimuth, elevation]).all():
        raise ValueError("azimuth_deg and elevation_deg must be finite.")

    # Approximate the VisPy Z-up turntable camera position around its target.
    camera_offset = np.asarray(
        [
            np.cos(elevation) * np.sin(azimuth),
            np.cos(elevation) * np.cos(azimuth),
            np.sin(elevation),
        ],
        dtype=np.float64,
    )
    view_forward = -camera_offset
    view_forward /= max(float(np.linalg.norm(view_forward)), 1e-12)

    world_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    screen_right = np.cross(world_up, view_forward)
    norm = float(np.linalg.norm(screen_right))
    if norm < 1e-8:
        # Near a top/bottom pole, choose a stable horizontal reference.
        screen_right = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        screen_right /= norm
    screen_up = np.cross(view_forward, screen_right)
    screen_up /= max(float(np.linalg.norm(screen_up)), 1e-12)

    basis = np.eye(3, dtype=np.float64)
    projected = np.column_stack(
        [
            basis @ screen_right,
            -(basis @ screen_up),  # Qt screen Y grows downward.
        ]
    )
    return projected.astype(np.float32)


def adaptive_world_axis_length(
    trajectory_points: np.ndarray | None,
    *,
    idle_length: float = 0.28,
    minimum: float = 0.08,
    maximum: float = 0.34,
) -> float:
    """Return a restrained world-origin axis length for the current live path.

    C1's generic ``XYZAxis`` was intentionally simple but visually dominant in
    arbitrary-gauge scenes.  C2 sizes the world-origin axes from the observed
    trajectory extent while keeping conservative bounds.  The value is purely a
    display scale and therefore never establishes metric authority.
    """

    idle_length = float(idle_length)
    minimum = float(minimum)
    maximum = float(maximum)
    if not np.isfinite([idle_length, minimum, maximum]).all():
        raise ValueError("axis length bounds must be finite.")
    if minimum <= 0.0 or maximum < minimum or not (minimum <= idle_length <= maximum):
        raise ValueError("axis length bounds are inconsistent.")
    if trajectory_points is None:
        return idle_length

    points = np.asarray(trajectory_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all():
        raise ValueError("trajectory_points must be a finite (N, 3) array.")
    if points.shape[0] < 2:
        return idle_length
    span = np.ptp(points, axis=0)
    diagonal = float(np.linalg.norm(span))
    if diagonal <= 1e-6:
        return idle_length
    return float(np.clip(diagonal * 0.30, minimum, maximum))


def trajectory_display_indices(count: int, *, max_points: int = 4096) -> np.ndarray:
    """Return ordered indices for bounded trajectory rendering.

    The authoritative UI-side trajectory history may contain many more samples;
    only the visual line is decimated.  Endpoints are always preserved.
    """

    count = int(count)
    max_points = int(max_points)
    if count < 0 or max_points < 2:
        raise ValueError("count must be non-negative and max_points must be >= 2.")
    if count == 0:
        return np.zeros((0,), dtype=np.int64)
    if count <= max_points:
        return np.arange(count, dtype=np.int64)
    step = int(np.ceil((count - 1) / float(max_points - 1)))
    indices = np.arange(0, count, step, dtype=np.int64)
    if indices[-1] != count - 1:
        indices = np.concatenate([indices, np.asarray([count - 1], dtype=np.int64)])
    if indices.size > max_points:
        # Evenly reselect while preserving both endpoints.
        positions = np.linspace(0, indices.size - 1, max_points).round().astype(np.int64)
        indices = indices[positions]
        indices[0] = 0
        indices[-1] = count - 1
    return indices

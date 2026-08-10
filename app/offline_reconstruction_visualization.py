"""Bounded offline sparse-SfM visualization payloads for Phase 8.8D1.

The production offline worker already writes the complete ``sfm_result.json``
for persistence/debugging.  D1 adds a deliberately small companion artifact for
interactive UI rendering so the Qt thread never has to parse tens of thousands
of landmark observations just to draw the sparse result.

All coordinates remain in the reconstruction gauge.  This module never converts
arbitrary monocular SfM units to metres or promotes sparse SfM to canonical
geometry authority.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

import numpy as np

OFFLINE_VISUALIZATION_SCHEMA = "eia.reconstruction.offline.visualization.v1"
MAX_OFFLINE_VISUALIZATION_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class OfflineSparseVisualization:
    """Validated bounded presentation payload for one completed offline SfM run."""

    session_id: str
    backend_id: str
    backend_version: str
    scale_status: str
    alignment_status: str
    pose_convention: str
    camera_matrices_reconstruction: np.ndarray
    camera_ids: tuple[str, ...]
    landmark_points_reconstruction: np.ndarray
    landmark_colors_rgb: np.ndarray | None
    registered_camera_total: int
    landmark_total: int
    mean_reprojection_error_px: float | None

    @property
    def has_geometry(self) -> bool:
        return bool(self.camera_matrices_reconstruction.shape[0] or self.landmark_points_reconstruction.shape[0])


def load_offline_sparse_visualization(path: str | Path) -> OfflineSparseVisualization:
    """Load and validate one D1 visualization artifact from an absolute path."""

    target = Path(path).expanduser()
    if not target.is_absolute():
        raise ValueError("Offline visualization artifact path must be absolute.")
    target = target.resolve()
    if not target.is_file():
        raise ValueError(f"Offline visualization artifact does not exist: {target}")
    size = target.stat().st_size
    if size <= 0 or size > MAX_OFFLINE_VISUALIZATION_BYTES:
        raise ValueError("Offline visualization artifact size is invalid or exceeds the UI safety bound.")
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read offline visualization artifact: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Offline visualization artifact root must be a JSON object.")
    return parse_offline_sparse_visualization(payload)


def parse_offline_sparse_visualization(payload: Mapping[str, object]) -> OfflineSparseVisualization:
    """Validate the bounded worker artifact without importing PyCOLMAP."""

    schema = str(payload.get("schema", "")).strip()
    if schema != OFFLINE_VISUALIZATION_SCHEMA:
        raise ValueError(f"Unsupported offline visualization schema: {schema!r}.")

    session_id = _nonempty(payload.get("session_id"), "session_id")
    backend_id = _nonempty(payload.get("backend_id"), "backend_id")
    backend_version = _nonempty(payload.get("backend_version"), "backend_version")
    scale_status = _nonempty(payload.get("scale_status"), "scale_status")
    alignment_status = _nonempty(payload.get("alignment_status"), "alignment_status")
    pose_convention = _nonempty(payload.get("pose_convention"), "pose_convention")
    if pose_convention != "T_reconstruction_from_camera":
        raise ValueError("Offline visualization pose convention must be T_reconstruction_from_camera.")
    # D1 must not silently upgrade raw monocular SfM to metric authority.
    if scale_status != "arbitrary":
        raise ValueError("D1 offline sparse visualization requires arbitrary scale status.")

    raw_cameras = payload.get("registered_cameras", [])
    if not isinstance(raw_cameras, list):
        raise ValueError("registered_cameras must be a JSON array.")
    matrices: list[np.ndarray] = []
    camera_ids: list[str] = []
    for index, item in enumerate(raw_cameras):
        if not isinstance(item, Mapping):
            raise ValueError("Every registered camera sample must be a JSON object.")
        camera_id = _nonempty(item.get("keyframe_id") or item.get("estimate_id"), f"registered_cameras[{index}].id")
        matrix = np.asarray(item.get("T_reconstruction_from_camera"), dtype=np.float64)
        _validate_pose(matrix, field_name=f"registered_cameras[{index}].T_reconstruction_from_camera")
        camera_ids.append(camera_id)
        matrices.append(np.array(matrix, copy=True))
    if len(camera_ids) != len(set(camera_ids)):
        raise ValueError("Offline visualization camera IDs must be unique.")
    camera_array = (
        np.asarray(matrices, dtype=np.float64)
        if matrices
        else np.zeros((0, 4, 4), dtype=np.float64)
    )

    points = np.asarray(payload.get("landmarks_reconstruction", []), dtype=np.float64)
    if points.size == 0:
        points = np.zeros((0, 3), dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all():
        raise ValueError("landmarks_reconstruction must be a finite (N, 3) JSON array.")

    colors_value = payload.get("landmark_colors_rgb")
    colors: np.ndarray | None = None
    if colors_value is not None:
        colors = np.asarray(colors_value)
        if colors.size == 0:
            colors = np.zeros((0, 3), dtype=np.uint8)
        if colors.ndim != 2 or colors.shape != points.shape:
            raise ValueError("landmark_colors_rgb must match landmarks_reconstruction shape.")
        if not np.issubdtype(colors.dtype, np.number):
            raise ValueError("landmark_colors_rgb must be numeric.")
        colors_float = colors.astype(np.float64)
        if not np.isfinite(colors_float).all() or np.any(colors_float < 0) or np.any(colors_float > 255):
            raise ValueError("landmark_colors_rgb values must lie within [0, 255].")
        if not np.allclose(colors_float, np.round(colors_float)):
            raise ValueError("landmark_colors_rgb values must be integers.")
        colors = np.asarray(np.round(colors_float), dtype=np.uint8)

    registered_total = _nonnegative_int(payload.get("registered_camera_total", camera_array.shape[0]), "registered_camera_total")
    landmark_total = _nonnegative_int(payload.get("landmark_total", points.shape[0]), "landmark_total")
    if registered_total < camera_array.shape[0]:
        raise ValueError("registered_camera_total cannot be smaller than the camera sample count.")
    if landmark_total < points.shape[0]:
        raise ValueError("landmark_total cannot be smaller than the landmark sample count.")

    error_value = payload.get("mean_reprojection_error_px")
    mean_error: float | None = None
    if error_value is not None:
        if isinstance(error_value, bool):
            raise ValueError("mean_reprojection_error_px must be numeric or null.")
        mean_error = float(error_value)
        if not np.isfinite(mean_error) or mean_error < 0.0:
            raise ValueError("mean_reprojection_error_px must be finite and non-negative.")

    return OfflineSparseVisualization(
        session_id=session_id,
        backend_id=backend_id,
        backend_version=backend_version,
        scale_status=scale_status,
        alignment_status=alignment_status,
        pose_convention=pose_convention,
        camera_matrices_reconstruction=np.ascontiguousarray(camera_array),
        camera_ids=tuple(camera_ids),
        landmark_points_reconstruction=np.ascontiguousarray(points),
        landmark_colors_rgb=None if colors is None else np.ascontiguousarray(colors),
        registered_camera_total=registered_total,
        landmark_total=landmark_total,
        mean_reprojection_error_px=mean_error,
    )


def _validate_pose(matrix: np.ndarray, *, field_name: str) -> None:
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{field_name} must be a finite homogeneous 4x4 matrix.")
    if not np.allclose(matrix[3], np.asarray([0.0, 0.0, 0.0, 1.0]), atol=1e-7):
        raise ValueError(f"{field_name} must have homogeneous final row [0, 0, 0, 1].")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError(f"{field_name} rotation must be orthonormal.")
    if not np.isclose(float(np.linalg.det(rotation)), 1.0, atol=1e-5):
        raise ValueError(f"{field_name} rotation must have determinant +1.")


def _nonempty(value: object, field_name: str) -> str:
    text = str(value if value is not None else "").strip()
    if not text:
        raise ValueError(f"{field_name} must be non-empty.")
    return text


def _nonnegative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a non-negative integer.")
    if isinstance(value, int):
        number = value
    elif isinstance(value, float) and np.isfinite(value) and value.is_integer():
        number = int(value)
    else:
        raise ValueError(f"{field_name} must be a non-negative integer.")
    if number < 0:
        raise ValueError(f"{field_name} must be a non-negative integer.")
    return number



__all__ = [
    "MAX_OFFLINE_VISUALIZATION_BYTES",
    "OFFLINE_VISUALIZATION_SCHEMA",
    "OfflineSparseVisualization",
    "load_offline_sparse_visualization",
    "parse_offline_sparse_visualization",
]

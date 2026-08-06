"""Canonical camera pose conventions and rigid-transform helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from core.contracts import RigidTransform3D
from core.contracts.validation import require_nonempty
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.validation import require_timestamp_ns


CANONICAL_CAMERA_AXES = "x_right_y_down_z_forward"
CANONICAL_WORLD_AXES = "x_right_y_up_z_back"
CANONICAL_LENGTH_UNIT = "m"
CANONICAL_POSE_FIELD = "T_world_from_camera"
CANONICAL_QUATERNION_ORDER = "wxyz"
CANONICAL_MATRIX_STORAGE = "row_major"
COORDINATE_CONVENTION_VERSION = 1


def invert_rigid_transform(transform: RigidTransform3D, *, method: str = "inverse") -> RigidTransform3D:
    """Return ``T_source_from_destination`` for one rigid transform."""

    if not isinstance(transform, RigidTransform3D):
        raise TypeError("transform must be a RigidTransform3D.")
    rotation = transform.matrix[:3, :3]
    translation = transform.matrix[:3, 3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -(rotation.T @ translation)
    return RigidTransform3D(
        transform.destination_frame_id,
        transform.source_frame_id,
        inverse,
        method=method,
        tolerance=transform.tolerance,
    )


def compose_rigid_transforms(
    first: RigidTransform3D,
    second: RigidTransform3D,
    *,
    method: str = "composition",
) -> RigidTransform3D:
    """Compose ``first`` then ``second`` using destination-from-source notation.

    ``first`` must be ``T_middle_from_source`` and ``second`` must be
    ``T_destination_from_middle``. The result is ``T_destination_from_source``.
    """

    if first.destination_frame_id != second.source_frame_id:
        raise ReconstructionValidationError(
            "Transform composition requires first.destination_frame_id to match "
            "second.source_frame_id."
        )
    return RigidTransform3D(
        first.source_frame_id,
        second.destination_frame_id,
        second.matrix @ first.matrix,
        method=method,
        tolerance=max(first.tolerance, second.tolerance),
    )


def normalize_quaternion_wxyz(value: object) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ReconstructionValidationError("Quaternion must be a finite WXYZ 4-vector.")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ReconstructionValidationError("Quaternion norm must be positive.")
    quaternion = quaternion / norm
    if quaternion[0] < 0.0:
        quaternion = -quaternion
    quaternion.setflags(write=False)
    return quaternion


def quaternion_wxyz_to_rotation_matrix(value: object) -> np.ndarray:
    w, x, y, z = normalize_quaternion_wxyz(value)
    matrix = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    matrix.setflags(write=False)
    return matrix


def rotation_matrix_to_quaternion_wxyz(value: object) -> np.ndarray:
    rotation = np.asarray(value, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ReconstructionValidationError("Rotation matrix must be finite with shape (3, 3).")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ReconstructionValidationError("Rotation matrix must be orthonormal.")
    determinant = float(np.linalg.det(rotation))
    if not np.isclose(determinant, 1.0, atol=1e-6):
        raise ReconstructionValidationError("Rotation matrix must have unit determinant.")

    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quaternion = np.array(
            [
                0.25 * scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            ]
        )
    else:
        diagonal = np.diag(rotation)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            quaternion = np.array(
                [
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                    0.25 * scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                ]
            )
        elif index == 1:
            scale = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            quaternion = np.array(
                [
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    0.25 * scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                ]
            )
        else:
            scale = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            quaternion = np.array(
                [
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    return normalize_quaternion_wxyz(quaternion)


@dataclass(frozen=True, slots=True)
class CameraPose:
    """Canonical camera-to-world pose stored as ``T_world_from_camera``."""

    pose_id: str
    camera_id: str
    frame_id: str
    timestamp_ns: int
    T_world_from_camera: RigidTransform3D
    tracking_confidence: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "pose_id", require_nonempty(self.pose_id, field_name="pose_id"))
        object.__setattr__(
            self, "camera_id", require_nonempty(self.camera_id, field_name="camera_id")
        )
        object.__setattr__(self, "frame_id", require_nonempty(self.frame_id, field_name="frame_id"))
        object.__setattr__(
            self, "timestamp_ns", require_timestamp_ns(self.timestamp_ns)
        )
        if not isinstance(self.T_world_from_camera, RigidTransform3D):
            raise TypeError("T_world_from_camera must be a RigidTransform3D.")
        if self.T_world_from_camera.source_frame_id == self.T_world_from_camera.destination_frame_id:
            raise ReconstructionValidationError(
                "A camera pose must map a camera frame into a distinct world frame."
            )
        if self.tracking_confidence is not None:
            confidence = float(self.tracking_confidence)
            if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                raise ReconstructionValidationError(
                    "tracking_confidence must be finite within [0, 1]."
                )
            object.__setattr__(self, "tracking_confidence", confidence)

    @property
    def T_camera_from_world(self) -> RigidTransform3D:
        return invert_rigid_transform(self.T_world_from_camera, method="pose_inverse")

    @property
    def quaternion_wxyz(self) -> np.ndarray:
        return rotation_matrix_to_quaternion_wxyz(self.T_world_from_camera.matrix[:3, :3])

    @property
    def translation_world_m(self) -> np.ndarray:
        translation = np.array(self.T_world_from_camera.matrix[:3, 3], copy=True)
        translation.setflags(write=False)
        return translation

    def as_dict(self) -> dict[str, Any]:
        return {
            "pose_id": self.pose_id,
            "camera_id": self.camera_id,
            "frame_id": self.frame_id,
            "timestamp_ns": self.timestamp_ns,
            "convention": CANONICAL_POSE_FIELD,
            "source_frame_id": self.T_world_from_camera.source_frame_id,
            "destination_frame_id": self.T_world_from_camera.destination_frame_id,
            "matrix": self.T_world_from_camera.matrix.tolist(),
            "quaternion_order": CANONICAL_QUATERNION_ORDER,
            "quaternion_wxyz": self.quaternion_wxyz.tolist(),
            "translation_world_m": self.translation_world_m.tolist(),
            "tracking_confidence": self.tracking_confidence,
        }

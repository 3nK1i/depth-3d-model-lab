from __future__ import annotations

import numpy as np
import pytest

from core.contracts import (
    AxisConvention,
    CalibrationStatus,
    CameraIntrinsics,
    CoordinateFrame,
    CoordinateFrameKind,
    ImageSize,
    RigidTransform3D,
)
from core.models.metadata import IntrinsicsSource
from core.reconstruction import (
    CANONICAL_LENGTH_UNIT,
    CANONICAL_POSE_FIELD,
    CameraDistortionModel,
    CameraPose,
    ReconstructionCamera,
    ReconstructionValidationError,
    compose_rigid_transforms,
    invert_rigid_transform,
    normalize_quaternion_wxyz,
    quaternion_wxyz_to_rotation_matrix,
    rotation_matrix_to_quaternion_wxyz,
)
from tests.reconstruction_helpers import make_camera, make_pose


def test_valid_reconstruction_camera_uses_canonical_axes_and_metres() -> None:
    camera = make_camera()
    assert camera.camera_frame.axis_convention is AxisConvention.X_RIGHT_Y_DOWN_Z_FORWARD
    assert camera.camera_frame.length_unit.value == CANONICAL_LENGTH_UNIT
    assert camera.distortion_model is CameraDistortionModel.NONE


def test_camera_rejects_invalid_focal_lengths() -> None:
    matrix = np.array([[0.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    with pytest.raises(Exception, match="focal lengths"):
        CameraIntrinsics(
            matrix,
            "rgb",
            IntrinsicsSource.ESTIMATED,
            CalibrationStatus.NOT_CALIBRATED,
        )


def test_image_size_rejects_invalid_dimensions() -> None:
    with pytest.raises(ValueError, match="positive"):
        ImageSize(0, 480)
    with pytest.raises(ValueError, match="positive"):
        ImageSize(640, -1)


def test_camera_rejects_principal_point_outside_image() -> None:
    intrinsics = CameraIntrinsics(
        np.array([[500.0, 0.0, 900.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]),
        "rgb",
        IntrinsicsSource.ESTIMATED,
        CalibrationStatus.NOT_CALIBRATED,
    )
    frame = CoordinateFrame(
        "camera",
        CoordinateFrameKind.CAMERA_GENERIC,
        AxisConvention.X_RIGHT_Y_DOWN_Z_FORWARD,
    )
    with pytest.raises(ReconstructionValidationError, match="principal point"):
        ReconstructionCamera("camera", ImageSize(640, 480), intrinsics, frame)


def test_distortion_coefficient_counts_are_model_specific() -> None:
    camera = make_camera()
    with pytest.raises(ReconstructionValidationError, match="coefficient counts"):
        ReconstructionCamera(
            camera.camera_id,
            camera.image_size,
            camera.intrinsics,
            camera.camera_frame,
            CameraDistortionModel.OPENCV_FISHEYE,
            (0.1, 0.2, 0.3),
        )


def test_invalid_rotation_matrix_is_rejected() -> None:
    matrix = np.eye(4)
    matrix[0, 0] = 2.0
    with pytest.raises(Exception, match="orthonormal"):
        RigidTransform3D("camera", "world", matrix)


def test_pose_inversion_round_trip() -> None:
    pose = make_pose(2)
    inverse = pose.T_camera_from_world
    round_trip = invert_rigid_transform(inverse)
    assert np.allclose(round_trip.matrix, pose.T_world_from_camera.matrix)
    assert inverse.source_frame_id == "world"
    assert inverse.destination_frame_id == "camera-1-frame"


def test_pose_composition_uses_destination_from_source_notation() -> None:
    first_matrix = np.eye(4)
    first_matrix[0, 3] = 1.0
    second_matrix = np.eye(4)
    second_matrix[1, 3] = 2.0
    first = RigidTransform3D("camera", "rig", first_matrix)
    second = RigidTransform3D("rig", "world", second_matrix)
    result = compose_rigid_transforms(first, second)
    assert result.source_frame_id == "camera"
    assert result.destination_frame_id == "world"
    assert np.allclose(result.matrix[:3, 3], [1.0, 2.0, 0.0])


def test_camera_to_world_and_world_to_camera_round_trip() -> None:
    pose = make_pose(3)
    identity = compose_rigid_transforms(
        pose.T_world_from_camera,
        pose.T_camera_from_world,
    )
    assert np.allclose(identity.matrix, np.eye(4), atol=1e-9)


def test_quaternion_normalization_and_matrix_round_trip() -> None:
    normalized = normalize_quaternion_wxyz([-2.0, 0.0, 0.0, 0.0])
    assert np.allclose(normalized, [1.0, 0.0, 0.0, 0.0])
    rotation = quaternion_wxyz_to_rotation_matrix([0.9238795325, 0.0, 0.3826834324, 0.0])
    restored = rotation_matrix_to_quaternion_wxyz(rotation)
    assert np.allclose(
        quaternion_wxyz_to_rotation_matrix(restored),
        rotation,
        atol=1e-9,
    )


def test_pose_serialization_declares_convention_explicitly() -> None:
    payload = make_pose(0).as_dict()
    assert payload["convention"] == CANONICAL_POSE_FIELD
    assert payload["source_frame_id"] == "camera-1-frame"
    assert payload["destination_frame_id"] == "world"
    assert payload["quaternion_order"] == "wxyz"

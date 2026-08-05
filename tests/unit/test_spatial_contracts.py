from __future__ import annotations

import numpy as np
import pytest

from core.contracts import (
    AxisConvention,
    CalibrationStatus,
    CameraIntrinsics,
    CoordinateFrame,
    CoordinateFrameKind,
    FrameReference,
    ImageSize,
    ImageTransform2D,
    PixelCoordinateSpace,
    PixelSpaceRole,
    RigidTransform3D,
)
from core.contracts.validation import ContractValidationError
from core.models.metadata import IntrinsicsSource


def _frame() -> FrameReference:
    return FrameReference(1, 2, 3.0, ImageSize(640, 480))


def _space(space_id: str, size: ImageSize, role: PixelSpaceRole) -> PixelCoordinateSpace:
    return PixelCoordinateSpace(space_id, _frame(), size, role)


def test_authoritative_image_transform_matches_crop_resize_padding() -> None:
    source = _space("rgb", ImageSize(640, 480), PixelSpaceRole.RGB)
    destination = _space("model", ImageSize(324, 244), PixelSpaceRole.MODEL_INPUT)
    matrix = np.array(
        [[0.5, 0.0, -8.0], [0.0, 0.5, -3.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    transform = ImageTransform2D(
        source_space=source,
        destination_space=destination,
        matrix=matrix,
        crop_xyxy=(20.0, 10.0, 620.0, 470.0),
        resized_size=ImageSize(300, 230),
        padding_ltrb=(2, 2, 22, 12),
        interpolation="linear",
    )
    assert np.allclose(transform.inverse_matrix @ transform.matrix, np.eye(3))


def test_image_transform_rejects_matrix_and_description_disagreement() -> None:
    source = _space("rgb", ImageSize(100, 100), PixelSpaceRole.RGB)
    destination = _space("mask", ImageSize(50, 50), PixelSpaceRole.MASK)
    with pytest.raises(ContractValidationError, match="disagrees"):
        ImageTransform2D(
            source_space=source,
            destination_space=destination,
            matrix=np.eye(3),
            resized_size=ImageSize(50, 50),
        )


def test_identity_rigid_transform_allows_equal_frame_ids_only_for_identity() -> None:
    identity = RigidTransform3D("camera", "camera", np.eye(4))
    assert np.array_equal(identity.matrix, np.eye(4))

    translated = np.eye(4)
    translated[0, 3] = 1.0
    with pytest.raises(ContractValidationError, match="identity"):
        RigidTransform3D("camera", "camera", translated)

    valid = RigidTransform3D("camera", "world", translated)
    assert valid.source_frame_id == "camera"
    assert valid.destination_frame_id == "world"


def test_rigid_transform_rotation_validation() -> None:
    invalid = np.eye(4)
    invalid[0, 0] = -1.0
    with pytest.raises(ContractValidationError, match="positive unit determinant"):
        RigidTransform3D("a", "b", invalid)


def test_predicted_and_estimated_intrinsics_remain_not_calibrated() -> None:
    matrix = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    intrinsics = CameraIntrinsics(
        matrix=matrix,
        pixel_space_id="rgb",
        source=IntrinsicsSource.MODEL_PREDICTED,
        calibration_status=CalibrationStatus.NOT_CALIBRATED,
    )
    assert intrinsics.calibration_status is CalibrationStatus.NOT_CALIBRATED

    with pytest.raises(ContractValidationError, match="Not Calibrated"):
        CameraIntrinsics(
            matrix=matrix,
            pixel_space_id="rgb",
            source=IntrinsicsSource.ESTIMATED,
            calibration_status=CalibrationStatus.CALIBRATED,
            calibration_record_id="bad",
        )


def test_future_coordinate_frames_are_contract_only() -> None:
    frame = CoordinateFrame(
        frame_id="map",
        kind=CoordinateFrameKind.RECONSTRUCTION_MAP,
        axis_convention=AxisConvention.X_RIGHT_Y_UP_Z_BACK,
    )
    assert frame.kind is CoordinateFrameKind.RECONSTRUCTION_MAP

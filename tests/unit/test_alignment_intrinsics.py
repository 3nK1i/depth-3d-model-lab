from __future__ import annotations

import numpy as np
import pytest

from core.alignment.factories import declare_crop_resize_padding, declare_full_frame_resize
from core.alignment.intrinsics import preserve_native_intrinsics, transform_intrinsics
from core.contracts.common import FrameReference, ImageSize
from core.contracts.spatial import (
    CalibrationStatus,
    CameraIntrinsics,
    PixelCoordinateSpace,
    PixelSpaceRole,
)
from core.contracts.validation import ContractValidationError
from core.models.metadata import IntrinsicsSource


def _spaces():
    frame = FrameReference(1, 1, 0.0, ImageSize(8, 6))
    source = PixelCoordinateSpace("rgb", frame, ImageSize(8, 6), PixelSpaceRole.RGB)
    destination = PixelCoordinateSpace("depth", frame, ImageSize(4, 3), PixelSpaceRole.DEPTH)
    return source, destination


def test_full_frame_resize_scales_focal_lengths_and_principal_point():
    source, destination = _spaces()
    intrinsics = CameraIntrinsics(
        np.array([[8.0, 0.0, 4.0], [0.0, 6.0, 3.0], [0.0, 0.0, 1.0]]),
        "rgb",
        IntrinsicsSource.MODEL_PREDICTED,
        CalibrationStatus.NOT_CALIBRATED,
    )
    transformed = transform_intrinsics(
        intrinsics, declare_full_frame_resize(source, destination)
    )
    assert transformed.pixel_space_id == "depth"
    assert transformed.matrix == pytest.approx(
        np.array([[4.0, 0.0, 2.0], [0.0, 3.0, 1.5], [0.0, 0.0, 1.0]])
    )
    assert transformed.calibration_status is CalibrationStatus.NOT_CALIBRATED


def test_crop_resize_padding_transforms_intrinsics_with_same_matrix():
    source, _ = _spaces()
    destination = PixelCoordinateSpace(
        "padded", source.frame_reference, ImageSize(6, 6), PixelSpaceRole.DEPTH
    )
    transform = declare_crop_resize_padding(
        source,
        destination,
        crop_xyxy=(2.0, 1.0, 6.0, 5.0),
        resized_size=ImageSize(4, 4),
        padding_ltrb=(1, 1, 1, 1),
    )
    intrinsics = CameraIntrinsics(
        np.array([[8.0, 0.0, 4.0], [0.0, 8.0, 3.0], [0.0, 0.0, 1.0]]),
        "rgb",
        IntrinsicsSource.ESTIMATED,
        CalibrationStatus.NOT_CALIBRATED,
    )
    transformed = transform_intrinsics(intrinsics, transform)
    assert transformed.matrix == pytest.approx(transform.matrix @ intrinsics.matrix)


def test_native_intrinsics_are_preserved_only_for_matching_target_space():
    intrinsics = CameraIntrinsics(
        np.eye(3),
        "depth",
        IntrinsicsSource.ESTIMATED,
        CalibrationStatus.NOT_CALIBRATED,
    )
    assert preserve_native_intrinsics(intrinsics, "depth") is intrinsics
    with pytest.raises(ContractValidationError):
        preserve_native_intrinsics(intrinsics, "other")

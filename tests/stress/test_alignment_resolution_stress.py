from __future__ import annotations

import numpy as np
import pytest

from core.alignment import AlignmentRequest, DepthAlignmentInput, align_frame, declare_full_frame_resize
from core.contracts.common import FrameReference, ImageSize, ModelProvenance
from core.contracts.spatial import CalibrationStatus, CameraIntrinsics, PixelCoordinateSpace, PixelSpaceRole
from core.models.metadata import ConfidenceKind, DepthScale, IntrinsicsSource
from core.perception.result import PerceptionFrameResult


@pytest.mark.parametrize(
    ("source_size", "depth_size"),
    [
        ((192, 108), (96, 54)),
        ((108, 192), (54, 96)),
        ((128, 128), (64, 64)),
        ((160, 90), (80, 60)),
        ((90, 160), (60, 80)),
    ],
)
def test_alignment_handles_landscape_portrait_square_and_nonuniform_resolutions(source_size, depth_size):
    frame = FrameReference(2, source_size[0] + depth_size[0], 0.0, ImageSize(*source_size))
    rgb_space = PixelCoordinateSpace("rgb", frame, ImageSize(*source_size), PixelSpaceRole.RGB)
    depth_space = PixelCoordinateSpace("depth", frame, ImageSize(*depth_size), PixelSpaceRole.DEPTH)
    depth = np.ones((depth_size[1], depth_size[0]), np.float32)
    intrinsics = CameraIntrinsics(
        np.array([[depth_size[0], 0.0, depth_size[0] / 2], [0.0, depth_size[1], depth_size[1] / 2], [0.0, 0.0, 1.0]]),
        "depth",
        IntrinsicsSource.ESTIMATED,
        CalibrationStatus.NOT_CALIBRATED,
    )
    depth_input = DepthAlignmentInput(
        frame,
        depth_space,
        depth,
        None,
        None,
        ConfidenceKind.NONE,
        intrinsics,
        DepthScale.RELATIVE,
        ModelProvenance("depth", "Depth", "depth"),
    )
    result = align_frame(
        AlignmentRequest(
            frame,
            np.zeros((source_size[1], source_size[0], 3), np.uint8),
            rgb_space,
            depth_input,
            PerceptionFrameResult(frame, (), ()),
            (declare_full_frame_resize(rgb_space, depth_space),),
        )
    )
    assert result.aligned_rgb.shape == (depth_size[1], depth_size[0], 3)
    assert result.depth_scale is DepthScale.RELATIVE

from __future__ import annotations

import numpy as np

from core.alignment import AlignmentRequest, DepthAlignmentInput, align_frame, declare_full_frame_resize
from core.contracts.common import FrameReference, ImageSize, ModelProvenance
from core.contracts.spatial import CalibrationStatus, CameraIntrinsics, PixelCoordinateSpace, PixelSpaceRole
from core.models.metadata import ConfidenceKind, DepthScale, IntrinsicsSource
from core.perception.result import PerceptionFrameResult


def test_repeated_alignment_does_not_mutate_inputs_or_accumulate_state():
    rgb_template = np.arange(48 * 64 * 3, dtype=np.uint8).reshape(48, 64, 3)
    depth_template = np.full((24, 32), 2.0, np.float32)
    for frame_id in range(120):
        frame = FrameReference(1, frame_id, float(frame_id), ImageSize(64, 48))
        rgb_space = PixelCoordinateSpace("rgb", frame, frame.source_size, PixelSpaceRole.RGB)
        depth_space = PixelCoordinateSpace("depth", frame, ImageSize(32, 24), PixelSpaceRole.DEPTH)
        intrinsics = CameraIntrinsics(
            np.array([[32.0, 0.0, 16.0], [0.0, 32.0, 12.0], [0.0, 0.0, 1.0]]),
            "depth",
            IntrinsicsSource.ESTIMATED,
            CalibrationStatus.NOT_CALIBRATED,
        )
        depth_input = DepthAlignmentInput(
            frame,
            depth_space,
            depth_template,
            None,
            None,
            ConfidenceKind.NONE,
            intrinsics,
            DepthScale.METRIC,
            ModelProvenance("depth", "Depth", "depth"),
        )
        result = align_frame(
            AlignmentRequest(
                frame,
                rgb_template,
                rgb_space,
                depth_input,
                PerceptionFrameResult(frame, (), ()),
                (declare_full_frame_resize(rgb_space, depth_space),),
            )
        )
        assert result.frame_reference.frame_id == frame_id
        assert result.depth.shape == (24, 32)
    assert rgb_template.flags.writeable
    assert depth_template.flags.writeable

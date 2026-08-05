from __future__ import annotations

import pickle

import numpy as np
import pytest

from core.alignment.contracts import (
    AlignmentPolicy,
    AlignmentRequest,
    AlignedFrameBundle,
    AlignedObjectObservation,
    AlignmentStatus,
    DepthAlignmentInput,
)
from core.alignment.factories import declare_full_frame_resize
from core.contracts.common import FrameReference, ImageSize, ModelProvenance
from core.contracts.spatial import (
    CalibrationStatus,
    CameraIntrinsics,
    PixelCoordinateSpace,
    PixelSpaceRole,
)
from core.contracts.validation import ContractValidationError
from core.models.metadata import ConfidenceKind, DepthScale, IntrinsicsSource
from core.perception.result import PerceptionFrameResult


def _base():
    frame = FrameReference(1, 2, 3.0, ImageSize(8, 6))
    rgb_space = PixelCoordinateSpace("rgb", frame, frame.source_size, PixelSpaceRole.RGB)
    depth_space = PixelCoordinateSpace("depth", frame, ImageSize(4, 3), PixelSpaceRole.DEPTH)
    intrinsics = CameraIntrinsics(
        np.array([[4.0, 0.0, 2.0], [0.0, 4.0, 1.5], [0.0, 0.0, 1.0]]),
        "depth",
        IntrinsicsSource.ESTIMATED,
        CalibrationStatus.NOT_CALIBRATED,
    )
    provenance = ModelProvenance("mock-depth", "Mock Depth", "depth")
    depth_input = DepthAlignmentInput(
        frame,
        depth_space,
        np.ones((3, 4), np.float32),
        np.ones((3, 4), bool),
        None,
        ConfidenceKind.NONE,
        intrinsics,
        DepthScale.SYNTHETIC,
        provenance,
    )
    perception = PerceptionFrameResult(frame, (), ())
    return frame, rgb_space, depth_space, depth_input, perception


def test_depth_contract_is_immutable_and_pickle_restores_readonly_arrays():
    _, _, _, depth_input, _ = _base()
    restored = pickle.loads(pickle.dumps(depth_input))
    assert not restored.depth.flags.writeable
    assert not restored.valid_mask.flags.writeable
    with pytest.raises(ValueError):
        restored.depth[0, 0] = 2


def test_depth_contract_rejects_shape_and_confidence_semantic_mismatch():
    frame, _, depth_space, _, _ = _base()
    intrinsics = CameraIntrinsics(
        np.eye(3), "depth", IntrinsicsSource.ESTIMATED, CalibrationStatus.NOT_CALIBRATED
    )
    provenance = ModelProvenance("d", "Depth", "depth")
    with pytest.raises(ContractValidationError):
        DepthAlignmentInput(
            frame,
            depth_space,
            np.ones((2, 4)),
            None,
            None,
            ConfidenceKind.NONE,
            intrinsics,
            DepthScale.UNKNOWN,
            provenance,
        )
    with pytest.raises(ContractValidationError):
        DepthAlignmentInput(
            frame,
            depth_space,
            np.ones((3, 4)),
            None,
            np.ones((3, 4)),
            ConfidenceKind.NONE,
            intrinsics,
            DepthScale.UNKNOWN,
            provenance,
        )


def test_alignment_request_requires_exact_cross_pipeline_frame_identity():
    frame, rgb_space, depth_space, depth_input, perception = _base()
    transform = declare_full_frame_resize(rgb_space, depth_space)
    request = AlignmentRequest(
        frame,
        np.zeros((6, 8, 3), np.uint8),
        rgb_space,
        depth_input,
        perception,
        (transform,),
    )
    assert not request.rgb.flags.writeable
    other = FrameReference(1, 3, 3.0, ImageSize(8, 6))
    other_perception = PerceptionFrameResult(other, (), ())
    with pytest.raises(ContractValidationError):
        AlignmentRequest(
            frame,
            np.zeros((6, 8, 3), np.uint8),
            rgb_space,
            depth_input,
            other_perception,
            (transform,),
        )


def test_alignment_policy_rejects_fractional_mask_resampling():
    from core.alignment.contracts import ResamplingMethod

    with pytest.raises(ContractValidationError):
        AlignmentPolicy(mask_interpolation=ResamplingMethod.LINEAR)


def test_aligned_object_and_bundle_validate_target_shapes():
    frame, _, depth_space, depth_input, _ = _base()
    observation = AlignedObjectObservation(
        "obj",
        0,
        "object",
        depth_space,
        aligned_mask=np.ones((3, 4), bool),
        aligned_foreground_pixels=12,
        foreground_retention_ratio=1.0,
    )
    bundle = AlignedFrameBundle(
        frame,
        depth_space,
        np.zeros((3, 4, 3), np.uint8),
        depth_input.depth,
        np.ones((3, 4), bool),
        np.ones((3, 4), bool),
        None,
        np.zeros((3, 4), bool),
        ConfidenceKind.NONE,
        depth_input.intrinsics,
        depth_input.depth_scale,
        1,
        depth_input.depth_model_provenance,
        (),
        (observation,),
        (),
        AlignmentStatus.SUCCESS,
    )
    restored = pickle.loads(pickle.dumps(bundle))
    assert not restored.aligned_rgb.flags.writeable
    assert not restored.aligned_observations[0].aligned_mask.flags.writeable
    with pytest.raises(ContractValidationError):
        AlignedObjectObservation(
            "bad",
            0,
            "object",
            depth_space,
            aligned_mask=np.ones((2, 4), bool),
            aligned_foreground_pixels=8,
        )

@pytest.mark.parametrize(
    "other",
    [
        FrameReference(1, 2, 3.1, ImageSize(8, 6)),
        FrameReference(1, 2, 3.0, ImageSize(9, 6)),
        FrameReference(2, 2, 3.0, ImageSize(8, 6)),
    ],
)
def test_alignment_request_rejects_timestamp_source_size_and_session_mismatch(other):
    frame, rgb_space, depth_space, depth_input, _ = _base()
    transform = declare_full_frame_resize(rgb_space, depth_space)
    perception = PerceptionFrameResult(other, (), ())
    with pytest.raises(ContractValidationError):
        AlignmentRequest(
            frame,
            np.zeros((6, 8, 3), np.uint8),
            rgb_space,
            depth_input,
            perception,
            (transform,),
        )

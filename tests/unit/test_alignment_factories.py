from __future__ import annotations

import numpy as np
import pytest

from core.alignment.factories import depth_input_from_inference_result, make_depth_space
from core.contracts.common import FrameReference, ImageSize
from core.contracts.spatial import CalibrationStatus
from core.contracts.validation import ContractValidationError
from core.models.metadata import ConfidenceKind, DepthScale, IntrinsicsSource
from core.runtime.messages import InferenceResult


def test_inference_result_factory_preserves_phase6_semantics():
    frame = FrameReference(7, 9, 4.0, ImageSize(8, 6))
    result = InferenceResult(
        7,
        9,
        "mock",
        np.ones((3, 4), np.float32),
        1.0,
        intrinsics=np.array([[4.0, 0.0, 2.0], [0.0, 4.0, 1.5], [0.0, 0.0, 1.0]]),
        valid_mask=np.ones((3, 4), bool),
        is_metric=True,
        model_name="Mock",
        precision="fp32",
        confidence=np.full((3, 4), 0.5, np.float32),
        depth_scale="metric",
        intrinsics_source="model_predicted",
        confidence_kind="confidence",
    )
    converted = depth_input_from_inference_result(result, frame)
    assert converted.depth_scale is DepthScale.METRIC
    assert converted.confidence_kind is ConfidenceKind.CONFIDENCE
    assert converted.intrinsics.source is IntrinsicsSource.MODEL_PREDICTED
    assert converted.intrinsics.calibration_status is CalibrationStatus.NOT_CALIBRATED
    assert converted.depth_space.size == ImageSize(4, 3)


def test_factory_rejects_frame_mismatch_and_missing_depth_intrinsics():
    frame = FrameReference(1, 1, 0.0, ImageSize(4, 3))
    mismatch = InferenceResult(1, 2, "m", np.ones((2, 2)), 1.0, intrinsics=np.eye(3))
    with pytest.raises(ContractValidationError):
        depth_input_from_inference_result(mismatch, frame)
    no_depth = InferenceResult(1, 1, "m", None, 1.0, intrinsics=np.eye(3))
    with pytest.raises(ContractValidationError):
        depth_input_from_inference_result(no_depth, frame)


def test_make_depth_space_handles_portrait_landscape_and_square():
    for shape in ((3, 5), (5, 3), (4, 4)):
        frame = FrameReference(1, shape[0], 0.0, ImageSize(10, 10))
        space = make_depth_space(frame, np.ones(shape))
        assert space.size == ImageSize(shape[1], shape[0])

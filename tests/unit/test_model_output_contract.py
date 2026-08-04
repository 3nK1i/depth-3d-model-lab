"""Normalized model output validation for Phase 6 multi-model adapters."""

from __future__ import annotations

import numpy as np
import pytest

from core.models.catalog import (
    DEPTH_PRO_MODEL_ID,
    MOGE2_MODEL_ID,
    MOCK_MODEL_ID,
    UNIDEPTH_V2_MODEL_ID,
)
from core.models.registry import get_model_descriptor
from core.models.result import GeometryResult
from core.models.validation import ModelOutputContractError, normalize_geometry_result
from core.runtime.messages import FrameRequest


def _frame() -> FrameRequest:
    return FrameRequest(
        session_id=7,
        frame_id=11,
        image=np.zeros((4, 6, 3), dtype=np.uint8),
    )


def _result(**overrides) -> GeometryResult:
    values = {
        "session_id": 7,
        "frame_id": 11,
        "timestamp": 1.0,
        "rgb": np.zeros((4, 6, 3), dtype=np.uint8),
        "depth": np.ones((4, 6), dtype=np.float32),
        "model_id": MOCK_MODEL_ID,
        "model_name": "",
        "is_metric": True,  # registry must correct this for the mock
    }
    values.update(overrides)
    return GeometryResult(**values)


def test_registry_normalizes_result_identity_and_depth_semantics() -> None:
    normalized = normalize_geometry_result(
        _result(),
        get_model_descriptor(MOCK_MODEL_ID),
        _frame(),
    )
    assert normalized.model_id == MOCK_MODEL_ID
    assert normalized.model_name == "Development Mock"
    assert normalized.is_metric is False
    assert normalized.depth_scale == "synthetic"
    assert normalized.intrinsics_source == "none"
    assert normalized.confidence_kind == "none"


def test_metric_model_semantics_and_predicted_intrinsics_are_normalized() -> None:
    result = _result(
        model_id=MOGE2_MODEL_ID,
        points=np.zeros((4, 6, 3), dtype=np.float32),
        normals=np.zeros((4, 6, 3), dtype=np.float32),
        intrinsics=np.eye(3, dtype=np.float32),
        valid_mask=np.ones((4, 6), dtype=bool),
    )
    normalized = normalize_geometry_result(
        result,
        get_model_descriptor(MOGE2_MODEL_ID),
        _frame(),
    )
    assert normalized.is_metric is True
    assert normalized.depth_scale == "metric"
    assert normalized.intrinsics_source == "model_predicted"


def test_metric_model_uses_estimated_intrinsics_fallback_when_prediction_is_missing() -> None:
    normalized = normalize_geometry_result(
        _result(model_id=MOGE2_MODEL_ID),
        get_model_descriptor(MOGE2_MODEL_ID),
        _frame(),
    )
    assert normalized.intrinsics is not None
    assert normalized.intrinsics.shape == (3, 3)
    assert normalized.intrinsics_source == "estimated"
    assert normalized.intrinsics[0, 0] > 0
    assert normalized.intrinsics[1, 1] > 0


def test_stale_or_wrong_model_identity_is_rejected() -> None:
    with pytest.raises(ModelOutputContractError, match="stale identifiers"):
        normalize_geometry_result(
            _result(frame_id=12),
            get_model_descriptor(MOCK_MODEL_ID),
            _frame(),
        )

    with pytest.raises(ModelOutputContractError, match="returned model id"):
        normalize_geometry_result(
            _result(model_id=MOGE2_MODEL_ID),
            get_model_descriptor(MOCK_MODEL_ID),
            _frame(),
        )


def test_depth_and_optional_output_shapes_are_validated() -> None:
    descriptor = get_model_descriptor(MOGE2_MODEL_ID)
    with pytest.raises(ModelOutputContractError, match="2D"):
        normalize_geometry_result(
            _result(model_id=MOGE2_MODEL_ID, depth=np.ones((4, 6, 1))),
            descriptor,
            _frame(),
        )

    with pytest.raises(ModelOutputContractError, match="3×3"):
        normalize_geometry_result(
            _result(
                model_id=MOGE2_MODEL_ID,
                intrinsics=np.eye(4, dtype=np.float32),
            ),
            descriptor,
            _frame(),
        )

    with pytest.raises(ModelOutputContractError, match="same H×W"):
        normalize_geometry_result(
            _result(
                model_id=MOGE2_MODEL_ID,
                valid_mask=np.ones((2, 3), dtype=bool),
            ),
            descriptor,
            _frame(),
        )


def test_adapter_cannot_return_outputs_not_declared_by_registry() -> None:
    with pytest.raises(ModelOutputContractError, match="undeclared output 'points'"):
        normalize_geometry_result(
            _result(points=np.zeros((4, 6, 3), dtype=np.float32)),
            get_model_descriptor(MOCK_MODEL_ID),
            _frame(),
        )


def test_point_stride_must_be_positive() -> None:
    with pytest.raises(ModelOutputContractError, match="point_stride"):
        normalize_geometry_result(
            _result(point_stride=0),
            get_model_descriptor(MOCK_MODEL_ID),
            _frame(),
        )

def test_sampled_point_map_shape_is_validated() -> None:
    descriptor = get_model_descriptor(MOGE2_MODEL_ID)
    with pytest.raises(ModelOutputContractError, match="sampled by point_stride"):
        normalize_geometry_result(
            _result(
                model_id=MOGE2_MODEL_ID,
                point_stride=2,
                points=np.zeros((4, 6, 3), dtype=np.float32),
            ),
            descriptor,
            _frame(),
        )

def test_geometry_result_phase5_positional_constructor_remains_compatible() -> None:
    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    depth = np.ones((2, 3), dtype=np.float32)
    result = GeometryResult(
        1,
        2,
        3.0,
        rgb,
        depth,
        None,
        None,
        None,
        None,
        None,
        None,
        1,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        True,
        "legacy-model",
        "Legacy Model",
        "FP32",
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        6.0,
    )
    assert result.model_id == "legacy-model"
    assert result.vram_total_gib == 6.0
    assert result.depth_scale == ""

def test_depth_pro_predicted_and_estimated_intrinsics_semantics() -> None:
    predicted = normalize_geometry_result(
        _result(
            model_id=DEPTH_PRO_MODEL_ID,
            intrinsics=np.array(
                [[500.0, 0.0, 2.5], [0.0, 500.0, 1.5], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            ),
            valid_mask=np.ones((4, 6), dtype=bool),
        ),
        get_model_descriptor(DEPTH_PRO_MODEL_ID),
        _frame(),
    )
    assert predicted.is_metric is True
    assert predicted.intrinsics_source == "model_predicted"

    estimated = normalize_geometry_result(
        _result(
            model_id=DEPTH_PRO_MODEL_ID,
            valid_mask=np.ones((4, 6), dtype=bool),
        ),
        get_model_descriptor(DEPTH_PRO_MODEL_ID),
        _frame(),
    )
    assert estimated.intrinsics is not None
    assert estimated.intrinsics_source == "estimated"



def test_unidepth_v2_uncertainty_and_geometry_semantics() -> None:
    result = _result(
        model_id=UNIDEPTH_V2_MODEL_ID,
        points=np.zeros((4, 6, 3), dtype=np.float32),
        intrinsics=np.array(
            [[500.0, 0.0, 2.5], [0.0, 500.0, 1.5], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
        confidence=np.full((4, 6), 0.2, dtype=np.float32),
        valid_mask=np.ones((4, 6), dtype=bool),
    )
    normalized = normalize_geometry_result(
        result, get_model_descriptor(UNIDEPTH_V2_MODEL_ID), _frame()
    )
    assert normalized.is_metric is True
    assert normalized.depth_scale == "metric"
    assert normalized.intrinsics_source == "model_predicted"
    assert normalized.confidence_kind == "uncertainty"

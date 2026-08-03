"""Development mock adapter tests used by lifecycle and worker validation."""

from __future__ import annotations

import time

import numpy as np
import pytest

from core.models.adapters.mock_adapter import MockGeometryModel, SimulatedOOMError
from core.models.catalog import MOCK_MODEL_ID
from core.runtime.messages import FrameRequest


def _loaded(**config) -> MockGeometryModel:
    model = MockGeometryModel()
    model.load("cpu", "fp32", {"latency_ms": 0.0, **config})
    return model


def test_mock_generates_deterministic_room_scale_depth() -> None:
    model = _loaded(depth_near_m=0.5, depth_far_m=4.0)
    image = np.zeros((3, 5, 3), dtype=np.uint8)

    result = model.infer(FrameRequest(2, 7, image))

    assert result.model_id == MOCK_MODEL_ID
    assert result.depth is not None and result.depth.shape == (3, 5)
    assert result.depth.dtype == np.float32
    assert result.depth.min() == pytest.approx(0.5)
    assert result.depth.max() == pytest.approx(4.0)
    assert result.precision == "FP32"
    assert result.is_metric is False
    assert result.worker_total_ms is not None


def test_mock_reports_nonnegative_queue_wait() -> None:
    model = _loaded()
    request = FrameRequest(1, 1, np.zeros((2, 2, 3), np.uint8), submitted_at=time.monotonic() - 0.01)
    result = model.infer(request)
    assert result.queue_wait_ms is not None and result.queue_wait_ms >= 0.0


def test_mock_can_simulate_oom() -> None:
    model = _loaded(simulate_oom=True)
    with pytest.raises(SimulatedOOMError, match="out-of-memory"):
        model.infer(FrameRequest(1, 1, np.zeros((2, 2, 3), np.uint8)))


def test_mock_requires_load_and_cannot_infer_after_unload() -> None:
    model = MockGeometryModel()
    with pytest.raises(RuntimeError, match="before load"):
        model.warmup()

    model.load("cpu", "fp32", {"latency_ms": 0.0})
    model.unload()
    with pytest.raises(RuntimeError, match="before load"):
        model.infer(FrameRequest(1, 1, np.zeros((2, 2, 3), np.uint8)))


def test_mock_metadata_never_presents_output_as_real_ai() -> None:
    metadata = MockGeometryModel().get_metadata()
    assert metadata["id"] == MOCK_MODEL_ID
    assert metadata["display_name"] == "Development Mock"
    assert metadata["is_metric"] is False
    assert metadata["outputs"]["points"] is False

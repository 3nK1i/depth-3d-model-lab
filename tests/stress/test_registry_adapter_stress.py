"""Dependency-light Phase 6 registry/factory lifecycle stress tests."""

from __future__ import annotations

import numpy as np
import pytest

from core.models.catalog import MOCK_MODEL_ID
from core.models.registry import (
    build_model,
    get_model_descriptor,
    resolve_model_runtime_config,
)
from core.models.runtime_config import ModelRuntimeContext
from core.models.validation import normalize_geometry_result
from core.runtime.messages import FrameRequest

pytestmark = pytest.mark.stress


def test_repeated_registry_factory_mock_lifecycle_is_stable() -> None:
    descriptor = get_model_descriptor(MOCK_MODEL_ID)
    config = resolve_model_runtime_config(
        MOCK_MODEL_ID,
        ModelRuntimeContext(overrides={"latency_ms": 0.0}),
    )
    image = np.zeros((24, 32, 3), dtype=np.uint8)

    for cycle in range(20):
        model = build_model(MOCK_MODEL_ID)
        model.load(
            device=str(config["device"]),
            precision=str(config["precision"]),
            config=config,
        )
        model.warmup()
        frame = FrameRequest(cycle + 1, cycle + 1, image, point_budget=500)
        result = normalize_geometry_result(model.infer(frame), descriptor, frame)

        assert result.model_id == MOCK_MODEL_ID
        assert result.depth is not None and result.depth.shape == image.shape[:2]
        assert result.depth_scale == "synthetic"
        assert model.unload()["released"] is True

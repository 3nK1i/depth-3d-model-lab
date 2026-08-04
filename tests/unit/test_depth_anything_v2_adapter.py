"""Depth Anything V2 adapter contract tests without external downloads."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from core.models.adapters.depth_anything_v2_adapter import (
    DepthAnythingV2CheckpointError,
    DepthAnythingV2ConfigurationError,
    DepthAnythingV2GeometryModel,
)
from core.models.catalog import DEPTH_ANYTHING_V2_MODEL_ID
from core.models.registry import get_model_descriptor
from core.models.validation import normalize_geometry_result
from core.runtime.messages import FrameRequest


class _Model:
    def __init__(self, malformed: bool = False) -> None:
        self.malformed = malformed
        self.loaded_state = None
        self.strict = None
        self.float_called = False
        self.to_device = None
        self.eval_called = False
        self.calls: list[tuple[np.ndarray, int]] = []

    def load_state_dict(self, state, strict=True):
        self.loaded_state = state
        self.strict = strict

    def float(self):
        self.float_called = True
        return self

    def to(self, device):
        self.to_device = device
        return self

    def eval(self):
        self.eval_called = True
        return self

    def infer_image(self, image, input_size):
        self.calls.append((image.copy(), input_size))
        if self.malformed:
            return np.ones((2, 2, 2), dtype=np.float32)
        h, w = image.shape[:2]
        return np.linspace(0.1, 2.0, h * w, dtype=np.float32).reshape(h, w)


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "Depth-Anything-V2"
    (source / "depth_anything_v2").mkdir(parents=True)
    (source / "depth_anything_v2" / "dpt.py").write_text("# test\n")
    return source


def _checkpoint(tmp_path: Path) -> Path:
    path = tmp_path / "model.pth"
    path.write_bytes(b"weights")
    return path


def _config(source: Path, checkpoint: Path, **overrides) -> dict:
    values = {
        "source_dir": str(source),
        "checkpoint": str(checkpoint),
        "encoder": "vits",
        "model_label": "Depth Anything V2 Test",
        "input_size": 518,
        "virtual_fov_degrees": 60.0,
        "point_budget": 4,
        "warmup_size": 64,
        "warmup_runs": 1,
    }
    values.update(overrides)
    return values


def _frame(height: int = 3, width: int = 4) -> FrameRequest:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[..., 0] = 10
    image[..., 1] = 20
    image[..., 2] = 30
    return FrameRequest(3, 7, image, point_budget=4)


def test_load_infer_normalize_and_unload_contract(tmp_path) -> None:
    source, checkpoint = _source(tmp_path), _checkpoint(tmp_path)
    model = _Model()
    adapter = DepthAnythingV2GeometryModel(
        model_loader=lambda *_args: model,
        checkpoint_loader=lambda _path: {"weight": torch.tensor(1)},
        torch_module=torch,
    )
    adapter.load("cpu", "fp16", _config(source, checkpoint))
    assert model.strict is True
    assert model.float_called is True
    assert str(model.to_device) == "cpu"
    assert model.eval_called is True

    result = adapter.infer(_frame())
    assert result.model_id == DEPTH_ANYTHING_V2_MODEL_ID
    assert result.model_name == "Depth Anything V2 Test"
    assert result.precision == "FP32"
    assert result.is_metric is False
    assert result.depth_scale == "relative"
    assert result.intrinsics_source == "estimated"
    assert result.depth.shape == (3, 4)
    assert result.points.shape == (2, 2, 3)
    assert result.colors.shape == (2, 2, 3)
    assert result.colors[0, 0].tolist() == [30, 20, 10]
    assert result.valid_mask.all()
    assert result.intrinsics.shape == (3, 3)
    # Higher official inverse depth must map to a nearer relative distance.
    assert result.depth[-1, -1] < result.depth[0, 0]
    assert model.calls[-1][1] == 518
    assert model.calls[-1][0][0, 0].tolist() == [10, 20, 30]  # official API expects BGR

    normalized = normalize_geometry_result(
        result, get_model_descriptor(DEPTH_ANYTHING_V2_MODEL_ID), _frame()
    )
    assert normalized.depth_scale == "relative"
    assert normalized.is_metric is False

    metadata = adapter.get_metadata()
    assert metadata["scale_status"] == "relative_not_metric"
    assert metadata["depth_representation"] == "relative_distance_from_inverse_depth"
    assert adapter.unload()["released"] is True


def test_relative_conversion_rejects_invalid_pixels_and_handles_flat_output() -> None:
    disparity = np.array([[0.0, np.nan, 2.0], [2.0, 2.0, np.inf]], dtype=np.float32)
    depth, valid = DepthAnythingV2GeometryModel._relative_depth_from_disparity(disparity)
    assert valid.tolist() == [[False, False, True], [True, True, False]]
    assert np.allclose(depth[valid], 1.0)
    assert np.all(depth[~valid] == 0.0)


def test_warmup_and_resolution_variants(tmp_path) -> None:
    source, checkpoint = _source(tmp_path), _checkpoint(tmp_path)
    model = _Model()
    adapter = DepthAnythingV2GeometryModel(
        model_loader=lambda *_args: model,
        checkpoint_loader=lambda _path: {},
        torch_module=torch,
    )
    adapter.load("cpu", "fp32", _config(source, checkpoint, warmup_runs=2))
    adapter.warmup()
    assert len(model.calls) == 2
    for h, w in ((3, 8), (8, 3), (5, 5)):
        assert adapter.infer(_frame(h, w)).depth.shape == (h, w)


def test_missing_assets_invalid_config_and_malformed_output(tmp_path) -> None:
    checkpoint = _checkpoint(tmp_path)
    adapter = DepthAnythingV2GeometryModel(
        model_loader=lambda *_args: _Model(),
        checkpoint_loader=lambda _path: {},
        torch_module=torch,
    )
    with pytest.raises(DepthAnythingV2CheckpointError, match="install_depth_anything_models"):
        adapter.load("cpu", "fp32", _config(tmp_path / "missing", checkpoint))
    source = _source(tmp_path)
    with pytest.raises(DepthAnythingV2CheckpointError, match="download_depth_anything_models"):
        adapter.load("cpu", "fp32", _config(source, tmp_path / "missing.pth"))
    with pytest.raises(DepthAnythingV2ConfigurationError, match="precision"):
        adapter.load("cpu", "bf16", _config(source, checkpoint))
    with pytest.raises(DepthAnythingV2ConfigurationError, match="encoder"):
        adapter.load("cpu", "fp32", _config(source, checkpoint, encoder="bad"))
    with pytest.raises(DepthAnythingV2ConfigurationError, match="virtual_fov"):
        adapter.load("cpu", "fp32", _config(source, checkpoint, virtual_fov_degrees=180))

    malformed = DepthAnythingV2GeometryModel(
        model_loader=lambda *_args: _Model(malformed=True),
        checkpoint_loader=lambda _path: {},
        torch_module=torch,
    )
    malformed.load("cpu", "fp32", _config(source, checkpoint))
    with pytest.raises(RuntimeError, match="2D"):
        malformed.infer(_frame())
    with pytest.raises(ValueError, match="H×W×3"):
        malformed._validated_bgr(np.zeros((3, 4), dtype=np.uint8))

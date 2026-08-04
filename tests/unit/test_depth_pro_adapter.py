"""Depth Pro adapter contract tests without external weights or downloads."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

from core.models.adapters.depth_pro_adapter import (
    DepthProCheckpointError,
    DepthProConfigurationError,
    DepthProGeometryModel,
)
from core.models.catalog import DEPTH_PRO_MODEL_ID
from core.models.registry import get_model_descriptor
from core.models.validation import normalize_geometry_result
from core.runtime.messages import FrameRequest


@dataclass(frozen=True)
class _FakeConfig:
    checkpoint_uri: str | None = None


class _FakeModel:
    def __init__(self, *, focal_length: float | None = 700.0) -> None:
        self.focal_length = focal_length
        self.eval_called = False
        self.infer_calls = 0
        self.last_tensor = None

    def eval(self):
        self.eval_called = True
        return self

    def infer(self, tensor, *, f_px=None):
        assert f_px is None
        self.infer_calls += 1
        self.last_tensor = tensor
        height, width = tensor.shape[-2:]
        output = {
            "depth": torch.full((height, width), 2.5, dtype=torch.float32),
        }
        if self.focal_length is not None:
            output["focallength_px"] = torch.tensor(self.focal_length)
        return output


class _Factory:
    def __init__(self, model: _FakeModel) -> None:
        self.model = model
        self.calls = []
        self.last_rgb = None

    def __call__(self, *, config, device, precision):
        self.calls.append((config, device, precision))

        def transform(rgb: np.ndarray):
            self.last_rgb = rgb.copy()
            return torch.from_numpy(rgb.copy()).permute(2, 0, 1).to(torch.float32)

        return self.model, transform


def _config(checkpoint: Path, **overrides) -> dict[str, object]:
    values: dict[str, object] = {
        "checkpoint": str(checkpoint),
        "model_label": "Depth Pro Test",
        "internal_resolution": 1536,
        "point_budget": 123_456,
        "warmup_size": 64,
        "warmup_runs": 1,
        "matmul_precision": "high",
        "hardware_tier": "efficient",
    }
    values.update(overrides)
    return values


def _frame() -> FrameRequest:
    image = np.zeros((3, 4, 3), dtype=np.uint8)
    image[..., 0] = 10  # B
    image[..., 1] = 20  # G
    image[..., 2] = 30  # R
    return FrameRequest(session_id=4, frame_id=9, image=image)


def test_load_infer_metadata_and_unload_follow_common_contract(tmp_path) -> None:
    checkpoint = tmp_path / "depth_pro.pt"
    checkpoint.write_bytes(b"test")
    model = _FakeModel(focal_length=800.0)
    factory = _Factory(model)
    adapter = DepthProGeometryModel(
        factory=factory,
        default_config=_FakeConfig(),
        torch_module=torch,
    )

    # CPU execution safely upgrades requested FP16 to FP32.
    adapter.load("cpu", "fp16", _config(checkpoint))
    assert model.eval_called is True
    config, device, dtype = factory.calls[0]
    assert config.checkpoint_uri == str(checkpoint)
    assert str(device) == "cpu"
    assert dtype is torch.float32

    result = adapter.infer(_frame())
    assert result.model_id == DEPTH_PRO_MODEL_ID
    assert result.model_name == "Depth Pro Test"
    assert result.precision == "FP32"
    assert result.is_metric is True
    assert result.depth.shape == (3, 4)
    assert np.allclose(result.depth, 2.5)
    assert result.valid_mask.dtype == np.bool_
    assert result.valid_mask.all()
    assert result.intrinsics is not None
    assert result.intrinsics[0, 0] == pytest.approx(800.0)
    assert result.intrinsics[1, 1] == pytest.approx(800.0)
    assert result.intrinsics[0, 2] == pytest.approx(1.5)
    assert result.intrinsics[1, 2] == pytest.approx(1.0)
    assert result.intrinsics_source == "model_predicted"
    assert factory.last_rgb is not None
    assert factory.last_rgb[0, 0].tolist() == [30, 20, 10]

    metadata = adapter.get_metadata()
    assert metadata["id"] == DEPTH_PRO_MODEL_ID
    assert metadata["checkpoint"] == str(checkpoint)
    assert metadata["precision"] == "FP32"
    assert metadata["focal_length_source"] == "model_predicted"
    assert metadata["outputs"]["intrinsics"] is True

    cleanup = adapter.unload()
    assert cleanup["released"] is True
    with pytest.raises(RuntimeError, match="not loaded"):
        adapter.infer(_frame())


def test_missing_or_invalid_focal_length_uses_registry_estimated_fallback(tmp_path) -> None:
    checkpoint = tmp_path / "depth_pro.pt"
    checkpoint.write_bytes(b"test")
    model = _FakeModel(focal_length=None)
    adapter = DepthProGeometryModel(
        factory=_Factory(model),
        default_config=_FakeConfig(),
        torch_module=torch,
    )
    adapter.load("cpu", "fp32", _config(checkpoint))

    raw = adapter.infer(_frame())
    assert raw.intrinsics is None
    normalized = normalize_geometry_result(
        raw,
        get_model_descriptor(DEPTH_PRO_MODEL_ID),
        _frame(),
    )
    assert normalized.intrinsics is not None
    assert normalized.intrinsics_source == "estimated"
    assert normalized.intrinsics[0, 0] > 0


def test_warmup_runs_configured_number_of_inferences(tmp_path) -> None:
    checkpoint = tmp_path / "depth_pro.pt"
    checkpoint.write_bytes(b"test")
    model = _FakeModel()
    adapter = DepthProGeometryModel(
        factory=_Factory(model),
        default_config=_FakeConfig(),
        torch_module=torch,
    )
    adapter.load("cpu", "fp32", _config(checkpoint, warmup_runs=2))
    adapter.warmup()
    assert model.infer_calls == 2


def test_missing_checkpoint_and_invalid_precision_fail_early(tmp_path) -> None:
    model = _FakeModel()
    adapter = DepthProGeometryModel(
        factory=_Factory(model),
        default_config=_FakeConfig(),
        torch_module=torch,
    )
    with pytest.raises(DepthProCheckpointError, match="download_depth_pro_checkpoint"):
        adapter.load("cpu", "fp32", _config(tmp_path / "missing.pt"))

    checkpoint = tmp_path / "depth_pro.pt"
    checkpoint.write_bytes(b"test")
    with pytest.raises(DepthProConfigurationError, match="Unsupported Depth Pro precision"):
        adapter.load("cpu", "bf16", _config(checkpoint))


def test_depth_resolution_and_input_shape_are_validated(tmp_path) -> None:
    checkpoint = tmp_path / "depth_pro.pt"
    checkpoint.write_bytes(b"test")

    class WrongShapeModel(_FakeModel):
        def infer(self, tensor, *, f_px=None):
            return {
                "depth": torch.ones((2, 2)),
                "focallength_px": torch.tensor(500.0),
            }

    adapter = DepthProGeometryModel(
        factory=_Factory(WrongShapeModel()),
        default_config=_FakeConfig(),
        torch_module=torch,
    )
    adapter.load("cpu", "fp32", _config(checkpoint))
    with pytest.raises(RuntimeError, match="does not match the input frame"):
        adapter.infer(_frame())

    with pytest.raises(ValueError, match="H×W×3"):
        adapter._validated_rgb(np.zeros((3, 4), dtype=np.uint8))

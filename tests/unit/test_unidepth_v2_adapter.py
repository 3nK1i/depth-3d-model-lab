"""UniDepthV2 adapter contract tests without external weights or downloads."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from core.models.adapters.unidepth_v2_adapter import (
    UniDepthV2CheckpointError,
    UniDepthV2ConfigurationError,
    UniDepthV2GeometryModel,
    _safe_unidepth_paddings,
)
from core.models.catalog import UNIDEPTH_V2_MODEL_ID
from core.models.registry import get_model_descriptor
from core.models.validation import normalize_geometry_result
from core.runtime.messages import FrameRequest


class _FakeModel:
    def __init__(self, *, include_intrinsics: bool = True) -> None:
        self.include_intrinsics = include_intrinsics
        self.resolution_level = -1
        self.interpolation_mode = "bilinear"
        self.eval_called = False
        self.half_called = False
        self.to_device = None
        self.infer_calls = 0
        self.last_tensor = None

    def half(self):
        self.half_called = True
        return self

    def to(self, device):
        self.to_device = device
        return self

    def eval(self):
        self.eval_called = True
        return self

    def infer(self, tensor):
        self.infer_calls += 1
        self.last_tensor = tensor
        height, width = tensor.shape[-2:]
        yy, xx = torch.meshgrid(
            torch.arange(height, dtype=torch.float32),
            torch.arange(width, dtype=torch.float32),
            indexing="ij",
        )
        depth = torch.full((1, 1, height, width), 2.0, dtype=torch.float32)
        points = torch.stack((xx, yy, depth[0, 0]), dim=0).unsqueeze(0)
        output = {
            "depth": depth,
            "points": points,
            "confidence": torch.full((1, 1, height, width), 0.25),
        }
        if self.include_intrinsics:
            output["intrinsics"] = torch.tensor(
                [[[500.0, 0.0, 1.5], [0.0, 510.0, 1.0], [0.0, 0.0, 1.0]]]
            )
        return output


class _Loader:
    def __init__(self, model: _FakeModel) -> None:
        self.model = model
        self.calls: list[tuple[str, bool]] = []

    def __call__(self, path: str, *, local_files_only: bool):
        self.calls.append((path, local_files_only))
        return self.model


def _checkpoint(root: Path) -> Path:
    path = root / "vits14"
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}")
    (path / "model.safetensors").write_bytes(b"weights")
    return path


def _config(checkpoint: Path, **overrides) -> dict[str, object]:
    values: dict[str, object] = {
        "checkpoint_dir": str(checkpoint),
        "repository": "lpiccinelli/unidepth-v2-vits14",
        "variant": "vits14",
        "model_label": "UniDepthV2 Test",
        "resolution_level": 2,
        "interpolation_mode": "bilinear",
        "point_budget": 6,
        "warmup_size": 64,
        "warmup_runs": 1,
        "matmul_precision": "high",
        "hardware_tier": "efficient",
    }
    values.update(overrides)
    return values


def _frame() -> FrameRequest:
    image = np.zeros((3, 4, 3), dtype=np.uint8)
    image[..., 0] = 10
    image[..., 1] = 20
    image[..., 2] = 30
    return FrameRequest(session_id=5, frame_id=12, image=image, point_budget=6)


def test_load_infer_normalize_metadata_and_unload(tmp_path) -> None:
    checkpoint = _checkpoint(tmp_path)
    model = _FakeModel()
    loader = _Loader(model)
    adapter = UniDepthV2GeometryModel(model_loader=loader, torch_module=torch)

    adapter.load("cpu", "fp16", _config(checkpoint))
    assert loader.calls == [(str(checkpoint), True)]
    assert model.eval_called is True
    assert model.half_called is False
    assert str(model.to_device) == "cpu"
    assert model.resolution_level == 2

    raw = adapter.infer(_frame())
    assert raw.model_id == UNIDEPTH_V2_MODEL_ID
    assert raw.model_name == "UniDepthV2 Test"
    assert raw.precision == "FP32"
    assert raw.depth.shape == (3, 4)
    assert raw.points is not None
    assert raw.points.shape == (2, 2, 3)
    assert raw.point_stride == 2
    assert raw.confidence is not None
    assert np.allclose(raw.confidence, 0.25)
    assert raw.valid_mask is not None and raw.valid_mask.all()
    assert raw.intrinsics is not None
    assert model.last_tensor[0, 0, 0].item() == 30
    assert model.last_tensor[1, 0, 0].item() == 20
    assert model.last_tensor[2, 0, 0].item() == 10

    normalized = normalize_geometry_result(
        raw, get_model_descriptor(UNIDEPTH_V2_MODEL_ID), _frame()
    )
    assert normalized.is_metric is True
    assert normalized.depth_scale == "metric"
    assert normalized.intrinsics_source == "model_predicted"
    assert normalized.confidence_kind == "uncertainty"

    metadata = adapter.get_metadata()
    assert metadata["id"] == UNIDEPTH_V2_MODEL_ID
    assert metadata["variant"] == "vits14"
    assert metadata["resolution_level"] == 2
    assert metadata["confidence_kind"] == "uncertainty"

    cleanup = adapter.unload()
    assert cleanup["released"] is True
    with pytest.raises(RuntimeError, match="not loaded"):
        adapter.infer(_frame())


def test_missing_intrinsics_uses_registry_estimated_fallback(tmp_path) -> None:
    checkpoint = _checkpoint(tmp_path)
    adapter = UniDepthV2GeometryModel(
        model_loader=_Loader(_FakeModel(include_intrinsics=False)),
        torch_module=torch,
    )
    adapter.load("cpu", "fp32", _config(checkpoint))
    raw = adapter.infer(_frame())
    assert raw.intrinsics is None
    normalized = normalize_geometry_result(
        raw, get_model_descriptor(UNIDEPTH_V2_MODEL_ID), _frame()
    )
    assert normalized.intrinsics is not None
    assert normalized.intrinsics_source == "estimated"


def test_warmup_and_fp16_conversion_are_profile_driven(tmp_path, monkeypatch) -> None:
    checkpoint = _checkpoint(tmp_path)
    model = _FakeModel()
    adapter = UniDepthV2GeometryModel(
        model_loader=_Loader(model), torch_module=torch
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    adapter.load("cuda:0", "fp16", _config(checkpoint, warmup_runs=2))
    assert model.half_called is True
    assert str(model.to_device) == "cuda:0"
    # Avoid CUDA events in this dependency-light unit test.
    adapter._device = torch.device("cpu")
    adapter.warmup()
    assert model.infer_calls == 2


def test_missing_checkpoint_and_invalid_configuration_fail_early(tmp_path) -> None:
    adapter = UniDepthV2GeometryModel(
        model_loader=_Loader(_FakeModel()), torch_module=torch
    )
    with pytest.raises(UniDepthV2CheckpointError, match="download_unidepth_v2_models"):
        adapter.load("cpu", "fp32", _config(tmp_path / "missing"))

    checkpoint = _checkpoint(tmp_path)
    with pytest.raises(UniDepthV2ConfigurationError, match="precision"):
        adapter.load("cpu", "bf16", _config(checkpoint))
    with pytest.raises(UniDepthV2ConfigurationError, match="resolution_level"):
        adapter.load("cpu", "fp32", _config(checkpoint, resolution_level=10))
    with pytest.raises(UniDepthV2ConfigurationError, match="interpolation"):
        adapter.load("cpu", "fp32", _config(checkpoint, interpolation_mode="invalid"))


def test_output_shapes_and_input_shape_are_validated(tmp_path) -> None:
    checkpoint = _checkpoint(tmp_path)

    class WrongShape(_FakeModel):
        def infer(self, tensor):
            output = super().infer(tensor)
            output["depth"] = torch.ones((1, 1, 2, 2))
            return output

    adapter = UniDepthV2GeometryModel(
        model_loader=_Loader(WrongShape()), torch_module=torch
    )
    adapter.load("cpu", "fp32", _config(checkpoint))
    with pytest.raises(RuntimeError, match="does not match the input frame"):
        adapter.infer(_frame())
    with pytest.raises(ValueError, match="H×W×3"):
        adapter._validated_rgb(np.zeros((3, 4), dtype=np.uint8))


def test_padding_guard_never_returns_negative_values() -> None:
    padding, shape = _safe_unidepth_paddings((206, 500), (0.5, 2.5))
    assert padding == (0, 0, 0, 0)
    assert shape == (206, 500)

    padding, shape = _safe_unidepth_paddings((100, 400), (0.5, 2.5))
    assert all(value >= 0 for value in padding)
    assert shape[0] >= 100 and shape[1] >= 400

    padding, shape = _safe_unidepth_paddings((400, 100), (0.5, 2.5))
    assert all(value >= 0 for value in padding)
    assert shape[0] >= 400 and shape[1] >= 100

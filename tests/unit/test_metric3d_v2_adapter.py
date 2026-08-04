"""Metric3D v2 adapter tests without downloads or external checkpoints."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from core.models.adapters.metric3d_v2_adapter import (
    Metric3DV2CheckpointError,
    Metric3DV2ConfigurationError,
    Metric3DV2GeometryModel,
)
from core.models.catalog import METRIC3D_V2_MODEL_ID
from core.models.registry import get_model_descriptor
from core.models.validation import normalize_geometry_result
from core.runtime.messages import FrameRequest


class _FakeModel:
    def __init__(self, *, malformed: bool = False) -> None:
        self.malformed = malformed
        self.loaded_state = None
        self.strict = None
        self.half_called = False
        self.float_called = False
        self.to_device = None
        self.eval_called = False
        self.infer_calls = 0
        self.last_tensor = None

    def load_state_dict(self, state, strict=False):
        self.loaded_state = state
        self.strict = strict
        return None

    def half(self):
        self.half_called = True
        return self

    def float(self):
        self.float_called = True
        return self

    def to(self, device):
        self.to_device = device
        return self

    def eval(self):
        self.eval_called = True
        return self

    def inference(self, payload):
        self.infer_calls += 1
        tensor = payload["input"]
        self.last_tensor = tensor.detach().cpu()
        if self.malformed:
            return {"depth": torch.ones((1, 1, 2, 2))}
        height, width = tensor.shape[-2:]
        depth = torch.ones((1, 1, height, width), dtype=tensor.dtype, device=tensor.device)
        confidence = torch.full_like(depth, 0.8)
        normals = torch.zeros((1, 4, height, width), dtype=tensor.dtype, device=tensor.device)
        normals[:, 2] = 1.0
        normals[:, 3] = 0.7
        return depth, confidence, {"prediction_normal": normals}


class _Loader:
    def __init__(self, model: _FakeModel) -> None:
        self.model = model
        self.calls: list[tuple[str, str]] = []

    def __call__(self, source: str, entrypoint: str):
        self.calls.append((source, entrypoint))
        return self.model


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "Metric3D"
    source.mkdir()
    (source / "hubconf.py").write_text("# test hub\n", encoding="utf-8")
    return source


def _checkpoint(tmp_path: Path, name: str = "model.pth") -> Path:
    path = tmp_path / name
    path.write_bytes(b"weights")
    return path


def _config(source: Path, checkpoint: Path, **overrides) -> dict:
    config = {
        "source_dir": str(source),
        "checkpoint": str(checkpoint),
        "entrypoint": "metric3d_vit_small",
        "model_label": "Metric3D v2 Test",
        "input_height": 64,
        "input_width": 64,
        "canonical_focal_length": 100.0,
        "virtual_fov_degrees": 60.0,
        "max_depth_m": 300.0,
        "point_budget": 4,
        "warmup_size": 64,
        "warmup_runs": 1,
    }
    config.update(overrides)
    return config


def _frame(height: int = 3, width: int = 4) -> FrameRequest:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[..., 0] = 10
    image[..., 1] = 20
    image[..., 2] = 30
    return FrameRequest(4, 9, image, point_budget=4)


def test_load_infer_normalize_and_unload_contract(tmp_path) -> None:
    source = _source(tmp_path)
    checkpoint = _checkpoint(tmp_path)
    model = _FakeModel()
    loader = _Loader(model)
    adapter = Metric3DV2GeometryModel(
        model_loader=loader,
        checkpoint_loader=lambda _path: {"model_state_dict": {"weight": torch.tensor(1)}},
        torch_module=torch,
    )

    adapter.load("cpu", "fp16", _config(source, checkpoint))
    assert loader.calls == [(str(source.resolve()), "metric3d_vit_small")]
    assert model.loaded_state is not None
    assert model.strict is False
    assert model.half_called is False
    assert model.float_called is True
    assert str(model.to_device) == "cpu"
    assert model.eval_called is True

    result = adapter.infer(_frame())
    assert result.model_id == METRIC3D_V2_MODEL_ID
    assert result.model_name == "Metric3D v2 Test"
    assert result.precision == "FP32"
    assert result.depth is not None and result.depth.shape == (3, 4)
    assert np.isfinite(result.depth).all() and (result.depth > 0).all()
    assert result.points is not None and result.points.shape == (2, 2, 3)
    assert result.colors is not None and result.colors.shape == (2, 2, 3)
    assert result.colors[0, 0].tolist() == [30, 20, 10]
    assert result.normals is not None and result.normals.shape == (2, 2, 3)
    assert np.allclose(result.normals[..., 2], 1.0)
    assert result.confidence is not None and np.allclose(result.confidence, 0.8)
    assert result.valid_mask is not None and result.valid_mask.all()
    assert result.intrinsics is not None and result.intrinsics.shape == (3, 3)
    assert result.intrinsics_source == "estimated"
    assert result.confidence_kind == "confidence"
    assert result.is_metric is True
    assert model.last_tensor is not None
    assert model.last_tensor.dtype == torch.float32

    normalized = normalize_geometry_result(
        result, get_model_descriptor(METRIC3D_V2_MODEL_ID), _frame()
    )
    assert normalized.depth_scale == "metric"
    assert normalized.intrinsics_source == "estimated"
    assert normalized.confidence_kind == "confidence"

    metadata = adapter.get_metadata()
    assert metadata["id"] == METRIC3D_V2_MODEL_ID
    assert metadata["entrypoint"] == "metric3d_vit_small"
    assert metadata["intrinsics_source"] == "estimated"
    assert metadata["metric_scale_status"] == "estimated_camera_intrinsics"

    cleanup = adapter.unload()
    assert cleanup["released"] is True
    with pytest.raises(RuntimeError, match="not loaded"):
        adapter.infer(_frame())


def test_preprocessing_handles_landscape_portrait_and_square(tmp_path) -> None:
    source = _source(tmp_path)
    checkpoint = _checkpoint(tmp_path)
    adapter = Metric3DV2GeometryModel(
        model_loader=_Loader(_FakeModel()),
        checkpoint_loader=lambda _path: {},
        torch_module=torch,
    )
    adapter.load("cpu", "fp32", _config(source, checkpoint))
    for height, width in ((3, 8), (8, 3), (5, 5)):
        prepared = adapter._prepare_input(_frame(height, width).image)
        assert prepared["normalized_chw"].shape == (3, 64, 64)
        assert all(value >= 0 for value in prepared["pad"])
        result = adapter.infer(_frame(height, width))
        assert result.depth.shape == (height, width)


def test_normals_accept_official_chw_and_hwc_shapes(tmp_path) -> None:
    source = _source(tmp_path)
    checkpoint = _checkpoint(tmp_path)
    adapter = Metric3DV2GeometryModel(
        model_loader=_Loader(_FakeModel()),
        checkpoint_loader=lambda _path: {},
        torch_module=torch,
    )
    adapter.load("cpu", "fp32", _config(source, checkpoint))
    prepared = adapter._prepare_input(_frame(4, 6).image)
    chw = np.zeros((4, 64, 64), dtype=np.float32)
    chw[2] = 1.0
    hwc = np.zeros((64, 64, 3), dtype=np.float32)
    hwc[..., 2] = 1.0
    assert adapter._postprocess_normals(chw, prepared).shape == (4, 6, 3)
    assert adapter._postprocess_normals(hwc, prepared).shape == (4, 6, 3)


def test_warmup_uses_configured_runs(tmp_path) -> None:
    source = _source(tmp_path)
    checkpoint = _checkpoint(tmp_path)
    model = _FakeModel()
    adapter = Metric3DV2GeometryModel(
        model_loader=_Loader(model),
        checkpoint_loader=lambda _path: {},
        torch_module=torch,
    )
    adapter.load("cpu", "fp32", _config(source, checkpoint, warmup_runs=2))
    adapter.warmup()
    assert model.infer_calls == 2


def test_missing_assets_and_invalid_configuration_fail_early(tmp_path) -> None:
    checkpoint = _checkpoint(tmp_path)
    adapter = Metric3DV2GeometryModel(
        model_loader=_Loader(_FakeModel()),
        checkpoint_loader=lambda _path: {},
        torch_module=torch,
    )
    with pytest.raises(Metric3DV2CheckpointError, match="install_metric3d_v2"):
        adapter.load("cpu", "fp32", _config(tmp_path / "missing", checkpoint))

    source = _source(tmp_path)
    with pytest.raises(Metric3DV2CheckpointError, match="download_metric3d_v2_models"):
        adapter.load("cpu", "fp32", _config(source, tmp_path / "missing.pth"))
    with pytest.raises(Metric3DV2ConfigurationError, match="precision"):
        adapter.load("cpu", "bf16", _config(source, checkpoint))
    with pytest.raises(Metric3DV2ConfigurationError, match="entrypoint"):
        adapter.load("cpu", "fp32", _config(source, checkpoint, entrypoint="bad"))
    with pytest.raises(Metric3DV2ConfigurationError, match="canonical_focal_length"):
        adapter.load("cpu", "fp32", _config(source, checkpoint, canonical_focal_length=0))
    with pytest.raises(Metric3DV2ConfigurationError, match="virtual_fov_degrees"):
        adapter.load("cpu", "fp32", _config(source, checkpoint, virtual_fov_degrees=180))


def test_invalid_image_and_model_output_are_rejected(tmp_path) -> None:
    source = _source(tmp_path)
    checkpoint = _checkpoint(tmp_path)
    adapter = Metric3DV2GeometryModel(
        model_loader=_Loader(_FakeModel(malformed=True)),
        checkpoint_loader=lambda _path: {},
        torch_module=torch,
    )
    adapter.load("cpu", "fp32", _config(source, checkpoint))
    with pytest.raises(RuntimeError, match="unexpected output"):
        adapter.infer(_frame())
    with pytest.raises(ValueError, match="H×W×3"):
        adapter._validated_rgb(np.zeros((3, 4), dtype=np.uint8))


def test_checkpoint_state_dictionary_validation(tmp_path) -> None:
    source = _source(tmp_path)
    checkpoint = _checkpoint(tmp_path)
    adapter = Metric3DV2GeometryModel(
        model_loader=_Loader(_FakeModel()),
        checkpoint_loader=lambda _path: [1, 2, 3],
        torch_module=torch,
    )
    with pytest.raises(RuntimeError, match="valid model state dictionary"):
        adapter.load("cpu", "fp32", _config(source, checkpoint))


class _ContextProbe:
    def __init__(self, events: list[str], name: str) -> None:
        self._events = events
        self._name = name

    def __enter__(self):
        self._events.append(f"enter:{self._name}")
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._events.append(f"exit:{self._name}")
        return False


class _AutocastProbeTorch:
    float16 = "float16"

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.events: list[str] = []

    def autocast(self, **kwargs):
        self.calls.append(kwargs)
        return _ContextProbe(self.events, "autocast")


def test_fp16_cuda_strategy_uses_autocast_without_half_model_conversion() -> None:
    probe = _AutocastProbeTorch()
    adapter = Metric3DV2GeometryModel(torch_module=probe)
    adapter._device = SimpleNamespace(type="cuda")
    adapter._precision = "fp16"

    assert adapter._supports_cuda_autocast(probe) is True
    with adapter._cuda_autocast_context():
        probe.events.append("inference")

    assert probe.calls == [{"device_type": "cuda", "dtype": "float16"}]
    assert probe.events == ["enter:autocast", "inference", "exit:autocast"]


class _CudaProbe:
    def __init__(self) -> None:
        self.amp = SimpleNamespace(autocast=None)

    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def get_device_capability(_device):
        return (7, 5)

    @staticmethod
    def is_tf32_supported() -> bool:
        return False


class _CudaLoadProbeTorch:
    float16 = "float16"

    def __init__(self) -> None:
        self.cuda = _CudaProbe()
        self.backends = SimpleNamespace(
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
            cudnn=SimpleNamespace(allow_tf32=False),
        )
        self.matmul_precision = None

    @staticmethod
    def device(value: str):
        return SimpleNamespace(type="cuda", value=value)

    def set_float32_matmul_precision(self, value: str) -> None:
        self.matmul_precision = value

    def autocast(self, **_kwargs):
        return _ContextProbe([], "autocast")


def test_cuda_fp16_load_keeps_metric3d_weights_in_fp32(tmp_path) -> None:
    source = _source(tmp_path)
    checkpoint = _checkpoint(tmp_path)
    model = _FakeModel()
    torch_probe = _CudaLoadProbeTorch()
    adapter = Metric3DV2GeometryModel(
        model_loader=_Loader(model),
        checkpoint_loader=lambda _path: {"model_state_dict": {}},
        torch_module=torch_probe,
    )

    adapter.load("cuda:0", "fp16", _config(source, checkpoint))

    assert adapter.precision_label == "FP16"
    assert model.half_called is False
    assert model.float_called is True
    assert model.to_device.type == "cuda"

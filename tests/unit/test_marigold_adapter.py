"""Marigold adapter contract tests without external downloads."""

from __future__ import annotations

from pathlib import Path
import warnings
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from core.models.adapters.marigold_adapter import (
    MarigoldCheckpointError,
    MarigoldConfigurationError,
    MarigoldGeometryModel,
    MarigoldOutOfMemoryError,
)
from core.models.catalog import MARIGOLD_MODEL_ID
from core.models.registry import get_model_descriptor
from core.models.validation import normalize_geometry_result
from core.runtime.messages import FrameRequest


class _VAE:
    def __init__(self) -> None:
        self.slicing = False
        self.tiling = False

    def enable_slicing(self):
        self.slicing = True

    def enable_tiling(self):
        self.tiling = True


class _Pipeline:
    prediction_type = "depth"

    def __init__(self, *, malformed: bool = False, oom: bool = False) -> None:
        self.vae = _VAE()
        self.malformed = malformed
        self.oom = oom
        self.to_device = None
        self.offload_gpu_id = None
        self.attention_slice = None
        self.calls = []
        self.hooks_removed = False

    def to(self, device):
        self.to_device = device
        return self

    def enable_model_cpu_offload(self, gpu_id=0):
        self.offload_gpu_id = gpu_id

    def enable_attention_slicing(self, value):
        self.attention_slice = value

    def remove_all_hooks(self):
        self.hooks_removed = True

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.oom:
            raise RuntimeError("CUDA out of memory")
        image = kwargs["image"]
        width, height = image.size
        if self.malformed:
            prediction = np.ones((1, 2, 2, 2), dtype=np.float32)
        else:
            prediction = np.linspace(
                0.0, 1.0, height * width, dtype=np.float32
            ).reshape(1, height, width, 1)
        uncertainty = None
        if kwargs["output_uncertainty"]:
            uncertainty = np.full((1, height, width, 1), 0.25, dtype=np.float32)
        return SimpleNamespace(prediction=prediction, uncertainty=uncertainty)


def _snapshot(tmp_path: Path) -> Path:
    path = tmp_path / "marigold"
    path.mkdir()
    (path / ".ready").write_text("{}\n", encoding="utf-8")
    (path / "model_index.json").write_text("{}\n", encoding="utf-8")
    return path


def _config(snapshot: Path, **overrides) -> dict:
    values = {
        "repository": "prs-eth/marigold-depth-v1-1",
        "checkpoint_dir": str(snapshot),
        "model_label": "Marigold Test",
        "weight_variant": "fp16",
        "prediction_type": "depth",
        "processing_resolution": 512,
        "num_inference_steps": 4,
        "ensemble_size": 3,
        "output_uncertainty": True,
        "model_cpu_offload": False,
        "attention_slicing": True,
        "vae_slicing": True,
        "vae_tiling": True,
        "seed": 7,
        "virtual_fov_degrees": 60.0,
        "point_budget": 4,
        "warmup_size": 64,
        "warmup_runs": 1,
        "warmup_inference_steps": 1,
        "inference_timeout_ms": 120_000,
        "stop_timeout_ms": 120_000,
    }
    values.update(overrides)
    return values


def _frame(height: int = 3, width: int = 4) -> FrameRequest:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[..., 0] = 10
    image[..., 1] = 20
    image[..., 2] = 30
    return FrameRequest(4, 9, image, point_budget=4)


def test_load_infer_normalize_and_unload_contract(tmp_path) -> None:
    snapshot = _snapshot(tmp_path)
    pipeline = _Pipeline()
    loader_calls = []

    def loader(path, **kwargs):
        loader_calls.append((path, kwargs))
        return pipeline

    adapter = MarigoldGeometryModel(pipeline_loader=loader, torch_module=torch)
    adapter.load("cpu", "fp16", _config(snapshot))
    assert loader_calls[-1][0] == str(snapshot)
    assert loader_calls[-1][1]["variant"] is None  # CPU is forced to FP32.
    assert str(pipeline.to_device) == "cpu"
    assert pipeline.attention_slice == "max"
    assert pipeline.vae.slicing is True
    assert pipeline.vae.tiling is True

    result = adapter.infer(_frame())
    assert result.model_id == MARIGOLD_MODEL_ID
    assert result.model_name == "Marigold Test"
    assert result.precision == "FP32"
    assert result.is_metric is False
    assert result.depth_scale == "relative"
    assert result.intrinsics_source == "estimated"
    assert result.confidence_kind == "uncertainty"
    assert result.depth.shape == (3, 4)
    assert result.points.shape == (2, 2, 3)
    assert result.colors.shape == (2, 2, 3)
    assert result.colors[0, 0].tolist() == [30, 20, 10]
    assert result.confidence.shape == (3, 4)
    assert np.allclose(result.confidence, 0.25)
    assert result.valid_mask.all()
    assert result.depth[0, 0] < result.depth[-1, -1]

    call = pipeline.calls[-1]
    assert call["num_inference_steps"] == 4
    assert call["ensemble_size"] == 3
    assert call["processing_resolution"] == 512
    assert call["output_uncertainty"] is True
    assert np.asarray(call["image"])[0, 0].tolist() == [30, 20, 10]

    normalized = normalize_geometry_result(
        result, get_model_descriptor(MARIGOLD_MODEL_ID), _frame()
    )
    assert normalized.depth_scale == "relative"
    assert normalized.is_metric is False
    assert normalized.confidence_kind == "uncertainty"

    metadata = adapter.get_metadata()
    assert metadata["metric_scale_status"] == "affine_invariant_relative"
    assert metadata["inference_timeout_ms"] == 120_000
    assert metadata["stop_timeout_ms"] == 120_000
    cleanup = adapter.unload()
    assert cleanup["released"] is True
    assert pipeline.hooks_removed is True


def test_warmup_uses_fast_single_prediction_and_discards_uncertainty(tmp_path) -> None:
    pipeline = _Pipeline()
    adapter = MarigoldGeometryModel(
        pipeline_loader=lambda *_args, **_kwargs: pipeline,
        torch_module=torch,
    )
    adapter.load("cpu", "fp32", _config(_snapshot(tmp_path)))
    adapter.warmup()
    assert len(pipeline.calls) == 1
    assert pipeline.calls[0]["num_inference_steps"] == 1
    assert pipeline.calls[0]["ensemble_size"] == 1
    assert pipeline.calls[0]["output_uncertainty"] is False


def test_disparity_output_is_converted_to_relative_distance(tmp_path) -> None:
    pipeline = _Pipeline()
    pipeline.prediction_type = "disparity"
    adapter = MarigoldGeometryModel(
        pipeline_loader=lambda *_args, **_kwargs: pipeline,
        torch_module=torch,
    )
    adapter.load("cpu", "fp32", _config(_snapshot(tmp_path), prediction_type="disparity"))
    result = adapter.infer(_frame())
    assert result.depth[0, 0] > result.depth[-1, -1]


def test_missing_snapshot_invalid_config_and_malformed_output(tmp_path) -> None:
    adapter = MarigoldGeometryModel(
        pipeline_loader=lambda *_args, **_kwargs: _Pipeline(),
        torch_module=torch,
    )
    with pytest.raises(MarigoldCheckpointError, match="download_marigold_model"):
        adapter.load("cpu", "fp32", _config(tmp_path / "missing"))
    snapshot = _snapshot(tmp_path)
    with pytest.raises(MarigoldConfigurationError, match="precision"):
        adapter.load("cpu", "bf16", _config(snapshot))
    with pytest.raises(MarigoldConfigurationError, match="ensemble_size"):
        adapter.load("cpu", "fp32", _config(snapshot, ensemble_size=2))
    with pytest.raises(MarigoldConfigurationError, match="prediction type"):
        adapter.load("cpu", "fp32", _config(snapshot, prediction_type="bad"))
    with pytest.raises(MarigoldConfigurationError, match="virtual_fov"):
        adapter.load("cpu", "fp32", _config(snapshot, virtual_fov_degrees=180))

    malformed = MarigoldGeometryModel(
        pipeline_loader=lambda *_args, **_kwargs: _Pipeline(malformed=True),
        torch_module=torch,
    )
    malformed.load("cpu", "fp32", _config(snapshot))
    with pytest.raises(RuntimeError, match="2D"):
        malformed.infer(_frame())


def test_cuda_oom_is_classified_and_cleanup_remains_safe(tmp_path) -> None:
    pipeline = _Pipeline(oom=True)
    adapter = MarigoldGeometryModel(
        pipeline_loader=lambda *_args, **_kwargs: pipeline,
        torch_module=torch,
    )
    adapter.load("cpu", "fp32", _config(_snapshot(tmp_path)))
    with pytest.raises(MarigoldOutOfMemoryError, match="exhausted GPU memory"):
        adapter.infer(_frame())
    assert adapter.unload()["released"] is True


def test_pillow_rgb_conversion_uses_non_deprecated_api(tmp_path) -> None:
    pipeline = _Pipeline()
    adapter = MarigoldGeometryModel(
        pipeline_loader=lambda *_args, **_kwargs: pipeline,
        torch_module=torch,
    )
    adapter.load("cpu", "fp32", _config(_snapshot(tmp_path)))

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        result = adapter.infer(_frame())

    assert result.depth.shape == (3, 4)
    assert np.asarray(pipeline.calls[-1]["image"])[0, 0].tolist() == [30, 20, 10]

"""Stateful metric Video Depth Anything adapter tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from core.models.adapters.video_depth_anything_adapter import (
    VideoDepthAnythingCheckpointError,
    VideoDepthAnythingConfigurationError,
    VideoDepthAnythingGeometryModel,
)
from core.models.catalog import VIDEO_DEPTH_ANYTHING_MODEL_ID
from core.models.registry import get_model_descriptor
from core.models.validation import normalize_geometry_result
from core.runtime.messages import FrameRequest


class _StreamingModel:
    def __init__(self, malformed: bool = False) -> None:
        self.malformed = malformed
        self.loaded_state = None
        self.strict = None
        self.float_called = False
        self.to_device = None
        self.eval_called = False
        self.calls: list[tuple[np.ndarray, int, str, bool]] = []
        self.transform = None
        self.frame_id_list = []
        self.frame_cache_list = []
        self.id = -1

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

    def infer_video_depth_one(self, frame, input_size=518, device="cpu", fp32=False):
        self.calls.append((frame.copy(), input_size, device, fp32))
        self.id += 1
        self.transform = object()
        self.frame_height, self.frame_width = frame.shape[:2]
        self.frame_id_list.append(self.id)
        self.frame_cache_list.append([self.id])
        if self.malformed:
            return np.ones((2, 2, 2), dtype=np.float32)
        h, w = frame.shape[:2]
        return np.linspace(0.5, 5.0, h * w, dtype=np.float32).reshape(h, w)


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "Video-Depth-Anything"
    (source / "video_depth_anything").mkdir(parents=True)
    (source / "video_depth_anything" / "video_depth_stream.py").write_text("# test\n")
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
        "model_label": "Metric Video Depth Anything Test",
        "input_size": 392,
        "max_resolution": 4,
        "virtual_fov_degrees": 60.0,
        "max_depth_m": 300.0,
        "point_budget": 4,
        "warmup_size": 64,
    }
    values.update(overrides)
    return values


def _frame(session: int = 1, frame_id: int = 1, height: int = 4, width: int = 8) -> FrameRequest:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[..., 0] = 10
    image[..., 1] = 20
    image[..., 2] = 30
    return FrameRequest(session, frame_id, image, point_budget=4)


def test_load_stream_infer_normalize_reset_and_unload(tmp_path) -> None:
    source, checkpoint = _source(tmp_path), _checkpoint(tmp_path)
    model = _StreamingModel()
    adapter = VideoDepthAnythingGeometryModel(
        model_loader=lambda *_args: model,
        checkpoint_loader=lambda _path: {"weight": torch.tensor(1)},
        torch_module=torch,
    )
    adapter.load("cpu", "fp16", _config(source, checkpoint))
    assert model.strict is True
    assert model.float_called and model.eval_called
    result = adapter.infer(_frame())
    assert result.model_id == VIDEO_DEPTH_ANYTHING_MODEL_ID
    assert result.model_name == "Metric Video Depth Anything Test"
    assert result.precision == "FP32"
    assert result.is_metric is True
    assert result.depth_scale == "metric"
    assert result.intrinsics_source == "estimated"
    assert result.depth.shape == (4, 8)
    assert result.points.shape[-1] == 3
    assert result.points.shape[0] * result.points.shape[1] <= 4
    assert result.colors.shape == result.points.shape
    assert result.colors[0, 0].tolist() == [30, 20, 10]
    # Adapter downscales the longest side to max_resolution and supplies RGB.
    call = model.calls[-1]
    assert call[0].shape == (2, 4, 3)
    assert call[0][0, 0].tolist() == [30, 20, 10]
    assert call[1:] == (392, "cpu", True)

    normalized = normalize_geometry_result(
        result, get_model_descriptor(VIDEO_DEPTH_ANYTHING_MODEL_ID), _frame()
    )
    assert normalized.is_metric is True
    assert normalized.depth_scale == "metric"
    metadata = adapter.get_metadata()
    assert metadata["stateful_sequence"] is True
    assert metadata["upstream_streaming_status"] == "experimental"

    reset = adapter.reset_sequence()
    assert reset["sequence_reset"] is True
    assert model.id == -1
    assert model.frame_cache_list == []
    assert model.frame_id_list == []
    assert model.transform is None
    assert adapter.unload()["released"] is True


def test_sequence_resets_on_session_frame_restart_and_resolution_change(tmp_path) -> None:
    source, checkpoint = _source(tmp_path), _checkpoint(tmp_path)
    model = _StreamingModel()
    adapter = VideoDepthAnythingGeometryModel(
        model_loader=lambda *_args: model,
        checkpoint_loader=lambda _path: {},
        torch_module=torch,
    )
    adapter.load("cpu", "fp32", _config(source, checkpoint, max_resolution=0))
    adapter.infer(_frame(session=1, frame_id=1))
    resets = adapter._sequence_resets
    adapter.infer(_frame(session=1, frame_id=2))
    assert adapter._sequence_resets == resets
    adapter.infer(_frame(session=2, frame_id=1))
    assert adapter._sequence_resets == resets + 1
    adapter.infer(_frame(session=2, frame_id=1))
    assert adapter._sequence_resets == resets + 2
    adapter.infer(_frame(session=2, frame_id=2, height=6, width=8))
    assert adapter._sequence_resets == resets + 3


def test_warmup_does_not_pollute_user_sequence(tmp_path) -> None:
    source, checkpoint = _source(tmp_path), _checkpoint(tmp_path)
    model = _StreamingModel()
    adapter = VideoDepthAnythingGeometryModel(
        model_loader=lambda *_args: model,
        checkpoint_loader=lambda _path: {},
        torch_module=torch,
    )
    adapter.load("cpu", "fp32", _config(source, checkpoint))
    adapter.warmup()
    assert model.id == -1
    assert model.transform is None
    result = adapter.infer(_frame())
    assert result.frame_id == 1
    assert model.id == 0


def test_missing_assets_invalid_config_and_malformed_output(tmp_path) -> None:
    checkpoint = _checkpoint(tmp_path)
    adapter = VideoDepthAnythingGeometryModel(
        model_loader=lambda *_args: _StreamingModel(),
        checkpoint_loader=lambda _path: {},
        torch_module=torch,
    )
    with pytest.raises(VideoDepthAnythingCheckpointError, match="install_depth_anything_models"):
        adapter.load("cpu", "fp32", _config(tmp_path / "missing", checkpoint))
    source = _source(tmp_path)
    with pytest.raises(VideoDepthAnythingCheckpointError, match="download_depth_anything_models"):
        adapter.load("cpu", "fp32", _config(source, tmp_path / "missing.pth"))
    with pytest.raises(VideoDepthAnythingConfigurationError, match="precision"):
        adapter.load("cpu", "bf16", _config(source, checkpoint))
    with pytest.raises(VideoDepthAnythingConfigurationError, match="encoder"):
        adapter.load("cpu", "fp32", _config(source, checkpoint, encoder="bad"))
    with pytest.raises(VideoDepthAnythingConfigurationError, match="max_depth_m"):
        adapter.load("cpu", "fp32", _config(source, checkpoint, max_depth_m=0))

    malformed = VideoDepthAnythingGeometryModel(
        model_loader=lambda *_args: _StreamingModel(malformed=True),
        checkpoint_loader=lambda _path: {},
        torch_module=torch,
    )
    malformed.load("cpu", "fp32", _config(source, checkpoint))
    with pytest.raises(RuntimeError, match="2D"):
        malformed.infer(_frame())
    with pytest.raises(ValueError, match="H×W×3"):
        malformed._validated_bgr(np.zeros((3, 4), dtype=np.uint8))

"""Dependency-light lifecycle stress for DA V2 and streaming metric VDA."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from core.models.adapters.depth_anything_v2_adapter import DepthAnythingV2GeometryModel
from core.models.adapters.video_depth_anything_adapter import VideoDepthAnythingGeometryModel
from core.runtime.messages import FrameRequest


class _ImageModel:
    def load_state_dict(self, _state, strict=True): return None
    def float(self): return self
    def to(self, _device): return self
    def eval(self): return self
    def infer_image(self, image, _input_size):
        h, w = image.shape[:2]
        return np.linspace(0.1, 2.0, h * w, dtype=np.float32).reshape(h, w)


class _VideoModel:
    def __init__(self):
        self.transform = None
        self.frame_id_list = []
        self.frame_cache_list = []
        self.id = -1
    def load_state_dict(self, _state, strict=True): return None
    def float(self): return self
    def to(self, _device): return self
    def eval(self): return self
    def infer_video_depth_one(self, frame, **_kwargs):
        self.id += 1
        self.transform = object()
        self.frame_id_list.append(self.id)
        self.frame_cache_list.append([self.id])
        return np.ones(frame.shape[:2], dtype=np.float32)


def test_repeated_depth_anything_lifecycles_are_stable(tmp_path: Path) -> None:
    image_source = tmp_path / "image"
    (image_source / "depth_anything_v2").mkdir(parents=True)
    (image_source / "depth_anything_v2/dpt.py").write_text("# test\n")
    video_source = tmp_path / "video"
    (video_source / "video_depth_anything").mkdir(parents=True)
    (video_source / "video_depth_anything/video_depth_stream.py").write_text("# test\n")
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"weights")
    frame = FrameRequest(1, 1, np.zeros((8, 8, 3), dtype=np.uint8), point_budget=16)

    for _ in range(20):
        image = DepthAnythingV2GeometryModel(
            model_loader=lambda *_args: _ImageModel(),
            checkpoint_loader=lambda _path: {},
            torch_module=torch,
        )
        image.load("cpu", "fp32", {
            "source_dir": str(image_source), "checkpoint": str(checkpoint),
            "encoder": "vits", "point_budget": 16, "warmup_size": 64,
        })
        assert image.infer(frame).depth_scale == "relative"
        assert image.unload()["released"] is True

        video_model = _VideoModel()
        video = VideoDepthAnythingGeometryModel(
            model_loader=lambda *_args, model=video_model: model,
            checkpoint_loader=lambda _path: {},
            torch_module=torch,
        )
        video.load("cpu", "fp32", {
            "source_dir": str(video_source), "checkpoint": str(checkpoint),
            "encoder": "vits", "point_budget": 16, "warmup_size": 64,
            "max_resolution": 0,
        })
        assert video.infer(frame).depth_scale == "metric"
        assert video.reset_sequence()["sequence_reset"] is True
        assert video_model.id == -1
        assert video.unload()["released"] is True

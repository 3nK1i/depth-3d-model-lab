"""Repeated Metric3D v2 lifecycle stress without external weights."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from core.models.adapters.metric3d_v2_adapter import Metric3DV2GeometryModel
from core.runtime.messages import FrameRequest


class _Model:
    def load_state_dict(self, _state, strict=False): return None
    def half(self): return self
    def to(self, _device): return self
    def eval(self): return self
    def inference(self, payload):
        tensor = payload["input"]
        h, w = tensor.shape[-2:]
        depth = torch.ones((1, 1, h, w), dtype=tensor.dtype)
        confidence = torch.ones_like(depth)
        normals = torch.zeros((1, 4, h, w), dtype=tensor.dtype)
        normals[:, 2] = 1.0
        return depth, confidence, {"prediction_normal": normals}


def test_repeated_load_infer_unload_is_stable(tmp_path: Path) -> None:
    source = tmp_path / "Metric3D"
    source.mkdir()
    (source / "hubconf.py").write_text("# test\n")
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"weights")
    frame = FrameRequest(
        1, 1, np.zeros((8, 8, 3), dtype=np.uint8), point_budget=16
    )
    config = {
        "source_dir": str(source),
        "checkpoint": str(checkpoint),
        "entrypoint": "metric3d_vit_small",
        "input_height": 64,
        "input_width": 64,
        "canonical_focal_length": 100.0,
        "point_budget": 16,
        "warmup_size": 64,
        "warmup_runs": 1,
    }
    for _ in range(25):
        adapter = Metric3DV2GeometryModel(
            model_loader=lambda *_args: _Model(),
            checkpoint_loader=lambda _path: {},
            torch_module=torch,
        )
        adapter.load("cpu", "fp32", config)
        result = adapter.infer(frame)
        assert result.depth.shape == (8, 8)
        assert result.normals is not None
        assert result.intrinsics_source == "estimated"
        assert adapter.unload()["released"] is True

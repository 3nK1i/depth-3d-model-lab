"""Repeated UniDepthV2 adapter lifecycle stress without external weights."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from core.models.adapters.unidepth_v2_adapter import UniDepthV2GeometryModel
from core.runtime.messages import FrameRequest


class _Model:
    def half(self): return self
    def to(self, _device): return self
    def eval(self): return self
    def infer(self, tensor):
        h, w = tensor.shape[-2:]
        depth = torch.ones((1, 1, h, w))
        points = torch.zeros((1, 3, h, w))
        points[:, 2] = depth[:, 0]
        return {
            "depth": depth,
            "points": points,
            "confidence": torch.ones((1, 1, h, w)),
            "intrinsics": torch.tensor([[[100.0, 0.0, 1.0], [0.0, 100.0, 1.0], [0.0, 0.0, 1.0]]]),
        }


def test_repeated_load_infer_unload_is_stable(tmp_path: Path) -> None:
    checkpoint = tmp_path / "vits14"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    frame = FrameRequest(1, 1, np.zeros((4, 4, 3), dtype=np.uint8), point_budget=16)
    config = {
        "checkpoint_dir": str(checkpoint),
        "resolution_level": 2,
        "point_budget": 16,
        "warmup_size": 64,
        "warmup_runs": 1,
    }
    for _ in range(25):
        adapter = UniDepthV2GeometryModel(
            model_loader=lambda *_args, **_kwargs: _Model(),
            torch_module=torch,
        )
        adapter.load("cpu", "fp32", config)
        result = adapter.infer(frame)
        assert result.depth.shape == (4, 4)
        assert result.confidence_kind == "uncertainty"
        assert adapter.unload()["released"] is True

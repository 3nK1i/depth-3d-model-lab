"""Dependency-free repeated Depth Pro adapter lifecycle stress."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from core.models.adapters.depth_pro_adapter import DepthProGeometryModel
from core.runtime.messages import FrameRequest


@dataclass(frozen=True)
class _Config:
    checkpoint_uri: str | None = None


class _Model:
    def eval(self):
        return self

    def infer(self, tensor, *, f_px=None):
        height, width = tensor.shape[-2:]
        return {
            "depth": torch.ones((height, width)),
            "focallength_px": torch.tensor(500.0),
        }


def _factory(*, config, device, precision):
    def transform(rgb):
        return torch.from_numpy(rgb.copy()).permute(2, 0, 1).float()

    return _Model(), transform


def test_depth_pro_repeated_load_infer_unload_is_stable(tmp_path) -> None:
    checkpoint = tmp_path / "depth_pro.pt"
    checkpoint.write_bytes(b"test")
    frame = FrameRequest(
        session_id=1,
        frame_id=1,
        image=np.zeros((16, 20, 3), dtype=np.uint8),
    )
    for _ in range(25):
        adapter = DepthProGeometryModel(
            factory=_factory,
            default_config=_Config(),
            torch_module=torch,
        )
        adapter.load(
            "cpu",
            "fp32",
            {
                "checkpoint": str(checkpoint),
                "warmup_size": 64,
                "warmup_runs": 1,
            },
        )
        adapter.warmup()
        result = adapter.infer(frame)
        assert result.depth is not None
        assert result.depth.shape == (16, 20)
        assert adapter.unload()["released"] is True

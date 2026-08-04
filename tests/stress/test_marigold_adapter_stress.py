"""Dependency-light repeated Marigold adapter lifecycle stress."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from core.models.adapters.marigold_adapter import MarigoldGeometryModel
from core.runtime.messages import FrameRequest


class _VAE:
    def enable_slicing(self):
        return None

    def enable_tiling(self):
        return None


class _Pipeline:
    prediction_type = "depth"

    def __init__(self) -> None:
        self.vae = _VAE()

    def to(self, _device):
        return self

    def enable_attention_slicing(self, _value):
        return None

    def remove_all_hooks(self):
        return None

    def __call__(self, **kwargs):
        width, height = kwargs["image"].size
        prediction = np.full((1, height, width, 1), 0.5, dtype=np.float32)
        return SimpleNamespace(prediction=prediction, uncertainty=None)


def test_repeated_load_infer_unload_cycles(tmp_path: Path) -> None:
    snapshot = tmp_path / "marigold"
    snapshot.mkdir()
    (snapshot / ".ready").write_text("{}\n")
    (snapshot / "model_index.json").write_text("{}\n")
    frame = FrameRequest(1, 1, np.zeros((16, 20, 3), dtype=np.uint8), point_budget=100)
    for _ in range(20):
        adapter = MarigoldGeometryModel(
            pipeline_loader=lambda *_args, **_kwargs: _Pipeline(),
            torch_module=torch,
        )
        adapter.load(
            "cpu",
            "fp32",
            {
                "checkpoint_dir": str(snapshot),
                "processing_resolution": 64,
                "num_inference_steps": 1,
                "ensemble_size": 1,
                "point_budget": 100,
                "warmup_size": 64,
            },
        )
        result = adapter.infer(frame)
        assert result.depth is not None
        assert result.points is not None
        assert adapter.unload()["released"] is True

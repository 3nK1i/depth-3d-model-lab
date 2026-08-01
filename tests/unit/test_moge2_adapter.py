"""MoGe-2 adapter contract tests with an injected local fake model."""

from __future__ import annotations

import numpy as np
import torch

from core.models.adapters.moge2_adapter import Moge2GeometryModel
from core.models.catalog import MOGE2_MODEL_ID
from core.runtime.messages import FrameRequest


class _FakeMoGeModel:
    last_instance: "_FakeMoGeModel | None" = None

    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.eval_called = False
        self.sdpa_called = False
        self.last_image: torch.Tensor | None = None
        self.last_kwargs: dict[str, object] = {}

    @classmethod
    def from_pretrained(cls, _repo: str, **_kwargs):
        instance = cls()
        cls.last_instance = instance
        return instance

    def to(self, device=None, dtype=None):
        if device is not None:
            self.device = torch.device(device)
        if dtype is not None:
            self.dtype = dtype
        return self

    def eval(self):
        self.eval_called = True
        return self

    def enable_pytorch_native_sdpa(self) -> None:
        self.sdpa_called = True

    def infer(self, image: torch.Tensor, **kwargs):
        self.last_image = image.detach().cpu().clone()
        self.last_kwargs = dict(kwargs)
        _, height, width = image.shape

        y, x = torch.meshgrid(
            torch.arange(height, dtype=torch.float32),
            torch.arange(width, dtype=torch.float32),
            indexing="ij",
        )
        depth = 1.0 + y / max(height, 1) + x / max(width, 1)
        points = torch.stack((x * 0.01, y * 0.01, depth), dim=-1)
        normal = torch.zeros((height, width, 3), dtype=torch.float32)
        normal[..., 2] = 1.0
        mask = torch.ones((height, width), dtype=torch.bool)
        mask[0, 0] = False
        intrinsics = torch.eye(3, dtype=torch.float32)
        return {
            "depth": depth,
            "points": points,
            "normal": normal,
            "mask": mask,
            "intrinsics": intrinsics,
        }


def _adapter() -> Moge2GeometryModel:
    adapter = Moge2GeometryModel(
        model_class=_FakeMoGeModel,
        torch_module=torch,
    )
    adapter.load(
        device="cpu",
        precision="fp16",  # CPU must safely resolve to FP32.
        config={
            "pretrained": "fake/moge-2-vits-normal",
            "model_label": "MoGe-2 ViT-S Normal",
            "num_tokens": 1200,
            "resolution_level": 5,
            "point_budget": 1_000,
            "warmup_size": 64,
        },
    )
    return adapter


def test_adapter_returns_metric_direct_geometry_with_bounded_points() -> None:
    adapter = _adapter()
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    image[0, 0] = (10, 20, 30)  # BGR
    result = adapter.infer(FrameRequest(7, 11, image, point_budget=1_000))

    assert result.model_id == MOGE2_MODEL_ID
    assert result.is_metric is True
    assert result.depth is not None and result.depth.shape == (100, 200)
    assert result.valid_mask is not None and result.valid_mask.shape == (100, 200)
    assert result.points is not None
    assert result.normals is not None
    assert result.points.shape[:2] == result.depth[:: result.point_stride, :: result.point_stride].shape
    assert result.points.shape[0] * result.points.shape[1] <= 1_000
    assert result.intrinsics is not None and result.intrinsics.shape == (3, 3)
    assert result.precision == "FP32"
    assert result.preprocess_ms is not None and result.preprocess_ms >= 0.0
    assert result.postprocess_ms is not None and result.postprocess_ms >= 0.0
    assert result.worker_total_ms is not None
    assert result.worker_total_ms >= result.latency_ms
    assert result.vram_allocated_gib is None
    assert result.vram_reserved_gib is None

    fake = _FakeMoGeModel.last_instance
    assert fake is not None and fake.last_image is not None
    np.testing.assert_allclose(
        fake.last_image[:, 0, 0].numpy(),
        np.array([30, 20, 10], dtype=np.float32) / 255.0,
    )
    assert fake.last_kwargs["num_tokens"] == 1200
    assert fake.last_kwargs["force_projection"] is True
    assert fake.last_kwargs["apply_mask"] is False
    assert fake.last_kwargs["use_fp16"] is False


def test_load_metadata_and_warmup_are_runtime_ready() -> None:
    adapter = _adapter()
    adapter.warmup()
    metadata = adapter.get_metadata()

    assert metadata["id"] == MOGE2_MODEL_ID
    assert metadata["display_name"] == "MoGe-2 ViT-S Normal"
    assert metadata["is_metric"] is True
    assert metadata["outputs"]["points"] is True
    assert metadata["outputs"]["normals"] is True
    assert metadata["precision"] == "FP32"

    fake = _FakeMoGeModel.last_instance
    assert fake is not None
    assert fake.eval_called is True
    assert fake.sdpa_called is True


def test_unload_prevents_further_inference() -> None:
    adapter = _adapter()
    adapter.unload()
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    try:
        adapter.infer(FrameRequest(1, 1, image))
    except RuntimeError as exc:
        assert "not loaded" in str(exc)
    else:  # pragma: no cover - explicit failure message
        raise AssertionError("Inference unexpectedly succeeded after unload")

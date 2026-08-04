"""Repeated isolated-contract benchmark lifecycle stress using Development Mock."""

from __future__ import annotations

from pathlib import Path

from core.benchmark.runner import BenchmarkOptions, benchmark_model
from core.hardware.gpu_info import AcceleratorBackend, HardwareInfo
from core.models.catalog import MOCK_MODEL_ID


def _cpu() -> HardwareInfo:
    return HardwareInfo(
        backend=AcceleratorBackend.CPU,
        device="cpu",
        device_index=None,
        name="CPU",
        total_memory_bytes=None,
        free_memory_bytes=None,
        compute_capability=None,
        supports_fp16=False,
        supports_bf16=False,
        torch_available=False,
        torch_cuda_available=False,
        torch_version=None,
        torch_cuda_version=None,
        compiled_architectures=(),
        torch_build_compatible=None,
    )


def test_benchmark_lifecycle_remains_stable_across_repeated_runs(tmp_path: Path) -> None:
    for _ in range(10):
        result = benchmark_model(
            MOCK_MODEL_ID,
            assets_root=tmp_path / "assets",
            options=BenchmarkOptions(iterations=1, point_budget=10_000),
            hardware=_cpu(),
        )
        assert result["status"] == "passed"
        assert result["cleanup"]["released"] is True
        assert result["output"]["point_count"] <= 10_000

"""Registry-driven benchmark execution tests."""

from __future__ import annotations

from pathlib import Path

from core.benchmark.runner import (
    BenchmarkOptions,
    benchmark_model,
    production_benchmark_descriptors,
)
from core.hardware.gpu_info import AcceleratorBackend, HardwareInfo
from core.hardware.profiles import ComputeProfile
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


def test_production_benchmark_is_registry_driven_and_excludes_mock() -> None:
    descriptors = production_benchmark_descriptors()
    ids = [item.model_id for item in descriptors]

    assert len(ids) == 7
    assert MOCK_MODEL_ID not in ids
    assert all(item.benchmark.included for item in descriptors)
    assert any(item.benchmark.workload == "sequence" for item in descriptors)
    assert any(item.benchmark.classification == "offline_quality_depth" for item in descriptors)


def test_mock_executes_benchmark_contract_without_model_dependencies(tmp_path: Path) -> None:
    result = benchmark_model(
        MOCK_MODEL_ID,
        assets_root=tmp_path / "assets",
        options=BenchmarkOptions(
            profile=ComputeProfile.AUTO,
            iterations=2,
            point_budget=12_345,
        ),
        hardware=_cpu(),
    )

    assert result["status"] == "passed"
    assert result["iterations"] == 2
    assert result["runtime"]["device"] == "cpu"
    assert result["runtime"]["precision"] == "FP32"
    assert result["output"]["resolution"] == [384, 640]
    assert result["output"]["resolutions"] == [[640, 384], [384, 640]]
    assert result["output"]["point_count"] <= 12_345
    assert result["output"]["depth_scale"] == "synthetic"
    assert result["timing"]["load_ms"] >= 0.0
    assert result["timing"]["unload_ms"] >= 0.0
    assert result["timing"]["end_to_end_ms"]["count"] == 2.0

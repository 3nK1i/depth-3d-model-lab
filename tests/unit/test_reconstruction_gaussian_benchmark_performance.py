from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

from core.reconstruction import (
    GaussianBenchmarkHardwareProfile,
    GaussianHardwareEvidence,
    GaussianPerformanceMetrics,
    gaussian_benchmark_hardware_requirement,
)

ROOT = Path(__file__).resolve().parents[2]


def _load_performance_module():
    path = ROOT / "scripts" / "gaussian_benchmark_workers" / "performance.py"
    spec = importlib.util.spec_from_file_location("eia_test_gaussian_performance_helpers", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_named_hardware_profiles_are_identity_contracts_not_fake_measurements() -> None:
    rtx2060 = gaussian_benchmark_hardware_requirement(GaussianBenchmarkHardwareProfile.RTX_2060_6GB)
    assert rtx2060.nominal_vram_mib == 6144.0
    assert rtx2060.matches(gpu_name="NVIDIA GeForce RTX 2060", total_vram_mib=6144.0)
    assert not rtx2060.matches(gpu_name="NVIDIA GeForce RTX 5090", total_vram_mib=32768.0)

    rtx5090 = gaussian_benchmark_hardware_requirement(GaussianBenchmarkHardwareProfile.RTX_5090_32GB)
    assert rtx5090.nominal_vram_mib == 32768.0
    assert rtx5090.matches(gpu_name="NVIDIA GeForce RTX 5090", total_vram_mib=32600.0)
    assert not rtx5090.matches(gpu_name="NVIDIA GeForce RTX 2060", total_vram_mib=6144.0)


def test_custom_hardware_profile_accepts_observed_gpu_without_claiming_named_target() -> None:
    custom = gaussian_benchmark_hardware_requirement(GaussianBenchmarkHardwareProfile.CUSTOM)
    assert custom.matches(gpu_name="NVIDIA Test GPU", total_vram_mib=12345.0)
    assert not custom.matches(gpu_name="", total_vram_mib=12345.0)


def test_detailed_performance_contract_preserves_step1_step2_readiness_semantics() -> None:
    legacy_complete = GaussianPerformanceMetrics(
        peak_vram_mib=2100.0,
        training_duration_s=4.0,
        render_fps=30.0,
    )
    assert legacy_complete.measured is True
    assert legacy_complete.detailed_measured is False

    detailed = GaussianPerformanceMetrics(
        peak_training_vram_mib=2100.0,
        peak_render_vram_mib=1800.0,
        training_duration_s=4.0,
        render_duration_s=2.0,
        render_frame_count=60,
        render_repeats=3,
        render_fps=30.0,
        measurement_scope="backend_process_tree_nvidia_smi",
        render_measurement_mode="backend_native_warm",
    )
    assert detailed.peak_vram_mib == 2100.0
    assert detailed.measured is True
    assert detailed.detailed_measured is True
    payload = detailed.as_dict()
    assert payload["peak_training_vram_mib"] == 2100.0
    assert payload["peak_render_vram_mib"] == 1800.0


def test_detailed_performance_rejects_invalid_render_counts() -> None:
    with pytest.raises(Exception, match="render_frame_count"):
        GaussianPerformanceMetrics(render_frame_count=0)


def test_hardware_evidence_requires_complete_runtime_provenance() -> None:
    partial = GaussianHardwareEvidence(gpu_name="RTX 2060")
    assert partial.measured is False
    evidence = GaussianHardwareEvidence(
        gpu_name="NVIDIA GeForce RTX 2060",
        gpu_uuid="GPU-test",
        total_vram_mib=6144.0,
        driver_version="610.43.02",
        cuda_runtime_version="12.8",
        compute_capability=(7, 5),
        profile_compatible=True,
    )
    assert evidence.measured is True
    assert evidence.as_dict()["compute_capability"] == [7, 5]


def test_nvidia_smi_hardware_probe_parser(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_performance_module()
    monkeypatch.setattr(
        module,
        "_run_text",
        lambda command: "0, NVIDIA GeForce RTX 2060, GPU-abc, 6144, 610.43.02\n",
    )
    probe = module.probe_gpu_hardware(0)
    assert probe.gpu_index == 0
    assert probe.name == "NVIDIA GeForce RTX 2060"
    assert probe.uuid == "GPU-abc"
    assert probe.total_vram_mib == 6144.0


def test_process_tree_sampler_sums_only_target_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_performance_module()
    monkeypatch.setattr(module, "_descendant_pids", lambda root_pid: {100, 101, 102})
    monkeypatch.setattr(
        module,
        "_query_compute_process_memory_mib",
        lambda gpu_uuid: {100: 100.0, 101: 200.0, 999: 4000.0},
    )
    sampler = module.ProcessTreeGpuSampler(root_pid=100, gpu_uuid="GPU-abc", interval_s=1.0)
    sampler._sample_once()
    assert sampler.peak_vram_mib == 300.0
    assert sampler.sample_count == 1

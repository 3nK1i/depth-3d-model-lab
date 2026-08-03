"""Dependency-free tests for NVIDIA/PyTorch hardware discovery parsing."""

from __future__ import annotations

import pytest

import core.hardware.gpu_info as gpu_info
from core.hardware.gpu_info import AcceleratorBackend


def test_parse_nvidia_smi_supports_multiple_devices_and_capabilities() -> None:
    output = (
        "0, NVIDIA GeForce RTX 2060, 6144, 5120, 7.5\n"
        "1, NVIDIA GeForce RTX 5090, 32768, 30000, 12.0\n"
    )
    devices = gpu_info._parse_nvidia_smi(output)

    assert len(devices) == 2
    assert devices[0].index == 0
    assert devices[0].compute_capability == (7, 5)
    assert devices[0].total_memory_bytes == 6144 * 1024**2
    assert devices[1].compute_capability == (12, 0)


def test_parse_nvidia_smi_skips_malformed_rows() -> None:
    devices = gpu_info._parse_nvidia_smi(
        "not-an-index, Bad, 100, 50, 7.5\n2, Good GPU, N/A, 20, N/A\n"
    )
    assert len(devices) == 1
    assert devices[0].index == 2
    assert devices[0].total_memory_bytes is None
    assert devices[0].compute_capability is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [("6144", 6144 * 1024**2), ("1.5", int(1.5 * 1024**2)), ("N/A", None), ("bad", None)],
)
def test_mib_conversion(value, expected) -> None:
    assert gpu_info._mib_to_bytes(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [("7.5", (7, 5)), ("12.0", (12, 0)), ("N/A", None), ("bad", None)],
)
def test_compute_capability_parsing(value, expected) -> None:
    assert gpu_info._parse_compute_capability(value) == expected


def test_detect_hardware_returns_cpu_when_torch_and_nvidia_are_unavailable(monkeypatch) -> None:
    gpu_info.detect_hardware.cache_clear()
    monkeypatch.setattr(
        gpu_info,
        "_read_torch_metadata",
        lambda: gpu_info._TorchMetadata(False, False, None, None, (), None),
    )
    monkeypatch.setattr(gpu_info, "_query_nvidia_smi", lambda: ())

    detected = gpu_info.detect_hardware(0)

    assert detected.backend is AcceleratorBackend.CPU
    assert detected.device == "cpu"
    assert detected.warning and "PyTorch is not installed" in detected.warning
    gpu_info.detect_hardware.cache_clear()


def test_gpu_snapshot_formats_short_label_and_readiness() -> None:
    hardware = gpu_info.HardwareInfo(
        backend=AcceleratorBackend.CUDA,
        device="cuda:0",
        device_index=0,
        name="NVIDIA GeForce RTX 2060",
        total_memory_bytes=6 * 1024**3,
        free_memory_bytes=5 * 1024**3,
        compute_capability=(7, 5),
        supports_fp16=True,
        supports_bf16=False,
        torch_available=True,
        torch_cuda_available=True,
        torch_version="2.x",
        torch_cuda_version="13.0",
        compiled_architectures=("sm_75",),
        torch_build_compatible=True,
    )
    assert hardware.cuda_ready
    assert hardware.short_label == "RTX 2060 · 6.0 GB"
    assert "Compute capability: 7.5" in hardware.details

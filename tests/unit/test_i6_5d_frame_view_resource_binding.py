from __future__ import annotations

from core.hardware.gpu_info import AcceleratorBackend, HardwareInfo
from app.reconstruction_frame_view_resources import (
    available_ram_bytes_from_meminfo,
    build_startup_frame_view_resource_snapshot,
)


def _hardware() -> HardwareInfo:
    return HardwareInfo(
        backend=AcceleratorBackend.CUDA,
        device="cuda:0",
        device_index=0,
        name="NVIDIA RTX Test",
        total_memory_bytes=8 * 1024**3,
        free_memory_bytes=6 * 1024**3,
        compute_capability=(8, 6),
        supports_fp16=True,
        supports_bf16=True,
        torch_available=True,
        torch_cuda_available=True,
        torch_version="test",
        torch_cuda_version="test",
        compiled_architectures=("8.6",),
        torch_build_compatible=True,
    )


def test_i6_5d_meminfo_parser_uses_memavailable(tmp_path) -> None:
    path = tmp_path / "meminfo"
    path.write_text("MemTotal: 16384 kB\nMemAvailable: 12288 kB\n", encoding="utf-8")
    assert available_ram_bytes_from_meminfo(path) == 12288 * 1024


def test_i6_5d_missing_meminfo_is_fail_soft(tmp_path) -> None:
    assert available_ram_bytes_from_meminfo(tmp_path / "missing") is None


def test_i6_5d_startup_snapshot_uses_existing_hardware_snapshot(tmp_path) -> None:
    path = tmp_path / "meminfo"
    path.write_text("MemAvailable: 8388608 kB\n", encoding="utf-8")
    snap = build_startup_frame_view_resource_snapshot(_hardware(), meminfo_path=path)
    assert snap.gpu_name == "NVIDIA RTX Test"
    assert snap.total_vram_bytes == 8 * 1024**3
    assert snap.free_vram_bytes == 6 * 1024**3
    assert snap.available_ram_bytes == 8 * 1024**3
    assert snap.cuda_ready is True

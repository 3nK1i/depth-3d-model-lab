"""Hardware-profile rules for CPU, RTX 2060-class, and RTX 5090-class GPUs."""

from __future__ import annotations

from core.hardware.gpu_info import AcceleratorBackend, HardwareInfo
from core.hardware.profiles import (
    ComputeProfile,
    InferencePrecision,
    available_precisions,
    recommended_profile,
    resolve_profile,
)


def _hardware(
    *,
    vram_gib: float | None,
    capability: tuple[int, int] | None = (7, 5),
    cuda_ready: bool = True,
) -> HardwareInfo:
    backend = AcceleratorBackend.CUDA if vram_gib is not None else AcceleratorBackend.CPU
    return HardwareInfo(
        backend=backend,
        device="cuda:0" if backend is AcceleratorBackend.CUDA else "cpu",
        device_index=0 if backend is AcceleratorBackend.CUDA else None,
        name="Test GPU" if backend is AcceleratorBackend.CUDA else "CPU",
        total_memory_bytes=None if vram_gib is None else int(vram_gib * 1024**3),
        free_memory_bytes=None if vram_gib is None else int(vram_gib * 0.8 * 1024**3),
        compute_capability=capability if backend is AcceleratorBackend.CUDA else None,
        supports_fp16=backend is AcceleratorBackend.CUDA,
        supports_bf16=bool(capability and capability[0] >= 8),
        torch_available=True,
        torch_cuda_available=cuda_ready and backend is AcceleratorBackend.CUDA,
        torch_version="test",
        torch_cuda_version="test" if backend is AcceleratorBackend.CUDA else None,
        compiled_architectures=("sm_75",) if backend is AcceleratorBackend.CUDA else (),
        torch_build_compatible=True if backend is AcceleratorBackend.CUDA else None,
    )


def test_recommended_profile_boundaries() -> None:
    assert recommended_profile(_hardware(vram_gib=None)) is ComputeProfile.EFFICIENT
    assert recommended_profile(_hardware(vram_gib=6.0)) is ComputeProfile.EFFICIENT
    assert recommended_profile(_hardware(vram_gib=8.0)) is ComputeProfile.BALANCED
    assert recommended_profile(_hardware(vram_gib=19.9)) is ComputeProfile.BALANCED
    assert recommended_profile(_hardware(vram_gib=20.0, capability=(12, 0))) is ComputeProfile.ULTRA


def test_auto_2060_resolves_to_vits_fp16() -> None:
    resolved = resolve_profile(ComputeProfile.AUTO, _hardware(vram_gib=6.0))
    assert resolved.effective is ComputeProfile.EFFICIENT
    assert resolved.precision is InferencePrecision.FP16
    assert resolved.device == "cuda:0"
    assert resolved.spec.model_repo == "Ruicheng/moge-2-vits-normal"
    assert resolved.as_model_config()["point_budget"] == 250_000


def test_unsafe_ultra_request_is_downgraded() -> None:
    resolved = resolve_profile(ComputeProfile.ULTRA, _hardware(vram_gib=6.0))
    assert resolved.requested is ComputeProfile.ULTRA
    assert resolved.effective is ComputeProfile.EFFICIENT
    assert resolved.downgraded
    assert resolved.warning and "safe tier" in resolved.warning


def test_cpu_always_uses_fp32_and_cpu_device() -> None:
    resolved = resolve_profile(ComputeProfile.AUTO, _hardware(vram_gib=None))
    assert resolved.device == "cpu"
    assert resolved.precision is InferencePrecision.FP32
    assert resolved.warning


def test_available_precisions_follow_capability() -> None:
    assert available_precisions(_hardware(vram_gib=None)) == (InferencePrecision.FP32,)
    assert available_precisions(_hardware(vram_gib=6.0, capability=(7, 5))) == (
        InferencePrecision.FP16,
        InferencePrecision.FP32,
    )
    assert available_precisions(_hardware(vram_gib=32.0, capability=(12, 0))) == (
        InferencePrecision.FP16,
        InferencePrecision.FP32,
        InferencePrecision.BF16,
    )

"""Hardware-profile rules for CPU, RTX 2060-class, and RTX 5090-class GPUs."""

from __future__ import annotations

from core.hardware.gpu_info import AcceleratorBackend, HardwareInfo
from core.hardware.profiles import (
    ComputeProfile,
    InferencePrecision,
    available_precisions,
    profile_is_safe,
    recommended_profile,
    resolve_profile,
)


def _hardware(
    *,
    vram_gib: float | None,
    free_vram_gib: float | None = None,
    capability: tuple[int, int] | None = (7, 5),
    cuda_ready: bool = True,
    name: str = "Test GPU",
    build_compatible: bool | None = True,
) -> HardwareInfo:
    backend = AcceleratorBackend.CUDA if vram_gib is not None else AcceleratorBackend.CPU
    if free_vram_gib is None and vram_gib is not None:
        free_vram_gib = vram_gib * 0.8
    return HardwareInfo(
        backend=backend,
        device="cuda:0" if backend is AcceleratorBackend.CUDA else "cpu",
        device_index=0 if backend is AcceleratorBackend.CUDA else None,
        name=name if backend is AcceleratorBackend.CUDA else "CPU",
        total_memory_bytes=None if vram_gib is None else int(vram_gib * 1024**3),
        free_memory_bytes=None if free_vram_gib is None else int(free_vram_gib * 1024**3),
        compute_capability=capability if backend is AcceleratorBackend.CUDA else None,
        supports_fp16=backend is AcceleratorBackend.CUDA,
        supports_bf16=bool(capability and capability[0] >= 8),
        torch_available=True,
        torch_cuda_available=cuda_ready and backend is AcceleratorBackend.CUDA,
        torch_version="test",
        torch_cuda_version="test" if backend is AcceleratorBackend.CUDA else None,
        compiled_architectures=("sm_75",) if backend is AcceleratorBackend.CUDA else (),
        torch_build_compatible=build_compatible if backend is AcceleratorBackend.CUDA else None,
    )


def test_recommended_profile_boundaries() -> None:
    assert recommended_profile(_hardware(vram_gib=None)) is ComputeProfile.EFFICIENT
    assert recommended_profile(_hardware(vram_gib=6.0)) is ComputeProfile.EFFICIENT
    assert recommended_profile(_hardware(vram_gib=8.0)) is ComputeProfile.BALANCED
    assert recommended_profile(_hardware(vram_gib=19.9)) is ComputeProfile.BALANCED
    assert recommended_profile(_hardware(vram_gib=20.0, capability=(12, 0))) is ComputeProfile.ULTRA


def test_auto_2060_resolves_to_tuned_vits_fp16() -> None:
    resolved = resolve_profile(
        ComputeProfile.AUTO,
        _hardware(vram_gib=6.0, free_vram_gib=5.2, name="NVIDIA GeForce RTX 2060"),
    )
    config = resolved.as_model_config()

    assert resolved.effective is ComputeProfile.EFFICIENT
    assert resolved.precision is InferencePrecision.FP16
    assert resolved.device == "cuda:0"
    assert resolved.spec.model_repo == "Ruicheng/moge-2-vits-normal"
    assert config["point_budget"] == 250_000
    assert config["live_point_budget"] == 100_000
    assert config["warmup_size"] == 256
    assert config["warmup_runs"] == 1
    assert config["allow_tf32"] is False
    assert config["enable_native_sdpa"] is True


def test_auto_5090_resolves_to_tuned_vitl_without_manual_config() -> None:
    resolved = resolve_profile(
        ComputeProfile.AUTO,
        _hardware(
            vram_gib=32.0,
            free_vram_gib=30.0,
            capability=(12, 0),
            name="NVIDIA GeForce RTX 5090",
        ),
    )
    config = resolved.as_model_config()

    assert resolved.effective is ComputeProfile.ULTRA
    assert resolved.precision is InferencePrecision.FP16
    assert config["pretrained"] == "Ruicheng/moge-2-vitl-normal"
    assert config["resolution_level"] == 9
    assert config["num_tokens"] == 2500
    assert config["point_budget"] == 1_000_000
    assert config["live_point_budget"] == 500_000
    assert config["warmup_size"] == 512
    assert config["warmup_runs"] == 2
    assert config["allow_tf32"] is True
    assert config["matmul_precision"] == "high"


def test_5090_memory_pressure_temporarily_uses_balanced() -> None:
    hardware = _hardware(
        vram_gib=32.0,
        free_vram_gib=10.0,
        capability=(12, 0),
        name="NVIDIA GeForce RTX 5090",
    )
    resolved = resolve_profile(ComputeProfile.AUTO, hardware)

    assert resolved.effective is ComputeProfile.BALANCED
    assert not resolved.downgraded  # Auto simply chooses the safe current tier.

    forced = resolve_profile(ComputeProfile.ULTRA, hardware)
    assert forced.effective is ComputeProfile.BALANCED
    assert forced.downgraded
    assert forced.memory_pressure
    assert forced.warning and "free VRAM" in forced.warning


def test_unsafe_ultra_request_is_downgraded() -> None:
    resolved = resolve_profile(ComputeProfile.ULTRA, _hardware(vram_gib=6.0))
    assert resolved.requested is ComputeProfile.ULTRA
    assert resolved.effective is ComputeProfile.EFFICIENT
    assert resolved.downgraded
    assert resolved.warning and "safe tier" in resolved.warning


def test_incompatible_cuda_build_uses_cpu_safe_runtime() -> None:
    resolved = resolve_profile(
        ComputeProfile.AUTO,
        _hardware(
            vram_gib=32.0,
            free_vram_gib=30.0,
            capability=(12, 0),
            build_compatible=False,
        ),
    )
    assert resolved.device == "cpu"
    assert resolved.precision is InferencePrecision.FP32
    assert resolved.effective is ComputeProfile.EFFICIENT


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


def test_profile_safety_helper_matches_resolver() -> None:
    gpu_2060 = _hardware(vram_gib=6.0, free_vram_gib=5.2)
    assert profile_is_safe(ComputeProfile.AUTO, gpu_2060)
    assert profile_is_safe(ComputeProfile.EFFICIENT, gpu_2060)
    assert not profile_is_safe(ComputeProfile.BALANCED, gpu_2060)
    assert not profile_is_safe(ComputeProfile.ULTRA, gpu_2060)

from __future__ import annotations

import pytest

from core.reconstruction import (
    BackendCapabilities,
    InputSourceKind,
    ReconstructionBackendKind,
    ReconstructionQualityProfile,
    ReconstructionValidationError,
    recommended_reconstruction_profile,
    resolve_reconstruction_profile,
)
from tests.reconstruction_helpers import make_hardware


def test_rtx2060_uses_efficient_reconstruction_policy() -> None:
    hardware = make_hardware(
        total_gib=6.0,
        free_gib=5.2,
        name="NVIDIA GeForce RTX 2060",
    )
    assert recommended_reconstruction_profile(hardware) is ReconstructionQualityProfile.EFFICIENT
    resolved = resolve_reconstruction_profile(hardware)
    assert resolved.effective is ReconstructionQualityProfile.EFFICIENT
    assert resolved.memory_policy.device == "cuda:0"
    assert resolved.memory_policy.gpu_budget_bytes > 0
    assert resolved.memory_policy.gpu_budget_bytes < int(5.2 * 1024**3)


def test_rtx5090_uses_high_quality_reconstruction_policy() -> None:
    hardware = make_hardware(
        total_gib=32.0,
        free_gib=30.0,
        capability=(12, 0),
        name="NVIDIA GeForce RTX 5090",
    )
    resolved = resolve_reconstruction_profile(hardware)
    assert resolved.effective is ReconstructionQualityProfile.HIGH_QUALITY
    assert resolved.memory_policy.max_active_keyframes == 800
    assert resolved.memory_policy.gpu_budget_bytes == int(22.0 * 1024**3)


def test_rtx5090_memory_pressure_selects_balanced() -> None:
    hardware = make_hardware(
        total_gib=32.0,
        free_gib=12.0,
        capability=(12, 0),
    )
    automatic = resolve_reconstruction_profile(hardware)
    assert automatic.effective is ReconstructionQualityProfile.BALANCED
    forced = resolve_reconstruction_profile(
        hardware, ReconstructionQualityProfile.HIGH_QUALITY
    )
    assert forced.effective is ReconstructionQualityProfile.BALANCED
    assert forced.downgraded
    assert forced.memory_pressure
    assert forced.warning and "free VRAM" in forced.warning


def test_cpu_policy_remains_dependency_light_and_bounded() -> None:
    resolved = resolve_reconstruction_profile(make_hardware(total_gib=None))
    assert resolved.effective is ReconstructionQualityProfile.EFFICIENT
    assert resolved.memory_policy.device == "cpu"
    assert resolved.memory_policy.gpu_budget_bytes == 0
    assert resolved.warning


def test_memory_sub_budgets_equal_total_gpu_budget() -> None:
    resolved = resolve_reconstruction_profile(
        make_hardware(total_gib=16.0, free_gib=14.0),
        ReconstructionQualityProfile.BALANCED,
    )
    policy = resolved.memory_policy
    assert (
        policy.frame_cache_bytes
        + policy.keyframe_cache_bytes
        + policy.working_memory_bytes
        == policy.gpu_budget_bytes
    )


def test_backend_capability_validation() -> None:
    capabilities = BackendCapabilities(
        backend_id="mock-sfm",
        display_name="Mock SfM",
        backend_version="1.0",
        kinds=(ReconstructionBackendKind.OFFLINE_SFM,),
        source_kinds=(InputSourceKind.IMAGE_SEQUENCE, InputSourceKind.RECORDED_VIDEO),
        quality_profiles=(
            ReconstructionQualityProfile.EFFICIENT,
            ReconstructionQualityProfile.BALANCED,
        ),
        supports_checkpoint=True,
        supports_resume=True,
        optional_dependencies=("optional-sfm",),
    )
    assert capabilities.supports_kind(ReconstructionBackendKind.OFFLINE_SFM)
    assert not capabilities.requires_cuda


def test_backend_resume_requires_checkpoint_support() -> None:
    with pytest.raises(ReconstructionValidationError, match="checkpoint"):
        BackendCapabilities(
            backend_id="bad",
            display_name="Bad Backend",
            backend_version="1",
            kinds=(ReconstructionBackendKind.LIVE_SLAM,),
            source_kinds=(InputSourceKind.USB_CAMERA,),
            quality_profiles=(ReconstructionQualityProfile.EFFICIENT,),
            supports_resume=True,
            supports_checkpoint=False,
        )


def test_cuda_required_backend_cannot_claim_cpu_support() -> None:
    with pytest.raises(ReconstructionValidationError, match="CUDA-required"):
        BackendCapabilities(
            backend_id="bad-cuda",
            display_name="Bad CUDA Backend",
            backend_version="1",
            kinds=(ReconstructionBackendKind.DENSE_FUSION,),
            source_kinds=(InputSourceKind.IMAGE_SEQUENCE,),
            quality_profiles=(ReconstructionQualityProfile.HIGH_QUALITY,),
            requires_cuda=True,
            supports_cpu=True,
        )


def test_zero_free_vram_never_falls_back_to_total_vram() -> None:
    hardware = make_hardware(total_gib=32.0, free_gib=0.0, capability=(12, 0))
    with pytest.raises(ReconstructionValidationError, match="No safe GPU memory"):
        resolve_reconstruction_profile(hardware)

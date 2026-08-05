from __future__ import annotations

import pytest

from core.contracts.validation import ContractValidationError
from core.perception.metadata import (
    AdapterCapability,
    CoResidencyPolicy,
    PerceptionCapabilities,
    PerceptionResourceMetadata,
    PerceptionTask,
    PreferredDevice,
)


def test_perception_resource_metadata_supports_2060_and_5090_profiles() -> None:
    resources = PerceptionResourceMetadata(
        estimated_vram_mb=4200,
        minimum_vram_mb=3000,
        preferred_device=PreferredDevice.CUDA,
        cpu_fallback_supported=True,
        co_residency_policy=CoResidencyPolicy.PREFER_ISOLATED,
        exclusive_gpu_required=False,
        supported_precisions=("fp32", "fp16"),
        supported_hardware_profiles=("rtx2060", "rtx5090"),
    )
    metadata = resources.as_dict()
    assert metadata["estimated_vram_mb"] == 4200
    assert metadata["supported_hardware_profiles"] == ["rtx2060", "rtx5090"]


def test_exclusive_gpu_and_co_residency_policy_must_agree() -> None:
    with pytest.raises(ContractValidationError, match="exclusive_gpu_required"):
        PerceptionResourceMetadata(
            exclusive_gpu_required=True,
            co_residency_policy=CoResidencyPolicy.ALLOW,
        )
    with pytest.raises(ContractValidationError, match="requires exclusive"):
        PerceptionResourceMetadata(
            exclusive_gpu_required=False,
            co_residency_policy=CoResidencyPolicy.EXCLUSIVE,
        )


def test_capabilities_are_specific_to_declared_operations() -> None:
    prompted = PerceptionCapabilities(
        tasks=(PerceptionTask.PROMPTED_SEGMENTATION,),
        adapter_capabilities=(AdapterCapability.PROMPTED_SEGMENTATION,),
    )
    assert AdapterCapability.FRAME_PERCEPTION not in prompted.adapter_capabilities

    with pytest.raises(ContractValidationError, match="prompted adapter"):
        PerceptionCapabilities(
            tasks=(PerceptionTask.PROMPTED_SEGMENTATION,),
            adapter_capabilities=(AdapterCapability.FRAME_PERCEPTION,),
        )

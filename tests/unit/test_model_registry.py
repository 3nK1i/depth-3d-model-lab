"""Phase 6 Step 1 model-registry contract tests."""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import pytest

from core.hardware.gpu_info import AcceleratorBackend, HardwareInfo
from core.hardware.profiles import ComputeProfile, resolve_profile
from core.models.adapters.mock_adapter import MockGeometryModel
from core.models.catalog import MOGE2_MODEL_ID, MOCK_MODEL_ID, SUPPORTED_MODEL_IDS
from core.models.metadata import (
    CheckpointSource,
    DepthScale,
    ModelCapabilities,
    ModelCheckpoint,
    ModelDescriptor,
    ModelOutputs,
)
from core.models.registry import (
    MODEL_REGISTRY,
    DuplicateModelError,
    ModelRegistry,
    ModelRegistryError,
    UnknownModelError,
    build_model,
    check_model_availability,
    get_default_model_descriptor,
    get_model_descriptor,
    iter_model_descriptors,
    merge_model_metadata,
    resolve_model_profile_metadata,
    resolve_model_runtime_config,
)
from core.models.runtime_config import ModelRuntimeContext


def _gpu(vram_gib: float = 6.0) -> HardwareInfo:
    return HardwareInfo(
        backend=AcceleratorBackend.CUDA,
        device="cuda:0",
        device_index=0,
        name="NVIDIA GeForce RTX 2060",
        total_memory_bytes=int(vram_gib * 1024**3),
        free_memory_bytes=int(vram_gib * 0.9 * 1024**3),
        compute_capability=(7, 5),
        supports_fp16=True,
        supports_bf16=False,
        torch_available=True,
        torch_cuda_available=True,
        torch_version="test",
        torch_cuda_version="test",
        compiled_architectures=("sm_75",),
        torch_build_compatible=True,
    )


def _descriptor(model_id: str = "test-model", *, default: bool = True) -> ModelDescriptor:
    return ModelDescriptor(
        model_id=model_id,
        display_name="Test Model",
        adapter_path="core.models.adapters.mock_adapter:MockGeometryModel",
        runtime_config_builder="core.models.runtime_config:build_static_runtime_config",
        profile_metadata_builder="core.models.runtime_config:build_mock_profile_metadata",
        capabilities=ModelCapabilities(
            outputs=ModelOutputs(depth=True),
            depth_scale=DepthScale.RELATIVE,
            source_types=("image",),
        ),
        default_config={"device": "cpu", "precision": "fp32"},
        default=default,
    )


def test_builtin_registry_is_authoritative_and_phase5_ids_remain_compatible() -> None:
    assert MODEL_REGISTRY.ids() == SUPPORTED_MODEL_IDS
    assert get_default_model_descriptor().model_id == MOGE2_MODEL_ID
    assert [item.model_id for item in iter_model_descriptors()] == list(SUPPORTED_MODEL_IDS)

    moge = get_model_descriptor(MOGE2_MODEL_ID)
    assert moge.capabilities.is_metric is True
    assert moge.capabilities.outputs.points is True
    assert moge.capabilities.intrinsics_fallback.value == "estimated"
    assert moge.uses_compute_profile is True
    assert moge.as_metadata()["uses_compute_profile"] is True

    mock = get_model_descriptor(MOCK_MODEL_ID)
    assert mock.development_only is True
    assert mock.capabilities.depth_scale is DepthScale.SYNTHETIC
    assert mock.uses_compute_profile is False


def test_worker_factory_builds_adapters_lazily_through_registry() -> None:
    assert isinstance(build_model(MOCK_MODEL_ID), MockGeometryModel)
    with pytest.raises(UnknownModelError, match="Unknown model id"):
        build_model("not-registered")


def test_runtime_config_dispatch_has_no_controller_model_branching() -> None:
    hardware = _gpu()
    resolved = resolve_profile(ComputeProfile.AUTO, hardware)
    context = ModelRuntimeContext(
        hardware_info=hardware,
        resolved_profile=resolved,
        point_budget=123_456,
    )

    moge_config = resolve_model_runtime_config(MOGE2_MODEL_ID, context)
    assert moge_config["pretrained"] == "Ruicheng/moge-2-vits-normal"
    assert moge_config["point_budget"] == 123_456
    assert moge_config["device"] == "cuda:0"

    mock_config = resolve_model_runtime_config(MOCK_MODEL_ID, context)
    assert mock_config == {
        "device": "cpu",
        "precision": "fp32",
        "latency_ms": 30.0,
    }


def test_profile_presentation_is_registry_dispatched() -> None:
    resolved = resolve_profile(ComputeProfile.AUTO, _gpu())
    moge = resolve_model_profile_metadata(MOGE2_MODEL_ID, resolved)
    assert moge.variant_label == "MoGe-2 ViT-S Normal"
    assert moge.detail_label == "Level 5 · 1200 tok"
    assert moge.point_budget == 250_000
    assert moge.live_point_budget == 100_000

    mock = resolve_model_profile_metadata(MOCK_MODEL_ID, resolved)
    assert mock.variant_label == "Synthetic Diagnostic"
    assert mock.detail_label == "No AI inference"
    assert mock.precision_label == "FP32"


def test_registry_metadata_overrides_adapter_capability_claims() -> None:
    descriptor = get_model_descriptor(MOGE2_MODEL_ID)
    merged = merge_model_metadata(
        descriptor,
        {
            "id": "wrong-id",
            "display_name": "Runtime Variant",
            "is_metric": False,
            "outputs": {"depth": False},
            "device": "cuda:0",
        },
    )
    assert merged["id"] == MOGE2_MODEL_ID
    assert merged["display_name"] == "Runtime Variant"
    assert merged["is_metric"] is True
    assert merged["outputs"]["depth"] is True
    assert merged["outputs"]["points"] is True
    assert merged["device"] == "cuda:0"
    assert merged["presentation"] == descriptor.presentation.as_dict()
    assert merged["license"] == descriptor.license.as_dict()


def test_dependency_and_hardware_availability_are_declared(monkeypatch) -> None:
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str):
        if name == "moge":
            return None
        return real_find_spec(name)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    availability = check_model_availability(MOGE2_MODEL_ID, _gpu())
    assert availability.available is False
    assert availability.missing_dependencies == ("moge",)
    assert "Missing dependencies" in availability.reason

    # The diagnostic adapter remains usable on a CUDA host because it selects
    # its CPU device through registry runtime config.
    mock = check_model_availability(MOCK_MODEL_ID, _gpu())
    assert mock.available is True


def test_local_checkpoint_availability_detects_missing_assets(tmp_path) -> None:
    registry = ModelRegistry()
    missing = tmp_path / "missing-model.pth"
    descriptor = ModelDescriptor(
        model_id="local-model",
        display_name="Local Model",
        adapter_path="core.models.adapters.mock_adapter:MockGeometryModel",
        runtime_config_builder="core.models.runtime_config:build_static_runtime_config",
        profile_metadata_builder="core.models.runtime_config:build_mock_profile_metadata",
        capabilities=ModelCapabilities(outputs=ModelOutputs(depth=True)),
        checkpoints=(
            ModelCheckpoint(
                str(missing),
                source=CheckpointSource.LOCAL,
            ),
        ),
        default=True,
    )
    registry.register(descriptor)

    availability = registry.availability("local-model")
    assert availability.available is False
    assert availability.checkpoint_status == "missing"
    assert availability.missing_assets == (str(missing),)

    missing.write_bytes(b"weights")
    availability = registry.availability("local-model")
    assert availability.available is True
    assert availability.checkpoint_status == "available"


def test_registry_rejects_duplicates_invalid_ids_and_multiple_defaults() -> None:
    registry = ModelRegistry()
    registry.register(_descriptor())

    with pytest.raises(DuplicateModelError):
        registry.register(_descriptor())

    with pytest.raises(ModelRegistryError, match="Only one"):
        registry.register(_descriptor("second-model", default=True))

    with pytest.raises(ModelRegistryError, match="Invalid model id"):
        ModelRegistry().register(_descriptor("Invalid Model"))


def test_registry_rejects_nonprimitive_runtime_config() -> None:
    registry = ModelRegistry()
    descriptor = ModelDescriptor(
        model_id="bad-config",
        display_name="Bad Config",
        adapter_path="core.models.adapters.mock_adapter:MockGeometryModel",
        runtime_config_builder="core.models.runtime_config:build_static_runtime_config",
        profile_metadata_builder="core.models.runtime_config:build_mock_profile_metadata",
        capabilities=ModelCapabilities(outputs=ModelOutputs(depth=True)),
        default_config={"bad": SimpleNamespace(value=1)},
        default=True,
    )
    with pytest.raises(ModelRegistryError, match="invalid default runtime config"):
        registry.register(descriptor)


def test_runtime_overrides_must_remain_primitive_for_ipc() -> None:
    with pytest.raises(TypeError, match="not primitive"):
        resolve_model_runtime_config(
            MOCK_MODEL_ID,
            ModelRuntimeContext(overrides={"bad": SimpleNamespace(value=1)}),
        )


def test_registry_rejects_inconsistent_intrinsics_and_duplicate_sources() -> None:
    no_provenance = _descriptor()
    no_provenance = ModelDescriptor(
        model_id="bad-intrinsics",
        display_name=no_provenance.display_name,
        adapter_path=no_provenance.adapter_path,
        runtime_config_builder=no_provenance.runtime_config_builder,
        profile_metadata_builder=no_provenance.profile_metadata_builder,
        capabilities=ModelCapabilities(
            outputs=ModelOutputs(depth=True, intrinsics=True),
            depth_scale=DepthScale.RELATIVE,
            source_types=("image",),
        ),
        default=True,
    )
    with pytest.raises(ModelRegistryError, match="without defining their provenance"):
        ModelRegistry().register(no_provenance)

    duplicate_sources = ModelDescriptor(
        model_id="duplicate-sources",
        display_name="Duplicate Sources",
        adapter_path="core.models.adapters.mock_adapter:MockGeometryModel",
        runtime_config_builder="core.models.runtime_config:build_static_runtime_config",
        profile_metadata_builder="core.models.runtime_config:build_mock_profile_metadata",
        capabilities=ModelCapabilities(
            outputs=ModelOutputs(depth=True),
            depth_scale=DepthScale.RELATIVE,
            source_types=("image", "image"),
        ),
        default=True,
    )
    with pytest.raises(ModelRegistryError, match="duplicate source types"):
        ModelRegistry().register(duplicate_sources)


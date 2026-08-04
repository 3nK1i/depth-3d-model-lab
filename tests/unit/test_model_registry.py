"""Phase 6 Step 1 model-registry contract tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.hardware.gpu_info import AcceleratorBackend, HardwareInfo
from core.hardware.profiles import ComputeProfile, resolve_profile
from core.models.adapters.depth_pro_adapter import DepthProGeometryModel
from core.models.adapters.depth_anything_v2_adapter import DepthAnythingV2GeometryModel
from core.models.adapters.mock_adapter import MockGeometryModel
from core.models.adapters.metric3d_v2_adapter import Metric3DV2GeometryModel
from core.models.adapters.marigold_adapter import MarigoldGeometryModel
from core.models.adapters.unidepth_v2_adapter import UniDepthV2GeometryModel
from core.models.adapters.video_depth_anything_adapter import VideoDepthAnythingGeometryModel
from core.models.catalog import (
    DEPTH_ANYTHING_V2_MODEL_ID,
    DEPTH_PRO_MODEL_ID,
    MARIGOLD_MODEL_ID,
    MOGE2_MODEL_ID,
    MOCK_MODEL_ID,
    METRIC3D_V2_MODEL_ID,
    SUPPORTED_MODEL_IDS,
    UNIDEPTH_V2_MODEL_ID,
    VIDEO_DEPTH_ANYTHING_MODEL_ID,
)
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
    assert isinstance(build_model(DEPTH_PRO_MODEL_ID), DepthProGeometryModel)
    assert isinstance(build_model(UNIDEPTH_V2_MODEL_ID), UniDepthV2GeometryModel)
    assert isinstance(build_model(METRIC3D_V2_MODEL_ID), Metric3DV2GeometryModel)
    assert isinstance(build_model(DEPTH_ANYTHING_V2_MODEL_ID), DepthAnythingV2GeometryModel)
    assert isinstance(build_model(VIDEO_DEPTH_ANYTHING_MODEL_ID), VideoDepthAnythingGeometryModel)
    assert isinstance(build_model(MARIGOLD_MODEL_ID), MarigoldGeometryModel)
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

def test_depth_pro_registry_contract_and_profile_dispatch() -> None:
    descriptor = get_model_descriptor(DEPTH_PRO_MODEL_ID)
    assert descriptor.capabilities.is_metric is True
    assert descriptor.capabilities.outputs.depth is True
    assert descriptor.capabilities.outputs.points is False
    assert descriptor.capabilities.outputs.intrinsics is True
    assert descriptor.capabilities.outputs.valid_mask is True
    assert descriptor.capabilities.intrinsics_source.value == "model_predicted"
    assert descriptor.capabilities.intrinsics_fallback.value == "estimated"
    assert descriptor.hardware.supported_precisions == ("fp32", "fp16")
    assert descriptor.checkpoints[0].source is CheckpointSource.LOCAL
    assert descriptor.uses_compute_profile is True

    hardware = _gpu()
    resolved = resolve_profile(ComputeProfile.AUTO, hardware)
    context = ModelRuntimeContext(
        hardware_info=hardware,
        resolved_profile=resolved,
        point_budget=222_222,
    )
    config = resolve_model_runtime_config(DEPTH_PRO_MODEL_ID, context)
    assert config["checkpoint"] == "checkpoints/depth_pro.pt"
    assert config["device"] == "cuda:0"
    assert config["precision"] == "fp16"
    assert config["point_budget"] == 222_222
    assert config["live_point_budget"] == 100_000
    assert config["internal_resolution"] == 1536

    metadata = resolve_model_profile_metadata(DEPTH_PRO_MODEL_ID, resolved)
    assert metadata.variant_label == "Depth Pro · DINOv2-L"
    assert metadata.detail_label == "1536² internal · Metric"
    assert metadata.precision_label == "FP16"
    assert metadata.point_budget == 250_000
    assert metadata.live_point_budget == 100_000


def test_depth_pro_availability_reports_dependency_and_project_local_checkpoint(
    monkeypatch,
    tmp_path,
) -> None:
    import core.models.registry as registry_module

    descriptor = get_model_descriptor(DEPTH_PRO_MODEL_ID)
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str):
        if name == "depth_pro":
            return None
        return real_find_spec(name)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(
        registry_module,
        "_resolve_local_asset",
        lambda _identifier: tmp_path / "depth_pro.pt",
    )
    availability = check_model_availability(DEPTH_PRO_MODEL_ID, _gpu())
    assert availability.available is False
    assert "depth-pro" in availability.missing_dependencies
    assert availability.checkpoint_status == "missing"
    assert availability.missing_assets == (descriptor.checkpoints[0].identifier,)

    # Once dependency and checkpoint exist, the same registry entry is ready.
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: object())
    (tmp_path / "depth_pro.pt").write_bytes(b"weights")
    availability = check_model_availability(DEPTH_PRO_MODEL_ID, _gpu())
    assert availability.available is True
    assert availability.checkpoint_status == "available"



def test_unidepth_v2_registry_contract_and_gpu_profile_dispatch() -> None:
    descriptor = get_model_descriptor(UNIDEPTH_V2_MODEL_ID)
    assert descriptor.capabilities.is_metric is True
    assert descriptor.capabilities.outputs.points is True
    assert descriptor.capabilities.outputs.intrinsics is True
    assert descriptor.capabilities.outputs.confidence is True
    assert descriptor.capabilities.confidence_kind.value == "uncertainty"
    assert descriptor.capabilities.intrinsics_fallback.value == "estimated"
    assert descriptor.hardware.supported_precisions == ("fp32", "fp16")
    assert descriptor.license.status == "noncommercial_review_required"

    efficient = resolve_profile(ComputeProfile.AUTO, _gpu(6.0))
    config = resolve_model_runtime_config(
        UNIDEPTH_V2_MODEL_ID,
        ModelRuntimeContext(resolved_profile=efficient, point_budget=111_111),
    )
    assert config["variant"] == "vits14"
    assert config["repository"] == "lpiccinelli/unidepth-v2-vits14"
    assert config["checkpoint_dir"] == "checkpoints/unidepth-v2/vits14"
    assert config["resolution_level"] == 2
    assert config["precision"] == "fp16"
    assert config["point_budget"] == 111_111

    ultra_hardware = _gpu(32.0)
    ultra_hardware = HardwareInfo(
        backend=ultra_hardware.backend, device=ultra_hardware.device,
        device_index=ultra_hardware.device_index, name="NVIDIA GeForce RTX 5090",
        total_memory_bytes=ultra_hardware.total_memory_bytes,
        free_memory_bytes=ultra_hardware.free_memory_bytes,
        compute_capability=(12, 0), supports_fp16=True, supports_bf16=True,
        torch_available=True, torch_cuda_available=True, torch_version="test",
        torch_cuda_version="test", compiled_architectures=("sm_120",),
        torch_build_compatible=True,
    )
    ultra = resolve_profile(ComputeProfile.AUTO, ultra_hardware)
    config = resolve_model_runtime_config(
        UNIDEPTH_V2_MODEL_ID, ModelRuntimeContext(resolved_profile=ultra)
    )
    assert config["variant"] == "vitl14"
    assert config["resolution_level"] == 8
    metadata = resolve_model_profile_metadata(UNIDEPTH_V2_MODEL_ID, ultra)
    assert metadata.variant_label == "UniDepthV2 ViT-L/14"
    assert metadata.detail_label == "Level 8 · Metric + uncertainty"
    assert metadata.precision_label == "FP16"


def test_metric3d_v2_registry_contract_and_gpu_profile_dispatch() -> None:
    descriptor = get_model_descriptor(METRIC3D_V2_MODEL_ID)
    assert descriptor.capabilities.is_metric is True
    assert descriptor.capabilities.outputs.points is True
    assert descriptor.capabilities.outputs.colors is True
    assert descriptor.capabilities.outputs.normals is True
    assert descriptor.capabilities.outputs.intrinsics is True
    assert descriptor.capabilities.outputs.confidence is True
    assert descriptor.capabilities.intrinsics_source.value == "estimated"
    assert descriptor.capabilities.confidence_kind.value == "confidence"
    assert descriptor.hardware.supported_precisions == ("fp32", "fp16")
    assert descriptor.license.name == "BSD 2-Clause"
    assert descriptor.license.status == "permissive_reviewed"

    efficient = resolve_profile(ComputeProfile.AUTO, _gpu(6.0))
    config = resolve_model_runtime_config(
        METRIC3D_V2_MODEL_ID,
        ModelRuntimeContext(resolved_profile=efficient, point_budget=123_456),
    )
    assert config["entrypoint"] == "metric3d_vit_small"
    assert config["checkpoint"].endswith("metric_depth_vit_small_800k.pth")
    assert config["variant"] == "vit-small"
    assert config["input_height"] == 616
    assert config["input_width"] == 1064
    assert config["precision"] == "fp16"
    assert config["point_budget"] == 123_456
    metadata = resolve_model_profile_metadata(METRIC3D_V2_MODEL_ID, efficient)
    assert metadata.variant_label == "Metric3D v2 ViT-S"
    assert metadata.detail_label == "1064×616 · Metric + normals"
    assert metadata.precision_label == "FP16"

    ultra_hardware = _gpu(32.0)
    ultra_hardware = HardwareInfo(
        backend=ultra_hardware.backend, device=ultra_hardware.device,
        device_index=ultra_hardware.device_index, name="NVIDIA GeForce RTX 5090",
        total_memory_bytes=ultra_hardware.total_memory_bytes,
        free_memory_bytes=ultra_hardware.free_memory_bytes,
        compute_capability=(12, 0), supports_fp16=True, supports_bf16=True,
        torch_available=True, torch_cuda_available=True, torch_version="test",
        torch_cuda_version="test", compiled_architectures=("sm_120",),
        torch_build_compatible=True,
    )
    ultra = resolve_profile(ComputeProfile.AUTO, ultra_hardware)
    config = resolve_model_runtime_config(
        METRIC3D_V2_MODEL_ID, ModelRuntimeContext(resolved_profile=ultra)
    )
    assert config["entrypoint"] == "metric3d_vit_large"
    assert config["checkpoint"].endswith("metric_depth_vit_large_800k.pth")
    assert config["variant"] == "vit-large"
    metadata = resolve_model_profile_metadata(METRIC3D_V2_MODEL_ID, ultra)
    assert metadata.variant_label == "Metric3D v2 ViT-L"
    assert metadata.precision_label == "FP16"


def test_metric3d_v2_availability_requires_source_and_checkpoint_markers(
    monkeypatch, tmp_path
) -> None:
    import core.models.registry as registry_module

    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: object() if name in {"torch", "torchvision", "cv2", "mmengine", "mmcv", "timm"} else real_find_spec(name),
    )
    monkeypatch.setattr(
        registry_module,
        "_resolve_local_asset",
        lambda identifier: tmp_path / Path(identifier).name,
    )
    availability = check_model_availability(METRIC3D_V2_MODEL_ID, _gpu())
    assert availability.available is False
    assert availability.checkpoint_status == "missing"
    assert len(availability.missing_assets) == 2

    (tmp_path / ".eia_metric3d_v2_ready.json").write_text("{}")
    (tmp_path / ".ready").write_text("{}")
    availability = check_model_availability(METRIC3D_V2_MODEL_ID, _gpu())
    assert availability.available is True
    assert availability.checkpoint_status == "managed_remote"



def test_depth_anything_v2_registry_contract_and_profile_dispatch() -> None:
    descriptor = get_model_descriptor(DEPTH_ANYTHING_V2_MODEL_ID)
    assert descriptor.capabilities.is_metric is False
    assert descriptor.capabilities.depth_scale is DepthScale.RELATIVE
    assert descriptor.capabilities.outputs.depth is True
    assert descriptor.capabilities.outputs.points is True
    assert descriptor.capabilities.outputs.colors is True
    assert descriptor.capabilities.outputs.intrinsics is True
    assert descriptor.capabilities.intrinsics_source.value == "estimated"
    assert descriptor.capabilities.stateful_sequence is False
    assert descriptor.license.status == "variant_dependent_review_required"

    efficient = resolve_profile(ComputeProfile.AUTO, _gpu(6.0))
    config = resolve_model_runtime_config(
        DEPTH_ANYTHING_V2_MODEL_ID,
        ModelRuntimeContext(resolved_profile=efficient, point_budget=123_456),
    )
    assert config["encoder"] == "vits"
    assert config["checkpoint"].endswith("depth_anything_v2_vits.pth")
    assert config["input_size"] == 518
    assert config["precision"] == "fp16"
    assert config["point_budget"] == 123_456
    metadata = resolve_model_profile_metadata(DEPTH_ANYTHING_V2_MODEL_ID, efficient)
    assert metadata.variant_label == "Depth Anything V2 Small"
    assert metadata.detail_label == "518 px · Relative depth"


def test_video_depth_anything_registry_contract_and_profile_dispatch() -> None:
    descriptor = get_model_descriptor(VIDEO_DEPTH_ANYTHING_MODEL_ID)
    assert descriptor.capabilities.is_metric is True
    assert descriptor.capabilities.depth_scale is DepthScale.METRIC
    assert descriptor.capabilities.outputs.points is True
    assert descriptor.capabilities.outputs.colors is True
    assert descriptor.capabilities.outputs.intrinsics is True
    assert descriptor.capabilities.intrinsics_source.value == "estimated"
    assert descriptor.capabilities.stateful_sequence is True
    assert descriptor.capabilities.supports_live is True
    assert descriptor.license.status == "variant_dependent_review_required"

    efficient = resolve_profile(ComputeProfile.AUTO, _gpu(6.0))
    config = resolve_model_runtime_config(
        VIDEO_DEPTH_ANYTHING_MODEL_ID,
        ModelRuntimeContext(resolved_profile=efficient, point_budget=77_777),
    )
    assert config["encoder"] == "vits"
    assert config["checkpoint"].endswith("metric_video_depth_anything_vits.pth")
    assert config["input_size"] == 392
    assert config["max_resolution"] == 960
    assert config["point_budget"] == 77_777
    metadata = resolve_model_profile_metadata(VIDEO_DEPTH_ANYTHING_MODEL_ID, efficient)
    assert metadata.variant_label == "Metric Video Depth Anything Small"
    assert metadata.detail_label == "392 px · Temporal metric"


def test_depth_anything_ultra_profiles_select_large_variants() -> None:
    hardware = _gpu(32.0)
    hardware = HardwareInfo(
        backend=hardware.backend,
        device=hardware.device,
        device_index=hardware.device_index,
        name="NVIDIA GeForce RTX 5090",
        total_memory_bytes=hardware.total_memory_bytes,
        free_memory_bytes=hardware.free_memory_bytes,
        compute_capability=(12, 0),
        supports_fp16=True,
        supports_bf16=True,
        torch_available=True,
        torch_cuda_available=True,
        torch_version="test",
        torch_cuda_version="test",
        compiled_architectures=("sm_120",),
        torch_build_compatible=True,
    )
    ultra = resolve_profile(ComputeProfile.AUTO, hardware)
    image = resolve_model_runtime_config(
        DEPTH_ANYTHING_V2_MODEL_ID, ModelRuntimeContext(resolved_profile=ultra)
    )
    video = resolve_model_runtime_config(
        VIDEO_DEPTH_ANYTHING_MODEL_ID, ModelRuntimeContext(resolved_profile=ultra)
    )
    assert image["encoder"] == "vitl"
    assert image["input_size"] == 700
    assert video["encoder"] == "vitl"
    assert video["input_size"] == 518
    assert video["max_resolution"] == 1920


def test_marigold_registry_profiles_and_capabilities() -> None:
    descriptor = get_model_descriptor(MARIGOLD_MODEL_ID)
    assert descriptor.capabilities.depth_scale is DepthScale.RELATIVE
    assert descriptor.capabilities.outputs.confidence is True
    assert descriptor.capabilities.confidence_kind.value == "uncertainty"
    assert descriptor.capabilities.supports_live is False
    assert descriptor.license.status == "review_required"

    efficient_hw = _gpu(6.0)
    efficient = resolve_profile(ComputeProfile.AUTO, efficient_hw)
    config = resolve_model_runtime_config(
        MARIGOLD_MODEL_ID,
        ModelRuntimeContext(hardware_info=efficient_hw, resolved_profile=efficient),
    )
    assert config["processing_resolution"] == 512
    assert config["num_inference_steps"] == 1
    assert config["ensemble_size"] == 1
    assert config["model_cpu_offload"] is True
    assert config["inference_timeout_ms"] == 120_000
    assert config["stop_timeout_ms"] == 120_000

    ultra_hw = _gpu(32.0)
    ultra = resolve_profile(ComputeProfile.AUTO, ultra_hw)
    ultra_config = resolve_model_runtime_config(
        MARIGOLD_MODEL_ID,
        ModelRuntimeContext(hardware_info=ultra_hw, resolved_profile=ultra),
    )
    assert ultra_config["processing_resolution"] == 768
    assert ultra_config["num_inference_steps"] == 10
    assert ultra_config["ensemble_size"] == 3
    assert ultra_config["output_uncertainty"] is True
    assert ultra_config["model_cpu_offload"] is False

    metadata = resolve_model_profile_metadata(MARIGOLD_MODEL_ID, efficient)
    assert "Fast" in metadata.variant_label
    assert "1 step" in metadata.detail_label

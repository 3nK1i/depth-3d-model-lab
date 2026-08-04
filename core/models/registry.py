"""Authoritative, lazy model registry for the multi-model platform.

The registry owns adapter construction, capabilities, dependencies, hardware
requirements, presentation metadata, and runtime configuration dispatch. It
uses dotted import paths so importing the UI never imports model packages or
initializes CUDA.
"""

from __future__ import annotations

import importlib
import importlib.util
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from core.models.base import BaseGeometryModel
from core.models.catalog import (
    DEPTH_ANYTHING_V2_DISPLAY_LABEL,
    DEPTH_ANYTHING_V2_MODEL_ID,
    DEPTH_PRO_DISPLAY_LABEL,
    DEPTH_PRO_MODEL_ID,
    MARIGOLD_DISPLAY_LABEL,
    MARIGOLD_MODEL_ID,
    MOCK_DISPLAY_LABEL,
    MOCK_MODEL_ID,
    MOGE2_DISPLAY_LABEL,
    MOGE2_MODEL_ID,
    METRIC3D_V2_DISPLAY_LABEL,
    METRIC3D_V2_MODEL_ID,
    UNIDEPTH_V2_DISPLAY_LABEL,
    UNIDEPTH_V2_MODEL_ID,
    VIDEO_DEPTH_ANYTHING_DISPLAY_LABEL,
    VIDEO_DEPTH_ANYTHING_MODEL_ID,
)
from core.models.metadata import (
    CheckpointSource,
    ConfidenceKind,
    DepthScale,
    HardwareRequirements,
    IntrinsicsSource,
    ModelCapabilities,
    ModelCheckpoint,
    ModelDependency,
    ModelLicense,
    ModelDescriptor,
    ModelOutputs,
    ModelPresentation,
    ModelProfileMetadata,
)
from core.models.runtime_config import ModelRuntimeContext, validate_runtime_config

_MODEL_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:[a-z0-9.-]*[a-z0-9])?$")


class ModelRegistryError(ValueError):
    """Base exception for invalid or unknown model registrations."""


class UnknownModelError(ModelRegistryError):
    """Raised when a model identifier is not registered."""


class DuplicateModelError(ModelRegistryError):
    """Raised when the same model identifier is registered twice."""


class AdapterContractError(TypeError):
    """Raised when a registry adapter path does not construct the common ABC."""


@dataclass(frozen=True, slots=True)
class ModelAvailability:
    """Dependency and hardware availability snapshot for one descriptor."""

    available: bool
    missing_dependencies: tuple[str, ...] = ()
    missing_assets: tuple[str, ...] = ()
    checkpoint_status: str = "not_required"
    hardware_supported: bool = True
    reason: str = ""


class ModelRegistry:
    """Validated registry with deterministic ordering and lazy imports."""

    def __init__(self) -> None:
        self._descriptors: dict[str, ModelDescriptor] = {}

    def register(self, descriptor: ModelDescriptor) -> None:
        _validate_descriptor(descriptor)
        if descriptor.model_id in self._descriptors:
            raise DuplicateModelError(
                f"Model id {descriptor.model_id!r} is already registered."
            )
        if descriptor.default and any(item.default for item in self._descriptors.values()):
            raise ModelRegistryError("Only one model may be marked as the default.")
        self._descriptors[descriptor.model_id] = descriptor

    def get(self, model_id: str) -> ModelDescriptor:
        try:
            return self._descriptors[str(model_id)]
        except KeyError:
            raise UnknownModelError(f"Unknown model id: {model_id!r}") from None

    def descriptors(
        self,
        *,
        selectable_only: bool = False,
        enabled_only: bool = False,
    ) -> tuple[ModelDescriptor, ...]:
        values: Iterable[ModelDescriptor] = self._descriptors.values()
        if selectable_only:
            values = (item for item in values if item.selectable)
        if enabled_only:
            values = (item for item in values if item.enabled)
        return tuple(values)

    def ids(self) -> tuple[str, ...]:
        return tuple(self._descriptors)

    def default(self) -> ModelDescriptor:
        defaults = [item for item in self._descriptors.values() if item.default]
        if len(defaults) != 1:
            raise ModelRegistryError(
                f"Registry must contain exactly one default model; found {len(defaults)}."
            )
        return defaults[0]

    def validate(self) -> None:
        if not self._descriptors:
            raise ModelRegistryError("Model registry is empty.")
        self.default()
        for descriptor in self._descriptors.values():
            _validate_descriptor(descriptor)

    def build(self, model_id: str) -> BaseGeometryModel:
        descriptor = self.get(model_id)
        adapter_class = _import_symbol(descriptor.adapter_path)
        if not isinstance(adapter_class, type):
            raise AdapterContractError(
                f"Adapter path {descriptor.adapter_path!r} did not resolve to a class."
            )
        adapter = adapter_class()
        if not isinstance(adapter, BaseGeometryModel):
            raise AdapterContractError(
                f"Adapter {descriptor.adapter_path!r} does not implement BaseGeometryModel."
            )
        return adapter

    def resolve_runtime_config(
        self,
        model_id: str,
        context: ModelRuntimeContext,
    ) -> dict[str, object]:
        descriptor = self.get(model_id)
        builder = _import_symbol(descriptor.runtime_config_builder)
        if not callable(builder):
            raise ModelRegistryError(
                f"Runtime config builder {descriptor.runtime_config_builder!r} is not callable."
            )
        config = builder(descriptor, context)
        if not isinstance(config, dict):
            raise ModelRegistryError(
                f"Runtime config builder for {model_id!r} must return a dict."
            )
        return dict(config)

    def resolve_profile_metadata(
        self,
        model_id: str,
        resolved_profile: Any,
    ) -> ModelProfileMetadata:
        descriptor = self.get(model_id)
        builder = _import_symbol(descriptor.profile_metadata_builder)
        if not callable(builder):
            raise ModelRegistryError(
                f"Profile metadata builder {descriptor.profile_metadata_builder!r} is not callable."
            )
        metadata = builder(descriptor, resolved_profile)
        if not isinstance(metadata, ModelProfileMetadata):
            raise ModelRegistryError(
                f"Profile metadata builder for {model_id!r} must return ModelProfileMetadata."
            )
        return metadata

    def availability(self, model_id: str, hardware_info: Any = None) -> ModelAvailability:
        descriptor = self.get(model_id)
        missing = tuple(
            dependency.distribution
            for dependency in descriptor.dependencies
            if not _dependency_available(dependency.import_name)
        )
        missing_assets = tuple(
            checkpoint.identifier
            for checkpoint in descriptor.checkpoints
            if checkpoint.required
            and checkpoint.source is CheckpointSource.LOCAL
            and not _resolve_local_asset(checkpoint.identifier).is_file()
        )
        if missing_assets:
            checkpoint_status = "missing"
        elif any(
            checkpoint.source is CheckpointSource.REMOTE
            for checkpoint in descriptor.checkpoints
        ):
            checkpoint_status = "managed_remote"
        elif descriptor.checkpoints:
            checkpoint_status = "available"
        else:
            checkpoint_status = "not_required"

        hardware_supported = True
        hardware_reason = ""
        if hardware_info is not None:
            backend = _hardware_backend_name(hardware_info)
            if backend and backend not in descriptor.hardware.supported_backends:
                hardware_supported = False
                hardware_reason = f"Backend {backend!r} is not supported."
            elif backend == "cuda":
                total_vram = getattr(hardware_info, "total_memory_gib", None)
                minimum = descriptor.hardware.minimum_cuda_vram_gib
                if minimum is not None and total_vram is not None and total_vram < minimum:
                    hardware_supported = False
                    hardware_reason = (
                        f"Requires at least {minimum:g} GiB CUDA VRAM; "
                        f"detected {float(total_vram):.1f} GiB."
                    )

        reasons: list[str] = []
        if missing:
            reasons.append("Missing dependencies: " + ", ".join(missing) + ".")
        if missing_assets:
            reasons.append("Missing local checkpoints: " + ", ".join(missing_assets) + ".")
        if hardware_reason:
            reasons.append(hardware_reason)
        if not descriptor.enabled:
            reasons.append("Model is disabled in the registry.")
        available = (
            descriptor.enabled
            and not missing
            and not missing_assets
            and hardware_supported
        )
        return ModelAvailability(
            available=available,
            missing_dependencies=missing,
            missing_assets=missing_assets,
            checkpoint_status=checkpoint_status,
            hardware_supported=hardware_supported,
            reason=" ".join(reasons),
        )


MODEL_REGISTRY = ModelRegistry()


def get_model_descriptor(model_id: str) -> ModelDescriptor:
    return MODEL_REGISTRY.get(model_id)


def get_default_model_descriptor() -> ModelDescriptor:
    return MODEL_REGISTRY.default()


def iter_model_descriptors(
    *,
    selectable_only: bool = False,
    enabled_only: bool = False,
) -> tuple[ModelDescriptor, ...]:
    return MODEL_REGISTRY.descriptors(
        selectable_only=selectable_only,
        enabled_only=enabled_only,
    )


def build_model(model_id: str) -> BaseGeometryModel:
    return MODEL_REGISTRY.build(model_id)


def resolve_model_runtime_config(
    model_id: str,
    context: ModelRuntimeContext,
) -> dict[str, object]:
    return MODEL_REGISTRY.resolve_runtime_config(model_id, context)


def resolve_model_profile_metadata(
    model_id: str,
    resolved_profile: Any,
) -> ModelProfileMetadata:
    return MODEL_REGISTRY.resolve_profile_metadata(model_id, resolved_profile)


def check_model_availability(model_id: str, hardware_info: Any = None) -> ModelAvailability:
    return MODEL_REGISTRY.availability(model_id, hardware_info)


def merge_model_metadata(
    descriptor: ModelDescriptor,
    runtime_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    """Keep registry contract fields authoritative while retaining telemetry."""

    runtime = dict(runtime_metadata or {})
    display_name = str(runtime.get("display_name") or descriptor.display_name)
    metadata = descriptor.metadata_with_runtime(**runtime)
    metadata["display_name"] = display_name
    return metadata


def _register_builtins() -> None:
    MODEL_REGISTRY.register(
        ModelDescriptor(
            model_id=MOGE2_MODEL_ID,
            display_name=MOGE2_DISPLAY_LABEL,
            adapter_path="core.models.adapters.moge2_adapter:Moge2GeometryModel",
            runtime_config_builder=(
                "core.models.runtime_config:build_profile_runtime_config"
            ),
            profile_metadata_builder=(
                "core.models.runtime_config:build_profile_metadata"
            ),
            capabilities=ModelCapabilities(
                outputs=ModelOutputs(
                    depth=True,
                    points=True,
                    normals=True,
                    intrinsics=True,
                    valid_mask=True,
                ),
                depth_scale=DepthScale.METRIC,
                intrinsics_source=IntrinsicsSource.MODEL_PREDICTED,
                intrinsics_fallback=IntrinsicsSource.ESTIMATED,
                confidence_kind=ConfidenceKind.NONE,
                source_types=("image", "video", "camera", "rtsp"),
                supports_live=True,
            ),
            dependencies=(
                ModelDependency(
                    import_name="torch",
                    distribution="torch",
                    requirement_file="requirements/models-moge2.txt",
                    purpose="Tensor inference runtime",
                ),
                ModelDependency(
                    import_name="moge",
                    distribution="moge",
                    requirement_file="requirements/models-moge2.txt",
                    purpose="Official MoGe-2 implementation",
                ),
            ),
            hardware=HardwareRequirements(
                supported_backends=("cpu", "cuda"),
                minimum_cuda_vram_gib=4.0,
                recommended_cuda_vram_gib=6.0,
                supported_precisions=("fp32", "fp16", "bf16"),
                notes="Automatic profiles adapt from RTX 2060 to RTX 5090 class GPUs.",
            ),
            presentation=ModelPresentation(
                description=(
                    "Metric monocular geometry with direct points, normals, mask, "
                    "and model-predicted intrinsics."
                ),
                header_label="MOGE-2",
                header_badge_kind="recommended",
                idle_badge="READY",
                active_badge="METRIC READY",
            ),
            license=ModelLicense(
                name="Upstream MoGe license",
                status="review_required",
                reference="https://github.com/microsoft/MoGe",
            ),
            checkpoints=(
                ModelCheckpoint("Ruicheng/moge-2-vits-normal"),
                ModelCheckpoint("Ruicheng/moge-2-vitb-normal"),
                ModelCheckpoint("Ruicheng/moge-2-vitl-normal"),
            ),
            uses_compute_profile=True,
            default=True,
        )
    )
    MODEL_REGISTRY.register(
        ModelDescriptor(
            model_id=DEPTH_PRO_MODEL_ID,
            display_name=DEPTH_PRO_DISPLAY_LABEL,
            adapter_path=(
                "core.models.adapters.depth_pro_adapter:DepthProGeometryModel"
            ),
            runtime_config_builder=(
                "core.models.runtime_config:build_depth_pro_runtime_config"
            ),
            profile_metadata_builder=(
                "core.models.runtime_config:build_depth_pro_profile_metadata"
            ),
            capabilities=ModelCapabilities(
                outputs=ModelOutputs(
                    depth=True,
                    intrinsics=True,
                    valid_mask=True,
                ),
                depth_scale=DepthScale.METRIC,
                intrinsics_source=IntrinsicsSource.MODEL_PREDICTED,
                intrinsics_fallback=IntrinsicsSource.ESTIMATED,
                confidence_kind=ConfidenceKind.NONE,
                source_types=("image", "video", "camera", "rtsp"),
                supports_live=True,
            ),
            dependencies=(
                ModelDependency(
                    import_name="torch",
                    distribution="torch",
                    requirement_file="requirements/models-depth-pro.txt",
                    purpose="Tensor inference runtime",
                ),
                ModelDependency(
                    import_name="torchvision",
                    distribution="torchvision",
                    requirement_file="requirements/models-depth-pro.txt",
                    purpose="Official preprocessing transforms",
                ),
                ModelDependency(
                    import_name="timm",
                    distribution="timm",
                    requirement_file="requirements/models-depth-pro.txt",
                    purpose="Vision Transformer components",
                ),
                ModelDependency(
                    import_name="pillow_heif",
                    distribution="pillow-heif",
                    requirement_file="requirements/models-depth-pro.txt",
                    purpose="Official image I/O dependency",
                ),
                ModelDependency(
                    import_name="matplotlib",
                    distribution="matplotlib",
                    requirement_file="requirements/models-depth-pro.txt",
                    purpose="Official package dependency",
                ),
                ModelDependency(
                    import_name="depth_pro",
                    distribution="depth-pro",
                    requirement_file="requirements/models-depth-pro.txt",
                    purpose="Official Apple Depth Pro implementation",
                ),
            ),
            hardware=HardwareRequirements(
                supported_backends=("cpu", "cuda"),
                minimum_cuda_vram_gib=6.0,
                recommended_cuda_vram_gib=8.0,
                supported_precisions=("fp32", "fp16"),
                notes=(
                    "The official architecture has a fixed 1536×1536 internal "
                    "inference size. RTX 2060 uses the conservative FP16 profile; "
                    "RTX 5090 uses the high-density profile automatically."
                ),
            ),
            presentation=ModelPresentation(
                description=(
                    "Zero-shot metric monocular depth with model-predicted focal "
                    "length and registry-managed estimated-intrinsics fallback."
                ),
                header_label="DEPTH PRO",
                header_badge_kind="compatible",
                idle_badge="READY",
                active_badge="METRIC READY",
            ),
            license=ModelLicense(
                name="Apple Depth Pro sample-code license",
                status="review_required",
                reference="https://github.com/apple/ml-depth-pro/blob/main/LICENSE",
            ),
            default_config={
                "checkpoint": "checkpoints/depth_pro.pt",
                "model_label": DEPTH_PRO_DISPLAY_LABEL,
                "internal_resolution": 1536,
                "point_budget": 250_000,
                "live_point_budget": 100_000,
                "warmup_size": 256,
                "warmup_runs": 1,
                "allow_tf32": False,
                "matmul_precision": "high",
                "hardware_tier": "efficient",
            },
            checkpoints=(
                ModelCheckpoint(
                    "checkpoints/depth_pro.pt",
                    source=CheckpointSource.LOCAL,
                ),
            ),
            uses_compute_profile=True,
        )
    )
    MODEL_REGISTRY.register(
        ModelDescriptor(
            model_id=UNIDEPTH_V2_MODEL_ID,
            display_name=UNIDEPTH_V2_DISPLAY_LABEL,
            adapter_path=(
                "core.models.adapters.unidepth_v2_adapter:UniDepthV2GeometryModel"
            ),
            runtime_config_builder=(
                "core.models.runtime_config:build_unidepth_v2_runtime_config"
            ),
            profile_metadata_builder=(
                "core.models.runtime_config:build_unidepth_v2_profile_metadata"
            ),
            capabilities=ModelCapabilities(
                outputs=ModelOutputs(
                    depth=True,
                    points=True,
                    intrinsics=True,
                    confidence=True,
                    valid_mask=True,
                ),
                depth_scale=DepthScale.METRIC,
                intrinsics_source=IntrinsicsSource.MODEL_PREDICTED,
                intrinsics_fallback=IntrinsicsSource.ESTIMATED,
                confidence_kind=ConfidenceKind.UNCERTAINTY,
                source_types=("image", "video", "camera", "rtsp"),
                supports_live=True,
            ),
            dependencies=(
                ModelDependency(
                    import_name="torch",
                    distribution="torch",
                    requirement_file="requirements/models-unidepth-v2.txt",
                    purpose="Tensor inference runtime",
                ),
                ModelDependency(
                    import_name="torchvision",
                    distribution="torchvision",
                    requirement_file="requirements/models-unidepth-v2.txt",
                    purpose="Official preprocessing operations",
                ),
                ModelDependency(
                    import_name="einops",
                    distribution="einops",
                    requirement_file="requirements/models-unidepth-v2.txt",
                    purpose="Tensor rearrangement operations",
                ),
                ModelDependency(
                    import_name="huggingface_hub",
                    distribution="huggingface-hub",
                    requirement_file="requirements/models-unidepth-v2.txt",
                    purpose="Local model snapshot management",
                ),
                ModelDependency(
                    import_name="timm",
                    distribution="timm",
                    requirement_file="requirements/models-unidepth-v2.txt",
                    purpose="Vision backbone components",
                ),
                ModelDependency(
                    import_name="unidepth",
                    distribution="unidepth",
                    requirement_file="requirements/models-unidepth-v2.txt",
                    purpose="Official UniDepthV2 implementation",
                ),
            ),
            hardware=HardwareRequirements(
                supported_backends=("cpu", "cuda"),
                minimum_cuda_vram_gib=4.0,
                recommended_cuda_vram_gib=8.0,
                supported_precisions=("fp32", "fp16"),
                notes=(
                    "Automatic profiles select ViT-S on RTX 2060, ViT-B on mid-range "
                    "GPUs, and ViT-L on RTX 5090 class hardware. The isolated V2 "
                    "installer avoids the legacy UniDepthV1 xFormers import path."
                ),
            ),
            presentation=ModelPresentation(
                description=(
                    "Universal metric monocular geometry with direct 3D points, "
                    "model-predicted intrinsics, and relative uncertainty."
                ),
                header_label="UNIDEPTH V2",
                header_badge_kind="compatible",
                idle_badge="READY",
                active_badge="METRIC + UNCERTAINTY",
            ),
            license=ModelLicense(
                name="Creative Commons Attribution-NonCommercial 4.0",
                status="noncommercial_review_required",
                reference="https://github.com/lpiccinelli-eth/UniDepth/blob/main/LICENSE",
            ),
            default_config={
                "checkpoint_dir": "checkpoints/unidepth-v2/vits14",
                "repository": "lpiccinelli/unidepth-v2-vits14",
                "variant": "vits14",
                "model_label": "UniDepthV2 ViT-S/14",
                "resolution_level": 2,
                "interpolation_mode": "bilinear",
                "point_budget": 250_000,
                "live_point_budget": 100_000,
                "warmup_size": 256,
                "warmup_runs": 1,
                "allow_tf32": False,
                "matmul_precision": "high",
                "hardware_tier": "efficient",
            },
            checkpoints=(
                ModelCheckpoint(
                    "checkpoints/unidepth-v2/.ready",
                    source=CheckpointSource.LOCAL,
                ),
                ModelCheckpoint(
                    "lpiccinelli/unidepth-v2-vits14",
                    source=CheckpointSource.REMOTE,
                    required=False,
                ),
                ModelCheckpoint(
                    "lpiccinelli/unidepth-v2-vitb14",
                    source=CheckpointSource.REMOTE,
                    required=False,
                ),
                ModelCheckpoint(
                    "lpiccinelli/unidepth-v2-vitl14",
                    source=CheckpointSource.REMOTE,
                    required=False,
                ),
            ),
            uses_compute_profile=True,
        )
    )
    MODEL_REGISTRY.register(
        ModelDescriptor(
            model_id=METRIC3D_V2_MODEL_ID,
            display_name=METRIC3D_V2_DISPLAY_LABEL,
            adapter_path=(
                "core.models.adapters.metric3d_v2_adapter:Metric3DV2GeometryModel"
            ),
            runtime_config_builder=(
                "core.models.runtime_config:build_metric3d_v2_runtime_config"
            ),
            profile_metadata_builder=(
                "core.models.runtime_config:build_metric3d_v2_profile_metadata"
            ),
            capabilities=ModelCapabilities(
                outputs=ModelOutputs(
                    depth=True,
                    points=True,
                    colors=True,
                    normals=True,
                    intrinsics=True,
                    confidence=True,
                    valid_mask=True,
                ),
                depth_scale=DepthScale.METRIC,
                intrinsics_source=IntrinsicsSource.ESTIMATED,
                confidence_kind=ConfidenceKind.CONFIDENCE,
                source_types=("image", "video", "camera", "rtsp"),
                supports_live=True,
            ),
            dependencies=(
                ModelDependency(
                    import_name="torch",
                    distribution="torch",
                    requirement_file="requirements/models-metric3d-v2.txt",
                    purpose="Tensor inference runtime",
                ),
                ModelDependency(
                    import_name="torchvision",
                    distribution="torchvision",
                    requirement_file="requirements/models-metric3d-v2.txt",
                    purpose="Official vision operations",
                ),
                ModelDependency(
                    import_name="cv2",
                    distribution="opencv-python",
                    requirement_file="requirements/models-metric3d-v2.txt",
                    purpose="Official image preprocessing",
                ),
                ModelDependency(
                    import_name="mmengine",
                    distribution="mmengine",
                    requirement_file="requirements/models-metric3d-v2.txt",
                    purpose="Official configuration loader",
                ),
                ModelDependency(
                    import_name="mmcv",
                    distribution="mmcv-lite",
                    requirement_file="requirements/models-metric3d-v2.txt",
                    purpose="OpenMMLab compatibility runtime",
                ),
                ModelDependency(
                    import_name="timm",
                    distribution="timm",
                    requirement_file="requirements/models-metric3d-v2.txt",
                    purpose="Vision Transformer components",
                ),
            ),
            hardware=HardwareRequirements(
                supported_backends=("cpu", "cuda"),
                minimum_cuda_vram_gib=5.0,
                recommended_cuda_vram_gib=8.0,
                supported_precisions=("fp32", "fp16"),
                notes=(
                    "Automatic profiles use ViT-S on RTX 2060 and ViT-L on RTX 5090. "
                    "Metric scale depends on camera intrinsics; estimated intrinsics are "
                    "used until calibration support is added."
                ),
            ),
            presentation=ModelPresentation(
                description=(
                    "Zero-shot metric depth and surface normals using estimated camera "
                    "intrinsics until calibration becomes available."
                ),
                header_label="METRIC3D V2",
                header_badge_kind="compatible",
                idle_badge="READY",
                active_badge="METRIC + NORMALS",
            ),
            license=ModelLicense(
                name="BSD 2-Clause",
                status="permissive_reviewed",
                reference="https://github.com/YvanYin/Metric3D/blob/main/LICENSE",
            ),
            default_config={
                "source_dir": "third_party/Metric3D",
                "checkpoint": (
                    "checkpoints/metric3d-v2/metric_depth_vit_small_800k.pth"
                ),
                "entrypoint": "metric3d_vit_small",
                "variant": "vit-small",
                "model_label": "Metric3D v2 ViT-S",
                "decoder_label": "RAFT-4iter",
                "input_height": 616,
                "input_width": 1064,
                "canonical_focal_length": 1000.0,
                "virtual_fov_degrees": 60.0,
                "max_depth_m": 300.0,
                "point_budget": 250_000,
                "live_point_budget": 100_000,
                "warmup_size": 256,
                "warmup_runs": 1,
                "allow_tf32": False,
                "matmul_precision": "high",
                "hardware_tier": "efficient",
            },
            checkpoints=(
                ModelCheckpoint(
                    "third_party/Metric3D/.eia_metric3d_v2_ready.json",
                    source=CheckpointSource.LOCAL,
                ),
                ModelCheckpoint(
                    "checkpoints/metric3d-v2/.ready",
                    source=CheckpointSource.LOCAL,
                ),
                ModelCheckpoint(
                    "JUGGHM/Metric3D:metric_depth_vit_small_800k.pth",
                    source=CheckpointSource.REMOTE,
                    required=False,
                ),
                ModelCheckpoint(
                    "JUGGHM/Metric3D:metric_depth_vit_large_800k.pth",
                    source=CheckpointSource.REMOTE,
                    required=False,
                ),
            ),
            uses_compute_profile=True,
        )
    )
    MODEL_REGISTRY.register(
        ModelDescriptor(
            model_id=DEPTH_ANYTHING_V2_MODEL_ID,
            display_name=DEPTH_ANYTHING_V2_DISPLAY_LABEL,
            adapter_path=(
                "core.models.adapters.depth_anything_v2_adapter:DepthAnythingV2GeometryModel"
            ),
            runtime_config_builder=(
                "core.models.runtime_config:build_depth_anything_v2_runtime_config"
            ),
            profile_metadata_builder=(
                "core.models.runtime_config:build_depth_anything_v2_profile_metadata"
            ),
            capabilities=ModelCapabilities(
                outputs=ModelOutputs(
                    depth=True,
                    points=True,
                    colors=True,
                    intrinsics=True,
                    valid_mask=True,
                ),
                depth_scale=DepthScale.RELATIVE,
                intrinsics_source=IntrinsicsSource.ESTIMATED,
                confidence_kind=ConfidenceKind.NONE,
                source_types=("image", "video", "camera", "rtsp"),
                supports_live=True,
            ),
            dependencies=(
                ModelDependency(
                    import_name="torch",
                    distribution="torch",
                    requirement_file="requirements/models-depth-anything-v2.txt",
                    purpose="Tensor inference runtime",
                ),
                ModelDependency(
                    import_name="torchvision",
                    distribution="torchvision",
                    requirement_file="requirements/models-depth-anything-v2.txt",
                    purpose="Official DINOv2/DPT operations",
                ),
                ModelDependency(
                    import_name="cv2",
                    distribution="opencv-python",
                    requirement_file="requirements/models-depth-anything-v2.txt",
                    purpose="Official image preprocessing",
                ),
            ),
            hardware=HardwareRequirements(
                supported_backends=("cpu", "cuda"),
                minimum_cuda_vram_gib=2.0,
                recommended_cuda_vram_gib=6.0,
                supported_precisions=("fp32", "fp16"),
                notes=(
                    "Automatic profiles use ViT-S on RTX 2060, ViT-B on mid-range "
                    "GPUs, and ViT-L on RTX 5090. Output scale is relative."
                ),
            ),
            presentation=ModelPresentation(
                description=(
                    "Robust frame-based relative depth. Point-cloud scale is arbitrary "
                    "and must not be used for metric measurements."
                ),
                header_label="DEPTH ANYTHING V2",
                header_badge_kind="compatible",
                idle_badge="READY",
                active_badge="RELATIVE READY",
            ),
            license=ModelLicense(
                name="Mixed: Small Apache-2.0; Base/Large CC BY-NC 4.0",
                status="variant_dependent_review_required",
                reference="https://github.com/DepthAnything/Depth-Anything-V2#license",
            ),
            default_config={
                "source_dir": "third_party/Depth-Anything-V2",
                "checkpoint": "checkpoints/depth-anything-v2/depth_anything_v2_vits.pth",
                "encoder": "vits",
                "model_label": "Depth Anything V2 Small",
                "input_size": 518,
                "virtual_fov_degrees": 60.0,
                "point_budget": 250_000,
                "live_point_budget": 100_000,
                "warmup_size": 256,
                "warmup_runs": 1,
                "allow_tf32": False,
                "matmul_precision": "high",
                "hardware_tier": "efficient",
            },
            checkpoints=(
                ModelCheckpoint(
                    "third_party/Depth-Anything-V2/.eia_depth_anything_v2_ready.json",
                    source=CheckpointSource.LOCAL,
                ),
                ModelCheckpoint(
                    "checkpoints/depth-anything-v2/.ready",
                    source=CheckpointSource.LOCAL,
                ),
                ModelCheckpoint(
                    "depth-anything/Depth-Anything-V2-Small:depth_anything_v2_vits.pth",
                    source=CheckpointSource.REMOTE,
                    required=False,
                ),
                ModelCheckpoint(
                    "depth-anything/Depth-Anything-V2-Base:depth_anything_v2_vitb.pth",
                    source=CheckpointSource.REMOTE,
                    required=False,
                ),
                ModelCheckpoint(
                    "depth-anything/Depth-Anything-V2-Large:depth_anything_v2_vitl.pth",
                    source=CheckpointSource.REMOTE,
                    required=False,
                ),
            ),
            uses_compute_profile=True,
        )
    )
    MODEL_REGISTRY.register(
        ModelDescriptor(
            model_id=VIDEO_DEPTH_ANYTHING_MODEL_ID,
            display_name=VIDEO_DEPTH_ANYTHING_DISPLAY_LABEL,
            adapter_path=(
                "core.models.adapters.video_depth_anything_adapter:VideoDepthAnythingGeometryModel"
            ),
            runtime_config_builder=(
                "core.models.runtime_config:build_video_depth_anything_runtime_config"
            ),
            profile_metadata_builder=(
                "core.models.runtime_config:build_video_depth_anything_profile_metadata"
            ),
            capabilities=ModelCapabilities(
                outputs=ModelOutputs(
                    depth=True,
                    points=True,
                    colors=True,
                    intrinsics=True,
                    valid_mask=True,
                ),
                depth_scale=DepthScale.METRIC,
                intrinsics_source=IntrinsicsSource.ESTIMATED,
                confidence_kind=ConfidenceKind.NONE,
                source_types=("image", "video", "camera", "rtsp"),
                supports_live=True,
                stateful_sequence=True,
            ),
            dependencies=(
                ModelDependency(
                    import_name="torch",
                    distribution="torch",
                    requirement_file="requirements/models-video-depth-anything.txt",
                    purpose="Tensor inference runtime",
                ),
                ModelDependency(
                    import_name="torchvision",
                    distribution="torchvision",
                    requirement_file="requirements/models-video-depth-anything.txt",
                    purpose="Official temporal DPT operations",
                ),
                ModelDependency(
                    import_name="cv2",
                    distribution="opencv-python",
                    requirement_file="requirements/models-video-depth-anything.txt",
                    purpose="Streaming frame preprocessing",
                ),
                ModelDependency(
                    import_name="einops",
                    distribution="einops",
                    requirement_file="requirements/models-video-depth-anything.txt",
                    purpose="Temporal tensor rearrangement",
                ),
            ),
            hardware=HardwareRequirements(
                supported_backends=("cpu", "cuda"),
                minimum_cuda_vram_gib=5.0,
                recommended_cuda_vram_gib=8.0,
                supported_precisions=("fp32", "fp16"),
                notes=(
                    "Official streaming temporal caches are stateful. RTX 2060 uses "
                    "the metric ViT-S checkpoint at 392 px; RTX 5090 uses metric ViT-L."
                ),
            ),
            presentation=ModelPresentation(
                description=(
                    "Official experimental streaming path for temporally consistent "
                    "metric depth. Estimated camera intrinsics are used until "
                    "calibration is available."
                ),
                header_label="VIDEO DEPTH ANYTHING",
                header_badge_kind="compatible",
                idle_badge="READY",
                active_badge="TEMPORAL METRIC",
            ),
            license=ModelLicense(
                name="Mixed: Small Apache-2.0; Base/Large CC BY-NC 4.0",
                status="variant_dependent_review_required",
                reference="https://github.com/DepthAnything/Video-Depth-Anything#license",
            ),
            default_config={
                "source_dir": "third_party/Video-Depth-Anything",
                "checkpoint": "checkpoints/video-depth-anything/metric_video_depth_anything_vits.pth",
                "encoder": "vits",
                "model_label": "Metric Video Depth Anything Small",
                "input_size": 392,
                "max_resolution": 960,
                "virtual_fov_degrees": 60.0,
                "max_depth_m": 300.0,
                "point_budget": 100_000,
                "live_point_budget": 100_000,
                "warmup_size": 224,
                "warmup_runs": 1,
                "allow_tf32": False,
                "matmul_precision": "high",
                "hardware_tier": "efficient",
            },
            checkpoints=(
                ModelCheckpoint(
                    "third_party/Video-Depth-Anything/.eia_video_depth_anything_ready.json",
                    source=CheckpointSource.LOCAL,
                ),
                ModelCheckpoint(
                    "checkpoints/video-depth-anything/.ready",
                    source=CheckpointSource.LOCAL,
                ),
                ModelCheckpoint(
                    "depth-anything/Metric-Video-Depth-Anything-Small:metric_video_depth_anything_vits.pth",
                    source=CheckpointSource.REMOTE,
                    required=False,
                ),
                ModelCheckpoint(
                    "depth-anything/Metric-Video-Depth-Anything-Base:metric_video_depth_anything_vitb.pth",
                    source=CheckpointSource.REMOTE,
                    required=False,
                ),
                ModelCheckpoint(
                    "depth-anything/Metric-Video-Depth-Anything-Large:metric_video_depth_anything_vitl.pth",
                    source=CheckpointSource.REMOTE,
                    required=False,
                ),
            ),
            uses_compute_profile=True,
        )
    )
    MODEL_REGISTRY.register(
        ModelDescriptor(
            model_id=MARIGOLD_MODEL_ID,
            display_name=MARIGOLD_DISPLAY_LABEL,
            adapter_path=(
                "core.models.adapters.marigold_adapter:MarigoldGeometryModel"
            ),
            runtime_config_builder=(
                "core.models.runtime_config:build_marigold_runtime_config"
            ),
            profile_metadata_builder=(
                "core.models.runtime_config:build_marigold_profile_metadata"
            ),
            capabilities=ModelCapabilities(
                outputs=ModelOutputs(
                    depth=True,
                    points=True,
                    colors=True,
                    intrinsics=True,
                    confidence=True,
                    valid_mask=True,
                ),
                depth_scale=DepthScale.RELATIVE,
                intrinsics_source=IntrinsicsSource.ESTIMATED,
                confidence_kind=ConfidenceKind.UNCERTAINTY,
                source_types=("image", "video", "camera", "rtsp"),
                supports_live=False,
            ),
            dependencies=(
                ModelDependency(
                    import_name="torch",
                    distribution="torch",
                    requirement_file="requirements/models-marigold.txt",
                    purpose="Tensor inference runtime",
                ),
                ModelDependency(
                    import_name="diffusers",
                    distribution="diffusers",
                    requirement_file="requirements/models-marigold.txt",
                    purpose="Official MarigoldDepthPipeline implementation",
                ),
                ModelDependency(
                    import_name="transformers",
                    distribution="transformers",
                    requirement_file="requirements/models-marigold.txt",
                    purpose="Stable Diffusion text encoder runtime",
                ),
                ModelDependency(
                    import_name="accelerate",
                    distribution="accelerate",
                    requirement_file="requirements/models-marigold.txt",
                    purpose="Memory-safe model CPU offload on 6 GB GPUs",
                ),
                ModelDependency(
                    import_name="huggingface_hub",
                    distribution="huggingface-hub",
                    requirement_file="requirements/models-marigold.txt",
                    purpose="Pinned local snapshot management",
                ),
                ModelDependency(
                    import_name="PIL",
                    distribution="Pillow",
                    requirement_file="requirements/models-marigold.txt",
                    purpose="Image conversion for the official pipeline",
                ),
            ),
            hardware=HardwareRequirements(
                supported_backends=("cpu", "cuda"),
                minimum_cuda_vram_gib=5.5,
                recommended_cuda_vram_gib=12.0,
                supported_precisions=("fp32", "fp16"),
                notes=(
                    "Diffusion-based, quality-oriented inference. RTX 2060 uses "
                    "FP16 model CPU offload and a one-step fast preset; RTX 5090 "
                    "uses a higher-resolution precision preset with uncertainty."
                ),
            ),
            presentation=ModelPresentation(
                description=(
                    "High-quality affine-invariant diffusion depth for snapshots and "
                    "offline inspection. It is not intended as a real-time model."
                ),
                header_label="MARIGOLD",
                header_badge_kind="experimental",
                idle_badge="QUALITY",
                active_badge="RELATIVE READY",
            ),
            license=ModelLicense(
                name="Code Apache-2.0; model CreativeML Open RAIL++-M",
                status="review_required",
                reference="https://huggingface.co/prs-eth/marigold-depth-v1-1",
            ),
            default_config={
                "repository": "prs-eth/marigold-depth-v1-1",
                "checkpoint_dir": "checkpoints/marigold/depth-v1-1",
                "model_label": "Marigold Depth v1.1",
                "weight_variant": "fp16",
                "prediction_type": "depth",
                "processing_resolution": 512,
                "num_inference_steps": 1,
                "ensemble_size": 1,
                "output_uncertainty": False,
                "model_cpu_offload": True,
                "attention_slicing": True,
                "vae_slicing": True,
                "vae_tiling": True,
                "seed": 0,
                "virtual_fov_degrees": 60.0,
                "point_budget": 100_000,
                "live_point_budget": 50_000,
                "warmup_size": 192,
                "warmup_runs": 1,
                "warmup_inference_steps": 1,
                "allow_tf32": False,
                "matmul_precision": "high",
                "hardware_tier": "efficient",
                "inference_timeout_ms": 120_000,
                "stop_timeout_ms": 120_000,
            },
            checkpoints=(
                ModelCheckpoint(
                    "checkpoints/marigold/depth-v1-1/.ready",
                    source=CheckpointSource.LOCAL,
                ),
                ModelCheckpoint(
                    "checkpoints/marigold/depth-v1-1/model_index.json",
                    source=CheckpointSource.LOCAL,
                ),
                ModelCheckpoint(
                    "prs-eth/marigold-depth-v1-1",
                    source=CheckpointSource.REMOTE,
                    required=False,
                ),
            ),
            uses_compute_profile=True,
        )
    )
    MODEL_REGISTRY.register(
        ModelDescriptor(
            model_id=MOCK_MODEL_ID,
            display_name="Development Mock",
            adapter_path="core.models.adapters.mock_adapter:MockGeometryModel",
            runtime_config_builder=(
                "core.models.runtime_config:build_static_runtime_config"
            ),
            profile_metadata_builder=(
                "core.models.runtime_config:build_mock_profile_metadata"
            ),
            capabilities=ModelCapabilities(
                outputs=ModelOutputs(depth=True),
                depth_scale=DepthScale.SYNTHETIC,
                intrinsics_source=IntrinsicsSource.NONE,
                source_types=("image", "video", "camera", "rtsp"),
                supports_live=True,
            ),
            hardware=HardwareRequirements(
                supported_backends=("cpu", "cuda"),
                supported_precisions=("fp32",),
                notes="Synthetic diagnostic adapter; no AI model is executed.",
            ),
            presentation=ModelPresentation(
                description=MOCK_DISPLAY_LABEL,
                header_label="MOCK",
                header_badge_kind="experimental",
                idle_badge="DEV ONLY",
                active_badge="MOCK ACTIVE",
            ),
            license=ModelLicense(
                name="EIA TECH internal development utility",
                status="internal",
            ),
            default_config={
                "device": "cpu",
                "precision": "fp32",
                "latency_ms": 30.0,
            },
            development_only=True,
        )
    )
    MODEL_REGISTRY.validate()


def _validate_descriptor(descriptor: ModelDescriptor) -> None:
    if not _MODEL_ID_PATTERN.fullmatch(descriptor.model_id):
        raise ModelRegistryError(
            f"Invalid model id {descriptor.model_id!r}; use lowercase registry-safe ids."
        )
    if not descriptor.display_name.strip():
        raise ModelRegistryError(f"Model {descriptor.model_id!r} has no display name.")
    _validate_symbol_path(descriptor.adapter_path, field_name="adapter_path")
    _validate_symbol_path(
        descriptor.runtime_config_builder,
        field_name="runtime_config_builder",
    )
    _validate_symbol_path(
        descriptor.profile_metadata_builder,
        field_name="profile_metadata_builder",
    )
    outputs = descriptor.capabilities.outputs
    if not outputs.depth:
        raise ModelRegistryError(
            f"Model {descriptor.model_id!r} must provide normalized depth output."
        )
    if (
        descriptor.capabilities.intrinsics_source is not IntrinsicsSource.NONE
        and not outputs.intrinsics
    ):
        raise ModelRegistryError(
            f"Model {descriptor.model_id!r} declares intrinsics provenance but no intrinsics output."
        )
    if outputs.confidence and descriptor.capabilities.confidence_kind is ConfidenceKind.NONE:
        raise ModelRegistryError(
            f"Model {descriptor.model_id!r} exposes confidence without defining its semantics."
        )
    if not outputs.confidence and descriptor.capabilities.confidence_kind is not ConfidenceKind.NONE:
        raise ModelRegistryError(
            f"Model {descriptor.model_id!r} defines confidence semantics without an output."
        )
    sources = tuple(str(value).lower() for value in descriptor.capabilities.source_types)
    valid_sources = {"image", "video", "camera", "rtsp"}
    if not sources:
        raise ModelRegistryError(
            f"Model {descriptor.model_id!r} must support at least one source type."
        )
    if len(set(sources)) != len(sources):
        raise ModelRegistryError(
            f"Model {descriptor.model_id!r} contains duplicate source types."
        )
    unknown_sources = set(sources) - valid_sources
    if unknown_sources:
        raise ModelRegistryError(
            f"Model {descriptor.model_id!r} has unsupported source types: "
            f"{sorted(unknown_sources)}."
        )
    if outputs.intrinsics and descriptor.capabilities.intrinsics_source is IntrinsicsSource.NONE:
        raise ModelRegistryError(
            f"Model {descriptor.model_id!r} exposes intrinsics without defining their provenance."
        )

    backends = tuple(str(value).lower() for value in descriptor.hardware.supported_backends)
    if not backends or set(backends) - {"cpu", "cuda"}:
        raise ModelRegistryError(
            f"Model {descriptor.model_id!r} has invalid hardware backends: {backends!r}."
        )
    minimum = descriptor.hardware.minimum_cuda_vram_gib
    recommended = descriptor.hardware.recommended_cuda_vram_gib
    if minimum is not None and minimum < 0:
        raise ModelRegistryError("Minimum CUDA VRAM cannot be negative.")
    if recommended is not None and recommended < 0:
        raise ModelRegistryError("Recommended CUDA VRAM cannot be negative.")
    if minimum is not None and recommended is not None and recommended < minimum:
        raise ModelRegistryError(
            f"Model {descriptor.model_id!r} recommends less VRAM than its minimum."
        )
    if not descriptor.hardware.supported_precisions:
        raise ModelRegistryError(
            f"Model {descriptor.model_id!r} must declare at least one precision."
        )

    dependency_names = [item.import_name for item in descriptor.dependencies]
    if any(not name.strip() for name in dependency_names):
        raise ModelRegistryError(
            f"Model {descriptor.model_id!r} contains an empty dependency import name."
        )
    if len(set(dependency_names)) != len(dependency_names):
        raise ModelRegistryError(
            f"Model {descriptor.model_id!r} contains duplicate dependencies."
        )
    checkpoint_ids = [item.identifier for item in descriptor.checkpoints]
    if any(not identifier.strip() for identifier in checkpoint_ids):
        raise ModelRegistryError(
            f"Model {descriptor.model_id!r} contains an empty checkpoint identifier."
        )
    if len(set(checkpoint_ids)) != len(checkpoint_ids):
        raise ModelRegistryError(
            f"Model {descriptor.model_id!r} contains duplicate checkpoints."
        )
    try:
        validate_runtime_config(descriptor.default_config)
    except TypeError as exc:
        raise ModelRegistryError(
            f"Model {descriptor.model_id!r} has an invalid default runtime config: {exc}"
        ) from exc


def _validate_symbol_path(path: str, *, field_name: str) -> None:
    module_name, separator, symbol_name = str(path).partition(":")
    if not separator or not module_name or not symbol_name:
        raise ModelRegistryError(
            f"{field_name} must use the 'package.module:Symbol' format; got {path!r}."
        )


def _import_symbol(path: str) -> Any:
    module_name, _, symbol_name = path.partition(":")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, symbol_name)
    except AttributeError:
        raise ModelRegistryError(
            f"Registry symbol {symbol_name!r} was not found in module {module_name!r}."
        ) from None


def _dependency_available(import_name: str) -> bool:
    try:
        return importlib.util.find_spec(import_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _resolve_local_asset(identifier: str) -> Path:
    """Resolve registry-local assets independently of the process CWD."""

    path = Path(identifier).expanduser()
    if path.is_absolute():
        return path
    project_root = Path(__file__).resolve().parents[2]
    return project_root / path


def _hardware_backend_name(hardware_info: Any) -> str:
    backend = getattr(hardware_info, "backend", "")
    value = getattr(backend, "value", backend)
    return str(value).lower()


_register_builtins()

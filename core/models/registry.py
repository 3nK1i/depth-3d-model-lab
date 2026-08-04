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
    MOCK_DISPLAY_LABEL,
    MOCK_MODEL_ID,
    MOGE2_DISPLAY_LABEL,
    MOGE2_MODEL_ID,
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
            and not Path(checkpoint.identifier).expanduser().is_file()
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


def _hardware_backend_name(hardware_info: Any) -> str:
    backend = getattr(hardware_info, "backend", "")
    value = getattr(backend, "value", backend)
    return str(value).lower()


_register_builtins()

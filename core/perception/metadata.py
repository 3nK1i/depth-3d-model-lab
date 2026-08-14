"""Dependency-light perception registry metadata and resource declarations."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from core.contracts.validation import (
    ContractValidationError,
    JsonValue,
    metadata_as_dict,
    normalize_metadata,
    require_nonempty,
)
from core.models.metadata import ModelDependency


class PerceptionTask(str, Enum):
    DETECTION = "detection"
    INSTANCE_SEGMENTATION = "instance_segmentation"
    PROMPTED_SEGMENTATION = "prompted_segmentation"
    AUTOMATIC_SEGMENTATION = "automatic_segmentation"
    TRACKING = "tracking"


class AdapterCapability(str, Enum):
    FRAME_PERCEPTION = "frame_perception"
    PROMPTED_SEGMENTATION = "prompted_segmentation"
    AUTOMATIC_SEGMENTATION = "automatic_segmentation"
    SEQUENCE_PERCEPTION = "sequence_perception"


class PreferredDevice(str, Enum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"


class CoResidencyPolicy(str, Enum):
    ALLOW = "allow"
    PREFER_ISOLATED = "prefer_isolated"
    EXCLUSIVE = "exclusive"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class PerceptionResourceMetadata:
    """Advisory metadata for future model-agnostic GPU scheduling."""

    estimated_vram_mb: int | None = None
    minimum_vram_mb: int | None = None
    preferred_device: PreferredDevice = PreferredDevice.AUTO
    cpu_fallback_supported: bool = False
    co_residency_policy: CoResidencyPolicy = CoResidencyPolicy.UNKNOWN
    exclusive_gpu_required: bool = False
    supported_precisions: tuple[str, ...] = ("fp32",)
    supported_hardware_profiles: tuple[str, ...] = ("rtx2060", "rtx5090")
    notes: str = ""

    def __post_init__(self) -> None:
        for field_name in ("estimated_vram_mb", "minimum_vram_mb"):
            value = getattr(self, field_name)
            if value is not None:
                if isinstance(value, bool) or int(value) <= 0:
                    raise ContractValidationError(f"{field_name} must be a positive integer.")
                object.__setattr__(self, field_name, int(value))
        if (
            self.estimated_vram_mb is not None
            and self.minimum_vram_mb is not None
            and self.estimated_vram_mb < self.minimum_vram_mb
        ):
            raise ContractValidationError(
                "estimated_vram_mb must be greater than or equal to minimum_vram_mb."
            )
        precisions = tuple(
            require_nonempty(value, field_name="supported precision").lower()
            for value in self.supported_precisions
        )
        profiles = tuple(
            require_nonempty(value, field_name="supported hardware profile").lower()
            for value in self.supported_hardware_profiles
        )
        if not precisions:
            raise ContractValidationError("supported_precisions must not be empty.")
        if not profiles:
            raise ContractValidationError("supported_hardware_profiles must not be empty.")
        if len(set(precisions)) != len(precisions):
            raise ContractValidationError("supported_precisions must be unique.")
        if len(set(profiles)) != len(profiles):
            raise ContractValidationError("supported_hardware_profiles must be unique.")
        if self.exclusive_gpu_required and self.co_residency_policy is not CoResidencyPolicy.EXCLUSIVE:
            raise ContractValidationError(
                "exclusive_gpu_required requires co_residency_policy=EXCLUSIVE."
            )
        if (
            self.co_residency_policy is CoResidencyPolicy.EXCLUSIVE
            and not self.exclusive_gpu_required
        ):
            raise ContractValidationError(
                "EXCLUSIVE co-residency requires exclusive_gpu_required=True."
            )
        object.__setattr__(self, "supported_precisions", precisions)
        object.__setattr__(self, "supported_hardware_profiles", profiles)
        object.__setattr__(self, "notes", str(self.notes).strip())

    def as_dict(self) -> dict[str, Any]:
        return {
            "estimated_vram_mb": self.estimated_vram_mb,
            "minimum_vram_mb": self.minimum_vram_mb,
            "preferred_device": self.preferred_device.value,
            "cpu_fallback_supported": self.cpu_fallback_supported,
            "co_residency_policy": self.co_residency_policy.value,
            "exclusive_gpu_required": self.exclusive_gpu_required,
            "supported_precisions": list(self.supported_precisions),
            "supported_hardware_profiles": list(self.supported_hardware_profiles),
            "notes": self.notes,
        }


@dataclass(frozen=True, slots=True)
class PerceptionCapabilities:
    tasks: tuple[PerceptionTask, ...]
    adapter_capabilities: tuple[AdapterCapability, ...]
    source_types: tuple[str, ...] = ("image",)
    supports_live: bool = False
    stateful_sequence: bool = False

    def __post_init__(self) -> None:
        if not self.tasks:
            raise ContractValidationError("Perception tasks must not be empty.")
        if not self.adapter_capabilities:
            raise ContractValidationError("Adapter capabilities must not be empty.")
        if len(set(self.tasks)) != len(self.tasks):
            raise ContractValidationError("Perception tasks must be unique.")
        if len(set(self.adapter_capabilities)) != len(self.adapter_capabilities):
            raise ContractValidationError("Adapter capabilities must be unique.")
        sources = tuple(
            require_nonempty(value, field_name="source type").lower()
            for value in self.source_types
        )
        if not sources:
            raise ContractValidationError("source_types must not be empty.")
        if self.stateful_sequence and AdapterCapability.SEQUENCE_PERCEPTION not in self.adapter_capabilities:
            raise ContractValidationError(
                "stateful_sequence requires the SEQUENCE_PERCEPTION adapter capability."
            )
        if (
            PerceptionTask.PROMPTED_SEGMENTATION in self.tasks
            and AdapterCapability.PROMPTED_SEGMENTATION not in self.adapter_capabilities
        ):
            raise ContractValidationError(
                "Prompted segmentation tasks require the prompted adapter capability."
            )
        if (
            PerceptionTask.AUTOMATIC_SEGMENTATION in self.tasks
            and AdapterCapability.AUTOMATIC_SEGMENTATION not in self.adapter_capabilities
        ):
            raise ContractValidationError(
                "Automatic segmentation tasks require the automatic segmentation adapter capability."
            )
        frame_tasks = {
            PerceptionTask.DETECTION,
            PerceptionTask.INSTANCE_SEGMENTATION,
            PerceptionTask.TRACKING,
        }
        if frame_tasks.intersection(self.tasks) and AdapterCapability.FRAME_PERCEPTION not in self.adapter_capabilities:
            raise ContractValidationError(
                "Frame perception tasks require the FRAME_PERCEPTION adapter capability."
            )
        object.__setattr__(self, "source_types", sources)

    def as_dict(self) -> dict[str, Any]:
        return {
            "tasks": [value.value for value in self.tasks],
            "adapter_capabilities": [value.value for value in self.adapter_capabilities],
            "source_types": list(self.source_types),
            "supports_live": self.supports_live,
            "stateful_sequence": self.stateful_sequence,
        }


@dataclass(frozen=True, slots=True)
class PerceptionDescriptor:
    model_id: str
    display_name: str
    adapter_path: str
    capabilities: PerceptionCapabilities
    resources: PerceptionResourceMetadata = field(default_factory=PerceptionResourceMetadata)
    dependencies: tuple[ModelDependency, ...] = ()
    default_config: Mapping[str, JsonValue] | tuple[tuple[str, JsonValue], ...] = field(
        default_factory=tuple
    )
    enabled: bool = True
    selectable: bool = True
    development_only: bool = False
    license_name: str = ""
    license_scope: str = ""
    checkpoints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", require_nonempty(self.model_id, field_name="model_id"))
        object.__setattr__(
            self, "display_name", require_nonempty(self.display_name, field_name="display_name")
        )
        adapter_path = require_nonempty(self.adapter_path, field_name="adapter_path")
        if ":" not in adapter_path:
            raise ContractValidationError(
                "adapter_path must use the 'module.path:Symbol' format."
            )
        object.__setattr__(self, "adapter_path", adapter_path)
        object.__setattr__(self, "default_config", normalize_metadata(self.default_config))
        object.__setattr__(self, "license_name", str(self.license_name).strip())
        object.__setattr__(self, "license_scope", str(self.license_scope).strip())
        checkpoints = tuple(require_nonempty(value, field_name="checkpoint") for value in self.checkpoints)
        if len(checkpoints) != len(set(checkpoints)):
            raise ContractValidationError("checkpoints must be unique.")
        object.__setattr__(self, "checkpoints", checkpoints)

    def as_metadata(self) -> dict[str, Any]:
        return {
            "id": self.model_id,
            "display_name": self.display_name,
            "adapter_path": self.adapter_path,
            "capabilities": self.capabilities.as_dict(),
            "resources": self.resources.as_dict(),
            "dependencies": [dependency.as_dict() for dependency in self.dependencies],
            "default_config": metadata_as_dict(self.default_config),  # type: ignore[arg-type]
            "enabled": self.enabled,
            "selectable": self.selectable,
            "development_only": self.development_only,
            "license_name": self.license_name,
            "license_scope": self.license_scope,
            "checkpoints": list(self.checkpoints),
        }

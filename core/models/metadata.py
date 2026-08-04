"""Lightweight model metadata shared by the UI and inference worker.

The classes in this module intentionally depend only on the Python standard
library. They describe what a model can produce, how it is presented, and
which optional dependencies or hardware it requires without importing the
actual model package or PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class DepthScale(str, Enum):
    """Scale semantics of a model's depth output."""

    METRIC = "metric"
    RELATIVE = "relative"
    SYNTHETIC = "synthetic"
    UNKNOWN = "unknown"


class IntrinsicsSource(str, Enum):
    """Origin of the camera intrinsics associated with a result."""

    MODEL_PREDICTED = "model_predicted"
    ESTIMATED = "estimated"
    CALIBRATED = "calibrated"
    NONE = "none"


class ConfidenceKind(str, Enum):
    """Meaning of the optional confidence-like output."""

    NONE = "none"
    CONFIDENCE = "confidence"
    UNCERTAINTY = "uncertainty"


class CheckpointSource(str, Enum):
    """How a registered checkpoint is expected to be resolved."""

    REMOTE = "remote"
    LOCAL = "local"


@dataclass(frozen=True, slots=True)
class ModelOutputs:
    """Normalized output fields the worker may return to the application."""

    depth: bool = True
    points: bool = False
    colors: bool = False
    normals: bool = False
    intrinsics: bool = False
    confidence: bool = False
    valid_mask: bool = False

    def as_dict(self) -> dict[str, bool]:
        return {
            "depth": self.depth,
            "points": self.points,
            "colors": self.colors,
            "normals": self.normals,
            "intrinsics": self.intrinsics,
            "confidence": self.confidence,
            "valid_mask": self.valid_mask,
        }


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    """Stable capability contract advertised by one registered model."""

    outputs: ModelOutputs = field(default_factory=ModelOutputs)
    depth_scale: DepthScale = DepthScale.UNKNOWN
    intrinsics_source: IntrinsicsSource = IntrinsicsSource.NONE
    intrinsics_fallback: IntrinsicsSource = IntrinsicsSource.NONE
    confidence_kind: ConfidenceKind = ConfidenceKind.NONE
    source_types: tuple[str, ...] = ("image",)
    supports_live: bool = False
    stateful_sequence: bool = False

    @property
    def is_metric(self) -> bool:
        return self.depth_scale is DepthScale.METRIC

    def as_dict(self) -> dict[str, Any]:
        sources = tuple(str(value).lower() for value in self.source_types)
        return {
            "outputs": self.outputs.as_dict(),
            "depth_scale": self.depth_scale.value,
            "is_metric": self.is_metric,
            "intrinsics_source": self.intrinsics_source.value,
            "intrinsics_fallback": self.intrinsics_fallback.value,
            "confidence_kind": self.confidence_kind.value,
            "source_types": list(sources),
            "supports_image": "image" in sources,
            "supports_video": "video" in sources,
            "supports_camera": "camera" in sources,
            "supports_rtsp": "rtsp" in sources,
            "supports_live": self.supports_live,
            "stateful_sequence": self.stateful_sequence,
        }


@dataclass(frozen=True, slots=True)
class ModelLicense:
    """License-review metadata without making unverified legal claims."""

    name: str = ""
    status: str = "unverified"
    reference: str = ""

    def as_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "status": self.status,
            "reference": self.reference,
        }


@dataclass(frozen=True, slots=True)
class ModelCheckpoint:
    """One registry-declared weight asset."""

    identifier: str
    source: CheckpointSource = CheckpointSource.REMOTE
    required: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "identifier": self.identifier,
            "source": self.source.value,
            "required": self.required,
        }


@dataclass(frozen=True, slots=True)
class ModelDependency:
    """One optional runtime package required by a model adapter."""

    import_name: str
    distribution: str
    requirement_file: str = ""
    purpose: str = ""

    def as_dict(self) -> dict[str, str]:
        return {
            "import_name": self.import_name,
            "distribution": self.distribution,
            "requirement_file": self.requirement_file,
            "purpose": self.purpose,
        }


@dataclass(frozen=True, slots=True)
class HardwareRequirements:
    """Registry-level hardware support and advisory thresholds."""

    supported_backends: tuple[str, ...] = ("cpu", "cuda")
    minimum_cuda_vram_gib: float | None = None
    recommended_cuda_vram_gib: float | None = None
    supported_precisions: tuple[str, ...] = ("fp32",)
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "supported_backends": list(self.supported_backends),
            "minimum_cuda_vram_gib": self.minimum_cuda_vram_gib,
            "recommended_cuda_vram_gib": self.recommended_cuda_vram_gib,
            "supported_precisions": list(self.supported_precisions),
            "notes": self.notes,
        }


@dataclass(frozen=True, slots=True)
class ModelPresentation:
    """UI labels that remain data-driven and independent of Qt imports."""

    description: str = ""
    header_label: str = ""
    header_badge_kind: str = "compatible"
    idle_badge: str = "READY"
    active_badge: str = "READY"
    unavailable_badge: str = "NOT INSTALLED"

    def as_dict(self) -> dict[str, str]:
        return {
            "description": self.description,
            "header_label": self.header_label,
            "header_badge_kind": self.header_badge_kind,
            "idle_badge": self.idle_badge,
            "active_badge": self.active_badge,
            "unavailable_badge": self.unavailable_badge,
        }


@dataclass(frozen=True, slots=True)
class ModelProfileMetadata:
    """Resolved model/profile values displayed by the generic settings UI."""

    variant_label: str
    detail_label: str
    precision_label: str
    point_budget: int
    live_point_budget: int
    tooltip: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "variant_label": self.variant_label,
            "detail_label": self.detail_label,
            "precision_label": self.precision_label,
            "point_budget": self.point_budget,
            "live_point_budget": self.live_point_budget,
            "tooltip": self.tooltip,
        }


@dataclass(frozen=True, slots=True)
class ModelDescriptor:
    """Authoritative declaration for one selectable model adapter."""

    model_id: str
    display_name: str
    adapter_path: str
    runtime_config_builder: str
    profile_metadata_builder: str
    capabilities: ModelCapabilities
    dependencies: tuple[ModelDependency, ...] = ()
    hardware: HardwareRequirements = field(default_factory=HardwareRequirements)
    presentation: ModelPresentation = field(default_factory=ModelPresentation)
    license: ModelLicense = field(default_factory=ModelLicense)
    default_config: Mapping[str, object] = field(default_factory=dict)
    checkpoints: tuple[ModelCheckpoint, ...] = ()
    uses_compute_profile: bool = False
    default: bool = False
    development_only: bool = False
    selectable: bool = True
    enabled: bool = True

    def as_metadata(self) -> dict[str, Any]:
        capabilities = self.capabilities.as_dict()
        return {
            "id": self.model_id,
            "registry_display_name": self.display_name,
            "display_name": self.display_name,
            "is_metric": capabilities["is_metric"],
            "depth_scale": capabilities["depth_scale"],
            "intrinsics_source": capabilities["intrinsics_source"],
            "intrinsics_fallback": capabilities["intrinsics_fallback"],
            "confidence_kind": capabilities["confidence_kind"],
            "outputs": capabilities["outputs"],
            "source_types": capabilities["source_types"],
            "supports_image": capabilities["supports_image"],
            "supports_video": capabilities["supports_video"],
            "supports_camera": capabilities["supports_camera"],
            "supports_rtsp": capabilities["supports_rtsp"],
            "supports_live": capabilities["supports_live"],
            "stateful_sequence": capabilities["stateful_sequence"],
            "development_only": self.development_only,
            "selectable": self.selectable,
            "enabled": self.enabled,
            "adapter_path": self.adapter_path,
            "runtime_config_builder": self.runtime_config_builder,
            "profile_metadata_builder": self.profile_metadata_builder,
            "dependencies": [dependency.as_dict() for dependency in self.dependencies],
            "hardware_requirements": self.hardware.as_dict(),
            "presentation": self.presentation.as_dict(),
            "license": self.license.as_dict(),
            "checkpoints": [checkpoint.as_dict() for checkpoint in self.checkpoints],
            "uses_compute_profile": self.uses_compute_profile,
        }

    def metadata_with_runtime(self, **runtime: object) -> dict[str, Any]:
        """Combine immutable registry metadata with adapter runtime facts.

        Runtime values may enrich telemetry, but they cannot override registry
        capabilities, dependencies, hardware, licensing, or presentation.
        ``display_name`` is the sole presentation exception because it may name
        the concrete checkpoint variant selected by a registry profile.
        """

        static_metadata = self.as_metadata()
        runtime_display_name = str(runtime.get("display_name") or self.display_name)
        metadata = dict(static_metadata)
        metadata.update(runtime)
        metadata.update(static_metadata)
        metadata["display_name"] = runtime_display_name
        return metadata

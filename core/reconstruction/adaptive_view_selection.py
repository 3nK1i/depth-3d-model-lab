"""I4.3F1 explainable, resource-aware dense-view selection foundation.

This module is deliberately provider-execution neutral.  It consumes the
*authoritative* PyCOLMAP SfM result and produces an explainable dense-view plan
for one already-selected provider implementation.  It never changes camera
poses, SfM landmarks, provider qualification, physical scale, measurement
status, or mesh authority.

I4.3F replaces historical fixed experimental view limits with a dynamic plan
based on actual registered views, sparse overlap/baseline evidence, image
quality/resolution, source span, hardware resources, estimated provider work,
and provider-specific requirements.  There is no permanent global view cap.

Production wiring into ingestion/provider execution is intentionally deferred
to the next narrow I4.3F slice after this contract is validated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite, log1p
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from core.contracts.validation import JsonValue, metadata_as_dict, normalize_metadata
from core.hardware.gpu_info import HardwareInfo
from core.reconstruction.dense_provider_architecture import (
    I43A_AUTHORITATIVE_FRAME_OWNER,
    DenseProviderFamily,
    DenseProviderImplementation,
    dense_provider_implementation_contracts,
)
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.sfm_contracts import OfflineSfmResult

I43F1_SCHEMA_VERSION = 1
I43F1_PHASE = "I4.3F1"
I43F1_PERMANENT_GLOBAL_VIEW_CAP = None
_EPS = 1e-12
_GIB = 1024**3


class AdaptiveViewOverrideMode(str, Enum):
    """Engineer-visible override modes; none is a permanent production cap."""

    ALL_REGISTERED = "all_registered"
    EXPLICIT_KEYFRAMES = "explicit_keyframes"


@dataclass(frozen=True, slots=True)
class AdaptiveViewSelectionOverride:
    mode: AdaptiveViewOverrideMode
    keyframe_ids: tuple[str, ...] = ()
    reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.mode, AdaptiveViewOverrideMode):
            raise TypeError("mode must be AdaptiveViewOverrideMode.")
        ids = tuple(str(value).strip() for value in self.keyframe_ids)
        if any(not value for value in ids) or len(ids) != len(set(ids)):
            raise ReconstructionValidationError(
                "Adaptive view override keyframe IDs must be non-empty and unique."
            )
        if self.mode is AdaptiveViewOverrideMode.ALL_REGISTERED and ids:
            raise ReconstructionValidationError(
                "ALL_REGISTERED override must not carry explicit keyframe IDs."
            )
        if self.mode is AdaptiveViewOverrideMode.EXPLICIT_KEYFRAMES and not ids:
            raise ReconstructionValidationError(
                "EXPLICIT_KEYFRAMES override requires at least one keyframe ID."
            )
        reason = str(self.reason).strip()
        if not reason:
            reason = (
                "engineer_requested_all_registered_views"
                if self.mode is AdaptiveViewOverrideMode.ALL_REGISTERED
                else "engineer_requested_explicit_keyframes"
            )
        object.__setattr__(self, "keyframe_ids", ids)
        object.__setattr__(self, "reason", reason)

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "keyframe_ids": list(self.keyframe_ids),
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class ProviderViewSelectionRequirements:
    """Provider-specific selection shape and relative work characteristics.

    ``relative_cost_per_megapixel`` is an explainable planning coefficient, not
    a memory or latency benchmark.  I4.3G/I5 later qualify real runtime/resource
    envelopes; F1 only needs a stable relative pressure signal.
    """

    implementation: DenseProviderImplementation
    family: DenseProviderFamily
    minimum_views: int
    relative_cost_per_megapixel: float
    overlap_priority: float
    baseline_priority: float
    temporal_span_priority: float
    quality_priority: float
    resolution_priority: float
    landmark_support_priority: float
    source_order_required: bool
    stateful_sequence: bool

    def __post_init__(self) -> None:
        if not isinstance(self.implementation, DenseProviderImplementation):
            raise TypeError("implementation must be DenseProviderImplementation.")
        if not isinstance(self.family, DenseProviderFamily):
            raise TypeError("family must be DenseProviderFamily.")
        if isinstance(self.minimum_views, bool) or int(self.minimum_views) < 2:
            raise ReconstructionValidationError("minimum_views must be at least 2.")
        object.__setattr__(self, "minimum_views", int(self.minimum_views))
        cost = float(self.relative_cost_per_megapixel)
        if not isfinite(cost) or cost <= 0.0:
            raise ReconstructionValidationError(
                "relative_cost_per_megapixel must be finite and positive."
            )
        object.__setattr__(self, "relative_cost_per_megapixel", cost)
        names = (
            "overlap_priority",
            "baseline_priority",
            "temporal_span_priority",
            "quality_priority",
            "resolution_priority",
            "landmark_support_priority",
        )
        values = []
        for name in names:
            value = float(getattr(self, name))
            if not isfinite(value) or value < 0.0:
                raise ReconstructionValidationError(f"{name} must be finite and non-negative.")
            values.append(value)
            object.__setattr__(self, name, value)
        if sum(values) <= 0.0:
            raise ReconstructionValidationError("Selection priorities must have positive total weight.")
        if not isinstance(self.source_order_required, bool) or not isinstance(self.stateful_sequence, bool):
            raise TypeError("source_order_required/stateful_sequence must be bool values.")

    @property
    def normalized_priorities(self) -> dict[str, float]:
        values = {
            "overlap": self.overlap_priority,
            "baseline": self.baseline_priority,
            "temporal_span": self.temporal_span_priority,
            "quality": self.quality_priority,
            "resolution": self.resolution_priority,
            "landmark_support": self.landmark_support_priority,
        }
        total = sum(values.values())
        return {key: value / total for key, value in values.items()}

    def as_dict(self) -> dict[str, Any]:
        return {
            "implementation": self.implementation.value,
            "family": self.family.value,
            "minimum_views": self.minimum_views,
            "relative_cost_per_megapixel": self.relative_cost_per_megapixel,
            "priorities": self.normalized_priorities,
            "source_order_required": self.source_order_required,
            "stateful_sequence": self.stateful_sequence,
        }


def _req(
    implementation: DenseProviderImplementation,
    *,
    minimum_views: int,
    cost: float,
    overlap: float,
    baseline: float,
    temporal: float,
    quality: float,
    resolution: float,
    landmark: float,
    source_order: bool = True,
    stateful: bool = False,
) -> ProviderViewSelectionRequirements:
    contract = next(
        item for item in dense_provider_implementation_contracts()
        if item.implementation is implementation
    )
    return ProviderViewSelectionRequirements(
        implementation=implementation,
        family=contract.family,
        minimum_views=minimum_views,
        relative_cost_per_megapixel=cost,
        overlap_priority=overlap,
        baseline_priority=baseline,
        temporal_span_priority=temporal,
        quality_priority=quality,
        resolution_priority=resolution,
        landmark_support_priority=landmark,
        source_order_required=source_order,
        stateful_sequence=stateful,
    )


_PROVIDER_REQUIREMENTS = {
    DenseProviderImplementation.COLMAP_MVS: _req(
        DenseProviderImplementation.COLMAP_MVS,
        minimum_views=3, cost=1.55,
        overlap=0.28, baseline=0.24, temporal=0.08, quality=0.14, resolution=0.08, landmark=0.18,
    ),
    DenseProviderImplementation.DA3: _req(
        DenseProviderImplementation.DA3,
        minimum_views=3, cost=1.35,
        overlap=0.20, baseline=0.18, temporal=0.14, quality=0.16, resolution=0.10, landmark=0.22,
    ),
    DenseProviderImplementation.ROLLING_DEPTH: _req(
        DenseProviderImplementation.ROLLING_DEPTH,
        minimum_views=3, cost=1.30,
        overlap=0.12, baseline=0.10, temporal=0.34, quality=0.18, resolution=0.08, landmark=0.18,
        stateful=True,
    ),
    DenseProviderImplementation.VIDEO_DEPTH_ANYTHING: _req(
        DenseProviderImplementation.VIDEO_DEPTH_ANYTHING,
        minimum_views=3, cost=1.15,
        overlap=0.12, baseline=0.10, temporal=0.34, quality=0.18, resolution=0.08, landmark=0.18,
        stateful=True,
    ),
    DenseProviderImplementation.METRIC3D_V2: _req(
        DenseProviderImplementation.METRIC3D_V2,
        minimum_views=2, cost=1.00,
        overlap=0.16, baseline=0.18, temporal=0.14, quality=0.20, resolution=0.12, landmark=0.20,
    ),
    DenseProviderImplementation.UNIDEPTH_V2: _req(
        DenseProviderImplementation.UNIDEPTH_V2,
        minimum_views=2, cost=1.10,
        overlap=0.16, baseline=0.18, temporal=0.14, quality=0.20, resolution=0.12, landmark=0.20,
    ),
    DenseProviderImplementation.MOGE_2: _req(
        DenseProviderImplementation.MOGE_2,
        minimum_views=2, cost=0.95,
        overlap=0.16, baseline=0.18, temporal=0.14, quality=0.20, resolution=0.12, landmark=0.20,
    ),
    DenseProviderImplementation.DEPTH_PRO: _req(
        DenseProviderImplementation.DEPTH_PRO,
        minimum_views=2, cost=1.20,
        overlap=0.16, baseline=0.18, temporal=0.14, quality=0.20, resolution=0.12, landmark=0.20,
    ),
    DenseProviderImplementation.DEPTH_ANYTHING_V2: _req(
        DenseProviderImplementation.DEPTH_ANYTHING_V2,
        minimum_views=2, cost=0.80,
        overlap=0.16, baseline=0.18, temporal=0.14, quality=0.20, resolution=0.12, landmark=0.20,
    ),
    DenseProviderImplementation.MARIGOLD: _req(
        DenseProviderImplementation.MARIGOLD,
        minimum_views=2, cost=1.45,
        overlap=0.16, baseline=0.18, temporal=0.14, quality=0.20, resolution=0.12, landmark=0.20,
    ),
}


def provider_view_selection_requirements(
    implementation: DenseProviderImplementation,
) -> ProviderViewSelectionRequirements:
    if not isinstance(implementation, DenseProviderImplementation):
        raise TypeError("implementation must be DenseProviderImplementation.")
    return _PROVIDER_REQUIREMENTS[implementation]


@dataclass(frozen=True, slots=True)
class AdaptiveSelectionResourceSnapshot:
    """Resource snapshot used for planning; values are observations, not limits."""

    gpu_name: str
    total_vram_bytes: int | None
    free_vram_bytes: int | None
    available_ram_bytes: int | None
    cuda_ready: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "gpu_name", str(self.gpu_name).strip() or "unknown")
        for field_name in ("total_vram_bytes", "free_vram_bytes", "available_ram_bytes"):
            value = getattr(self, field_name)
            if value is not None:
                if isinstance(value, bool) or int(value) < 0:
                    raise ReconstructionValidationError(f"{field_name} must be non-negative when provided.")
                object.__setattr__(self, field_name, int(value))
        if not isinstance(self.cuda_ready, bool):
            raise TypeError("cuda_ready must be a bool.")

    @classmethod
    def from_hardware(
        cls,
        hardware: HardwareInfo,
        *,
        available_ram_bytes: int | None,
    ) -> "AdaptiveSelectionResourceSnapshot":
        if not isinstance(hardware, HardwareInfo):
            raise TypeError("hardware must be HardwareInfo.")
        return cls(
            gpu_name=hardware.name,
            total_vram_bytes=hardware.total_memory_bytes,
            free_vram_bytes=hardware.free_memory_bytes,
            available_ram_bytes=available_ram_bytes,
            cuda_ready=hardware.cuda_ready,
        )

    @property
    def free_vram_gib(self) -> float | None:
        return None if self.free_vram_bytes is None else self.free_vram_bytes / _GIB

    @property
    def available_ram_gib(self) -> float | None:
        return None if self.available_ram_bytes is None else self.available_ram_bytes / _GIB

    def planning_capacity_units(self) -> float | None:
        """Return transparent heuristic capacity units, not claimed GiB usage.

        RAM and currently free VRAM contribute separately.  The coefficients are
        intentionally exposed through the snapshot metadata and are used only to
        decide whether *all* views fit the planning budget.  Real provider runtime
        envelopes remain an I4.3G/I5 qualification task.
        """

        ram = self.available_ram_gib
        vram = self.free_vram_gib
        if ram is None and vram is None:
            return None
        ram_term = 0.0 if ram is None else max(0.0, ram) * 24.0
        vram_term = 0.0 if vram is None else max(0.0, vram) * 32.0
        if not self.cuda_ready:
            vram_term = 0.0
        return max(0.0, ram_term + vram_term)

    def as_dict(self) -> dict[str, Any]:
        return {
            "gpu_name": self.gpu_name,
            "total_vram_bytes": self.total_vram_bytes,
            "free_vram_bytes": self.free_vram_bytes,
            "available_ram_bytes": self.available_ram_bytes,
            "cuda_ready": self.cuda_ready,
            "free_vram_gib": self.free_vram_gib,
            "available_ram_gib": self.available_ram_gib,
            "planning_capacity_units": self.planning_capacity_units(),
            "planning_capacity_formula": "24*available_ram_gib + 32*free_vram_gib_when_cuda_ready",
            "capacity_is_empirical_runtime_claim": False,
        }


@dataclass(frozen=True, slots=True)
class AdaptiveViewSelectionPolicy:
    """Explainable planner controls; deliberately contains no fixed view cap."""

    minimum_budget_utilization: float = 0.70
    preserve_source_endpoints: bool = True
    inspect_image_sharpness: bool = True
    fallback_sharpness_score: float = 0.50
    schema_version: int = I43F1_SCHEMA_VERSION

    def __post_init__(self) -> None:
        value = float(self.minimum_budget_utilization)
        if not isfinite(value) or not 0.0 < value <= 1.0:
            raise ReconstructionValidationError(
                "minimum_budget_utilization must be finite and within (0, 1]."
            )
        fallback = float(self.fallback_sharpness_score)
        if not isfinite(fallback) or not 0.0 <= fallback <= 1.0:
            raise ReconstructionValidationError(
                "fallback_sharpness_score must be finite and within [0, 1]."
            )
        if not isinstance(self.preserve_source_endpoints, bool) or not isinstance(self.inspect_image_sharpness, bool):
            raise TypeError("preserve_source_endpoints/inspect_image_sharpness must be bool values.")
        if int(self.schema_version) != I43F1_SCHEMA_VERSION:
            raise ReconstructionValidationError("Unsupported I4.3F1 policy schema version.")
        object.__setattr__(self, "minimum_budget_utilization", value)
        object.__setattr__(self, "fallback_sharpness_score", fallback)
        object.__setattr__(self, "schema_version", I43F1_SCHEMA_VERSION)

    def as_dict(self) -> dict[str, Any]:
        return {
            "minimum_budget_utilization": self.minimum_budget_utilization,
            "preserve_source_endpoints": self.preserve_source_endpoints,
            "inspect_image_sharpness": self.inspect_image_sharpness,
            "fallback_sharpness_score": self.fallback_sharpness_score,
            "permanent_global_view_cap": None,
        }


@dataclass(frozen=True, slots=True)
class AdaptiveViewEvidence:
    keyframe_id: str
    frame_id: str
    source_sequence_index: int
    timestamp_ns: int
    width: int
    height: int
    megapixels: float
    sharpness_score: float
    sharpness_available: bool
    landmark_observation_count: int
    landmark_support_score: float
    source_span_position: float
    estimated_work_units: float

    def __post_init__(self) -> None:
        for name in ("keyframe_id", "frame_id"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ReconstructionValidationError(f"{name} must not be empty.")
            object.__setattr__(self, name, value)
        for name in ("source_sequence_index", "timestamp_ns", "width", "height", "landmark_observation_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) < 0:
                raise ReconstructionValidationError(f"{name} must be non-negative.")
            object.__setattr__(self, name, int(value))
        if self.width <= 0 or self.height <= 0:
            raise ReconstructionValidationError("View evidence dimensions must be positive.")
        for name in ("megapixels", "sharpness_score", "landmark_support_score", "source_span_position", "estimated_work_units"):
            value = float(getattr(self, name))
            if not isfinite(value) or value < 0.0:
                raise ReconstructionValidationError(f"{name} must be finite and non-negative.")
            object.__setattr__(self, name, value)
        for name in ("sharpness_score", "landmark_support_score", "source_span_position"):
            if getattr(self, name) > 1.0 + 1e-12:
                raise ReconstructionValidationError(f"{name} must be within [0, 1].")
        if not isinstance(self.sharpness_available, bool):
            raise TypeError("sharpness_available must be bool.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "keyframe_id": self.keyframe_id,
            "frame_id": self.frame_id,
            "source_sequence_index": self.source_sequence_index,
            "timestamp_ns": self.timestamp_ns,
            "image_size": {"width": self.width, "height": self.height},
            "megapixels": self.megapixels,
            "sharpness_score": self.sharpness_score,
            "sharpness_available": self.sharpness_available,
            "landmark_observation_count": self.landmark_observation_count,
            "landmark_support_score": self.landmark_support_score,
            "source_span_position": self.source_span_position,
            "estimated_work_units": self.estimated_work_units,
        }


@dataclass(frozen=True, slots=True)
class AdaptiveViewDecision:
    keyframe_id: str
    selected: bool
    score: float
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        keyframe = str(self.keyframe_id).strip()
        if not keyframe:
            raise ReconstructionValidationError("Adaptive view decision keyframe_id must not be empty.")
        score = float(self.score)
        if not isfinite(score) or not 0.0 <= score <= 1.0:
            raise ReconstructionValidationError("Adaptive view decision score must be within [0, 1].")
        reasons = tuple(str(value).strip() for value in self.reason_codes)
        if not reasons or any(not value for value in reasons):
            raise ReconstructionValidationError("Adaptive view decisions require non-empty reason codes.")
        if not isinstance(self.selected, bool):
            raise TypeError("selected must be bool.")
        object.__setattr__(self, "keyframe_id", keyframe)
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "reason_codes", reasons)

    def as_dict(self) -> dict[str, Any]:
        return {
            "keyframe_id": self.keyframe_id,
            "selected": self.selected,
            "score": self.score,
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True, slots=True)
class AdaptiveViewSelectionPlan:
    implementation: DenseProviderImplementation
    source_view_count: int
    selected_keyframe_ids: tuple[str, ...]
    source_order_keyframe_ids: tuple[str, ...]
    adaptive_selection_evaluated: bool
    reduction_applied: bool
    engineer_override_applied: bool
    reason_codes: tuple[str, ...]
    estimated_source_work_units: float
    estimated_selected_work_units: float
    planning_capacity_units: float | None
    resource_limited_target_view_count: int | None
    requirements: ProviderViewSelectionRequirements
    resources: AdaptiveSelectionResourceSnapshot
    evidence: tuple[AdaptiveViewEvidence, ...]
    decisions: tuple[AdaptiveViewDecision, ...]
    authority_metadata: Mapping[str, JsonValue] | Sequence[tuple[str, JsonValue]] = field(default_factory=tuple)
    permanent_global_view_cap: int | None = I43F1_PERMANENT_GLOBAL_VIEW_CAP
    schema_version: int = I43F1_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.implementation, DenseProviderImplementation):
            raise TypeError("implementation must be DenseProviderImplementation.")
        if isinstance(self.source_view_count, bool) or int(self.source_view_count) < 2:
            raise ReconstructionValidationError("source_view_count must be at least 2.")
        source_count = int(self.source_view_count)
        selected = tuple(str(value).strip() for value in self.selected_keyframe_ids)
        source_order = tuple(str(value).strip() for value in self.source_order_keyframe_ids)
        if any(not value for value in selected + source_order):
            raise ReconstructionValidationError("Selection-plan keyframe IDs must not be empty.")
        if len(source_order) != source_count or len(source_order) != len(set(source_order)):
            raise ReconstructionValidationError("source_order_keyframe_ids must represent every source view exactly once.")
        if not selected or len(selected) != len(set(selected)) or any(value not in source_order for value in selected):
            raise ReconstructionValidationError("selected_keyframe_ids must be a unique non-empty subset of source views.")
        if len(selected) < self.requirements.minimum_views:
            raise ReconstructionValidationError("Adaptive plan selected fewer than the provider minimum views.")
        if self.permanent_global_view_cap is not None:
            raise ReconstructionValidationError("I4.3F forbids a permanent global view cap.")
        if not self.adaptive_selection_evaluated:
            raise ReconstructionValidationError("I4.3F plans must record that adaptive selection was evaluated.")
        if bool(self.reduction_applied) != (len(selected) < source_count):
            raise ReconstructionValidationError("reduction_applied must match selected/source view counts.")
        if self.resource_limited_target_view_count is not None:
            target = int(self.resource_limited_target_view_count)
            if target < self.requirements.minimum_views or target > source_count:
                raise ReconstructionValidationError("resource_limited_target_view_count is invalid.")
            object.__setattr__(self, "resource_limited_target_view_count", target)
        reasons = tuple(str(value).strip() for value in self.reason_codes)
        if not reasons or any(not value for value in reasons):
            raise ReconstructionValidationError("Adaptive plans require engineer-visible reason codes.")
        evidence = tuple(self.evidence)
        decisions = tuple(self.decisions)
        if len(evidence) != source_count or len(decisions) != source_count:
            raise ReconstructionValidationError("Adaptive plan evidence/decisions must cover every source view.")
        if {item.keyframe_id for item in evidence} != set(source_order):
            raise ReconstructionValidationError("Adaptive plan evidence keyframes drifted from source views.")
        if {item.keyframe_id for item in decisions} != set(source_order):
            raise ReconstructionValidationError("Adaptive plan decision keyframes drifted from source views.")
        selected_from_decisions = {item.keyframe_id for item in decisions if item.selected}
        if selected_from_decisions != set(selected):
            raise ReconstructionValidationError("Adaptive plan decisions do not match selected_keyframe_ids.")
        if self.requirements.source_order_required:
            index = {value: i for i, value in enumerate(source_order)}
            if tuple(sorted(selected, key=index.__getitem__)) != selected:
                raise ReconstructionValidationError("Selected views must preserve authoritative source order.")
        source_work = float(self.estimated_source_work_units)
        selected_work = float(self.estimated_selected_work_units)
        if not isfinite(source_work) or source_work <= 0.0 or not isfinite(selected_work) or selected_work <= 0.0:
            raise ReconstructionValidationError("Estimated selection work must be finite and positive.")
        if selected_work > source_work + 1e-9:
            raise ReconstructionValidationError("Selected work cannot exceed source work.")
        if self.planning_capacity_units is not None:
            capacity = float(self.planning_capacity_units)
            if not isfinite(capacity) or capacity < 0.0:
                raise ReconstructionValidationError("planning_capacity_units must be non-negative when provided.")
            object.__setattr__(self, "planning_capacity_units", capacity)
        if not isinstance(self.resources, AdaptiveSelectionResourceSnapshot):
            raise TypeError("resources must be AdaptiveSelectionResourceSnapshot.")
        if not isinstance(self.requirements, ProviderViewSelectionRequirements):
            raise TypeError("requirements must be ProviderViewSelectionRequirements.")
        if self.requirements.implementation is not self.implementation:
            raise ReconstructionValidationError("Provider requirements do not match plan implementation.")
        authority = normalize_metadata(self.authority_metadata)
        authority_dict = metadata_as_dict(authority)
        required_false = (
            "metric_scale_claimed",
            "measurement_authoritative",
            "certified_metrology",
            "surface_mesh_generated",
            "provider_selected",
            "provider_promoted",
            "auto_policy_implemented",
        )
        if authority_dict.get("camera_pose_authority_owner") != I43A_AUTHORITATIVE_FRAME_OWNER:
            raise ReconstructionValidationError("I4.3F must preserve PyCOLMAP SfM camera-pose authority.")
        if any(bool(authority_dict.get(name, False)) for name in required_false):
            raise ReconstructionValidationError("I4.3F selection cannot promote provider/Auto/metric/measurement/metrology/mesh authority.")
        if int(self.schema_version) != I43F1_SCHEMA_VERSION:
            raise ReconstructionValidationError("Unsupported I4.3F1 plan schema version.")
        object.__setattr__(self, "source_view_count", source_count)
        object.__setattr__(self, "selected_keyframe_ids", selected)
        object.__setattr__(self, "source_order_keyframe_ids", source_order)
        object.__setattr__(self, "reason_codes", reasons)
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "decisions", decisions)
        object.__setattr__(self, "estimated_source_work_units", source_work)
        object.__setattr__(self, "estimated_selected_work_units", selected_work)
        object.__setattr__(self, "authority_metadata", authority)
        object.__setattr__(self, "schema_version", I43F1_SCHEMA_VERSION)

    @property
    def selected_view_count(self) -> int:
        return len(self.selected_keyframe_ids)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "phase": I43F1_PHASE,
            "implementation": self.implementation.value,
            "source_view_count": self.source_view_count,
            "selected_view_count": self.selected_view_count,
            "selected_keyframe_ids": list(self.selected_keyframe_ids),
            "source_order_keyframe_ids": list(self.source_order_keyframe_ids),
            "adaptive_selection_evaluated": self.adaptive_selection_evaluated,
            "reduction_applied": self.reduction_applied,
            "engineer_override_applied": self.engineer_override_applied,
            "reason_codes": list(self.reason_codes),
            "estimated_source_work_units": self.estimated_source_work_units,
            "estimated_selected_work_units": self.estimated_selected_work_units,
            "planning_capacity_units": self.planning_capacity_units,
            "resource_limited_target_view_count": self.resource_limited_target_view_count,
            "permanent_global_view_cap": self.permanent_global_view_cap,
            "requirements": self.requirements.as_dict(),
            "resources": self.resources.as_dict(),
            "evidence": [item.as_dict() for item in self.evidence],
            "decisions": [item.as_dict() for item in self.decisions],
            "authority": metadata_as_dict(self.authority_metadata),
        }


def _workspace_order(sfm_result: OfflineSfmResult) -> tuple[tuple[Any, Any], ...]:
    cameras = {item.keyframe_id: item for item in sfm_result.cameras}
    rows = tuple(
        (image, cameras[image.keyframe_id])
        for image in sorted(sfm_result.workspace.images, key=lambda item: item.order)
        if image.keyframe_id in cameras
    )
    if len(rows) != len(sfm_result.cameras):
        raise ReconstructionValidationError(
            "I4.3F requires every registered SfM camera to resolve to one workspace image."
        )
    if len(rows) < 2:
        raise ReconstructionValidationError("I4.3F requires at least two registered SfM views.")
    return rows


def _landmark_sets(sfm_result: OfflineSfmResult) -> dict[str, set[str]]:
    result = {camera.keyframe_id: set() for camera in sfm_result.cameras}
    for landmark in sfm_result.landmarks:
        for observation in landmark.observations:
            if observation.keyframe_id in result:
                result[observation.keyframe_id].add(landmark.landmark_id)
    return result


def _overlap_matrix(keyframe_ids: tuple[str, ...], landmark_sets: Mapping[str, set[str]]) -> np.ndarray:
    n = len(keyframe_ids)
    matrix = np.eye(n, dtype=np.float64)
    for i in range(n):
        left = landmark_sets[keyframe_ids[i]]
        for j in range(i + 1, n):
            right = landmark_sets[keyframe_ids[j]]
            denom = min(len(left), len(right))
            value = 0.0 if denom <= 0 else len(left.intersection(right)) / float(denom)
            matrix[i, j] = matrix[j, i] = float(np.clip(value, 0.0, 1.0))
    return matrix


def _baseline_matrix(cameras: Sequence[Any]) -> np.ndarray:
    centers = np.vstack([np.asarray(item.camera_center_reconstruction, dtype=np.float64) for item in cameras])
    delta = centers[:, None, :] - centers[None, :, :]
    distance = np.linalg.norm(delta, axis=2)
    nonzero = distance[distance > _EPS]
    if nonzero.size == 0:
        return np.zeros_like(distance)
    scale = float(np.quantile(nonzero, 0.75))
    scale = max(scale, _EPS)
    return np.clip(distance / scale, 0.0, 1.0)


def _sharpness_for_image(path: Path, *, fallback: float) -> tuple[float, bool, float]:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.size <= 0:
        return fallback, False, 0.0
    variance = float(cv2.Laplacian(image, cv2.CV_64F).var())
    if not isfinite(variance) or variance < 0.0:
        return fallback, False, 0.0
    return 0.0, True, variance


def _normalize(values: Sequence[float], *, fallback: float = 0.5) -> tuple[float, ...]:
    array = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(array)
    if not finite.any():
        return tuple(float(fallback) for _ in values)
    valid = array[finite]
    low = float(np.min(valid))
    high = float(np.max(valid))
    if high - low <= _EPS:
        return tuple(float(fallback) for _ in values)
    result = np.full(len(array), float(fallback), dtype=np.float64)
    result[finite] = (array[finite] - low) / (high - low)
    return tuple(float(np.clip(value, 0.0, 1.0)) for value in result)


def _build_evidence(
    sfm_result: OfflineSfmResult,
    requirements: ProviderViewSelectionRequirements,
    policy: AdaptiveViewSelectionPolicy,
) -> tuple[tuple[AdaptiveViewEvidence, ...], np.ndarray, np.ndarray]:
    rows = _workspace_order(sfm_result)
    keyframe_ids = tuple(image.keyframe_id for image, _ in rows)
    landmarks = _landmark_sets(sfm_result)
    overlap = _overlap_matrix(keyframe_ids, landmarks)
    baseline = _baseline_matrix([camera for _, camera in rows])
    root = Path(sfm_result.workspace.workspace_root).expanduser().resolve()

    raw_sharpness: list[float] = []
    sharpness_available: list[bool] = []
    for image, _ in rows:
        if not policy.inspect_image_sharpness:
            raw_sharpness.append(float("nan"))
            sharpness_available.append(False)
            continue
        _, available, variance = _sharpness_for_image(
            root / image.staged_image_path,
            fallback=policy.fallback_sharpness_score,
        )
        raw_sharpness.append(log1p(variance) if available else float("nan"))
        sharpness_available.append(available)
    normalized_sharpness = _normalize(raw_sharpness, fallback=policy.fallback_sharpness_score)

    counts = [len(landmarks[keyframe]) for keyframe in keyframe_ids]
    normalized_landmarks = _normalize(counts, fallback=0.5)
    source_indices = [image.source_sequence_index for image, _ in rows]
    minimum_index = min(source_indices)
    maximum_index = max(source_indices)
    span = max(1, maximum_index - minimum_index)

    evidence: list[AdaptiveViewEvidence] = []
    for index, (image, _) in enumerate(rows):
        megapixels = (image.image_size.width * image.image_size.height) / 1_000_000.0
        evidence.append(
            AdaptiveViewEvidence(
                keyframe_id=image.keyframe_id,
                frame_id=image.frame_id,
                source_sequence_index=image.source_sequence_index,
                timestamp_ns=image.timestamp_ns,
                width=image.image_size.width,
                height=image.image_size.height,
                megapixels=megapixels,
                sharpness_score=normalized_sharpness[index],
                sharpness_available=sharpness_available[index],
                landmark_observation_count=counts[index],
                landmark_support_score=normalized_landmarks[index],
                source_span_position=(image.source_sequence_index - minimum_index) / float(span),
                estimated_work_units=max(0.01, megapixels) * requirements.relative_cost_per_megapixel,
            )
        )
    return tuple(evidence), overlap, baseline


def _target_count_from_resources(
    evidence: tuple[AdaptiveViewEvidence, ...],
    requirements: ProviderViewSelectionRequirements,
    resources: AdaptiveSelectionResourceSnapshot,
    policy: AdaptiveViewSelectionPolicy,
) -> tuple[int, float | None]:
    source_count = len(evidence)
    capacity = resources.planning_capacity_units()
    if capacity is None:
        return source_count, None
    source_work = sum(item.estimated_work_units for item in evidence)
    usable = capacity * policy.minimum_budget_utilization
    if source_work <= usable + 1e-9:
        return source_count, capacity
    mean_work = source_work / float(source_count)
    if mean_work <= _EPS:
        return source_count, capacity
    dynamic = int(np.floor(usable / mean_work))
    dynamic = max(requirements.minimum_views, min(source_count, dynamic))
    return dynamic, capacity


def _span_gap_score(candidate: int, selected: set[int], positions: np.ndarray) -> float:
    if not selected:
        return 1.0
    return float(min(1.0, min(abs(float(positions[candidate] - positions[index])) for index in selected) * 2.0))


def _candidate_score(
    candidate: int,
    selected: set[int],
    evidence: tuple[AdaptiveViewEvidence, ...],
    overlap: np.ndarray,
    baseline: np.ndarray,
    weights: Mapping[str, float],
) -> float:
    item = evidence[candidate]
    if selected:
        overlap_support = max(float(overlap[candidate, index]) for index in selected)
        baseline_novelty = min(1.0, min(float(baseline[candidate, index]) for index in selected))
    else:
        overlap_support = 1.0
        baseline_novelty = 1.0
    positions = np.asarray([row.source_span_position for row in evidence], dtype=np.float64)
    temporal_gap = _span_gap_score(candidate, selected, positions)
    resolution_values = np.asarray([row.megapixels for row in evidence], dtype=np.float64)
    max_resolution = max(float(np.max(resolution_values)), _EPS)
    resolution_score = float(np.clip(item.megapixels / max_resolution, 0.0, 1.0))
    score = (
        weights["overlap"] * overlap_support
        + weights["baseline"] * baseline_novelty
        + weights["temporal_span"] * temporal_gap
        + weights["quality"] * item.sharpness_score
        + weights["resolution"] * resolution_score
        + weights["landmark_support"] * item.landmark_support_score
    )
    return float(np.clip(score, 0.0, 1.0))


def _greedy_select(
    evidence: tuple[AdaptiveViewEvidence, ...],
    overlap: np.ndarray,
    baseline: np.ndarray,
    requirements: ProviderViewSelectionRequirements,
    policy: AdaptiveViewSelectionPolicy,
    target_count: int,
) -> tuple[set[int], dict[int, float], dict[int, tuple[str, ...]]]:
    n = len(evidence)
    if target_count >= n:
        selected = set(range(n))
        return (
            selected,
            {index: 1.0 for index in range(n)},
            {index: ("all_registered_views_fit_resource_budget",) for index in range(n)},
        )

    weights = requirements.normalized_priorities
    selected: set[int] = set()
    score_by_index: dict[int, float] = {}
    reasons: dict[int, tuple[str, ...]] = {}

    if policy.preserve_source_endpoints and n >= 2:
        selected.update((0, n - 1))
        score_by_index[0] = 1.0
        score_by_index[n - 1] = 1.0
        reasons[0] = ("source_span_endpoint",)
        reasons[n - 1] = ("source_span_endpoint",)

    # Seed the strongest sparse/quality view when endpoints alone are not enough.
    if len(selected) < min(target_count, requirements.minimum_views):
        candidates = [index for index in range(n) if index not in selected]
        if candidates:
            seed = max(
                candidates,
                key=lambda index: (
                    evidence[index].landmark_support_score + evidence[index].sharpness_score,
                    -evidence[index].source_sequence_index,
                ),
            )
            selected.add(seed)
            score_by_index[seed] = 1.0
            reasons[seed] = ("high_sparse_support_quality_seed",)

    while len(selected) < target_count:
        candidates = [index for index in range(n) if index not in selected]
        scored = [
            (
                _candidate_score(index, selected, evidence, overlap, baseline, weights),
                -evidence[index].source_sequence_index,
                index,
            )
            for index in candidates
        ]
        score, _, chosen = max(scored)
        selected.add(chosen)
        score_by_index[chosen] = score
        reasons[chosen] = (
            "adaptive_overlap_baseline_quality_resource_selection",
        )

    for index in range(n):
        if index not in selected:
            score = _candidate_score(index, selected, evidence, overlap, baseline, weights)
            score_by_index[index] = score
            reasons[index] = ("resource_budget_deferred_view",)
    return selected, score_by_index, reasons


def build_adaptive_view_selection_plan(
    sfm_result: OfflineSfmResult,
    *,
    implementation: DenseProviderImplementation,
    resources: AdaptiveSelectionResourceSnapshot,
    policy: AdaptiveViewSelectionPolicy | None = None,
    override: AdaptiveViewSelectionOverride | None = None,
) -> AdaptiveViewSelectionPlan:
    """Build one deterministic, explainable dense-view plan.

    The planner consumes only registered authoritative SfM views.  It does not
    run or choose a provider.  Auto provider selection remains deferred to I5.
    """

    if not isinstance(sfm_result, OfflineSfmResult):
        raise TypeError("sfm_result must be OfflineSfmResult.")
    if not isinstance(implementation, DenseProviderImplementation):
        raise TypeError("implementation must be DenseProviderImplementation.")
    if not isinstance(resources, AdaptiveSelectionResourceSnapshot):
        raise TypeError("resources must be AdaptiveSelectionResourceSnapshot.")
    active_policy = policy or AdaptiveViewSelectionPolicy()
    if not isinstance(active_policy, AdaptiveViewSelectionPolicy):
        raise TypeError("policy must be AdaptiveViewSelectionPolicy or None.")
    if override is not None and not isinstance(override, AdaptiveViewSelectionOverride):
        raise TypeError("override must be AdaptiveViewSelectionOverride or None.")

    requirements = provider_view_selection_requirements(implementation)
    evidence, overlap, baseline = _build_evidence(
        sfm_result,
        requirements,
        active_policy,
    )
    if len(evidence) < requirements.minimum_views:
        raise ReconstructionValidationError(
            f"I4.3F {implementation.value} requires at least {requirements.minimum_views} registered views."
        )
    source_order = tuple(item.keyframe_id for item in evidence)
    source_work = float(sum(item.estimated_work_units for item in evidence))

    target_count, capacity = _target_count_from_resources(
        evidence,
        requirements,
        resources,
        active_policy,
    )
    reason_codes: list[str] = []
    override_applied = override is not None

    if override is not None:
        if override.mode is AdaptiveViewOverrideMode.ALL_REGISTERED:
            selected_indices = set(range(len(evidence)))
            score_by_index = {index: 1.0 for index in selected_indices}
            decision_reasons = {
                index: ("engineer_override_all_registered", override.reason)
                for index in selected_indices
            }
            target_count = len(evidence)
            reason_codes.extend(("engineer_override_applied", "engineer_selected_all_registered_views"))
        else:
            known = {item.keyframe_id for item in evidence}
            unknown = [value for value in override.keyframe_ids if value not in known]
            if unknown:
                raise ReconstructionValidationError(
                    f"I4.3F engineer override references unregistered keyframe: {unknown[0]}"
                )
            if len(override.keyframe_ids) < requirements.minimum_views:
                raise ReconstructionValidationError(
                    "I4.3F engineer override cannot violate the provider minimum-view requirement."
                )
            selected_ids = set(override.keyframe_ids)
            selected_indices = {
                index for index, item in enumerate(evidence) if item.keyframe_id in selected_ids
            }
            score_by_index = {index: (1.0 if index in selected_indices else 0.0) for index in range(len(evidence))}
            decision_reasons = {
                index: (
                    ("engineer_override_explicit_view", override.reason)
                    if index in selected_indices
                    else ("engineer_override_excluded_view", override.reason)
                )
                for index in range(len(evidence))
            }
            target_count = len(selected_indices)
            reason_codes.extend(("engineer_override_applied", "engineer_selected_explicit_keyframes"))
    else:
        selected_indices, score_by_index, decision_reasons = _greedy_select(
            evidence,
            overlap,
            baseline,
            requirements,
            active_policy,
            target_count,
        )
        if len(selected_indices) == len(evidence):
            reason_codes.append("all_registered_views_fit_resource_budget")
        else:
            reason_codes.extend((
                "resource_pressure_adaptive_reduction",
                "sparse_overlap_baseline_quality_diversity_preserved",
            ))
        if requirements.stateful_sequence:
            reason_codes.append("temporal_provider_source_order_preserved")

    selected_ids = tuple(
        item.keyframe_id for index, item in enumerate(evidence) if index in selected_indices
    )
    selected_work = float(sum(evidence[index].estimated_work_units for index in selected_indices))
    decisions = tuple(
        AdaptiveViewDecision(
            keyframe_id=item.keyframe_id,
            selected=index in selected_indices,
            score=score_by_index[index],
            reason_codes=decision_reasons[index],
        )
        for index, item in enumerate(evidence)
    )

    return AdaptiveViewSelectionPlan(
        implementation=implementation,
        source_view_count=len(evidence),
        selected_keyframe_ids=selected_ids,
        source_order_keyframe_ids=source_order,
        adaptive_selection_evaluated=True,
        reduction_applied=len(selected_ids) < len(evidence),
        engineer_override_applied=override_applied,
        reason_codes=tuple(dict.fromkeys(reason_codes)),
        estimated_source_work_units=source_work,
        estimated_selected_work_units=selected_work,
        planning_capacity_units=capacity,
        resource_limited_target_view_count=target_count if override is None else None,
        requirements=requirements,
        resources=resources,
        evidence=evidence,
        decisions=decisions,
        authority_metadata={
            "camera_pose_authority_owner": I43A_AUTHORITATIVE_FRAME_OWNER,
            "authoritative_sfm_modified": False,
            "provider_selected": False,
            "provider_promoted": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
        },
        permanent_global_view_cap=None,
    )


def i4_3f1_architecture_snapshot() -> dict[str, Any]:
    """Execution-free F1 contract snapshot used by tests/binding smoke."""

    requirements = [
        provider_view_selection_requirements(item.implementation).as_dict()
        for item in dense_provider_implementation_contracts()
    ]
    return {
        "schema_version": I43F1_SCHEMA_VERSION,
        "phase": I43F1_PHASE,
        "status": "pass",
        "pass": True,
        "scope": {
            "adaptive_dense_view_planner_contract": True,
            "production_wiring_performed": False,
            "pre_sfm_adaptive_frame_planner_deferred_to": "I4.3F2",
            "provider_execution_wiring_deferred_to": "I4.3F2",
        },
        "selection": {
            "registered_sfm_views_only": True,
            "evidence_considered": [
                "sfm_registration",
                "sparse_landmark_support",
                "pairwise_sparse_overlap",
                "camera_baseline_diversity",
                "image_sharpness",
                "image_resolution",
                "source_temporal_span",
                "available_vram",
                "available_ram",
                "estimated_provider_work",
                "provider_requirements",
            ],
            "engineer_override_supported": True,
            "engineer_visible_reason_codes": True,
            "permanent_global_view_cap": None,
            "all_registered_views_remain_eligible": True,
        },
        "provider_requirements": requirements,
        "authority": {
            "camera_pose_authority_owner": I43A_AUTHORITATIVE_FRAME_OWNER,
            "authoritative_sfm_modified": False,
            "provider_selected": False,
            "provider_promoted": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
        },
        "next": "I4.3F2 production frame/view-selection wiring after F1 contract validation.",
    }


def _require_snapshot_pass(snapshot: Mapping[str, Any], *, phase: str) -> None:
    if not isinstance(snapshot, Mapping):
        raise TypeError(f"{phase} snapshot must be a mapping.")
    if snapshot.get("phase") != phase or snapshot.get("status") != "pass" or snapshot.get("pass") is not True:
        raise ReconstructionValidationError(f"I4.3F1 requires a passing {phase} snapshot.")


def validate_i4_3f1_predecessors(
    i4_3a_snapshot: Mapping[str, Any],
    i4_3e6_snapshot: Mapping[str, Any],
) -> None:
    """Fail closed unless I4.3A and the complete E1-E6 chain authorize F1."""

    _require_snapshot_pass(i4_3a_snapshot, phase="I4.3A")
    architecture = i4_3a_snapshot.get("architecture")
    if not isinstance(architecture, Mapping):
        raise ReconstructionValidationError("I4.3F1 I4.3A snapshot is missing architecture evidence.")
    if architecture.get("authoritative_camera_frame_owner") != I43A_AUTHORITATIVE_FRAME_OWNER:
        raise ReconstructionValidationError("I4.3F1 refuses non-PyCOLMAP camera authority.")
    provider_families = architecture.get("provider_family_contracts")
    if isinstance(provider_families, list):
        for contract in provider_families:
            if not isinstance(contract, Mapping):
                raise ReconstructionValidationError("I4.3F1 provider-family contract evidence is malformed.")
            if contract.get("permanent_global_view_cap") is not None:
                raise ReconstructionValidationError("I4.3F1 refuses predecessor permanent global view caps.")
            if contract.get("adaptive_view_selection_required") is not True:
                raise ReconstructionValidationError("I4.3F1 requires adaptive-view selection on every provider family.")

    _require_snapshot_pass(i4_3e6_snapshot, phase="I4.3E6")
    qualification = i4_3e6_snapshot.get("qualification")
    if not isinstance(qualification, Mapping):
        raise ReconstructionValidationError("I4.3F1 E6 snapshot is missing qualification evidence.")
    if qualification.get("i4_3e_provider_adapters_complete") is not True:
        raise ReconstructionValidationError("I4.3F1 requires all I4.3E provider adapters to be complete.")
    if qualification.get("remaining_i4_3e_providers_pending") not in ([], tuple()):
        raise ReconstructionValidationError("I4.3F1 refuses an incomplete I4.3E provider chain.")
    authority = i4_3e6_snapshot.get("authority")
    if not isinstance(authority, Mapping):
        raise ReconstructionValidationError("I4.3F1 E6 snapshot is missing authority evidence.")
    for name in (
        "provider_promoted",
        "metric_scale_claimed",
        "measurement_authoritative",
        "certified_metrology",
        "surface_mesh_generated",
    ):
        if bool(authority.get(name, False)):
            raise ReconstructionValidationError(f"I4.3F1 refuses promoted predecessor authority: {name}.")


def build_i4_3f1_binding_snapshot(
    *,
    i4_3a_snapshot: Mapping[str, Any],
    i4_3e6_snapshot: Mapping[str, Any],
    resources: AdaptiveSelectionResourceSnapshot,
) -> dict[str, Any]:
    validate_i4_3f1_predecessors(i4_3a_snapshot, i4_3e6_snapshot)
    if not isinstance(resources, AdaptiveSelectionResourceSnapshot):
        raise TypeError("resources must be AdaptiveSelectionResourceSnapshot.")
    snapshot = i4_3f1_architecture_snapshot()
    snapshot.update(
        {
            "resources": resources.as_dict(),
            "decision": {
                "i4_3f1_adaptive_selection_contract_closed": True,
                "i4_3f2_production_wiring_authorized": True,
                "provider_selected": False,
                "provider_promoted": False,
                "auto_policy_implemented": False,
                "permanent_global_view_cap": None,
            },
        }
    )
    return snapshot

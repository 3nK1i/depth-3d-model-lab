"""I4.3A canonical production dense-provider architecture contracts.

This module is deliberately execution-free.  It freezes the contract surface for
I4.3B-H without probing, starting, selecting, or promoting a concrete dense
backend.

The architecture distinguishes three different concepts that must never be
collapsed into one enum:

* selection mode: Auto policy vs an explicitly requested concrete provider;
* provider family: MVS, multi-view AI depth, temporal depth, single-image depth,
  and a future evidence-gated hybrid family;
* provider implementation: COLMAP MVS, DA3, RollingDepth, Video Depth Anything,
  Metric3D v2, UniDepthV2, MoGe-2, Depth Pro, Depth Anything V2, and Marigold.

PyCOLMAP SfM remains the sole authoritative camera/global reconstruction frame.
Dense providers may contribute dense geometry/evidence in that frame, but dense
provider execution alone never establishes physical metric scale, measurement
or metrology authority, or a surface mesh.

I4.3A also freezes the progressive-preview event contract required by the v1.0
UI.  The event contract does not implement a UI or force unfinished geometry to
be authoritative; provisional geometry is explicitly replaceable/non-authoritative.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from core.contracts.validation import JsonValue, metadata_as_dict, normalize_metadata, require_nonempty
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.orchestration import ReconstructionProviderExecution

I43A_SCHEMA_VERSION = 2
I43A_PHASE = "I4.3A"
I43A_AUTO_POLICY_ID = "auto"
I43A_AUTHORITATIVE_FRAME_OWNER = "pycolmap_sfm"
I43A_NO_PERMANENT_VIEW_CAP = None


class DenseProviderSelectionMode(str, Enum):
    """How a provider is requested; concrete provider identity is separate."""

    AUTO = "auto"
    EXPLICIT_PROVIDER = "explicit_provider"


class DenseProviderFamily(str, Enum):
    MVS = "mvs"
    MULTI_VIEW_AI_DEPTH = "multi_view_ai_depth"
    TEMPORAL_DEPTH = "temporal_depth"
    SINGLE_IMAGE_DEPTH = "single_image_depth"
    HYBRID = "hybrid"


class DenseProviderImplementation(str, Enum):
    COLMAP_MVS = "colmap_mvs"
    DA3 = "da3"
    ROLLING_DEPTH = "rolling_depth"
    VIDEO_DEPTH_ANYTHING = "video_depth_anything"
    METRIC3D_V2 = "metric3d_v2"
    UNIDEPTH_V2 = "unidepth_v2"
    MOGE_2 = "moge_2"
    DEPTH_PRO = "depth_pro"
    DEPTH_ANYTHING_V2 = "depth_anything_v2"
    MARIGOLD = "marigold"


class DenseProviderNativeOutput(str, Enum):
    DENSE_GEOMETRY = "dense_geometry"
    DEPTH_EVIDENCE = "depth_evidence"
    COMPOSITE_GEOMETRY = "composite_geometry"


class DenseProviderFrameBinding(str, Enum):
    AUTHORITATIVE_SFM_FRAME = "authoritative_sfm_frame"
    ALIGN_TO_AUTHORITATIVE_SFM = "align_to_authoritative_sfm"


class DenseProviderQualificationState(str, Enum):
    UNQUALIFIED = "unqualified"
    CANDIDATE = "candidate"
    QUALIFIED = "qualified"


class DenseProviderRunStatus(str, Enum):
    COMPLETED = "completed"
    DEGRADED = "degraded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DenseProviderProgressStage(str, Enum):
    PREPARING = "preparing"
    VIEW_SELECTION = "view_selection"
    PROVIDER_EXECUTION = "provider_execution"
    ALIGNMENT = "alignment"
    FUSION_FILTERING = "fusion_filtering"
    FINALIZATION = "finalization"


@dataclass(frozen=True, slots=True)
class DenseProviderFamilyContract:
    family: DenseProviderFamily
    native_output: DenseProviderNativeOutput
    frame_binding: DenseProviderFrameBinding
    implementation_authorized: bool = True
    requires_authoritative_sfm: bool = True
    preserves_authoritative_sfm_frame: bool = True
    adaptive_view_selection_required: bool = True
    progressive_status_required: bool = True
    provisional_geometry_contract_supported: bool = True
    permanent_global_view_cap: int | None = I43A_NO_PERMANENT_VIEW_CAP
    establishes_metric_scale: bool = False
    establishes_camera_pose_authority: bool = False
    establishes_measurement_authority: bool = False
    establishes_certified_metrology: bool = False
    generates_surface_mesh: bool = False
    schema_version: int = I43A_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.family, DenseProviderFamily):
            raise TypeError("family must be DenseProviderFamily.")
        if not isinstance(self.native_output, DenseProviderNativeOutput):
            raise TypeError("native_output must be DenseProviderNativeOutput.")
        if not isinstance(self.frame_binding, DenseProviderFrameBinding):
            raise TypeError("frame_binding must be DenseProviderFrameBinding.")

        expected_output = {
            DenseProviderFamily.MVS: DenseProviderNativeOutput.DENSE_GEOMETRY,
            DenseProviderFamily.MULTI_VIEW_AI_DEPTH: DenseProviderNativeOutput.DEPTH_EVIDENCE,
            DenseProviderFamily.TEMPORAL_DEPTH: DenseProviderNativeOutput.DEPTH_EVIDENCE,
            DenseProviderFamily.SINGLE_IMAGE_DEPTH: DenseProviderNativeOutput.DEPTH_EVIDENCE,
            DenseProviderFamily.HYBRID: DenseProviderNativeOutput.COMPOSITE_GEOMETRY,
        }[self.family]
        expected_binding = {
            DenseProviderFamily.MVS: DenseProviderFrameBinding.AUTHORITATIVE_SFM_FRAME,
            DenseProviderFamily.MULTI_VIEW_AI_DEPTH: DenseProviderFrameBinding.ALIGN_TO_AUTHORITATIVE_SFM,
            DenseProviderFamily.TEMPORAL_DEPTH: DenseProviderFrameBinding.ALIGN_TO_AUTHORITATIVE_SFM,
            DenseProviderFamily.SINGLE_IMAGE_DEPTH: DenseProviderFrameBinding.ALIGN_TO_AUTHORITATIVE_SFM,
            DenseProviderFamily.HYBRID: DenseProviderFrameBinding.AUTHORITATIVE_SFM_FRAME,
        }[self.family]
        if self.native_output is not expected_output or self.frame_binding is not expected_binding:
            raise ReconstructionValidationError("Dense-provider family native-output/frame-binding contract drifted.")
        if self.family is DenseProviderFamily.HYBRID and self.implementation_authorized:
            raise ReconstructionValidationError(
                "Hybrid implementation is not authorized by I4.3A; it remains an evidence-gated future experiment."
            )
        if self.family is not DenseProviderFamily.HYBRID and not self.implementation_authorized:
            raise ReconstructionValidationError("Current I4.3 provider families must remain implementation-authorized.")
        if not self.requires_authoritative_sfm or not self.preserves_authoritative_sfm_frame:
            raise ReconstructionValidationError("I4.3 providers must consume/preserve the authoritative SfM frame.")
        if not self.adaptive_view_selection_required:
            raise ReconstructionValidationError("I4.3 production view selection must remain adaptive and explainable.")
        if not self.progressive_status_required or not self.provisional_geometry_contract_supported:
            raise ReconstructionValidationError(
                "I4.3 families must expose progressive status and the provisional-geometry preview contract."
            )
        if self.permanent_global_view_cap is not None:
            raise ReconstructionValidationError("I4.3 forbids a permanent global dense-view cap.")
        forbidden_authority = (
            self.establishes_metric_scale,
            self.establishes_camera_pose_authority,
            self.establishes_measurement_authority,
            self.establishes_certified_metrology,
            self.generates_surface_mesh,
        )
        if any(forbidden_authority):
            raise ReconstructionValidationError(
                "Dense-provider family execution cannot establish metric/camera-pose/measurement/metrology/mesh authority."
            )
        if int(self.schema_version) != I43A_SCHEMA_VERSION:
            raise ReconstructionValidationError("Unsupported I4.3A dense-provider family schema version.")
        object.__setattr__(self, "schema_version", I43A_SCHEMA_VERSION)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "family": self.family.value,
            "native_output": self.native_output.value,
            "frame_binding": self.frame_binding.value,
            "implementation_authorized": self.implementation_authorized,
            "requires_authoritative_sfm": self.requires_authoritative_sfm,
            "preserves_authoritative_sfm_frame": self.preserves_authoritative_sfm_frame,
            "adaptive_view_selection_required": self.adaptive_view_selection_required,
            "progressive_status_required": self.progressive_status_required,
            "provisional_geometry_contract_supported": self.provisional_geometry_contract_supported,
            "permanent_global_view_cap": self.permanent_global_view_cap,
            "establishes_metric_scale": self.establishes_metric_scale,
            "establishes_camera_pose_authority": self.establishes_camera_pose_authority,
            "establishes_measurement_authority": self.establishes_measurement_authority,
            "establishes_certified_metrology": self.establishes_certified_metrology,
            "generates_surface_mesh": self.generates_surface_mesh,
        }


def dense_provider_family_contracts() -> tuple[DenseProviderFamilyContract, ...]:
    return (
        DenseProviderFamilyContract(
            family=DenseProviderFamily.MVS,
            native_output=DenseProviderNativeOutput.DENSE_GEOMETRY,
            frame_binding=DenseProviderFrameBinding.AUTHORITATIVE_SFM_FRAME,
        ),
        DenseProviderFamilyContract(
            family=DenseProviderFamily.MULTI_VIEW_AI_DEPTH,
            native_output=DenseProviderNativeOutput.DEPTH_EVIDENCE,
            frame_binding=DenseProviderFrameBinding.ALIGN_TO_AUTHORITATIVE_SFM,
        ),
        DenseProviderFamilyContract(
            family=DenseProviderFamily.TEMPORAL_DEPTH,
            native_output=DenseProviderNativeOutput.DEPTH_EVIDENCE,
            frame_binding=DenseProviderFrameBinding.ALIGN_TO_AUTHORITATIVE_SFM,
        ),
        DenseProviderFamilyContract(
            family=DenseProviderFamily.SINGLE_IMAGE_DEPTH,
            native_output=DenseProviderNativeOutput.DEPTH_EVIDENCE,
            frame_binding=DenseProviderFrameBinding.ALIGN_TO_AUTHORITATIVE_SFM,
        ),
        DenseProviderFamilyContract(
            family=DenseProviderFamily.HYBRID,
            native_output=DenseProviderNativeOutput.COMPOSITE_GEOMETRY,
            frame_binding=DenseProviderFrameBinding.AUTHORITATIVE_SFM_FRAME,
            implementation_authorized=False,
        ),
    )


_FAMILY_BY_ID = {item.family: item for item in dense_provider_family_contracts()}


@dataclass(frozen=True, slots=True)
class DenseProviderImplementationContract:
    implementation: DenseProviderImplementation
    label: str
    family: DenseProviderFamily
    integration_stage: str
    existing_adapter_expected: bool
    schema_version: int = I43A_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.implementation, DenseProviderImplementation):
            raise TypeError("implementation must be DenseProviderImplementation.")
        label = require_nonempty(self.label, field_name="label")
        if not isinstance(self.family, DenseProviderFamily):
            raise TypeError("family must be DenseProviderFamily.")
        if self.family is DenseProviderFamily.HYBRID:
            raise ReconstructionValidationError("I4.3A does not register a concrete Hybrid provider implementation.")
        stage = require_nonempty(self.integration_stage, field_name="integration_stage")
        if stage not in {"I4.3B", "I4.3C", "I4.3D", "I4.3E"}:
            raise ReconstructionValidationError("Provider implementation integration_stage is outside I4.3B-E.")
        if int(self.schema_version) != I43A_SCHEMA_VERSION:
            raise ReconstructionValidationError("Unsupported I4.3A implementation-contract schema version.")
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "integration_stage", stage)
        object.__setattr__(self, "schema_version", I43A_SCHEMA_VERSION)

    @property
    def family_contract(self) -> DenseProviderFamilyContract:
        return _FAMILY_BY_ID[self.family]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "implementation": self.implementation.value,
            "label": self.label,
            "family": self.family.value,
            "integration_stage": self.integration_stage,
            "existing_adapter_expected": self.existing_adapter_expected,
            "native_output": self.family_contract.native_output.value,
            "frame_binding": self.family_contract.frame_binding.value,
        }


def dense_provider_implementation_contracts() -> tuple[DenseProviderImplementationContract, ...]:
    """Stable I4.3 implementation identities; availability/qualification are runtime facts later."""

    return (
        DenseProviderImplementationContract(
            DenseProviderImplementation.COLMAP_MVS,
            "COLMAP MVS",
            DenseProviderFamily.MVS,
            "I4.3B",
            True,
        ),
        DenseProviderImplementationContract(
            DenseProviderImplementation.DA3,
            "Depth Anything 3",
            DenseProviderFamily.MULTI_VIEW_AI_DEPTH,
            "I4.3C",
            True,
        ),
        DenseProviderImplementationContract(
            DenseProviderImplementation.ROLLING_DEPTH,
            "RollingDepth",
            DenseProviderFamily.TEMPORAL_DEPTH,
            "I4.3D",
            False,
        ),
        DenseProviderImplementationContract(
            DenseProviderImplementation.VIDEO_DEPTH_ANYTHING,
            "Video Depth Anything",
            DenseProviderFamily.TEMPORAL_DEPTH,
            "I4.3D",
            True,
        ),
        DenseProviderImplementationContract(
            DenseProviderImplementation.METRIC3D_V2,
            "Metric3D v2",
            DenseProviderFamily.SINGLE_IMAGE_DEPTH,
            "I4.3E",
            True,
        ),
        DenseProviderImplementationContract(
            DenseProviderImplementation.UNIDEPTH_V2,
            "UniDepthV2",
            DenseProviderFamily.SINGLE_IMAGE_DEPTH,
            "I4.3E",
            True,
        ),
        DenseProviderImplementationContract(
            DenseProviderImplementation.MOGE_2,
            "MoGe-2",
            DenseProviderFamily.SINGLE_IMAGE_DEPTH,
            "I4.3E",
            True,
        ),
        DenseProviderImplementationContract(
            DenseProviderImplementation.DEPTH_PRO,
            "Depth Pro",
            DenseProviderFamily.SINGLE_IMAGE_DEPTH,
            "I4.3E",
            True,
        ),
        DenseProviderImplementationContract(
            DenseProviderImplementation.DEPTH_ANYTHING_V2,
            "Depth Anything V2",
            DenseProviderFamily.SINGLE_IMAGE_DEPTH,
            "I4.3E",
            True,
        ),
        DenseProviderImplementationContract(
            DenseProviderImplementation.MARIGOLD,
            "Marigold",
            DenseProviderFamily.SINGLE_IMAGE_DEPTH,
            "I4.3E",
            True,
        ),
    )


_IMPLEMENTATION_BY_ID = {item.implementation: item for item in dense_provider_implementation_contracts()}


@dataclass(frozen=True, slots=True)
class DenseProviderDescriptor:
    provider_id: str
    implementation: DenseProviderImplementation
    execution: ReconstructionProviderExecution
    backend_version: str
    adapter_version: str
    qualification_state: DenseProviderQualificationState = DenseProviderQualificationState.CANDIDATE
    available: bool = False
    unavailable_reason: str = "adapter_not_bound"
    supports_engineer_override: bool = True
    metadata: Mapping[str, JsonValue] | Sequence[tuple[str, JsonValue]] = field(default_factory=tuple)
    schema_version: int = I43A_SCHEMA_VERSION

    def __post_init__(self) -> None:
        provider_id = require_nonempty(self.provider_id, field_name="provider_id")
        if provider_id == I43A_AUTO_POLICY_ID:
            raise ReconstructionValidationError("The Auto policy ID cannot be registered as an executable provider.")
        if not isinstance(self.implementation, DenseProviderImplementation):
            raise TypeError("implementation must be DenseProviderImplementation.")
        if not isinstance(self.execution, ReconstructionProviderExecution):
            raise TypeError("execution must be ReconstructionProviderExecution.")
        backend_version = require_nonempty(self.backend_version, field_name="backend_version")
        adapter_version = require_nonempty(self.adapter_version, field_name="adapter_version")
        if not isinstance(self.qualification_state, DenseProviderQualificationState):
            raise TypeError("qualification_state must be DenseProviderQualificationState.")
        reason = str(self.unavailable_reason).strip()
        if self.available and reason:
            raise ReconstructionValidationError("Available dense providers must not carry unavailable_reason.")
        if not self.available and not reason:
            raise ReconstructionValidationError("Unavailable dense providers require unavailable_reason.")
        if not isinstance(self.supports_engineer_override, bool):
            raise TypeError("supports_engineer_override must be a bool.")
        if int(self.schema_version) != I43A_SCHEMA_VERSION:
            raise ReconstructionValidationError("Unsupported I4.3A dense-provider descriptor schema version.")
        object.__setattr__(self, "provider_id", provider_id)
        object.__setattr__(self, "backend_version", backend_version)
        object.__setattr__(self, "adapter_version", adapter_version)
        object.__setattr__(self, "unavailable_reason", reason)
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata))
        object.__setattr__(self, "schema_version", I43A_SCHEMA_VERSION)

    @property
    def implementation_contract(self) -> DenseProviderImplementationContract:
        return _IMPLEMENTATION_BY_ID[self.implementation]

    @property
    def family(self) -> DenseProviderFamily:
        return self.implementation_contract.family

    @property
    def production_qualified(self) -> bool:
        return self.qualification_state is DenseProviderQualificationState.QUALIFIED

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "provider_id": self.provider_id,
            "implementation": self.implementation.value,
            "family": self.family.value,
            "execution": self.execution.value,
            "backend_version": self.backend_version,
            "adapter_version": self.adapter_version,
            "qualification_state": self.qualification_state.value,
            "production_qualified": self.production_qualified,
            "available": self.available,
            "unavailable_reason": self.unavailable_reason,
            "supports_engineer_override": self.supports_engineer_override,
            "implementation_contract": self.implementation_contract.as_dict(),
            "metadata": metadata_as_dict(self.metadata),  # type: ignore[arg-type]
        }


@dataclass(frozen=True, slots=True)
class DenseProviderSelectionRequest:
    mode: DenseProviderSelectionMode
    available_view_count: int
    authoritative_sfm_available: bool
    authoritative_sfm_frame_id: str = ""
    requested_provider_id: str = ""
    allow_fallback: bool = True
    engineer_override_provider_id: str = ""
    schema_version: int = I43A_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.mode, DenseProviderSelectionMode):
            raise TypeError("mode must be DenseProviderSelectionMode.")
        view_count = int(self.available_view_count)
        if view_count <= 0:
            raise ReconstructionValidationError("available_view_count must be positive.")
        if not isinstance(self.authoritative_sfm_available, bool):
            raise TypeError("authoritative_sfm_available must be a bool.")
        frame_id = str(self.authoritative_sfm_frame_id).strip()
        if self.authoritative_sfm_available and not frame_id:
            raise ReconstructionValidationError("Available authoritative SfM requires authoritative_sfm_frame_id.")
        if not self.authoritative_sfm_available and frame_id:
            raise ReconstructionValidationError("Unavailable authoritative SfM cannot carry a frame ID.")
        requested = str(self.requested_provider_id).strip()
        override = str(self.engineer_override_provider_id).strip()
        if requested == I43A_AUTO_POLICY_ID or override == I43A_AUTO_POLICY_ID:
            raise ReconstructionValidationError("Concrete provider fields cannot use the Auto policy ID.")
        if self.mode is DenseProviderSelectionMode.AUTO:
            if requested:
                raise ReconstructionValidationError("Auto requests cannot carry requested_provider_id.")
        else:
            if not requested:
                raise ReconstructionValidationError("Explicit-provider requests require requested_provider_id.")
            if override:
                raise ReconstructionValidationError("Explicit-provider requests do not use engineer_override_provider_id.")
        if not isinstance(self.allow_fallback, bool):
            raise TypeError("allow_fallback must be a bool.")
        if int(self.schema_version) != I43A_SCHEMA_VERSION:
            raise ReconstructionValidationError("Unsupported I4.3A selection-request schema version.")
        object.__setattr__(self, "available_view_count", view_count)
        object.__setattr__(self, "authoritative_sfm_frame_id", frame_id)
        object.__setattr__(self, "requested_provider_id", requested)
        object.__setattr__(self, "engineer_override_provider_id", override)
        object.__setattr__(self, "schema_version", I43A_SCHEMA_VERSION)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode.value,
            "available_view_count": self.available_view_count,
            "authoritative_sfm_available": self.authoritative_sfm_available,
            "authoritative_sfm_frame_id": self.authoritative_sfm_frame_id,
            "requested_provider_id": self.requested_provider_id,
            "allow_fallback": self.allow_fallback,
            "engineer_override_provider_id": self.engineer_override_provider_id,
        }


@dataclass(frozen=True, slots=True)
class DenseProviderSelectionDecision:
    request_mode: DenseProviderSelectionMode
    requested_provider_id: str
    selected_provider_id: str
    selected_implementation: DenseProviderImplementation | None
    considered_provider_ids: tuple[str, ...]
    automatic: bool
    engineer_override_applied: bool
    fallback_used: bool
    reason_codes: tuple[str, ...]
    schema_version: int = I43A_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.request_mode, DenseProviderSelectionMode):
            raise TypeError("request_mode must be DenseProviderSelectionMode.")
        requested = str(self.requested_provider_id).strip()
        selected = str(self.selected_provider_id).strip()
        considered = tuple(str(value).strip() for value in self.considered_provider_ids)
        if any(not value for value in considered) or len(set(considered)) != len(considered):
            raise ReconstructionValidationError("considered_provider_ids must be unique and non-empty.")
        if I43A_AUTO_POLICY_ID in considered or requested == I43A_AUTO_POLICY_ID or selected == I43A_AUTO_POLICY_ID:
            raise ReconstructionValidationError("Auto is never an executable/considered provider ID.")
        if self.request_mode is DenseProviderSelectionMode.AUTO and requested:
            raise ReconstructionValidationError("Auto decisions cannot carry requested_provider_id.")
        if self.request_mode is DenseProviderSelectionMode.EXPLICIT_PROVIDER and not requested:
            raise ReconstructionValidationError("Explicit-provider decisions require requested_provider_id.")
        if selected:
            if selected not in considered:
                raise ReconstructionValidationError("Selected provider must appear in considered_provider_ids.")
            if not isinstance(self.selected_implementation, DenseProviderImplementation):
                raise ReconstructionValidationError("Selected decisions require selected_implementation.")
        elif self.selected_implementation is not None:
            raise ReconstructionValidationError("Unselected decisions cannot carry selected_implementation.")
        if self.request_mode is DenseProviderSelectionMode.EXPLICIT_PROVIDER and selected:
            if not self.fallback_used and selected != requested:
                raise ReconstructionValidationError("Explicit-provider selection must match requested_provider_id unless fallback is explicit.")
        expected_automatic = self.request_mode is DenseProviderSelectionMode.AUTO and not bool(self.engineer_override_applied)
        if bool(self.automatic) != expected_automatic:
            raise ReconstructionValidationError(
                "automatic must reflect Auto policy execution and must be false when an engineer override is applied."
            )
        if self.fallback_used and not selected:
            raise ReconstructionValidationError("Fallback cannot be reported without a selected provider.")
        reasons = tuple(str(value).strip() for value in self.reason_codes)
        if any(not value for value in reasons):
            raise ReconstructionValidationError("reason_codes must not contain empty values.")
        if not selected and not reasons:
            raise ReconstructionValidationError("Unselected decisions require an explicit reason code.")
        if int(self.schema_version) != I43A_SCHEMA_VERSION:
            raise ReconstructionValidationError("Unsupported I4.3A selection-decision schema version.")
        object.__setattr__(self, "requested_provider_id", requested)
        object.__setattr__(self, "selected_provider_id", selected)
        object.__setattr__(self, "considered_provider_ids", considered)
        object.__setattr__(self, "engineer_override_applied", bool(self.engineer_override_applied))
        object.__setattr__(self, "fallback_used", bool(self.fallback_used))
        object.__setattr__(self, "reason_codes", reasons)
        object.__setattr__(self, "schema_version", I43A_SCHEMA_VERSION)

    def as_dict(self) -> dict[str, Any]:
        family = (
            _IMPLEMENTATION_BY_ID[self.selected_implementation].family.value
            if self.selected_implementation is not None
            else ""
        )
        return {
            "schema_version": self.schema_version,
            "request_mode": self.request_mode.value,
            "requested_provider_id": self.requested_provider_id,
            "selected_provider_id": self.selected_provider_id,
            "selected_implementation": self.selected_implementation.value if self.selected_implementation else "",
            "selected_family": family,
            "considered_provider_ids": list(self.considered_provider_ids),
            "automatic": self.automatic,
            "engineer_override_applied": self.engineer_override_applied,
            "fallback_used": self.fallback_used,
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True, slots=True)
class DenseProviderProgressEvent:
    """UI-neutral progress snapshot; provisional geometry is never final authority."""

    provider_id: str
    implementation: DenseProviderImplementation
    stage: DenseProviderProgressStage
    stage_detail: str
    progress_fraction: float
    processed_views: int
    total_views: int
    provisional_geometry_available: bool = False
    provisional_geometry_ref: str = ""
    preview_revision: int = 0
    gpu_vram_used_mib: float | None = None
    process_ram_used_mib: float | None = None
    degraded_reason: str = ""
    fallback_reason: str = ""
    provisional_geometry_authoritative: bool = False
    metric_scale_claimed: bool = False
    measurement_authoritative: bool = False
    certified_metrology: bool = False
    surface_mesh_generated: bool = False
    schema_version: int = I43A_SCHEMA_VERSION

    def __post_init__(self) -> None:
        provider_id = require_nonempty(self.provider_id, field_name="provider_id")
        if provider_id == I43A_AUTO_POLICY_ID:
            raise ReconstructionValidationError("Auto cannot emit provider progress as an executable backend.")
        if not isinstance(self.implementation, DenseProviderImplementation):
            raise TypeError("implementation must be DenseProviderImplementation.")
        if not isinstance(self.stage, DenseProviderProgressStage):
            raise TypeError("stage must be DenseProviderProgressStage.")
        detail = require_nonempty(self.stage_detail, field_name="stage_detail")
        progress = float(self.progress_fraction)
        if progress < 0.0 or progress > 1.0:
            raise ReconstructionValidationError("progress_fraction must be within [0, 1].")
        processed = int(self.processed_views)
        total = int(self.total_views)
        if processed < 0 or total < 0 or processed > total:
            raise ReconstructionValidationError("processed_views/total_views are invalid.")
        preview_ref = str(self.provisional_geometry_ref).strip()
        revision = int(self.preview_revision)
        if revision < 0:
            raise ReconstructionValidationError("preview_revision must be non-negative.")
        if self.provisional_geometry_available:
            if not preview_ref or revision <= 0:
                raise ReconstructionValidationError(
                    "Available provisional geometry requires a reference and a positive preview_revision."
                )
        elif preview_ref:
            raise ReconstructionValidationError("Unavailable provisional geometry cannot carry a geometry reference.")
        for field_name, value in (
            ("gpu_vram_used_mib", self.gpu_vram_used_mib),
            ("process_ram_used_mib", self.process_ram_used_mib),
        ):
            if value is not None and float(value) < 0:
                raise ReconstructionValidationError(f"{field_name} must be non-negative when provided.")
        if self.provisional_geometry_authoritative:
            raise ReconstructionValidationError("Progressive preview geometry is provisional and cannot be authoritative.")
        if any((self.metric_scale_claimed, self.measurement_authoritative, self.certified_metrology, self.surface_mesh_generated)):
            raise ReconstructionValidationError(
                "Progress events cannot establish metric/measurement/metrology/mesh authority."
            )
        if int(self.schema_version) != I43A_SCHEMA_VERSION:
            raise ReconstructionValidationError("Unsupported I4.3A progress-event schema version.")
        object.__setattr__(self, "provider_id", provider_id)
        object.__setattr__(self, "stage_detail", detail)
        object.__setattr__(self, "progress_fraction", progress)
        object.__setattr__(self, "processed_views", processed)
        object.__setattr__(self, "total_views", total)
        object.__setattr__(self, "provisional_geometry_ref", preview_ref)
        object.__setattr__(self, "preview_revision", revision)
        object.__setattr__(self, "degraded_reason", str(self.degraded_reason).strip())
        object.__setattr__(self, "fallback_reason", str(self.fallback_reason).strip())
        object.__setattr__(self, "schema_version", I43A_SCHEMA_VERSION)

    @property
    def family(self) -> DenseProviderFamily:
        return _IMPLEMENTATION_BY_ID[self.implementation].family

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "provider_id": self.provider_id,
            "implementation": self.implementation.value,
            "family": self.family.value,
            "stage": self.stage.value,
            "stage_detail": self.stage_detail,
            "progress_fraction": self.progress_fraction,
            "processed_views": self.processed_views,
            "total_views": self.total_views,
            "provisional_geometry_available": self.provisional_geometry_available,
            "provisional_geometry_ref": self.provisional_geometry_ref,
            "preview_revision": self.preview_revision,
            "resource_usage": {
                "gpu_vram_used_mib": self.gpu_vram_used_mib,
                "process_ram_used_mib": self.process_ram_used_mib,
            },
            "degraded_reason": self.degraded_reason,
            "fallback_reason": self.fallback_reason,
            "authority": {
                "provisional_geometry_authoritative": self.provisional_geometry_authoritative,
                "metric_scale_claimed": self.metric_scale_claimed,
                "measurement_authoritative": self.measurement_authoritative,
                "certified_metrology": self.certified_metrology,
                "surface_mesh_generated": self.surface_mesh_generated,
            },
        }


@dataclass(frozen=True, slots=True)
class DenseProviderExecutionResult:
    provider_id: str
    implementation: DenseProviderImplementation
    status: DenseProviderRunStatus
    source_view_count: int
    selected_view_count: int
    adaptive_view_selection: bool
    view_selection_reason: str
    permanent_global_view_cap: int | None
    canonical_geometry_ref: str
    canonical_frame_id: str
    native_artifact_refs: tuple[str, ...] = ()
    degradation_reasons: tuple[str, ...] = ()
    failure_reason: str = ""
    progress_event_count: int = 0
    final_preview_revision: int = 0
    canonical_geometry_authoritative: bool = True
    single_authoritative_sfm_frame_preserved: bool = True
    camera_pose_authority_owner: str = I43A_AUTHORITATIVE_FRAME_OWNER
    metric_scale_claimed: bool = False
    measurement_authoritative: bool = False
    certified_metrology: bool = False
    surface_mesh_generated: bool = False
    production_filter_policy_changed: bool = False
    provider_promotion_performed: bool = False
    schema_version: int = I43A_SCHEMA_VERSION

    def __post_init__(self) -> None:
        provider_id = require_nonempty(self.provider_id, field_name="provider_id")
        if provider_id == I43A_AUTO_POLICY_ID:
            raise ReconstructionValidationError("Auto cannot produce a dense-provider execution result.")
        if not isinstance(self.implementation, DenseProviderImplementation):
            raise TypeError("implementation must be DenseProviderImplementation.")
        if not isinstance(self.status, DenseProviderRunStatus):
            raise TypeError("status must be DenseProviderRunStatus.")
        source_count = int(self.source_view_count)
        selected_count = int(self.selected_view_count)
        if source_count <= 0 or selected_count < 0 or selected_count > source_count:
            raise ReconstructionValidationError("Dense-provider source/selected view counts are invalid.")
        if self.status in {DenseProviderRunStatus.COMPLETED, DenseProviderRunStatus.DEGRADED} and selected_count <= 0:
            raise ReconstructionValidationError("Successful/degraded dense-provider results require selected views.")
        if self.permanent_global_view_cap is not None:
            raise ReconstructionValidationError("I4.3 forbids a permanent global dense-view cap.")
        completed_like = self.status in {DenseProviderRunStatus.COMPLETED, DenseProviderRunStatus.DEGRADED}
        selection_reason = str(self.view_selection_reason).strip()
        if completed_like and selected_count < source_count:
            if not self.adaptive_view_selection or not selection_reason:
                raise ReconstructionValidationError(
                    "Reduced view use must be adaptive and carry an engineer-visible selection reason."
                )
        canonical_ref = str(self.canonical_geometry_ref).strip()
        frame_id = str(self.canonical_frame_id).strip()
        if completed_like and (not canonical_ref or not frame_id):
            raise ReconstructionValidationError("Completed/degraded provider results require canonical geometry/frame references.")
        if not completed_like and (canonical_ref or frame_id or self.canonical_geometry_authoritative):
            raise ReconstructionValidationError("Failed/cancelled provider results cannot claim canonical geometry.")
        if completed_like and (not self.canonical_geometry_authoritative or not self.single_authoritative_sfm_frame_preserved):
            raise ReconstructionValidationError("Completed dense geometry must preserve canonical authority in the SfM frame.")
        owner = str(self.camera_pose_authority_owner).strip()
        if owner != I43A_AUTHORITATIVE_FRAME_OWNER:
            raise ReconstructionValidationError("Dense providers cannot replace PyCOLMAP SfM camera-pose authority.")
        forbidden = (
            self.metric_scale_claimed,
            self.measurement_authoritative,
            self.certified_metrology,
            self.surface_mesh_generated,
            self.production_filter_policy_changed,
            self.provider_promotion_performed,
        )
        if any(forbidden):
            raise ReconstructionValidationError(
                "I4.3A dense-provider execution cannot establish metric/measurement/metrology/mesh authority, "
                "change production filtering, or promote a provider."
            )
        native_refs = tuple(str(value).strip() for value in self.native_artifact_refs)
        if any(not value for value in native_refs):
            raise ReconstructionValidationError("native_artifact_refs must not contain empty values.")
        degradation = tuple(str(value).strip() for value in self.degradation_reasons)
        if any(not value for value in degradation):
            raise ReconstructionValidationError("degradation_reasons must not contain empty values.")
        failure = str(self.failure_reason).strip()
        if self.status is DenseProviderRunStatus.DEGRADED and not degradation:
            raise ReconstructionValidationError("Degraded provider results require degradation_reasons.")
        if self.status in {DenseProviderRunStatus.FAILED, DenseProviderRunStatus.CANCELLED} and not failure:
            raise ReconstructionValidationError("Failed/cancelled provider results require failure_reason.")
        if completed_like and failure:
            raise ReconstructionValidationError("Completed/degraded provider results cannot carry failure_reason.")
        progress_count = int(self.progress_event_count)
        final_revision = int(self.final_preview_revision)
        if progress_count < 0 or final_revision < 0:
            raise ReconstructionValidationError("Progress-event count/revision must be non-negative.")
        if final_revision > 0 and progress_count <= 0:
            raise ReconstructionValidationError("A final preview revision requires at least one progress event.")
        if int(self.schema_version) != I43A_SCHEMA_VERSION:
            raise ReconstructionValidationError("Unsupported I4.3A execution-result schema version.")
        object.__setattr__(self, "provider_id", provider_id)
        object.__setattr__(self, "source_view_count", source_count)
        object.__setattr__(self, "selected_view_count", selected_count)
        object.__setattr__(self, "adaptive_view_selection", bool(self.adaptive_view_selection))
        object.__setattr__(self, "view_selection_reason", selection_reason)
        object.__setattr__(self, "canonical_geometry_ref", canonical_ref)
        object.__setattr__(self, "canonical_frame_id", frame_id)
        object.__setattr__(self, "native_artifact_refs", native_refs)
        object.__setattr__(self, "degradation_reasons", degradation)
        object.__setattr__(self, "failure_reason", failure)
        object.__setattr__(self, "progress_event_count", progress_count)
        object.__setattr__(self, "final_preview_revision", final_revision)
        object.__setattr__(self, "camera_pose_authority_owner", owner)
        object.__setattr__(self, "schema_version", I43A_SCHEMA_VERSION)

    @property
    def family(self) -> DenseProviderFamily:
        return _IMPLEMENTATION_BY_ID[self.implementation].family

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "provider_id": self.provider_id,
            "implementation": self.implementation.value,
            "provider_family": self.family.value,
            "status": self.status.value,
            "source_view_count": self.source_view_count,
            "selected_view_count": self.selected_view_count,
            "adaptive_view_selection": self.adaptive_view_selection,
            "view_selection_reason": self.view_selection_reason,
            "permanent_global_view_cap": self.permanent_global_view_cap,
            "canonical_geometry_ref": self.canonical_geometry_ref,
            "canonical_frame_id": self.canonical_frame_id,
            "native_artifact_refs": list(self.native_artifact_refs),
            "degradation_reasons": list(self.degradation_reasons),
            "failure_reason": self.failure_reason,
            "progress_event_count": self.progress_event_count,
            "final_preview_revision": self.final_preview_revision,
            "authority": {
                "canonical_geometry_authoritative": self.canonical_geometry_authoritative,
                "single_authoritative_sfm_frame_preserved": self.single_authoritative_sfm_frame_preserved,
                "camera_pose_authority_owner": self.camera_pose_authority_owner,
                "metric_scale_claimed": self.metric_scale_claimed,
                "measurement_authoritative": self.measurement_authoritative,
                "certified_metrology": self.certified_metrology,
                "surface_mesh_generated": self.surface_mesh_generated,
                "production_filter_policy_changed": self.production_filter_policy_changed,
                "provider_promotion_performed": self.provider_promotion_performed,
            },
        }


class DenseProviderAdapter(Protocol):
    """Interface I4.3B-E adapters must implement; I4.3A executes nothing."""

    @property
    def descriptor(self) -> DenseProviderDescriptor: ...

    def execute(
        self,
        request: DenseProviderSelectionRequest,
        decision: DenseProviderSelectionDecision,
    ) -> DenseProviderExecutionResult: ...


class DenseProviderProgressSink(Protocol):
    """UI-neutral sink used later by worker/controller bridges for live preview."""

    def publish(self, event: DenseProviderProgressEvent) -> None: ...


def validate_dense_provider_catalog(providers: Sequence[DenseProviderDescriptor]) -> tuple[DenseProviderDescriptor, ...]:
    values = tuple(providers)
    if not values:
        raise ReconstructionValidationError("Dense-provider catalog requires at least one concrete provider.")
    if any(not isinstance(value, DenseProviderDescriptor) for value in values):
        raise TypeError("Dense-provider catalog must contain DenseProviderDescriptor values.")
    ids = tuple(value.provider_id for value in values)
    if len(set(ids)) != len(ids):
        raise ReconstructionValidationError("Dense-provider IDs must be globally unique.")
    implementations = tuple(value.implementation for value in values)
    if len(set(implementations)) != len(implementations):
        raise ReconstructionValidationError("I4.3A allows one canonical provider descriptor per implementation identity.")
    return tuple(sorted(values, key=lambda item: (item.family.value, item.implementation.value, item.provider_id)))


def validate_i4_3a_authorization(closure: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the persisted I4.2F gate before any I4.3 architecture work."""

    if not isinstance(closure, Mapping):
        raise TypeError("I4.2F closure must be a mapping.")
    if closure.get("phase") != "I4.2F" or closure.get("status") != "pass" or closure.get("pass") is not True:
        raise ReconstructionValidationError("I4.3A requires a passing I4.2F closure.")
    decision = closure.get("decision")
    if not isinstance(decision, Mapping):
        raise ReconstructionValidationError("I4.2F closure is missing its decision record.")
    required_decisions = {
        "i4_2_closed": True,
        "i4_2f_closed": True,
        "i4_3_provider_architecture_authorized": True,
        "dense_provider_promoted": False,
        "production_d12_policy_change_authorized": False,
    }
    for key, expected in required_decisions.items():
        if decision.get(key) is not expected:
            raise ReconstructionValidationError(f"I4.3A authorization gate failed: decision.{key} must be {expected}.")
    authority = closure.get("authority")
    if not isinstance(authority, Mapping):
        raise ReconstructionValidationError("I4.2F closure is missing its authority record.")
    for key in (
        "authoritative_sparse_modified",
        "bundle_adjustment_rerun",
        "certified_metrology",
        "dense_provider_promoted",
        "measurement_authoritative",
        "metric_scale_claimed",
        "patch_match_rerun",
        "pose_refinement",
        "production_filter_policy_changed",
        "production_manifold_applied_to_authoritative_scene",
        "sfm_rerun",
        "stereo_fusion_rerun",
        "surface_mesh_generated",
    ):
        if authority.get(key) is not False:
            raise ReconstructionValidationError(f"I4.3A authorization gate failed: authority.{key} must remain false.")
    conclusion = closure.get("scientific_conclusion")
    if not isinstance(conclusion, Mapping):
        raise ReconstructionValidationError("I4.2F closure is missing its scientific conclusion.")
    if conclusion.get("temple_ring_detail_preservation_qualified") is not True:
        raise ReconstructionValidationError("I4.3A requires qualified TempleRing detail preservation.")
    if conclusion.get("temple_ring_driven_d12_policy_correction_required") is not False:
        raise ReconstructionValidationError("I4.3A requires the TempleRing d12 correction question to be closed.")
    return {
        "i4_2_closed": True,
        "i4_2f_closed": True,
        "i4_3_provider_architecture_authorized": True,
        "d12_safe_on_temple_ring_within_tested_transfer": bool(
            conclusion.get("d12_sparse_safe_on_temple_ring_within_tested_transfer", False)
        ),
        "global_universal_detail_safety_claimed": bool(conclusion.get("global_universal_detail_safety_claimed", False)),
    }


def read_i4_2f_closure(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError(f"Unable to read I4.2F closure: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError("I4.2F closure JSON must be an object.")
    return payload


def build_i4_3a_architecture_snapshot(*, i4_2f_closure: Mapping[str, Any]) -> dict[str, Any]:
    authorization = validate_i4_3a_authorization(i4_2f_closure)
    families = dense_provider_family_contracts()
    implementations = dense_provider_implementation_contracts()
    return {
        "schema_version": I43A_SCHEMA_VERSION,
        "phase": I43A_PHASE,
        "status": "pass",
        "pass": True,
        "authorization": authorization,
        "architecture": {
            "selection_modes": [mode.value for mode in DenseProviderSelectionMode],
            "provider_families": [family.value for family in DenseProviderFamily],
            "provider_implementations": [item.value for item in DenseProviderImplementation],
            "auto_is_selection_policy_only": True,
            "auto_is_executable_provider": False,
            "explicit_selection_targets_concrete_provider": True,
            "authoritative_camera_frame_owner": I43A_AUTHORITATIVE_FRAME_OWNER,
            "single_authoritative_sfm_frame_required": True,
            "provider_family_contracts": [item.as_dict() for item in families],
            "provider_implementation_contracts": [item.as_dict() for item in implementations],
            "permanent_global_view_cap": None,
            "adaptive_engineer_visible_view_selection_required": True,
            "engineer_override_supported_by_contract": True,
            "fallback_must_be_explicit_in_decision": True,
            "failure_and_degradation_are_distinct": True,
            "scenario_specific_qualification_required": True,
            "universal_provider_winner_assumed": False,
        },
        "progressive_preview_contract": {
            "required_for_v1_0": True,
            "ui_implemented_in_i4_3a": False,
            "events_are_ui_neutral": True,
            "progress_stages": [stage.value for stage in DenseProviderProgressStage],
            "processed_total_views_supported": True,
            "resource_usage_supported": True,
            "provisional_geometry_supported": True,
            "provisional_geometry_is_replaceable": True,
            "provisional_geometry_authoritative": False,
            "preview_revision_required_when_geometry_available": True,
            "final_canonical_geometry_remains_separate": True,
        },
        "qualification_plan": {
            "i4_3g_required": True,
            "complete_room_mvs_vs_da3_required": True,
            "temple_ring_mvs_vs_ai_where_useful": True,
            "house_mvs_vs_da3_evidence_retained": True,
            "temporal_video_comparison_required": True,
            "low_texture_comparison_required": True,
            "weak_overlap_comparison_required": True,
            "hybrid_family_defined_as_future_hypothesis": True,
            "hybrid_implementation_authorized": False,
            "hybrid_requires_new_evidence": True,
        },
        "i4_3_roadmap": [
            "I4.3B MVS production provider adapter",
            "I4.3C DA3 production provider adapter",
            "I4.3D RollingDepth + Video Depth Anything temporal providers",
            "I4.3E remaining installed depth-provider adapters/qualification",
            "I4.3F adaptive frame/view selection",
            "I4.3G controlled cross-provider scenario qualification including complete-room MVS vs DA3",
            "I4.3H provider architecture closure",
        ],
        "authority": {
            "backend_started": False,
            "provider_selected": False,
            "provider_promoted": False,
            "auto_policy_implemented": False,
            "hybrid_implementation_authorized": False,
            "production_filter_policy_changed": False,
            "authoritative_sparse_modified": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
        },
        "next": (
            "I4.3B MVS production provider adapter may bind the already-qualified COLMAP/PyCOLMAP MVS path "
            "behind these contracts. Auto policy, DA3/temporal/single-image bindings, UI wiring, and hybrid "
            "experimentation remain deferred to their explicit later I4.3/I5 steps."
        ),
    }


def write_i4_3a_architecture_snapshot(path: str | Path, snapshot: Mapping[str, Any]) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(snapshot), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


__all__ = [
    "I43A_SCHEMA_VERSION",
    "I43A_PHASE",
    "I43A_AUTO_POLICY_ID",
    "I43A_AUTHORITATIVE_FRAME_OWNER",
    "DenseProviderSelectionMode",
    "DenseProviderFamily",
    "DenseProviderImplementation",
    "DenseProviderNativeOutput",
    "DenseProviderFrameBinding",
    "DenseProviderQualificationState",
    "DenseProviderRunStatus",
    "DenseProviderProgressStage",
    "DenseProviderFamilyContract",
    "DenseProviderImplementationContract",
    "DenseProviderDescriptor",
    "DenseProviderSelectionRequest",
    "DenseProviderSelectionDecision",
    "DenseProviderProgressEvent",
    "DenseProviderExecutionResult",
    "DenseProviderAdapter",
    "DenseProviderProgressSink",
    "dense_provider_family_contracts",
    "dense_provider_implementation_contracts",
    "validate_dense_provider_catalog",
    "validate_i4_3a_authorization",
    "read_i4_2f_closure",
    "build_i4_3a_architecture_snapshot",
    "write_i4_3a_architecture_snapshot",
]

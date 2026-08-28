"""I4.3F2 production wiring for adaptive pre-SfM and dense-view planning.

This module bridges the execution-free I4.3F1 planner into production without
selecting a provider, changing PyCOLMAP authority, or introducing a permanent
view cap.  Resource-derived targets are per-run planning decisions only.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
from math import floor, isfinite
from typing import Any, Mapping, Sequence

from core.contracts.validation import JsonValue, metadata_as_dict, normalize_metadata, require_nonempty
from core.reconstruction.adaptive_view_selection import (
    AdaptiveSelectionResourceSnapshot,
    AdaptiveViewSelectionOverride,
    AdaptiveViewSelectionPlan,
    build_adaptive_view_selection_plan,
)
from core.reconstruction.dense_provider_architecture import (
    DenseProviderDescriptor,
    DenseProviderExecutionResult,
    DenseProviderSelectionDecision,
    DenseProviderSelectionMode,
    DenseProviderSelectionRequest,
    I43A_AUTHORITATIVE_FRAME_OWNER,
)
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.sfm_contracts import (
    OfflineSfmResult,
    SfmLandmarkEstimate,
    SfmStageStatistics,
    SfmWorkspaceManifest,
)

I43F2_SCHEMA_VERSION = 1
I43F2_PHASE = "I4.3F2"
I43F2_PERMANENT_GLOBAL_VIEW_CAP = None
# Transparent planning coefficient only; not a measured runtime/memory claim.
I43F2_PRE_SFM_WORK_UNITS_PER_FRAME = 4.0


@dataclass(frozen=True, slots=True)
class AdaptivePreSfmFramePlan:
    """Provider-neutral pre-SfM frame budget for one concrete source/run."""

    source_kind: str
    estimated_total_frames: int | None
    expected_frame_rate_hz: float | None
    target_sampled_frames: int | None
    resource_target_frames: int | None
    engineer_max_keyframes_override: int | None
    adaptive_reduction_expected: bool
    engineer_override_applied: bool
    reason_codes: tuple[str, ...]
    planning_capacity_units: float | None
    resources: AdaptiveSelectionResourceSnapshot
    permanent_global_view_cap: int | None = I43F2_PERMANENT_GLOBAL_VIEW_CAP
    schema_version: int = I43F2_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_kind", require_nonempty(self.source_kind, field_name="source_kind"))
        total = self.estimated_total_frames
        if total is not None:
            if isinstance(total, bool) or int(total) <= 0:
                raise ReconstructionValidationError("estimated_total_frames must be positive when known.")
            object.__setattr__(self, "estimated_total_frames", int(total))
        fps = self.expected_frame_rate_hz
        if fps is not None:
            value = float(fps)
            if not isfinite(value) or value <= 0.0:
                raise ReconstructionValidationError("expected_frame_rate_hz must be finite and positive when known.")
            object.__setattr__(self, "expected_frame_rate_hz", value)
        for name in ("target_sampled_frames", "resource_target_frames", "engineer_max_keyframes_override"):
            value = getattr(self, name)
            if value is not None:
                if isinstance(value, bool) or int(value) < 3:
                    raise ReconstructionValidationError(f"{name} must be at least 3 when present.")
                object.__setattr__(self, name, int(value))
        if self.target_sampled_frames is not None and total is not None and self.target_sampled_frames > total:
            raise ReconstructionValidationError("target_sampled_frames cannot exceed a known source-frame count.")
        if self.permanent_global_view_cap is not None:
            raise ReconstructionValidationError("I4.3F2 forbids a permanent global frame/view cap.")
        if not isinstance(self.resources, AdaptiveSelectionResourceSnapshot):
            raise TypeError("resources must be AdaptiveSelectionResourceSnapshot.")
        if not isinstance(self.adaptive_reduction_expected, bool) or not isinstance(self.engineer_override_applied, bool):
            raise TypeError("adaptive_reduction_expected/engineer_override_applied must be bool values.")
        reasons = tuple(str(item).strip() for item in self.reason_codes)
        if not reasons or any(not item for item in reasons):
            raise ReconstructionValidationError("Adaptive pre-SfM plans require engineer-visible reason codes.")
        if int(self.schema_version) != I43F2_SCHEMA_VERSION:
            raise ReconstructionValidationError("Unsupported I4.3F2 pre-SfM plan schema version.")
        object.__setattr__(self, "reason_codes", reasons)
        object.__setattr__(self, "schema_version", I43F2_SCHEMA_VERSION)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_kind": self.source_kind,
            "estimated_total_frames": self.estimated_total_frames,
            "expected_frame_rate_hz": self.expected_frame_rate_hz,
            "target_sampled_frames": self.target_sampled_frames,
            "resource_target_frames": self.resource_target_frames,
            "engineer_max_keyframes_override": self.engineer_max_keyframes_override,
            "adaptive_reduction_expected": self.adaptive_reduction_expected,
            "engineer_override_applied": self.engineer_override_applied,
            "reason_codes": list(self.reason_codes),
            "planning_capacity_units": self.planning_capacity_units,
            "planning_work_units_per_frame": I43F2_PRE_SFM_WORK_UNITS_PER_FRAME,
            "planning_capacity_is_empirical_runtime_claim": False,
            "resources": self.resources.as_dict(),
            "permanent_global_view_cap": None,
        }


def build_adaptive_pre_sfm_frame_plan(
    *,
    source_kind: str,
    estimated_total_frames: int | None,
    expected_frame_rate_hz: float | None,
    resources: AdaptiveSelectionResourceSnapshot,
    engineer_max_keyframes_override: int | None = None,
) -> AdaptivePreSfmFramePlan:
    """Resolve one provider-neutral frame target before SfM.

    The resource coefficient is deliberately transparent and conservative.  It
    is not a benchmark, and it never becomes a permanent application cap.
    Explicit historical ``max_keyframes`` values remain supported only as an
    engineer override and are recorded as such.
    """

    if not isinstance(resources, AdaptiveSelectionResourceSnapshot):
        raise TypeError("resources must be AdaptiveSelectionResourceSnapshot.")
    kind = require_nonempty(source_kind, field_name="source_kind")
    total = estimated_total_frames
    if total is not None:
        if isinstance(total, bool) or int(total) <= 0:
            raise ReconstructionValidationError("estimated_total_frames must be positive when known.")
        total = int(total)
    fps = expected_frame_rate_hz
    if fps is not None:
        fps = float(fps)
        if not isfinite(fps) or fps <= 0.0:
            raise ReconstructionValidationError("expected_frame_rate_hz must be finite and positive when known.")

    capacity = resources.planning_capacity_units()
    resource_target: int | None = None
    if capacity is not None:
        resource_target = max(3, int(floor(capacity / I43F2_PRE_SFM_WORK_UNITS_PER_FRAME)))

    override = engineer_max_keyframes_override
    reasons: list[str] = ["adaptive_pre_sfm_planning_evaluated", "permanent_global_view_cap_none"]
    if override is not None:
        if isinstance(override, bool) or int(override) < 3:
            raise ReconstructionValidationError("Engineer max-keyframe override must be at least 3.")
        override = int(override)
        target = override if total is None else min(total, override)
        reasons.extend(("engineer_max_keyframes_override_applied", "source_span_sampling_required_when_reduced"))
        engineer_override = True
    else:
        engineer_override = False
        if total is None:
            target = resource_target
        elif resource_target is None or total <= resource_target:
            target = total
        else:
            target = resource_target
        if total is not None and target == total:
            reasons.append("all_source_frames_fit_pre_sfm_resource_budget")
        elif target is not None:
            reasons.extend(("resource_pressure_adaptive_pre_sfm_reduction", "source_span_sampling_required_when_reduced"))
        else:
            reasons.append("resource_capacity_unknown_no_fixed_pre_sfm_cap")

    reduction = bool(total is not None and target is not None and target < total)
    return AdaptivePreSfmFramePlan(
        source_kind=kind,
        estimated_total_frames=total,
        expected_frame_rate_hz=fps,
        target_sampled_frames=target,
        resource_target_frames=resource_target,
        engineer_max_keyframes_override=override,
        adaptive_reduction_expected=reduction,
        engineer_override_applied=engineer_override,
        reason_codes=tuple(dict.fromkeys(reasons)),
        planning_capacity_units=capacity,
        resources=resources,
        permanent_global_view_cap=None,
    )


def _sfm_authority_digest(result: OfflineSfmResult) -> str:
    payload = {
        "session_id": result.session_id,
        "backend_id": result.backend_id,
        "backend_version": result.backend_version,
        "gauge": result.gauge.as_dict(),
        "cameras": [item.as_dict() for item in result.cameras],
        "camera_models": [item.as_dict() for item in result.camera_models],
        "landmarks": [item.as_dict() for item in result.landmarks],
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def project_authoritative_sfm_for_provider(
    sfm_result: OfflineSfmResult,
    plan: AdaptiveViewSelectionPlan,
) -> OfflineSfmResult:
    """Create an immutable selected-view projection of authoritative SfM.

    Camera transforms/landmark coordinates are copied unchanged.  Landmarks are
    filtered only because the generic SfM contract requires every observation to
    reference a workspace image and every retained landmark to have >=2 selected
    observations.  The original result object and its workspace files are never
    changed.
    """

    if not isinstance(sfm_result, OfflineSfmResult):
        raise TypeError("sfm_result must be OfflineSfmResult.")
    if not isinstance(plan, AdaptiveViewSelectionPlan):
        raise TypeError("plan must be AdaptiveViewSelectionPlan.")
    registered = {camera.keyframe_id for camera in sfm_result.cameras}
    if set(plan.source_order_keyframe_ids) != registered:
        raise ReconstructionValidationError("I4.3F2 plan does not cover the authoritative registered-view set exactly.")
    selected = set(plan.selected_keyframe_ids)
    if not selected:
        raise ReconstructionValidationError("I4.3F2 provider projection selected no registered views.")
    if not plan.reduction_applied:
        return sfm_result

    workspace_by_id = {item.keyframe_id: item for item in sfm_result.workspace.images}
    ordered_images = [workspace_by_id[key] for key in plan.selected_keyframe_ids]
    projected_images = tuple(replace(item, order=index) for index, item in enumerate(ordered_images))
    projected_workspace = SfmWorkspaceManifest(
        session_id=sfm_result.workspace.session_id,
        source=sfm_result.workspace.source,
        workspace_root=sfm_result.workspace.workspace_root,
        images_directory=sfm_result.workspace.images_directory,
        database_path=sfm_result.workspace.database_path,
        sparse_directory=sfm_result.workspace.sparse_directory,
        checkpoints_directory=sfm_result.workspace.checkpoints_directory,
        reports_directory=sfm_result.workspace.reports_directory,
        logs_directory=sfm_result.workspace.logs_directory,
        images=projected_images,
    )
    camera_by_id = {item.keyframe_id: item for item in sfm_result.cameras}
    projected_cameras = tuple(camera_by_id[key] for key in plan.selected_keyframe_ids)
    camera_ids = {item.camera_id for item in projected_cameras}
    projected_camera_models = tuple(item for item in sfm_result.camera_models if item.camera_id in camera_ids)

    projected_landmarks: list[SfmLandmarkEstimate] = []
    for landmark in sfm_result.landmarks:
        observations = tuple(item for item in landmark.observations if item.keyframe_id in selected)
        if len(observations) < 2:
            continue
        projected_landmarks.append(replace(landmark, observations=observations))
    observation_count = sum(len(item.observations) for item in projected_landmarks)
    metadata = metadata_as_dict(sfm_result.metadata)  # type: ignore[arg-type]
    metadata["i4_3f2_provider_input_projection"] = {
        "projection_only": True,
        "authoritative_sfm_modified": False,
        "original_authority_sha256": _sfm_authority_digest(sfm_result),
        "original_registered_views": sfm_result.statistics.registered_keyframes,
        "selected_registered_views": len(projected_cameras),
        "selected_keyframe_ids": list(plan.selected_keyframe_ids),
        "selection_reason_codes": list(plan.reason_codes),
        "permanent_global_view_cap": None,
        "camera_pose_authority_owner": I43A_AUTHORITATIVE_FRAME_OWNER,
    }
    return OfflineSfmResult(
        session_id=sfm_result.session_id,
        backend_id=sfm_result.backend_id,
        backend_version=sfm_result.backend_version,
        gauge=sfm_result.gauge,
        cameras=projected_cameras,
        camera_models=projected_camera_models,
        landmarks=tuple(projected_landmarks),
        statistics=SfmStageStatistics(
            input_keyframes=len(projected_images),
            registered_keyframes=len(projected_cameras),
            landmarks=len(projected_landmarks),
            observations=observation_count,
            elapsed_ns=sfm_result.statistics.elapsed_ns,
            mean_reprojection_error_px=sfm_result.statistics.mean_reprojection_error_px,
        ),
        workspace=projected_workspace,
        warnings=sfm_result.warnings,
        metadata=metadata,
    )


@dataclass(frozen=True, slots=True)
class AdaptiveProviderExecutionPreparation:
    provider: DenseProviderDescriptor
    reconstruction_frame_id: str
    original_sfm_result: OfflineSfmResult
    provider_sfm_result: OfflineSfmResult
    plan: AdaptiveViewSelectionPlan
    selection_request: DenseProviderSelectionRequest
    selection_decision: DenseProviderSelectionDecision
    original_authority_sha256: str
    authority_metadata: Mapping[str, JsonValue] | Sequence[tuple[str, JsonValue]] = field(default_factory=tuple)
    schema_version: int = I43F2_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.provider, DenseProviderDescriptor):
            raise TypeError("provider must be DenseProviderDescriptor.")
        frame_id = require_nonempty(self.reconstruction_frame_id, field_name="reconstruction_frame_id")
        if self.provider.implementation is not self.plan.implementation:
            raise ReconstructionValidationError("I4.3F2 provider descriptor/selection-plan implementation drifted.")
        if self.selection_request.available_view_count != self.plan.selected_view_count:
            raise ReconstructionValidationError("I4.3F2 selection request must expose the selected provider-input view count.")
        if self.provider_sfm_result.statistics.registered_keyframes != self.plan.selected_view_count:
            raise ReconstructionValidationError("I4.3F2 provider projection count drifted from the adaptive plan.")
        if self.original_sfm_result.statistics.registered_keyframes != self.plan.source_view_count:
            raise ReconstructionValidationError("I4.3F2 original SfM count drifted from the adaptive plan.")
        if _sfm_authority_digest(self.original_sfm_result) != self.original_authority_sha256:
            raise ReconstructionValidationError("I4.3F2 original authoritative SfM changed during preparation.")
        if self.selection_request.authoritative_sfm_frame_id != frame_id:
            raise ReconstructionValidationError("I4.3F2 request/frame identity drifted.")
        if int(self.schema_version) != I43F2_SCHEMA_VERSION:
            raise ReconstructionValidationError("Unsupported I4.3F2 execution-preparation schema version.")
        object.__setattr__(self, "reconstruction_frame_id", frame_id)
        object.__setattr__(self, "authority_metadata", normalize_metadata(self.authority_metadata))
        object.__setattr__(self, "schema_version", I43F2_SCHEMA_VERSION)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "provider": self.provider.as_dict(),
            "reconstruction_frame_id": self.reconstruction_frame_id,
            "source_view_count": self.plan.source_view_count,
            "selected_view_count": self.plan.selected_view_count,
            "selection_plan": self.plan.as_dict(),
            "selection_request": self.selection_request.as_dict(),
            "selection_decision": self.selection_decision.as_dict(),
            "original_authority_sha256": self.original_authority_sha256,
            "provider_input_projection_used": self.provider_sfm_result is not self.original_sfm_result,
            "authority": metadata_as_dict(self.authority_metadata),  # type: ignore[arg-type]
        }


def prepare_adaptive_provider_execution(
    sfm_result: OfflineSfmResult,
    *,
    provider: DenseProviderDescriptor,
    reconstruction_frame_id: str,
    resources: AdaptiveSelectionResourceSnapshot,
    override: AdaptiveViewSelectionOverride | None = None,
) -> AdaptiveProviderExecutionPreparation:
    """Build adaptive provider input + explicit I4.3A selection contracts."""

    if not isinstance(provider, DenseProviderDescriptor):
        raise TypeError("provider must be DenseProviderDescriptor.")
    if not provider.available:
        raise ReconstructionValidationError(f"I4.3F2 provider is unavailable: {provider.unavailable_reason}")
    frame_id = require_nonempty(reconstruction_frame_id, field_name="reconstruction_frame_id")
    original_digest = _sfm_authority_digest(sfm_result)
    plan = build_adaptive_view_selection_plan(
        sfm_result,
        implementation=provider.implementation,
        resources=resources,
        override=override,
    )
    provider_sfm = project_authoritative_sfm_for_provider(sfm_result, plan)
    request = DenseProviderSelectionRequest(
        mode=DenseProviderSelectionMode.EXPLICIT_PROVIDER,
        available_view_count=plan.selected_view_count,
        authoritative_sfm_available=True,
        authoritative_sfm_frame_id=frame_id,
        requested_provider_id=provider.provider_id,
        allow_fallback=False,
    )
    decision = DenseProviderSelectionDecision(
        request_mode=DenseProviderSelectionMode.EXPLICIT_PROVIDER,
        requested_provider_id=provider.provider_id,
        selected_provider_id=provider.provider_id,
        selected_implementation=provider.implementation,
        considered_provider_ids=(provider.provider_id,),
        automatic=False,
        engineer_override_applied=False,
        fallback_used=False,
        reason_codes=("explicit_provider_selection", "i4_3f2_adaptive_view_plan_bound"),
    )
    return AdaptiveProviderExecutionPreparation(
        provider=provider,
        reconstruction_frame_id=frame_id,
        original_sfm_result=sfm_result,
        provider_sfm_result=provider_sfm,
        plan=plan,
        selection_request=request,
        selection_decision=decision,
        original_authority_sha256=original_digest,
        authority_metadata={
            "camera_pose_authority_owner": I43A_AUTHORITATIVE_FRAME_OWNER,
            "authoritative_sfm_modified": False,
            "provider_selected_by_auto": False,
            "provider_promoted": False,
            "hybrid_authorized": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
            "permanent_global_view_cap": None,
        },
    )


def finalize_adaptive_provider_execution_result(
    result: DenseProviderExecutionResult,
    preparation: AdaptiveProviderExecutionPreparation,
) -> DenseProviderExecutionResult:
    """Rebind provider-local result counts to the original authoritative source set."""

    if not isinstance(result, DenseProviderExecutionResult):
        raise TypeError("result must be DenseProviderExecutionResult.")
    if not isinstance(preparation, AdaptiveProviderExecutionPreparation):
        raise TypeError("preparation must be AdaptiveProviderExecutionPreparation.")
    plan = preparation.plan
    if result.provider_id != preparation.provider.provider_id or result.implementation is not plan.implementation:
        raise ReconstructionValidationError("I4.3F2 provider result identity drifted from preparation.")
    if result.source_view_count != plan.selected_view_count or result.selected_view_count != plan.selected_view_count:
        raise ReconstructionValidationError("I4.3F2 provider-local result did not execute the prepared selected-view set.")
    if _sfm_authority_digest(preparation.original_sfm_result) != preparation.original_authority_sha256:
        raise ReconstructionValidationError("I4.3F2 original authoritative SfM changed during provider execution.")
    reason = ";".join(plan.reason_codes)
    return replace(
        result,
        source_view_count=plan.source_view_count,
        selected_view_count=plan.selected_view_count,
        adaptive_view_selection=True,
        view_selection_reason=reason,
        permanent_global_view_cap=None,
    )


def build_i4_3f2_production_readiness_snapshot(
    *,
    i4_3f1_snapshot: Mapping[str, Any],
    resources: AdaptiveSelectionResourceSnapshot,
) -> dict[str, Any]:
    if not isinstance(i4_3f1_snapshot, Mapping):
        raise TypeError("i4_3f1_snapshot must be a mapping.")
    if i4_3f1_snapshot.get("phase") != "I4.3F1" or i4_3f1_snapshot.get("status") != "pass":
        raise ReconstructionValidationError("I4.3F2 requires a passing I4.3F1 binding snapshot.")
    decision = i4_3f1_snapshot.get("decision")
    if not isinstance(decision, Mapping) or decision.get("i4_3f2_production_wiring_authorized") is not True:
        raise ReconstructionValidationError("I4.3F2 requires explicit F1 production-wiring authorization.")
    if decision.get("permanent_global_view_cap") is not None:
        raise ReconstructionValidationError("I4.3F2 refuses a predecessor permanent global view cap.")
    return {
        "schema_version": I43F2_SCHEMA_VERSION,
        "phase": I43F2_PHASE,
        "status": "pass",
        "pass": True,
        "scope": {
            "production_offline_worker_pre_sfm_adaptive_planning": True,
            "legacy_default_max_keyframes_removed": True,
            "explicit_engineer_max_keyframes_override_supported": True,
            "post_sfm_provider_projection_bridge": True,
            "provider_execution_result_rebinding": True,
            "mvs_selected_sparse_subset_supported_without_source_mutation": True,
            "provider_execution_started_by_binding_smoke": False,
        },
        "resources": resources.as_dict(),
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
            "permanent_global_view_cap": None,
        },
        "decision": {
            "i4_3f2_production_wiring_closed": True,
            "i4_3f3_production_validation_authorized": True,
            "provider_selected": False,
            "auto_policy_implemented": False,
            "permanent_global_view_cap": None,
        },
        "next": "I4.3F3 production validation and I4.3F closure.",
    }


__all__ = [
    "I43F2_SCHEMA_VERSION",
    "I43F2_PHASE",
    "I43F2_PRE_SFM_WORK_UNITS_PER_FRAME",
    "AdaptivePreSfmFramePlan",
    "AdaptiveProviderExecutionPreparation",
    "build_adaptive_pre_sfm_frame_plan",
    "project_authoritative_sfm_for_provider",
    "prepare_adaptive_provider_execution",
    "finalize_adaptive_provider_execution_result",
    "build_i4_3f2_production_readiness_snapshot",
]

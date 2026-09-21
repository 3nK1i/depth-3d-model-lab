"""I6.5B deterministic adaptive frame/view planning and hardware bounds.

This module resolves the I6.5A frame/view request contract using the already
qualified I4.3F2 resource-planning heuristic.  It is intentionally execution
free: no source sampling is performed, no worker/controller/UI behavior is
changed, and no retry-until-pass policy is introduced.

The I4.3F2 resource target remains a transparent planning heuristic, not an
empirical runtime guarantee.  I6.5B records the exact resource snapshot and
both the raw and source-clamped planning targets so later controller/UI work can
explain requested versus effective behavior without silently changing modes.

PyCOLMAP remains the sole canonical camera/global-frame authority.  Planned
source views are never treated as registered cameras.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any

from core.reconstruction.adaptive_production_planning import (
    I43F2_PRE_SFM_WORK_UNITS_PER_FRAME,
    build_adaptive_pre_sfm_frame_plan,
)
from core.reconstruction.adaptive_view_selection import AdaptiveSelectionResourceSnapshot
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.reconstruction_frame_view_plan_contract import (
    I65A_CAMERA_AUTHORITY_OWNER,
    I65A_MINIMUM_RECONSTRUCTION_VIEWS,
    FrameViewPlanMode,
    FrameViewPlanRequest,
    FrameViewPlanResolution,
    FrameViewSourceProvenance,
)

I65B_SCHEMA_VERSION = 1
I65B_PHASE = "I6.5B"
I65B_PERMANENT_GLOBAL_VIEW_CAP = None
I65B_RESOURCE_FORMULA = "24*available_ram_gib + 32*free_vram_gib_when_cuda_ready"


def _canonical_sha256(payload: Any) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _unique_reasons(*groups: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    values: list[str] = []
    for group in groups:
        for value in group:
            text = str(value or "").strip()
            if text and text not in values:
                values.append(text)
    if not values:
        raise ReconstructionValidationError("A deterministic frame/view plan requires reason codes.")
    return tuple(values)


@dataclass(frozen=True, slots=True)
class FrameViewHardwareBounds:
    """Hardware/resource provenance behind one deterministic planning decision.

    ``raw_resource_target_views`` is the unbounded I4.3F2 heuristic result.
    ``effective_resource_target_views`` is clamped to a known source count for
    compatibility with the I6.5A product contract.  Neither value is a measured
    runtime guarantee or a permanent application cap.
    """

    resources: AdaptiveSelectionResourceSnapshot
    raw_resource_target_views: int | None
    effective_resource_target_views: int | None
    planning_capacity_units: float | None
    source_count_clamped: bool
    planning_work_units_per_frame: float = I43F2_PRE_SFM_WORK_UNITS_PER_FRAME
    planning_capacity_is_empirical_runtime_claim: bool = False
    permanent_global_view_cap: int | None = I65B_PERMANENT_GLOBAL_VIEW_CAP
    schema_version: int = I65B_SCHEMA_VERSION
    hardware_context_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.resources, AdaptiveSelectionResourceSnapshot):
            raise TypeError("resources must be AdaptiveSelectionResourceSnapshot.")
        for name in ("raw_resource_target_views", "effective_resource_target_views"):
            value = getattr(self, name)
            if value is not None:
                if isinstance(value, bool) or int(value) < I65A_MINIMUM_RECONSTRUCTION_VIEWS:
                    raise ReconstructionValidationError(
                        f"{name} must be at least {I65A_MINIMUM_RECONSTRUCTION_VIEWS} when present."
                    )
                object.__setattr__(self, name, int(value))
        raw = self.raw_resource_target_views
        effective = self.effective_resource_target_views
        if raw is None and effective is not None:
            raise ReconstructionValidationError(
                "effective_resource_target_views cannot exist without a raw resource target."
            )
        if raw is not None and effective is not None and effective > raw:
            raise ReconstructionValidationError(
                "effective_resource_target_views cannot exceed raw_resource_target_views."
            )
        capacity = self.planning_capacity_units
        if capacity is not None:
            capacity = float(capacity)
            if capacity < 0.0:
                raise ReconstructionValidationError("planning_capacity_units must be non-negative.")
            object.__setattr__(self, "planning_capacity_units", capacity)
        work = float(self.planning_work_units_per_frame)
        if work <= 0.0:
            raise ReconstructionValidationError("planning_work_units_per_frame must be positive.")
        object.__setattr__(self, "planning_work_units_per_frame", work)
        if not isinstance(self.source_count_clamped, bool):
            raise TypeError("source_count_clamped must be bool.")
        if self.planning_capacity_is_empirical_runtime_claim is not False:
            raise ReconstructionValidationError(
                "I6.5B resource planning must not be represented as an empirical runtime guarantee."
            )
        if self.permanent_global_view_cap is not None:
            raise ReconstructionValidationError("I6.5B forbids a permanent global view cap.")
        if int(self.schema_version) != I65B_SCHEMA_VERSION:
            raise ReconstructionValidationError("Unsupported I6.5B hardware-bounds schema version.")
        object.__setattr__(self, "schema_version", I65B_SCHEMA_VERSION)
        object.__setattr__(self, "hardware_context_sha256", _canonical_sha256(self._digest_payload()))

    def _digest_payload(self) -> dict[str, Any]:
        return {
            "phase": I65B_PHASE,
            "schema_version": I65B_SCHEMA_VERSION,
            "resources": self.resources.as_dict(),
            "raw_resource_target_views": self.raw_resource_target_views,
            "effective_resource_target_views": self.effective_resource_target_views,
            "planning_capacity_units": self.planning_capacity_units,
            "planning_work_units_per_frame": self.planning_work_units_per_frame,
            "planning_capacity_formula": I65B_RESOURCE_FORMULA,
            "planning_capacity_is_empirical_runtime_claim": False,
            "source_count_clamped": self.source_count_clamped,
            "permanent_global_view_cap": None,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            **self._digest_payload(),
            "hardware_context_sha256": self.hardware_context_sha256,
        }


@dataclass(frozen=True, slots=True)
class DeterministicFrameViewPlan:
    """Resolved I6.5A plan plus I6.5B hardware provenance and safety policy."""

    resolution: FrameViewPlanResolution
    hardware_bounds: FrameViewHardwareBounds
    resource_bound_applied: bool
    all_eligible_exceeds_resource_bound: bool
    execution_resource_preflight_required: bool = True
    schema_version: int = I65B_SCHEMA_VERSION
    phase: str = I65B_PHASE
    planning_provenance_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.resolution, FrameViewPlanResolution):
            raise TypeError("resolution must be FrameViewPlanResolution.")
        if not isinstance(self.hardware_bounds, FrameViewHardwareBounds):
            raise TypeError("hardware_bounds must be FrameViewHardwareBounds.")
        for name in (
            "resource_bound_applied",
            "all_eligible_exceeds_resource_bound",
            "execution_resource_preflight_required",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool.")
        if self.phase != I65B_PHASE or int(self.schema_version) != I65B_SCHEMA_VERSION:
            raise ReconstructionValidationError("Unsupported I6.5B deterministic-plan schema version.")
        if self.resolution.request.mode is FrameViewPlanMode.ALL_ELIGIBLE and self.resource_bound_applied:
            raise ReconstructionValidationError(
                "ALL_ELIGIBLE cannot silently receive the adaptive resource cap."
            )
        if self.all_eligible_exceeds_resource_bound and self.resolution.request.mode is not FrameViewPlanMode.ALL_ELIGIBLE:
            raise ReconstructionValidationError(
                "all_eligible_exceeds_resource_bound is only meaningful for ALL_ELIGIBLE requests."
            )
        object.__setattr__(self, "schema_version", I65B_SCHEMA_VERSION)
        object.__setattr__(self, "phase", I65B_PHASE)
        object.__setattr__(self, "planning_provenance_sha256", _canonical_sha256(self._digest_payload()))

    def _digest_payload(self) -> dict[str, Any]:
        return {
            "phase": I65B_PHASE,
            "schema_version": I65B_SCHEMA_VERSION,
            "plan_input_sha256": self.resolution.plan_input_sha256,
            "hardware_context_sha256": self.hardware_bounds.hardware_context_sha256,
            "resource_bound_applied": self.resource_bound_applied,
            "all_eligible_exceeds_resource_bound": self.all_eligible_exceeds_resource_bound,
            "execution_resource_preflight_required": self.execution_resource_preflight_required,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            **self._digest_payload(),
            "planning_provenance_sha256": self.planning_provenance_sha256,
            "resolution": self.resolution.as_dict(),
            "hardware_bounds": self.hardware_bounds.as_dict(),
            "policy": {
                "legacy_i4_3f2_resource_planner_reused": True,
                "request_mode_substituted": False,
                "retry_until_pass": False,
                "silent_replan_after_sfm_failure": False,
                "resource_target_is_empirical_runtime_guarantee": False,
                "permanent_global_view_cap": None,
            },
            "effects": {
                "sampling_executed": False,
                "ingestion_modified": False,
                "controller_modified": False,
                "ui_modified": False,
                "worker_modified": False,
                "provider_selected": False,
                "provider_activated": False,
                "production_auto_activated": False,
                "execution_started": False,
                "scientific_threshold_modified": False,
                "production_default_modified": False,
            },
            "authority": {
                "camera_pose_authority_owner": I65A_CAMERA_AUTHORITY_OWNER,
                "planned_views_are_registered_cameras": False,
                "pycolmap_registration_outcome_claimed": False,
                "metric_scale_claimed": False,
                "certified_metrology": False,
            },
        }


def _clamped_resource_target(raw_target: int | None, source_total: int | None) -> tuple[int | None, bool]:
    if raw_target is None:
        return None, False
    target = int(raw_target)
    if source_total is not None and target > source_total:
        return int(source_total), True
    return target, False


def resolve_deterministic_frame_view_plan(
    *,
    source: FrameViewSourceProvenance,
    request: FrameViewPlanRequest,
    resources: AdaptiveSelectionResourceSnapshot,
) -> DeterministicFrameViewPlan:
    """Resolve one deterministic pre-SfM product plan without executing it.

    ADAPTIVE reuses the existing I4.3F2 resource target.
    MANUAL_LIMIT treats the requested count as a maximum and applies source and
    resource bounds deterministically.
    ALL_ELIGIBLE preserves the mode without a silent cap; resource pressure is
    exposed as advisory provenance and remains subject to execution preflight.
    """

    if not isinstance(source, FrameViewSourceProvenance):
        raise TypeError("source must be FrameViewSourceProvenance.")
    if not isinstance(request, FrameViewPlanRequest):
        raise TypeError("request must be FrameViewPlanRequest.")
    if not isinstance(resources, AdaptiveSelectionResourceSnapshot):
        raise TypeError("resources must be AdaptiveSelectionResourceSnapshot.")

    total = source.estimated_total_views
    if total is not None and total < I65A_MINIMUM_RECONSTRUCTION_VIEWS:
        raise ReconstructionValidationError(
            f"Frame/view planning requires at least {I65A_MINIMUM_RECONSTRUCTION_VIEWS} known source views."
        )

    legacy = build_adaptive_pre_sfm_frame_plan(
        source_kind=source.source_kind,
        estimated_total_frames=total,
        expected_frame_rate_hz=source.expected_frame_rate_hz,
        resources=resources,
        engineer_max_keyframes_override=None,
    )
    raw_resource_target = legacy.resource_target_frames
    effective_resource_target, source_clamped = _clamped_resource_target(raw_resource_target, total)
    hardware = FrameViewHardwareBounds(
        resources=resources,
        raw_resource_target_views=raw_resource_target,
        effective_resource_target_views=effective_resource_target,
        planning_capacity_units=legacy.planning_capacity_units,
        source_count_clamped=source_clamped,
    )

    base_reasons = list(legacy.reason_codes)
    base_reasons.extend((
        "i6_5b_deterministic_resolution",
        "legacy_i4_3f2_resource_planner_reused",
        "resource_target_is_planning_heuristic_not_runtime_guarantee",
        "retry_until_pass_forbidden",
    ))

    resource_bound_applied = False
    all_eligible_exceeds = False

    if request.mode is FrameViewPlanMode.ADAPTIVE:
        target = legacy.target_sampled_frames
        if target is not None and total is not None:
            target = min(int(target), int(total))
        if target is not None and target < I65A_MINIMUM_RECONSTRUCTION_VIEWS:
            raise ReconstructionValidationError("Adaptive planning resolved fewer than three views.")
        resource_bound_applied = bool(
            target is not None
            and effective_resource_target is not None
            and target == effective_resource_target
            and (total is None or target < total)
        )
        reasons = _unique_reasons(base_reasons, ["adaptive_mode_preserved"])

    elif request.mode is FrameViewPlanMode.MANUAL_LIMIT:
        requested = request.requested_max_views
        if requested is None:
            raise ReconstructionValidationError("MANUAL_LIMIT requires requested_max_views.")
        target = int(requested)
        reasons_extra: list[str] = ["manual_limit_requested", "manual_limit_is_maximum_not_exact_count"]
        if total is not None and target > total:
            target = int(total)
            reasons_extra.append("manual_limit_source_count_bound_applied")
        before_resource = target
        if effective_resource_target is not None and target > effective_resource_target:
            target = int(effective_resource_target)
            resource_bound_applied = True
            reasons_extra.append("manual_limit_resource_bound_applied")
        else:
            resource_bound_applied = False
            if effective_resource_target is None:
                reasons_extra.append("manual_limit_resource_bound_unresolved")
            elif before_resource <= effective_resource_target:
                reasons_extra.append("manual_limit_within_resource_planning_bound")
        if target < I65A_MINIMUM_RECONSTRUCTION_VIEWS:
            raise ReconstructionValidationError("Manual planning resolved fewer than three views.")
        reasons = _unique_reasons(base_reasons, reasons_extra)

    elif request.mode is FrameViewPlanMode.ALL_ELIGIBLE:
        target = None
        reasons_extra = ["all_eligible_requested", "all_eligible_resource_target_advisory_only"]
        if total is not None and effective_resource_target is not None:
            if total > effective_resource_target:
                all_eligible_exceeds = True
                reasons_extra.append("all_eligible_exceeds_resource_planning_bound")
            else:
                reasons_extra.append("all_eligible_within_resource_planning_bound")
        elif effective_resource_target is None:
            reasons_extra.append("all_eligible_resource_capacity_unresolved")
        else:
            reasons_extra.append("all_eligible_source_count_unresolved")
        reasons = _unique_reasons(base_reasons, reasons_extra)

    else:  # pragma: no cover - enum exhaustiveness guard
        raise ReconstructionValidationError(f"Unsupported frame/view planning mode: {request.mode!r}")

    resolution = FrameViewPlanResolution(
        source=source,
        request=request,
        effective_target_views=target,
        resource_target_views=effective_resource_target,
        reason_codes=reasons,
        permanent_global_view_cap=None,
    )
    return DeterministicFrameViewPlan(
        resolution=resolution,
        hardware_bounds=hardware,
        resource_bound_applied=resource_bound_applied,
        all_eligible_exceeds_resource_bound=all_eligible_exceeds,
        execution_resource_preflight_required=True,
    )


__all__ = [
    "I65B_SCHEMA_VERSION",
    "I65B_PHASE",
    "I65B_PERMANENT_GLOBAL_VIEW_CAP",
    "I65B_RESOURCE_FORMULA",
    "FrameViewHardwareBounds",
    "DeterministicFrameViewPlan",
    "resolve_deterministic_frame_view_plan",
]

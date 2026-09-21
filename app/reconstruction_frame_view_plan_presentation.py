"""I6.5D professional frame/view-plan presentation.

This module converts the frozen I6.5A-C planning state into compact UI text.
It has no Qt dependency and no execution authority: it never samples source
frames, mutates worker settings, starts SfM, or claims that a planned view is a
PyCOLMAP-registered camera.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.reconstruction_frame_view_plan_state import (
    FrameViewPlanIntentOrigin,
    FrameViewPlanStateStatus,
    ReconstructionFrameViewPlanState,
)
from core.reconstruction.reconstruction_frame_view_plan_contract import FrameViewPlanMode

I65D_PHASE = "I6.5D"
I65D_PRESENTATION_PROFILE = "professional_frame_view_plan_ui"
I65D_DEFAULT_MANUAL_LIMIT = 64


def _gib(value: int | None) -> str:
    if value is None:
        return "unknown"
    return f"{float(value) / (1024.0 ** 3):.1f} GiB"


@dataclass(frozen=True, slots=True)
class ReconstructionFrameViewPlanPresentation:
    status_text: str
    status_kind: str
    status_tooltip: str
    mode_text: str
    source_text: str
    source_tooltip: str
    planned_text: str
    planned_tooltip: str
    resource_text: str
    resource_tooltip: str
    hardware_text: str
    hardware_tooltip: str
    provenance_text: str
    provenance_tooltip: str
    boundary_text: str
    boundary_tooltip: str
    manual_limit_value: int
    manual_limit_enabled: bool
    apply_enabled: bool
    reset_enabled: bool


def present_reconstruction_frame_view_plan(
    state: ReconstructionFrameViewPlanState,
) -> ReconstructionFrameViewPlanPresentation:
    """Build a compact, fail-closed I6.5D presentation from passive state."""

    if not isinstance(state, ReconstructionFrameViewPlanState):
        raise TypeError("state must be ReconstructionFrameViewPlanState.")

    request = state.request
    mode_text = {
        FrameViewPlanMode.ADAPTIVE: "Adaptive",
        FrameViewPlanMode.ALL_ELIGIBLE: "All eligible",
        FrameViewPlanMode.MANUAL_LIMIT: f"Manual limit · max {request.requested_max_views}",
    }[request.mode]
    manual_limit_value = (
        int(request.requested_max_views)
        if request.requested_max_views is not None
        else I65D_DEFAULT_MANUAL_LIMIT
    )

    source = state.source
    if source is None:
        source_text = "No source"
        source_tooltip = "Choose an offline image sequence or recorded video to resolve a pre-SfM plan."
    elif source.estimated_total_views is None:
        source_text = "View count unresolved"
        source_tooltip = (
            f"{source.source_kind} • {source.source_uri}. The source identity is known, "
            "but a frame count is not invented before ingestion."
        )
    else:
        source_text = f"{source.estimated_total_views} source views"
        source_tooltip = (
            f"{source.source_kind} • planning identity {source.source_fingerprint_sha256}. "
            "This is planning provenance, not a PyCOLMAP registration result."
        )

    plan = state.plan
    if plan is None:
        planned_text = "Not resolved"
        resource_text = "Not resolved"
        hardware_text = "Waiting for planning resources" if source is not None else "Startup resources available when source is ready"
        provenance_text = "Not available"
        planned_tooltip = "No effective pre-SfM plan has been resolved yet."
        resource_tooltip = "The resource-aware planning bound is not available in the current state."
        hardware_tooltip = "Resource observations are planning inputs only; they are not runtime guarantees."
        provenance_tooltip = "A deterministic plan provenance SHA-256 is produced only for resolved plans."
    else:
        total = plan.resolution.source.estimated_total_views
        effective = plan.resolution.effective_target_views
        if request.mode is FrameViewPlanMode.ALL_ELIGIBLE:
            planned_text = f"All {total} eligible" if total is not None else "All eligible"
        elif effective is not None and total is not None:
            planned_text = f"{effective} planned of {total}"
        elif effective is not None:
            planned_text = f"Up to {effective} planned"
        else:
            planned_text = "All eligible"
        planned_tooltip = (
            "This is a deterministic pre-SfM planning target. It is not a claim that PyCOLMAP "
            "will register the same number of cameras."
        )

        bound = plan.hardware_bounds.effective_resource_target_views
        resource_text = (
            f"{bound}-view planning bound" if bound is not None else "Planning bound unresolved"
        )
        if plan.all_eligible_exceeds_resource_bound:
            resource_text += " · review"
        resource_tooltip = (
            "The resource target reuses the qualified I4.3F2 planning heuristic. It is advisory for "
            "ALL_ELIGIBLE and is never represented as an empirical runtime guarantee."
        )

        resources = plan.hardware_bounds.resources
        hardware_text = resources.gpu_name
        hardware_tooltip = (
            f"GPU: {resources.gpu_name}\n"
            f"Free VRAM snapshot: {_gib(resources.free_vram_bytes)}\n"
            f"Available RAM snapshot: {_gib(resources.available_ram_bytes)}\n"
            f"CUDA ready: {'yes' if resources.cuda_ready else 'no'}\n"
            "These observations are captured outside ReconstructionController and used only for deterministic planning."
        )
        provenance_text = f"{plan.planning_provenance_sha256[:12]}…"
        provenance_tooltip = (
            f"Deterministic planning provenance SHA-256: {plan.planning_provenance_sha256}\n"
            f"Hardware context SHA-256: {plan.hardware_bounds.hardware_context_sha256}"
        )

    if state.status is FrameViewPlanStateStatus.WAITING_SOURCE:
        status_text, status_kind = "Waiting for source", "neutral"
        status_tooltip = "Choose a supported offline source; the current planning request is retained."
    elif state.status is FrameViewPlanStateStatus.UNSUPPORTED_WORKFLOW:
        status_text, status_kind = "Not applicable", "neutral"
        status_tooltip = "Adaptive pre-SfM frame/view planning applies to offline image sequences and recorded video."
    elif state.status is FrameViewPlanStateStatus.WAITING_RESOURCES:
        status_text, status_kind = "Waiting for resources", "neutral"
        status_tooltip = "The source is ready, but an explicit application resource snapshot has not been supplied."
    elif state.status is FrameViewPlanStateStatus.BLOCKED:
        status_text, status_kind = "Plan unavailable", "warning"
        status_tooltip = "The current source/request/resource combination could not produce a valid deterministic plan."
    elif request.mode is FrameViewPlanMode.ALL_ELIGIBLE and plan is not None and plan.all_eligible_exceeds_resource_bound:
        status_text, status_kind = "All eligible · resource review", "warning"
        status_tooltip = (
            "All eligible views remain requested. The resource heuristic suggests a smaller planning bound, "
            "but I6.5 never silently substitutes Adaptive or caps this request."
        )
    elif request.mode is FrameViewPlanMode.ALL_ELIGIBLE:
        status_text, status_kind = "All eligible ready", "ready"
        status_tooltip = "All eligible source views remain requested; execution preflight still applies."
    elif request.mode is FrameViewPlanMode.MANUAL_LIMIT:
        status_text, status_kind = "Manual plan ready", "ready"
        status_tooltip = "The manual value is a maximum; the effective plan may be lower after source/resource bounds."
    else:
        status_text, status_kind = "Adaptive plan ready", "ready"
        status_tooltip = "The deterministic adaptive target is resolved from source provenance and explicit resources."

    return ReconstructionFrameViewPlanPresentation(
        status_text=status_text,
        status_kind=status_kind,
        status_tooltip=status_tooltip,
        mode_text=mode_text,
        source_text=source_text,
        source_tooltip=source_tooltip,
        planned_text=planned_text,
        planned_tooltip=planned_tooltip,
        resource_text=resource_text,
        resource_tooltip=resource_tooltip,
        hardware_text=hardware_text,
        hardware_tooltip=hardware_tooltip,
        provenance_text=provenance_text,
        provenance_tooltip=provenance_tooltip,
        boundary_text="Pre-SfM intent • no frames sampled or cameras registered",
        boundary_tooltip=(
            "Applying these controls records planning intent only. Manual limit is a maximum, ALL_ELIGIBLE is not "
            "silently capped, no worker settings are mutated, and no reconstruction starts. PyCOLMAP remains the "
            "sole canonical camera/global-frame authority; a planned view is not a registered camera."
        ),
        manual_limit_value=manual_limit_value,
        manual_limit_enabled=request.mode is FrameViewPlanMode.MANUAL_LIMIT,
        apply_enabled=True,
        reset_enabled=state.intent_origin is FrameViewPlanIntentOrigin.EXPLICIT_USER_OVERRIDE,
    )


__all__ = [
    "I65D_PHASE",
    "I65D_PRESENTATION_PROFILE",
    "I65D_DEFAULT_MANUAL_LIMIT",
    "ReconstructionFrameViewPlanPresentation",
    "present_reconstruction_frame_view_plan",
]

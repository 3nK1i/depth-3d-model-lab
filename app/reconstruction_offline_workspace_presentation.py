"""I6.1C presentation adapter for the passive offline reconstruction workspace.

The I6.1A workspace snapshot is authoritative product state.  This module only
reduces that state to compact, professional UI text/kinds/tooltips for the
existing Reconstruction page.  It does not start/cancel work, select providers,
change persistence, promote metric authority, or activate production Auto.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.reconstruction_offline_workspace import (
    OfflineReconstructionWorkspaceSnapshot,
    OfflineWorkspaceEvidenceIdentityStatus,
    OfflineWorkspaceScaleStatus,
)


@dataclass(frozen=True, slots=True)
class OfflineReconstructionWorkspacePresentation:
    workspace_text: str
    workspace_status_kind: str
    workspace_tooltip: str
    authority_text: str
    authority_status_kind: str
    authority_tooltip: str


def present_offline_reconstruction_workspace(
    snapshot: OfflineReconstructionWorkspaceSnapshot,
) -> OfflineReconstructionWorkspacePresentation:
    """Build compact presentation from one immutable workspace snapshot."""

    if not isinstance(snapshot, OfflineReconstructionWorkspaceSnapshot):
        raise TypeError("snapshot must be OfflineReconstructionWorkspaceSnapshot.")

    workspace_text, workspace_kind = _workspace_state(snapshot)
    workspace_tooltip = _workspace_tooltip(snapshot)
    authority_text, authority_kind = _authority_state(snapshot)
    authority_tooltip = _authority_tooltip(snapshot)

    return OfflineReconstructionWorkspacePresentation(
        workspace_text=workspace_text,
        workspace_status_kind=workspace_kind,
        workspace_tooltip=workspace_tooltip,
        authority_text=authority_text,
        authority_status_kind=authority_kind,
        authority_tooltip=authority_tooltip,
    )


def _workspace_state(
    snapshot: OfflineReconstructionWorkspaceSnapshot,
) -> tuple[str, str]:
    if snapshot.evidence_identity_status is OfflineWorkspaceEvidenceIdentityStatus.CONFLICT:
        return "Evidence conflict", "error"

    if snapshot.persistent_scene_available:
        return "Scene ready", "ready"

    if snapshot.partial_result_available:
        return "Partial result", "warning"

    state = snapshot.offline_run_state
    if state == "starting":
        return "Preparing reconstruction", "running"
    if state == "running":
        return "Reconstructing", "running"
    if state == "cancelling":
        return "Cancelling", "warning"
    if state == "failed":
        return "Reconstruction failed", "error"
    if state == "cancelled":
        return "Cancelled", "warning"
    if state == "completed":
        return "Completed • no scene", "warning"

    if snapshot.source_ready:
        return "Ready to reconstruct", "ready"

    if snapshot.offline_workflow_supported:
        return "Source needs attention", "warning"

    return "Offline idle", "neutral"


def _workspace_tooltip(snapshot: OfflineReconstructionWorkspaceSnapshot) -> str:
    if snapshot.evidence_identity_status is OfflineWorkspaceEvidenceIdentityStatus.CONFLICT:
        return (
            "Offline reconstruction evidence identity conflict. The workspace fails closed; "
            f"no incompatible state is merged. {snapshot.identity_details}"
        ).strip()

    parts: list[str] = []
    if snapshot.workflow == "image_sequence":
        parts.append("Offline workflow: image sequence.")
    elif snapshot.workflow == "recorded_video":
        parts.append("Offline workflow: recorded video.")
    else:
        parts.append(
            "Offline workspace is idle. Select an image sequence or recorded video to use "
            "the offline reconstruction workflow."
        )

    if snapshot.offline_run_state in {"starting", "running", "cancelling"}:
        if snapshot.current_stage:
            parts.append(f"Stage: {_humanize(snapshot.current_stage)}.")
        if snapshot.progress_fraction is not None:
            parts.append(f"Progress: {round(snapshot.progress_fraction * 100.0):d}%.")
        if snapshot.progress_message:
            parts.append(snapshot.progress_message)
    elif snapshot.persistent_scene_available:
        parts.append("A persistent canonical reconstruction scene is available.")
    elif snapshot.partial_result_available:
        parts.append(
            "A bounded partial reconstruction result is available; it is not a persistent "
            "canonical scene."
        )
    elif snapshot.source_ready:
        parts.append("Source validation is ready; reconstruction has not been started by this UI binding.")
    elif snapshot.offline_workflow_supported:
        parts.append("The selected offline source is not currently ready to run.")

    if snapshot.checkpoint_available:
        resume = "resume-compatible" if snapshot.checkpoint_resume_allowed else "not resume-compatible"
        parts.append(f"Checkpoint linked ({resume}).")

    return " ".join(parts)


def _authority_state(
    snapshot: OfflineReconstructionWorkspaceSnapshot,
) -> tuple[str, str]:
    scale = snapshot.scale_status
    if scale is OfflineWorkspaceScaleStatus.CALIBRATED_METRIC:
        return "PyCOLMAP • calibrated metric", "ready"
    if scale is OfflineWorkspaceScaleStatus.METRIC:
        return "PyCOLMAP • metric coordinates", "ready"
    if scale is OfflineWorkspaceScaleStatus.ARBITRARY:
        return "PyCOLMAP • arbitrary scale", "warning"
    return "PyCOLMAP • scale unqualified", "neutral"


def _authority_tooltip(snapshot: OfflineReconstructionWorkspaceSnapshot) -> str:
    scale_text = {
        OfflineWorkspaceScaleStatus.UNQUALIFIED: "Scale qualification: unqualified.",
        OfflineWorkspaceScaleStatus.ARBITRARY: "Scale qualification: arbitrary SfM gauge.",
        OfflineWorkspaceScaleStatus.METRIC: "Scale qualification: metric coordinates authorized by persistent evidence.",
        OfflineWorkspaceScaleStatus.CALIBRATED_METRIC: "Scale qualification: calibrated metric scale authorized by persistent evidence.",
    }[snapshot.scale_status]
    camera_text = (
        "Canonical camera/global-frame authority: PyCOLMAP SfM."
        if snapshot.camera_authority_owner == "pycolmap_sfm"
        else f"Canonical camera/global-frame authority: {snapshot.camera_authority_owner}."
    )
    geometry_text = (
        "Canonical geometry authority: available."
        if snapshot.canonical_geometry_authoritative
        else "Canonical geometry authority: not established by the current offline result."
    )
    return " ".join(
        (
            camera_text,
            scale_text,
            geometry_text,
            "Certified metrology: no.",
            "Production Auto: deferred; the qualified policy remains offline/shadow only.",
        )
    )


def _humanize(value: str) -> str:
    text = str(value or "").strip().replace("_", " ")
    return text[:1].upper() + text[1:] if text else ""


__all__ = [
    "OfflineReconstructionWorkspacePresentation",
    "present_offline_reconstruction_workspace",
]

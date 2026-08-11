"""Phase 8.8G4 failure/recovery presentation contracts.

G4 does not add another recovery engine.  Automatic live-worker recovery stays
owned by the existing hardened live runtime, checkpoint compatibility/preparation
stays owned by G3, and a manual retry starts a fresh attempt through the existing
selected-workflow start path.  This module only classifies already-authoritative
runtime/session evidence into conservative UX state.

A fresh retry is intentionally distinct from checkpoint resume: it creates a new
runtime attempt from the currently selected source and is disabled while retained
scene/checkpoint evidence would make that new attempt ambiguous.  G4 never turns
appearance artifacts into geometry authority and never certifies metrology.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.reconstruction_checkpoint_resume import (
    ReconstructionCheckpointResumePresentation,
    ReconstructionCheckpointResumeState,
)
from app.reconstruction_session_model import (
    ReconstructionSessionIdentityStatus,
    ReconstructionSessionModel,
)
from core.reconstruction import LiveCaptureRuntimeState, LiveCaptureRuntimeStats


class ReconstructionFailureRecoveryState(str, Enum):
    STABLE = "stable"
    RUNTIME_ACTIVE = "runtime_active"
    AUTO_RECOVERING = "auto_recovering"
    RETRY_AVAILABLE = "retry_available"
    CANCELLED_RETRY_AVAILABLE = "cancelled_retry_available"
    PRESERVED_EVIDENCE = "preserved_evidence"
    IDENTITY_CONFLICT = "identity_conflict"
    SOURCE_REQUIRED = "source_required"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class ReconstructionFailureRecoveryPresentation:
    """Compact G4 UI state; it never executes recovery by itself."""

    state: ReconstructionFailureRecoveryState
    can_retry_from_source: bool
    text: str
    details: str
    automatic_recovery: bool
    retry_starts_new_session: bool
    preserved_scene: bool
    preserved_checkpoint: bool
    canonical_geometry_authoritative: bool
    certified_metrology: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.state, ReconstructionFailureRecoveryState):
            raise TypeError("state must be ReconstructionFailureRecoveryState.")
        for field_name in (
            "can_retry_from_source",
            "automatic_recovery",
            "retry_starts_new_session",
            "preserved_scene",
            "preserved_checkpoint",
            "canonical_geometry_authoritative",
            "certified_metrology",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be bool.")
        text = str(self.text).strip()
        details = str(self.details).strip()
        if not text or not details:
            raise ValueError("G4 recovery presentation text must be non-empty.")
        if self.can_retry_from_source and not self.retry_starts_new_session:
            raise ValueError("A G4 retry must be explicitly represented as a fresh-session attempt.")
        if bool(self.certified_metrology):
            raise ValueError("G4 failure/recovery UX cannot certify metrology.")
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "details", details)
        object.__setattr__(self, "certified_metrology", False)

    @property
    def status_kind(self) -> str:
        if self.state in {
            ReconstructionFailureRecoveryState.IDENTITY_CONFLICT,
            ReconstructionFailureRecoveryState.SOURCE_REQUIRED,
        }:
            return "warning"
        if self.state in {
            ReconstructionFailureRecoveryState.AUTO_RECOVERING,
            ReconstructionFailureRecoveryState.RUNTIME_ACTIVE,
        }:
            return "running"
        if self.state in {
            ReconstructionFailureRecoveryState.RETRY_AVAILABLE,
            ReconstructionFailureRecoveryState.CANCELLED_RETRY_AVAILABLE,
            ReconstructionFailureRecoveryState.PRESERVED_EVIDENCE,
        }:
            return "warning"
        return "neutral"


def build_reconstruction_failure_recovery_presentation(
    model: ReconstructionSessionModel | None,
    *,
    live_state: LiveCaptureRuntimeState,
    offline_state: object,
    live_stats: LiveCaptureRuntimeStats | None,
    source_ready: bool,
    last_error: str,
    checkpoint_resume: ReconstructionCheckpointResumePresentation | None,
    controller_active: bool,
    controller_shutdown: bool,
) -> ReconstructionFailureRecoveryPresentation:
    """Derive conservative G4 recovery UX from existing runtime/session evidence."""

    if model is not None and not isinstance(model, ReconstructionSessionModel):
        raise TypeError("model must be ReconstructionSessionModel or None.")
    if not isinstance(live_state, LiveCaptureRuntimeState):
        raise TypeError("live_state must be LiveCaptureRuntimeState.")
    if live_stats is not None and not isinstance(live_stats, LiveCaptureRuntimeStats):
        raise TypeError("live_stats must be LiveCaptureRuntimeStats or None.")
    if checkpoint_resume is not None and not isinstance(
        checkpoint_resume, ReconstructionCheckpointResumePresentation
    ):
        raise TypeError("checkpoint_resume must be ReconstructionCheckpointResumePresentation or None.")

    offline_value = str(getattr(offline_state, "value", offline_state)).strip().lower()
    error = str(last_error).strip()
    has_scene = bool(model and model.scene_id)
    has_checkpoint = bool(model and model.checkpoint_id)
    authority = bool(model and model.canonical_geometry_authoritative)

    common = dict(
        retry_starts_new_session=True,
        preserved_scene=has_scene,
        preserved_checkpoint=has_checkpoint,
        canonical_geometry_authoritative=authority,
        certified_metrology=False,
    )

    if controller_shutdown:
        return ReconstructionFailureRecoveryPresentation(
            state=ReconstructionFailureRecoveryState.UNAVAILABLE,
            can_retry_from_source=False,
            text="Recovery actions unavailable",
            details="The reconstruction controller is shut down; no recovery action can be started.",
            automatic_recovery=False,
            **common,
        )

    if live_state is LiveCaptureRuntimeState.RECOVERING:
        restarts = live_stats.worker_restarts if live_stats is not None else 0
        detail = error or (live_stats.last_error if live_stats is not None else "")
        suffix = f" • {restarts} restart" + ("s" if restarts != 1 else "") if restarts else ""
        return ReconstructionFailureRecoveryPresentation(
            state=ReconstructionFailureRecoveryState.AUTO_RECOVERING,
            can_retry_from_source=False,
            text=f"Automatic live recovery{suffix}",
            details=(
                "The existing live runtime owns bounded worker restart/recovery. Manual retry remains "
                "disabled while that recovery attempt is active."
                + (f" Last error: {detail}" if detail else "")
            ),
            automatic_recovery=True,
            **common,
        )

    failed = live_state is LiveCaptureRuntimeState.FAILED or offline_value == "failed"
    cancelled = offline_value == "cancelled"

    if not failed and not cancelled:
        if controller_active:
            return ReconstructionFailureRecoveryPresentation(
                state=ReconstructionFailureRecoveryState.RUNTIME_ACTIVE,
                can_retry_from_source=False,
                text="Recovery standby • runtime active",
                details="The current reconstruction is active. Existing runtime recovery/cancellation policies remain authoritative.",
                automatic_recovery=False,
                **common,
            )
        return ReconstructionFailureRecoveryPresentation(
            state=ReconstructionFailureRecoveryState.STABLE,
            can_retry_from_source=False,
            text="No recovery action required",
            details="No terminal reconstruction failure or cancellation is currently focused.",
            automatic_recovery=False,
            **common,
        )

    if model is not None and model.identity_status is ReconstructionSessionIdentityStatus.CONFLICT:
        return ReconstructionFailureRecoveryPresentation(
            state=ReconstructionFailureRecoveryState.IDENTITY_CONFLICT,
            can_retry_from_source=False,
            text="Recovery blocked • session identity conflict",
            details=(
                "Runtime/scene/checkpoint evidence belongs to different session identities. Resolve that "
                "conflict explicitly before starting another recovery attempt."
                + (f" Failure: {error}" if error else "")
            ),
            automatic_recovery=False,
            **common,
        )

    if has_scene or has_checkpoint:
        if checkpoint_resume is not None and checkpoint_resume.state is ReconstructionCheckpointResumeState.PREPARED:
            route = "A validated checkpoint resume handoff is prepared; resumed execution remains explicit outside G4."
        elif checkpoint_resume is not None and checkpoint_resume.state is ReconstructionCheckpointResumeState.COMPATIBLE:
            route = "A compatible checkpoint is available; use G3 Prepare Resume if checkpoint continuation is intended."
        elif has_scene and has_checkpoint:
            route = "Persistent scene and checkpoint evidence are preserved. Use their explicit controls before starting a new attempt."
        elif has_checkpoint:
            route = "Checkpoint evidence is preserved. Validate/prepare it through the G3 checkpoint controls before starting a new attempt."
        else:
            route = "Persistent scene evidence is preserved. Save/reload it explicitly before starting a new attempt."
        return ReconstructionFailureRecoveryPresentation(
            state=ReconstructionFailureRecoveryState.PRESERVED_EVIDENCE,
            can_retry_from_source=False,
            text="Failure contained • evidence preserved" if failed else "Cancellation contained • evidence preserved",
            details=route + (f" Last failure: {error}" if error else ""),
            automatic_recovery=False,
            **common,
        )

    if not source_ready:
        return ReconstructionFailureRecoveryPresentation(
            state=ReconstructionFailureRecoveryState.SOURCE_REQUIRED,
            can_retry_from_source=False,
            text="Recovery needs a ready source",
            details=(
                "The previous attempt is terminal, but the selected source is not ready. Re-select or reconnect "
                "the source before retrying."
                + (f" Last failure: {error}" if error else "")
            ),
            automatic_recovery=False,
            **common,
        )

    if cancelled:
        return ReconstructionFailureRecoveryPresentation(
            state=ReconstructionFailureRecoveryState.CANCELLED_RETRY_AVAILABLE,
            can_retry_from_source=True,
            text="Cancelled safely • retry from source",
            details=(
                "Retry starts a fresh reconstruction session from the currently selected source. It does not "
                "resume a checkpoint or reuse the cancelled worker process."
            ),
            automatic_recovery=False,
            **common,
        )

    return ReconstructionFailureRecoveryPresentation(
        state=ReconstructionFailureRecoveryState.RETRY_AVAILABLE,
        can_retry_from_source=True,
        text="Failure contained • retry from source",
        details=(
            "Retry starts a fresh reconstruction session and fresh offline output when applicable. It does not "
            "resume a checkpoint or manually restart an in-flight live worker."
            + (f" Last failure: {error}" if error else "")
        ),
        automatic_recovery=False,
        **common,
    )


__all__ = [
    "ReconstructionFailureRecoveryPresentation",
    "ReconstructionFailureRecoveryState",
    "build_reconstruction_failure_recovery_presentation",
]

"""Phase 8.8G3 checkpoint/resume presentation and resume-context contracts.

G3 exposes the already-qualified whole-engine checkpoint/resume contracts to the
application UX. It does not invent another checkpoint format, execute a resumed
pipeline, recover failed runtimes, or change reconstruction authority. A resume
must remain an explicit two-part operation: a compatible current pipeline context
is validated first, then a bounded ``ReconstructionResumePreparation`` is handed
off for later execution by the existing isolated engine path.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Collection, Sequence

from app.reconstruction_session_model import (
    ReconstructionSessionIdentityStatus,
    ReconstructionSessionModel,
)
from core.reconstruction import (
    ReconstructionEngineCheckpoint,
    ReconstructionPipelinePlan,
    ReconstructionProviderDescriptor,
    ReconstructionResumeCompatibilityResult,
)


class ReconstructionCheckpointResumeState(str, Enum):
    UNAVAILABLE = "unavailable"
    RUNTIME_ACTIVE = "runtime_active"
    LOAD_ONLY = "load_only"
    CONTEXT_REQUIRED = "context_required"
    IDENTITY_CONFLICT = "identity_conflict"
    INCOMPATIBLE = "incompatible"
    COMPATIBLE = "compatible"
    PREPARED = "prepared"


@dataclass(frozen=True, slots=True)
class ReconstructionResumeContext:
    """Current generic pipeline identity used to validate one loaded checkpoint."""

    plan: ReconstructionPipelinePlan
    providers: tuple[ReconstructionProviderDescriptor, ...]
    available_payload_refs: frozenset[str]

    def __init__(
        self,
        *,
        plan: ReconstructionPipelinePlan,
        providers: Sequence[ReconstructionProviderDescriptor],
        available_payload_refs: Collection[str],
    ) -> None:
        if not isinstance(plan, ReconstructionPipelinePlan):
            raise TypeError("plan must be ReconstructionPipelinePlan.")
        provider_values = tuple(providers)
        if any(not isinstance(item, ReconstructionProviderDescriptor) for item in provider_values):
            raise TypeError("providers must contain ReconstructionProviderDescriptor values.")
        if len({item.provider_id for item in provider_values}) != len(provider_values):
            raise ValueError("resume-context providers must have unique provider IDs.")
        refs = frozenset(str(value).strip() for value in available_payload_refs if str(value).strip())
        object.__setattr__(self, "plan", plan)
        object.__setattr__(self, "providers", provider_values)
        object.__setattr__(self, "available_payload_refs", refs)


@dataclass(frozen=True, slots=True)
class ReconstructionCheckpointResumePresentation:
    """Passive G3 UI state for checkpoint loading and resume preparation."""

    state: ReconstructionCheckpointResumeState
    can_load_checkpoint: bool
    can_prepare_resume: bool
    text: str
    details: str
    next_stage_text: str
    reused_stage_count: int
    canonical_geometry_authoritative: bool
    certified_metrology: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.state, ReconstructionCheckpointResumeState):
            raise TypeError("state must be ReconstructionCheckpointResumeState.")
        if not isinstance(self.can_load_checkpoint, bool):
            raise TypeError("can_load_checkpoint must be bool.")
        if not isinstance(self.can_prepare_resume, bool):
            raise TypeError("can_prepare_resume must be bool.")
        text = str(self.text).strip()
        details = str(self.details).strip()
        next_stage = str(self.next_stage_text).strip()
        if not text or not details or not next_stage:
            raise ValueError("G3 checkpoint/resume presentation text must be non-empty.")
        reused = int(self.reused_stage_count)
        if reused < 0:
            raise ValueError("reused_stage_count must be non-negative.")
        if bool(self.certified_metrology):
            raise ValueError("G3 checkpoint/resume UX cannot certify metrology.")
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "details", details)
        object.__setattr__(self, "next_stage_text", next_stage)
        object.__setattr__(self, "reused_stage_count", reused)
        object.__setattr__(self, "canonical_geometry_authoritative", bool(self.canonical_geometry_authoritative))
        object.__setattr__(self, "certified_metrology", False)

    @property
    def status_kind(self) -> str:
        if self.state in {
            ReconstructionCheckpointResumeState.IDENTITY_CONFLICT,
            ReconstructionCheckpointResumeState.INCOMPATIBLE,
        }:
            return "warning"
        if self.state is ReconstructionCheckpointResumeState.RUNTIME_ACTIVE:
            return "running"
        if self.state in {
            ReconstructionCheckpointResumeState.COMPATIBLE,
            ReconstructionCheckpointResumeState.PREPARED,
        }:
            return "ready"
        return "neutral"


def build_reconstruction_checkpoint_resume_presentation(
    model: ReconstructionSessionModel | None,
    *,
    checkpoint: ReconstructionEngineCheckpoint | None,
    compatibility: ReconstructionResumeCompatibilityResult | None,
    context_available: bool,
    prepared: bool,
    controller_active: bool,
    controller_shutdown: bool,
) -> ReconstructionCheckpointResumePresentation:
    """Derive G3 checkpoint/resume UX without launching resumed execution."""

    if model is not None and not isinstance(model, ReconstructionSessionModel):
        raise TypeError("model must be ReconstructionSessionModel or None.")
    if checkpoint is not None and not isinstance(checkpoint, ReconstructionEngineCheckpoint):
        raise TypeError("checkpoint must be ReconstructionEngineCheckpoint or None.")
    if compatibility is not None and not isinstance(
        compatibility, ReconstructionResumeCompatibilityResult
    ):
        raise TypeError("compatibility must be ReconstructionResumeCompatibilityResult or None.")

    authority = bool(model and model.canonical_geometry_authoritative)
    next_stage = checkpoint.next_stage.value if checkpoint is not None else "—"
    reused = checkpoint.completed_stage_count if checkpoint is not None else 0

    if controller_shutdown:
        return ReconstructionCheckpointResumePresentation(
            state=ReconstructionCheckpointResumeState.UNAVAILABLE,
            can_load_checkpoint=False,
            can_prepare_resume=False,
            text="Checkpoint actions unavailable",
            details="The reconstruction controller is shut down; checkpoint interaction is disabled.",
            next_stage_text=next_stage,
            reused_stage_count=reused,
            canonical_geometry_authoritative=authority,
        )

    if controller_active:
        return ReconstructionCheckpointResumePresentation(
            state=ReconstructionCheckpointResumeState.RUNTIME_ACTIVE,
            can_load_checkpoint=False,
            can_prepare_resume=False,
            text="Checkpoint actions paused • runtime active",
            details=(
                "Checkpoint loading and resume preparation are idle-only. Stop or finish the active "
                "reconstruction before changing checkpoint state."
            ),
            next_stage_text=next_stage,
            reused_stage_count=reused,
            canonical_geometry_authoritative=authority,
        )

    if checkpoint is None:
        return ReconstructionCheckpointResumePresentation(
            state=ReconstructionCheckpointResumeState.LOAD_ONLY,
            can_load_checkpoint=True,
            can_prepare_resume=False,
            text="Load reconstruction checkpoint",
            details=(
                "Loading verifies the existing whole-engine checkpoint manifest and SHA-256 sidecar. "
                "It does not start or resume reconstruction."
            ),
            next_stage_text="—",
            reused_stage_count=0,
            canonical_geometry_authoritative=authority,
        )

    if model is not None and model.identity_status is ReconstructionSessionIdentityStatus.CONFLICT:
        return ReconstructionCheckpointResumePresentation(
            state=ReconstructionCheckpointResumeState.IDENTITY_CONFLICT,
            can_load_checkpoint=True,
            can_prepare_resume=False,
            text="Checkpoint loaded • session identity conflict",
            details=(
                "The loaded checkpoint does not agree with the focused runtime/scene session identity. "
                "G3 refuses resume preparation until the conflict is resolved explicitly."
            ),
            next_stage_text=next_stage,
            reused_stage_count=reused,
            canonical_geometry_authoritative=authority,
        )

    if not context_available:
        return ReconstructionCheckpointResumePresentation(
            state=ReconstructionCheckpointResumeState.CONTEXT_REQUIRED,
            can_load_checkpoint=True,
            can_prepare_resume=False,
            text="Checkpoint loaded • resume context required",
            details=(
                "A checkpoint is present, but the current pipeline plan/provider/payload context has not "
                "been registered. G3 will not guess compatibility from checkpoint metadata alone."
            ),
            next_stage_text=next_stage,
            reused_stage_count=reused,
            canonical_geometry_authoritative=authority,
        )

    if compatibility is None:
        return ReconstructionCheckpointResumePresentation(
            state=ReconstructionCheckpointResumeState.CONTEXT_REQUIRED,
            can_load_checkpoint=True,
            can_prepare_resume=False,
            text="Checkpoint loaded • validation pending",
            details="A current resume context exists, but checkpoint compatibility has not been validated yet.",
            next_stage_text=next_stage,
            reused_stage_count=reused,
            canonical_geometry_authoritative=authority,
        )

    if not compatibility.compatible:
        reasons = "; ".join(compatibility.reasons) or "Checkpoint compatibility failed."
        return ReconstructionCheckpointResumePresentation(
            state=ReconstructionCheckpointResumeState.INCOMPATIBLE,
            can_load_checkpoint=True,
            can_prepare_resume=False,
            text="Checkpoint incompatible • resume blocked",
            details=reasons,
            next_stage_text="—",
            reused_stage_count=compatibility.reused_stage_count,
            canonical_geometry_authoritative=authority,
        )

    next_compatible = compatibility.next_stage.value if compatibility.next_stage is not None else next_stage
    if prepared:
        return ReconstructionCheckpointResumePresentation(
            state=ReconstructionCheckpointResumeState.PREPARED,
            can_load_checkpoint=True,
            can_prepare_resume=False,
            text=f"Resume prepared • next {next_compatible}",
            details=(
                f"Compatibility is verified and {compatibility.reused_stage_count} committed stage(s) will be reused. "
                "G3 emits a preparation handoff only; it does not launch the resumed engine automatically."
            ),
            next_stage_text=next_compatible,
            reused_stage_count=compatibility.reused_stage_count,
            canonical_geometry_authoritative=authority,
        )

    return ReconstructionCheckpointResumePresentation(
        state=ReconstructionCheckpointResumeState.COMPATIBLE,
        can_load_checkpoint=True,
        can_prepare_resume=True,
        text=f"Checkpoint compatible • next {next_compatible}",
        details=(
            f"Compatibility is verified. Preparing resume will reuse {compatibility.reused_stage_count} committed "
            "stage(s) and hand the validated prefix to the existing engine path. No resumed execution is launched by G3."
        ),
        next_stage_text=next_compatible,
        reused_stage_count=compatibility.reused_stage_count,
        canonical_geometry_authoritative=authority,
    )


__all__ = [
    "ReconstructionCheckpointResumePresentation",
    "ReconstructionCheckpointResumeState",
    "ReconstructionResumeContext",
    "build_reconstruction_checkpoint_resume_presentation",
]

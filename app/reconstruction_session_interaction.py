"""Phase 8.8G2 save/reload presentation for persistent reconstruction scenes.

G2 deliberately reuses the existing application persistence bridge and the G1
session model. It does not introduce a second session format, copy external
artifact payloads, resume checkpoints, recover failed runtimes, or change any
reconstruction authority. The interaction is scene-focused and remains safe for
generic future reconstruction use cases.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.reconstruction_session_model import (
    ReconstructionSessionIdentityStatus,
    ReconstructionSessionModel,
)


class ReconstructionSceneInteractionState(str, Enum):
    UNAVAILABLE = "unavailable"
    RUNTIME_ACTIVE = "runtime_active"
    LOAD_ONLY = "load_only"
    READY = "ready"
    IDENTITY_CONFLICT = "identity_conflict"


@dataclass(frozen=True, slots=True)
class ReconstructionSceneInteractionPresentation:
    """Passive G2 availability model for scene load/save/reload actions."""

    state: ReconstructionSceneInteractionState
    can_load: bool
    can_save: bool
    can_reload: bool
    text: str
    details: str
    canonical_geometry_authoritative: bool
    certified_metrology: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.state, ReconstructionSceneInteractionState):
            raise TypeError("state must be ReconstructionSceneInteractionState.")
        for field_name in ("can_load", "can_save", "can_reload"):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be bool.")
        text = str(self.text).strip()
        details = str(self.details).strip()
        if not text or not details:
            raise ValueError("G2 interaction presentation text/details must be non-empty.")
        if bool(self.certified_metrology):
            raise ValueError("G2 scene interaction cannot certify metrology.")
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "details", details)
        object.__setattr__(
            self,
            "canonical_geometry_authoritative",
            bool(self.canonical_geometry_authoritative),
        )
        object.__setattr__(self, "certified_metrology", False)

    @property
    def status_kind(self) -> str:
        if self.state is ReconstructionSceneInteractionState.IDENTITY_CONFLICT:
            return "warning"
        if self.state is ReconstructionSceneInteractionState.RUNTIME_ACTIVE:
            return "running"
        if self.state is ReconstructionSceneInteractionState.READY:
            return "ready"
        return "neutral"


def build_reconstruction_scene_interaction_presentation(
    model: ReconstructionSessionModel | None,
    *,
    controller_active: bool,
    controller_shutdown: bool,
) -> ReconstructionSceneInteractionPresentation:
    """Derive bounded UI availability without performing persistence actions."""

    if model is not None and not isinstance(model, ReconstructionSessionModel):
        raise TypeError("model must be ReconstructionSessionModel or None.")

    has_scene = bool(model and model.scene_id)
    has_manifest = bool(model and model.scene_manifest)
    authority = bool(model and model.canonical_geometry_authoritative)

    if controller_shutdown:
        return ReconstructionSceneInteractionPresentation(
            state=ReconstructionSceneInteractionState.UNAVAILABLE,
            can_load=False,
            can_save=False,
            can_reload=False,
            text="Scene actions unavailable",
            details="The reconstruction controller is shut down; persistent-scene interaction is disabled.",
            canonical_geometry_authoritative=authority,
        )

    if controller_active:
        return ReconstructionSceneInteractionPresentation(
            state=ReconstructionSceneInteractionState.RUNTIME_ACTIVE,
            can_load=False,
            can_save=False,
            can_reload=False,
            text="Scene actions paused • runtime active",
            details=(
                "Load, save and reload are idle-only persistence actions. Stop or finish the active "
                "reconstruction before changing persistent scene state."
            ),
            canonical_geometry_authoritative=authority,
        )

    if not has_scene:
        return ReconstructionSceneInteractionPresentation(
            state=ReconstructionSceneInteractionState.LOAD_ONLY,
            can_load=True,
            can_save=False,
            can_reload=False,
            text="Load persistent scene",
            details=(
                "Load verifies the persistent-scene manifest checksum and all recorded file references. "
                "It does not start reconstruction or resume a checkpoint."
            ),
            canonical_geometry_authoritative=False,
        )

    conflict = model is not None and (
        model.identity_status is ReconstructionSessionIdentityStatus.CONFLICT
    )
    return ReconstructionSceneInteractionPresentation(
        state=(
            ReconstructionSceneInteractionState.IDENTITY_CONFLICT
            if conflict
            else ReconstructionSceneInteractionState.READY
        ),
        can_load=True,
        can_save=True,
        can_reload=has_manifest,
        text=(
            "Scene actions ready • identity conflict visible"
            if conflict
            else "Scene actions ready • guarded persistence"
        ),
        details=(
            "Load may select another persistent scene. Save writes the current scene manifest and checksum "
            "to a chosen folder; referenced artifact payloads remain at their recorded paths, so this is not an "
            "export bundle. Reload re-reads the current manifest and requires the same session identity. "
            "No checkpoint resume is executed."
        ),
        canonical_geometry_authoritative=authority,
    )


__all__ = [
    "ReconstructionSceneInteractionPresentation",
    "ReconstructionSceneInteractionState",
    "build_reconstruction_scene_interaction_presentation",
]

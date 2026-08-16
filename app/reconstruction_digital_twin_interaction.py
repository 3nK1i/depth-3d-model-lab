"""Phase 9.6D unified inspection/measurement/semantic interaction policy.

This module projects already-qualified Reconstruction interaction state.  It does
not pick geometry, derive measurements, alter semantic membership, infer QC, or
launch appearance backends. Canonical geometry remains the only picking and
measurement authority.

Interactive Gaussian is co-rendered in the canonical VisPy scene. Interactive
NeRF is a mouse-transparent appearance image driven by the canonical viewport
camera. During semantic inspection or ruler use, NeRF may be presentation-only
dimmed/frozen so canonical overlays stay legible; the user's persisted/display
setting is not mutated. Static appearance previews are intentionally treated as
unsynchronised display artifacts and are never promoted to picking surfaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


CANONICAL_VIEW = "canonical_geometry"
GAUSSIAN_VIEW = "gaussian_scene"
NERF_VIEW = "nerf_scene"
NERF_SEMANTIC_ASSIST_MAX_OPACITY_PERCENT = 60
NERF_MEASUREMENT_ASSIST_MAX_OPACITY_PERCENT = 40


class DigitalTwinInteractionMode(str, Enum):
    NO_SCENE = "no_scene"
    CANONICAL = "canonical"
    GAUSSIAN_INTERACTIVE = "gaussian_interactive"
    NERF_INTERACTIVE = "nerf_interactive"
    STATIC_PREVIEW = "static_preview"


@dataclass(frozen=True, slots=True)
class DigitalTwinInteractionPresentation:
    """Read-only Phase 9.6D interaction state.

    ``effective_nerf_opacity_cap_percent`` is a transient presentation cap only.
    It never changes the user's NeRF display setting or the persisted artifact.
    """

    mode: DigitalTwinInteractionMode
    scene_id: str
    active_view_representation: str
    canonical_surface_loaded: bool
    canonical_geometry_authoritative: bool
    semantic_context_available: bool
    semantic_selection_active: bool
    measurement_available: bool
    measurement_active: bool
    gaussian_interactive: bool
    nerf_interactive: bool
    static_preview: bool
    certified_metrology: bool = False
    picking_authority: str = "canonical_geometry"
    measurement_authority: str = "canonical_geometry"

    def __post_init__(self) -> None:
        if not isinstance(self.mode, DigitalTwinInteractionMode):
            raise TypeError("mode must be DigitalTwinInteractionMode.")
        scene_id = str(self.scene_id).strip()
        active = str(self.active_view_representation).strip().lower() or CANONICAL_VIEW
        if active not in {CANONICAL_VIEW, GAUSSIAN_VIEW, NERF_VIEW}:
            raise ValueError(f"Unsupported Digital Twin interaction representation: {active!r}")
        object.__setattr__(self, "scene_id", scene_id)
        object.__setattr__(self, "active_view_representation", active)
        for name in (
            "canonical_surface_loaded",
            "canonical_geometry_authoritative",
            "semantic_context_available",
            "semantic_selection_active",
            "measurement_available",
            "measurement_active",
            "gaussian_interactive",
            "nerf_interactive",
            "static_preview",
        ):
            object.__setattr__(self, name, bool(getattr(self, name)))
        if bool(self.certified_metrology):
            raise ValueError("Phase 9.6D interaction presentation cannot certify metrology.")
        object.__setattr__(self, "certified_metrology", False)

        appearance_count = sum(
            bool(value)
            for value in (self.gaussian_interactive, self.nerf_interactive, self.static_preview)
        )
        if appearance_count > 1:
            raise ValueError("Only one appearance interaction mode may be active at a time.")

        if self.mode is DigitalTwinInteractionMode.NO_SCENE:
            if scene_id or self.canonical_geometry_authoritative or self.canonical_surface_loaded:
                raise ValueError("NO_SCENE interaction cannot expose canonical scene state.")
            if self.semantic_context_available or self.semantic_selection_active:
                raise ValueError("NO_SCENE interaction cannot expose semantic scene state.")
            if self.measurement_available or self.measurement_active or appearance_count:
                raise ValueError("NO_SCENE interaction must fail closed without scene interaction.")
            object.__setattr__(self, "picking_authority", "none")
            object.__setattr__(self, "measurement_authority", "none")
            return

        if not scene_id or not self.canonical_geometry_authoritative:
            raise ValueError("Scene-bound interaction requires canonical geometry authority.")
        if str(self.picking_authority).strip().lower() != CANONICAL_VIEW:
            raise ValueError("Digital Twin picking authority must remain canonical geometry.")
        if str(self.measurement_authority).strip().lower() != CANONICAL_VIEW:
            raise ValueError("Digital Twin measurement authority must remain canonical geometry.")
        object.__setattr__(self, "picking_authority", CANONICAL_VIEW)
        object.__setattr__(self, "measurement_authority", CANONICAL_VIEW)

        if self.semantic_selection_active and not self.semantic_context_available:
            raise ValueError("Semantic selection requires available semantic context.")
        if self.measurement_available != self.canonical_surface_loaded:
            raise ValueError("Interactive ruler availability must follow the loaded canonical surface.")
        if self.measurement_active and not self.measurement_available:
            raise ValueError("Active measurement requires a loaded canonical surface.")
        if self.gaussian_interactive and active != GAUSSIAN_VIEW:
            raise ValueError("Interactive Gaussian state requires Gaussian as the active view.")
        if self.nerf_interactive and active != NERF_VIEW:
            raise ValueError("Interactive NeRF state requires NeRF as the active view.")
        if self.static_preview and active == CANONICAL_VIEW:
            raise ValueError("Static preview requires an appearance representation.")

        expected = DigitalTwinInteractionMode.CANONICAL
        if self.gaussian_interactive:
            expected = DigitalTwinInteractionMode.GAUSSIAN_INTERACTIVE
        elif self.nerf_interactive:
            expected = DigitalTwinInteractionMode.NERF_INTERACTIVE
        elif self.static_preview:
            expected = DigitalTwinInteractionMode.STATIC_PREVIEW
        if self.mode is not expected:
            raise ValueError("Digital Twin interaction mode does not match presentation state.")

    @property
    def effective_nerf_opacity_cap_percent(self) -> int | None:
        if not self.nerf_interactive:
            return None
        if self.measurement_active:
            return NERF_MEASUREMENT_ASSIST_MAX_OPACITY_PERCENT
        if self.semantic_selection_active:
            return NERF_SEMANTIC_ASSIST_MAX_OPACITY_PERCENT
        return None

    @property
    def appearance_frozen_for_measurement(self) -> bool:
        return bool(self.nerf_interactive and self.measurement_active)

    @property
    def static_preview_blocks_canonical_picking(self) -> bool:
        return self.static_preview

    @property
    def interaction_text(self) -> str:
        if self.mode is DigitalTwinInteractionMode.NO_SCENE:
            return "Waiting for canonical interaction"
        if self.mode is DigitalTwinInteractionMode.STATIC_PREVIEW:
            return "Static appearance preview • canonical interaction guarded"
        if self.measurement_active:
            if self.mode is DigitalTwinInteractionMode.NERF_INTERACTIVE:
                return "Canonical ruler active • NeRF frozen + dimmed"
            if self.mode is DigitalTwinInteractionMode.GAUSSIAN_INTERACTIVE:
                return "Canonical ruler active • Gaussian remains appearance only"
            return "Canonical surface ruler active"
        if self.semantic_selection_active:
            if self.mode is DigitalTwinInteractionMode.NERF_INTERACTIVE:
                return "Semantic inspection • NeRF dimmed for canonical overlay"
            if self.mode is DigitalTwinInteractionMode.GAUSSIAN_INTERACTIVE:
                return "Semantic inspection • Gaussian remains appearance context"
            return "Canonical semantic inspection active"
        if self.measurement_available and self.semantic_context_available:
            return "Canonical inspection + ruler ready"
        if self.measurement_available:
            return "Canonical ruler ready"
        if self.semantic_context_available:
            return "Semantic inspection ready • ruler unavailable"
        return "Canonical interaction ready"

    @property
    def semantic_text(self) -> str:
        if not self.semantic_context_available:
            return "No semantic context"
        if self.semantic_selection_active:
            return "Selected semantic object • canonical membership/context retained"
        return "Semantic context available • no object selected"

    @property
    def measurement_text(self) -> str:
        if self.measurement_active:
            return "Canonical surface ruler active • uncertainty/rejection preserved"
        if self.measurement_available:
            return "Canonical surface ruler available"
        return "Ruler unavailable • no loaded canonical surface"

    @property
    def authority_text(self) -> str:
        if self.mode is DigitalTwinInteractionMode.NO_SCENE:
            return "No scene interaction authority"
        return (
            "Canonical geometry owns picking + measurement • semantics are context only • "
            "appearance cannot create scale/QC/metrology authority"
        )

    @property
    def appearance_text(self) -> str:
        if self.mode is DigitalTwinInteractionMode.GAUSSIAN_INTERACTIVE:
            return "Gaussian co-rendered • canonical interaction retained"
        if self.mode is DigitalTwinInteractionMode.NERF_INTERACTIVE:
            if self.measurement_active:
                return "NeRF frame frozen/dimmed during ruler • canonical clicks pass through"
            if self.semantic_selection_active:
                return "NeRF dimmed transiently so canonical semantic overlay remains legible"
            return "NeRF mouse-transparent • canonical interaction retained"
        if self.mode is DigitalTwinInteractionMode.STATIC_PREVIEW:
            return "Static preview is not camera-synchronized • never a picking surface"
        return "No neural appearance interaction active"


def build_digital_twin_interaction_presentation(
    composition: Any = None,
    *,
    canonical_surface_loaded: bool = False,
    semantic_context_available: bool = False,
    semantic_selection_active: bool = False,
    measurement_active: bool = False,
    gaussian_interactive: bool = False,
    nerf_interactive: bool = False,
    gaussian_preview_visible: bool = False,
    nerf_preview_visible: bool = False,
) -> DigitalTwinInteractionPresentation:
    """Project existing scene/UI state without changing any authority domain."""

    scene_id = str(getattr(composition, "scene_id", "")).strip() if composition is not None else ""
    canonical_authority = bool(
        composition is not None and getattr(composition, "canonical_geometry_authoritative", False)
    )
    if not scene_id or not canonical_authority:
        return DigitalTwinInteractionPresentation(
            mode=DigitalTwinInteractionMode.NO_SCENE,
            scene_id="",
            active_view_representation=CANONICAL_VIEW,
            canonical_surface_loaded=False,
            canonical_geometry_authoritative=False,
            semantic_context_available=False,
            semantic_selection_active=False,
            measurement_available=False,
            measurement_active=False,
            gaussian_interactive=False,
            nerf_interactive=False,
            static_preview=False,
            picking_authority="none",
            measurement_authority="none",
        )

    active = str(getattr(composition, "active_view_representation", CANONICAL_VIEW)).strip().lower()
    gaussian_interactive = bool(gaussian_interactive)
    nerf_interactive = bool(nerf_interactive)
    static_preview = bool(gaussian_preview_visible or nerf_preview_visible)
    if bool(gaussian_preview_visible) and bool(nerf_preview_visible):
        raise ValueError("Gaussian and NeRF static previews cannot be visible simultaneously.")
    if gaussian_interactive and (nerf_interactive or static_preview):
        raise ValueError("Interactive Gaussian cannot coexist with another appearance interaction mode.")
    if nerf_interactive and static_preview:
        raise ValueError("Interactive NeRF cannot coexist with a static appearance preview.")

    mode = DigitalTwinInteractionMode.CANONICAL
    if gaussian_interactive:
        mode = DigitalTwinInteractionMode.GAUSSIAN_INTERACTIVE
    elif nerf_interactive:
        mode = DigitalTwinInteractionMode.NERF_INTERACTIVE
    elif static_preview:
        mode = DigitalTwinInteractionMode.STATIC_PREVIEW

    return DigitalTwinInteractionPresentation(
        mode=mode,
        scene_id=scene_id,
        active_view_representation=active,
        canonical_surface_loaded=bool(canonical_surface_loaded),
        canonical_geometry_authoritative=True,
        semantic_context_available=bool(semantic_context_available),
        semantic_selection_active=bool(semantic_selection_active),
        measurement_available=bool(canonical_surface_loaded),
        measurement_active=bool(measurement_active),
        gaussian_interactive=gaussian_interactive,
        nerf_interactive=nerf_interactive,
        static_preview=static_preview,
    )

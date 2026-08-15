"""Phase 9.6A unified Digital Twin workspace presentation and authority model.

This module composes already-qualified Reconstruction presentation state. It is
intentionally dependency-light and presentation-only: it does not load scene
payloads, mutate persistence, start neural runtimes, perform measurements, run
inspection/QC, or establish new geometry/metric authority.

Canonical geometry remains the spatial, picking and measurement authority.
Gaussian and NeRF remain optional appearance-only representations. QC/reference
remains a separate evidence domain and metrology is never certified here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


_CANONICAL_VIEW = "canonical_geometry"
_GAUSSIAN_VIEW = "gaussian_scene"
_NERF_VIEW = "nerf_scene"
_SUPPORTED_VIEWS = {_CANONICAL_VIEW, _GAUSSIAN_VIEW, _NERF_VIEW}


class DigitalTwinWorkspaceState(str, Enum):
    NO_SCENE = "no_scene"
    CANONICAL = "canonical"
    COMPOSED = "composed"
    DEGRADED = "degraded"


@dataclass(frozen=True, slots=True)
class DigitalTwinWorkspacePresentation:
    """Read-only composition of the existing Digital Twin presentation domains."""

    state: DigitalTwinWorkspaceState
    scene_id: str
    scene_state: str
    canonical_geometry_authoritative: bool
    canonical_surface_loaded: bool
    semantic_context_available: bool
    gaussian_available: bool
    nerf_available: bool
    active_view_representation: str
    gaussian_runtime_active: bool
    nerf_runtime_active: bool
    calibration_available: bool
    scale_state: str
    scale_text: str
    certified_metrology: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.state, DigitalTwinWorkspaceState):
            raise TypeError("state must be DigitalTwinWorkspaceState.")
        active = str(self.active_view_representation).strip().lower()
        if active not in _SUPPORTED_VIEWS:
            raise ValueError(f"Unsupported Digital Twin active view: {active or '<empty>'}")
        object.__setattr__(self, "active_view_representation", active)
        object.__setattr__(self, "scene_id", str(self.scene_id).strip())
        object.__setattr__(self, "scene_state", str(self.scene_state).strip().lower())
        object.__setattr__(self, "scale_state", str(self.scale_state).strip().lower() or "no_scene")
        object.__setattr__(self, "scale_text", str(self.scale_text).strip() or "—")
        for field_name in (
            "canonical_geometry_authoritative",
            "canonical_surface_loaded",
            "semantic_context_available",
            "gaussian_available",
            "nerf_available",
            "gaussian_runtime_active",
            "nerf_runtime_active",
            "calibration_available",
        ):
            object.__setattr__(self, field_name, bool(getattr(self, field_name)))
        if bool(self.certified_metrology):
            raise ValueError("Phase 9.6A workspace composition cannot certify metrology.")
        object.__setattr__(self, "certified_metrology", False)

        has_scene = bool(self.scene_id)
        if self.state is DigitalTwinWorkspaceState.NO_SCENE:
            if has_scene or self.canonical_geometry_authoritative:
                raise ValueError("NO_SCENE workspace cannot expose canonical scene authority.")
            if active != _CANONICAL_VIEW:
                raise ValueError("NO_SCENE workspace must fail closed to the canonical view.")
        else:
            if not has_scene:
                raise ValueError("Available Digital Twin workspace requires persistent scene identity.")
            if not self.canonical_geometry_authoritative:
                raise ValueError("Digital Twin workspace requires canonical geometry authority.")

        if active == _GAUSSIAN_VIEW and not self.gaussian_available:
            raise ValueError("Gaussian view cannot be active without a persisted Gaussian representation.")
        if active == _NERF_VIEW and not self.nerf_available:
            raise ValueError("NeRF view cannot be active without a persisted NeRF representation.")
        if self.gaussian_runtime_active and active != _GAUSSIAN_VIEW:
            raise ValueError("Gaussian runtime/display activity must remain bound to the Gaussian view.")
        if self.nerf_runtime_active and active != _NERF_VIEW:
            raise ValueError("NeRF runtime activity must remain bound to the NeRF view.")
        if self.gaussian_runtime_active and self.nerf_runtime_active:
            raise ValueError("Gaussian and NeRF heavyweight presentation activity cannot be active together.")

    @property
    def workspace_text(self) -> str:
        if self.state is DigitalTwinWorkspaceState.NO_SCENE:
            return "No persistent Digital Twin scene"
        if self.state is DigitalTwinWorkspaceState.DEGRADED:
            return "Degraded Digital Twin • canonical authority retained"
        if self.state is DigitalTwinWorkspaceState.COMPOSED:
            return "Unified Digital Twin • composed scene"
        return "Canonical Digital Twin • ready"

    @property
    def active_view_text(self) -> str:
        if self.active_view_representation == _GAUSSIAN_VIEW:
            return "Gaussian appearance • canonical spatial authority"
        if self.active_view_representation == _NERF_VIEW:
            return "NeRF appearance • canonical spatial authority"
        return "Canonical geometry"

    @property
    def composition_text(self) -> str:
        if self.state is DigitalTwinWorkspaceState.NO_SCENE:
            return "Waiting for persistent scene"
        layers = ["Canonical"]
        if self.semantic_context_available:
            layers.append("Semantics")
        if self.gaussian_available:
            layers.append("Gaussian")
        if self.nerf_available:
            layers.append("NeRF")
        return " • ".join(layers)

    @property
    def authority_text(self) -> str:
        if self.state is DigitalTwinWorkspaceState.NO_SCENE:
            return "No canonical scene authority"
        return "Canonical geometry • picking + measurement"

    @property
    def safety_text(self) -> str:
        return "Appearance only overlays • QC/reference separate • metrology not certified"

    @property
    def resource_text(self) -> str:
        if self.nerf_runtime_active:
            return "NeRF isolated worker active • Gaussian not active"
        if self.gaussian_runtime_active:
            return "Gaussian interactive display active • NeRF worker idle"
        if self.gaussian_available or self.nerf_available:
            return "Neural appearance idle/on demand"
        return "No neural appearance resource required"

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "scene_id": self.scene_id,
            "scene_state": self.scene_state,
            "canonical_geometry_authoritative": self.canonical_geometry_authoritative,
            "canonical_surface_loaded": self.canonical_surface_loaded,
            "semantic_context_available": self.semantic_context_available,
            "gaussian_available": self.gaussian_available,
            "nerf_available": self.nerf_available,
            "active_view_representation": self.active_view_representation,
            "gaussian_runtime_active": self.gaussian_runtime_active,
            "nerf_runtime_active": self.nerf_runtime_active,
            "calibration_available": self.calibration_available,
            "scale_state": self.scale_state,
            "scale_text": self.scale_text,
            "certified_metrology": False,
            "geometry_authority": "canonical_geometry",
            "picking_authority": "canonical_geometry",
            "measurement_authority": "canonical_geometry",
            "appearance_authority": "appearance_only",
            "qc_reference_authority": "separate_evidence_domain",
        }


def _appearance_kinds(presentation: Any) -> set[str]:
    if presentation is None:
        return set()
    result: set[str] = set()
    for entry in tuple(getattr(presentation, "entries", ()) or ()):
        kind = str(getattr(entry, "representation_kind", "")).strip().lower()
        if kind in {_GAUSSIAN_VIEW, _NERF_VIEW}:
            if bool(getattr(entry, "canonical_geometry_authoritative", False)):
                raise ValueError("Appearance representation unexpectedly claims canonical geometry authority.")
            if bool(getattr(entry, "certified_metrology", False)):
                raise ValueError("Appearance representation unexpectedly claims certified metrology.")
            result.add(kind)
    return result


def build_digital_twin_workspace_presentation(
    canonical_scene: Any = None,
    *,
    canonical_surface_loaded: bool = False,
    semantic_context_available: bool = False,
    appearance_presentation: Any = None,
    active_view_representation: str = _CANONICAL_VIEW,
    gaussian_runtime_active: bool = False,
    nerf_runtime_active: bool = False,
) -> DigitalTwinWorkspacePresentation:
    """Compose existing qualified presentation state without changing authority."""

    available = bool(canonical_scene is not None and getattr(canonical_scene, "available", False))
    if not available:
        return DigitalTwinWorkspacePresentation(
            state=DigitalTwinWorkspaceState.NO_SCENE,
            scene_id="",
            scene_state="",
            canonical_geometry_authoritative=False,
            canonical_surface_loaded=False,
            semantic_context_available=False,
            gaussian_available=False,
            nerf_available=False,
            active_view_representation=_CANONICAL_VIEW,
            gaussian_runtime_active=False,
            nerf_runtime_active=False,
            calibration_available=False,
            scale_state="no_scene",
            scale_text="—",
            certified_metrology=False,
        )

    if bool(getattr(canonical_scene, "certified_metrology", False)):
        raise ValueError("Canonical scene presentation unexpectedly claims certified metrology.")
    canonical_authority = bool(getattr(canonical_scene, "canonical_geometry_authoritative", False))
    if not canonical_authority:
        raise ValueError("Digital Twin composition requires canonical geometry authority.")

    kinds = _appearance_kinds(appearance_presentation)
    active = str(active_view_representation).strip().lower() or _CANONICAL_VIEW
    scene_state = str(getattr(canonical_scene, "scene_state", "")).strip().lower()
    supporting = bool(semantic_context_available or kinds)
    if scene_state == "degraded":
        state = DigitalTwinWorkspaceState.DEGRADED
    elif supporting:
        state = DigitalTwinWorkspaceState.COMPOSED
    else:
        state = DigitalTwinWorkspaceState.CANONICAL

    raw_scale_state = getattr(canonical_scene, "scale_state", "not_qualified")
    scale_state = str(getattr(raw_scale_state, "value", raw_scale_state)).strip().lower()
    scale_text = str(getattr(canonical_scene, "scale_text", "")).strip() or "Not qualified"

    return DigitalTwinWorkspacePresentation(
        state=state,
        scene_id=str(getattr(canonical_scene, "scene_id", "")).strip(),
        scene_state=scene_state,
        canonical_geometry_authoritative=True,
        canonical_surface_loaded=bool(canonical_surface_loaded),
        semantic_context_available=bool(semantic_context_available),
        gaussian_available=_GAUSSIAN_VIEW in kinds,
        nerf_available=_NERF_VIEW in kinds,
        active_view_representation=active,
        gaussian_runtime_active=bool(gaussian_runtime_active),
        nerf_runtime_active=bool(nerf_runtime_active),
        calibration_available=bool(getattr(canonical_scene, "calibration_available", False)),
        scale_state=scale_state,
        scale_text=scale_text,
        certified_metrology=False,
    )


__all__ = [
    "DigitalTwinWorkspacePresentation",
    "DigitalTwinWorkspaceState",
    "build_digital_twin_workspace_presentation",
]

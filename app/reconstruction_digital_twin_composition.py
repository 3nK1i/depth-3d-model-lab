"""Phase 9.6B unified Digital Twin scene-composition projection.

This module describes how already-qualified Reconstruction presentation layers
coexist in one persistent Digital Twin workspace. It is deliberately read-only
and dependency-light: it does not load resources, render, mutate persistence,
launch neural runtimes, perform measurement, or establish metric/QC authority.

Canonical geometry remains the sole spatial, picking and measurement authority.
Semantic objects and cameras are supporting context. Gaussian and NeRF are
optional appearance-only layers and can never become geometry/scale/QC authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


_CANONICAL = "canonical_surface"
_SEMANTICS = "semantic_objects"
_CAMERAS = "world_cameras"
_GAUSSIAN = "gaussian_scene"
_NERF = "nerf_scene"
_CANONICAL_VIEW = "canonical_geometry"
_SUPPORTED_VIEWS = {_CANONICAL_VIEW, _GAUSSIAN, _NERF}


class DigitalTwinLayerRole(str, Enum):
    AUTHORITATIVE_GEOMETRY = "authoritative_geometry"
    SUPPORTING_CONTEXT = "supporting_context"
    APPEARANCE_ONLY = "appearance_only"


class DigitalTwinCompositionState(str, Enum):
    NO_SCENE = "no_scene"
    CANONICAL = "canonical"
    COMPOSED = "composed"
    DEGRADED = "degraded"


@dataclass(frozen=True, slots=True)
class DigitalTwinCompositionLayer:
    key: str
    label: str
    role: DigitalTwinLayerRole
    available: bool
    visible: bool
    source_id: str = ""
    canonical_geometry_authoritative: bool = False
    certified_metrology: bool = False

    def __post_init__(self) -> None:
        key = str(self.key).strip().lower()
        if key not in {_CANONICAL, _SEMANTICS, _CAMERAS, _GAUSSIAN, _NERF}:
            raise ValueError(f"Unsupported Digital Twin composition layer: {key or '<empty>'}")
        if not isinstance(self.role, DigitalTwinLayerRole):
            raise TypeError("role must be DigitalTwinLayerRole.")
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "label", str(self.label).strip() or key.replace("_", " ").title())
        object.__setattr__(self, "source_id", str(self.source_id).strip())
        object.__setattr__(self, "available", bool(self.available))
        object.__setattr__(self, "visible", bool(self.visible))
        if self.visible and not self.available:
            raise ValueError(f"Visible Digital Twin layer {key!r} must be available.")
        if bool(self.certified_metrology):
            raise ValueError("Digital Twin composition layers cannot certify metrology.")
        object.__setattr__(self, "certified_metrology", False)

        authority = bool(self.canonical_geometry_authoritative)
        if key == _CANONICAL:
            if self.role is not DigitalTwinLayerRole.AUTHORITATIVE_GEOMETRY:
                raise ValueError("Canonical surface must retain authoritative-geometry role.")
            if self.available and not authority:
                raise ValueError("Available canonical surface must retain canonical geometry authority.")
        else:
            if authority:
                raise ValueError(f"Non-canonical layer {key!r} cannot claim canonical geometry authority.")
            if key in {_GAUSSIAN, _NERF} and self.role is not DigitalTwinLayerRole.APPEARANCE_ONLY:
                raise ValueError("Gaussian and NeRF layers must remain appearance only.")
            if key in {_SEMANTICS, _CAMERAS} and self.role is not DigitalTwinLayerRole.SUPPORTING_CONTEXT:
                raise ValueError("Semantic and camera layers must remain supporting context.")
        object.__setattr__(self, "canonical_geometry_authoritative", authority)

    @property
    def authority_text(self) -> str:
        if self.key == _CANONICAL:
            return "Canonical spatial • picking • measurement authority"
        if self.key == _SEMANTICS:
            return "Semantic membership/context only"
        if self.key == _CAMERAS:
            return "Camera/world context only • not scale evidence"
        return "Appearance only • no geometry/scale/measurement/QC authority"


@dataclass(frozen=True, slots=True)
class DigitalTwinSceneComposition:
    state: DigitalTwinCompositionState
    scene_id: str
    active_view_representation: str
    layers: tuple[DigitalTwinCompositionLayer, ...]
    canonical_geometry_authoritative: bool
    certified_metrology: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.state, DigitalTwinCompositionState):
            raise TypeError("state must be DigitalTwinCompositionState.")
        scene_id = str(self.scene_id).strip()
        active = str(self.active_view_representation).strip().lower() or _CANONICAL_VIEW
        if active not in _SUPPORTED_VIEWS:
            raise ValueError(f"Unsupported active Digital Twin representation: {active!r}")
        layers = tuple(self.layers)
        if any(not isinstance(layer, DigitalTwinCompositionLayer) for layer in layers):
            raise TypeError("layers must contain DigitalTwinCompositionLayer values.")
        if len({layer.key for layer in layers}) != len(layers):
            raise ValueError("Digital Twin composition layer keys must be unique.")
        if bool(self.certified_metrology):
            raise ValueError("Phase 9.6B composition cannot certify metrology.")
        object.__setattr__(self, "scene_id", scene_id)
        object.__setattr__(self, "active_view_representation", active)
        object.__setattr__(self, "layers", layers)
        object.__setattr__(self, "certified_metrology", False)

        canonical = self.layer(_CANONICAL)
        if self.state is DigitalTwinCompositionState.NO_SCENE:
            if scene_id or bool(self.canonical_geometry_authoritative) or any(layer.available for layer in layers):
                raise ValueError("NO_SCENE composition cannot expose scene layers or canonical authority.")
            if active != _CANONICAL_VIEW:
                raise ValueError("NO_SCENE composition must fail closed to canonical view.")
            return

        if not scene_id or not bool(self.canonical_geometry_authoritative):
            raise ValueError("Digital Twin scene composition requires persistent canonical scene authority.")
        if canonical is None or not canonical.available or not canonical.canonical_geometry_authoritative:
            raise ValueError("Digital Twin scene composition requires the canonical authority layer.")
        object.__setattr__(self, "canonical_geometry_authoritative", True)

        gaussian = self.layer(_GAUSSIAN)
        nerf = self.layer(_NERF)
        if active == _GAUSSIAN and not bool(gaussian and gaussian.available):
            raise ValueError("Gaussian cannot be active without a scene-bound persisted Gaussian layer.")
        if active == _NERF and not bool(nerf and nerf.available):
            raise ValueError("NeRF cannot be active without a scene-bound persisted NeRF layer.")
        visible_appearance = [layer for layer in (gaussian, nerf) if layer is not None and layer.visible]
        if len(visible_appearance) > 1:
            raise ValueError("Gaussian and NeRF appearance layers cannot be visible simultaneously.")
        if active == _GAUSSIAN and gaussian is not None and not gaussian.visible:
            raise ValueError("Active Gaussian representation must be visible.")
        if active == _NERF and nerf is not None and not nerf.visible:
            raise ValueError("Active NeRF representation must be visible.")

    def layer(self, key: str) -> DigitalTwinCompositionLayer | None:
        needle = str(key).strip().lower()
        return next((layer for layer in self.layers if layer.key == needle), None)

    @property
    def available_layers(self) -> tuple[DigitalTwinCompositionLayer, ...]:
        return tuple(layer for layer in self.layers if layer.available)

    @property
    def visible_layers(self) -> tuple[DigitalTwinCompositionLayer, ...]:
        return tuple(layer for layer in self.layers if layer.visible)

    @property
    def composition_text(self) -> str:
        if self.state is DigitalTwinCompositionState.NO_SCENE:
            return "Waiting for persistent scene"
        return " • ".join(layer.label for layer in self.available_layers)

    @property
    def visibility_text(self) -> str:
        if self.state is DigitalTwinCompositionState.NO_SCENE:
            return "No scene layers visible"
        visible = self.visible_layers
        if not visible:
            return "Scene evidence loaded • all optional display layers hidden"
        return "Visible: " + " • ".join(layer.label for layer in visible)

    @property
    def authority_text(self) -> str:
        if self.state is DigitalTwinCompositionState.NO_SCENE:
            return "No canonical scene authority"
        return "Canonical geometry only • semantics/cameras support • neural layers appearance only"

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "scene_id": self.scene_id,
            "active_view_representation": self.active_view_representation,
            "canonical_geometry_authoritative": bool(self.canonical_geometry_authoritative),
            "certified_metrology": False,
            "picking_authority": "canonical_geometry" if self.scene_id else "none",
            "measurement_authority": "canonical_geometry" if self.scene_id else "none",
            "layers": tuple(
                {
                    "key": layer.key,
                    "label": layer.label,
                    "role": layer.role.value,
                    "available": layer.available,
                    "visible": layer.visible,
                    "source_id": layer.source_id,
                    "canonical_geometry_authoritative": layer.canonical_geometry_authoritative,
                    "certified_metrology": False,
                }
                for layer in self.layers
            ),
        }


def _scene_id(value: Any) -> str:
    return str(getattr(value, "scene_id", "")).strip() if value is not None else ""


def _validate_scene_bound(name: str, presentation: Any, expected_scene_id: str) -> None:
    if presentation is None:
        return
    actual = _scene_id(presentation)
    if actual and actual != expected_scene_id:
        raise ValueError(f"{name} scene identity does not match the canonical Digital Twin scene.")
    if bool(getattr(presentation, "certified_metrology", False)):
        raise ValueError(f"{name} presentation unexpectedly claims certified metrology.")


def _appearance_entries(presentation: Any, expected_scene_id: str) -> dict[str, Any]:
    if presentation is None:
        return {}
    _validate_scene_bound("Appearance", presentation, expected_scene_id)
    result: dict[str, Any] = {}
    for entry in tuple(getattr(presentation, "entries", ()) or ()):
        kind = str(getattr(entry, "representation_kind", "")).strip().lower()
        if kind not in {_GAUSSIAN, _NERF}:
            continue
        if bool(getattr(entry, "canonical_geometry_authoritative", False)):
            raise ValueError("Appearance layer unexpectedly claims canonical geometry authority.")
        if bool(getattr(entry, "certified_metrology", False)):
            raise ValueError("Appearance layer unexpectedly claims certified metrology.")
        result[kind] = entry
    return result


def build_digital_twin_scene_composition(
    canonical_scene: Any = None,
    *,
    canonical_surface_loaded: bool = False,
    canonical_surface_visible: bool = False,
    semantic_presentation: Any = None,
    semantic_visible: bool = False,
    camera_presentation: Any = None,
    camera_visible: bool = False,
    appearance_presentation: Any = None,
    active_view_representation: str = _CANONICAL_VIEW,
    gaussian_visible: bool = False,
    nerf_visible: bool = False,
) -> DigitalTwinSceneComposition:
    """Compose existing scene-bound presentation state without changing authority."""

    available = bool(canonical_scene is not None and getattr(canonical_scene, "available", False))
    if not available:
        empty = tuple(
            DigitalTwinCompositionLayer(key, label, role, False, False)
            for key, label, role in (
                (_CANONICAL, "Canonical", DigitalTwinLayerRole.AUTHORITATIVE_GEOMETRY),
                (_SEMANTICS, "Semantics", DigitalTwinLayerRole.SUPPORTING_CONTEXT),
                (_CAMERAS, "Cameras", DigitalTwinLayerRole.SUPPORTING_CONTEXT),
                (_GAUSSIAN, "Gaussian", DigitalTwinLayerRole.APPEARANCE_ONLY),
                (_NERF, "NeRF", DigitalTwinLayerRole.APPEARANCE_ONLY),
            )
        )
        return DigitalTwinSceneComposition(
            state=DigitalTwinCompositionState.NO_SCENE,
            scene_id="",
            active_view_representation=_CANONICAL_VIEW,
            layers=empty,
            canonical_geometry_authoritative=False,
            certified_metrology=False,
        )

    if bool(getattr(canonical_scene, "certified_metrology", False)):
        raise ValueError("Canonical scene presentation unexpectedly claims certified metrology.")
    if not bool(getattr(canonical_scene, "canonical_geometry_authoritative", False)):
        raise ValueError("Digital Twin scene composition requires canonical geometry authority.")
    scene_id = _scene_id(canonical_scene)
    if not scene_id:
        raise ValueError("Digital Twin scene composition requires persistent scene identity.")

    _validate_scene_bound("Semantic", semantic_presentation, scene_id)
    _validate_scene_bound("Camera", camera_presentation, scene_id)
    appearance = _appearance_entries(appearance_presentation, scene_id)

    semantics_available = semantic_presentation is not None
    cameras_available = camera_presentation is not None
    gaussian_available = _GAUSSIAN in appearance
    nerf_available = _NERF in appearance
    active = str(active_view_representation).strip().lower() or _CANONICAL_VIEW

    layers = (
        DigitalTwinCompositionLayer(
            _CANONICAL,
            "Canonical",
            DigitalTwinLayerRole.AUTHORITATIVE_GEOMETRY,
            True,
            bool(canonical_surface_loaded and canonical_surface_visible),
            source_id=str(getattr(canonical_scene, "artifact_id", "")).strip(),
            canonical_geometry_authoritative=True,
        ),
        DigitalTwinCompositionLayer(
            _SEMANTICS,
            "Semantics",
            DigitalTwinLayerRole.SUPPORTING_CONTEXT,
            semantics_available,
            bool(semantics_available and semantic_visible),
            source_id=str(getattr(semantic_presentation, "semantic_artifact_id", "")).strip(),
        ),
        DigitalTwinCompositionLayer(
            _CAMERAS,
            "Cameras",
            DigitalTwinLayerRole.SUPPORTING_CONTEXT,
            cameras_available,
            bool(cameras_available and camera_visible),
            source_id=str(getattr(camera_presentation, "artifact_id", "")).strip(),
        ),
        DigitalTwinCompositionLayer(
            _GAUSSIAN,
            "Gaussian",
            DigitalTwinLayerRole.APPEARANCE_ONLY,
            gaussian_available,
            bool(gaussian_available and gaussian_visible),
            source_id=str(getattr(appearance.get(_GAUSSIAN), "artifact_id", "")).strip(),
        ),
        DigitalTwinCompositionLayer(
            _NERF,
            "NeRF",
            DigitalTwinLayerRole.APPEARANCE_ONLY,
            nerf_available,
            bool(nerf_available and nerf_visible),
            source_id=str(getattr(appearance.get(_NERF), "artifact_id", "")).strip(),
        ),
    )

    scene_state = str(getattr(canonical_scene, "scene_state", "")).strip().lower()
    if scene_state == "degraded":
        state = DigitalTwinCompositionState.DEGRADED
    elif any(layer.available for layer in layers[1:]):
        state = DigitalTwinCompositionState.COMPOSED
    else:
        state = DigitalTwinCompositionState.CANONICAL

    return DigitalTwinSceneComposition(
        state=state,
        scene_id=scene_id,
        active_view_representation=active,
        layers=layers,
        canonical_geometry_authoritative=True,
        certified_metrology=False,
    )


__all__ = [
    "DigitalTwinCompositionLayer",
    "DigitalTwinCompositionState",
    "DigitalTwinLayerRole",
    "DigitalTwinSceneComposition",
    "build_digital_twin_scene_composition",
]

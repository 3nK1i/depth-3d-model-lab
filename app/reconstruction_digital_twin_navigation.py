"""Phase 9.6C cross-representation navigation/synchronization projection.

The Reconstruction viewport already owns one qualified turntable presentation
camera. Interactive Gaussian appearance is rendered in that same VisPy scene,
and interactive NeRF requests are derived from the same canvas camera snapshot.
This module describes that existing relationship without creating another
camera, renderer, spatial frame, or persistence authority.

Canonical geometry remains the sole spatial, picking and measurement authority.
The shared viewport camera is presentation/navigation state only. Persisted
canonical camera poses remain supporting context and do not become scale
evidence. Gaussian and NeRF remain appearance-only consumers of navigation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Any, Mapping


_CANONICAL_VIEW = "canonical_geometry"
_GAUSSIAN_VIEW = "gaussian_scene"
_NERF_VIEW = "nerf_scene"
_SUPPORTED_VIEWS = {_CANONICAL_VIEW, _GAUSSIAN_VIEW, _NERF_VIEW}


class DigitalTwinNavigationMode(str, Enum):
    NO_SCENE = "no_scene"
    CANONICAL = "canonical"
    GAUSSIAN_INTERACTIVE = "gaussian_interactive"
    NERF_INTERACTIVE = "nerf_interactive"
    STATIC_PREVIEW = "static_preview"


@dataclass(frozen=True, slots=True)
class DigitalTwinNavigationSync:
    """Read-only Phase 9.6C view synchronization state.

    ``navigation_source`` deliberately names the presentation camera rather than
    an authority domain. The camera can drive appearance rendering, but it does
    not supersede canonical scene geometry or establish metric scale.
    """

    mode: DigitalTwinNavigationMode
    scene_id: str
    active_view_representation: str
    center_viewer: tuple[float, float, float] | None
    distance: float | None
    azimuth_degrees: float | None
    elevation_degrees: float | None
    fov_degrees: float | None
    gaussian_interactive: bool
    nerf_interactive: bool
    static_preview: bool
    view_preserved_on_same_scene_reload: bool
    canonical_geometry_authoritative: bool
    certified_metrology: bool = False
    navigation_source: str = "canonical_viewport_camera"
    spatial_authority: str = "canonical_geometry"

    def __post_init__(self) -> None:
        if not isinstance(self.mode, DigitalTwinNavigationMode):
            raise TypeError("mode must be DigitalTwinNavigationMode.")
        scene_id = str(self.scene_id).strip()
        active = str(self.active_view_representation).strip().lower() or _CANONICAL_VIEW
        if active not in _SUPPORTED_VIEWS:
            raise ValueError(f"Unsupported Digital Twin navigation representation: {active!r}")
        if bool(self.certified_metrology):
            raise ValueError("Phase 9.6C navigation cannot certify metrology.")
        navigation_source = str(self.navigation_source).strip().lower()
        spatial_authority = str(self.spatial_authority).strip().lower()

        object.__setattr__(self, "scene_id", scene_id)
        object.__setattr__(self, "active_view_representation", active)
        object.__setattr__(self, "gaussian_interactive", bool(self.gaussian_interactive))
        object.__setattr__(self, "nerf_interactive", bool(self.nerf_interactive))
        object.__setattr__(self, "static_preview", bool(self.static_preview))
        object.__setattr__(
            self,
            "view_preserved_on_same_scene_reload",
            bool(self.view_preserved_on_same_scene_reload),
        )
        object.__setattr__(self, "certified_metrology", False)

        appearance_flags = sum(
            bool(value)
            for value in (
                self.gaussian_interactive,
                self.nerf_interactive,
                self.static_preview,
            )
        )
        if appearance_flags > 1:
            raise ValueError("Digital Twin navigation permits only one active appearance navigation mode.")
        if self.gaussian_interactive and active != _GAUSSIAN_VIEW:
            raise ValueError("Interactive Gaussian navigation requires Gaussian as the active appearance view.")
        if self.nerf_interactive and active != _NERF_VIEW:
            raise ValueError("Interactive NeRF navigation requires NeRF as the active appearance view.")
        if self.static_preview and active == _CANONICAL_VIEW:
            raise ValueError("Static appearance preview requires an appearance representation as the active view.")

        if self.mode is DigitalTwinNavigationMode.NO_SCENE:
            if navigation_source not in {"", "none"} or spatial_authority not in {"", "none"}:
                raise ValueError("NO_SCENE navigation cannot expose navigation or spatial authority.")
            object.__setattr__(self, "navigation_source", "none")
            object.__setattr__(self, "spatial_authority", "none")
            if scene_id or bool(self.canonical_geometry_authoritative):
                raise ValueError("NO_SCENE navigation cannot expose scene identity or canonical authority.")
            if active != _CANONICAL_VIEW or appearance_flags:
                raise ValueError("NO_SCENE navigation must fail closed to canonical view.")
            if any(
                value is not None
                for value in (
                    self.center_viewer,
                    self.distance,
                    self.azimuth_degrees,
                    self.elevation_degrees,
                    self.fov_degrees,
                )
            ):
                raise ValueError("NO_SCENE navigation must not expose a scene camera snapshot.")
            return

        if navigation_source != "canonical_viewport_camera":
            raise ValueError("Digital Twin navigation must remain sourced from the canonical viewport camera.")
        if spatial_authority != "canonical_geometry":
            raise ValueError("Digital Twin navigation cannot establish a separate spatial authority.")
        object.__setattr__(self, "navigation_source", "canonical_viewport_camera")
        object.__setattr__(self, "spatial_authority", "canonical_geometry")
        if not scene_id or not bool(self.canonical_geometry_authoritative):
            raise ValueError("Scene-bound Digital Twin navigation requires canonical scene authority.")
        object.__setattr__(self, "canonical_geometry_authoritative", True)
        center = self.center_viewer
        if center is None or len(center) != 3:
            raise ValueError("Scene-bound Digital Twin navigation requires a 3D viewport center.")
        normalized_center = tuple(float(value) for value in center)
        if not all(isfinite(value) for value in normalized_center):
            raise ValueError("Digital Twin viewport center must be finite.")
        object.__setattr__(self, "center_viewer", normalized_center)

        numeric = {
            "distance": self.distance,
            "azimuth_degrees": self.azimuth_degrees,
            "elevation_degrees": self.elevation_degrees,
            "fov_degrees": self.fov_degrees,
        }
        for name, value in numeric.items():
            if value is None or not isfinite(float(value)):
                raise ValueError(f"Digital Twin navigation {name} must be finite.")
            object.__setattr__(self, name, float(value))
        if float(self.distance) <= 0.0:
            raise ValueError("Digital Twin navigation distance must be positive.")
        if not 5.0 <= float(self.fov_degrees) <= 140.0:
            raise ValueError("Digital Twin navigation field of view is outside the supported range.")

        expected_mode = DigitalTwinNavigationMode.CANONICAL
        if self.gaussian_interactive:
            expected_mode = DigitalTwinNavigationMode.GAUSSIAN_INTERACTIVE
        elif self.nerf_interactive:
            expected_mode = DigitalTwinNavigationMode.NERF_INTERACTIVE
        elif self.static_preview:
            expected_mode = DigitalTwinNavigationMode.STATIC_PREVIEW
        if self.mode is not expected_mode:
            raise ValueError("Digital Twin navigation mode does not match the active presentation state.")

    @property
    def navigation_text(self) -> str:
        if self.mode is DigitalTwinNavigationMode.NO_SCENE:
            return "Waiting for persistent scene"
        if self.mode is DigitalTwinNavigationMode.GAUSSIAN_INTERACTIVE:
            return "Shared canonical camera • Gaussian co-navigates"
        if self.mode is DigitalTwinNavigationMode.NERF_INTERACTIVE:
            return "Shared canonical camera • NeRF follows view requests"
        if self.mode is DigitalTwinNavigationMode.STATIC_PREVIEW:
            label = "Gaussian" if self.active_view_representation == _GAUSSIAN_VIEW else "NeRF"
            return f"Canonical navigation retained • {label} static preview"
        return "Canonical viewport camera • shared navigation source"

    @property
    def camera_text(self) -> str:
        if self.mode is DigitalTwinNavigationMode.NO_SCENE or self.center_viewer is None:
            return "No scene-bound presentation camera"
        x, y, z = self.center_viewer
        return (
            f"center=({x:.3f}, {y:.3f}, {z:.3f}) • distance={float(self.distance):.3f} • "
            f"az={float(self.azimuth_degrees):.1f}° • el={float(self.elevation_degrees):.1f}° • "
            f"fov={float(self.fov_degrees):.1f}°"
        )

    @property
    def scene_switch_text(self) -> str:
        if self.mode is DigitalTwinNavigationMode.NO_SCENE:
            return "No scene-bound navigation"
        if self.view_preserved_on_same_scene_reload:
            return "Same-scene reload • canonical view preserved"
        return "Scene-bound canonical view • one shared navigation source"

    @property
    def authority_text(self) -> str:
        if self.mode is DigitalTwinNavigationMode.NO_SCENE:
            return "No scene authority"
        return (
            "Navigation camera is presentation state only • canonical geometry remains spatial, "
            "picking and measurement authority"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "scene_id": self.scene_id,
            "active_view_representation": self.active_view_representation,
            "center_viewer": self.center_viewer,
            "distance": self.distance,
            "azimuth_degrees": self.azimuth_degrees,
            "elevation_degrees": self.elevation_degrees,
            "fov_degrees": self.fov_degrees,
            "gaussian_interactive": self.gaussian_interactive,
            "nerf_interactive": self.nerf_interactive,
            "static_preview": self.static_preview,
            "view_preserved_on_same_scene_reload": self.view_preserved_on_same_scene_reload,
            "navigation_source": self.navigation_source,
            "spatial_authority": self.spatial_authority,
            "picking_authority": "canonical_geometry" if self.scene_id else "none",
            "measurement_authority": "canonical_geometry" if self.scene_id else "none",
            "appearance_spatial_authority": False,
            "canonical_geometry_authoritative": bool(self.canonical_geometry_authoritative),
            "certified_metrology": False,
        }


def _composition_value(composition: Any, name: str, default: Any = None) -> Any:
    return getattr(composition, name, default)


def _layer(composition: Any, key: str) -> Any | None:
    layer_method = getattr(composition, "layer", None)
    if not callable(layer_method):
        return None
    return layer_method(key)


def _normalized_camera_state(camera_state: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(camera_state, Mapping):
        raise TypeError("Digital Twin navigation requires a canonical viewport camera mapping.")
    missing = {
        "center_viewer",
        "distance",
        "azimuth_degrees",
        "elevation_degrees",
        "fov_degrees",
    } - set(camera_state)
    if missing:
        raise ValueError("Canonical viewport camera snapshot is incomplete: " + ", ".join(sorted(missing)))
    center_raw = camera_state["center_viewer"]
    try:
        center = tuple(float(value) for value in center_raw)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError("Canonical viewport center is invalid.") from exc
    return {
        "center_viewer": center,
        "distance": float(camera_state["distance"]),
        "azimuth_degrees": float(camera_state["azimuth_degrees"]),
        "elevation_degrees": float(camera_state["elevation_degrees"]),
        "fov_degrees": float(camera_state["fov_degrees"]),
    }


def build_digital_twin_navigation_sync(
    composition: Any,
    *,
    camera_state: Mapping[str, object] | None = None,
    gaussian_interactive: bool = False,
    nerf_interactive: bool = False,
    gaussian_preview_visible: bool = False,
    nerf_preview_visible: bool = False,
    view_preserved_on_same_scene_reload: bool = False,
) -> DigitalTwinNavigationSync:
    """Describe one shared navigation source across the existing scene views.

    Interactive Gaussian consumes the canvas camera implicitly because it is a
    visual in the same scene graph. Interactive NeRF consumes the same camera
    explicitly through ``interactive_render_camera_state``. Static previews are
    intentionally reported as static rather than falsely camera-synchronized.
    """

    if composition is None:
        raise ValueError("Digital Twin navigation requires a valid unified scene composition.")

    raw_state = _composition_value(composition, "state", "")
    state_value = str(getattr(raw_state, "value", raw_state)).strip().lower()
    scene_id = str(_composition_value(composition, "scene_id", "")).strip()
    active = (
        str(_composition_value(composition, "active_view_representation", _CANONICAL_VIEW))
        .strip()
        .lower()
        or _CANONICAL_VIEW
    )
    canonical_authority = bool(_composition_value(composition, "canonical_geometry_authoritative", False))

    if bool(_composition_value(composition, "certified_metrology", False)):
        raise ValueError("Digital Twin composition unexpectedly claims certified metrology.")
    if state_value == "no_scene":
        if any((gaussian_interactive, nerf_interactive, gaussian_preview_visible, nerf_preview_visible)):
            raise ValueError("Appearance navigation cannot remain active without a canonical scene.")
        return DigitalTwinNavigationSync(
            mode=DigitalTwinNavigationMode.NO_SCENE,
            scene_id="",
            active_view_representation=_CANONICAL_VIEW,
            center_viewer=None,
            distance=None,
            azimuth_degrees=None,
            elevation_degrees=None,
            fov_degrees=None,
            gaussian_interactive=False,
            nerf_interactive=False,
            static_preview=False,
            view_preserved_on_same_scene_reload=False,
            canonical_geometry_authoritative=False,
            certified_metrology=False,
            navigation_source="none",
            spatial_authority="none",
        )

    if not scene_id or not canonical_authority:
        raise ValueError("Digital Twin navigation requires the canonical scene authority established by composition.")
    if active not in _SUPPORTED_VIEWS:
        raise ValueError(f"Unsupported Digital Twin active view: {active!r}")

    flags = (
        bool(gaussian_interactive),
        bool(nerf_interactive),
        bool(gaussian_preview_visible),
        bool(nerf_preview_visible),
    )
    if sum(flags) > 1:
        raise ValueError("Gaussian/NeRF interactive and preview modes must remain mutually exclusive.")

    gaussian_layer = _layer(composition, "gaussian_scene")
    nerf_layer = _layer(composition, "nerf_scene")
    if gaussian_interactive or gaussian_preview_visible:
        if active != _GAUSSIAN_VIEW:
            raise ValueError("Visible Gaussian appearance must match the active unified representation.")
        if gaussian_layer is None or not bool(getattr(gaussian_layer, "available", False)):
            raise ValueError("Gaussian navigation cannot bind an unavailable scene layer.")
        if not bool(getattr(gaussian_layer, "visible", False)):
            raise ValueError("Visible Gaussian navigation requires the unified Gaussian layer to be visible.")
    if nerf_interactive or nerf_preview_visible:
        if active != _NERF_VIEW:
            raise ValueError("Visible NeRF appearance must match the active unified representation.")
        if nerf_layer is None or not bool(getattr(nerf_layer, "available", False)):
            raise ValueError("NeRF navigation cannot bind an unavailable scene layer.")
        if not bool(getattr(nerf_layer, "visible", False)):
            raise ValueError("Visible NeRF navigation requires the unified NeRF layer to be visible.")
    if active == _CANONICAL_VIEW and any(flags):
        raise ValueError("Canonical active view cannot expose an appearance navigation mode.")

    if camera_state is None:
        raise ValueError("Scene-bound Digital Twin navigation requires the canonical viewport camera snapshot.")
    camera = _normalized_camera_state(camera_state)

    mode = DigitalTwinNavigationMode.CANONICAL
    if gaussian_interactive:
        mode = DigitalTwinNavigationMode.GAUSSIAN_INTERACTIVE
    elif nerf_interactive:
        mode = DigitalTwinNavigationMode.NERF_INTERACTIVE
    elif gaussian_preview_visible or nerf_preview_visible:
        mode = DigitalTwinNavigationMode.STATIC_PREVIEW

    return DigitalTwinNavigationSync(
        mode=mode,
        scene_id=scene_id,
        active_view_representation=active,
        gaussian_interactive=bool(gaussian_interactive),
        nerf_interactive=bool(nerf_interactive),
        static_preview=bool(gaussian_preview_visible or nerf_preview_visible),
        view_preserved_on_same_scene_reload=bool(view_preserved_on_same_scene_reload),
        canonical_geometry_authoritative=True,
        certified_metrology=False,
        **camera,
    )


__all__ = [
    "DigitalTwinNavigationMode",
    "DigitalTwinNavigationSync",
    "build_digital_twin_navigation_sync",
]

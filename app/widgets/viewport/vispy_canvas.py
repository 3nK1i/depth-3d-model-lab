"""VisPy 3D point-cloud canvas and explicit viewport interaction modes.

The SceneCanvas, camera and visuals are created once. Live frames only update
existing GPU buffers. Step 3.1 adds real Select/Orbit/Pan/Zoom modes, immediate
point-style updates, background/grid controls, and keeps camera state stable
across geometry refreshes.
"""

from __future__ import annotations

import math

import numpy as np
from PyQt6.QtCore import QEvent, QObject, Qt, pyqtSignal
from vispy import scene

DEFAULT_POINT_SIZE = 4.0
DEFAULT_CAMERA_DISTANCE = 5.0
DEFAULT_FOV = 45.0
DEFAULT_INTERACTION_MODE = "orbit"
_VALID_INTERACTION_MODES = {"select", "orbit", "pan", "zoom"}

_BACKGROUND_COLORS: dict[str, str | tuple[float, float, float, float]] = {
    "dark": "#090b0e",
    "black": "#000000",
    "white": "#f2f3f5",
    "transparent": (0.0, 0.0, 0.0, 0.0),
}


class _ModeTurntableCamera(scene.cameras.TurntableCamera):
    """Turntable camera whose left-drag action follows the selected tool."""

    def __init__(self, *args, interaction_mode: str = DEFAULT_INTERACTION_MODE, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._interaction_mode = DEFAULT_INTERACTION_MODE
        self.set_interaction_mode(interaction_mode)

    @property
    def interaction_mode(self) -> str:
        return self._interaction_mode

    def set_interaction_mode(self, mode: str) -> None:
        normalized = str(mode).strip().lower()
        if normalized not in _VALID_INTERACTION_MODES:
            raise ValueError(f"Unsupported viewport interaction mode: {mode!r}")
        self._interaction_mode = normalized
        self._event_value = None

    def viewbox_mouse_event(self, event) -> None:
        """Apply only the interaction represented by the selected toolbar tool."""
        if event.handled or not self.interactive:
            return

        if self._interaction_mode == "select":
            if event.type == "mouse_release":
                self._event_value = None
            elif event.type in {"mouse_press", "mouse_move", "mouse_wheel", "gesture_zoom"}:
                event.handled = True
            return

        if event.type in {"mouse_wheel", "gesture_zoom"}:
            # Wheel/pinch zoom remains useful in Orbit and Pan modes.
            super().viewbox_mouse_event(event)
            return

        if event.type == "mouse_release":
            self._event_value = None
            return

        if event.type == "mouse_press":
            event.handled = True
            return

        if event.type != "mouse_move" or event.press_event is None:
            return

        mouse_event = event.mouse_event
        press = mouse_event.press_event
        if press is None:
            return
        p1 = press.pos
        p2 = mouse_event.pos
        delta = p2 - p1
        buttons = tuple(event.buttons or ())

        # Right drag remains a secondary zoom gesture in Orbit and Pan.
        if 2 in buttons:
            self._zoom_from_drag(delta)
            event.handled = True
            return
        if 1 not in buttons:
            return

        if self._interaction_mode == "orbit":
            self._update_rotation(event)
        elif self._interaction_mode == "pan":
            self._pan_from_drag(p1, p2)
        elif self._interaction_mode == "zoom":
            self._zoom_from_drag(delta)
        event.handled = True

    def _zoom_from_drag(self, delta: np.ndarray) -> None:
        if self._event_value is None or not isinstance(self._event_value, tuple):
            self._event_value = (self.scale_factor, self.distance)
        if len(self._event_value) != 2:
            self._event_value = (self.scale_factor, self.distance)
        zoom_y = (1.0 + self.zoom_factor) ** float(delta[1])
        self.scale_factor = float(self._event_value[0]) * zoom_y
        if self.distance is not None and self._event_value[1] is not None:
            self.distance = float(self._event_value[1]) * zoom_y
        self.view_changed()

    def _pan_from_drag(self, p1: np.ndarray, p2: np.ndarray) -> None:
        norm = float(np.mean(self._viewbox.size)) if self._viewbox is not None else 1.0
        norm = max(norm, 1.0)
        if self._event_value is None or (
            isinstance(self._event_value, tuple) and len(self._event_value) == 2
        ):
            self._event_value = tuple(self.center)

        dist = (p1 - p2) / norm * self.scale_factor
        dist[1] *= -1.0
        dx, dy, dz = self._dist_to_trans(dist)
        flip = self._flip_factors
        up, forward, right = self._get_dim_vectors()
        dx, dy, dz = right * dx + forward * dy + up * dz
        dx, dy, dz = flip[0] * dx, flip[1] * dy, dz * flip[2]
        center = self._event_value
        self.center = (
            float(center[0] + dx),
            float(center[1] + dy),
            float(center[2] + dz),
        )


class PointCloudCanvas(QObject):
    """Own the VisPy SceneCanvas, camera, grid and point-cloud visual."""

    interaction_started = pyqtSignal()
    interaction_ended = pyqtSignal()
    escape_pressed = pyqtSignal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)

        self._background_name = "dark"
        self._canvas = scene.SceneCanvas(
            # Do not use VisPy's ``keys="interactive"`` shortcut here: it
            # binds Escape to closing the embedded canvas and F11 to a second,
            # independent fullscreen lifecycle. The Qt window owns both.
            keys=None,
            bgcolor=_BACKGROUND_COLORS[self._background_name],
            show=False,
            vsync=True,
        )
        self._view = self._canvas.central_widget.add_view()
        self._camera = _ModeTurntableCamera(
            fov=DEFAULT_FOV,
            distance=DEFAULT_CAMERA_DISTANCE,
            interaction_mode=DEFAULT_INTERACTION_MODE,
        )
        self._view.camera = self._camera

        self._grid_visible = True
        self._grid = self._create_grid_visual()

        self._markers = scene.visuals.Markers(parent=self._view.scene)
        self._point_size = DEFAULT_POINT_SIZE
        self._smoothing = 0.5
        self._markers.antialias = self._antialias_amount(self._smoothing)
        self._has_points = False
        self._points_bounds: tuple[tuple[float, float], ...] | None = None
        self._last_world_points: np.ndarray | None = None
        self._last_colors: np.ndarray | None = None

        self._is_interacting = False
        self._canvas.events.mouse_press.connect(self._on_mouse_press)
        self._canvas.events.mouse_release.connect(self._on_mouse_release)
        self.native.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.native.installEventFilter(self)

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt override
        if (
            watched is self.native
            and event.type() == QEvent.Type.KeyPress
            and event.key() == Qt.Key.Key_Escape
        ):
            self.escape_pressed.emit()
            return True
        return super().eventFilter(watched, event)

    @property
    def native(self):
        """The Qt-native widget passed to ``ViewportPanel.attach_canvas``."""
        return self._canvas.native

    # -- Data update -----------------------------------------------------

    def update_points(self, points: np.ndarray, colors: np.ndarray) -> None:
        """Update the existing marker buffers using camera-space geometry."""
        if points.shape[0] == 0:
            self.clear_points()
            return

        world_points = np.empty_like(points, dtype=np.float32)
        world_points[:, 0] = points[:, 0]
        world_points[:, 1] = points[:, 2]
        world_points[:, 2] = -points[:, 1]

        color_float = (
            colors.astype(np.float32) / 255.0
            if colors.dtype == np.uint8
            else np.asarray(colors, dtype=np.float32)
        )
        self._last_world_points = np.ascontiguousarray(world_points)
        self._last_colors = np.ascontiguousarray(color_float)
        self._refresh_markers()
        self._has_points = True
        self._points_bounds = (
            (float(world_points[:, 0].min()), float(world_points[:, 0].max())),
            (float(world_points[:, 1].min()), float(world_points[:, 1].max())),
            (float(world_points[:, 2].min()), float(world_points[:, 2].max())),
        )

    def clear_points(self) -> None:
        self._markers.set_data(pos=np.zeros((0, 3), dtype=np.float32))
        self._last_world_points = None
        self._last_colors = None
        self._has_points = False
        self._points_bounds = None

    def _refresh_markers(self) -> None:
        if self._last_world_points is None or self._last_colors is None:
            return
        self._markers.set_data(
            pos=self._last_world_points,
            face_color=self._last_colors,
            size=self._point_size,
            edge_width=0,
        )
        self._canvas.update()

    # -- Camera control --------------------------------------------------

    def fit_scene(self) -> None:
        if self._has_points and self._points_bounds is not None:
            (x_min, x_max), (y_min, y_max), (z_min, z_max) = self._points_bounds
            center = (
                (x_min + x_max) / 2.0,
                (y_min + y_max) / 2.0,
                (z_min + z_max) / 2.0,
            )
            radius = 0.5 * math.sqrt(
                (x_max - x_min) ** 2
                + (y_max - y_min) ** 2
                + (z_max - z_min) ** 2
            )
            radius = max(radius, 0.1)
            half_fov_rad = math.radians(self._camera.fov / 2.0)
            distance = (radius / math.tan(half_fov_rad)) * 1.2
            self._camera.center = center
            self._camera.distance = distance
        else:
            self._camera.center = (0.0, 0.0, 0.0)
            self._camera.distance = DEFAULT_CAMERA_DISTANCE

    def reset_view(self) -> None:
        self._camera.center = (0.0, 0.0, 0.0)
        self._camera.distance = DEFAULT_CAMERA_DISTANCE
        self._camera.fov = DEFAULT_FOV
        self._camera.azimuth = 30.0
        self._camera.elevation = 30.0
        self.fit_scene()

    def set_interaction_mode(self, mode: str) -> None:
        self._camera.set_interaction_mode(mode)
        cursors = {
            "select": Qt.CursorShape.ArrowCursor,
            "orbit": Qt.CursorShape.OpenHandCursor,
            "pan": Qt.CursorShape.SizeAllCursor,
            "zoom": Qt.CursorShape.SizeVerCursor,
        }
        self.native.setCursor(cursors[mode])

    @property
    def interaction_mode(self) -> str:
        return self._camera.interaction_mode

    # -- Render settings -------------------------------------------------

    def set_point_size(self, size: float) -> None:
        normalized = max(0.5, float(size))
        if math.isclose(normalized, self._point_size, rel_tol=0.0, abs_tol=1e-6):
            return
        self._point_size = normalized
        self._refresh_markers()

    def set_smoothing(self, value: float) -> None:
        normalized = float(np.clip(value, 0.0, 1.0))
        self._smoothing = normalized
        self._markers.antialias = self._antialias_amount(normalized)
        self._canvas.update()

    @staticmethod
    def _antialias_amount(value: float) -> float:
        return 0.25 + value * 2.75

    def set_background(self, background: str) -> None:
        name = str(background).strip().lower()
        if name not in _BACKGROUND_COLORS:
            raise ValueError(f"Unsupported viewport background: {background!r}")
        if name == self._background_name:
            return
        self._background_name = name
        self._canvas.bgcolor = _BACKGROUND_COLORS[name]
        self._replace_grid_visual()
        self._canvas.update()

    def set_grid_visible(self, visible: bool) -> None:
        self._grid_visible = bool(visible)
        self._grid.visible = self._grid_visible
        self._canvas.update()

    def _create_grid_visual(self):
        color = (0.0, 0.0, 0.0, 0.16) if self._background_name == "white" else (1, 1, 1, 0.08)
        grid = scene.visuals.GridLines(color=color, parent=self._view.scene)
        grid.visible = self._grid_visible
        return grid

    def _replace_grid_visual(self) -> None:
        old_grid = self._grid
        old_grid.parent = None
        self._grid = self._create_grid_visual()

    # -- Interaction state ----------------------------------------------

    def _on_mouse_press(self, _event) -> None:
        if self.interaction_mode == "select":
            return
        self._is_interacting = True
        self.interaction_started.emit()

    def _on_mouse_release(self, _event) -> None:
        if not self._is_interacting:
            return
        self._is_interacting = False
        self.interaction_ended.emit()

    @property
    def is_interacting(self) -> bool:
        return self._is_interacting

    @property
    def has_points(self) -> bool:
        return self._has_points

    @property
    def point_size(self) -> float:
        return self._point_size

    @property
    def smoothing(self) -> float:
        return self._smoothing

    @property
    def background_name(self) -> str:
        return self._background_name

    @property
    def grid_visible(self) -> bool:
        return self._grid_visible

    def close(self) -> None:
        self._canvas.close()

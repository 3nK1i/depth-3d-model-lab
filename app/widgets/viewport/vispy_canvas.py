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


class ReconstructionCanvas(QObject):
    """VisPy foundation for the Phase 8.8C Reconstruction viewport.

    C1-C5 provide the live reconstruction viewer. D1 reuses the same bounded
    scene primitives for completed offline sparse SfM results. D2 adds bounded
    registered-camera structure inspection and selection without mixing live
    runtime state or promoting sparse geometry to canonical/metric authority.
    """

    interaction_started = pyqtSignal()
    interaction_ended = pyqtSignal()
    escape_pressed = pyqtSignal()

    MAX_TRAJECTORY_POSES = 20_000
    MAX_TRAJECTORY_DISPLAY_POINTS = 4_096
    MAX_SPARSE_LANDMARK_DISPLAY_POINTS = 8_192
    MAX_KEYFRAME_DISPLAY_FRUSTUMS = 96
    MAX_OFFLINE_CAMERA_PATH_POINTS = 1_024

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        from app.reconstruction_visualization import camera_frustum_segments_in_viewer

        # Keep an import reference on the instance so tests can assert that the
        # canvas uses the dependency-light presentation transform rather than
        # reinterpreting reconstruction poses itself.
        self._frustum_builder = camera_frustum_segments_in_viewer
        self._background_name = "dark"
        self._canvas = scene.SceneCanvas(
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
        self._camera.azimuth = 32.0
        self._camera.elevation = 24.0
        self._view.camera = self._camera

        self._grid_visible = True
        self._axes_visible = True
        self._grid = self._create_grid_visual()
        self._world_axis_length = 0.28
        self._axis_x, self._axis_y, self._axis_z, self._axis_origin = self._create_world_axes_visuals()

        self._camera_frustum = scene.visuals.Line(
            pos=np.zeros((0, 3), dtype=np.float32),
            connect="segments",
            color=(0.88, 0.16, 0.24, 1.0),
            width=2.0,
            parent=self._view.scene,
        )
        self._camera_center = scene.visuals.Markers(parent=self._view.scene)
        self._camera_center.set_data(
            pos=np.zeros((0, 3), dtype=np.float32),
            face_color=(0.88, 0.16, 0.24, 1.0),
            size=6.0,
            edge_width=0,
        )
        self._trajectory_line = scene.visuals.Line(
            pos=np.zeros((0, 3), dtype=np.float32),
            connect="strip",
            color=(0.28, 0.72, 0.95, 0.85),
            width=1.5,
            parent=self._view.scene,
        )
        self._trajectory_visible = True
        self._trajectory_line.visible = True
        self._trajectory_start = scene.visuals.Markers(parent=self._view.scene)
        self._trajectory_start.set_data(
            pos=np.zeros((0, 3), dtype=np.float32),
            face_color=(0.28, 0.72, 0.95, 0.90),
            size=5.0,
            edge_width=0,
        )
        self._trajectory_start.visible = True
        self._sparse_landmarks = scene.visuals.Markers(parent=self._view.scene)
        self._sparse_landmarks.set_data(
            pos=np.zeros((0, 3), dtype=np.float32),
            face_color=(0.72, 0.82, 0.90, 0.82),
            size=2.6,
            edge_width=0,
        )
        self._sparse_landmarks_visible = True
        self._sparse_landmarks.visible = True
        self._sparse_landmark_positions = np.zeros((0, 3), dtype=np.float32)
        self._sparse_landmark_total_count = 0
        self._sparse_landmark_display_count = 0
        self._sparse_map_revision: int | None = None
        self._sparse_landmark_colors = None
        self._offline_visualization_loaded = False
        self._offline_session_id: str | None = None
        self._offline_scale_status: str | None = None
        self._keyframe_frustums = scene.visuals.Line(
            pos=np.zeros((0, 3), dtype=np.float32),
            connect="segments",
            color=(0.94, 0.66, 0.24, 0.52),
            width=1.0,
            parent=self._view.scene,
        )
        self._keyframe_centers = scene.visuals.Markers(parent=self._view.scene)
        self._keyframe_centers.set_data(
            pos=np.zeros((0, 3), dtype=np.float32),
            face_color=(0.94, 0.66, 0.24, 0.74),
            size=3.4,
            edge_width=0,
        )
        self._keyframes_visible = True
        self._keyframe_frustums.visible = True
        self._keyframe_centers.visible = True
        self._offline_camera_path = scene.visuals.Line(
            pos=np.zeros((0, 3), dtype=np.float32),
            connect="strip",
            color=(0.94, 0.66, 0.24, 0.30),
            width=1.0,
            parent=self._view.scene,
        )
        self._offline_camera_path.visible = False
        self._selected_offline_camera_frustum = scene.visuals.Line(
            pos=np.zeros((0, 3), dtype=np.float32),
            connect="segments",
            color=(1.0, 0.84, 0.30, 0.98),
            width=2.4,
            parent=self._view.scene,
        )
        self._selected_offline_camera_center = scene.visuals.Markers(parent=self._view.scene)
        self._selected_offline_camera_center.set_data(
            pos=np.zeros((0, 3), dtype=np.float32),
            face_color=(1.0, 0.84, 0.30, 1.0),
            size=7.0,
            edge_width=0,
        )
        self._selected_offline_camera_frustum.visible = False
        self._selected_offline_camera_center.visible = False
        self._offline_camera_matrices = np.zeros((0, 4, 4), dtype=np.float64)
        self._offline_camera_positions = np.zeros((0, 3), dtype=np.float32)
        self._offline_camera_ids: tuple[str, ...] = ()
        self._offline_camera_path_display_count = 0
        self._selected_offline_camera_index: int | None = None
        self._keyframe_matrices = np.zeros((0, 4, 4), dtype=np.float64)
        self._keyframe_positions = np.zeros((0, 3), dtype=np.float32)
        self._keyframe_ids: tuple[str, ...] = ()
        self._keyframe_total_count = 0
        self._keyframe_display_count = 0
        self._keyframe_map_revision: int | None = None
        self._trajectory_positions: list[np.ndarray] = []
        self._trajectory_display_count = 0
        self._current_camera_position: np.ndarray | None = None
        self._has_pose = False
        self._is_interacting = False
        self._intrinsics_matrix: np.ndarray | None = None
        self._image_size: tuple[int, int] | None = None
        self._closed = False

        self._canvas.events.mouse_press.connect(self._on_mouse_press)
        self._canvas.events.mouse_release.connect(self._on_mouse_release)
        self.native.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.native.installEventFilter(self)
        self.reset_view()

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
        return self._canvas.native

    def configure_camera_geometry(
        self,
        *,
        intrinsics_matrix: np.ndarray | None,
        image_size: tuple[int, int] | None,
    ) -> None:
        if intrinsics_matrix is None or image_size is None:
            self._intrinsics_matrix = None
            self._image_size = None
            return
        matrix = np.asarray(intrinsics_matrix, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError("intrinsics_matrix must be a finite 3x3 array.")
        width, height = int(image_size[0]), int(image_size[1])
        if width <= 0 or height <= 0:
            raise ValueError("image_size must contain positive width and height.")
        self._intrinsics_matrix = np.array(matrix, copy=True)
        self._image_size = (width, height)

    def update_live_pose(self, T_reconstruction_from_camera: np.ndarray) -> None:
        from app.reconstruction_visualization import camera_position_in_viewer

        frustum = self._frustum_builder(
            T_reconstruction_from_camera,
            intrinsics_matrix=self._intrinsics_matrix,
            image_size=self._image_size,
        )
        position = camera_position_in_viewer(T_reconstruction_from_camera).astype(np.float32)
        self._camera_frustum.set_data(pos=frustum, connect="segments")
        self._camera_center.set_data(
            pos=position.reshape(1, 3),
            face_color=(0.88, 0.16, 0.24, 1.0),
            size=6.0,
            edge_width=0,
        )
        self._current_camera_position = position
        self._has_pose = True
        self._trajectory_positions.append(np.array(position, copy=True))
        if len(self._trajectory_positions) > self.MAX_TRAJECTORY_POSES:
            del self._trajectory_positions[: len(self._trajectory_positions) - self.MAX_TRAJECTORY_POSES]
        self._refresh_world_axis_scale()
        if self._trajectory_visible:
            self._refresh_trajectory_visual()
        self._canvas.update()

    def clear_live_state(self) -> None:
        self._camera_frustum.set_data(pos=np.zeros((0, 3), dtype=np.float32), connect="segments")
        self._camera_center.set_data(pos=np.zeros((0, 3), dtype=np.float32))
        self._trajectory_positions.clear()
        self._trajectory_line.set_data(pos=np.zeros((0, 3), dtype=np.float32), connect="strip")
        self._trajectory_start.set_data(pos=np.zeros((0, 3), dtype=np.float32))
        self._trajectory_display_count = 0
        self._sparse_landmarks.set_data(pos=np.zeros((0, 3), dtype=np.float32))
        self._sparse_landmark_positions = np.zeros((0, 3), dtype=np.float32)
        self._sparse_landmark_total_count = 0
        self._sparse_landmark_display_count = 0
        self._sparse_map_revision = None
        self._sparse_landmark_colors = None
        self._offline_visualization_loaded = False
        self._offline_session_id = None
        self._offline_scale_status = None
        self._offline_camera_path.set_data(pos=np.zeros((0, 3), dtype=np.float32), connect="strip")
        self._offline_camera_path.visible = False
        self._selected_offline_camera_frustum.set_data(
            pos=np.zeros((0, 3), dtype=np.float32), connect="segments"
        )
        self._selected_offline_camera_center.set_data(pos=np.zeros((0, 3), dtype=np.float32))
        self._selected_offline_camera_frustum.visible = False
        self._selected_offline_camera_center.visible = False
        self._offline_camera_matrices = np.zeros((0, 4, 4), dtype=np.float64)
        self._offline_camera_positions = np.zeros((0, 3), dtype=np.float32)
        self._offline_camera_ids = ()
        self._offline_camera_path_display_count = 0
        self._selected_offline_camera_index = None
        self._keyframe_frustums.set_data(pos=np.zeros((0, 3), dtype=np.float32), connect="segments")
        self._keyframe_centers.set_data(pos=np.zeros((0, 3), dtype=np.float32))
        self._keyframe_matrices = np.zeros((0, 4, 4), dtype=np.float64)
        self._keyframe_positions = np.zeros((0, 3), dtype=np.float32)
        self._keyframe_ids = ()
        self._keyframe_total_count = 0
        self._keyframe_display_count = 0
        self._keyframe_map_revision = None
        self._world_axis_length = 0.28
        self._update_world_axes_visuals()
        self._current_camera_position = None
        self._has_pose = False
        self._canvas.update()

    def set_trajectory_visible(self, visible: bool) -> None:
        self._trajectory_visible = bool(visible)
        self._trajectory_line.visible = self._trajectory_visible
        self._trajectory_start.visible = self._trajectory_visible
        if self._trajectory_visible:
            self._refresh_trajectory_visual()
        self._canvas.update()

    def update_sparse_landmarks(
        self,
        points_reconstruction: np.ndarray,
        *,
        colors_rgb: np.ndarray | None = None,
        total_count: int | None = None,
        revision: int | None = None,
    ) -> None:
        from app.reconstruction_visualization import (
            sparse_landmark_display_indices,
            sparse_landmarks_to_viewer,
        )

        points = np.asarray(points_reconstruction, dtype=np.float64)
        if points.size == 0:
            points = np.zeros((0, 3), dtype=np.float64)
        if points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all():
            raise ValueError("points_reconstruction must be a finite (N, 3) array.")
        viewer = sparse_landmarks_to_viewer(points)
        indices = sparse_landmark_display_indices(
            viewer.shape[0], max_points=self.MAX_SPARSE_LANDMARK_DISPLAY_POINTS
        )
        display = np.ascontiguousarray(viewer[indices])
        display_colors = None
        if colors_rgb is not None:
            colors = np.asarray(colors_rgb)
            if colors.size == 0:
                colors = np.zeros((0, 3), dtype=np.uint8)
            if colors.ndim != 2 or colors.shape != points.shape:
                raise ValueError("colors_rgb must match points_reconstruction shape.")
            colors_float = colors.astype(np.float64)
            if not np.isfinite(colors_float).all() or np.any(colors_float < 0) or np.any(colors_float > 255):
                raise ValueError("colors_rgb values must lie within [0, 255].")
            display_colors = np.ascontiguousarray(colors_float[indices] / 255.0, dtype=np.float32)
            alpha = np.full((display_colors.shape[0], 1), 0.88, dtype=np.float32)
            display_colors = np.concatenate([display_colors, alpha], axis=1)
        self._sparse_landmark_positions = display
        self._sparse_landmark_colors = display_colors
        self._sparse_landmark_total_count = max(
            int(total_count) if total_count is not None else int(points.shape[0]),
            int(points.shape[0]),
        )
        self._sparse_landmark_display_count = int(display.shape[0])
        self._sparse_map_revision = None if revision is None else int(revision)
        self._sparse_landmarks.set_data(
            pos=display,
            face_color=(0.72, 0.82, 0.90, 0.82) if display_colors is None else display_colors,
            size=2.6,
            edge_width=0,
        )
        self._sparse_landmarks.visible = self._sparse_landmarks_visible
        self._refresh_world_axis_scale()
        self._canvas.update()

    def show_offline_sparse_reconstruction(
        self,
        *,
        session_id: str,
        scale_status: str,
        points_reconstruction: np.ndarray,
        camera_matrices_reconstruction: np.ndarray,
        camera_ids: tuple[str, ...] | list[str],
        landmark_colors_rgb: np.ndarray | None = None,
        landmark_total: int | None = None,
        registered_camera_total: int | None = None,
    ) -> None:
        """Replace live visuals with one completed, bounded offline sparse SfM result."""

        sid = str(session_id).strip()
        status = str(scale_status).strip().lower()
        if not sid:
            raise ValueError("session_id must be non-empty.")
        if status != "arbitrary":
            raise ValueError("D1 offline sparse visualization requires arbitrary scale status.")
        self.clear_live_state()
        self.update_sparse_landmarks(
            points_reconstruction,
            colors_rgb=landmark_colors_rgb,
            total_count=landmark_total,
            revision=None,
        )
        matrices = np.asarray(camera_matrices_reconstruction, dtype=np.float64)
        ids = tuple(str(value).strip() for value in camera_ids)
        if matrices.size == 0:
            matrices = np.zeros((0, 4, 4), dtype=np.float64)
        if matrices.ndim != 3 or matrices.shape[1:] != (4, 4) or not np.isfinite(matrices).all():
            raise ValueError("camera_matrices_reconstruction must be a finite (N, 4, 4) array.")
        if len(ids) != matrices.shape[0] or any(not value for value in ids) or len(set(ids)) != len(ids):
            raise ValueError("camera_ids must match the offline camera samples and remain unique.")
        from app.reconstruction_visualization import keyframe_centers_in_viewer

        self._offline_camera_matrices = np.ascontiguousarray(matrices)
        self._offline_camera_positions = keyframe_centers_in_viewer(self._offline_camera_matrices)
        self._offline_camera_ids = ids
        self._selected_offline_camera_index = None
        self.update_keyframe_poses(
            matrices,
            keyframe_ids=ids,
            total_count=registered_camera_total,
            revision=None,
        )
        self._offline_visualization_loaded = bool(self.has_sparse_map or self.has_keyframes)
        self._refresh_offline_camera_path()
        self._offline_session_id = sid
        self._offline_scale_status = status
        self._has_pose = False
        self._current_camera_position = None
        self._refresh_world_axis_scale()
        self._canvas.update()

    def _refresh_offline_camera_path(self) -> None:
        if self._offline_camera_positions.shape[0] < 2:
            self._offline_camera_path.set_data(
                pos=np.zeros((0, 3), dtype=np.float32), connect="strip"
            )
            self._offline_camera_path_display_count = int(self._offline_camera_positions.shape[0])
            self._offline_camera_path.visible = False
            return
        from app.reconstruction_visualization import keyframe_display_indices

        indices = keyframe_display_indices(
            self._offline_camera_positions.shape[0],
            max_keyframes=self.MAX_OFFLINE_CAMERA_PATH_POINTS,
        )
        display = np.ascontiguousarray(self._offline_camera_positions[indices])
        self._offline_camera_path.set_data(pos=display, connect="strip")
        self._offline_camera_path_display_count = int(display.shape[0])
        self._offline_camera_path.visible = bool(
            self._offline_visualization_loaded and self._keyframes_visible
        )

    def select_offline_camera(self, camera_index: int) -> None:
        """Highlight one registered offline camera without changing reconstruction state."""

        index = int(camera_index)
        if not self._offline_visualization_loaded:
            raise ValueError("No offline sparse reconstruction is loaded.")
        if index < 0 or index >= self._offline_camera_matrices.shape[0]:
            raise IndexError("Offline registered camera index is out of range.")
        self._selected_offline_camera_index = index
        self._refresh_selected_offline_camera()
        self._canvas.update()

    def clear_offline_camera_selection(self) -> None:
        self._selected_offline_camera_index = None
        self._selected_offline_camera_frustum.set_data(
            pos=np.zeros((0, 3), dtype=np.float32), connect="segments"
        )
        self._selected_offline_camera_center.set_data(pos=np.zeros((0, 3), dtype=np.float32))
        self._selected_offline_camera_frustum.visible = False
        self._selected_offline_camera_center.visible = False
        self._canvas.update()

    def _refresh_selected_offline_camera(self) -> None:
        index = self._selected_offline_camera_index
        if index is None:
            return
        if index < 0 or index >= self._offline_camera_matrices.shape[0]:
            self.clear_offline_camera_selection()
            return
        from app.reconstruction_visualization import keyframe_frustum_segments_in_viewer

        matrix = self._offline_camera_matrices[index : index + 1]
        depth = float(np.clip(self._world_axis_length * 0.34, 0.05, 0.15))
        frustum = keyframe_frustum_segments_in_viewer(matrix, depth=depth)
        center = self._offline_camera_positions[index : index + 1]
        self._selected_offline_camera_frustum.set_data(pos=frustum, connect="segments")
        self._selected_offline_camera_center.set_data(
            pos=center,
            face_color=(1.0, 0.84, 0.30, 1.0),
            size=7.0,
            edge_width=0,
        )
        visible = bool(self._keyframes_visible and self._offline_visualization_loaded)
        self._selected_offline_camera_frustum.visible = visible
        self._selected_offline_camera_center.visible = visible

    def focus_selected_offline_camera(self) -> None:
        index = self._selected_offline_camera_index
        if index is None or index < 0 or index >= self._offline_camera_positions.shape[0]:
            if index is not None:
                self.clear_offline_camera_selection()
            if self._current_camera_position is None:
                self.fit_scene()
            else:
                self._camera.center = tuple(float(value) for value in self._current_camera_position)
                self._camera.distance = 2.0
                self._camera.view_changed()
            return
        position = self._offline_camera_positions[index]
        self._camera.center = tuple(float(value) for value in position)
        self._camera.distance = max(float(self._world_axis_length) * 5.0, 0.8)
        self._camera.view_changed()

    def set_sparse_landmarks_visible(self, visible: bool) -> None:
        self._sparse_landmarks_visible = bool(visible)
        self._sparse_landmarks.visible = self._sparse_landmarks_visible
        self._canvas.update()

    def update_keyframe_poses(
        self,
        matrices_reconstruction: np.ndarray,
        *,
        keyframe_ids: tuple[str, ...] | list[str] = (),
        total_count: int | None = None,
        revision: int | None = None,
    ) -> None:
        from app.reconstruction_visualization import (
            keyframe_centers_in_viewer,
            keyframe_display_indices,
        )

        matrices = np.asarray(matrices_reconstruction, dtype=np.float64)
        if matrices.size == 0:
            matrices = np.zeros((0, 4, 4), dtype=np.float64)
        if matrices.ndim != 3 or matrices.shape[1:] != (4, 4) or not np.isfinite(matrices).all():
            raise ValueError("matrices_reconstruction must be a finite (N, 4, 4) array.")
        ids = tuple(str(value).strip() for value in keyframe_ids)
        if ids and len(ids) != matrices.shape[0]:
            raise ValueError("keyframe_ids must match matrices_reconstruction length.")
        if ids and (any(not value for value in ids) or len(set(ids)) != len(ids)):
            raise ValueError("keyframe_ids must be non-empty and unique.")

        indices = keyframe_display_indices(
            matrices.shape[0], max_keyframes=self.MAX_KEYFRAME_DISPLAY_FRUSTUMS
        )
        self._keyframe_matrices = np.ascontiguousarray(matrices[indices])
        self._keyframe_positions = keyframe_centers_in_viewer(self._keyframe_matrices)
        self._keyframe_ids = tuple(ids[index] for index in indices) if ids else ()
        self._keyframe_total_count = max(
            int(total_count) if total_count is not None else int(matrices.shape[0]),
            int(matrices.shape[0]),
        )
        self._keyframe_display_count = int(self._keyframe_matrices.shape[0])
        self._keyframe_map_revision = None if revision is None else int(revision)
        self._refresh_world_axis_scale()
        self._refresh_keyframe_visual()
        self._canvas.update()

    def set_keyframes_visible(self, visible: bool) -> None:
        self._keyframes_visible = bool(visible)
        self._keyframe_frustums.visible = self._keyframes_visible
        self._keyframe_centers.visible = self._keyframes_visible
        self._offline_camera_path.visible = bool(
            self._keyframes_visible and self._offline_visualization_loaded
            and self._offline_camera_positions.shape[0] >= 2
        )
        selected_visible = bool(
            self._keyframes_visible and self._offline_visualization_loaded
            and self._selected_offline_camera_index is not None
        )
        self._selected_offline_camera_frustum.visible = selected_visible
        self._selected_offline_camera_center.visible = selected_visible
        if self._keyframes_visible:
            self._refresh_keyframe_visual()
            if self._selected_offline_camera_index is not None:
                self._refresh_selected_offline_camera()
        self._canvas.update()

    def _refresh_keyframe_visual(self) -> None:
        from app.reconstruction_visualization import keyframe_frustum_segments_in_viewer

        if self._keyframe_matrices.shape[0] == 0:
            self._keyframe_frustums.set_data(
                pos=np.zeros((0, 3), dtype=np.float32), connect="segments"
            )
            self._keyframe_centers.set_data(pos=np.zeros((0, 3), dtype=np.float32))
            return
        depth = float(np.clip(self._world_axis_length * 0.26, 0.035, 0.11))
        segments = keyframe_frustum_segments_in_viewer(self._keyframe_matrices, depth=depth)
        self._keyframe_frustums.set_data(pos=segments, connect="segments")
        self._keyframe_centers.set_data(
            pos=self._keyframe_positions,
            face_color=(0.94, 0.66, 0.24, 0.74),
            size=3.4,
            edge_width=0,
        )
        self._keyframe_frustums.visible = self._keyframes_visible
        self._keyframe_centers.visible = self._keyframes_visible

    def _refresh_trajectory_visual(self) -> None:
        from app.reconstruction_visualization import trajectory_display_indices

        if not self._trajectory_positions:
            self._trajectory_line.set_data(pos=np.zeros((0, 3), dtype=np.float32), connect="strip")
            self._trajectory_start.set_data(pos=np.zeros((0, 3), dtype=np.float32))
            self._trajectory_display_count = 0
            return
        points = np.asarray(self._trajectory_positions, dtype=np.float32)
        indices = trajectory_display_indices(
            points.shape[0], max_points=self.MAX_TRAJECTORY_DISPLAY_POINTS
        )
        display = np.ascontiguousarray(points[indices])
        self._trajectory_line.set_data(pos=display, connect="strip")
        self._trajectory_start.set_data(
            pos=points[:1],
            face_color=(0.28, 0.72, 0.95, 0.90),
            size=5.0,
            edge_width=0,
        )
        self._trajectory_display_count = int(display.shape[0])

    def _refresh_world_axis_scale(self) -> None:
        from app.reconstruction_visualization import adaptive_world_axis_length

        groups: list[np.ndarray] = []
        if self._trajectory_positions:
            groups.append(np.asarray(self._trajectory_positions, dtype=np.float64))
        if self._sparse_landmark_positions.size:
            groups.append(np.asarray(self._sparse_landmark_positions, dtype=np.float64))
        if self._keyframe_positions.size:
            groups.append(np.asarray(self._keyframe_positions, dtype=np.float64))
        points = np.vstack(groups) if groups else None
        target = adaptive_world_axis_length(points)
        if abs(target - self._world_axis_length) <= max(0.01, self._world_axis_length * 0.12):
            return
        self._world_axis_length = float(target)
        self._update_world_axes_visuals()
        if self._keyframe_matrices.shape[0]:
            self._refresh_keyframe_visual()
        if self._selected_offline_camera_index is not None:
            self._refresh_selected_offline_camera()

    def _create_world_axes_visuals(self):
        axis_x = scene.visuals.Line(
            connect="segments", color=(0.84, 0.28, 0.34, 0.72), width=1.25, parent=self._view.scene
        )
        axis_y = scene.visuals.Line(
            connect="segments", color=(0.30, 0.72, 0.48, 0.72), width=1.25, parent=self._view.scene
        )
        axis_z = scene.visuals.Line(
            connect="segments", color=(0.34, 0.52, 0.90, 0.72), width=1.25, parent=self._view.scene
        )
        self._axis_x_head = scene.visuals.Line(
            connect="segments", color=(0.84, 0.28, 0.34, 0.72), width=1.25, parent=self._view.scene
        )
        self._axis_y_head = scene.visuals.Line(
            connect="segments", color=(0.30, 0.72, 0.48, 0.72), width=1.25, parent=self._view.scene
        )
        self._axis_z_head = scene.visuals.Line(
            connect="segments", color=(0.34, 0.52, 0.90, 0.72), width=1.25, parent=self._view.scene
        )
        origin = scene.visuals.Markers(parent=self._view.scene)
        origin.set_data(
            pos=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
            face_color=(0.70, 0.72, 0.76, 0.78),
            size=4.0,
            edge_width=0,
        )
        self._axis_x, self._axis_y, self._axis_z, self._axis_origin = axis_x, axis_y, axis_z, origin
        self._update_world_axes_visuals()
        return axis_x, axis_y, axis_z, origin

    def _update_world_axes_visuals(self) -> None:
        length = float(self._world_axis_length)
        origin = np.zeros(3, dtype=np.float32)
        self._axis_x.set_data(
            pos=np.asarray([origin, [length, 0.0, 0.0]], dtype=np.float32), connect="segments"
        )
        self._axis_y.set_data(
            pos=np.asarray([origin, [0.0, length, 0.0]], dtype=np.float32), connect="segments"
        )
        self._axis_z.set_data(
            pos=np.asarray([origin, [0.0, 0.0, length]], dtype=np.float32), connect="segments"
        )
        from app.reconstruction_visualization import world_axis_arrowhead_segments

        self._axis_x_head.set_data(pos=world_axis_arrowhead_segments(0, length), connect="segments")
        self._axis_y_head.set_data(pos=world_axis_arrowhead_segments(1, length), connect="segments")
        self._axis_z_head.set_data(pos=world_axis_arrowhead_segments(2, length), connect="segments")
        for visual in (
            self._axis_x, self._axis_y, self._axis_z,
            self._axis_x_head, self._axis_y_head, self._axis_z_head,
            self._axis_origin,
        ):
            visual.visible = self._axes_visible

    def fit_scene(self) -> None:
        groups: list[np.ndarray] = []
        if self._trajectory_positions:
            groups.append(np.asarray(self._trajectory_positions, dtype=np.float64))
        if self._sparse_landmark_positions.size:
            groups.append(np.asarray(self._sparse_landmark_positions, dtype=np.float64))
        if self._keyframe_positions.size:
            groups.append(np.asarray(self._keyframe_positions, dtype=np.float64))
        if groups:
            points = np.vstack(groups)
            center = points.mean(axis=0)
            span = np.ptp(points, axis=0)
            radius = max(float(np.linalg.norm(span)) * 0.6, 0.5)
            self._camera.center = tuple(float(v) for v in center)
            self._camera.distance = max(radius * 2.4, 1.5)
        elif self._current_camera_position is not None:
            self.focus_current_camera()
        else:
            self._camera.center = (0.0, 0.0, 0.0)
            self._camera.distance = DEFAULT_CAMERA_DISTANCE
        self._camera.view_changed()

    def focus_current_camera(self) -> None:
        if self._selected_offline_camera_index is not None and self._offline_visualization_loaded:
            self.focus_selected_offline_camera()
            return
        if self._current_camera_position is None:
            self.fit_scene()
            return
        self._camera.center = tuple(float(v) for v in self._current_camera_position)
        self._camera.distance = 2.0
        self._camera.view_changed()

    def reset_view(self) -> None:
        self._camera.center = (0.0, 0.0, 0.0)
        self._camera.distance = DEFAULT_CAMERA_DISTANCE
        self._camera.fov = DEFAULT_FOV
        self._camera.azimuth = 32.0
        self._camera.elevation = 24.0
        self._camera.view_changed()

    def set_interaction_mode(self, mode: str) -> None:
        self._camera.set_interaction_mode(mode)
        cursors = {
            "select": Qt.CursorShape.ArrowCursor,
            "orbit": Qt.CursorShape.OpenHandCursor,
            "pan": Qt.CursorShape.SizeAllCursor,
            "zoom": Qt.CursorShape.SizeVerCursor,
        }
        self.native.setCursor(cursors[mode])

    def set_grid_visible(self, visible: bool) -> None:
        self._grid_visible = bool(visible)
        self._grid.visible = self._grid_visible
        self._canvas.update()

    def set_axes_visible(self, visible: bool) -> None:
        self._axes_visible = bool(visible)
        for visual in (
            self._axis_x, self._axis_y, self._axis_z,
            self._axis_x_head, self._axis_y_head, self._axis_z_head,
            self._axis_origin,
        ):
            visual.visible = self._axes_visible
        self._canvas.update()

    def orientation_gizmo_axes(self) -> np.ndarray:
        from app.reconstruction_visualization import orientation_gizmo_projection

        return orientation_gizmo_projection(self._camera.azimuth, self._camera.elevation)

    def _create_grid_visual(self):
        grid = scene.visuals.GridLines(color=(1, 1, 1, 0.08), parent=self._view.scene)
        grid.visible = self._grid_visible
        return grid

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
    def interaction_mode(self) -> str:
        return self._camera.interaction_mode

    @property
    def trajectory_count(self) -> int:
        return len(self._trajectory_positions)

    @property
    def trajectory_visible(self) -> bool:
        return self._trajectory_visible

    @property
    def trajectory_display_count(self) -> int:
        return self._trajectory_display_count

    @property
    def sparse_landmarks_visible(self) -> bool:
        return self._sparse_landmarks_visible

    @property
    def sparse_landmark_count(self) -> int:
        return self._sparse_landmark_total_count

    @property
    def sparse_landmark_display_count(self) -> int:
        return self._sparse_landmark_display_count

    @property
    def sparse_map_revision(self) -> int | None:
        return self._sparse_map_revision

    @property
    def has_sparse_map(self) -> bool:
        return self._sparse_landmark_display_count > 0

    @property
    def keyframes_visible(self) -> bool:
        return self._keyframes_visible

    @property
    def keyframe_count(self) -> int:
        return self._keyframe_total_count

    @property
    def keyframe_display_count(self) -> int:
        return self._keyframe_display_count

    @property
    def keyframe_map_revision(self) -> int | None:
        return self._keyframe_map_revision

    @property
    def keyframe_ids(self) -> tuple[str, ...]:
        return self._keyframe_ids

    @property
    def has_keyframes(self) -> bool:
        return self._keyframe_display_count > 0

    @property
    def offline_registered_camera_count(self) -> int:
        return int(self._offline_camera_matrices.shape[0])

    @property
    def offline_camera_ids(self) -> tuple[str, ...]:
        return self._offline_camera_ids

    @property
    def offline_camera_path_visible(self) -> bool:
        return bool(self._offline_camera_path.visible)

    @property
    def offline_camera_path_display_count(self) -> int:
        return int(self._offline_camera_path_display_count)

    @property
    def selected_offline_camera_index(self) -> int | None:
        return self._selected_offline_camera_index

    @property
    def selected_offline_camera_id(self) -> str | None:
        index = self._selected_offline_camera_index
        if index is None or index < 0 or index >= len(self._offline_camera_ids):
            return None
        return self._offline_camera_ids[index]

    @property
    def selected_offline_camera_visible(self) -> bool:
        return bool(
            self._selected_offline_camera_index is not None
            and self._selected_offline_camera_frustum.visible
        )

    @property
    def offline_visualization_loaded(self) -> bool:
        return bool(self._offline_visualization_loaded)

    @property
    def offline_session_id(self) -> str | None:
        return self._offline_session_id

    @property
    def offline_scale_status(self) -> str | None:
        return self._offline_scale_status

    @property
    def world_axis_length(self) -> float:
        return float(self._world_axis_length)

    @property
    def current_camera_position(self) -> np.ndarray | None:
        if self._current_camera_position is None:
            return None
        return np.array(self._current_camera_position, copy=True)

    @property
    def has_pose(self) -> bool:
        return self._has_pose

    @property
    def grid_visible(self) -> bool:
        return self._grid_visible

    @property
    def axes_visible(self) -> bool:
        return self._axes_visible

    @property
    def is_interacting(self) -> bool:
        return self._is_interacting

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._canvas.close()

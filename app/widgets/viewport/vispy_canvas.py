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
_VALID_INTERACTION_MODES = {"select", "orbit", "pan", "zoom", "measure"}

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

        if self._interaction_mode in {"select", "measure"}:
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
            "measure": Qt.CursorShape.CrossCursor,
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
    measurement_point_picked = pyqtSignal(object)
    measurement_cancel_requested = pyqtSignal()

    MAX_TRAJECTORY_POSES = 20_000
    MAX_TRAJECTORY_DISPLAY_POINTS = 4_096
    MAX_SPARSE_LANDMARK_DISPLAY_POINTS = 8_192
    MAX_KEYFRAME_DISPLAY_FRUSTUMS = 96
    MAX_OFFLINE_CAMERA_PATH_POINTS = 1_024
    MAX_CANONICAL_CAMERA_FRUSTUMS = 192
    MAX_CANONICAL_CAMERA_TRAJECTORY_POINTS = 4_096
    MAX_SEMANTIC_OBJECT_DISPLAY = 256
    MAX_CANONICAL_SEMANTIC_LABEL_DISPLAY = 64
    CANONICAL_INSPECTION_OVERLAY_FONT_SIZE = 14.0

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
        # Phase 9.3F layer switches are presentation-only session preferences.
        # They never unload evidence or change reconstruction/measurement authority.
        self._canonical_mesh_layer_visible = True
        self._canonical_camera_frustums_layer_visible = True
        self._canonical_camera_trajectory_layer_visible = True
        self._semantic_support_layer_visible = True
        self._canonical_semantic_labels_layer_visible = True
        self._canonical_semantic_selection_layer_visible = True
        self._canonical_inspection_overlay_layer_visible = True
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
        self._offline_sparse_envelope = scene.visuals.Line(
            pos=np.zeros((0, 3), dtype=np.float32),
            connect="segments",
            color=(0.36, 0.84, 0.64, 0.62),
            width=1.3,
            parent=self._view.scene,
        )
        self._offline_sparse_envelope.visible = False
        self._offline_sparse_envelope_visible = True
        self._offline_sparse_envelope_available = False
        self._offline_sparse_envelope_min = None
        self._offline_sparse_envelope_max = None
        # Phase 9.3B adds one persistent canonical surface visual to the
        # existing Reconstruction SceneCanvas. The mesh is presentation-only;
        # canonical authority remains with PersistentReconstructionScene.
        self._canonical_mesh = scene.visuals.Mesh(parent=self._view.scene, shading="smooth")
        self._canonical_mesh.visible = False
        self._canonical_mesh_loaded = False
        self._canonical_mesh_scene_id = ""
        self._canonical_mesh_artifact_id = ""
        self._canonical_mesh_source_path = ""
        self._canonical_mesh_vertex_count = 0
        self._canonical_mesh_face_count = 0
        self._canonical_mesh_bounds_min = None
        self._canonical_mesh_bounds_max = None
        self._canonical_mesh_vertices_viewer = np.zeros((0, 3), dtype=np.float32)
        self._canonical_mesh_faces = np.zeros((0, 3), dtype=np.uint32)
        # Interactive ruler overlay is presentation-only and always resolves
        # points against the canonical mesh, never against Gaussian/NeRF data.
        self._interactive_measure_line = scene.visuals.Line(
            pos=np.zeros((0, 3), dtype=np.float32),
            connect="strip",
            color=(0.23, 0.78, 0.85, 0.98),
            width=2.4,
            parent=self._view.scene,
        )
        self._interactive_measure_markers = scene.visuals.Markers(parent=self._view.scene)
        self._interactive_measure_markers.set_data(
            pos=np.zeros((0, 3), dtype=np.float32),
            face_color=(0.23, 0.78, 0.85, 1.0),
            edge_color=(0.92, 0.99, 1.0, 1.0),
            size=9.0,
            edge_width=1.2,
        )
        self._interactive_measure_line.visible = False
        self._interactive_measure_markers.visible = False
        # Phase 9.3C keeps persistent canonical-scene cameras separate from
        # live/offline sparse-SfM visuals.  They share the same reconstruction
        # coordinate transform but never inherit canonical geometry authority.
        self._canonical_camera_frustums = scene.visuals.Line(
            pos=np.zeros((0, 3), dtype=np.float32),
            connect="segments",
            color=(0.42, 0.78, 0.98, 0.64),
            width=1.15,
            parent=self._view.scene,
        )
        self._canonical_camera_centers = scene.visuals.Markers(parent=self._view.scene)
        self._canonical_camera_centers.set_data(
            pos=np.zeros((0, 3), dtype=np.float32),
            face_color=(0.42, 0.78, 0.98, 0.82),
            size=4.2,
            edge_width=0,
        )
        self._canonical_camera_trajectory = scene.visuals.Line(
            pos=np.zeros((0, 3), dtype=np.float32),
            connect="strip",
            color=(0.42, 0.78, 0.98, 0.72),
            width=1.45,
            parent=self._view.scene,
        )
        self._canonical_camera_start = scene.visuals.Markers(parent=self._view.scene)
        self._canonical_camera_start.set_data(
            pos=np.zeros((0, 3), dtype=np.float32),
            face_color=(0.42, 0.78, 0.98, 0.95),
            size=5.6,
            edge_width=0,
        )
        self._canonical_camera_loaded = False
        self._canonical_camera_scene_id = ""
        self._canonical_camera_artifact_id = ""
        self._canonical_camera_provider_id = ""
        self._canonical_camera_matrices = np.zeros((0, 4, 4), dtype=np.float64)
        self._canonical_camera_positions = np.zeros((0, 3), dtype=np.float32)
        self._canonical_camera_ids: tuple[str, ...] = ()
        self._canonical_camera_total_count = 0
        self._canonical_camera_frustum_display_count = 0
        self._canonical_camera_trajectory_display_count = 0
        self._canonical_camera_frustums.visible = False
        self._canonical_camera_centers.visible = False
        self._canonical_camera_trajectory.visible = False
        self._canonical_camera_start.visible = False
        self._semantic_object_boxes = scene.visuals.Line(
            pos=np.zeros((0, 3), dtype=np.float32),
            connect="segments",
            color=(0.64, 0.46, 0.94, 0.48),
            width=1.2,
            parent=self._view.scene,
        )
        self._semantic_object_centers = scene.visuals.Markers(parent=self._view.scene)
        self._semantic_object_centers.set_data(
            pos=np.zeros((0, 3), dtype=np.float32),
            face_color=(0.64, 0.46, 0.94, 0.76),
            size=4.2,
            edge_width=0,
        )
        self._selected_semantic_object_box = scene.visuals.Line(
            pos=np.zeros((0, 3), dtype=np.float32),
            connect="segments",
            color=(1.0, 0.80, 0.22, 0.98),
            width=2.4,
            parent=self._view.scene,
        )
        self._selected_semantic_object_center = scene.visuals.Markers(parent=self._view.scene)
        self._selected_semantic_object_center.set_data(
            pos=np.zeros((0, 3), dtype=np.float32),
            face_color=(1.0, 0.80, 0.22, 1.0),
            size=7.0,
            edge_width=0,
        )
        # Phase 9.3D keeps semantic surface membership separate from the
        # canonical MeshVisual. Only a selected object's already-persisted
        # canonical face support is drawn as a transient presentation overlay.
        self._selected_canonical_semantic_mesh = scene.visuals.Mesh(
            parent=self._view.scene,
            shading="smooth",
            color=(1.0, 0.66, 0.12, 0.58),
        )
        self._selected_canonical_semantic_mesh.visible = False
        try:
            self._selected_canonical_semantic_mesh.set_gl_state(
                "translucent",
                depth_test=True,
                blend=True,
                polygon_offset_fill=True,
                polygon_offset=(1.0, 1.0),
            )
        except (TypeError, ValueError):
            # Older qualified VisPy releases may not expose polygon-offset
            # keywords through Visual.set_gl_state(). The highlight remains
            # valid without the optional depth-fighting mitigation.
            self._selected_canonical_semantic_mesh.set_gl_state(
                "translucent", depth_test=True, blend=True
            )
        self._canonical_semantic_labels = scene.visuals.Text(
            text="",
            pos=np.zeros((1, 3), dtype=np.float32),
            color=(0.88, 0.82, 0.98, 0.78),
            font_size=8.0,
            anchor_x="center",
            anchor_y="bottom",
            parent=self._view.scene,
        )
        self._canonical_semantic_labels.visible = False
        self._selected_canonical_semantic_label = scene.visuals.Text(
            text="",
            pos=np.zeros((1, 3), dtype=np.float32),
            color=(1.0, 0.84, 0.34, 1.0),
            font_size=12.0,
            bold=True,
            anchor_x="center",
            anchor_y="bottom",
            parent=self._view.scene,
        )
        self._selected_canonical_semantic_label.visible = False
        # Phase 9.3E uses one bounded 3D text callout for source-authored
        # measurement validity and exact selected-object QC. Numeric values are
        # supplied only after E3 validity qualification; this visual never
        # derives dimensions from semantic support geometry.
        self._canonical_inspection_overlay = scene.visuals.Text(
            text="",
            pos=np.zeros((1, 3), dtype=np.float32),
            color=(0.94, 0.95, 0.98, 0.98),
            # Phase 9.3E pre-commit readability refinement: keep the bounded
            # callout compact, but make uncertainty/rejection/QC text clearly
            # legible at the project's 1366x768 minimum desktop target.
            font_size=self.CANONICAL_INSPECTION_OVERLAY_FONT_SIZE,
            bold=True,
            anchor_x="center",
            anchor_y="bottom",
            parent=self._view.scene,
        )
        self._canonical_inspection_overlay.visible = False
        self._canonical_inspection_overlay_scene_id = ""
        self._canonical_inspection_overlay_object_id = ""
        self._canonical_inspection_overlay_text = ""
        self._canonical_inspection_overlay_anchor = np.zeros((3,), dtype=np.float32)
        # Phase 9.3J genuine canonical dimension lines. Endpoints are supplied
        # only by the qualified measurement-registration presentation; this
        # visual never derives dimensions from semantic support bounds.
        self._canonical_registered_dimensions = scene.visuals.Line(
            pos=np.zeros((0, 3), dtype=np.float32),
            connect="segments",
            color=(1.0, 0.79, 0.20, 1.0),
            width=2.4,
            parent=self._view.scene,
        )
        self._canonical_registered_dimensions.visible = False
        self._canonical_registered_dimension_endpoints = scene.visuals.Markers(parent=self._view.scene)
        self._canonical_registered_dimension_endpoints.set_data(
            pos=np.zeros((0, 3), dtype=np.float32),
            face_color=(1.0, 0.79, 0.20, 1.0),
            edge_color=(0.08, 0.08, 0.08, 1.0),
            size=7.0,
            edge_width=1.0,
        )
        self._canonical_registered_dimension_endpoints.visible = False
        self._canonical_registered_dimension_labels = scene.visuals.Text(
            text="",
            pos=np.zeros((1, 3), dtype=np.float32),
            color=(1.0, 0.88, 0.42, 1.0),
            font_size=12.0,
            bold=True,
            anchor_x="center",
            anchor_y="bottom",
            parent=self._view.scene,
        )
        self._canonical_registered_dimension_labels.visible = False
        self._canonical_registered_dimension_scene_id = ""
        self._canonical_registered_dimension_object_id = ""
        self._canonical_registered_dimension_count = 0
        self._canonical_registered_dimension_text: tuple[str, ...] = ()
        self._canonical_semantic_loaded = False
        self._canonical_semantic_scene_id = ""
        self._canonical_semantic_artifact_id = ""
        self._canonical_semantic_canonical_artifact_id = ""
        self._canonical_semantic_object_ids: tuple[str, ...] = ()
        self._canonical_semantic_labels_text: tuple[str, ...] = ()
        self._canonical_semantic_label_positions = np.zeros((0, 3), dtype=np.float32)
        self._canonical_semantic_surface_vertices: tuple[np.ndarray, ...] = ()
        self._canonical_semantic_surface_faces: tuple[np.ndarray, ...] = ()
        self._canonical_semantic_label_display_count = 0
        self._selected_canonical_semantic_surface_loaded = False
        self._semantic_objects_visible = True
        self._semantic_objects_available = False
        self._semantic_bounds_min = np.zeros((0, 3), dtype=np.float64)
        self._semantic_bounds_max = np.zeros((0, 3), dtype=np.float64)
        self._semantic_centroids_viewer = np.zeros((0, 3), dtype=np.float32)
        self._semantic_object_ids: tuple[str, ...] = ()
        self._semantic_object_total_count = 0
        self._selected_semantic_object_index: int | None = None
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

        position = camera_position_in_viewer(T_reconstruction_from_camera).astype(np.float32)
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
        from app.reconstruction_visualization import adaptive_camera_frustum_depth

        frustum = self._frustum_builder(
            T_reconstruction_from_camera,
            intrinsics_matrix=self._intrinsics_matrix,
            image_size=self._image_size,
            depth=adaptive_camera_frustum_depth(self._world_axis_length),
        )
        self._camera_frustum.set_data(pos=frustum, connect="segments")
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
        self._offline_sparse_envelope.set_data(
            pos=np.zeros((0, 3), dtype=np.float32), connect="segments"
        )
        self._offline_sparse_envelope.visible = False
        self._offline_sparse_envelope_available = False
        self._offline_sparse_envelope_min = None
        self._offline_sparse_envelope_max = None
        self.clear_semantic_objects(update_canvas=False)
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
        self._refresh_canonical_camera_layer_visibility()
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

    def set_offline_sparse_envelope(
        self,
        minimum_reconstruction: np.ndarray | tuple[float, float, float],
        maximum_reconstruction: np.ndarray | tuple[float, float, float],
    ) -> None:
        """Show one robust D4 sparse-SfM presentation envelope.

        The box is a visualization aid in arbitrary reconstruction gauge. It is
        never promoted to canonical geometry or object dimensions.
        """

        lower = np.asarray(minimum_reconstruction, dtype=np.float64)
        upper = np.asarray(maximum_reconstruction, dtype=np.float64)
        if lower.shape != (3,) or upper.shape != (3,):
            raise ValueError("Offline sparse envelope bounds must be 3-vectors.")
        if not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(upper < lower):
            raise ValueError("Offline sparse envelope bounds must be finite and ordered.")
        from app.reconstruction_visualization import sparse_landmarks_to_viewer

        x0, y0, z0 = lower
        x1, y1, z1 = upper
        corners = np.asarray(
            [
                [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
            ],
            dtype=np.float64,
        )
        edges = (
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        )
        viewer = sparse_landmarks_to_viewer(corners)
        segments = np.asarray([viewer[index] for edge in edges for index in edge], dtype=np.float32)
        self._offline_sparse_envelope.set_data(pos=segments, connect="segments")
        self._offline_sparse_envelope_min = np.array(lower, copy=True)
        self._offline_sparse_envelope_max = np.array(upper, copy=True)
        self._offline_sparse_envelope_available = True
        self._offline_sparse_envelope.visible = bool(
            self._offline_sparse_envelope_visible and self._offline_visualization_loaded
        )
        self._canvas.update()

    def clear_offline_sparse_envelope(self) -> None:
        self._offline_sparse_envelope.set_data(
            pos=np.zeros((0, 3), dtype=np.float32), connect="segments"
        )
        self._offline_sparse_envelope.visible = False
        self._offline_sparse_envelope_available = False
        self._offline_sparse_envelope_min = None
        self._offline_sparse_envelope_max = None
        self._canvas.update()

    def set_offline_sparse_envelope_visible(self, visible: bool) -> None:
        self._offline_sparse_envelope_visible = bool(visible)
        self._offline_sparse_envelope.visible = bool(
            self._offline_sparse_envelope_visible
            and self._offline_sparse_envelope_available
            and self._offline_visualization_loaded
        )
        self._canvas.update()

    def set_canonical_mesh(
        self,
        *,
        vertices_viewer: np.ndarray,
        faces: np.ndarray,
        vertex_colors_rgba: np.ndarray,
        scene_id: str,
        artifact_id: str,
        source_path: str,
    ) -> None:
        """Display one already-validated persistent canonical surface.

        Coordinates are already in the application's presentation-only Z-up
        viewer frame. This method never establishes scale or geometry authority.
        """

        vertices = np.asarray(vertices_viewer, dtype=np.float32)
        triangles = np.asarray(faces, dtype=np.uint32)
        colors = np.asarray(vertex_colors_rgba, dtype=np.float32)
        if vertices.ndim != 2 or vertices.shape[1:] != (3,) or vertices.shape[0] < 3:
            raise ValueError("vertices_viewer must contain at least three (N, 3) vertices.")
        if not np.isfinite(vertices).all():
            raise ValueError("vertices_viewer must be finite.")
        if triangles.ndim != 2 or triangles.shape[1:] != (3,) or triangles.shape[0] < 1:
            raise ValueError("faces must contain at least one triangular face.")
        if np.any(triangles >= vertices.shape[0]):
            raise ValueError("faces reference vertices outside the canonical mesh.")
        if colors.shape != (vertices.shape[0], 4):
            raise ValueError("vertex_colors_rgba must have shape (N, 4).")
        if not np.isfinite(colors).all() or np.any(colors < 0.0) or np.any(colors > 1.0):
            raise ValueError("vertex_colors_rgba must be finite in [0, 1].")
        scene_id = str(scene_id).strip()
        artifact_id = str(artifact_id).strip()
        source_path = str(source_path).strip()
        if not scene_id or not artifact_id or not source_path:
            raise ValueError("Canonical mesh scene/artifact/source identity is required.")

        vertices = np.ascontiguousarray(vertices)
        triangles = np.ascontiguousarray(triangles)
        colors = np.ascontiguousarray(colors)
        self._canonical_mesh.set_data(
            vertices=vertices,
            faces=triangles,
            vertex_colors=colors,
        )
        self._canonical_mesh_loaded = True
        self._canonical_mesh.visible = bool(self._canonical_mesh_layer_visible)
        self._canonical_mesh_scene_id = scene_id
        self._canonical_mesh_artifact_id = artifact_id
        self._canonical_mesh_source_path = source_path
        self._canonical_mesh_vertex_count = int(vertices.shape[0])
        self._canonical_mesh_face_count = int(triangles.shape[0])
        self._canonical_mesh_bounds_min = vertices.min(axis=0).astype(np.float32)
        self._canonical_mesh_bounds_max = vertices.max(axis=0).astype(np.float32)
        self._canonical_mesh_vertices_viewer = np.array(vertices, copy=True)
        self._canonical_mesh_faces = np.array(triangles, copy=True)
        self.clear_interactive_measurement_overlay(update_canvas=False)
        self._refresh_world_axis_scale()
        self._canvas.update()

    def clear_canonical_mesh(self, *, update_canvas: bool = True) -> None:
        # Semantic surface overlays are valid only while this exact canonical
        # mesh remains loaded. Clear them first so direct canvas lifecycle calls
        # cannot leave stale membership visuals behind.
        self.clear_canonical_semantic_objects(update_canvas=False)
        # MeshVisual.set_data() immediately recomputes bounds.  Some qualified
        # VisPy releases therefore reject a zero-vertex mesh before visibility
        # can be disabled.  Never use an empty set_data() call as a clear
        # operation: hide the current node and, when it owns loaded surface
        # data, detach it from the scenegraph and replace it with one fresh
        # empty hidden Mesh visual.  The next canonical load overwrites that
        # visual normally through set_canonical_mesh().
        self._canonical_mesh.visible = False
        if self._canonical_mesh_loaded:
            self._canonical_mesh.parent = None
            self._canonical_mesh = scene.visuals.Mesh(
                parent=self._view.scene,
                shading="smooth",
            )
            self._canonical_mesh.visible = False
        self._canonical_mesh_loaded = False
        self._canonical_mesh_scene_id = ""
        self._canonical_mesh_artifact_id = ""
        self._canonical_mesh_source_path = ""
        self._canonical_mesh_vertex_count = 0
        self._canonical_mesh_face_count = 0
        self._canonical_mesh_bounds_min = None
        self._canonical_mesh_bounds_max = None
        self._canonical_mesh_vertices_viewer = np.zeros((0, 3), dtype=np.float32)
        self._canonical_mesh_faces = np.zeros((0, 3), dtype=np.uint32)
        self.clear_interactive_measurement_overlay(update_canvas=False)
        self._refresh_world_axis_scale()
        if update_canvas:
            self._canvas.update()

    def set_canonical_mesh_visible(self, visible: bool) -> None:
        """Toggle only the materialized canonical surface presentation layer."""

        self._canonical_mesh_layer_visible = bool(visible)
        self._canonical_mesh.visible = bool(
            self._canonical_mesh_loaded and self._canonical_mesh_layer_visible
        )
        self._canvas.update()

    def set_canonical_camera_frustums_visible(self, visible: bool) -> None:
        self._canonical_camera_frustums_layer_visible = bool(visible)
        self._refresh_canonical_camera_layer_visibility()
        self._canvas.update()

    def set_canonical_camera_trajectory_visible(self, visible: bool) -> None:
        self._canonical_camera_trajectory_layer_visible = bool(visible)
        self._refresh_canonical_camera_layer_visibility()
        self._canvas.update()

    def _refresh_canonical_camera_layer_visibility(self) -> None:
        frustums = bool(
            self._canonical_camera_loaded
            and self._keyframes_visible
            and self._canonical_camera_frustums_layer_visible
        )
        trajectory = bool(
            self._canonical_camera_loaded
            and self._trajectory_visible
            and self._canonical_camera_trajectory_layer_visible
        )
        self._canonical_camera_frustums.visible = frustums
        self._canonical_camera_centers.visible = frustums
        self._canonical_camera_trajectory.visible = bool(
            trajectory and self._canonical_camera_trajectory_display_count >= 2
        )
        self._canonical_camera_start.visible = trajectory

    def set_canonical_camera_trajectory(
        self,
        *,
        matrices_reconstruction: np.ndarray,
        camera_ids: tuple[str, ...] | list[str],
        scene_id: str,
        artifact_id: str,
        provider_id: str,
        total_count: int | None = None,
    ) -> None:
        """Render persisted camera frustums/path in the canonical reconstruction frame.

        The camera trajectory remains supporting geometry/presentation context.  This
        method does not establish scale or canonical geometry authority.
        """

        from app.reconstruction_visualization import (
            adaptive_camera_frustum_depth,
            keyframe_centers_in_viewer,
            keyframe_display_indices,
            keyframe_frustum_segments_in_viewer,
            trajectory_display_indices,
        )

        matrices = np.asarray(matrices_reconstruction, dtype=np.float64)
        if matrices.ndim != 3 or matrices.shape[1:] != (4, 4) or matrices.shape[0] < 1:
            raise ValueError("matrices_reconstruction must have shape (N, 4, 4) with N >= 1.")
        if not np.isfinite(matrices).all():
            raise ValueError("matrices_reconstruction must be finite.")
        ids = tuple(str(value).strip() for value in camera_ids)
        if len(ids) != matrices.shape[0] or any(not value for value in ids):
            raise ValueError("camera_ids must contain one non-empty ID per canonical camera pose.")
        if len(set(ids)) != len(ids):
            raise ValueError("camera_ids must be unique.")
        scene_id = str(scene_id).strip()
        artifact_id = str(artifact_id).strip()
        provider_id = str(provider_id).strip()
        if not scene_id or not artifact_id or not provider_id:
            raise ValueError("Canonical camera scene/artifact/provider identity is required.")
        total = matrices.shape[0] if total_count is None else int(total_count)
        if total < matrices.shape[0]:
            raise ValueError("total_count cannot be smaller than persisted canonical camera poses.")

        positions = keyframe_centers_in_viewer(matrices)
        frustum_indices = keyframe_display_indices(
            matrices.shape[0], max_keyframes=self.MAX_CANONICAL_CAMERA_FRUSTUMS
        )
        trajectory_indices = trajectory_display_indices(
            positions.shape[0], max_points=self.MAX_CANONICAL_CAMERA_TRAJECTORY_POINTS
        )
        frustum_matrices = np.ascontiguousarray(matrices[frustum_indices])
        frustum_positions = np.ascontiguousarray(positions[frustum_indices])
        trajectory_positions = np.ascontiguousarray(positions[trajectory_indices])

        depth = adaptive_camera_frustum_depth(self._world_axis_length)
        frustum_segments = keyframe_frustum_segments_in_viewer(frustum_matrices, depth=depth)
        self._canonical_camera_frustums.set_data(pos=frustum_segments, connect="segments")
        self._canonical_camera_centers.set_data(
            pos=frustum_positions,
            face_color=(0.42, 0.78, 0.98, 0.82),
            size=4.2,
            edge_width=0,
        )
        self._canonical_camera_trajectory.set_data(pos=trajectory_positions, connect="strip")
        self._canonical_camera_start.set_data(
            pos=positions[:1],
            face_color=(0.42, 0.78, 0.98, 0.95),
            size=5.6,
            edge_width=0,
        )
        self._canonical_camera_loaded = True
        self._canonical_camera_scene_id = scene_id
        self._canonical_camera_artifact_id = artifact_id
        self._canonical_camera_provider_id = provider_id
        self._canonical_camera_matrices = np.ascontiguousarray(matrices)
        self._canonical_camera_positions = np.ascontiguousarray(positions)
        self._canonical_camera_ids = ids
        self._canonical_camera_total_count = total
        self._canonical_camera_frustum_display_count = int(frustum_matrices.shape[0])
        self._canonical_camera_trajectory_display_count = int(trajectory_positions.shape[0])
        self._refresh_canonical_camera_layer_visibility()
        self._refresh_world_axis_scale()
        # Frustum depth depends on the refreshed scene scale. Rebuild once if
        # the axis scale changed materially after camera positions were added.
        depth = adaptive_camera_frustum_depth(self._world_axis_length)
        frustum_segments = keyframe_frustum_segments_in_viewer(frustum_matrices, depth=depth)
        self._canonical_camera_frustums.set_data(pos=frustum_segments, connect="segments")
        self._canvas.update()

    def clear_canonical_camera_trajectory(self, *, update_canvas: bool = True) -> None:
        self._canonical_camera_frustums.set_data(
            pos=np.zeros((0, 3), dtype=np.float32), connect="segments"
        )
        self._canonical_camera_centers.set_data(pos=np.zeros((0, 3), dtype=np.float32))
        self._canonical_camera_trajectory.set_data(
            pos=np.zeros((0, 3), dtype=np.float32), connect="strip"
        )
        self._canonical_camera_start.set_data(pos=np.zeros((0, 3), dtype=np.float32))
        self._canonical_camera_frustums.visible = False
        self._canonical_camera_centers.visible = False
        self._canonical_camera_trajectory.visible = False
        self._canonical_camera_start.visible = False
        self._canonical_camera_loaded = False
        self._canonical_camera_scene_id = ""
        self._canonical_camera_artifact_id = ""
        self._canonical_camera_provider_id = ""
        self._canonical_camera_matrices = np.zeros((0, 4, 4), dtype=np.float64)
        self._canonical_camera_positions = np.zeros((0, 3), dtype=np.float32)
        self._canonical_camera_ids = ()
        self._canonical_camera_total_count = 0
        self._canonical_camera_frustum_display_count = 0
        self._canonical_camera_trajectory_display_count = 0
        self._refresh_world_axis_scale()
        if update_canvas:
            self._canvas.update()

    def set_canonical_semantic_objects(
        self,
        *,
        object_ids: tuple[str, ...] | list[str],
        labels: tuple[str, ...] | list[str],
        label_positions_viewer: np.ndarray,
        surface_vertices_viewer: tuple[np.ndarray, ...] | list[np.ndarray],
        surface_faces: tuple[np.ndarray, ...] | list[np.ndarray],
        scene_id: str,
        semantic_artifact_id: str,
        canonical_artifact_id: str,
        preserve_selected_mesh_visual: bool = False,
    ) -> None:
        """Load validated 9.3D semantic membership for canonical presentation.

        The supplied compact surface meshes are derived from persisted semantic
        face indices and the already-loaded canonical viewer mesh. They remain
        presentation overlays only and never replace that canonical mesh.
        """

        if not self._canonical_mesh_loaded:
            raise ValueError("Canonical semantic binding requires a loaded canonical mesh.")
        scene_id = str(scene_id).strip()
        semantic_artifact_id = str(semantic_artifact_id).strip()
        canonical_artifact_id = str(canonical_artifact_id).strip()
        if not scene_id or not semantic_artifact_id or not canonical_artifact_id:
            raise ValueError("Canonical semantic scene/artifact identity is required.")
        if scene_id != self._canonical_mesh_scene_id:
            raise ValueError("Canonical semantic scene identity does not match the loaded mesh.")
        if canonical_artifact_id != self._canonical_mesh_artifact_id:
            raise ValueError("Canonical semantic binding targets a different canonical artifact.")

        ids = tuple(str(value).strip() for value in object_ids)
        label_values = tuple(str(value).strip() for value in labels)
        positions = np.asarray(label_positions_viewer, dtype=np.float32)
        vertices_values = tuple(np.asarray(value, dtype=np.float32) for value in surface_vertices_viewer)
        face_values = tuple(np.asarray(value, dtype=np.uint32) for value in surface_faces)
        count = len(ids)
        if count > self.MAX_SEMANTIC_OBJECT_DISPLAY:
            raise ValueError("Canonical semantic presentation exceeds the bounded display limit.")
        if any(not value for value in ids) or len(set(ids)) != count:
            raise ValueError("Canonical semantic object_ids must be unique and non-empty.")
        if self._semantic_objects_available and ids != self._semantic_object_ids:
            raise ValueError("Canonical semantic object ordering must match the active semantic presentation.")
        if len(label_values) != count or any(not value for value in label_values):
            raise ValueError("Canonical semantic labels must match object_ids and remain non-empty.")
        if positions.shape != (count, 3) or not np.isfinite(positions).all():
            raise ValueError("Canonical semantic label positions must be finite with shape (N, 3).")
        if len(vertices_values) != count or len(face_values) != count:
            raise ValueError("Canonical semantic surface arrays must match object_ids.")

        normalized_vertices: list[np.ndarray] = []
        normalized_faces: list[np.ndarray] = []
        for vertices, faces in zip(vertices_values, face_values, strict=True):
            if vertices.size == 0:
                vertices = np.zeros((0, 3), dtype=np.float32)
            if faces.size == 0:
                faces = np.zeros((0, 3), dtype=np.uint32)
            if vertices.ndim != 2 or vertices.shape[1:] != (3,) or not np.isfinite(vertices).all():
                raise ValueError("Canonical semantic surface vertices must be finite (N, 3) arrays.")
            if faces.ndim != 2 or faces.shape[1:] != (3,):
                raise ValueError("Canonical semantic surface faces must have shape (M, 3).")
            if faces.shape[0] and (not vertices.shape[0] or np.any(faces >= vertices.shape[0])):
                raise ValueError("Canonical semantic surface faces reference missing local vertices.")
            normalized_vertices.append(np.ascontiguousarray(vertices, dtype=np.float32))
            normalized_faces.append(np.ascontiguousarray(faces, dtype=np.uint32))

        self.clear_canonical_semantic_objects(
            update_canvas=False,
            preserve_selected_mesh_visual=preserve_selected_mesh_visual,
        )
        self._canonical_semantic_loaded = bool(count)
        self._canonical_semantic_scene_id = scene_id
        self._canonical_semantic_artifact_id = semantic_artifact_id
        self._canonical_semantic_canonical_artifact_id = canonical_artifact_id
        self._canonical_semantic_object_ids = ids
        self._canonical_semantic_labels_text = label_values
        self._canonical_semantic_label_positions = np.ascontiguousarray(positions)
        self._canonical_semantic_surface_vertices = tuple(normalized_vertices)
        self._canonical_semantic_surface_faces = tuple(normalized_faces)

        label_count = min(count, self.MAX_CANONICAL_SEMANTIC_LABEL_DISPLAY)
        self._canonical_semantic_label_display_count = label_count
        if label_count:
            self._canonical_semantic_labels.text = list(label_values[:label_count])
            self._canonical_semantic_labels.pos = self._canonical_semantic_label_positions[:label_count]
        else:
            self._canonical_semantic_labels.text = ""
        self._refresh_semantic_layer_visibility()
        if self._selected_semantic_object_index is not None:
            self._refresh_selected_canonical_semantic_object()
        self._canvas.update()

    def clear_canonical_semantic_objects(
        self,
        *,
        update_canvas: bool = True,
        preserve_selected_mesh_visual: bool = False,
    ) -> None:
        self.clear_canonical_inspection_overlay(update_canvas=False)
        self.clear_canonical_registered_dimensions(update_canvas=False)
        self._canonical_semantic_labels.visible = False
        self._canonical_semantic_labels.text = ""
        self._selected_canonical_semantic_label.visible = False
        self._selected_canonical_semantic_label.text = ""
        self._clear_selected_canonical_semantic_mesh(
            preserve_visual=preserve_selected_mesh_visual
        )
        self._canonical_semantic_loaded = False
        self._canonical_semantic_scene_id = ""
        self._canonical_semantic_artifact_id = ""
        self._canonical_semantic_canonical_artifact_id = ""
        self._canonical_semantic_object_ids = ()
        self._canonical_semantic_labels_text = ()
        self._canonical_semantic_label_positions = np.zeros((0, 3), dtype=np.float32)
        self._canonical_semantic_surface_vertices = ()
        self._canonical_semantic_surface_faces = ()
        self._canonical_semantic_label_display_count = 0
        if update_canvas:
            self._canvas.update()

    def set_canonical_inspection_overlay(
        self,
        *,
        scene_id: str,
        object_id: str,
        anchor_viewer: np.ndarray,
        text: str,
    ) -> None:
        """Show bounded 9.3E source evidence for the selected canonical object.

        The text is already authority-qualified by the application presentation
        layer.  This canvas method only validates scene/object identity and
        places the callout in the existing canonical viewer coordinates.
        """

        if not self._canonical_semantic_loaded:
            raise ValueError("Canonical inspection overlay requires loaded canonical semantics.")
        scene_id = str(scene_id).strip()
        object_id = str(object_id).strip()
        text = str(text).strip()
        if not scene_id or not object_id or not text:
            raise ValueError("Canonical inspection overlay requires scene, object, and text.")
        if scene_id != self._canonical_semantic_scene_id:
            raise ValueError("Canonical inspection overlay scene does not match loaded semantics.")
        if object_id not in self._canonical_semantic_object_ids:
            raise ValueError("Canonical inspection overlay targets an unknown semantic object.")
        selected = self.selected_semantic_object_id
        if selected != object_id:
            raise ValueError("Canonical inspection overlay must target the currently selected object.")
        anchor = np.asarray(anchor_viewer, dtype=np.float32)
        if anchor.shape != (3,) or not np.isfinite(anchor).all():
            raise ValueError("Canonical inspection overlay anchor must be one finite 3D point.")

        self._canonical_inspection_overlay_scene_id = scene_id
        self._canonical_inspection_overlay_object_id = object_id
        self._canonical_inspection_overlay_text = text
        self._canonical_inspection_overlay_anchor = np.ascontiguousarray(anchor)
        self._canonical_inspection_overlay.text = text
        self._canonical_inspection_overlay.pos = self._canonical_inspection_overlay_anchor.reshape(1, 3)
        self._refresh_canonical_inspection_overlay_visibility()
        self._canvas.update()

    def clear_canonical_inspection_overlay(self, *, update_canvas: bool = True) -> None:
        self._canonical_inspection_overlay.visible = False
        self._canonical_inspection_overlay.text = ""
        self._canonical_inspection_overlay_scene_id = ""
        self._canonical_inspection_overlay_object_id = ""
        self._canonical_inspection_overlay_text = ""
        self._canonical_inspection_overlay_anchor = np.zeros((3,), dtype=np.float32)
        if update_canvas:
            self._canvas.update()

    def _refresh_canonical_inspection_overlay_visibility(self) -> None:
        selected = self.selected_semantic_object_id
        self._canonical_inspection_overlay.visible = bool(
            self._canonical_inspection_overlay_layer_visible
            and self._semantic_objects_visible
            and self._semantic_objects_available
            and self._canonical_semantic_loaded
            and self._canonical_inspection_overlay_text
            and selected is not None
            and selected == self._canonical_inspection_overlay_object_id
            and self._canonical_inspection_overlay_scene_id == self._canonical_semantic_scene_id
        )

    def set_canonical_registered_dimensions(
        self,
        *,
        scene_id: str,
        object_id: str,
        starts_reconstruction: np.ndarray,
        ends_reconstruction: np.ndarray,
        labels: tuple[str, ...] | list[str],
    ) -> None:
        """Render up to three already-qualified measurement dimensions."""

        if not self._canonical_semantic_loaded:
            raise ValueError("Canonical registered dimensions require loaded canonical semantics.")
        scene_id = str(scene_id).strip()
        object_id = str(object_id).strip()
        if scene_id != self._canonical_semantic_scene_id or object_id not in self._canonical_semantic_object_ids:
            raise ValueError("Canonical registered dimension binding does not match the loaded semantic scene.")
        if self.selected_semantic_object_id != object_id:
            raise ValueError("Canonical registered dimensions must target the selected semantic object.")
        starts = np.asarray(starts_reconstruction, dtype=np.float64)
        ends = np.asarray(ends_reconstruction, dtype=np.float64)
        texts = tuple(str(value).strip() for value in labels)
        if starts.ndim != 2 or starts.shape[1:] != (3,) or ends.shape != starts.shape:
            raise ValueError("Registered dimension endpoints must have matching shape (N, 3).")
        if not 1 <= starts.shape[0] <= 3 or len(texts) != starts.shape[0] or any(not value for value in texts):
            raise ValueError("Registered dimensions require 1-3 non-empty labeled segments.")
        if not np.isfinite(starts).all() or not np.isfinite(ends).all():
            raise ValueError("Registered dimension endpoints must be finite.")
        from app.reconstruction_visualization import reconstruction_points_to_viewer

        start_view = reconstruction_points_to_viewer(starts).astype(np.float32, copy=False)
        end_view = reconstruction_points_to_viewer(ends).astype(np.float32, copy=False)
        segments = np.empty((starts.shape[0] * 2, 3), dtype=np.float32)
        segments[0::2] = start_view
        segments[1::2] = end_view
        endpoints = np.concatenate([start_view, end_view], axis=0)
        midpoints = (start_view + end_view) * 0.5
        self._canonical_registered_dimensions.set_data(pos=segments, connect="segments")
        self._canonical_registered_dimension_endpoints.set_data(
            pos=endpoints,
            face_color=(1.0, 0.79, 0.20, 1.0),
            edge_color=(0.08, 0.08, 0.08, 1.0),
            size=7.0,
            edge_width=1.0,
        )
        self._canonical_registered_dimension_labels.text = list(texts)
        self._canonical_registered_dimension_labels.pos = midpoints
        self._canonical_registered_dimension_scene_id = scene_id
        self._canonical_registered_dimension_object_id = object_id
        self._canonical_registered_dimension_count = int(starts.shape[0])
        self._canonical_registered_dimension_text = texts
        self._refresh_canonical_registered_dimension_visibility()
        self._canvas.update()

    def clear_canonical_registered_dimensions(self, *, update_canvas: bool = True) -> None:
        self._canonical_registered_dimensions.visible = False
        self._canonical_registered_dimension_endpoints.visible = False
        self._canonical_registered_dimension_labels.visible = False
        self._canonical_registered_dimension_labels.text = ""
        self._canonical_registered_dimension_scene_id = ""
        self._canonical_registered_dimension_object_id = ""
        self._canonical_registered_dimension_count = 0
        self._canonical_registered_dimension_text = ()
        if update_canvas:
            self._canvas.update()

    def _refresh_canonical_registered_dimension_visibility(self) -> None:
        selected = self.selected_semantic_object_id
        visible = bool(
            self._canonical_inspection_overlay_layer_visible
            and self._semantic_objects_visible
            and self._semantic_objects_available
            and self._canonical_semantic_loaded
            and self._canonical_registered_dimension_count
            and selected == self._canonical_registered_dimension_object_id
            and self._canonical_registered_dimension_scene_id == self._canonical_semantic_scene_id
        )
        self._canonical_registered_dimensions.visible = visible
        self._canonical_registered_dimension_endpoints.visible = visible
        self._canonical_registered_dimension_labels.visible = visible

    def _clear_selected_canonical_semantic_mesh(self, *, preserve_visual: bool = False) -> None:
        # Match the qualified 9.3B MeshVisual lifecycle: never submit a zero-
        # vertex mesh to VisPy, because MeshData bounds reduction may crash.
        # An exact same-scene rebind may safely keep the previous non-empty
        # MeshVisual hidden until new non-empty membership is validated and
        # written in place. Genuine clear/different-scene paths still detach
        # and recreate the node so stale semantic geometry cannot survive.
        self._selected_canonical_semantic_mesh.visible = False
        if preserve_visual and self._selected_canonical_semantic_surface_loaded:
            return
        if self._selected_canonical_semantic_surface_loaded:
            self._selected_canonical_semantic_mesh.parent = None
            self._selected_canonical_semantic_mesh = scene.visuals.Mesh(
                parent=self._view.scene,
                shading="smooth",
                color=(1.0, 0.66, 0.12, 0.58),
            )
            self._selected_canonical_semantic_mesh.visible = False
            try:
                self._selected_canonical_semantic_mesh.set_gl_state(
                    "translucent",
                    depth_test=True,
                    blend=True,
                    polygon_offset_fill=True,
                    polygon_offset=(1.0, 1.0),
                )
            except (TypeError, ValueError):
                self._selected_canonical_semantic_mesh.set_gl_state(
                    "translucent", depth_test=True, blend=True
                )
        self._selected_canonical_semantic_surface_loaded = False

    def _refresh_selected_canonical_semantic_object(self) -> None:
        self._selected_canonical_semantic_label.visible = False
        self._selected_canonical_semantic_label.text = ""
        index = self._selected_semantic_object_index
        if (
            index is None
            or not self._canonical_semantic_loaded
            or index < 0
            or index >= len(self._canonical_semantic_object_ids)
        ):
            self._clear_selected_canonical_semantic_mesh()
            return
        if index >= len(self._semantic_object_ids) or (
            self._canonical_semantic_object_ids[index] != self._semantic_object_ids[index]
        ):
            self._clear_selected_canonical_semantic_mesh()
            return

        self._selected_canonical_semantic_label.text = self._canonical_semantic_labels_text[index]
        self._selected_canonical_semantic_label.pos = self._canonical_semantic_label_positions[index : index + 1]
        master = bool(self._semantic_objects_visible and self._semantic_objects_available)
        self._selected_canonical_semantic_label.visible = bool(
            master and self._canonical_semantic_labels_layer_visible
        )

        vertices = self._canonical_semantic_surface_vertices[index]
        faces = self._canonical_semantic_surface_faces[index]
        if vertices.shape[0] and faces.shape[0]:
            # Switching directly between two valid non-empty object surfaces does
            # not require the 9.3B empty-mesh clear lifecycle. Reuse the existing
            # MeshVisual and replace its non-empty data in place; detach/recreate is
            # reserved for a genuine clear/unavailable surface.
            self._selected_canonical_semantic_mesh.set_data(vertices=vertices, faces=faces)
            self._selected_canonical_semantic_surface_loaded = True
            self._selected_canonical_semantic_mesh.visible = bool(
                master and self._canonical_semantic_selection_layer_visible
            )
        else:
            self._clear_selected_canonical_semantic_mesh()

    def set_semantic_objects(
        self,
        *,
        bounds_min_reconstruction: np.ndarray,
        bounds_max_reconstruction: np.ndarray,
        centroids_reconstruction: np.ndarray,
        object_ids: tuple[str, ...] | list[str],
        total_count: int | None = None,
    ) -> None:
        """Render bounded D5 semantic-context object boxes and centroids.

        These wireframes are presentation-only semantic support bounds. They
        are not measurement boxes, defect decisions, or canonical geometry.
        """

        lower = np.asarray(bounds_min_reconstruction, dtype=np.float64)
        upper = np.asarray(bounds_max_reconstruction, dtype=np.float64)
        centroids = np.asarray(centroids_reconstruction, dtype=np.float64)
        if lower.size == 0:
            lower = np.zeros((0, 3), dtype=np.float64)
        if upper.size == 0:
            upper = np.zeros((0, 3), dtype=np.float64)
        if centroids.size == 0:
            centroids = np.zeros((0, 3), dtype=np.float64)
        if lower.ndim != 2 or lower.shape[1:] != (3,):
            raise ValueError("bounds_min_reconstruction must be a finite (N, 3) array.")
        if upper.shape != lower.shape or centroids.shape != lower.shape:
            raise ValueError("Semantic bounds and centroids must share shape (N, 3).")
        if not np.isfinite(lower).all() or not np.isfinite(upper).all() or not np.isfinite(centroids).all():
            raise ValueError("Semantic bounds and centroids must be finite.")
        if np.any(upper < lower):
            raise ValueError("Semantic bounds must be ordered.")
        if np.any(centroids < lower - 1e-8) or np.any(centroids > upper + 1e-8):
            raise ValueError("Semantic centroids must lie inside their bounds.")
        ids = tuple(str(value).strip() for value in object_ids)
        if len(ids) != lower.shape[0] or any(not value for value in ids) or len(set(ids)) != len(ids):
            raise ValueError("object_ids must match semantic objects and remain unique.")
        if lower.shape[0] > self.MAX_SEMANTIC_OBJECT_DISPLAY:
            raise ValueError("Semantic object presentation exceeds the bounded display limit.")

        from app.offline_reconstruction_semantics import (
            semantic_bounds_segments_in_viewer,
            semantic_centroid_in_viewer,
        )

        segments = [
            semantic_bounds_segments_in_viewer(lower[index], upper[index])
            for index in range(lower.shape[0])
        ]
        combined = (
            np.ascontiguousarray(np.vstack(segments), dtype=np.float32)
            if segments
            else np.zeros((0, 3), dtype=np.float32)
        )
        center_viewer = np.asarray(
            [semantic_centroid_in_viewer(value) for value in centroids],
            dtype=np.float32,
        ).reshape((-1, 3))
        self._semantic_object_boxes.set_data(pos=combined, connect="segments")
        self._semantic_object_centers.set_data(
            pos=center_viewer,
            face_color=(0.64, 0.46, 0.94, 0.76),
            size=4.2,
            edge_width=0,
        )
        self._semantic_bounds_min = np.ascontiguousarray(lower)
        self._semantic_bounds_max = np.ascontiguousarray(upper)
        self._semantic_centroids_viewer = np.ascontiguousarray(center_viewer)
        self._semantic_object_ids = ids
        self._semantic_object_total_count = max(
            int(total_count) if total_count is not None else int(lower.shape[0]),
            int(lower.shape[0]),
        )
        self._semantic_objects_available = bool(lower.shape[0])
        self._selected_semantic_object_index = None
        self._refresh_semantic_layer_visibility()
        self._selected_semantic_object_box.visible = False
        self._selected_semantic_object_center.visible = False
        self._refresh_world_axis_scale()
        self._canvas.update()

    def clear_semantic_objects(self, *, update_canvas: bool = True) -> None:
        self.clear_canonical_semantic_objects(update_canvas=False)
        self._semantic_object_boxes.set_data(
            pos=np.zeros((0, 3), dtype=np.float32), connect="segments"
        )
        self._semantic_object_centers.set_data(pos=np.zeros((0, 3), dtype=np.float32))
        self._selected_semantic_object_box.set_data(
            pos=np.zeros((0, 3), dtype=np.float32), connect="segments"
        )
        self._selected_semantic_object_center.set_data(pos=np.zeros((0, 3), dtype=np.float32))
        self._semantic_object_boxes.visible = False
        self._semantic_object_centers.visible = False
        self._selected_semantic_object_box.visible = False
        self._selected_semantic_object_center.visible = False
        self._semantic_objects_available = False
        self._semantic_bounds_min = np.zeros((0, 3), dtype=np.float64)
        self._semantic_bounds_max = np.zeros((0, 3), dtype=np.float64)
        self._semantic_centroids_viewer = np.zeros((0, 3), dtype=np.float32)
        self._semantic_object_ids = ()
        self._semantic_object_total_count = 0
        self._selected_semantic_object_index = None
        if update_canvas:
            self._canvas.update()

    def set_semantic_objects_visible(self, visible: bool) -> None:
        """Master semantic-context visibility used by the legacy quick toolbar.

        Phase 9.3F keeps this as a compatibility master gate and adds finer
        presentation-only child layers for support, labels, selection and
        inspection evidence.
        """

        self._semantic_objects_visible = bool(visible)
        self._refresh_semantic_layer_visibility()
        self._canvas.update()

    def set_semantic_support_visible(self, visible: bool) -> None:
        self._semantic_support_layer_visible = bool(visible)
        self._refresh_semantic_layer_visibility()
        self._canvas.update()

    def set_canonical_semantic_labels_visible(self, visible: bool) -> None:
        self._canonical_semantic_labels_layer_visible = bool(visible)
        self._refresh_semantic_layer_visibility()
        self._canvas.update()

    def set_canonical_semantic_selection_visible(self, visible: bool) -> None:
        self._canonical_semantic_selection_layer_visible = bool(visible)
        self._refresh_semantic_layer_visibility()
        self._canvas.update()

    def set_canonical_inspection_overlay_visible(self, visible: bool) -> None:
        self._canonical_inspection_overlay_layer_visible = bool(visible)
        self._refresh_canonical_inspection_overlay_visibility()
        self._refresh_canonical_registered_dimension_visibility()
        self._canvas.update()

    def _refresh_semantic_layer_visibility(self) -> None:
        master = bool(self._semantic_objects_visible and self._semantic_objects_available)
        support = bool(master and self._semantic_support_layer_visible)
        selected = self._selected_semantic_object_index is not None
        self._semantic_object_boxes.visible = support
        self._semantic_object_centers.visible = support
        self._selected_semantic_object_box.visible = bool(support and selected)
        self._selected_semantic_object_center.visible = bool(support and selected)
        self._canonical_semantic_labels.visible = bool(
            master
            and self._canonical_semantic_labels_layer_visible
            and self._canonical_semantic_loaded
            and self._canonical_semantic_label_display_count
        )
        self._selected_canonical_semantic_label.visible = bool(
            master
            and selected
            and self._canonical_semantic_labels_layer_visible
            and self._canonical_semantic_loaded
        )
        self._selected_canonical_semantic_mesh.visible = bool(
            master
            and selected
            and self._canonical_semantic_selection_layer_visible
            and self._canonical_semantic_loaded
            and self._selected_canonical_semantic_surface_loaded
        )
        self._refresh_canonical_inspection_overlay_visibility()
        self._refresh_canonical_registered_dimension_visibility()

    def select_semantic_object(self, object_index: int) -> None:
        index = int(object_index)
        if not self._semantic_objects_available:
            raise ValueError("No semantic object presentation is loaded.")
        if index < 0 or index >= self._semantic_bounds_min.shape[0]:
            raise IndexError("Semantic object index is out of range.")
        self._selected_semantic_object_index = index
        self._refresh_selected_semantic_object()
        self._refresh_canonical_inspection_overlay_visibility()
        self._canvas.update()

    def clear_semantic_object_selection(self) -> None:
        self._selected_semantic_object_index = None
        self._selected_semantic_object_box.set_data(
            pos=np.zeros((0, 3), dtype=np.float32), connect="segments"
        )
        self._selected_semantic_object_center.set_data(pos=np.zeros((0, 3), dtype=np.float32))
        self._selected_semantic_object_box.visible = False
        self._selected_semantic_object_center.visible = False
        self._selected_canonical_semantic_label.visible = False
        self._selected_canonical_semantic_label.text = ""
        self._clear_selected_canonical_semantic_mesh()
        self.clear_canonical_inspection_overlay(update_canvas=False)
        self._canvas.update()

    def _refresh_selected_semantic_object(self) -> None:
        index = self._selected_semantic_object_index
        if index is None:
            return
        if index < 0 or index >= self._semantic_bounds_min.shape[0]:
            self.clear_semantic_object_selection()
            return
        from app.offline_reconstruction_semantics import semantic_bounds_segments_in_viewer

        segments = semantic_bounds_segments_in_viewer(
            self._semantic_bounds_min[index], self._semantic_bounds_max[index]
        )
        center = self._semantic_centroids_viewer[index : index + 1]
        self._selected_semantic_object_box.set_data(pos=segments, connect="segments")
        self._selected_semantic_object_center.set_data(
            pos=center,
            face_color=(1.0, 0.80, 0.22, 1.0),
            size=7.0,
            edge_width=0,
        )
        visible = bool(
            self._semantic_objects_visible
            and self._semantic_objects_available
            and self._semantic_support_layer_visible
        )
        self._selected_semantic_object_box.visible = visible
        self._selected_semantic_object_center.visible = visible
        self._refresh_selected_canonical_semantic_object()

    def focus_selected_semantic_object(self) -> None:
        index = self._selected_semantic_object_index
        if index is None or index < 0 or index >= self._semantic_centroids_viewer.shape[0]:
            self.fit_scene()
            return
        center = self._semantic_centroids_viewer[index]
        span = self._semantic_bounds_max[index] - self._semantic_bounds_min[index]
        radius = max(float(np.linalg.norm(span)) * 0.5, 0.08)
        self._camera.center = tuple(float(value) for value in center)
        self._camera.distance = max(radius * 3.2, 0.7)
        self._camera.view_changed()

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
        from app.reconstruction_visualization import adaptive_camera_frustum_depth

        depth = adaptive_camera_frustum_depth(self._world_axis_length)
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
        self._refresh_canonical_camera_layer_visibility()
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
        if self._canonical_camera_positions.size:
            groups.append(np.asarray(self._canonical_camera_positions, dtype=np.float64))
        if (
            self._canonical_mesh_loaded
            and self._canonical_mesh_bounds_min is not None
            and self._canonical_mesh_bounds_max is not None
        ):
            lower = np.asarray(self._canonical_mesh_bounds_min, dtype=np.float64)
            upper = np.asarray(self._canonical_mesh_bounds_max, dtype=np.float64)
            groups.append(
                np.asarray(
                    [
                        [x, y, z]
                        for x in (lower[0], upper[0])
                        for y in (lower[1], upper[1])
                        for z in (lower[2], upper[2])
                    ],
                    dtype=np.float64,
                )
            )
        if self._semantic_centroids_viewer.size:
            groups.append(np.asarray(self._semantic_centroids_viewer, dtype=np.float64))
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
        if self._canonical_camera_loaded and self._canonical_camera_matrices.shape[0]:
            from app.reconstruction_visualization import (
                adaptive_camera_frustum_depth,
                keyframe_display_indices,
                keyframe_frustum_segments_in_viewer,
            )
            indices = keyframe_display_indices(
                self._canonical_camera_matrices.shape[0],
                max_keyframes=self.MAX_CANONICAL_CAMERA_FRUSTUMS,
            )
            segments = keyframe_frustum_segments_in_viewer(
                self._canonical_camera_matrices[indices],
                depth=adaptive_camera_frustum_depth(self._world_axis_length),
            )
            self._canonical_camera_frustums.set_data(pos=segments, connect="segments")

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
        if self._canonical_camera_positions.size:
            groups.append(np.asarray(self._canonical_camera_positions, dtype=np.float64))
        if (
            self._canonical_mesh_loaded
            and self._canonical_mesh_bounds_min is not None
            and self._canonical_mesh_bounds_max is not None
        ):
            lower = np.asarray(self._canonical_mesh_bounds_min, dtype=np.float64)
            upper = np.asarray(self._canonical_mesh_bounds_max, dtype=np.float64)
            groups.append(
                np.asarray(
                    [
                        [x, y, z]
                        for x in (lower[0], upper[0])
                        for y in (lower[1], upper[1])
                        for z in (lower[2], upper[2])
                    ],
                    dtype=np.float64,
                )
            )
        if self._semantic_centroids_viewer.size:
            groups.append(np.asarray(self._semantic_centroids_viewer, dtype=np.float64))
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
            "measure": Qt.CursorShape.CrossCursor,
        }
        self.native.setCursor(cursors[mode])

    def _pick_canonical_surface_point(self, click_xy) -> np.ndarray | None:
        if not self._canonical_mesh_loaded or self._canonical_mesh_vertices_viewer.size == 0:
            return None
        click = np.asarray(click_xy, dtype=np.float64)
        if click.shape != (2,) or not np.isfinite(click).all():
            return None
        try:
            transform = self._canonical_mesh.transforms.get_transform("visual", "canvas")
            mapped = np.asarray(transform.map(self._canonical_mesh_vertices_viewer), dtype=np.float64)
            from app.reconstruction_interactive_measurement import pick_projected_surface_point
            from app.reconstruction_visualization import viewer_point_to_reconstruction

            picked = pick_projected_surface_point(
                vertices_viewer=self._canonical_mesh_vertices_viewer,
                faces=self._canonical_mesh_faces,
                mapped_homogeneous=mapped,
                click_xy=click,
            )
            if picked is None:
                return None
            return viewer_point_to_reconstruction(np.asarray(picked.point_viewer, dtype=np.float64))
        except (AttributeError, TypeError, ValueError, RuntimeError):
            # Picking is a convenience interaction. A transform/presentation
            # failure must fail closed rather than invent a surface point.
            return None

    def set_interactive_measurement_overlay(
        self,
        points_reconstruction,
        *,
        update_canvas: bool = True,
    ) -> None:
        from app.reconstruction_visualization import reconstruction_points_to_viewer

        points = np.asarray(points_reconstruction, dtype=np.float64)
        if points.size == 0:
            self.clear_interactive_measurement_overlay(update_canvas=update_canvas)
            return
        if points.ndim != 2 or points.shape[1:] != (3,) or points.shape[0] not in {1, 2}:
            raise ValueError("Interactive measurement overlay requires one or two 3D points.")
        if not np.isfinite(points).all():
            raise ValueError("Interactive measurement overlay points must be finite.")
        viewer = reconstruction_points_to_viewer(points).astype(np.float32)
        self._interactive_measure_markers.set_data(
            pos=viewer,
            face_color=(0.23, 0.78, 0.85, 1.0),
            edge_color=(0.92, 0.99, 1.0, 1.0),
            size=9.0,
            edge_width=1.2,
        )
        self._interactive_measure_markers.visible = True
        if viewer.shape[0] == 2:
            self._interactive_measure_line.set_data(pos=viewer, connect="strip")
            self._interactive_measure_line.visible = True
        else:
            self._interactive_measure_line.visible = False
        if update_canvas:
            self._canvas.update()

    def clear_interactive_measurement_overlay(self, *, update_canvas: bool = True) -> None:
        if hasattr(self, "_interactive_measure_line"):
            self._interactive_measure_line.visible = False
            self._interactive_measure_markers.visible = False
        if update_canvas:
            self._canvas.update()

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

    def _on_mouse_press(self, event) -> None:
        if self.interaction_mode == "measure":
            button = getattr(event, "button", None)
            if button == 2:
                self.measurement_cancel_requested.emit()
                return
            if button not in {None, 1}:
                return
            point = self._pick_canonical_surface_point(getattr(event, "pos", ()))
            if point is not None:
                self.measurement_point_picked.emit(point)
            return
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
    def offline_sparse_envelope_visible(self) -> bool:
        return bool(self._offline_sparse_envelope.visible)

    @property
    def offline_sparse_envelope_available(self) -> bool:
        return bool(self._offline_sparse_envelope_available)

    @property
    def offline_sparse_envelope_bounds(self) -> tuple[np.ndarray, np.ndarray] | None:
        if self._offline_sparse_envelope_min is None or self._offline_sparse_envelope_max is None:
            return None
        return (
            np.array(self._offline_sparse_envelope_min, copy=True),
            np.array(self._offline_sparse_envelope_max, copy=True),
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
    def canonical_mesh_layer_visible(self) -> bool:
        return bool(self._canonical_mesh_layer_visible)

    @property
    def canonical_camera_frustums_layer_visible(self) -> bool:
        return bool(self._canonical_camera_frustums_layer_visible)

    @property
    def canonical_camera_trajectory_layer_visible(self) -> bool:
        return bool(self._canonical_camera_trajectory_layer_visible)

    @property
    def semantic_support_layer_visible(self) -> bool:
        return bool(self._semantic_support_layer_visible)

    @property
    def canonical_semantic_labels_layer_visible(self) -> bool:
        return bool(self._canonical_semantic_labels_layer_visible)

    @property
    def canonical_semantic_selection_layer_visible(self) -> bool:
        return bool(self._canonical_semantic_selection_layer_visible)

    @property
    def canonical_inspection_overlay_layer_visible(self) -> bool:
        return bool(self._canonical_inspection_overlay_layer_visible)

    @property
    def canonical_mesh_loaded(self) -> bool:
        return bool(self._canonical_mesh_loaded)

    @property
    def canonical_mesh_visible(self) -> bool:
        return bool(self._canonical_mesh_loaded and self._canonical_mesh.visible)

    @property
    def canonical_mesh_vertex_count(self) -> int:
        return int(self._canonical_mesh_vertex_count)

    @property
    def canonical_mesh_face_count(self) -> int:
        return int(self._canonical_mesh_face_count)

    @property
    def canonical_mesh_scene_id(self) -> str:
        return self._canonical_mesh_scene_id

    @property
    def canonical_mesh_artifact_id(self) -> str:
        return self._canonical_mesh_artifact_id

    @property
    def canonical_mesh_source_path(self) -> str:
        return self._canonical_mesh_source_path

    @property
    def canonical_mesh_bounds(self) -> tuple[np.ndarray, np.ndarray] | None:
        if self._canonical_mesh_bounds_min is None or self._canonical_mesh_bounds_max is None:
            return None
        return (
            np.asarray(self._canonical_mesh_bounds_min, dtype=np.float32).copy(),
            np.asarray(self._canonical_mesh_bounds_max, dtype=np.float32).copy(),
        )

    @property
    def canonical_camera_loaded(self) -> bool:
        return bool(self._canonical_camera_loaded)

    @property
    def canonical_camera_scene_id(self) -> str:
        return self._canonical_camera_scene_id

    @property
    def canonical_camera_artifact_id(self) -> str:
        return self._canonical_camera_artifact_id

    @property
    def canonical_camera_provider_id(self) -> str:
        return self._canonical_camera_provider_id

    @property
    def canonical_camera_ids(self) -> tuple[str, ...]:
        return self._canonical_camera_ids

    @property
    def canonical_camera_count(self) -> int:
        return int(self._canonical_camera_total_count)

    @property
    def canonical_camera_persisted_count(self) -> int:
        return int(self._canonical_camera_matrices.shape[0])

    @property
    def canonical_camera_frustum_display_count(self) -> int:
        return int(self._canonical_camera_frustum_display_count)

    @property
    def canonical_camera_trajectory_display_count(self) -> int:
        return int(self._canonical_camera_trajectory_display_count)

    @property
    def canonical_camera_frustums_visible(self) -> bool:
        return bool(self._canonical_camera_loaded and self._canonical_camera_frustums.visible)

    @property
    def canonical_camera_trajectory_visible(self) -> bool:
        return bool(self._canonical_camera_loaded and self._canonical_camera_trajectory.visible)

    @property
    def canonical_semantic_loaded(self) -> bool:
        return bool(self._canonical_semantic_loaded)

    @property
    def canonical_semantic_scene_id(self) -> str:
        return self._canonical_semantic_scene_id

    @property
    def canonical_semantic_artifact_id(self) -> str:
        return self._canonical_semantic_artifact_id

    @property
    def canonical_semantic_object_ids(self) -> tuple[str, ...]:
        return self._canonical_semantic_object_ids

    @property
    def canonical_semantic_surface_binding_count(self) -> int:
        return sum(bool(faces.shape[0]) for faces in self._canonical_semantic_surface_faces)

    @property
    def canonical_semantic_label_display_count(self) -> int:
        return int(self._canonical_semantic_label_display_count)

    @property
    def canonical_semantic_labels_visible(self) -> bool:
        return bool(self._canonical_semantic_loaded and self._canonical_semantic_labels.visible)

    @property
    def selected_canonical_semantic_surface_visible(self) -> bool:
        return bool(
            self._selected_canonical_semantic_surface_loaded
            and self._selected_canonical_semantic_mesh.visible
        )

    @property
    def selected_canonical_semantic_label_visible(self) -> bool:
        return bool(self._selected_canonical_semantic_label.visible)

    @property
    def canonical_inspection_overlay_visible(self) -> bool:
        return bool(self._canonical_inspection_overlay.visible)

    @property
    def canonical_inspection_overlay_text(self) -> str:
        return self._canonical_inspection_overlay_text

    @property
    def canonical_inspection_overlay_object_id(self) -> str:
        return self._canonical_inspection_overlay_object_id

    @property
    def canonical_registered_dimension_count(self) -> int:
        return int(self._canonical_registered_dimension_count)

    @property
    def canonical_registered_dimensions_visible(self) -> bool:
        return bool(self._canonical_registered_dimensions.visible)

    @property
    def canonical_registered_dimension_object_id(self) -> str:
        return self._canonical_registered_dimension_object_id

    @property
    def canonical_registered_dimension_text(self) -> tuple[str, ...]:
        return self._canonical_registered_dimension_text

    @property
    def semantic_objects_available(self) -> bool:
        return bool(self._semantic_objects_available)

    @property
    def semantic_objects_visible(self) -> bool:
        return bool(self._semantic_objects_visible and self._semantic_objects_available)

    @property
    def semantic_object_count(self) -> int:
        return int(self._semantic_object_total_count)

    @property
    def semantic_object_display_count(self) -> int:
        return int(self._semantic_bounds_min.shape[0])

    @property
    def semantic_object_ids(self) -> tuple[str, ...]:
        return self._semantic_object_ids

    @property
    def selected_semantic_object_index(self) -> int | None:
        return self._selected_semantic_object_index

    @property
    def selected_semantic_object_id(self) -> str | None:
        index = self._selected_semantic_object_index
        if index is None or index < 0 or index >= len(self._semantic_object_ids):
            return None
        return self._semantic_object_ids[index]

    @property
    def selected_semantic_object_visible(self) -> bool:
        return bool(
            self._semantic_objects_visible
            and self._semantic_objects_available
            and self._selected_semantic_object_index is not None
            and self._selected_semantic_object_box.visible
        )

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
        self.clear_canonical_camera_trajectory(update_canvas=False)
        self.clear_canonical_semantic_objects(update_canvas=False)
        self.clear_canonical_mesh(update_canvas=False)
        self._canvas.close()

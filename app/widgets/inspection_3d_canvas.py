"""Interactive VisPy object-point and fitted-box view for Measure & Inspect."""
from __future__ import annotations

import math
from typing import Callable, Mapping

import numpy as np
from PyQt6.QtCore import QObject, QPointF, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import QGridLayout, QSizePolicy, QWidget
from vispy import scene

from core.geometry.orientation_overlay import project_turntable_axes
from core.inspection.contracts import InspectionBoxMethod, InspectionObjectSummary, InspectionPointMode

_BOX_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
    (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
)
_BOX_COLORS = {
    "axis_aligned": (0.75, 0.77, 0.80, 0.9),
    "camera_aligned": (0.23, 0.78, 0.85, 0.95),
    "oriented": (0.91, 0.66, 0.24, 0.95),
}
_AXIS_COLORS = {
    "x": QColor("#e0293e"),
    "y": QColor("#2ecc71"),
    "z": QColor("#3a7de8"),
}


class _OrientationGizmo(QWidget):
    """Compact camera-aware XYZ triad anchored in viewport screen space."""

    def __init__(
        self,
        camera_state: Callable[[], tuple[float, float, float]],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._camera_state = camera_state
        self._last_state: tuple[float, float, float] | None = None
        self.setFixedSize(88, 72)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAutoFillBackground(False)

        # TurntableCamera does not expose one stable public Qt signal across all
        # supported VisPy releases, so a very small visible-only poll keeps the
        # overlay synchronized without coupling to private event internals.
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._refresh_from_camera)
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().showEvent(event)
        if not self._timer.isActive():
            self._timer.start()
        self._refresh_from_camera(force=True)

    def hideEvent(self, event) -> None:  # noqa: N802 - Qt override
        self._timer.stop()
        super().hideEvent(event)

    def _refresh_from_camera(self, *, force: bool = False) -> None:
        try:
            state = tuple(round(float(value), 4) for value in self._camera_state())
        except (AttributeError, TypeError, ValueError):
            state = (45.0, 25.0, 0.0)
        if force or state != self._last_state:
            self._last_state = state
            self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            azimuth, elevation, roll = self._last_state or self._camera_state()
            projected = project_turntable_axes(azimuth, elevation, roll)

            origin = QPointF(34.0, 54.0)
            scale = 29.0
            font = QFont()
            font.setPointSize(7)
            font.setBold(True)
            painter.setFont(font)

            # Draw axes pointing away from the camera first so nearer arrows
            # remain visually dominant where projections overlap.
            for name in sorted(projected, key=lambda axis: projected[axis][2]):
                screen_x, screen_y, _depth = projected[name]
                length = math.hypot(screen_x, screen_y)
                if length < 0.20:
                    # Preserve orientation for axes nearly aligned with the
                    # view direction while retaining a small visible cue.
                    if length < 1e-9:
                        screen_x, screen_y = 0.0, -1.0
                        length = 1.0
                    multiplier = 0.20 / length
                    screen_x *= multiplier
                    screen_y *= multiplier

                end = QPointF(
                    origin.x() + (screen_x * scale),
                    origin.y() + (screen_y * scale),
                )
                color = _AXIS_COLORS[name]
                self._draw_arrow(painter, origin, end, color)

                direction_x = end.x() - origin.x()
                direction_y = end.y() - origin.y()
                direction_norm = max(math.hypot(direction_x, direction_y), 1.0)
                label_position = QPointF(
                    end.x() + (direction_x / direction_norm) * 6.0 - 2.5,
                    end.y() + (direction_y / direction_norm) * 6.0 + 3.0,
                )
                painter.setPen(QPen(color))
                painter.drawText(label_position, name.upper())

            # Small neutral origin dot improves legibility without adding an
            # opaque panel around the overlay.
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#d9dde4"))
            painter.drawEllipse(origin, 2.2, 2.2)
        finally:
            if painter.isActive():
                painter.end()

    @staticmethod
    def _draw_arrow(painter: QPainter, start: QPointF, end: QPointF, color: QColor) -> None:
        pen = QPen(color)
        pen.setWidthF(1.8)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)

        angle = math.atan2(end.y() - start.y(), end.x() - start.x())
        head_length = 6.0
        head_angle = math.radians(24.0)
        shaft_end = QPointF(
            end.x() - (head_length * 0.6) * math.cos(angle),
            end.y() - (head_length * 0.6) * math.sin(angle),
        )
        painter.drawLine(start, shaft_end)

        left = QPointF(
            end.x() - head_length * math.cos(angle - head_angle),
            end.y() - head_length * math.sin(angle - head_angle),
        )
        right = QPointF(
            end.x() - head_length * math.cos(angle + head_angle),
            end.y() - head_length * math.sin(angle + head_angle),
        )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawPolygon(QPolygonF([end, left, right]))


class Inspection3DCanvas(QObject):
    """One persistent SceneCanvas; data changes update GPU buffers only."""

    escape_pressed = pyqtSignal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._canvas = scene.SceneCanvas(keys=None, bgcolor="#090b0e", show=False, vsync=True)
        self._view = self._canvas.central_widget.add_view()
        self._camera = scene.cameras.TurntableCamera(fov=45.0, distance=5.0)
        self._view.camera = self._camera
        self._markers = scene.visuals.Markers(parent=self._view.scene)
        self._lines: dict[str, scene.visuals.Line] = {
            name: scene.visuals.Line(parent=self._view.scene, method="gl", width=2.0)
            for name in _BOX_COLORS
        }

        # The orientation triad is intentionally not a scene visual.  It is a
        # fixed screen-space overlay, so pan/zoom/fit never moves it or includes
        # it in scene bounds.
        self._native_host = QWidget()
        self._native_host.setObjectName("inspection3DViewportHost")
        self._native_host.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        host_layout = QGridLayout(self._native_host)
        host_layout.setContentsMargins(0, 0, 0, 0)
        host_layout.setSpacing(0)
        host_layout.addWidget(self._canvas.native, 0, 0)
        self._orientation_gizmo = _OrientationGizmo(self._camera_orientation, self._native_host)
        host_layout.addWidget(
            self._orientation_gizmo,
            0,
            0,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom,
        )
        host_layout.setRowStretch(0, 1)
        host_layout.setColumnStretch(0, 1)

        self._has_points = False
        self._world_points = np.zeros((0, 3), dtype=np.float32)
        self._colors = np.zeros((0, 4), dtype=np.float32)
        self._point_size = 3.0
        self._show_boxes = True
        self._box_method = InspectionBoxMethod.AUTO
        self._last_object: InspectionObjectSummary | None = None
        self._last_arrays: Mapping[str, np.ndarray] = {}
        self._point_mode = InspectionPointMode.FILTERED

    @property
    def native(self):
        return self._native_host

    def close(self) -> None:
        self._orientation_gizmo.stop()
        self._canvas.close()
        self._native_host.close()

    def clear(self) -> None:
        self._markers.set_data(pos=np.zeros((0, 3), dtype=np.float32))
        for visual in self._lines.values():
            visual.set_data(pos=np.zeros((0, 3), dtype=np.float32))
        self._has_points = False
        self._world_points = np.zeros((0, 3), dtype=np.float32)

    def set_object(
        self,
        obj: InspectionObjectSummary | None,
        arrays: Mapping[str, np.ndarray],
        *,
        point_mode: InspectionPointMode,
        box_method: InspectionBoxMethod,
        show_boxes: bool,
    ) -> None:
        self._last_object = obj
        self._last_arrays = arrays
        self._point_mode = point_mode
        self._box_method = box_method
        self._show_boxes = bool(show_boxes)
        if obj is None:
            self.clear()
            return
        points_key = obj.raw_points_key if point_mode is InspectionPointMode.RAW else obj.filtered_points_key
        colors_key = obj.raw_colors_key if point_mode is InspectionPointMode.RAW else obj.filtered_colors_key
        points = np.asarray(arrays.get(points_key, np.zeros((0, 3))), dtype=np.float32)
        colors = np.asarray(arrays.get(colors_key, np.zeros((len(points), 3))), dtype=np.uint8)
        if colors.shape != (len(points), 3):
            colors = np.tile(np.array([[46, 204, 113]], dtype=np.uint8), (len(points), 1))
        self._set_points(points, colors)
        self._set_boxes(obj, box_method, show_boxes)
        self.fit_scene()

    def refresh_style(self, *, point_size: float | None = None, show_boxes: bool | None = None) -> None:
        if point_size is not None:
            self._point_size = max(1.0, float(point_size))
            if self._has_points:
                self._markers.set_data(
                    pos=self._world_points,
                    face_color=self._colors,
                    size=self._point_size,
                    edge_width=0,
                )
        if show_boxes is not None and self._last_object is not None:
            self._show_boxes = bool(show_boxes)
            self._set_boxes(self._last_object, self._box_method, self._show_boxes)
        self._canvas.update()

    def reset_view(self) -> None:
        self._camera.azimuth = 45.0
        self._camera.elevation = 25.0
        self.fit_scene()
        self._orientation_gizmo._refresh_from_camera(force=True)

    def fit_scene(self) -> None:
        if not self._has_points or not len(self._world_points):
            self._camera.center = (0.0, 0.0, 0.0)
            self._camera.distance = 5.0
            return
        lo = self._world_points.min(axis=0)
        hi = self._world_points.max(axis=0)
        center = (lo + hi) * 0.5
        radius = max(float(np.linalg.norm(hi - lo) * 0.5), 0.05)
        self._camera.center = tuple(float(v) for v in center)
        self._camera.distance = radius / max(math.tan(math.radians(self._camera.fov / 2.0)), 1e-6) * 1.35

    def _camera_orientation(self) -> tuple[float, float, float]:
        return (
            float(getattr(self._camera, "azimuth", 45.0)),
            float(getattr(self._camera, "elevation", 25.0)),
            float(getattr(self._camera, "roll", 0.0)),
        )

    @staticmethod
    def _to_world(points: np.ndarray) -> np.ndarray:
        world = np.empty_like(points, dtype=np.float32)
        world[:, 0] = points[:, 0]
        world[:, 1] = points[:, 2]
        world[:, 2] = -points[:, 1]
        return world

    def _set_points(self, points: np.ndarray, colors: np.ndarray) -> None:
        if points.ndim != 2 or points.shape[1:] != (3,) or not len(points):
            self.clear()
            return
        self._world_points = np.ascontiguousarray(self._to_world(points))
        rgba = np.column_stack((colors.astype(np.float32) / 255.0, np.ones(len(colors), dtype=np.float32)))
        self._colors = np.ascontiguousarray(rgba)
        self._markers.set_data(
            pos=self._world_points,
            face_color=self._colors,
            size=self._point_size,
            edge_width=0,
        )
        self._has_points = True

    def _set_boxes(self, obj: InspectionObjectSummary, method: InspectionBoxMethod, show: bool) -> None:
        payloads = dict(obj.boxes)
        preferred_name = next(
            (name for name, payload in obj.boxes if payload.get("box_id") == obj.preferred_box_id),
            "",
        )
        for name, visual in self._lines.items():
            payload = payloads.get(name)
            visible = show and payload is not None and (
                (method is InspectionBoxMethod.AUTO and name == preferred_name)
                or (method is not InspectionBoxMethod.AUTO and name == method.value)
            )
            if not visible:
                visual.set_data(pos=np.zeros((0, 3), dtype=np.float32))
                continue
            corners = np.asarray(payload["corners_xyz_m"], dtype=np.float32)
            vertices = []
            for first, second in _BOX_EDGES:
                vertices.extend((corners[first], corners[second]))
            world = self._to_world(np.asarray(vertices, dtype=np.float32))
            color = (0.30, 1.0, 0.30, 1.0) if payload.get("box_id") == obj.preferred_box_id else _BOX_COLORS[name]
            visual.set_data(pos=world, color=color, connect="segments", width=3.0 if payload.get("box_id") == obj.preferred_box_id else 2.0)

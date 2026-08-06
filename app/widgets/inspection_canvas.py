"""Interactive aligned-RGB inspection canvas with safe manual mask editing."""
from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QWidget

from core.inspection.contracts import InspectionBoxMethod, InspectionObjectSummary, MaskEditMode

_PALETTE = (
    QColor(46, 204, 113), QColor(58, 198, 216), QColor(232, 168, 60),
    QColor(224, 41, 62), QColor(164, 140, 232), QColor(255, 128, 64),
)
_BOX_COLORS = {
    "axis_aligned": QColor(190, 194, 202),
    "camera_aligned": QColor(58, 198, 216),
    "oriented": QColor(232, 168, 60),
}


def _qimage_from_rgb(array: np.ndarray) -> QImage:
    rgb = np.ascontiguousarray(array, dtype=np.uint8)
    height, width = rgb.shape[:2]
    image = QImage(rgb.data, width, height, width * 3, QImage.Format.Format_RGB888)
    return image.copy()


class InspectionImageCanvas(QWidget):
    """Scalable image surface with object hit testing and include/exclude brush."""

    object_clicked = pyqtSignal(str)
    mask_edited = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("inspectionImageCanvas")
        self.setMinimumSize(480, 320)
        self.setMouseTracking(True)
        self._rgb: np.ndarray | None = None
        self._pixmap = QPixmap()
        self._objects: tuple[InspectionObjectSummary, ...] = ()
        self._arrays: Mapping[str, np.ndarray] = {}
        self._selected_id = ""
        self._editable_mask: np.ndarray | None = None
        self._show_masks = True
        self._show_boxes = True
        self._show_labels = True
        self._show_uncertainty = True
        self._box_method = InspectionBoxMethod.AUTO
        self._edit_mode = MaskEditMode.SELECT
        self._brush_radius = 16
        self._editing_enabled = False
        self._painting = False
        self._image_rect = QRectF()
        self._hover_image_point: tuple[int, int] | None = None
        self.setCursor(Qt.CursorShape.ArrowCursor)

    def clear(self) -> None:
        self._rgb = None
        self._pixmap = QPixmap()
        self._objects = ()
        self._arrays = {}
        self._selected_id = ""
        self._editable_mask = None
        self.update()

    def set_bundle(
        self,
        rgb: np.ndarray,
        objects: Sequence[InspectionObjectSummary],
        arrays: Mapping[str, np.ndarray],
    ) -> None:
        source = np.asarray(rgb, dtype=np.uint8)
        if source.ndim != 3 or source.shape[2] != 3:
            raise ValueError("Inspection RGB must be HxWx3.")
        self._rgb = np.array(source, copy=True)
        self._pixmap = QPixmap.fromImage(_qimage_from_rgb(self._rgb))
        self._objects = tuple(objects)
        self._arrays = arrays
        if self._objects:
            self.set_selected_object(self._objects[0].observation_id)
        else:
            self._selected_id = ""
            self._editable_mask = None
        self.update()

    def set_selected_object(self, observation_id: str) -> None:
        self._selected_id = str(observation_id)
        obj = self.selected_object
        self._editable_mask = (
            np.array(self._arrays[obj.mask_key], dtype=bool, copy=True) if obj is not None else None
        )
        self.update()

    @property
    def selected_object(self) -> InspectionObjectSummary | None:
        return next((value for value in self._objects if value.observation_id == self._selected_id), None)

    @property
    def editable_mask(self) -> np.ndarray | None:
        return None if self._editable_mask is None else np.array(self._editable_mask, copy=True)

    def set_editable_mask(self, mask: np.ndarray) -> None:
        if self._rgb is None:
            return
        value = np.asarray(mask, dtype=bool)
        if value.shape != self._rgb.shape[:2]:
            raise ValueError("Editable mask must match the aligned RGB dimensions.")
        self._editable_mask = np.array(value, copy=True)
        self.update()

    def reset_selected_mask(self) -> None:
        obj = self.selected_object
        if obj is None:
            return
        self._editable_mask = np.array(self._arrays[obj.mask_key], dtype=bool, copy=True)
        self.update()

    def set_overlay_options(self, *, masks: bool, boxes: bool, labels: bool, uncertainty: bool) -> None:
        self._show_masks = bool(masks)
        self._show_boxes = bool(boxes)
        self._show_labels = bool(labels)
        self._show_uncertainty = bool(uncertainty)
        self.update()

    def set_box_method(self, method: InspectionBoxMethod) -> None:
        self._box_method = method
        self.update()

    def set_edit_mode(self, mode: MaskEditMode) -> None:
        self._edit_mode = mode
        self.setCursor(
            Qt.CursorShape.CrossCursor
            if mode in {MaskEditMode.INCLUDE, MaskEditMode.EXCLUDE}
            else Qt.CursorShape.ArrowCursor
        )
        self.update()

    def set_brush_radius(self, radius: int) -> None:
        self._brush_radius = max(2, int(radius))
        self.update()

    def set_editing_enabled(self, enabled: bool) -> None:
        self._editing_enabled = bool(enabled)
        if not enabled and self._edit_mode is not MaskEditMode.SELECT:
            self.set_edit_mode(MaskEditMode.SELECT)

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        if not painter.isActive():
            return
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.fillRect(self.rect(), QColor("#090b0e"))
            if self._rgb is None or self._pixmap.isNull():
                painter.setPen(QColor("#9298a3"))
                painter.drawText(
                    self.rect(),
                    Qt.AlignmentFlag.AlignCenter,
                    "Capture a source frame to begin inspection",
                )
                return
            target = self._fit_rect(self._rgb.shape[1], self._rgb.shape[0])
            self._image_rect = target
            painter.drawPixmap(target.toRect(), self._pixmap)
            self._paint_masks(painter, target)
            if self._show_boxes:
                self._paint_boxes(painter, target)
            if self._show_labels:
                self._paint_labels(painter, target)
            self._paint_selection(painter, target)
            self._paint_brush(painter, target)
        finally:
            # A paint-event exception must never leave Qt's backing-store
            # painter active. This also prevents warning cascades on shutdown.
            if painter.isActive():
                painter.end()

    def _fit_rect(self, image_width: int, image_height: int) -> QRectF:
        margin = 8.0
        available_w = max(1.0, self.width() - margin * 2)
        available_h = max(1.0, self.height() - margin * 2)
        scale = min(available_w / image_width, available_h / image_height)
        width = image_width * scale
        height = image_height * scale
        return QRectF((self.width() - width) / 2.0, (self.height() - height) / 2.0, width, height)

    def _map_point(self, x: float, y: float, target: QRectF) -> QPointF:
        assert self._rgb is not None
        return QPointF(
            target.left() + x * target.width() / self._rgb.shape[1],
            target.top() + y * target.height() / self._rgb.shape[0],
        )

    def _mask_for_object(self, obj: InspectionObjectSummary) -> np.ndarray:
        if obj.observation_id == self._selected_id and self._editable_mask is not None:
            return self._editable_mask
        return np.asarray(self._arrays[obj.mask_key], dtype=bool)

    def _paint_masks(self, painter: QPainter, target: QRectF) -> None:
        if not self._show_masks or self._rgb is None:
            return
        for index, obj in enumerate(self._objects):
            mask = self._mask_for_object(obj)
            if not np.any(mask):
                continue
            rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
            color = _PALETTE[index % len(_PALETTE)]
            rgba[mask] = (color.red(), color.green(), color.blue(), 82 if obj.observation_id != self._selected_id else 112)
            image = QImage(
                np.ascontiguousarray(rgba).data,
                rgba.shape[1], rgba.shape[0], rgba.shape[1] * 4,
                QImage.Format.Format_RGBA8888,
            ).copy()
            painter.drawImage(target, image)

    def _box_names(self, obj: InspectionObjectSummary) -> tuple[str, ...]:
        if self._box_method is InspectionBoxMethod.AUTO:
            preferred = next((name for name, payload in obj.boxes if payload.get("box_id") == obj.preferred_box_id), "")
            return (preferred,) if preferred else tuple(name for name, _ in obj.boxes)
        return (self._box_method.value,)

    def _paint_boxes(self, painter: QPainter, target: QRectF) -> None:
        for index, obj in enumerate(self._objects):
            boxes = dict(obj.boxes)
            for name in self._box_names(obj):
                payload = boxes.get(name)
                if not payload:
                    continue
                preferred = payload.get("box_id") == obj.preferred_box_id
                color = _PALETTE[index % len(_PALETTE)] if preferred else _BOX_COLORS.get(name, QColor("#d0d0d0"))
                pen = QPen(color, 3.0 if obj.observation_id == self._selected_id else 1.5)
                pen.setCosmetic(True)
                painter.setPen(pen)
                for segment in payload.get("projected_segments", ()):
                    p0 = self._map_point(float(segment[0][0]), float(segment[0][1]), target)
                    p1 = self._map_point(float(segment[1][0]), float(segment[1][1]), target)
                    painter.drawLine(p0, p1)

    def _paint_labels(self, painter: QPainter, target: QRectF) -> None:
        if self._rgb is None:
            return
        for index, obj in enumerate(self._objects):
            box = obj.detection_box_xyxy
            if box is None:
                continue
            anchor = self._map_point(box[0], box[1], target)
            label = obj.class_name
            if obj.detection_confidence is not None:
                label += f"  {obj.detection_confidence:.2f}"
            if obj.track_id:
                label += f"  {obj.track_id}"
            if self._show_uncertainty:
                width = next((m for m in obj.measurements if m.get("kind") == "width" and m.get("status") == "available"), None)
                if width:
                    label += f"  W {width['value_si'] * 1000:.0f}±{width['expanded_uncertainty_si'] * 1000:.0f} mm"
            color = _PALETTE[index % len(_PALETTE)]
            metrics = painter.fontMetrics()
            # QFontMetrics returns QRect, while the scaled image coordinates
            # are floating-point. Keep the whole label path in QRectF/QPointF.
            rect = QRectF(metrics.boundingRect(label)).adjusted(-6.0, -4.0, 6.0, 4.0)
            rect.moveBottomLeft(anchor + QPointF(0.0, -4.0))
            if rect.left() < target.left():
                rect.moveLeft(target.left())
            if rect.right() > target.right():
                rect.moveRight(target.right())
            if rect.top() < target.top():
                rect.moveTop(target.top())
            painter.fillRect(rect, QColor(10, 12, 15, 220))
            painter.setPen(color)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, label)

    def _paint_selection(self, painter: QPainter, target: QRectF) -> None:
        obj = self.selected_object
        if obj is None or obj.detection_box_xyxy is None:
            return
        x0, y0, x1, y1 = obj.detection_box_xyxy
        p0, p1 = self._map_point(x0, y0, target), self._map_point(x1, y1, target)
        painter.setPen(QPen(QColor("#ffffff"), 2.0, Qt.PenStyle.DashLine))
        painter.drawRect(QRectF(p0, p1).normalized())

    def _paint_brush(self, painter: QPainter, target: QRectF) -> None:
        if not self._editing_enabled or self._edit_mode is MaskEditMode.SELECT or self._hover_image_point is None or self._rgb is None:
            return
        x, y = self._hover_image_point
        center = self._map_point(x, y, target)
        radius = self._brush_radius * target.width() / self._rgb.shape[1]
        color = QColor("#2ecc71") if self._edit_mode is MaskEditMode.INCLUDE else QColor("#e0293e")
        painter.setPen(QPen(color, 2.0))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(center, radius, radius)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton or self._rgb is None:
            return
        point = self._widget_to_image(event.position())
        if point is None:
            return
        if self._editing_enabled and self._edit_mode in {MaskEditMode.INCLUDE, MaskEditMode.EXCLUDE} and self._editable_mask is not None:
            self._painting = True
            self._apply_brush(*point)
            return
        selected = self._hit_test(*point)
        if selected:
            self.object_clicked.emit(selected)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        point = self._widget_to_image(event.position())
        self._hover_image_point = point
        if self._painting and point is not None:
            self._apply_brush(*point)
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self._painting:
            self._painting = False
            self.mask_edited.emit(self._selected_id)

    def leaveEvent(self, _event) -> None:  # noqa: N802
        self._hover_image_point = None
        self.update()

    def _widget_to_image(self, point: QPointF) -> tuple[int, int] | None:
        if self._rgb is None or not self._image_rect.contains(point):
            return None
        x = int((point.x() - self._image_rect.left()) * self._rgb.shape[1] / self._image_rect.width())
        y = int((point.y() - self._image_rect.top()) * self._rgb.shape[0] / self._image_rect.height())
        return min(max(x, 0), self._rgb.shape[1] - 1), min(max(y, 0), self._rgb.shape[0] - 1)

    def _hit_test(self, x: int, y: int) -> str:
        matches = []
        for obj in self._objects:
            if obj.detection_box_xyxy is None:
                continue
            x0, y0, x1, y1 = obj.detection_box_xyxy
            if x0 <= x < x1 and y0 <= y < y1:
                matches.append(((x1 - x0) * (y1 - y0), obj.observation_id))
        return min(matches)[1] if matches else ""

    def _apply_brush(self, x: int, y: int) -> None:
        if self._editable_mask is None:
            return
        height, width = self._editable_mask.shape
        yy, xx = np.ogrid[:height, :width]
        circle = (xx - x) ** 2 + (yy - y) ** 2 <= self._brush_radius ** 2
        self._editable_mask[circle] = self._edit_mode is MaskEditMode.INCLUDE
        self.update()

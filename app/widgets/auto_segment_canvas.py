"""Interactive mask/prompt canvas for the Phase 9.2 Auto Segment workspace."""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QWidget

from core.perception.semantic_workspace import SemanticReviewStatus, SemanticWorkspace


_PALETTE = (
    QColor("#27c4d8"), QColor("#e1762d"), QColor("#9b6ee8"), QColor("#59c17a"),
    QColor("#e05283"), QColor("#d5b942"), QColor("#4f8be8"), QColor("#77b7a7"),
)


class AutoSegmentCanvas(QWidget):
    object_clicked = pyqtSignal(str)
    prompt_point_clicked = pyqtSignal(int, int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(420, 300)
        self.setMouseTracking(True)
        self._bgr: np.ndarray | None = None
        self._pixmap = QPixmap()
        self._workspace: SemanticWorkspace | None = None
        self._selected_id = ""
        self._image_rect = QRectF()
        self._show_masks = True
        self._show_labels = True
        self._interaction_mode = "select"
        self._positive_points: tuple[tuple[float, float], ...] = ()
        self._negative_points: tuple[tuple[float, float], ...] = ()

    @property
    def interaction_mode(self) -> str:
        return self._interaction_mode

    def set_image(self, image_bgr: np.ndarray | None) -> None:
        if image_bgr is None:
            self._bgr = None
            self._pixmap = QPixmap()
            self.update()
            return
        bgr = np.ascontiguousarray(np.asarray(image_bgr, dtype=np.uint8))
        if bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError("Auto Segment canvas image must be HxWx3 BGR.")
        rgb = np.ascontiguousarray(bgr[..., ::-1])
        image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888).copy()
        self._bgr = bgr.copy()
        self._pixmap = QPixmap.fromImage(image)
        self.update()

    def set_workspace(self, workspace: SemanticWorkspace | None) -> None:
        self._workspace = workspace
        if workspace is None or workspace.object_by_id(self._selected_id) is None:
            self._selected_id = ""
        self.update()

    def set_selected_object(self, object_id: str) -> None:
        self._selected_id = str(object_id).strip()
        self.update()

    def set_overlay_options(self, *, masks: bool, labels: bool) -> None:
        self._show_masks = bool(masks)
        self._show_labels = bool(labels)
        self.update()

    def set_interaction_mode(self, mode: str) -> None:
        value = str(mode).strip().lower()
        if value not in {"select", "segment_new", "refine_positive", "refine_negative"}:
            raise ValueError(f"Unsupported Auto Segment canvas interaction mode: {mode!r}.")
        self._interaction_mode = value
        self.setCursor(
            Qt.CursorShape.ArrowCursor if value == "select" else Qt.CursorShape.CrossCursor
        )
        self.update()

    def set_prompt_points(
        self,
        *,
        positive: tuple[tuple[float, float], ...] = (),
        negative: tuple[tuple[float, float], ...] = (),
    ) -> None:
        self._positive_points = tuple((float(x), float(y)) for x, y in positive)
        self._negative_points = tuple((float(x), float(y)) for x, y in negative)
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        if not painter.isActive():
            return
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.fillRect(self.rect(), QColor("#090b0e"))
            if self._bgr is None or self._pixmap.isNull():
                painter.setPen(QColor("#9298a3"))
                painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Open an image or use the captured frame")
                return
            target = self._fit_rect(self._bgr.shape[1], self._bgr.shape[0])
            self._image_rect = target
            painter.drawPixmap(target.toRect(), self._pixmap)
            self._paint_workspace(painter, target)
            self._paint_prompt_points(painter, target)
        finally:
            if painter.isActive():
                painter.end()

    def _fit_rect(self, width: int, height: int) -> QRectF:
        margin = 8.0
        aw, ah = max(1.0, self.width() - 2 * margin), max(1.0, self.height() - 2 * margin)
        scale = min(aw / width, ah / height)
        w, h = width * scale, height * scale
        return QRectF((self.width() - w) / 2.0, (self.height() - h) / 2.0, w, h)

    def _paint_workspace(self, painter: QPainter, target: QRectF) -> None:
        if self._workspace is None or self._bgr is None:
            return
        for index, item in enumerate(self._workspace.objects):
            mask_obj = item.proposal_mask if item.proposal_mask is not None else None
            if mask_obj is None:
                continue
            mask = np.asarray(mask_obj.payload.data, dtype=bool)
            color = _PALETTE[index % len(_PALETTE)]
            if self._show_masks and np.any(mask):
                rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
                if item.review_status is SemanticReviewStatus.REJECTED:
                    alpha = 34
                elif item.object_id == self._selected_id:
                    alpha = 124
                elif item.review_status is SemanticReviewStatus.ACCEPTED:
                    alpha = 88
                else:
                    alpha = 66
                rgba[mask] = (color.red(), color.green(), color.blue(), alpha)
                overlay = QImage(
                    np.ascontiguousarray(rgba).data,
                    rgba.shape[1], rgba.shape[0], rgba.shape[1] * 4,
                    QImage.Format.Format_RGBA8888,
                ).copy()
                painter.drawImage(target, overlay)
            if item.object_id == self._selected_id:
                ys, xs = np.nonzero(mask)
                if xs.size:
                    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
                    p0 = self._map_point(float(x0), float(y0), target)
                    p1 = self._map_point(float(x1 + 1), float(y1 + 1), target)
                    painter.setPen(QPen(QColor("#ffffff"), 2.0, Qt.PenStyle.DashLine))
                    painter.drawRect(QRectF(p0, p1).normalized())
            if self._show_labels and (item.class_name or item.object_id == self._selected_id):
                ys, xs = np.nonzero(mask)
                if xs.size:
                    anchor = self._map_point(float(np.median(xs)), float(np.median(ys)), target)
                    label = item.instance_name or item.class_name or "Unlabeled proposal"
                    rect = QRectF(painter.fontMetrics().boundingRect(label)).adjusted(-6, -3, 6, 3)
                    rect.moveCenter(anchor)
                    painter.fillRect(rect, QColor(8, 10, 13, 220))
                    painter.setPen(color)
                    painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, label)

    def _paint_prompt_points(self, painter: QPainter, target: QRectF) -> None:
        for positive, points in (
            (True, self._positive_points),
            (False, self._negative_points),
        ):
            fill = QColor("#42df83") if positive else QColor("#ff5168")
            for x, y in points:
                point = self._map_point(x, y, target)
                painter.setPen(QPen(QColor("#ffffff"), 1.5))
                painter.setBrush(fill)
                painter.drawEllipse(point, 6.0, 6.0)
                painter.setPen(QPen(QColor("#07100b" if positive else "#16060a"), 1.6))
                painter.drawLine(QPointF(point.x() - 3.0, point.y()), QPointF(point.x() + 3.0, point.y()))
                if positive:
                    painter.drawLine(QPointF(point.x(), point.y() - 3.0), QPointF(point.x(), point.y() + 3.0))

    def _map_point(self, x: float, y: float, target: QRectF) -> QPointF:
        assert self._bgr is not None
        return QPointF(
            target.left() + x * target.width() / self._bgr.shape[1],
            target.top() + y * target.height() / self._bgr.shape[0],
        )

    def _widget_to_image(self, point: QPointF) -> tuple[int, int] | None:
        if self._bgr is None or not self._image_rect.contains(point):
            return None
        x = int((point.x() - self._image_rect.left()) * self._bgr.shape[1] / self._image_rect.width())
        y = int((point.y() - self._image_rect.top()) * self._bgr.shape[0] / self._image_rect.height())
        return min(max(x, 0), self._bgr.shape[1] - 1), min(max(y, 0), self._bgr.shape[0] - 1)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            return
        point = self._widget_to_image(event.position())
        if point is None:
            return
        x, y = point
        if self._interaction_mode != "select":
            self.prompt_point_clicked.emit(x, y)
            return
        if self._workspace is None:
            return
        matches: list[tuple[int, str]] = []
        for item in self._workspace.objects:
            if item.proposal_mask is None:
                continue
            mask = np.asarray(item.proposal_mask.payload.data, dtype=bool)
            if mask[y, x]:
                matches.append((int(np.count_nonzero(mask)), item.object_id))
        if matches:
            self.object_clicked.emit(min(matches)[1])

"""Reusable scalable image/depth display surface.

The widget retains the full-resolution source pixmap and derives the visible
pixmap whenever the panel is resized or its inspection state changes. This
avoids the stale-scaling behaviour of assigning a one-off scaled pixmap to a
plain QLabel.
"""

from __future__ import annotations

from PyQt6.QtCore import QRect, Qt
from PyQt6.QtGui import QColor, QPainter, QPen, QPixmap, QTransform
from PyQt6.QtWidgets import QLabel, QSizePolicy, QWidget


class ScalablePixmapLabel(QLabel):
    """Centered pixmap surface with fit/1:1, rotation, and optional grid."""

    def __init__(self, placeholder: str, parent: QWidget | None = None) -> None:
        super().__init__(placeholder, parent)
        self.setObjectName("mediaSurface")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(160)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._placeholder = placeholder
        self._source_pixmap = QPixmap()
        self._rotation_quadrants = 0
        self._fit_to_view = True
        self._grid_visible = False
        self._displayed_rect = QRect()

    @property
    def has_pixmap(self) -> bool:
        return not self._source_pixmap.isNull()

    @property
    def fit_to_view(self) -> bool:
        return self._fit_to_view

    @property
    def grid_visible(self) -> bool:
        return self._grid_visible

    @property
    def rotation_degrees(self) -> int:
        return (self._rotation_quadrants % 4) * 90

    def set_source_pixmap(self, pixmap: QPixmap) -> None:
        self._source_pixmap = QPixmap(pixmap)
        self.setText("")
        self._refresh_visible_pixmap()

    def clear_source(self) -> None:
        self._source_pixmap = QPixmap()
        self._rotation_quadrants = 0
        self._displayed_rect = QRect()
        self.clear()
        self.setText(self._placeholder)
        self.update()

    def current_full_resolution_pixmap(self) -> QPixmap:
        """Return the current rotated frame without display scaling."""
        if self._source_pixmap.isNull():
            return QPixmap()
        if self._rotation_quadrants % 4 == 0:
            return QPixmap(self._source_pixmap)
        transform = QTransform().rotate(self.rotation_degrees)
        return self._source_pixmap.transformed(
            transform,
            Qt.TransformationMode.SmoothTransformation,
        )

    def rotate_clockwise(self) -> None:
        if self._source_pixmap.isNull():
            return
        self._rotation_quadrants = (self._rotation_quadrants + 1) % 4
        self._refresh_visible_pixmap()

    def set_fit_to_view(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled == self._fit_to_view:
            return
        self._fit_to_view = enabled
        self._refresh_visible_pixmap()

    def set_grid_visible(self, visible: bool) -> None:
        self._grid_visible = bool(visible)
        self.update()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        self._refresh_visible_pixmap()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().paintEvent(event)
        if not self._grid_visible or self._displayed_rect.isEmpty():
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pen = QPen(QColor(255, 255, 255, 72), 1)
        pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)

        rect = self._displayed_rect
        for fraction in (1 / 3, 2 / 3):
            x = rect.left() + round(rect.width() * fraction)
            y = rect.top() + round(rect.height() * fraction)
            painter.drawLine(x, rect.top(), x, rect.bottom())
            painter.drawLine(rect.left(), y, rect.right(), y)
        painter.end()

    def _refresh_visible_pixmap(self) -> None:
        full = self.current_full_resolution_pixmap()
        if full.isNull():
            self._displayed_rect = QRect()
            return

        available = self.contentsRect().size()
        if self._fit_to_view and available.width() > 0 and available.height() > 0:
            visible = full.scaled(
                available,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        else:
            visible = full

        self.setPixmap(visible)
        x = self.contentsRect().left() + max(0, (self.contentsRect().width() - visible.width()) // 2)
        y = self.contentsRect().top() + max(0, (self.contentsRect().height() - visible.height()) // 2)
        self._displayed_rect = QRect(x, y, visible.width(), visible.height())
        self.update()

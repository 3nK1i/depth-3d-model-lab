"""Screen-space orientation gizmo for the Reconstruction viewport."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from PyQt6.QtCore import QPointF, Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import QWidget

from app.theme import PALETTE


class ReconstructionOrientationGizmo(QWidget):
    """Compact bottom-right X/Y/Z orientation indicator.

    Unlike the scene's world-origin axes, this widget is fixed in screen space
    and follows the viewer camera orientation.  It is intentionally read-only in
    C2; axis-click view snapping can be added later without changing geometry
    authority.
    """

    def __init__(
        self,
        projection_provider: Callable[[], np.ndarray],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("reconstructionOrientationGizmo")
        self.setFixedSize(84, 84)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setToolTip("Viewer orientation (X / Y / Z)")
        self._projection_provider = projection_provider
        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self.update)
        self._timer.start()

    def shutdown(self) -> None:
        self._timer.stop()

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

            try:
                projected = np.asarray(self._projection_provider(), dtype=np.float64)
            except Exception:
                projected = np.eye(3, 2, dtype=np.float64)
            if projected.shape != (3, 2) or not np.isfinite(projected).all():
                projected = np.eye(3, 2, dtype=np.float64)

            # Keep the triad visually aligned with the bottom-left safety badge.
            # The widget itself remains bottom-aligned; only the painted triad sits
            # lower now that there is no surrounding card to centre it within.
            origin = QPointF(self.width() * 0.54, self.height() * 0.67)
            length = 22.0
            colors = (
                QColor("#e95b66"),  # X
                QColor("#55c98a"),  # Y
                QColor("#5f88ed"),  # Z
            )
            labels = ("X", "Y", "Z")

            # Draw shorter/foreshortened axes first so the most visible directions
            # remain legible where endpoints overlap.
            order = sorted(range(3), key=lambda i: float(np.linalg.norm(projected[i])))
            label_font = QFont(self.font())
            label_font.setPointSize(7)
            label_font.setWeight(QFont.Weight.DemiBold)
            painter.setFont(label_font)
            for index in order:
                direction = projected[index]
                endpoint = QPointF(
                    origin.x() + float(direction[0]) * length,
                    origin.y() + float(direction[1]) * length,
                )
                pen = QPen(colors[index], 1.6)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                painter.setPen(pen)
                painter.drawLine(origin, endpoint)
                painter.setBrush(colors[index])
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(endpoint, 2.2, 2.2)
                painter.setPen(colors[index])
                label_offset_x = 4.0 if direction[0] >= 0.0 else -9.0
                label_offset_y = 3.0 if direction[1] >= 0.0 else -4.0
                painter.drawText(
                    QPointF(endpoint.x() + label_offset_x, endpoint.y() + label_offset_y),
                    labels[index],
                )

            painter.setBrush(QColor(PALETTE.viewport_text))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(origin, 2.6, 2.6)
        finally:
            painter.end()

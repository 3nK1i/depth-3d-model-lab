"""Reusable pill-style toggle switch, QPainter-drawn.

Matches the target UI's toggle switches (Grid / Axes / Background Removal
/ VSync): a rounded track that fills EIA-red when on, with a white knob
that slides across. Fully vector-drawn and DPI-aware; no image assets.
"""

from __future__ import annotations

from PyQt6.QtCore import (
    QEasingCurve,
    QPropertyAnimation,
    QRectF,
    Qt,
    pyqtProperty,
    pyqtSignal,
)
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import QAbstractButton, QWidget

from app.theme import PALETTE


class ToggleSwitch(QAbstractButton):
    """A checkable pill switch with an animated knob."""

    toggled_changed = pyqtSignal(bool)

    def __init__(
        self,
        checked: bool = False,
        *,
        width: int = 40,
        height: int = 22,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setCheckable(True)
        self.setChecked(checked)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._w = width
        self._h = height
        self.setFixedSize(width, height)

        margin = 3
        self._knob_min = margin
        self._knob_max = width - height + margin
        self._knob_pos = float(self._knob_max if checked else self._knob_min)

        self._anim = QPropertyAnimation(self, b"knob_position", self)
        self._anim.setDuration(140)
        self._anim.setEasingCurve(QEasingCurve.Type.InOutCubic)

        self.toggled.connect(self._on_toggled)

    def _on_toggled(self, checked: bool) -> None:
        self._anim.stop()
        self._anim.setStartValue(self._knob_pos)
        self._anim.setEndValue(self._knob_max if checked else self._knob_min)
        self._anim.start()
        self.toggled_changed.emit(checked)

    def get_knob_position(self) -> float:
        return self._knob_pos

    def set_knob_position(self, value: float) -> None:
        self._knob_pos = value
        self.update()

    knob_position = pyqtProperty(float, fget=get_knob_position, fset=set_knob_position)

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        enabled = self.isEnabled()
        if self.isChecked():
            track = QColor(PALETTE.accent if enabled else PALETTE.accent_disabled)
        else:
            track = QColor(PALETTE.border_light if enabled else PALETTE.border)

        radius = self._h / 2
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(track)
        painter.drawRoundedRect(QRectF(0, 0, self._w, self._h), radius, radius)

        knob_d = self._h - 6
        knob_color = QColor("#ffffff" if enabled else "#9a9ba3")
        painter.setBrush(knob_color)
        painter.drawEllipse(QRectF(self._knob_pos, 3, knob_d, knob_d))
        painter.end()

    def sizeHint(self):  # noqa: N802 - Qt override
        from PyQt6.QtCore import QSize

        return QSize(self._w, self._h)
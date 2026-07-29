"""A slider paired with a live numeric value label, matching the target.

The target UI shows each depth/render slider as: NAME on the left, a red
track in the middle, and a formatted value (e.g. ``0.1 m``, ``0.6``) on
the right. This widget encapsulates that row so the settings panel does
not re-implement it per control.
"""

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QSlider, QWidget


class LabeledSlider(QWidget):
    """Label + horizontal slider + formatted value readout."""

    value_changed = pyqtSignal(int)

    def __init__(
        self,
        label_text: str,
        *,
        minimum: int = 0,
        maximum: int = 100,
        value: int = 0,
        value_formatter: Callable[[int], str] | None = None,
        label_width: int = 116,
        value_width: int = 44,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._formatter = value_formatter or (lambda v: str(v))

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        self._label = QLabel(label_text)
        self._label.setObjectName("settingLabel")
        self._label.setFixedWidth(label_width)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(minimum)
        self._slider.setMaximum(maximum)
        self._slider.setValue(value)

        self._value = QLabel(self._formatter(value))
        self._value.setObjectName("settingValue")
        self._value.setFixedWidth(value_width)
        self._value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self._slider.valueChanged.connect(self._on_changed)

        row.addWidget(self._label)
        row.addWidget(self._slider, stretch=1)
        row.addWidget(self._value)

    def _on_changed(self, value: int) -> None:
        self._value.setText(self._formatter(value))
        self.value_changed.emit(value)

    @property
    def slider(self) -> QSlider:
        return self._slider

    def value(self) -> int:
        return self._slider.value()
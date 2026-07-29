"""Reusable controls for the right settings sidebar.

Every settings card composes these rather than styling widgets ad hoc, so
alignment, heights and typography stay identical across all three cards.

Row grid contract (shared by all cards, giving one common right edge):

    col 0 -> label            (stretch)
    col 1 -> slider / filler  (stretch)
    col 2 -> value box        (fixed VALUE_BOX_WIDTH)

Combo and toggle rows span columns 1-2 so their right edge lines up with
the slider value boxes.
"""

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPainter
from PyQt6.QtWidgets import QComboBox, QLabel, QSizePolicy, QSlider, QWidget

from app.icons import render_pixmap
from app.theme import PALETTE
from app.widgets.toggle_switch import ToggleSwitch

CONTROL_HEIGHT = 29
VALUE_BOX_WIDTH = 48
ROW_VSPACING = 6
COL_HSPACING = 10


class SettingsComboBox(QComboBox):
    """Compact combo box that paints its own vector down-chevron.

    Qt draws no arrow once ``::drop-down`` is restyled and no image is
    supplied, which is why the previous combos had no chevron at all.
    Painting it here keeps the icon vector and DPI-aware.
    """

    CHEVRON_SIZE = 12
    CHEVRON_MARGIN = 10

    def __init__(self, items: list[str] | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("settingsCombo")
        self.setFixedHeight(CONTROL_HEIGHT)
        # Without these a QComboBox demands the width of its longest item,
        # which pushes the whole card wider than the sidebar.
        self.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.setMinimumContentsLength(6)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        if items:
            self.addItems(items)
        self._chevron = render_pixmap(
            "chevron_down", PALETTE.settings_text_secondary, self.CHEVRON_SIZE, stroke_width=2.6
        )

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        x = self.width() - self.CHEVRON_SIZE - self.CHEVRON_MARGIN
        y = (self.height() - self.CHEVRON_SIZE) // 2
        painter.drawPixmap(x, y, self._chevron)
        painter.end()


class ValueBox(QLabel):
    """Small bordered readout shown at the right of a slider row."""

    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setObjectName("valueBox")
        self.setFixedWidth(VALUE_BOX_WIDTH)
        self.setFixedHeight(20)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)


class StatusBadge(QLabel):
    """Compact inline badge (e.g. RECOMMENDED) that never wraps."""

    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setObjectName("statusBadge")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFixedHeight(CONTROL_HEIGHT - 12)
        self.setSizePolicy(self.sizePolicy().horizontalPolicy(), self.sizePolicy().verticalPolicy())


class RowLabel(QLabel):
    """Settings row caption. ``uppercase`` matches the MODEL card style."""

    def __init__(self, text: str, uppercase: bool = False, parent: QWidget | None = None) -> None:
        super().__init__(text.upper() if uppercase else text, parent)
        self.setObjectName("settingsRowLabelUpper" if uppercase else "settingsRowLabel")
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)


class SettingsSlider(QSlider):
    """Horizontal slider with the neutral-grey knob used in the sidebar."""

    def __init__(
        self,
        minimum: int,
        maximum: int,
        value: int,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(Qt.Orientation.Horizontal, parent)
        self.setObjectName("settingsSlider")
        self.setMinimum(minimum)
        self.setMaximum(maximum)
        self.setValue(value)
        self.setFixedHeight(20)


class SliderValueBinding(QWidget):
    """Invisible helper binding a slider to its value box via a formatter."""

    value_changed = pyqtSignal(int)

    def __init__(
        self,
        slider: SettingsSlider,
        value_box: ValueBox,
        formatter: Callable[[int], str],
    ) -> None:
        super().__init__()
        self.hide()
        self._slider = slider
        self._box = value_box
        self._formatter = formatter
        value_box.setText(formatter(slider.value()))
        slider.valueChanged.connect(self._on_changed)

    def _on_changed(self, value: int) -> None:
        self._box.setText(self._formatter(value))
        self.value_changed.emit(value)


def make_toggle(checked: bool) -> ToggleSwitch:
    """Sidebar toggle switch at the compact target proportions."""
    return ToggleSwitch(checked=checked, width=42, height=21)
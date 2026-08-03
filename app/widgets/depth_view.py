"""Depth-map visualization panel with synchronized, functional controls."""

from __future__ import annotations

from PyQt6.QtCore import QSignalBlocker, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QLinearGradient, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.widgets.icon_button import IconButton
from app.widgets.media_canvas import ScalablePixmapLabel

_COLORMAPS = ("Turbo", "Inferno", "Magma", "Viridis", "Greyscale")
_MIN_DEPTH_PRESETS = (0.1, 0.3, 0.5, 1.0, 2.0)
_MAX_DEPTH_PRESETS = (3.0, 5.0, 8.0, 10.0, 20.0, 50.0)


_COLORMAP_STOPS: dict[str, tuple[str, ...]] = {
    "Turbo": ("#30123b", "#4662d7", "#20b5aa", "#a4fc3c", "#f9ba38", "#7a0403"),
    "Inferno": ("#000004", "#420a68", "#932667", "#dd513a", "#fca50a", "#fcffa4"),
    "Magma": ("#000004", "#3b0f70", "#8c2981", "#de4968", "#fe9f6d", "#fcfdbf"),
    "Viridis": ("#440154", "#414487", "#2a788e", "#22a884", "#7ad151", "#fde725"),
    "Greyscale": ("#000000", "#333333", "#666666", "#999999", "#cccccc", "#ffffff"),
}


def _format_metres(value: float) -> str:
    return f"{float(value):.1f} m"


def _parse_metres(text: str) -> float:
    return float(text.strip().lower().removesuffix("m").strip())


class _ColorBar(QFrame):
    """Vertical gradient that follows the selected depth colormap."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._colormap_name = "Turbo"
        self.setFixedWidth(14)
        self.setMinimumHeight(120)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

    def set_colormap(self, name: str) -> None:
        normalized = str(name).strip().title()
        if normalized == "Greyscale":
            key = "Greyscale"
        else:
            key = next(
                (candidate for candidate in _COLORMAP_STOPS if candidate.lower() == normalized.lower()),
                "Turbo",
            )
        if key != self._colormap_name:
            self._colormap_name = key
            self.update()

    @property
    def colormap_name(self) -> str:
        return self._colormap_name

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        gradient = QLinearGradient(0, self.height(), 0, 0)
        colors = _COLORMAP_STOPS[self._colormap_name]
        denominator = max(1, len(colors) - 1)
        for index, color in enumerate(colors):
            gradient.setColorAt(index / denominator, QColor(color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(gradient)
        painter.drawRoundedRect(self.rect(), 3, 3)
        painter.end()


class _ColorBarColumn(QWidget):
    """Color bar plus FAR/NEAR value captions above and below it."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        col = QVBoxLayout(self)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(3)

        far = QLabel("FAR")
        far.setObjectName("metricLabel")
        far.setStyleSheet("color: #e0293e; font-size: 9px; font-weight: 700;")
        far.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._far_value = QLabel("8.0 m")
        self._far_value.setObjectName("muted")
        self._far_value.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._bar = _ColorBar()
        bar_row = QHBoxLayout()
        bar_row.addStretch(1)
        bar_row.addWidget(self._bar)
        bar_row.addStretch(1)

        self._near_value = QLabel("0.1 m")
        self._near_value.setObjectName("muted")
        self._near_value.setAlignment(Qt.AlignmentFlag.AlignCenter)

        col.addWidget(far)
        col.addWidget(self._far_value)
        col.addLayout(bar_row, stretch=1)
        col.addWidget(self._near_value)

    def set_range(self, near_m: float, far_m: float) -> None:
        self._near_value.setText(_format_metres(near_m))
        self._far_value.setText(_format_metres(far_m))

    def set_colormap(self, name: str) -> None:
        self._bar.set_colormap(name)


class DepthMapView(QFrame):
    """Displays the colorized depth map and mirrors the sidebar controls."""

    colormap_changed = pyqtSignal(str)
    depth_range_changed = pyqtSignal(float, float)
    fullscreen_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("panel")
        self._build_ui()
        self._fullscreen_btn.setEnabled(False)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)
        layout.addLayout(self._build_title_row())

        body = QHBoxLayout()
        body.setSpacing(8)
        self._display = ScalablePixmapLabel("No depth output")
        body.addWidget(self._display, stretch=1)
        self._color_bar = _ColorBarColumn()
        body.addWidget(self._color_bar)
        layout.addLayout(body, stretch=1)

        layout.addWidget(self._build_controls_bar())

    def _build_title_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        dot = QLabel("●")
        dot.setStyleSheet("color: #e0293e; font-size: 9px;")
        title = QLabel("DEPTH MAP")
        title.setObjectName("panelTitle")
        row.addWidget(dot)
        row.addWidget(title)
        row.addStretch(1)
        return row

    def _build_controls_bar(self) -> QFrame:
        controls = QFrame()
        controls.setObjectName("depthControlsBar")
        row = QHBoxLayout(controls)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        colormap_label = QLabel("Colormap")
        colormap_label.setObjectName("depthControlLabel")
        self._colormap_combo = self._make_combo(_COLORMAPS, minimum_width=110)

        min_label = QLabel("Min Depth")
        min_label.setObjectName("depthControlLabel")
        self._min_depth = self._make_combo(
            tuple(_format_metres(v) for v in _MIN_DEPTH_PRESETS),
            minimum_width=80,
        )

        max_label = QLabel("Max Depth")
        max_label.setObjectName("depthControlLabel")
        self._max_depth = self._make_combo(
            tuple(_format_metres(v) for v in _MAX_DEPTH_PRESETS),
            minimum_width=80,
        )
        self._max_depth.setCurrentText("8.0 m")

        row.addWidget(colormap_label)
        row.addWidget(self._colormap_combo)
        row.addSpacing(4)
        row.addWidget(min_label)
        row.addWidget(self._min_depth)
        row.addWidget(max_label)
        row.addWidget(self._max_depth)
        row.addStretch(1)

        self._fullscreen_btn = IconButton(
            "fullscreen",
            size=30,
            icon_size=16,
            tooltip="Focus the Depth Map panel (Esc to restore)",
            checkable=True,
        )
        row.addWidget(self._fullscreen_btn)

        self._colormap_combo.currentTextChanged.connect(self._on_colormap_changed)
        self._min_depth.currentTextChanged.connect(lambda _text: self._on_range_changed("min"))
        self._max_depth.currentTextChanged.connect(lambda _text: self._on_range_changed("max"))
        self._fullscreen_btn.clicked.connect(self.fullscreen_requested.emit)
        return controls

    @staticmethod
    def _make_combo(items: tuple[str, ...], *, minimum_width: int) -> QComboBox:
        combo = QComboBox()
        combo.setObjectName("depthControlCombo")
        combo.addItems(list(items))
        combo.setProperty("presetDepthValues", list(items))
        combo.setMinimumWidth(minimum_width)
        combo.setFixedHeight(30)
        return combo

    def set_depth_frame(self, pixmap: QPixmap) -> None:
        self._display.set_source_pixmap(pixmap)
        self._fullscreen_btn.setEnabled(True)

    def clear_frame(self) -> None:
        self._display.clear_source()
        self._fullscreen_btn.setChecked(False)
        self._fullscreen_btn.setEnabled(False)

    def sync_controls(self, colormap: str, near_m: float, far_m: float) -> None:
        """Mirror sidebar values without feeding signals back to the controller."""
        blockers = (
            QSignalBlocker(self._colormap_combo),
            QSignalBlocker(self._min_depth),
            QSignalBlocker(self._max_depth),
        )
        self._set_combo_text(self._colormap_combo, colormap)
        self._set_combo_text(self._min_depth, _format_metres(near_m))
        self._set_combo_text(self._max_depth, _format_metres(far_m))
        self._color_bar.set_colormap(colormap)
        self._color_bar.set_range(near_m, far_m)
        del blockers

    def set_fullscreen_active(self, active: bool) -> None:
        self._fullscreen_btn.setChecked(bool(active))

    def _on_colormap_changed(self, name: str) -> None:
        self._color_bar.set_colormap(name)
        self.colormap_changed.emit(name)

    def _on_range_changed(self, changed: str) -> None:
        near = _parse_metres(self._min_depth.currentText())
        far = _parse_metres(self._max_depth.currentText())
        if near >= far:
            if changed == "min":
                far = near + 0.1
                blocker = QSignalBlocker(self._max_depth)
                self._set_combo_text(self._max_depth, _format_metres(far))
                del blocker
            else:
                near = max(0.0, far - 0.1)
                blocker = QSignalBlocker(self._min_depth)
                self._set_combo_text(self._min_depth, _format_metres(near))
                del blocker
        self._color_bar.set_range(near, far)
        self.depth_range_changed.emit(near, far)

    @staticmethod
    def _set_combo_text(combo: QComboBox, text: str) -> None:
        presets = set(combo.property("presetDepthValues") or [])
        previous_dynamic = str(combo.property("dynamicDepthValue") or "")
        if previous_dynamic and previous_dynamic != text:
            previous_index = combo.findText(previous_dynamic)
            if previous_index >= 0:
                combo.removeItem(previous_index)
            combo.setProperty("dynamicDepthValue", "")

        index = combo.findText(text)
        if index < 0:
            combo.addItem(text)
            combo.setProperty("dynamicDepthValue", text)
            index = combo.findText(text)
        elif text in presets and previous_dynamic == text:
            combo.setProperty("dynamicDepthValue", "")
        combo.setCurrentIndex(index)

    @property
    def colormap_combo(self) -> QComboBox:
        return self._colormap_combo

    @property
    def min_depth_m(self) -> float:
        return _parse_metres(self._min_depth.currentText())

    @property
    def max_depth_m(self) -> float:
        return _parse_metres(self._max_depth.currentText())

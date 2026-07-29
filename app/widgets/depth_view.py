"""Depth map visualization panel.

Matches the target: the colorized depth image with a vertical color bar
on the right (FAR→NEAR, Turbo gradient, 8.0 m / 0.1 m labels), and a
bottom control row with Colormap / Min Depth / Max Depth dropdowns, a
settings gear and a fullscreen action.

Colormapping must never mutate the original depth array (spec §11); that
rule lives in :mod:`core.geometry.colors`. This widget only displays an
already-colorized image handed to it via :meth:`set_depth_frame`.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QLinearGradient, QPainter, QPixmap
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


class _ColorBar(QFrame):
    """Vertical Turbo-style gradient bar with FAR/NEAR range labels."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedWidth(14)
        self.setMinimumHeight(120)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        gradient = QLinearGradient(0, 0, 0, self.height())
        # Turbo-ish: far (red/orange) at top -> near (blue) at bottom.
        gradient.setColorAt(0.0, Qt.GlobalColor.red)
        gradient.setColorAt(0.25, Qt.GlobalColor.yellow)
        gradient.setColorAt(0.5, Qt.GlobalColor.green)
        gradient.setColorAt(0.75, Qt.GlobalColor.cyan)
        gradient.setColorAt(1.0, Qt.GlobalColor.blue)
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

        bar_row = QHBoxLayout()
        bar_row.addStretch(1)
        bar_row.addWidget(_ColorBar())
        bar_row.addStretch(1)

        self._near_value = QLabel("0.1 m")
        self._near_value.setObjectName("muted")
        self._near_value.setAlignment(Qt.AlignmentFlag.AlignCenter)

        col.addWidget(far)
        col.addWidget(self._far_value)
        col.addLayout(bar_row, stretch=1)
        col.addWidget(self._near_value)

    def set_range(self, near_m: float, far_m: float) -> None:
        self._near_value.setText(f"{near_m:.1f} m")
        self._far_value.setText(f"{far_m:.1f} m")


class DepthMapView(QFrame):
    """Displays the colorized depth map for the current frame."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("panel")
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)
        layout.addLayout(self._build_title_row())

        body = QHBoxLayout()
        body.setSpacing(8)
        body.addWidget(self._build_display_area(), stretch=1)
        self._color_bar = _ColorBarColumn()
        body.addWidget(self._color_bar)
        layout.addLayout(body, stretch=1)

        layout.addLayout(self._build_controls_row())

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

    def _build_display_area(self) -> QLabel:
        self._display = QLabel("No depth output")
        self._display.setObjectName("mediaSurface")
        self._display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._display.setMinimumHeight(160)
        self._display.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        return self._display

    def _build_controls_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(8)

        colormap_label = QLabel("Colormap")
        colormap_label.setObjectName("muted")
        self._colormap_combo = QComboBox()
        self._colormap_combo.addItems(["Turbo", "Inferno", "Magma", "Viridis", "Greyscale"])
        self._colormap_combo.setMinimumWidth(110)

        min_label = QLabel("Min Depth")
        min_label.setObjectName("muted")
        self._min_depth = QComboBox()
        self._min_depth.addItems(["0.1 m", "0.5 m", "1.0 m"])
        self._min_depth.setMinimumWidth(80)

        max_label = QLabel("Max Depth")
        max_label.setObjectName("muted")
        self._max_depth = QComboBox()
        self._max_depth.addItems(["8.0 m", "5.0 m", "10.0 m"])
        self._max_depth.setMinimumWidth(80)

        row.addWidget(colormap_label)
        row.addWidget(self._colormap_combo)
        row.addSpacing(6)
        row.addWidget(min_label)
        row.addWidget(self._min_depth)
        row.addWidget(max_label)
        row.addWidget(self._max_depth)
        row.addStretch(1)

        row.addWidget(IconButton("settings", size=28, icon_size=15, tooltip="Depth settings"))
        row.addWidget(IconButton("fullscreen", size=28, icon_size=15, tooltip="Fullscreen"))
        return row

    def set_depth_frame(self, pixmap: QPixmap) -> None:
        self._display.setPixmap(
            pixmap.scaled(
                self._display.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def clear_frame(self) -> None:
        self._display.clear()
        self._display.setText("No depth output")

    @property
    def colormap_combo(self) -> QComboBox:
        return self._colormap_combo
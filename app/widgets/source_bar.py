"""Source selection bar: Image / Video / Camera / RTSP + dynamic config.

Matches the target: source-type buttons with vector icons, per-source
config controls with sensible minimum widths so they never clip (the
earlier "Ca / 12 / 30" truncation), and a CONNECTED/DISCONNECTED badge.

Phase 1 builds the visual shell and source-type switching only. Real
device enumeration, file browsing and RTSP fields are wired in Phase 2
against :mod:`core.capture`.
"""

from __future__ import annotations

from enum import Enum

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QStackedWidget,
    QWidget,
)

from app.icons import render_icon
from app.theme import PALETTE
from app.widgets.badge import BadgeKind, BadgeLabel
from app.widgets.icon_button import IconButton


class SourceType(Enum):
    IMAGE = "image"
    VIDEO = "video"
    CAMERA = "camera"
    RTSP = "rtsp"


_SOURCE_META: dict[SourceType, tuple[str, str]] = {
    SourceType.IMAGE: ("image", "IMAGE"),
    SourceType.VIDEO: ("video", "VIDEO"),
    SourceType.CAMERA: ("camera", "CAMERA"),
    SourceType.RTSP: ("rtsp", "RTSP"),
}


class SourceBar(QFrame):
    """Row of source-type buttons plus a stacked area of per-source controls."""

    source_type_changed = pyqtSignal(SourceType)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("sourceBar")
        self._type_buttons: dict[SourceType, QPushButton] = {}
        self._active_type = SourceType.CAMERA
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 8, 12, 8)
        layout.setSpacing(12)

        label = QLabel("INPUT SOURCE")
        label.setObjectName("sectionTitle")
        layout.addWidget(label)

        type_row = QHBoxLayout()
        type_row.setSpacing(6)
        for source_type in SourceType:
            icon_name, text = _SOURCE_META[source_type]
            btn = QPushButton(f"  {text}")
            btn.setObjectName("sourceTypeButton")
            btn.setCheckable(True)
            btn.setIcon(render_icon(icon_name, PALETTE.text_secondary, 15))
            btn.clicked.connect(lambda _c, t=source_type: self._on_type_clicked(t))
            self._type_buttons[source_type] = btn
            type_row.addWidget(btn)
        layout.addLayout(type_row)

        self._stack = QStackedWidget()
        for source_type in SourceType:
            self._stack.addWidget(self._build_config_widget(source_type))
        layout.addWidget(self._stack)

        self._connection_badge = BadgeLabel(BadgeKind.DISCONNECTED_FLAT, "DISCONNECTED")
        layout.addWidget(self._connection_badge)
        layout.addStretch(1)

        self._sync_type_button_icons()
        self._type_buttons[self._active_type].setChecked(True)
        self._stack.setCurrentIndex(list(SourceType).index(self._active_type))

    def _build_config_widget(self, source_type: SourceType) -> QWidget:
        widget = QWidget()
        row = QHBoxLayout(widget)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        if source_type is SourceType.IMAGE:
            browse = QPushButton("Browse…")
            recent = QComboBox()
            recent.addItem("Recent files…")
            recent.setMinimumWidth(160)
            row.addWidget(browse)
            row.addWidget(recent)
        elif source_type is SourceType.VIDEO:
            browse = QPushButton("Browse…")
            path = QLineEdit()
            path.setPlaceholderText("No video loaded")
            path.setReadOnly(True)
            path.setMinimumWidth(220)
            row.addWidget(browse)
            row.addWidget(path)
        elif source_type is SourceType.CAMERA:
            device = QComboBox()
            device.addItem("Camera 0")
            device.setMinimumWidth(110)
            resolution = QComboBox()
            resolution.addItems(["1280x720", "1920x1080", "640x480"])
            resolution.setMinimumWidth(110)
            fps = QComboBox()
            fps.addItems(["30 FPS", "60 FPS", "15 FPS"])
            fps.setMinimumWidth(90)
            refresh = IconButton("refresh", size=32, icon_size=15, tooltip="Refresh devices")
            row.addWidget(device)
            row.addWidget(resolution)
            row.addWidget(fps)
            row.addWidget(refresh)
        elif source_type is SourceType.RTSP:
            url = QLineEdit()
            url.setPlaceholderText("rtsp://host:port/stream")
            url.setMinimumWidth(240)
            test_btn = QPushButton("Test")
            row.addWidget(url)
            row.addWidget(test_btn)

        return widget

    def _on_type_clicked(self, source_type: SourceType) -> None:
        if source_type == self._active_type:
            self._type_buttons[source_type].setChecked(True)
            return
        self._active_type = source_type
        for t, btn in self._type_buttons.items():
            btn.setChecked(t == source_type)
        self._sync_type_button_icons()
        self._stack.setCurrentIndex(list(SourceType).index(source_type))
        self.source_type_changed.emit(source_type)

    def _sync_type_button_icons(self) -> None:
        """Tint the checked source button's icon EIA-red to match its text."""
        for t, btn in self._type_buttons.items():
            icon_name, _ = _SOURCE_META[t]
            color = PALETTE.accent if t == self._active_type else PALETTE.text_secondary
            btn.setIcon(render_icon(icon_name, color, 15))

    def set_connection_state(self, connected: bool) -> None:
        if connected:
            self._connection_badge.set_kind(BadgeKind.CONNECTED_FLAT, "CONNECTED")
        else:
            self._connection_badge.set_kind(BadgeKind.DISCONNECTED_FLAT, "DISCONNECTED")

    @property
    def active_type(self) -> SourceType:
        return self._active_type
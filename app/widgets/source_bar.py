"""Source selection: two grouped cards — source type, and source config.

Matches the target: a segmented Image/Video/Camera/RTSP control (no
individual button outlines, translucent red fill + bottom accent line
on the active one) in its own card, and a second card holding the
per-source config controls plus the connection badge. Both cards are
built on the shared :class:`ToolbarGroupCard` shell so they carry
identical height/padding/background — no more single continuous bar.

Phase 1 builds the visual shell and source-type switching only. Real
device enumeration, file browsing and RTSP fields are wired in Phase 2
against :mod:`core.capture`.
"""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QWidget,
)

from app.icons import render_icon
from app.theme import PALETTE
from app.widgets.settings_controls import SettingsComboBox
from app.widgets.source_connection_badge import SourceConnectionBadge
from app.widgets.source_toolbar_snapshot import ConnectionState, SourceType
from app.widgets.toolbar_group_card import ToolbarGroupCard

_SOURCE_META: dict[SourceType, tuple[str, str]] = {
    SourceType.IMAGE: ("image", "IMAGE"),
    SourceType.VIDEO: ("video", "VIDEO"),
    SourceType.CAMERA: ("camera", "CAMERA"),
    SourceType.RTSP: ("rtsp", "RTSP"),
}

# Approximate width shares within SourceBar's own two-card row (spec §2:
# source type ~34%, source configuration ~30%, relative to each other
# here since SourceRuntimeBar applies the overall 3-card proportions).
_TYPE_CARD_STRETCH = 53
_CONFIG_CARD_STRETCH = 47


class SourceBar(QWidget):
    """Source-type card + source-configuration card, side by side."""

    source_type_changed = pyqtSignal(SourceType)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._type_buttons: dict[SourceType, QPushButton] = {}
        self._active_type = SourceType.CAMERA
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        layout.addWidget(self._build_type_card(), _TYPE_CARD_STRETCH)
        layout.addWidget(self._build_config_card(), _CONFIG_CARD_STRETCH)

    def _build_type_card(self) -> ToolbarGroupCard:
        card = ToolbarGroupCard()
        row = card.content_layout

        label = QLabel("INPUT SOURCE")
        label.setObjectName("toolbarSectionTitle")
        row.addWidget(label)
        row.addSpacing(4)

        for source_type in SourceType:
            icon_name, text = _SOURCE_META[source_type]
            btn = QPushButton(f"  {text}")
            btn.setObjectName("sourceTypeButton")
            btn.setCheckable(True)
            btn.setIcon(render_icon(icon_name, PALETTE.toolbar_text_muted, 15))
            btn.clicked.connect(lambda _c, t=source_type: self._on_type_clicked(t))
            self._type_buttons[source_type] = btn
            row.addWidget(btn)

        self._sync_type_button_icons()
        self._type_buttons[self._active_type].setChecked(True)
        return card

    def _build_config_card(self) -> ToolbarGroupCard:
        card = ToolbarGroupCard()
        row = card.content_layout

        self._stack = QStackedWidget()
        self._stack.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        for source_type in SourceType:
            self._stack.addWidget(self._build_config_widget(source_type))
        row.addWidget(self._stack, stretch=1)

        self._connection_badge = SourceConnectionBadge()
        row.addWidget(self._connection_badge)

        self._stack.setCurrentIndex(list(SourceType).index(self._active_type))
        return card

    def _build_config_widget(self, source_type: SourceType) -> QWidget:
        widget = QWidget()
        row = QHBoxLayout(widget)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(7)

        if source_type is SourceType.IMAGE:
            browse = QPushButton("Browse…")
            recent = SettingsComboBox(["Recent files…"])
            recent.setMinimumWidth(150)
            row.addWidget(browse)
            row.addWidget(recent)
        elif source_type is SourceType.VIDEO:
            browse = QPushButton("Browse…")
            path = QLineEdit()
            path.setPlaceholderText("No video loaded")
            path.setReadOnly(True)
            path.setMinimumWidth(200)
            row.addWidget(browse)
            row.addWidget(path)
        elif source_type is SourceType.CAMERA:
            # No refresh button in this target-matching composition (spec
            # §7): device rescan stays available contextually elsewhere
            # without disturbing this exact row.
            device = SettingsComboBox(["Camera 0"])
            device.setMinimumWidth(100)
            resolution = SettingsComboBox(["1280\u00d7720", "1920\u00d71080", "640\u00d7480"])
            resolution.setMinimumWidth(100)
            fps = SettingsComboBox(["30 FPS", "60 FPS", "15 FPS"])
            fps.setMinimumWidth(80)
            row.addWidget(device)
            row.addWidget(resolution)
            row.addWidget(fps)
        elif source_type is SourceType.RTSP:
            url = QLineEdit()
            url.setPlaceholderText("rtsp://host:port/stream")
            url.setMinimumWidth(200)
            test_btn = QPushButton("Test")
            row.addWidget(url)
            row.addWidget(test_btn)

        row.addStretch(1)
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
            color = PALETTE.accent if t == self._active_type else PALETTE.toolbar_text_muted
            btn.setIcon(render_icon(icon_name, color, 15))

    def set_connection_state(self, connected: bool) -> None:
        """Backward-compatible bool API; see :meth:`set_connection_enum`
        for the full 5-state vocabulary (spec §8)."""
        self.set_connection_enum(
            ConnectionState.CONNECTED if connected else ConnectionState.DISCONNECTED
        )

    def set_connection_enum(self, state: ConnectionState) -> None:
        self._connection_badge.set_state(state)

    @property
    def active_type(self) -> SourceType:
        return self._active_type
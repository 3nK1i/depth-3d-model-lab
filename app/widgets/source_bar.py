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

from PyQt6.QtCore import QTimer, pyqtSignal
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
_SOURCE_TYPE_TEXT_MIN_WIDTH = 350


class SourceBar(QWidget):
    """Source-type card + source-configuration card, side by side."""

    source_type_changed = pyqtSignal(SourceType)
    browse_image_clicked = pyqtSignal()
    browse_video_clicked = pyqtSignal()
    rtsp_test_clicked = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._type_buttons: dict[SourceType, QPushButton] = {}
        self._active_type = SourceType.CAMERA
        self._compact_source_types = False
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self._type_card = self._build_type_card()
        layout.addWidget(self._type_card, _TYPE_CARD_STRETCH)
        layout.addWidget(self._build_config_card(), _CONFIG_CARD_STRETCH)
        QTimer.singleShot(0, self._refresh_source_type_presentation)

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
            btn.setToolTip(text.title() if source_type is not SourceType.RTSP else "RTSP")
            btn.setAccessibleName(text.title() if source_type is not SourceType.RTSP else "RTSP")
            btn.clicked.connect(lambda _c, t=source_type: self._on_type_clicked(t))
            btn.setMinimumWidth(0)
            btn.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            self._type_buttons[source_type] = btn
            row.addWidget(btn, stretch=1)

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
            self._image_browse_btn = QPushButton("Browse…")
            self._image_browse_btn.clicked.connect(self.browse_image_clicked.emit)
            recent = SettingsComboBox(["Recent files…"])
            recent.setMinimumWidth(0)
            recent.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            self._image_recent_combo = recent
            row.addWidget(self._image_browse_btn)
            row.addWidget(recent, stretch=1)
        elif source_type is SourceType.VIDEO:
            self._video_browse_btn = QPushButton("Browse…")
            self._video_browse_btn.clicked.connect(self.browse_video_clicked.emit)
            self._video_path_edit = QLineEdit()
            self._video_path_edit.setPlaceholderText("No video loaded")
            self._video_path_edit.setReadOnly(True)
            self._video_path_edit.setMinimumWidth(0)
            self._video_path_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            row.addWidget(self._video_browse_btn)
            row.addWidget(self._video_path_edit, stretch=1)
        elif source_type is SourceType.CAMERA:
            # No refresh button in this target-matching composition (spec
            # §7): device rescan stays available contextually elsewhere
            # without disturbing this exact row.
            self._camera_device_combo = SettingsComboBox(["Camera 0"])
            self._camera_device_combo.setMinimumWidth(0)
            self._camera_device_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            self._camera_resolution_combo = SettingsComboBox(
                ["1280\u00d7720", "1920\u00d71080", "640\u00d7480"]
            )
            self._camera_resolution_combo.setMinimumWidth(0)
            self._camera_resolution_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            self._camera_fps_combo = SettingsComboBox(["30 FPS", "60 FPS", "15 FPS"])
            self._camera_fps_combo.setMinimumWidth(0)
            self._camera_fps_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            row.addWidget(self._camera_device_combo, stretch=10)
            row.addWidget(self._camera_resolution_combo, stretch=12)
            row.addWidget(self._camera_fps_combo, stretch=8)
        elif source_type is SourceType.RTSP:
            self._rtsp_url_edit = QLineEdit()
            self._rtsp_url_edit.setPlaceholderText("rtsp://host:port/stream")
            self._rtsp_url_edit.setMinimumWidth(0)
            self._rtsp_url_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            self._rtsp_test_btn = QPushButton("Test")
            self._rtsp_test_btn.clicked.connect(self.rtsp_test_clicked.emit)
            row.addWidget(self._rtsp_url_edit, stretch=1)
            row.addWidget(self._rtsp_test_btn)

        row.addStretch(1)
        return widget


    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        # Child card geometry settles after the parent resize/layout pass.
        QTimer.singleShot(0, self._refresh_source_type_presentation)

    def _refresh_source_type_presentation(self) -> None:
        if not hasattr(self, "_type_card"):
            return
        compact = self._type_card.width() < _SOURCE_TYPE_TEXT_MIN_WIDTH
        presentation_matches = all(
            (button.text() == "") == compact for button in self._type_buttons.values()
        )
        if compact == self._compact_source_types and presentation_matches:
            return
        self._compact_source_types = compact
        for source_type, button in self._type_buttons.items():
            _, text = _SOURCE_META[source_type]
            button.setText("" if compact else f"  {text}")

    @property
    def compact_source_types(self) -> bool:
        return self._compact_source_types

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

    def set_camera_devices(self, device_indices: list[int]) -> None:
        """Replace the camera device combo's contents with real detected
        indices. Falls back to a single "Camera 0" placeholder if none
        were found, so the combo is never left empty."""
        combo = self._camera_device_combo
        current_text = combo.currentText()
        combo.clear()
        if device_indices:
            combo.addItems([f"Camera {i}" for i in device_indices])
        else:
            combo.addItem("Camera 0")
        restore_index = combo.findText(current_text)
        combo.setCurrentIndex(restore_index if restore_index >= 0 else 0)

    def camera_config(self) -> tuple[int, int, int, int]:
        """Return (device_index, width, height, fps) parsed from the
        camera config combos."""
        device_text = self._camera_device_combo.currentText()
        device_index = int(device_text.split()[-1]) if device_text.split()[-1].isdigit() else 0
        res_text = self._camera_resolution_combo.currentText()
        width_str, height_str = res_text.split("\u00d7")
        fps_text = self._camera_fps_combo.currentText()
        fps = int(fps_text.split()[0])
        return device_index, int(width_str), int(height_str), fps

    def rtsp_url(self) -> str:
        return self._rtsp_url_edit.text().strip()

    def set_video_path_display(self, path: str) -> None:
        self._video_path_edit.setText(path)

    def set_source_controls_enabled(self, enabled: bool) -> None:
        """Enable or lock source type and configuration controls together."""
        enabled = bool(enabled)
        for button in self._type_buttons.values():
            button.setEnabled(enabled)
        self._stack.setEnabled(enabled)

    @property
    def active_type(self) -> SourceType:
        return self._active_type

"""Original RGB source panel with a functional floating inspection toolbar."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from app.theme import PALETTE
from app.widgets.icon_button import IconButton
from app.widgets.media_canvas import ScalablePixmapLabel


class OriginalSourceView(QFrame):
    """Displays the raw source frame and local, non-destructive view tools."""

    fullscreen_requested = pyqtSignal()
    frame_saved = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("panel")
        self._is_live = False
        self._activity_text = "IDLE"
        self._tool_buttons: list[IconButton] = []
        self._build_ui()
        self._set_tools_enabled(False)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)
        layout.addLayout(self._build_title_row())
        layout.addWidget(self._build_display_surface(), stretch=1)

    def _build_title_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        dot = QLabel("●")
        dot.setStyleSheet(f"color: {PALETTE.accent}; font-size: 9px;")
        title = QLabel("ORIGINAL SOURCE")
        title.setObjectName("panelTitle")
        row.addWidget(dot)
        row.addWidget(title)
        row.addStretch(1)

        self._live_label = QLabel("○ IDLE")
        self._live_label.setObjectName("muted")
        row.addWidget(self._live_label)
        return row

    def _build_display_surface(self) -> QFrame:
        surface = QFrame()
        surface.setObjectName("mediaSurfaceHost")
        stack = QGridLayout(surface)
        stack.setContentsMargins(0, 0, 0, 0)
        stack.setSpacing(0)

        self._display = ScalablePixmapLabel("No source connected")
        stack.addWidget(self._display, 0, 0)

        self._toolbar_card = QFrame()
        self._toolbar_card.setObjectName("mediaToolbarCard")
        toolbar = QHBoxLayout(self._toolbar_card)
        toolbar.setContentsMargins(7, 5, 7, 5)
        toolbar.setSpacing(4)

        self._save_btn = self._make_tool(
            "screenshot",
            "Save the current source frame",
        )
        self._rotate_btn = self._make_tool(
            "refresh",
            "Rotate preview clockwise (display only)",
        )
        self._fullscreen_btn = self._make_tool(
            "fullscreen",
            "Focus the Original Source panel (Esc to restore)",
            checkable=True,
        )
        self._fit_btn = self._make_tool(
            "fit",
            "Fit image to panel / show at actual pixel size",
            checkable=True,
        )
        self._fit_btn.setChecked(True)
        self._grid_btn = self._make_tool(
            "grid",
            "Toggle rule-of-thirds inspection grid",
            checkable=True,
        )

        for button in self._tool_buttons:
            toolbar.addWidget(button)

        self._save_btn.clicked.connect(self._save_current_frame)
        self._rotate_btn.clicked.connect(self._display.rotate_clockwise)
        self._fullscreen_btn.clicked.connect(self.fullscreen_requested.emit)
        self._fit_btn.toggled.connect(self._display.set_fit_to_view)
        self._grid_btn.toggled.connect(self._display.set_grid_visible)

        stack.addWidget(
            self._toolbar_card,
            0,
            0,
            alignment=Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom,
        )
        stack.setContentsMargins(0, 0, 0, 10)
        return surface

    def _make_tool(
        self,
        icon_name: str,
        tooltip: str,
        *,
        checkable: bool = False,
    ) -> IconButton:
        button = IconButton(
            icon_name,
            size=30,
            icon_size=16,
            tooltip=tooltip,
            checkable=checkable,
        )
        self._tool_buttons.append(button)
        return button

    def set_frame(self, pixmap: QPixmap) -> None:
        """Display a new RGB frame while retaining full resolution."""
        self._display.set_source_pixmap(pixmap)
        self._set_tools_enabled(True)

    def clear_frame(self) -> None:
        self._display.clear_source()
        self._grid_btn.setChecked(False)
        self._fit_btn.setChecked(True)
        self._set_tools_enabled(False)

    def set_live(self, is_live: bool) -> None:
        """Show LIVE for continuous sources, or IDLE when inactive."""
        if is_live:
            self._set_activity("LIVE", active=True)
        else:
            self._set_activity("IDLE", active=False)

    def set_ready(self) -> None:
        """Show READY for a loaded static image source."""
        self._set_activity("READY", active=True)

    def set_fullscreen_active(self, active: bool) -> None:
        self._fullscreen_btn.setChecked(bool(active))

    def _set_activity(self, text: str, *, active: bool) -> None:
        self._is_live = text == "LIVE"
        self._activity_text = text
        self._live_label.setText(f"● {text}" if active else f"○ {text}")
        self._live_label.setStyleSheet(
            "color: #2ecc71; font-weight: 700; font-size: 11px;"
            if active
            else "color: #5f6169; font-size: 11px;"
        )

    def _set_tools_enabled(self, enabled: bool) -> None:
        for button in self._tool_buttons:
            button.setEnabled(enabled)

    def _save_current_frame(self) -> None:
        pixmap = self._display.current_full_resolution_pixmap()
        if pixmap.isNull():
            return
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save Current Source Frame",
            "source-frame.png",
            "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;Bitmap (*.bmp)",
        )
        if not path:
            return

        output = Path(path)
        if not output.suffix:
            output = output.with_suffix(".jpg" if "JPEG" in selected_filter else ".png")
        if pixmap.save(str(output)):
            self.frame_saved.emit(str(output))

    @property
    def activity_text(self) -> str:
        return self._activity_text

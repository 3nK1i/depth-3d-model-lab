"""Original RGB source panel.

Phase 1 renders the panel chrome (title, live indicator, display-mode
toolbar) with a placeholder image area. Real frames are pushed in via
:meth:`set_frame` starting in Phase 2, once :mod:`core.capture` exists.
This panel must stay independent of inference — it keeps showing frames
even while inference is stopped or paused (spec §5).
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.widgets.icon_button import IconButton


class OriginalSourceView(QFrame):
    """Displays the raw, unprocessed source frame."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("panel")
        self._is_live = False
        self._activity_text = "IDLE"
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)
        layout.addLayout(self._build_title_row())
        layout.addWidget(self._build_display_area(), stretch=1)
        layout.addLayout(self._build_toolbar_row())

    def _build_title_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        dot = QLabel("●")
        dot.setStyleSheet("color: #e0293e; font-size: 9px;")
        title = QLabel("ORIGINAL SOURCE")
        title.setObjectName("panelTitle")
        row.addWidget(dot)
        row.addWidget(title)
        row.addStretch(1)

        self._live_label = QLabel("○ IDLE")
        self._live_label.setObjectName("muted")
        row.addWidget(self._live_label)
        return row

    def _build_display_area(self) -> QLabel:
        self._display = QLabel("No source connected")
        self._display.setObjectName("mediaSurface")
        self._display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._display.setMinimumHeight(160)
        self._display.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        return self._display

    def _build_toolbar_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addStretch(1)
        for icon_name, tooltip in (
            ("screenshot", "Screenshot"),
            ("capture", "Snapshot"),
            ("refresh", "Refresh"),
            ("fullscreen", "Fullscreen"),
            ("fit", "Fit"),
            ("grid", "Grid"),
        ):
            row.addWidget(IconButton(icon_name, size=28, icon_size=15, tooltip=tooltip))
        row.addStretch(1)
        return row

    def set_frame(self, pixmap: QPixmap) -> None:
        """Display a new RGB frame. Called by the source controller."""
        self._display.setPixmap(
            pixmap.scaled(
                self._display.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def clear_frame(self) -> None:
        self._display.clear()
        self._display.setText("No source connected")

    def set_live(self, is_live: bool) -> None:
        """Show LIVE for continuous sources, or IDLE when inactive."""
        if is_live:
            self._set_activity("LIVE", active=True)
        else:
            self._set_activity("IDLE", active=False)

    def set_ready(self) -> None:
        """Show READY for a loaded static image source."""
        self._set_activity("READY", active=True)

    def _set_activity(self, text: str, *, active: bool) -> None:
        self._is_live = text == "LIVE"
        self._activity_text = text
        self._live_label.setText(f"● {text}" if active else f"○ {text}")
        self._live_label.setStyleSheet(
            "color: #2ecc71; font-weight: 700; font-size: 11px;"
            if active
            else "color: #5f6169; font-size: 11px;"
        )

    @property
    def activity_text(self) -> str:
        return self._activity_text

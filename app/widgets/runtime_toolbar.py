"""Runtime action card: Connect/Disconnect, Start/Pause/Resume, Stop.

Matches the target: one grouped card, translucent tinted buttons (green
Start, amber Pause, red Stop) instead of solid opaque fills, and only
one of Connect/Disconnect visible at a time rather than both shown
simultaneously with one greyed out.

Phase 1 wires this to a minimal local enable/disable demo so the shell is
interactive and testable. The real enable/disable rules come from the
central application state machine in Phase 3 (spec §14) — at that point
:meth:`apply_button_states` becomes the single seam the controller calls,
and the ad-hoc toggling in this file's demo should be removed in favor
of it. The method's signature is unchanged from the previous version so
the Phase-1 demo lifecycle in main_window.py needed no changes.
"""

from __future__ import annotations

from enum import Enum

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QPushButton, QWidget

from app.icons import render_icon
from app.theme import PALETTE
from app.widgets.toolbar_group_card import ToolbarGroupCard

RUNTIME_BUTTON_HEIGHT = 34


class RuntimeButton(Enum):
    CONNECT = "connect"
    DISCONNECT = "disconnect"
    START = "start"
    PAUSE_RESUME = "pause_resume"
    STOP = "stop"


class RuntimeToolbar(QWidget):
    """Runtime action card: Disconnect + Start/Pause/Resume + Stop."""

    connect_clicked = pyqtSignal()
    disconnect_clicked = pyqtSignal()
    start_clicked = pyqtSignal()
    pause_resume_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._is_paused = False
        self._build_ui()
        # Phase-1 demo defaults: source not connected yet.
        self.apply_button_states(
            connect_enabled=True,
            disconnect_enabled=False,
            start_enabled=False,
            pause_resume_enabled=False,
            stop_enabled=False,
        )

    def _build_ui(self) -> None:
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        card = ToolbarGroupCard()
        row = card.content_layout
        row.setSpacing(9)

        self._connect_btn = QPushButton("Connect")
        self._connect_btn.setObjectName("toolbarDisconnectButton")
        self._connect_btn.setFixedHeight(RUNTIME_BUTTON_HEIGHT)
        self._connect_btn.setMinimumWidth(100)
        self._connect_btn.clicked.connect(self.connect_clicked.emit)

        self._disconnect_btn = QPushButton("Disconnect")
        self._disconnect_btn.setObjectName("toolbarDisconnectButton")
        self._disconnect_btn.setFixedHeight(RUNTIME_BUTTON_HEIGHT)
        self._disconnect_btn.setMinimumWidth(100)
        self._disconnect_btn.clicked.connect(self.disconnect_clicked.emit)

        self._start_btn = QPushButton("  Start Inference")
        self._start_btn.setObjectName("toolbarStartButton")
        self._start_btn.setFixedHeight(RUNTIME_BUTTON_HEIGHT)
        self._start_btn.setMinimumWidth(140)
        self._start_btn.setIcon(render_icon("play", PALETTE.toolbar_green_fg, 14))
        self._start_btn.clicked.connect(self.start_clicked.emit)

        self._pause_resume_btn = QPushButton("  Pause")
        self._pause_resume_btn.setObjectName("toolbarPauseButton")
        self._pause_resume_btn.setFixedHeight(RUNTIME_BUTTON_HEIGHT)
        self._pause_resume_btn.setMinimumWidth(90)
        self._pause_resume_btn.setIcon(render_icon("pause", PALETTE.toolbar_amber_fg, 14))
        self._pause_resume_btn.clicked.connect(self.pause_resume_clicked.emit)

        self._stop_btn = QPushButton("  Stop Inference")
        self._stop_btn.setObjectName("toolbarStopButton")
        self._stop_btn.setFixedHeight(RUNTIME_BUTTON_HEIGHT)
        self._stop_btn.setMinimumWidth(140)
        self._stop_btn.setIcon(render_icon("stop", PALETTE.toolbar_red_fg, 14))
        self._stop_btn.clicked.connect(self.stop_clicked.emit)

        row.addWidget(self._connect_btn)
        row.addWidget(self._disconnect_btn)
        row.addWidget(self._start_btn)
        row.addWidget(self._pause_resume_btn)
        row.addWidget(self._stop_btn)

        outer.addWidget(card)

    def apply_button_states(
        self,
        *,
        connect_enabled: bool,
        disconnect_enabled: bool,
        start_enabled: bool,
        pause_resume_enabled: bool,
        stop_enabled: bool,
    ) -> None:
        """Explicitly set every button's enabled/visible state.

        Only one of Connect/Disconnect is ever visible at once (spec
        §10): Disconnect shows exactly when ``disconnect_enabled`` is
        true, Connect otherwise. Centralizing this as one call (rather
        than toggling buttons individually all over the codebase) is the
        pattern the Phase 3 state machine controller will call into
        directly.
        """
        self._connect_btn.setVisible(not disconnect_enabled)
        self._connect_btn.setEnabled(connect_enabled)
        self._disconnect_btn.setVisible(disconnect_enabled)
        self._disconnect_btn.setEnabled(disconnect_enabled)
        self._start_btn.setEnabled(start_enabled)
        self._pause_resume_btn.setEnabled(pause_resume_enabled)
        self._stop_btn.setEnabled(stop_enabled)

    def set_paused_label(self, is_paused: bool) -> None:
        self._is_paused = is_paused
        if is_paused:
            self._pause_resume_btn.setText("  Resume")
            self._pause_resume_btn.setIcon(render_icon("play", PALETTE.toolbar_amber_fg, 14))
        else:
            self._pause_resume_btn.setText("  Pause")
            self._pause_resume_btn.setIcon(render_icon("pause", PALETTE.toolbar_amber_fg, 14))

    @property
    def is_paused(self) -> bool:
        return self._is_paused
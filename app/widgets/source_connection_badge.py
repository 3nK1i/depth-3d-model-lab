"""Compact connection-status badge for the source configuration card.

Supports every state the spec requires (DISCONNECTED / CONNECTING /
CONNECTED / RECONNECTING / ERROR) as a proper translucent-background
badge rather than plain colored text, matching the same visual language
as the performance footer's status badge.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QWidget

from app.theme import PALETTE
from app.widgets.source_toolbar_snapshot import ConnectionState, connection_label

_STYLE: dict[ConnectionState, tuple[str, str, str]] = {
    # state -> (bg, fg, border)
    ConnectionState.CONNECTED: (
        PALETTE.footer_status_green_bg,
        PALETTE.footer_status_green_fg,
        PALETTE.footer_status_green_border,
    ),
    ConnectionState.CONNECTING: (
        PALETTE.footer_status_amber_bg,
        PALETTE.footer_status_amber_fg,
        PALETTE.footer_status_amber_border,
    ),
    ConnectionState.RECONNECTING: (
        PALETTE.footer_status_amber_bg,
        PALETTE.footer_status_amber_fg,
        PALETTE.footer_status_amber_border,
    ),
    ConnectionState.ERROR: (
        PALETTE.footer_status_red_bg,
        PALETTE.footer_status_red_fg,
        PALETTE.footer_status_red_border,
    ),
    ConnectionState.DISCONNECTED: (
        PALETTE.footer_status_neutral_bg,
        PALETTE.footer_status_neutral_fg,
        PALETTE.footer_status_neutral_border,
    ),
}


class SourceConnectionBadge(QLabel):
    """Small pill badge showing the current source connection state."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("sourceConnectionBadge")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFixedHeight(24)
        self.set_state(ConnectionState.DISCONNECTED)

    def set_state(self, state: ConnectionState) -> None:
        bg, fg, border = _STYLE[state]
        self.setText(connection_label(state))
        self.setStyleSheet(
            f"background-color: {bg}; color: {fg}; border: 1px solid {border};"
        )
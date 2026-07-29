"""Reusable grouped card for the source/runtime toolbar row.

Each of the three toolbar sections (source type, source configuration,
runtime actions) is its own independent bordered card rather than one
continuous bar — this widget is the shared shell so all three carry
identical height, padding and background (spec §1, §21).
"""

from __future__ import annotations

from PyQt6.QtWidgets import QFrame, QHBoxLayout, QSizePolicy, QWidget

TOOLBAR_CARD_HEIGHT = 50


class ToolbarGroupCard(QFrame):
    """A single dark bordered card holding one toolbar section's controls."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("toolbarGroupCard")
        self.setFixedHeight(TOOLBAR_CARD_HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(9, 7, 9, 7)
        self._layout.setSpacing(8)

    @property
    def content_layout(self) -> QHBoxLayout:
        return self._layout
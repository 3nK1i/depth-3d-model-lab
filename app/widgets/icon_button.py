"""Reusable vector-icon button.

Wraps :func:`app.icons.render_icon` and swaps the rendered icon color on
hover / checked / disabled so a single widget handles every icon-button
state without duplicating QPainter logic across the codebase.
"""

from __future__ import annotations

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QEnterEvent
from PyQt6.QtWidgets import QPushButton, QWidget

from app.icons import render_icon
from app.theme import PALETTE


class IconButton(QPushButton):
    """A square, borderless button showing a single vector icon."""

    def __init__(
        self,
        icon_name: str,
        *,
        size: int = 30,
        icon_size: int = 16,
        color: str = PALETTE.text_secondary,
        hover_color: str = PALETTE.text_primary,
        checked_color: str = PALETTE.accent,
        disabled_color: str = PALETTE.text_muted,
        tooltip: str = "",
        checkable: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._icon_name = icon_name
        self._icon_px = icon_size
        self._color = color
        self._hover_color = hover_color
        self._checked_color = checked_color
        self._disabled_color = disabled_color
        self._hovered = False

        self.setObjectName("iconButton")
        self.setFixedSize(size, size)
        self.setIconSize(QSize(icon_size, icon_size))
        self.setCheckable(checkable)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        if tooltip:
            self.setToolTip(tooltip)

        self.toggled.connect(lambda _checked: self._refresh_icon())
        self._refresh_icon()

    def _current_color(self) -> str:
        if not self.isEnabled():
            return self._disabled_color
        if self.isCheckable() and self.isChecked():
            return self._checked_color
        if self._hovered:
            return self._hover_color
        return self._color

    def _refresh_icon(self) -> None:
        self.setIcon(render_icon(self._icon_name, self._current_color(), self._icon_px))

    def set_icon_name(self, icon_name: str) -> None:
        self._icon_name = icon_name
        self._refresh_icon()

    def enterEvent(self, event: QEnterEvent) -> None:  # noqa: N802 - Qt override
        self._hovered = True
        self._refresh_icon()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt override
        self._hovered = False
        self._refresh_icon()
        super().leaveEvent(event)

    def changeEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().changeEvent(event)
        self._refresh_icon()
"""Small colored badge label, used for status and compatibility indicators."""

from __future__ import annotations

from enum import Enum

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel

from app.theme import PALETTE, badge_style


class BadgeKind(Enum):
    """Semantic badge kinds mapped to a background/foreground color pair."""

    READY = "ready"
    LOADING = "loading"
    ERROR = "error"
    RECOMMENDED = "recommended"
    COMPATIBLE = "compatible"
    EXPERIMENTAL = "experimental"
    SNAPSHOT_ONLY = "snapshot_only"
    RECONSTRUCTION_ONLY = "reconstruction_only"
    NOT_RECOMMENDED = "not_recommended"
    NOT_INSTALLED = "not_installed"
    # Flat (no chip background) status text, e.g. the source connection
    # badge sitting directly on the source bar.
    CONNECTED_FLAT = "connected_flat"
    DISCONNECTED_FLAT = "disconnected_flat"


_STYLE_MAP: dict[BadgeKind, tuple[str, str]] = {
    BadgeKind.READY: (PALETTE.badge_recommended_bg, PALETTE.badge_recommended_fg),
    BadgeKind.LOADING: (PALETTE.badge_experimental_bg, PALETTE.badge_experimental_fg),
    BadgeKind.ERROR: (PALETTE.badge_not_recommended_bg, PALETTE.badge_not_recommended_fg),
    BadgeKind.RECOMMENDED: (PALETTE.badge_recommended_bg, PALETTE.badge_recommended_fg),
    BadgeKind.COMPATIBLE: (PALETTE.badge_compatible_bg, PALETTE.badge_compatible_fg),
    BadgeKind.EXPERIMENTAL: (PALETTE.badge_experimental_bg, PALETTE.badge_experimental_fg),
    BadgeKind.SNAPSHOT_ONLY: (PALETTE.badge_snapshot_only_bg, PALETTE.badge_snapshot_only_fg),
    BadgeKind.RECONSTRUCTION_ONLY: (
        PALETTE.badge_reconstruction_only_bg,
        PALETTE.badge_reconstruction_only_fg,
    ),
    BadgeKind.NOT_RECOMMENDED: (
        PALETTE.badge_not_recommended_bg,
        PALETTE.badge_not_recommended_fg,
    ),
    BadgeKind.NOT_INSTALLED: (PALETTE.badge_not_installed_bg, PALETTE.badge_not_installed_fg),
    BadgeKind.CONNECTED_FLAT: ("transparent", PALETTE.badge_recommended_fg),
    BadgeKind.DISCONNECTED_FLAT: ("transparent", PALETTE.badge_not_recommended_fg),
}

_TEXT_MAP: dict[BadgeKind, str] = {
    BadgeKind.READY: "READY",
    BadgeKind.LOADING: "LOADING",
    BadgeKind.ERROR: "ERROR",
    BadgeKind.RECOMMENDED: "RECOMMENDED",
    BadgeKind.COMPATIBLE: "COMPATIBLE",
    BadgeKind.EXPERIMENTAL: "EXPERIMENTAL",
    BadgeKind.SNAPSHOT_ONLY: "SNAPSHOT ONLY",
    BadgeKind.RECONSTRUCTION_ONLY: "RECONSTRUCTION ONLY",
    BadgeKind.NOT_RECOMMENDED: "NOT RECOMMENDED",
    BadgeKind.NOT_INSTALLED: "NOT INSTALLED",
    BadgeKind.CONNECTED_FLAT: "CONNECTED",
    BadgeKind.DISCONNECTED_FLAT: "DISCONNECTED",
}


class BadgeLabel(QLabel):
    """A small, pill-shaped status label (e.g. READY, RECOMMENDED)."""

    def __init__(self, kind: BadgeKind = BadgeKind.READY, text: str | None = None) -> None:
        super().__init__(text or _TEXT_MAP[kind])
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._kind = kind
        self.set_kind(kind, text)

    def set_kind(self, kind: BadgeKind, text: str | None = None) -> None:
        """Update the badge's semantic kind (and optionally its text)."""
        self._kind = kind
        bg, fg = _STYLE_MAP[kind]
        self.setStyleSheet(badge_style(bg, fg))
        self.setText(text or _TEXT_MAP[kind])

    @property
    def kind(self) -> BadgeKind:
        return self._kind
"""Primary navigation: Live 3D / Measure & Inspect / Stable Video / Snapshot / Reconstruction / Benchmark.

Phase 8.8B5 enables the already-built Reconstruction workspace through a
deliberate additive gate while preserving the legacy operational-mode set.
"""

from __future__ import annotations

from enum import Enum

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QPushButton, QWidget

from app.theme import PALETTE


class AppMode(Enum):
    LIVE_3D = "live_3d"
    INSPECTION = "inspection"
    STABLE_VIDEO = "stable_video"
    SNAPSHOT = "snapshot"
    RECONSTRUCTION = "reconstruction"
    BENCHMARK = "benchmark"


_MODE_LABELS: dict[AppMode, str] = {
    AppMode.LIVE_3D: "◈  LIVE 3D",
    AppMode.INSPECTION: "⌖  MEASURE & INSPECT",
    AppMode.STABLE_VIDEO: "▷  STABLE VIDEO",
    AppMode.SNAPSHOT: "◫  SNAPSHOT",
    AppMode.RECONSTRUCTION: "⛶  RECONSTRUCTION",
    AppMode.BENCHMARK: "▤  BENCHMARK",
}

# Stable pre-8.8 operational set. Keep this explicit because many historical
# backward-compatibility gates verify that these five modes remain available.
_ENABLED_MODES = {AppMode.LIVE_3D, AppMode.INSPECTION, AppMode.STABLE_VIDEO, AppMode.SNAPSHOT, AppMode.BENCHMARK}

# Phase 8.8B5 deliberately enables Reconstruction without rewriting the legacy
# mode contract above. This makes the rollout explicit and easy to audit.
_RECONSTRUCTION_NAVIGATION_ENABLED = True


def _mode_is_enabled(mode: AppMode) -> bool:
    return mode in _ENABLED_MODES or (
        _RECONSTRUCTION_NAVIGATION_ENABLED and mode is AppMode.RECONSTRUCTION
    )


class ModeTabs(QFrame):
    """Horizontal row of primary navigation buttons."""

    mode_changed = pyqtSignal(AppMode)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("modeTabs")
        self._buttons: dict[AppMode, QPushButton] = {}
        self._active_mode = AppMode.LIVE_3D
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 6, 16, 0)
        layout.setSpacing(4)

        for mode in AppMode:
            btn = QPushButton(_MODE_LABELS[mode])
            btn.setCheckable(True)
            btn.setFlat(True)
            btn.setStyleSheet(self._tab_button_qss())
            enabled = _mode_is_enabled(mode)
            btn.setEnabled(enabled)
            if not enabled:
                btn.setToolTip("Not yet implemented in this milestone")
            btn.clicked.connect(lambda _checked, m=mode: self._on_clicked(m))
            self._buttons[mode] = btn
            layout.addWidget(btn)

        layout.addStretch(1)
        self._buttons[self._active_mode].setChecked(True)

    @staticmethod
    def _tab_button_qss() -> str:
        return f"""
            QPushButton {{
                background: transparent;
                border: none;
                border-bottom: 2px solid transparent;
                padding: 10px 16px;
                color: {PALETTE.text_secondary};
                font-weight: 700;
                font-size: 12px;
                letter-spacing: 0.3px;
            }}
            QPushButton:checked {{
                color: {PALETTE.accent};
                border-bottom: 2px solid {PALETTE.accent};
            }}
            QPushButton:disabled {{
                color: {PALETTE.text_muted};
            }}
            QPushButton:hover:!disabled:!checked {{
                color: {PALETTE.text_primary};
            }}
        """

    def _on_clicked(self, mode: AppMode) -> None:
        if mode == self._active_mode:
            self._buttons[mode].setChecked(True)
            return
        for m, btn in self._buttons.items():
            btn.setChecked(m == mode)
        self._active_mode = mode
        self.mode_changed.emit(mode)


    def set_navigation_locked(self, locked: bool) -> None:
        """Keep the active mode available while a long-running mode owns the UI."""
        for mode, button in self._buttons.items():
            base_enabled = _mode_is_enabled(mode)
            button.setEnabled(base_enabled and (not locked or mode is self._active_mode))

    def set_active_mode(self, mode: AppMode) -> None:
        """Select an enabled mode programmatically without emitting twice."""
        if not _mode_is_enabled(mode):
            raise ValueError(f"Mode is not enabled: {mode.value}")
        if mode is self._active_mode:
            self._buttons[mode].setChecked(True)
            return
        for item, button in self._buttons.items():
            button.setChecked(item is mode)
        self._active_mode = mode
        self.mode_changed.emit(mode)

    @property
    def active_mode(self) -> AppMode:
        return self._active_mode

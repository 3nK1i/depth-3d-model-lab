"""Compact progressive-disclosure group for the Reconstruction settings sidebar.

Phase 8.8G5 keeps every existing Reconstruction control and authority statement
available while reducing default vertical density.  The component is purely a
presentation container: expanding or collapsing a group never mutates
reconstruction state, controller state, persistence, or backend execution.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QToolButton, QVBoxLayout, QWidget


class ReconstructionSettingsDisclosureGroup(QFrame):
    """Settings group with a compact chevron-controlled content body."""

    expansion_changed = pyqtSignal(bool)

    def __init__(
        self,
        title: str,
        *,
        expanded: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._title = str(title).strip()
        if not self._title:
            raise ValueError("Reconstruction settings disclosure title must be non-empty.")
        self._expanded = bool(expanded)
        self.setObjectName("reconstructionSettingsGroup")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 9, 10, 9)
        outer.setSpacing(7)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(6)

        self._title_label = QLabel(self._title)
        self._title_label.setObjectName("reconstructionSettingsGroupTitle")
        header.addWidget(self._title_label)
        header.addStretch(1)

        self._toggle_button = QToolButton()
        self._toggle_button.setObjectName("reconstructionSettingsDisclosureToggle")
        self._toggle_button.setCheckable(True)
        self._toggle_button.setChecked(self._expanded)
        self._toggle_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._toggle_button.setFixedSize(20, 18)
        self._toggle_button.toggled.connect(self._on_toggled)
        header.addWidget(self._toggle_button)
        outer.addLayout(header)

        self._body = QWidget()
        self._body.setObjectName("reconstructionSettingsDisclosureBody")
        self._content_layout = QVBoxLayout(self._body)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(7)
        self._body.setVisible(self._expanded)
        outer.addWidget(self._body)

        self._refresh_toggle()

    @property
    def content_layout(self) -> QVBoxLayout:
        return self._content_layout

    @property
    def expanded(self) -> bool:
        return self._expanded

    @property
    def title(self) -> str:
        return self._title

    def set_expanded(self, expanded: bool) -> None:
        requested = bool(expanded)
        if self._toggle_button.isChecked() == requested:
            # Keep the body synchronized even if an external test or stylesheet
            # changed visibility directly.
            self._expanded = requested
            self._body.setVisible(requested)
            self._refresh_toggle()
            self.updateGeometry()
            return
        self._toggle_button.setChecked(requested)

    def _on_toggled(self, expanded: bool) -> None:
        self._expanded = bool(expanded)
        self._body.setVisible(self._expanded)
        self._refresh_toggle()
        self.updateGeometry()
        self.expansion_changed.emit(self._expanded)

    def _refresh_toggle(self) -> None:
        self._toggle_button.setText("▾" if self._expanded else "▸")
        action = "Collapse" if self._expanded else "Expand"
        self._toggle_button.setToolTip(f"{action} {self._title.lower()} details")
        self._toggle_button.setAccessibleName(f"{action} {self._title} details")

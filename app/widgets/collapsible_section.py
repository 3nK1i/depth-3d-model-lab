"""A titled, collapsible content section (spec §16: "use collapsible sections")."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFrame, QPushButton, QVBoxLayout, QWidget


class CollapsibleSection(QFrame):
    """A section with a clickable header that shows/hides its body."""

    def __init__(self, title: str, expanded: bool = True, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._expanded = expanded
        self._build_ui(title)

    def _build_ui(self, title: str) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 4, 0, 4)
        outer.setSpacing(4)

        self._toggle_btn = QPushButton(f"{'▾' if self._expanded else '▸'}  {title}")
        self._toggle_btn.setCheckable(True)
        self._toggle_btn.setChecked(self._expanded)
        self._toggle_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._toggle_btn.setStyleSheet(
            "text-align: left; background: transparent; border: none; "
            "color: #e0293e; font-weight: 700; font-size: 11px; "
            "letter-spacing: 0.5px; padding: 4px 0;"
        )
        self._toggle_btn.clicked.connect(self._on_toggled)
        outer.addWidget(self._toggle_btn)

        self._title = title
        self._body = QWidget()
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(4, 0, 4, 4)
        self._body_layout.setSpacing(8)
        self._body.setVisible(self._expanded)
        outer.addWidget(self._body)

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet("color: #2a2b30;")
        outer.addWidget(divider)

    def _on_toggled(self) -> None:
        self._expanded = self._toggle_btn.isChecked()
        self._body.setVisible(self._expanded)
        self._toggle_btn.setText(f"{'▾' if self._expanded else '▸'}  {self._title}")

    def add_widget(self, widget: QWidget) -> None:
        self._body_layout.addWidget(widget)

    def add_layout(self, layout) -> None:  # noqa: ANN001 - QLayout has no shared base import here
        self._body_layout.addLayout(layout)

    @property
    def expanded(self) -> bool:
        return self._expanded

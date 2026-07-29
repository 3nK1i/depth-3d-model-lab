"""A single label + control row for the settings panel.

Uniform two-column layout: a fixed-width caption on the left and the
control filling the rest. Keeps every dropdown/toggle row aligned and,
crucially, lets the control shrink instead of overflowing the panel
(the cause of the earlier right-edge clipping).
"""

from __future__ import annotations

from PyQt6.QtWidgets import QHBoxLayout, QLabel, QWidget

LABEL_WIDTH = 116


class SettingRow(QWidget):
    """Caption + arbitrary control, laid out as one aligned row."""

    def __init__(
        self,
        label_text: str,
        control: QWidget,
        *,
        label_width: int = LABEL_WIDTH,
        stretch_control: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        label = QLabel(label_text)
        label.setObjectName("settingLabel")
        label.setFixedWidth(label_width)
        row.addWidget(label)

        if stretch_control:
            row.addWidget(control, stretch=1)
        else:
            row.addWidget(control)
            row.addStretch(1)
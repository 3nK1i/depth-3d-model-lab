"""Independent, content-sized accordion card for the settings sidebar.

Each card is its own surface: background, border, radius, padding and a
compact header with a single chevron on the right that flips between
up (expanded) and down (collapsed). Cards never expand vertically — the
sidebar's stretch lives *after* the last card, so unused space stays
outside them.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.icons import render_icon
from app.theme import PALETTE
from app.widgets.settings_controls import (
    COL_HSPACING,
    ROW_VSPACING,
    VALUE_BOX_WIDTH,
    RowLabel,
)

CARD_PADDING_H = 15
CARD_PADDING_V = 7


class _CardHeader(QWidget):
    """Compact header: optional red accent chevron, light title, toggle chevron."""

    toggled = pyqtSignal(bool)

    def __init__(self, title: str, expanded: bool, accent_marker: bool) -> None:
        super().__init__()
        self._expanded = expanded

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(7)

        if accent_marker:
            marker = QLabel("▾")
            marker.setStyleSheet(
                f"color: {PALETTE.accent}; font-size: 10px; background: transparent;"
            )
            row.addWidget(marker)

        self._title = QLabel(title)
        self._title.setObjectName("settingsCardTitle")
        row.addWidget(self._title)
        row.addStretch(1)

        self._chevron = QPushButton()
        self._chevron.setObjectName("cardChevron")
        self._chevron.setFixedSize(20, 20)
        self._chevron.setCursor(Qt.CursorShape.PointingHandCursor)
        self._chevron.clicked.connect(self._on_clicked)
        row.addWidget(self._chevron)

        self._refresh_chevron()

    def _refresh_chevron(self) -> None:
        name = "chevron_up" if self._expanded else "chevron_down"
        self._chevron.setIcon(render_icon(name, PALETTE.settings_text_secondary, 13))

    def _on_clicked(self) -> None:
        self._expanded = not self._expanded
        self._refresh_chevron()
        self.toggled.emit(self._expanded)

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt override
        self._on_clicked()
        super().mousePressEvent(event)


class CollapsibleSettingsCard(QFrame):
    """A self-contained settings card that sizes itself to its content."""

    expansion_changed = pyqtSignal(bool)

    def __init__(
        self,
        title: str,
        *,
        expanded: bool = True,
        accent_marker: bool = False,
        uppercase_labels: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("settingsCard")
        # Expanding (not Maximum): when the sidebar has more height than the
        # cards' combined sizeHint, each expanded card grows into its
        # stretch share (set by SettingsSidebar) instead of leftover space
        # collecting below the last card.
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        self._uppercase = uppercase_labels

        outer = QVBoxLayout(self)
        outer.setContentsMargins(CARD_PADDING_H, CARD_PADDING_V, CARD_PADDING_H, CARD_PADDING_V)
        outer.setSpacing(6)

        self._header = _CardHeader(title, expanded, accent_marker)
        self._header.toggled.connect(self._on_toggled)
        outer.addWidget(self._header)

        self._body = QWidget()
        self._body.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        self._grid = QGridLayout(self._body)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(COL_HSPACING)
        self._grid.setVerticalSpacing(ROW_VSPACING)
        # col0 label | col1 slider/filler | col2 fixed value box
        # Labels get less share than controls so long combo texts
        # ("MoGe-2 ViT-L Normal", "Auto (RTX 5090)") are never elided.
        self._grid.setColumnStretch(0, 4)
        self._grid.setColumnStretch(1, 6)
        self._grid.setColumnStretch(2, 0)
        self._grid.setColumnMinimumWidth(2, VALUE_BOX_WIDTH)
        self._body.setVisible(expanded)
        outer.addWidget(self._body, stretch=1)

        self._row = 0

    def _on_toggled(self, expanded: bool) -> None:
        self._body.setVisible(expanded)
        self.updateGeometry()
        self.expansion_changed.emit(expanded)

    def _mark_row_stretch(self) -> None:
        """Give the just-added row a stretch share.

        Every data row gets equal stretch (rather than one blank row at
        the end), so when the card is taller than its content needs, the
        extra height becomes balanced spacing *between* rows instead of
        one dead gap at the bottom.
        """
        self._grid.setRowStretch(self._row, 1)

    # -- Row builders ------------------------------------------------------

    def add_wide_row(self, left: QWidget, right: QWidget) -> None:
        """A row where ``left`` takes most of the width and ``right`` stays
        compact beside it — used for the model selector + badge."""
        self._grid.addWidget(left, self._row, 0, 1, 2, Qt.AlignmentFlag.AlignVCenter)
        self._grid.addWidget(
            right, self._row, 2, 1, 1, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self._mark_row_stretch()
        self._row += 1

    def add_control_row(self, label_text: str, control: QWidget) -> None:
        """Label in col 0; control spans cols 1-2 to share the right edge."""
        self._grid.addWidget(
            RowLabel(label_text, uppercase=self._uppercase),
            self._row,
            0,
            Qt.AlignmentFlag.AlignVCenter,
        )
        self._grid.addWidget(control, self._row, 1, 1, 2, Qt.AlignmentFlag.AlignVCenter)
        self._mark_row_stretch()
        self._row += 1

    def add_slider_row(self, label_text: str, slider: QWidget, value_box: QWidget) -> None:
        """Three-column row: Label | Slider | boxed value."""
        self._grid.addWidget(
            RowLabel(label_text, uppercase=self._uppercase),
            self._row,
            0,
            Qt.AlignmentFlag.AlignVCenter,
        )
        self._grid.addWidget(slider, self._row, 1, Qt.AlignmentFlag.AlignVCenter)
        self._grid.addWidget(
            value_box,
            self._row,
            2,
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
        )
        self._mark_row_stretch()
        self._row += 1

    def add_toggle_row(self, label_text: str, toggle: QWidget) -> None:
        """Toggle pinned to the same right edge as the value boxes."""
        self._grid.addWidget(
            RowLabel(label_text, uppercase=self._uppercase),
            self._row,
            0,
            1,
            2,
            Qt.AlignmentFlag.AlignVCenter,
        )
        self._grid.addWidget(
            toggle,
            self._row,
            2,
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
        )
        self._mark_row_stretch()
        self._row += 1
        self._row += 1
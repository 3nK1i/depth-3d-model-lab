"""Parent composition: the three grouped cards on one toolbar row.

    SourceRuntimeBar
    └── QHBoxLayout (spacing=10, zero margins)
        ├── SourceBar          (source-type card + source-config card)
        └── RuntimeToolbar     (runtime-action card)

No outer border wraps all three — that was the single-continuous-bar
bug this file replaces. The page background shows through the 10px gaps
between cards (spec §1). Stretch factors approximate the target's width
distribution (source ~64% combined, runtime ~36%) without a manual
spacer, so the runtime card naturally lands at the row's right edge.
"""

from __future__ import annotations

from PyQt6.QtWidgets import QHBoxLayout, QWidget

from app.widgets.runtime_toolbar import RuntimeToolbar
from app.widgets.source_bar import SourceBar

# SourceBar already splits its own share into type/config cards
# internally (53:47) — this outer ratio is what fixes their combined
# share against the runtime card to roughly the spec's 64:36 target.
_SOURCE_STRETCH = 64
_RUNTIME_STRETCH = 36


class SourceRuntimeBar(QWidget):
    """Composes :class:`SourceBar` and :class:`RuntimeToolbar` as siblings."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.source_bar = SourceBar()
        self.runtime_toolbar = RuntimeToolbar()

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.addWidget(self.source_bar, _SOURCE_STRETCH)
        layout.addWidget(self.runtime_toolbar, _RUNTIME_STRETCH)
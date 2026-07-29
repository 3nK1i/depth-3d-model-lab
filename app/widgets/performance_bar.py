"""Bottom performance/status bar.

Shows placeholder dashes in Phase 1 — spec §17 is explicit that displayed
values must be measured, never fabricated, so this widget only exposes
setter methods for a future performance monitor to call; it does not
generate any numbers itself.
"""

from __future__ import annotations

from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from app.theme import FOOTER_HEIGHT
from app.widgets.badge import BadgeKind, BadgeLabel

_PLACEHOLDER = "—"


class PerformanceBar(QFrame):
    """Single-row status/performance footer."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("panel")
        self.setFixedHeight(FOOTER_HEIGHT)
        self._value_labels: dict[str, QLabel] = {}
        self._build_ui()

    def _metric(self, key: str, label_text: str) -> QVBoxLayout:
        col = QVBoxLayout()
        col.setSpacing(0)
        label = QLabel(label_text)
        label.setObjectName("metricLabel")
        value = QLabel(_PLACEHOLDER)
        value.setObjectName("metricValue")
        self._value_labels[key] = value
        col.addWidget(label)
        col.addWidget(value)
        return col

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 0, 14, 0)
        layout.setSpacing(22)

        status_col = QVBoxLayout()
        status_col.setSpacing(0)
        status_label = QLabel("STATUS")
        status_label.setObjectName("metricLabel")
        self._status_badge = BadgeLabel(BadgeKind.NOT_INSTALLED, "IDLE")
        status_col.addWidget(status_label)
        status_col.addWidget(self._status_badge)
        layout.addLayout(status_col)

        for key, label_text in (
            ("cam_fps", "CAM FPS"),
            ("inf_fps", "INF FPS"),
            ("latency", "LATENCY"),
            ("render_fps", "RENDER FPS"),
            ("points", "POINTS"),
            ("resolution", "RESOLUTION"),
            ("vram", "VRAM USAGE"),
            ("model", "MODEL"),
            ("precision", "PRECISION"),
            ("uptime", "UPTIME"),
        ):
            layout.addLayout(self._metric(key, label_text))

        layout.addStretch(1)

    def set_status(self, kind: BadgeKind, text: str) -> None:
        self._status_badge.set_kind(kind, text)

    def set_metric(self, key: str, value: str) -> None:
        """Set a single metric's displayed text. Unknown keys are ignored."""
        label = self._value_labels.get(key)
        if label is not None:
            label.setText(value)

    def reset_metrics(self) -> None:
        for label in self._value_labels.values():
            label.setText(_PLACEHOLDER)
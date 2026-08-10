"""Compact viewport tracking/map diagnostics HUD for Phase 8.8C5."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from app.reconstruction_diagnostics import LiveTrackingDiagnosticsPresentation


class ReconstructionDiagnosticsHud(QFrame):
    """Screen-space presentation-only live diagnostics overlay."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("reconstructionDiagnosticsHud")
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setMinimumWidth(255)
        self.setMaximumWidth(330)
        self._presentation = LiveTrackingDiagnosticsPresentation()
        self._build_ui()
        self.set_presentation(self._presentation)

    @property
    def presentation(self) -> LiveTrackingDiagnosticsPresentation:
        return self._presentation

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(9, 7, 9, 7)
        layout.setSpacing(4)

        header = QHBoxLayout()
        header.setSpacing(8)
        title = QLabel("TRACKING DIAGNOSTICS")
        title.setObjectName("reconstructionDiagnosticsTitle")
        header.addWidget(title)
        header.addStretch(1)
        self._tracking_badge = QLabel("WAITING")
        self._tracking_badge.setObjectName("reconstructionDiagnosticsBadge")
        self._tracking_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.addWidget(self._tracking_badge)
        layout.addLayout(header)

        self._latency_value = self._row(layout, "Frame", "Waiting for samples")
        self._map_value = self._row(layout, "Map", "No live map yet")
        self._growth_value = self._row(layout, "Growth", "—")
        self._features_value = self._row(layout, "Features", "Not reported by Stella")

    @staticmethod
    def _row(layout: QVBoxLayout, label_text: str, value_text: str) -> QLabel:
        row = QHBoxLayout()
        row.setSpacing(8)
        label = QLabel(label_text)
        label.setObjectName("reconstructionDiagnosticsLabel")
        row.addWidget(label)
        row.addStretch(1)
        value = QLabel(value_text)
        value.setObjectName("reconstructionDiagnosticsValue")
        value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(value)
        layout.addLayout(row)
        return value

    def set_presentation(self, presentation: LiveTrackingDiagnosticsPresentation) -> None:
        if not isinstance(presentation, LiveTrackingDiagnosticsPresentation):
            raise TypeError("presentation must be LiveTrackingDiagnosticsPresentation.")
        self._presentation = presentation
        self._tracking_badge.setText(presentation.display_tracking_label)
        self._tracking_badge.setProperty("statusKind", presentation.status_kind)
        self._refresh_style(self._tracking_badge)
        self._latency_value.setText(presentation.latency_text)
        self._map_value.setText(presentation.map_text)
        self._growth_value.setText(presentation.map_growth_text)
        self._features_value.setText(presentation.feature_text)
        tooltip = presentation.reason or "Live diagnostics are presentation-only SLAM telemetry."
        self.setToolTip(tooltip)

    @staticmethod
    def _refresh_style(widget: QWidget) -> None:
        style = widget.style()
        style.unpolish(widget)
        style.polish(widget)
        widget.update()

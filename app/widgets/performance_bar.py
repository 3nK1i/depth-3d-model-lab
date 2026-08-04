"""Performance/status footer: eleven metric regions spanning the full width.

One continuous card (spec: no separate rounded card per metric), divided
by thin vertical separators, with explicit stretch proportions on every
cell so the whole footer width is used and nothing collects on the left
with a dead region on the right — the bug this rework fixes.

Reusable pieces: :class:`MetricCell` (label-over-value), a specialised
:class:`StatusMetricCell` (label + colored state badge) and
:class:`VramMetricCell` (value + compact progress bar), plus
:class:`ModelMetricCell` (elidable name + info button). All read from one
:class:`PerformanceSnapshot` via :meth:`PerformanceBar.update_snapshot`,
so the widget never needs to know whether the data is Phase 1 preview
data or a real runtime monitor's numbers.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QMessageBox,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.icons import render_icon
from app.theme import FOOTER_HEIGHT, PALETTE
from app.widgets.badge import BadgeKind
from app.widgets.performance_snapshot import (
    PLACEHOLDER,
    PerformanceSnapshot,
    format_fps,
    format_latency,
    format_or_dash,
    format_points,
    format_resolution,
    format_uptime,
    format_vram,
    status_color,
    vram_ratio,
)

# Stretch proportions across the eleven regions, summing to 100 (spec §4).
_STRETCH = {
    "status": 11,
    "cam_fps": 7,
    "inf_fps": 7,
    "latency": 7,
    "render_fps": 7,
    "points": 7,
    "resolution": 8,
    "vram": 15,
    "model": 13,
    "precision": 7,
    "uptime": 11,
}

_STATUS_STYLE: dict[str, tuple[str, str, str]] = {
    # color role -> (bg, fg, border)
    "green": (
        PALETTE.footer_status_green_bg,
        PALETTE.footer_status_green_fg,
        PALETTE.footer_status_green_border,
    ),
    "amber": (
        PALETTE.footer_status_amber_bg,
        PALETTE.footer_status_amber_fg,
        PALETTE.footer_status_amber_border,
    ),
    "red": (
        PALETTE.footer_status_red_bg,
        PALETTE.footer_status_red_fg,
        PALETTE.footer_status_red_border,
    ),
    "neutral": (
        PALETTE.footer_status_neutral_bg,
        PALETTE.footer_status_neutral_fg,
        PALETTE.footer_status_neutral_border,
    ),
}

# Backward-compatible shim: earlier callers (the Phase-1 demo lifecycle in
# main_window.py) pass a header/settings BadgeKind + free text rather than
# the footer's own status vocabulary. Map each kind to a color role so
# that call site does not need to change.
_BADGE_KIND_TO_COLOR: dict[BadgeKind, str] = {
    BadgeKind.RECOMMENDED: "green",
    BadgeKind.READY: "green",
    BadgeKind.EXPERIMENTAL: "amber",
    BadgeKind.NOT_RECOMMENDED: "red",
    BadgeKind.ERROR: "red",
    BadgeKind.COMPATIBLE: "neutral",
    BadgeKind.NOT_INSTALLED: "neutral",
    BadgeKind.LOADING: "neutral",
    BadgeKind.SNAPSHOT_ONLY: "neutral",
    BadgeKind.RECONSTRUCTION_ONLY: "neutral",
}


class _VSeparator(QFrame):
    """Thin vertical line between metric regions."""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("footerSeparator")
        self.setFixedWidth(1)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)


class MetricCell(QWidget):
    """Label-over-value metric region (CAM FPS, LATENCY, POINTS, ...)."""

    def __init__(self, label_text: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        col = QVBoxLayout(self)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(2)

        label = QLabel(label_text)
        label.setObjectName("footerCellLabel")
        self._value = QLabel(PLACEHOLDER)
        self._value.setObjectName("footerCellValue")

        col.addStretch(1)
        col.addWidget(label)
        col.addWidget(self._value)
        col.addStretch(1)

    def set_value(self, text: str, *, positive: bool = False, muted: bool = False) -> None:
        self._value.setText(text)
        if positive:
            object_name = "footerCellValuePositive"
        elif muted:
            object_name = "footerCellValueMuted"
        else:
            object_name = "footerCellValue"
        if self._value.objectName() != object_name:
            self._value.setObjectName(object_name)
            self._value.style().unpolish(self._value)
            self._value.style().polish(self._value)

    def set_tooltip(self, text: str) -> None:
        self.setToolTip(text)
        self._value.setToolTip(text)


class StatusMetricCell(QWidget):
    """STATUS label + compact colored state badge, on one row."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        col = QVBoxLayout(self)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(2)

        label = QLabel("STATUS")
        label.setObjectName("footerCellLabel")

        self._badge = QLabel("IDLE")
        self._badge.setObjectName("footerStatusBadge")
        self._badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._badge.setFixedHeight(22)

        badge_row = QHBoxLayout()
        badge_row.setContentsMargins(0, 0, 0, 0)
        badge_row.addWidget(self._badge)
        badge_row.addStretch(1)

        col.addStretch(1)
        col.addWidget(label)
        col.addLayout(badge_row)
        col.addStretch(1)

        self.set_state("IDLE")

    def set_state(self, status_text: str, color_role: str | None = None) -> None:
        role = color_role or status_color(status_text)
        bg, fg, border = _STATUS_STYLE.get(role, _STATUS_STYLE["neutral"])
        self._badge.setText(status_text.upper())
        self._badge.setStyleSheet(
            f"background-color: {bg}; color: {fg}; border: 1px solid {border};"
        )

    def set_tooltip(self, text: str) -> None:
        self.setToolTip(text)
        self._badge.setToolTip(text)


class CompactProgressBar(QWidget):
    """A thin rounded track with a green fill — no native QProgressBar frame."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._ratio = 0.0
        self.setFixedHeight(5)
        self.setMinimumWidth(40)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_ratio(self, ratio: float) -> None:
        self._ratio = max(0.0, min(1.0, ratio))
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(Qt.PenStyle.NoPen)

        radius = self.height() / 2.0
        painter.setBrush(QColor(PALETTE.footer_track_bg))
        painter.drawRoundedRect(self.rect(), radius, radius)

        if self._ratio > 0:
            fill_width = max(self.height(), self.width() * self._ratio)
            painter.setBrush(QColor(PALETTE.footer_value_positive))
            painter.drawRoundedRect(0, 0, int(fill_width), self.height(), radius, radius)
        painter.end()


class VramMetricCell(QWidget):
    """MODEL VRAM: allocator memory attributed to the inference process."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        col = QVBoxLayout(self)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(2)

        label = QLabel("MODEL VRAM")
        label.setObjectName("footerCellLabel")

        self._value = QLabel(PLACEHOLDER)
        self._value.setObjectName("footerCellValuePositive")
        self._bar = CompactProgressBar()

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)
        row.addWidget(self._value)
        row.addWidget(self._bar, stretch=1)

        col.addStretch(1)
        col.addWidget(label)
        col.addLayout(row)
        col.addStretch(1)

    def set_vram(self, used_gb: float | None, total_gb: float | None) -> None:
        self._value.setText(format_vram(used_gb, total_gb))
        self._bar.set_ratio(vram_ratio(used_gb, total_gb))

    def set_tooltip(self, text: str) -> None:
        self.setToolTip(text)
        self._value.setToolTip(text)
        self._bar.setToolTip(text)


class ModelMetricCell(QWidget):
    """MODEL: elidable one-line name, full name in a tooltip, info button."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        col = QVBoxLayout(self)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(2)

        label = QLabel("MODEL")
        label.setObjectName("footerCellLabel")

        self._value = QLabel(PLACEHOLDER)
        self._value.setObjectName("footerCellValue")
        self._value.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        self._info_btn = QLabel()  # placeholder swapped for a real button below
        info_btn = self._build_info_button()

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        row.addWidget(self._value, stretch=1)
        row.addWidget(info_btn)

        col.addStretch(1)
        col.addWidget(label)
        col.addLayout(row)
        col.addStretch(1)

        self._full_name = ""
        self._details = ""

    def _build_info_button(self) -> QPushButton:
        btn = QPushButton()
        btn.setObjectName("modelInfoButton")
        btn.setFixedSize(22, 22)
        btn.setIcon(render_icon("info", PALETTE.footer_label, 13))
        btn.setToolTip("Model details")
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.clicked.connect(self._show_details)
        self._info_btn = btn
        return btn

    def set_model_name(self, full_name: str | None) -> None:
        self._full_name = full_name or ""
        text = format_or_dash(full_name)
        metrics = self._value.fontMetrics()
        elided = metrics.elidedText(text, Qt.TextElideMode.ElideRight, max(self._value.width(), 60))
        self._value.setText(elided)
        self._value.setToolTip(self._details or self._full_name)

    def set_details(self, details: str) -> None:
        self._details = details
        tooltip = details or self._full_name or "Model details"
        self._value.setToolTip(tooltip)
        self._info_btn.setToolTip(tooltip)

    def _show_details(self) -> None:
        QMessageBox.information(
            self,
            "Model & Runtime Details",
            self._details or self._full_name or "No model details are available yet.",
        )

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        if self._full_name:
            self.set_model_name(self._full_name)


class PerformanceBar(QFrame):
    """Single-row performance/status footer: eleven full-width regions."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("performanceFooter")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(FOOTER_HEIGHT)
        self._snapshot = PerformanceSnapshot.empty()
        self._build_ui()
        self.update_snapshot(self._snapshot)

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 6, 16, 6)
        layout.setSpacing(12)

        self._status_cell = StatusMetricCell()
        self._cam_fps = MetricCell("CAM FPS")
        self._inf_fps = MetricCell("INF FPS")
        self._latency = MetricCell("LATENCY")
        self._render_fps = MetricCell("RENDER FPS")
        self._points = MetricCell("POINTS")
        self._resolution = MetricCell("RESOLUTION")
        self._vram = VramMetricCell()
        self._model = ModelMetricCell()
        self._precision = MetricCell("PRECISION")
        self._uptime = MetricCell("UPTIME")

        cells = (
            ("status", self._status_cell),
            ("cam_fps", self._cam_fps),
            ("inf_fps", self._inf_fps),
            ("latency", self._latency),
            ("render_fps", self._render_fps),
            ("points", self._points),
            ("resolution", self._resolution),
            ("vram", self._vram),
            ("model", self._model),
            ("precision", self._precision),
            ("uptime", self._uptime),
        )
        for index, (key, cell) in enumerate(cells):
            layout.addWidget(cell, _STRETCH[key])
            if index < len(cells) - 1:
                layout.addWidget(_VSeparator())
        # No addStretch(): the eleven cells above already span 100% of the
        # width via their stretch factors — an extra stretch here is what
        # produced the large empty region on the right.

    # -- Public API ---------------------------------------------------------

    def update_snapshot(self, snapshot: PerformanceSnapshot) -> None:
        """Update every cell from one structured snapshot."""
        self._snapshot = snapshot
        self._status_cell.set_state(snapshot.status)
        self._cam_fps.set_value(format_fps(snapshot.camera_fps), positive=True)
        self._inf_fps.set_value(format_fps(snapshot.inference_fps), positive=True)
        self._latency.set_value(format_latency(snapshot.latency_ms))
        self._render_fps.set_value(format_fps(snapshot.render_fps), positive=True)
        self._points.set_value(format_points(snapshot.point_count))
        self._resolution.set_value(format_resolution(snapshot.resolution))
        self._vram.set_vram(snapshot.vram_used_gb, snapshot.vram_total_gb)
        self._model.set_model_name(snapshot.model_name)
        self._precision.set_value(format_or_dash(snapshot.precision))
        self._uptime.set_value(format_uptime(snapshot.uptime_seconds))

        self._status_cell.set_tooltip(snapshot.status_tooltip())
        self._cam_fps.set_tooltip(snapshot.camera_fps_tooltip())
        self._inf_fps.set_tooltip(snapshot.inference_fps_tooltip())
        self._latency.set_tooltip(snapshot.latency_tooltip())
        self._render_fps.set_tooltip(snapshot.render_fps_tooltip())
        self._points.set_tooltip(snapshot.points_tooltip())
        self._resolution.set_tooltip("Depth-result resolution returned by the active model.")
        self._vram.set_tooltip(snapshot.vram_tooltip())
        self._model.set_details(snapshot.model_tooltip())
        self._precision.set_tooltip("Actual precision reported by the loaded worker model.")
        self._uptime.set_tooltip("Elapsed wall-clock time for the current inference session.")

    def clear_metrics(self) -> None:
        """Reset to the IDLE state: neutral badge, every metric unavailable."""
        self.update_snapshot(PerformanceSnapshot.empty())

    # -- Backward-compatible shim for the Phase-1 demo lifecycle -----------
    # main_window.py's temporary demo calls these with a header/settings
    # BadgeKind + free text; this maps that vocabulary onto the footer's
    # own status color roles so the call sites do not need to change.

    def set_status(self, kind: BadgeKind, text: str) -> None:
        role = _BADGE_KIND_TO_COLOR.get(kind, "neutral")
        self._snapshot = self._snapshot.with_status(text)
        self._status_cell.set_state(text, color_role=role)
        self._status_cell.set_tooltip(self._snapshot.status_tooltip())

    @property
    def snapshot(self) -> PerformanceSnapshot:
        """Return the last snapshot rendered by the footer."""
        return self._snapshot

    def reset_metrics(self) -> None:
        self.clear_metrics()

"""Top header bar: branding, GPU/VRAM/CUDA status, and window actions.

Matches the target: a red rounded logo tile, the wordmark + subtitle on
a single baseline, a red PRO badge and version, the hardware block
(GPU / VRAM / CUDA) grouped toward the right, a status badge, and vector
icon actions (help / settings / minimize / maximize / close).

Phase 1 shows placeholder hardware values; real values are pushed in via
``update_gpu_status`` / ``update_cuda_status`` from the hardware layer in
a later phase.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from app.widgets.badge import BadgeKind, BadgeLabel
from app.theme import PALETTE

APP_VERSION = "v2.0.0"


class _LogoTile(QLabel):
    """Company logo. Loads assets/logo/logo.png (scaled to the header
    height); falls back to a small red glyph tile if the file is missing.

    The logo replaces the text wordmark, so ``LOGO_MAX_WIDTH`` is wide
    enough for a full icon+wordmark lockup image. Reduce it if your PNG
    is just the square mark.
    """

    LOGO_HEIGHT = 30
    LOGO_MAX_WIDTH = 150

    def __init__(self) -> None:
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        logo_path = Path(__file__).resolve().parents[2] / "assets" / "logo" / "logo.png"
        pixmap = QPixmap(str(logo_path)) if logo_path.exists() else QPixmap()

        if not pixmap.isNull():
            scaled = pixmap.scaledToHeight(self.LOGO_HEIGHT, Qt.TransformationMode.SmoothTransformation)
            if scaled.width() > self.LOGO_MAX_WIDTH:
                scaled = pixmap.scaledToWidth(self.LOGO_MAX_WIDTH, Qt.TransformationMode.SmoothTransformation)
            self.setPixmap(scaled)
            self.setFixedSize(scaled.width(), self.LOGO_HEIGHT)
        else:
            # Fallback: small red tile with a glyph if logo.png is absent.
            self.setText("◨")
            self.setFixedSize(self.LOGO_HEIGHT, self.LOGO_HEIGHT)
            self.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.setStyleSheet(
                f"background-color: {PALETTE.accent}; color: white; "
                f"border-radius: 6px; font-size: 16px; font-weight: 900;"
            )


class HeaderWidget(QFrame):
    """Application header bar shown at the very top of the main window."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("headerBar")
        self.setFixedHeight(56)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 0, 12, 0)
        layout.setSpacing(14)

        layout.addLayout(self._build_brand_block())
        layout.addStretch(1)
        layout.addLayout(self._build_hardware_block())
        layout.addSpacing(14)
        layout.addWidget(self._build_status_badge())

    def _build_brand_block(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(12)
        row.addWidget(_LogoTile())

        title_row = QHBoxLayout()
        title_row.setSpacing(8)

        product = QLabel("DEPTH & 3D MODEL LAB")
        product.setObjectName("productName")
        title_row.addWidget(product)

        pro_badge = QLabel("PRO")
        pro_badge.setObjectName("proBadge")
        pro_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_row.addWidget(pro_badge)

        version = QLabel(APP_VERSION)
        version.setObjectName("muted")
        title_row.addWidget(version)
        title_row.addStretch(1)

        row.addLayout(title_row)
        return row

    def _build_hardware_block(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(24)
        row.addLayout(self._metric_pair("GPU", "Detecting…", "gpu_value", PALETTE.success))
        row.addLayout(self._metric_pair("VRAM", "— / — GB", "vram_value", PALETTE.text_primary))
        row.addLayout(self._metric_pair("CUDA", "—", "cuda_value", PALETTE.success))
        return row

    def _metric_pair(self, label_text: str, initial: str, name: str, value_color: str) -> QVBoxLayout:
        col = QVBoxLayout()
        col.setSpacing(0)
        label = QLabel(label_text)
        label.setObjectName("metricLabel")
        value = QLabel(initial)
        value.setStyleSheet(
            f"font-family: 'JetBrains Mono', monospace; font-size: 12px; "
            f"font-weight: 600; color: {value_color};"
        )
        setattr(self, f"_{name}", value)
        col.addWidget(label)
        col.addWidget(value)
        return col

    def _build_status_badge(self) -> BadgeLabel:
        self._status_badge = BadgeLabel(BadgeKind.LOADING, "INITIALIZING")
        return self._status_badge

    # -- Public update API ------------------------------------------------

    def update_gpu_status(self, gpu_name: str, vram_used_gb: float, vram_total_gb: float) -> None:
        self._gpu_value.setText(gpu_name)
        self._vram_value.setText(f"{vram_used_gb:.1f} / {vram_total_gb:.0f} GB")

    def update_cuda_status(self, cuda_version: str | None) -> None:
        self._cuda_value.setText(cuda_version or "Not available")

    def set_status(self, kind: BadgeKind, text: str | None = None) -> None:
        self._status_badge.set_kind(kind, text)
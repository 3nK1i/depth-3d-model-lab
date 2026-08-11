"""Reusable Phase 8.8 neural-appearance preview overlay."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QResizeEvent
from PyQt6.QtWidgets import QFrame, QLabel, QVBoxLayout


class ReconstructionAppearancePreview(QFrame):
    """Aspect-preserving static appearance preview inside Reconstruction viewport.

    F2/F3 use backend-authored preview images only.  The widget performs no
    neural rendering and is intentionally reusable across appearance kinds.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("reconstructionAppearancePreview")
        self._source = QPixmap()
        self._source_path = ""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(8)

        self._title = QLabel("APPEARANCE PREVIEW • APPEARANCE ONLY")
        self._title.setObjectName("reconstructionAppearancePreviewTitle")
        self._title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._title)

        self._image = QLabel()
        self._image.setObjectName("reconstructionAppearancePreviewImage")
        self._image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image.setMinimumSize(1, 1)
        layout.addWidget(self._image, stretch=1)

        self._footer = QLabel("Backend-authored static render • canonical geometry remains authoritative")
        self._footer.setObjectName("reconstructionAppearancePreviewFooter")
        self._footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._footer.setWordWrap(True)
        layout.addWidget(self._footer)
        self.hide()

    @property
    def source_path(self) -> str:
        return self._source_path

    @property
    def has_image(self) -> bool:
        return not self._source.isNull()

    def set_preview(self, path: str | Path, *, title: str = "GAUSSIAN PREVIEW • APPEARANCE ONLY") -> None:
        source = Path(path).expanduser().resolve()
        pixmap = QPixmap(str(source))
        if pixmap.isNull():
            raise ValueError(f"Appearance preview image could not be decoded: {source}")
        self._source = pixmap
        self._source_path = str(source)
        self._title.setText(str(title).strip() or "APPEARANCE PREVIEW • APPEARANCE ONLY")
        self._rescale()

    def clear_preview(self) -> None:
        self._source = QPixmap()
        self._source_path = ""
        self._image.clear()
        self.hide()

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self) -> None:
        if self._source.isNull():
            self._image.clear()
            return
        size = self._image.size()
        if size.width() <= 1 or size.height() <= 1:
            return
        scaled = self._source.scaled(
            size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._image.setPixmap(scaled)

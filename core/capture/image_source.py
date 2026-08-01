"""Static image source: loads one file and keeps it ready until disconnect.

Unlike video/camera/RTSP, there is no acquisition loop — a single frame
is read and emitted once. The source remains CONNECTED because the loaded
image can be re-emitted on demand via :meth:`resend` without touching disk.
An explicit :meth:`~core.capture.base_source.BaseSource.stop` transitions it
to STOPPED when the user disconnects.
"""

from __future__ import annotations

import cv2

from core.capture.base_source import BaseSource, SourceStatus

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")


class ImageSource(BaseSource):
    """Loads a single image file and emits it as one frame."""

    def __init__(self, path: str, parent=None) -> None:
        super().__init__(parent)
        self._path = path
        self._last_image = None

    def _run(self) -> None:
        image = cv2.imread(self._path, cv2.IMREAD_COLOR)
        if image is None:
            self._emit_error(f"Could not read image: {self._path}")
            return
        self._last_image = image
        self._set_status(SourceStatus.CONNECTED)
        self._emit_frame(image)
        # No acquisition loop is needed, but the loaded image remains a
        # valid connected source until the user explicitly disconnects.

    def resend(self) -> None:
        """Re-emit the already-loaded image without touching disk again."""
        if self._last_image is not None:
            self._emit_frame(self._last_image)

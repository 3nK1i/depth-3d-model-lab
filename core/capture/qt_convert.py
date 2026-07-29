"""Convert OpenCV (BGR ndarray) frames to Qt pixmaps for display.

Kept as one small helper so this conversion isn't duplicated across
every place a frame reaches the UI.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtGui import QImage, QPixmap


def ndarray_to_qpixmap(image: np.ndarray) -> QPixmap:
    """Convert a BGR (OpenCV convention) HxWx3 uint8 array to a QPixmap."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected an HxWx3 BGR array, got shape {image.shape}")

    height, width, _ = image.shape
    rgb = np.ascontiguousarray(image[:, :, ::-1])  # BGR -> RGB
    bytes_per_line = width * 3
    qimage = QImage(rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
    # .copy() detaches from the numpy buffer, which may be reused/freed
    # by the next frame before Qt is done with it otherwise.
    return QPixmap.fromImage(qimage.copy())
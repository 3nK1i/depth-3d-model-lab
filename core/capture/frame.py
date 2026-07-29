"""Shared frame data structure passed from capture sources to the UI.

One dataclass so every source (image/video/camera/RTSP) hands the UI the
same shape regardless of how the frame was produced.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True, slots=True)
class SourceFrame:
    """A single captured frame plus the metadata the UI/inference need.

    ``frame_id`` increments per frame from a given source connection and
    is what lets a later consumer (Phase 3's inference worker) detect and
    drop stale results after a stop/restart — see spec §13.
    """

    image: np.ndarray  # HxWx3, BGR (OpenCV convention), uint8
    frame_id: int
    timestamp: float = field(default_factory=time.monotonic)

    @property
    def height(self) -> int:
        return self.image.shape[0]

    @property
    def width(self) -> int:
        return self.image.shape[1]
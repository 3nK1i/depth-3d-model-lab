"""Capture sources: image, video, camera, RTSP.

Silences OpenCV's own C++-level console logging here, once, before any
capture code runs. Probing non-existent camera indices (device
enumeration) and RTSP reconnect attempts are both expected, routine
occurrences in this module — OpenCV logs a WARN/ERROR line for each one
regardless, which is harmless noise, not a real problem. Genuine
failures are still reported through this package's own
``error_occurred`` signals, so nothing is actually being hidden from the
application — only OpenCV's low-level stderr chatter.

The log-level API itself is not consistent across OpenCV builds/binary
distributions (``cv2.utils.logging`` exists on some, the older top-level
``cv2.setLogLevel`` on others, neither on some minimal builds), so every
attempt is wrapped defensively — this must never be able to crash the
app at import time just because a particular OpenCV build doesn't
expose one of these.
"""

import cv2


def _silence_opencv_logging() -> None:
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
        return
    except AttributeError:
        pass

    try:
        # Older/alternate binding: a top-level function taking a plain
        # int (0 == silent) rather than the cv2.utils.logging enum.
        cv2.setLogLevel(0)
        return
    except AttributeError:
        pass

    # Neither API is present on this OpenCV build. Nothing to do —
    # capture still works correctly, the only cost is console noise.


_silence_opencv_logging()
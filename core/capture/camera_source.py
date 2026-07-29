"""USB/built-in camera source (spec §5C).

Opens a device index with OpenCV and continuously grabs frames on its
own thread. Camera preview works independently of inference — this
source knows nothing about inference state at all.
"""

from __future__ import annotations


import cv2

from PyQt6.QtCore import QObject, QThread, pyqtSignal

from core.capture.base_source import BaseSource, SourceStatus


class CameraBackend:
    AUTO = cv2.CAP_ANY
    V4L2 = cv2.CAP_V4L2
    GSTREAMER = cv2.CAP_GSTREAMER


class CameraSource(BaseSource):
    """Continuously grabs frames from a camera device index."""

    def __init__(
        self,
        device_index: int = 0,
        *,
        width: int | None = None,
        height: int | None = None,
        fps: int | None = None,
        backend: int = CameraBackend.AUTO,
        mirror: bool = False,
        rotate_degrees: int = 0,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._device_index = device_index
        self._width = width
        self._height = height
        self._fps = fps
        self._backend = backend
        self._mirror = mirror
        self._rotate_degrees = rotate_degrees

    def _run(self) -> None:
        capture = cv2.VideoCapture(self._device_index, self._backend)
        if self._width:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        if self._height:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        if self._fps:
            capture.set(cv2.CAP_PROP_FPS, self._fps)

        if not capture.isOpened():
            self._emit_error(f"Could not open camera device {self._device_index}")
            return

        self._set_status(SourceStatus.CONNECTED)
        while not self._stop_requested:
            ok, frame = capture.read()
            if not ok:
                self._emit_error(f"Camera device {self._device_index} read failed")
                break
            frame = self._apply_transforms(frame)
            self._emit_frame(frame)

        capture.release()
        if not self._stop_requested:
            return  # error path already reported above
        self._set_status(SourceStatus.STOPPED)

    def _apply_transforms(self, frame):
        if self._mirror:
            frame = cv2.flip(frame, 1)
        if self._rotate_degrees == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif self._rotate_degrees == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif self._rotate_degrees == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    def set_mirror(self, mirror: bool) -> None:
        self._mirror = mirror

    def set_rotation(self, degrees: int) -> None:
        self._rotate_degrees = degrees % 360


def enumerate_cameras(max_index: int = 8) -> list[int]:
    """Probe device indices 0..max_index-1 and return the ones that open.

    A simple, synchronous probe — fine to call from the UI thread since
    it is only used for the (infrequent) device-list refresh action, not
    the acquisition loop itself.
    """
    found: list[int] = []
    for index in range(max_index):
        capture = cv2.VideoCapture(index)
        if capture.isOpened():
            found.append(index)
        capture.release()
    return found


class CameraEnumerator(QObject):
    """One-shot background device probe.

    Opening non-existent device indices can take noticeably longer on
    real V4L2/GStreamer backends than in a clean sandbox, so this must
    never run on the UI thread — it's a tiny throwaway QThread, not the
    long-lived per-source thread pattern :class:`BaseSource` uses.
    """

    devices_found = pyqtSignal(list)

    def __init__(self, max_index: int = 8, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._max_index = max_index
        self._thread: QThread | None = None

    def start(self) -> None:
        self._thread = QThread()
        self.moveToThread(self._thread)
        self._thread.started.connect(self._run)
        self._thread.start()

    def _run(self) -> None:
        found = enumerate_cameras(self._max_index)
        self.devices_found.emit(found)
        if self._thread is not None:
            self._thread.quit()
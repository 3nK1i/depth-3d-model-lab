"""Video file source with playback controls (spec §5B).

Runs an OpenCV ``VideoCapture`` read loop on its own thread, paced to the
file's own frame rate. Supports play/pause/stop/restart/seek/loop/speed
without tearing down the capture — pausing simply stops advancing the
read position, it does not stop the thread.
"""

from __future__ import annotations

import time

import cv2

from core.capture.base_source import BaseSource, SourceStatus


class VideoSource(BaseSource):
    """Plays back a video file, emitting frames at (playback-speed ×
    native) FPS. Independent of any inference state (spec §5)."""

    def __init__(self, path: str, parent=None) -> None:
        super().__init__(parent)
        self._path = path
        self._capture: cv2.VideoCapture | None = None
        self._playing = True
        self._loop = True
        self._speed = 1.0
        self._seek_to_frame: int | None = None
        self._native_fps: float | None = None

    def _run(self) -> None:
        capture = cv2.VideoCapture(self._path)
        if not capture.isOpened():
            self._emit_error(f"Could not open video: {self._path}")
            return
        self._capture = capture

        fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        self._native_fps = fps
        frame_interval = 1.0 / fps
        self._set_status(SourceStatus.CONNECTED)

        while not self._stop_requested:
            if self._seek_to_frame is not None:
                capture.set(cv2.CAP_PROP_POS_FRAMES, self._seek_to_frame)
                self._seek_to_frame = None

            if not self._playing:
                time.sleep(0.03)
                continue

            start = time.monotonic()
            ok, frame = capture.read()
            if not ok:
                if self._loop:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
            self._emit_frame(frame)

            elapsed = time.monotonic() - start
            target = frame_interval / max(self._speed, 0.01)
            remaining = target - elapsed
            if remaining > 0:
                time.sleep(remaining)

        capture.release()
        self._capture = None
        self._set_status(SourceStatus.STOPPED)

    # -- Playback controls (safe to call from the UI thread: these only
    # set plain attributes the loop reads each iteration) ------------------

    def play(self) -> None:
        self._playing = True

    def pause(self) -> None:
        self._playing = False

    def seek_to_frame(self, frame_index: int) -> None:
        self._seek_to_frame = max(0, frame_index)

    def set_loop(self, loop: bool) -> None:
        self._loop = loop

    def set_speed(self, speed: float) -> None:
        self._speed = max(0.05, speed)

    @property
    def is_playing(self) -> bool:
        return self._playing

    @property
    def total_frames(self) -> int:
        if self._capture is None:
            return 0
        return int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def current_frame_index(self) -> int:
        if self._capture is None:
            return 0
        return int(self._capture.get(cv2.CAP_PROP_POS_FRAMES))

    @property
    def path(self) -> str:
        return self._path

    @property
    def native_fps(self) -> float | None:
        return self._native_fps

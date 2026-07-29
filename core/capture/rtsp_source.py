"""RTSP network stream source (spec §5D).

Adds automatic reconnect with a bounded attempt count on top of the same
continuous-grab pattern as :class:`~core.capture.camera_source.CameraSource`.
Never logs or stores the URL's password in plain text (spec §5D, §19).
"""

from __future__ import annotations

import os
import re
import time

import cv2

from core.capture.base_source import BaseSource, SourceStatus

_PASSWORD_PATTERN = re.compile(r"(rtsp://[^:]+:)([^@]+)(@)")


def sanitize_rtsp_url(url: str) -> str:
    """Redact a password embedded in an rtsp://user:pass@host URL for
    safe logging/display. Never call this the *only* protection —
    callers must also avoid persisting the raw URL to disk in plain text."""
    return _PASSWORD_PATTERN.sub(r"\1***\3", url)


class RtspSource(BaseSource):
    """Continuously grabs frames from an RTSP stream, with reconnect."""

    def __init__(
        self,
        url: str,
        *,
        use_tcp: bool = True,
        low_latency: bool = True,
        connection_timeout_ms: int = 5000,
        max_reconnect_attempts: int = 5,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._url = url
        self._use_tcp = use_tcp
        self._low_latency = low_latency
        self._connection_timeout_ms = connection_timeout_ms
        self._max_reconnect_attempts = max_reconnect_attempts
        self._reconnect_attempts = 0

    def _open_capture(self) -> cv2.VideoCapture:
        transport = "tcp" if self._use_tcp else "udp"
        # OPENCV_FFMPEG_CAPTURE_OPTIONS is read by the FFmpeg backend for
        # transport selection; harmless if the backend ignores it.
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;{transport}"

        # CAP_PROP_OPEN_TIMEOUT_MSEC must be passed via the constructor's
        # params list, NOT set on the VideoCapture afterward — the
        # constructor performs the actual blocking connect, so setting it
        # post-construction is too late to have any effect and an
        # unreachable host can then block the worker thread for the OS's
        # own TCP timeout (potentially 60s+), not the configured value.
        params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self._connection_timeout_ms]
        if self._low_latency:
            params += [cv2.CAP_PROP_BUFFERSIZE, 1]
        capture = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG, params)
        return capture

    def _interruptible_sleep(self, seconds: float) -> None:
        """Sleep in small chunks, checking ``_stop_requested`` between each.

        A plain ``time.sleep(10.0)`` cannot be interrupted once started —
        ``stop()`` would have to wait out the full duration, which is
        exactly the kind of blocking-call-swallows-the-stop-signal bug
        the safe-stop design (Phase 3) exists to prevent. Reconnect
        backoff can be several seconds, so it must be interruptible here,
        not just cooperative at the top of the loop.
        """
        deadline = time.monotonic() + seconds
        while not self._stop_requested and time.monotonic() < deadline:
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))

    def _run(self) -> None:
        while not self._stop_requested:
            capture = self._open_capture()
            if not capture.isOpened():
                self._reconnect_attempts += 1
                if self._reconnect_attempts > self._max_reconnect_attempts:
                    self._emit_error(
                        f"RTSP connection failed after {self._max_reconnect_attempts} attempts"
                    )
                    return
                self._set_status(SourceStatus.RECONNECTING)
                self._interruptible_sleep(min(2.0 * self._reconnect_attempts, 10.0))
                continue

            self._reconnect_attempts = 0
            self._set_status(SourceStatus.CONNECTED)
            self._read_until_disconnected(capture)
            capture.release()

        self._set_status(SourceStatus.STOPPED)

    def _read_until_disconnected(self, capture: cv2.VideoCapture) -> None:
        while not self._stop_requested:
            ok, frame = capture.read()
            if not ok:
                return  # triggers a reconnect in the outer loop
            self._emit_frame(frame)

    @property
    def reconnect_attempts(self) -> int:
        return self._reconnect_attempts
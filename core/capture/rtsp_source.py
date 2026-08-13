"""RTSP network stream source (spec §5D).

Adds automatic reconnect with a bounded attempt count on top of the same
continuous-grab pattern as :class:`~core.capture.camera_source.CameraSource`.
Never logs or stores the URL's password in plain text (spec §5D, §19).
"""

from __future__ import annotations

import math
import os
import re
import time

import cv2
from PyQt6.QtCore import QThread

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
        read_timeout_ms: int = 1500,
        max_reconnect_attempts: int = 5,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._url = url
        self._use_tcp = use_tcp
        self._low_latency = low_latency
        self._connection_timeout_ms = connection_timeout_ms
        self._read_timeout_ms = read_timeout_ms
        self._max_reconnect_attempts = max_reconnect_attempts
        self._reconnect_attempts = 0
        self._native_fps: float | None = None

    def _open_capture(self) -> cv2.VideoCapture:
        transport = "tcp" if self._use_tcp else "udp"
        # OPENCV_FFMPEG_CAPTURE_OPTIONS is read by the FFmpeg backend for
        # transport selection; harmless if the backend ignores it.
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;{transport}"

        # Open/read timeouts are FFmpeg open-only properties and must be
        # supplied when VideoCapture is constructed. They bound the two native
        # calls that can otherwise outlive BaseSource.stop() and leave a Qt
        # capture thread running during application teardown. CAP_PROP_BUFFERSIZE
        # intentionally remains post-open: this FFmpeg build rejects it when
        # supplied in the constructor params list.
        params = [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
            self._connection_timeout_ms,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC,
            self._read_timeout_ms,
        ]
        self._native_fps = None
        capture = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG, params)

        if capture.isOpened():
            try:
                native_fps = float(capture.get(cv2.CAP_PROP_FPS))
            except (TypeError, ValueError, OverflowError):
                native_fps = 0.0
            if math.isfinite(native_fps) and 0.0 < native_fps < 1000.0:
                self._native_fps = native_fps

        if self._low_latency and capture.isOpened():
            # Best-effort only: this backend build silently fails to
            # apply it (set() returns False) rather than erroring, so a
            # low-latency hint that doesn't take effect must never break
            # the connection — hence setting it here, after open, not in
            # the params list above.
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture

    def stop(self) -> None:
        """Stop RTSP capture without destroying a still-running QThread.

        FFmpeg open/read calls are bounded above, so the normal wait remains
        finite. If a backend ever ignores those timeouts, the final unbounded
        wait is deliberately safer than releasing the last QThread reference
        while it is still running (Qt aborts in that situation). Native thread
        termination is never used.
        """
        self._stop_requested = True
        thread = self._thread
        if thread is not None:
            thread.quit()
            if QThread.currentThread() is thread:
                # A source-status callback may request stop from the capture
                # thread itself (for example the bounded RTSP connection test).
                # Waiting on the current QThread would deadlock; _run() will
                # observe the stop flag and return, and the owner can reap the
                # finished thread on its next external stop/shutdown call.
                return
            shutdown_wait_ms = max(
                2000,
                int(self._connection_timeout_ms) + int(self._read_timeout_ms) + 500,
            )
            if not thread.wait(shutdown_wait_ms):
                # Last-resort ownership barrier: never drop the QThread object
                # while Qt still reports it as running. The FFmpeg timeouts
                # above should make this path exceptional.
                thread.wait()
            self._thread = None
        self._set_status(SourceStatus.STOPPED)

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

    @property
    def native_fps(self) -> float | None:
        """Frame rate reported by the active RTSP decoder, when trustworthy."""
        return self._native_fps

    @property
    def sanitized_url(self) -> str:
        return sanitize_rtsp_url(self._url)

    @property
    def source_fingerprint(self) -> str:
        # The raw locator remains private/in-memory; only its one-way identity
        # leaves the capture object.
        import hashlib

        return hashlib.sha256(self._url.encode("utf-8")).hexdigest()

from __future__ import annotations

import math

import cv2
import pytest

pytest.importorskip("PyQt6")

from core.capture.rtsp_source import RtspSource


class _Capture:
    def __init__(self, *, opened: bool = True, fps: float = 30.0) -> None:
        self._opened = opened
        self._fps = fps
        self.set_calls: list[tuple[int, float]] = []

    def isOpened(self) -> bool:
        return self._opened

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FPS:
            return self._fps
        return 0.0

    def set(self, prop: int, value: float) -> bool:
        self.set_calls.append((prop, value))
        return True


class _Thread:
    def __init__(self, wait_results: list[bool]) -> None:
        self.wait_results = list(wait_results)
        self.quit_calls = 0
        self.wait_calls: list[tuple[object, ...]] = []

    def quit(self) -> None:
        self.quit_calls += 1

    def wait(self, *args) -> bool:
        self.wait_calls.append(args)
        if self.wait_results:
            return self.wait_results.pop(0)
        return True


def test_rtsp_native_fps_is_unknown_before_decoder_open() -> None:
    source = RtspSource("rtsp://camera.local/live")
    assert source.native_fps is None


def test_rtsp_open_captures_valid_native_fps(monkeypatch) -> None:
    capture = _Capture(opened=True, fps=30.0)
    monkeypatch.setattr("core.capture.rtsp_source.cv2.VideoCapture", lambda *_args: capture)

    source = RtspSource("rtsp://camera.local/live")
    assert source._open_capture() is capture
    assert source.native_fps == 30.0


def test_rtsp_open_binds_open_and_read_timeouts(monkeypatch) -> None:
    capture = _Capture(opened=True, fps=30.0)
    calls: list[tuple[object, ...]] = []

    def _video_capture(*args):
        calls.append(args)
        return capture

    monkeypatch.setattr("core.capture.rtsp_source.cv2.VideoCapture", _video_capture)

    source = RtspSource(
        "rtsp://camera.local/live",
        connection_timeout_ms=4200,
        read_timeout_ms=1300,
    )
    source._open_capture()

    assert len(calls) == 1
    params = calls[0][2]
    assert params == [
        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
        4200,
        cv2.CAP_PROP_READ_TIMEOUT_MSEC,
        1300,
    ]
    assert cv2.CAP_PROP_BUFFERSIZE not in params


@pytest.mark.parametrize("fps", [0.0, -1.0, math.nan, math.inf, -math.inf, 1000.0, 1001.0])
def test_rtsp_open_rejects_untrustworthy_native_fps(monkeypatch, fps: float) -> None:
    capture = _Capture(opened=True, fps=fps)
    monkeypatch.setattr("core.capture.rtsp_source.cv2.VideoCapture", lambda *_args: capture)

    source = RtspSource("rtsp://camera.local/live")
    source._open_capture()
    assert source.native_fps is None


def test_rtsp_failed_reopen_clears_stale_native_fps(monkeypatch) -> None:
    source = RtspSource("rtsp://camera.local/live")
    source._native_fps = 25.0
    capture = _Capture(opened=False, fps=30.0)
    monkeypatch.setattr("core.capture.rtsp_source.cv2.VideoCapture", lambda *_args: capture)

    source._open_capture()
    assert source.native_fps is None


def test_rtsp_stop_waits_for_timeout_budget_before_dropping_thread() -> None:
    source = RtspSource(
        "rtsp://camera.local/live",
        connection_timeout_ms=5000,
        read_timeout_ms=1500,
    )
    thread = _Thread([True])
    source._thread = thread  # type: ignore[assignment]

    source.stop()

    assert source._stop_requested is True
    assert thread.quit_calls == 1
    assert thread.wait_calls == [(7000,)]
    assert source._thread is None


def test_rtsp_stop_uses_ownership_barrier_if_bounded_wait_expires() -> None:
    source = RtspSource("rtsp://camera.local/live")
    thread = _Thread([False, True])
    source._thread = thread  # type: ignore[assignment]

    source.stop()

    assert thread.quit_calls == 1
    assert thread.wait_calls[0] == (7000,)
    assert thread.wait_calls[1] == ()
    assert source._thread is None

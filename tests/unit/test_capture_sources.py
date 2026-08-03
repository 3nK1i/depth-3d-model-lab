"""Capture-source contracts without requiring physical cameras or RTSP servers."""

from __future__ import annotations

import os

import cv2
import numpy as np

from core.capture.base_source import BaseSource, SourceStatus
from core.capture.camera_source import CameraSource, enumerate_cameras
from core.capture.frame import SourceFrame
from core.capture.rtsp_source import RtspSource, sanitize_rtsp_url
from core.capture.video_source import VideoSource


class _Capture:
    def __init__(self, *, opened=True, frames=None, fps=30.0) -> None:
        self.opened = opened
        self.frames = list(frames or [])
        self.fps = fps
        self.released = False
        self.set_calls: list[tuple[int, float]] = []
        self.position = 0

    def isOpened(self):
        return self.opened

    def read(self):
        if not self.frames:
            return False, None
        self.position += 1
        return True, self.frames.pop(0)

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return self.position + len(self.frames)
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return self.position
        return 0

    def set(self, prop, value):
        self.set_calls.append((prop, value))
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self.position = int(value)
        return True

    def release(self):
        self.released = True


class _DummySource(BaseSource):
    def _run(self) -> None:
        return


def test_source_frame_dimensions_and_monotonic_metadata() -> None:
    frame = SourceFrame(np.zeros((7, 11, 3), dtype=np.uint8), frame_id=5)
    assert frame.height == 7
    assert frame.width == 11
    assert frame.timestamp > 0


def test_base_source_assigns_incrementing_frame_ids() -> None:
    source = _DummySource()
    received = []
    source.frame_ready.connect(received.append)
    image = np.zeros((2, 3, 3), np.uint8)
    source._emit_frame(image)
    source._emit_frame(image)
    assert [frame.frame_id for frame in received] == [0, 1]


def test_video_controls_clamp_values() -> None:
    source = VideoSource("unused.mp4")
    source.pause()
    assert not source.is_playing
    source.play()
    assert source.is_playing
    source.seek_to_frame(-50)
    assert source._seek_to_frame == 0
    source.set_speed(0.0)
    assert source._speed == 0.05
    source.set_loop(False)
    assert source._loop is False


def test_video_run_emits_frames_and_stops(monkeypatch) -> None:
    frames = [np.full((2, 3, 3), 1, np.uint8), np.full((2, 3, 3), 2, np.uint8)]
    capture = _Capture(frames=frames, fps=1000.0)
    monkeypatch.setattr("core.capture.video_source.cv2.VideoCapture", lambda *_: capture)

    source = VideoSource("test.mp4")
    source.set_loop(False)
    statuses = []
    received = []
    source.status_changed.connect(statuses.append)
    source.frame_ready.connect(received.append)
    source._run()

    assert statuses == [SourceStatus.CONNECTED, SourceStatus.STOPPED]
    assert [frame.frame_id for frame in received] == [0, 1]
    assert capture.released


def test_camera_transforms_are_applied() -> None:
    source = CameraSource(mirror=True, rotate_degrees=90)
    frame = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    expected = cv2.rotate(cv2.flip(frame, 1), cv2.ROTATE_90_CLOCKWISE)
    np.testing.assert_array_equal(source._apply_transforms(frame), expected)


def test_camera_enumeration_releases_every_probe(monkeypatch) -> None:
    captures = {index: _Capture(opened=index in {0, 2}) for index in range(4)}
    monkeypatch.setattr(
        "core.capture.camera_source.cv2.VideoCapture",
        lambda index: captures[index],
    )
    assert enumerate_cameras(4) == [0, 2]
    assert all(capture.released for capture in captures.values())


def test_rtsp_password_is_redacted_without_changing_other_parts() -> None:
    assert sanitize_rtsp_url("rtsp://user:secret@10.0.0.5:554/live") == (
        "rtsp://user:***@10.0.0.5:554/live"
    )
    assert sanitize_rtsp_url("rtsp://10.0.0.5/live") == "rtsp://10.0.0.5/live"


def test_rtsp_open_uses_timeout_and_sets_low_latency_after_open(monkeypatch) -> None:
    capture = _Capture(opened=True)
    calls = []

    def _open(*args):
        calls.append(args)
        return capture

    monkeypatch.setattr("core.capture.rtsp_source.cv2.VideoCapture", _open)
    source = RtspSource("rtsp://camera/live", connection_timeout_ms=4321)
    assert source._open_capture() is capture

    assert calls[0][0] == "rtsp://camera/live"
    assert calls[0][1] == cv2.CAP_FFMPEG
    assert calls[0][2] == [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 4321]
    assert (cv2.CAP_PROP_BUFFERSIZE, 1) in capture.set_calls
    assert os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] == "rtsp_transport;tcp"


def test_rtsp_gives_bounded_error_after_reconnect_limit(monkeypatch) -> None:
    source = RtspSource("rtsp://camera/live", max_reconnect_attempts=2)
    monkeypatch.setattr(source, "_open_capture", lambda: _Capture(opened=False))
    monkeypatch.setattr(source, "_interruptible_sleep", lambda _seconds: None)
    statuses = []
    errors = []
    source.status_changed.connect(statuses.append)
    source.error_occurred.connect(errors.append)

    source._run()

    assert statuses.count(SourceStatus.RECONNECTING) == 2
    assert statuses[-1] is SourceStatus.ERROR
    assert errors == ["RTSP connection failed after 2 attempts"]

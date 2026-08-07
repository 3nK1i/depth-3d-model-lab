from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from core.capture.camera_source import CameraSource
from core.capture.rtsp_source import RtspSource
from core.capture.video_source import VideoSource


def test_camera_exposes_read_only_live_identity_without_opening_device() -> None:
    source = CameraSource(3, width=1280, height=720, fps=30)
    assert source.device_index == 3
    assert source.requested_width == 1280
    assert source.requested_height == 720
    assert source.requested_fps == 30


def test_rtsp_exposes_only_sanitized_locator_and_one_way_identity() -> None:
    source = RtspSource("rtsp://user:password@camera.local/live")
    assert source.sanitized_url == "rtsp://user:***@camera.local/live"
    assert "password" not in source.sanitized_url
    assert len(source.source_fingerprint) == 64


def test_video_exposes_path_without_changing_playback_contract(tmp_path) -> None:
    path = tmp_path / "clip.mp4"
    source = VideoSource(str(path))
    assert source.path == str(path)
    assert source.native_fps is None

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from core.reconstruction import (
    ImageSequenceReader,
    RecordedVideoReader,
    TimestampOrigin,
    make_rtsp_source_metadata,
    make_usb_source_metadata,
)
from tests.reconstruction_ingestion_helpers import textured_frame


def test_image_directory_uses_natural_filename_order(tmp_path: Path) -> None:
    for name, index in (("frame10.png", 10), ("frame2.png", 2), ("frame1.png", 1)):
        assert cv2.imwrite(str(tmp_path / name), textured_frame(index))
    reader = ImageSequenceReader.from_directory(tmp_path, frame_rate_hz=5.0)
    assert [path.name for path in reader.paths] == ["frame1.png", "frame2.png", "frame10.png"]
    frames = list(reader)
    assert [frame.sequence_index for frame in frames] == [0, 1, 2]
    assert [frame.timestamp_ns for frame in frames] == [0, 200_000_000, 400_000_000]


def test_explicit_image_list_can_preserve_caller_order(tmp_path: Path) -> None:
    paths = []
    for name, index in (("b.png", 2), ("a.png", 1)):
        path = tmp_path / name
        assert cv2.imwrite(str(path), textured_frame(index))
        paths.append(path)
    reader = ImageSequenceReader.from_paths(paths, preserve_order=True)
    assert [path.name for path in reader.paths] == ["b.png", "a.png"]


def test_corrupt_image_is_reported_without_aborting_sequence(tmp_path: Path) -> None:
    good = tmp_path / "frame1.png"
    bad = tmp_path / "frame2.png"
    assert cv2.imwrite(str(good), textured_frame(1))
    bad.write_bytes(b"not a real image")
    reader = ImageSequenceReader.from_directory(tmp_path)
    frames = list(reader)
    assert len(frames) == 1
    assert len(reader.issues) == 1
    assert reader.issues[0].sequence_index == 1
    assert reader.issues[0].reasons[0].value == "decode_failed"


class _VideoCapture:
    def __init__(self, *, frames, fps=10.0, frame_count=3, pts_ms=None, opened=True):
        self.frames = list(frames)
        self.fps = fps
        self.frame_count = frame_count
        self.pts_ms = list(pts_ms or [])
        self.opened = opened
        self.read_index = 0
        self.released = False

    def isOpened(self):
        return self.opened

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return self.frame_count
        if prop == cv2.CAP_PROP_POS_MSEC:
            return self.pts_ms[max(0, self.read_index - 1)]
        return 0.0

    def read(self):
        if not self.frames:
            return False, None
        frame = self.frames.pop(0)
        self.read_index += 1
        return True, frame

    def release(self):
        self.released = True


def test_video_reader_preserves_pts_and_falls_back_when_non_monotonic(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"synthetic-container")
    probe = _VideoCapture(frames=[], fps=10.0, frame_count=3)
    decode = _VideoCapture(
        frames=[textured_frame(i) for i in range(3)],
        fps=10.0,
        frame_count=3,
        pts_ms=[0.0, 0.0, 100.0],
    )
    captures = iter((probe, decode))
    monkeypatch.setattr(
        "core.reconstruction.sources.cv2.VideoCapture", lambda *_args: next(captures)
    )
    reader = RecordedVideoReader(path)
    frames = list(reader)
    assert [item.timestamp_ns for item in frames] == [0, 100_000_000, 200_000_000]
    assert [item.timestamp_origin for item in frames] == [
        TimestampOrigin.PRESENTATION_TIMESTAMP,
        TimestampOrigin.SYNTHETIC,
        TimestampOrigin.SYNTHETIC,
    ]
    assert probe.released and decode.released
    assert not reader.issues


def test_live_source_metadata_is_declared_without_opening_devices() -> None:
    usb = make_usb_source_metadata(2, width=1280, height=720, frame_rate_hz=30.0)
    assert usb.source_uri == "usb://2"
    assert usb.expected_image_size is not None
    assert usb.expected_image_size.as_dict() == {"width": 1280, "height": 720}
    rtsp = make_rtsp_source_metadata("rtsp://user:secret@10.0.0.5/live")
    assert "secret" not in rtsp.source_uri
    assert "***" in rtsp.source_uri

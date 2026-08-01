"""Static-image capture lifecycle regression tests."""

from __future__ import annotations

import numpy as np

from core.capture.base_source import SourceStatus
from core.capture.image_source import ImageSource


def test_loaded_image_stays_connected_until_explicit_stop(monkeypatch):
    image = np.full((12, 16, 3), 127, dtype=np.uint8)
    monkeypatch.setattr("core.capture.image_source.cv2.imread", lambda *_a, **_k: image)

    source = ImageSource("example.png")
    statuses: list[SourceStatus] = []
    frames = []
    source.status_changed.connect(statuses.append)
    source.frame_ready.connect(frames.append)

    source._run()

    assert statuses == [SourceStatus.CONNECTED]
    assert source.status is SourceStatus.CONNECTED
    assert len(frames) == 1

    source.stop()

    assert statuses[-1] is SourceStatus.STOPPED
    assert source.status is SourceStatus.STOPPED


def test_resend_reuses_loaded_image_without_disk_io(monkeypatch):
    image = np.full((8, 10, 3), 42, dtype=np.uint8)
    reads = 0

    def _read(*_args, **_kwargs):
        nonlocal reads
        reads += 1
        return image

    monkeypatch.setattr("core.capture.image_source.cv2.imread", _read)

    source = ImageSource("example.png")
    frames = []
    source.frame_ready.connect(frames.append)

    source._run()
    source.resend()

    assert reads == 1
    assert len(frames) == 2
    assert np.array_equal(frames[0].image, frames[1].image)

from pathlib import Path

import numpy as np

from core.stable_video.export import StableVideoExportSession


class _FakeWriter:
    def __init__(self, path, *_args):
        self.path = Path(path)
        self.path.touch()
        self.frames = []
        self.opened = True

    def isOpened(self):
        return self.opened

    def write(self, frame):
        self.frames.append(np.array(frame, copy=True))

    def release(self):
        self.opened = False


def test_stable_video_export_is_atomic_and_writes_reports(tmp_path, monkeypatch):
    writers = []

    def factory(*args, **kwargs):
        writer = _FakeWriter(*args, **kwargs)
        writers.append(writer)
        return writer

    monkeypatch.setattr("core.stable_video.export.cv2.VideoWriter", factory)
    monkeypatch.setattr("core.stable_video.export.cv2.VideoWriter_fourcc", lambda *_: 0)

    output = tmp_path / "run"
    session = StableVideoExportSession(
        output,
        fps=20,
        model_id="mock",
        model_name="Mock",
        temporal_mode="generic",
        smoothing_strength=0.5,
        export_numeric_depth=True,
    )
    rgb = np.zeros((24, 32, 3), dtype=np.uint8)
    depth = np.linspace(1, 5, 24 * 32, dtype=np.float32).reshape(24, 32)
    session.append(rgb_bgr=rgb, depth=depth, performance={"latency_ms": 10.0})

    assert not output.exists()
    assert session.partial_dir.exists()
    summary = session.finalize()

    assert summary.completed is True
    assert summary.frame_count == 1
    assert output.exists()
    assert (output / "rgb_video.mp4").exists()
    assert (output / "depth_video.mp4").exists()
    assert (output / "performance.csv").exists()
    assert (output / "run_manifest.json").exists()
    assert (output / "depth_frames" / "depth_000000.npy").exists()
    assert len(writers) == 2
    assert len(writers[0].frames) == 1
    assert len(writers[1].frames) == 1


def test_stable_video_export_resizes_resolution_changes(tmp_path, monkeypatch):
    writers = []
    def factory(*args, **kwargs):
        writer = _FakeWriter(*args, **kwargs)
        writers.append(writer)
        return writer
    monkeypatch.setattr("core.stable_video.export.cv2.VideoWriter", factory)
    monkeypatch.setattr("core.stable_video.export.cv2.VideoWriter_fourcc", lambda *_: 0)
    session = StableVideoExportSession(
        tmp_path / "run", fps=10, model_id="mock", model_name="Mock",
        temporal_mode="generic", smoothing_strength=0.0,
    )
    session.append(
        rgb_bgr=np.zeros((10, 12, 3), dtype=np.uint8),
        depth=np.ones((10, 12), dtype=np.float32), performance={},
    )
    session.append(
        rgb_bgr=np.zeros((11, 13, 3), dtype=np.uint8),
        depth=np.ones((11, 13), dtype=np.float32), performance={},
    )
    summary = session.finalize()
    assert summary.frame_count == 2
    assert writers[0].frames[1].shape == (10, 12, 3)

from pathlib import Path

import numpy as np

from app.snapshot_controller import SnapshotRunState
from app.widgets.snapshot_page import SnapshotPage


def test_snapshot_page_capture_and_state(qtbot):
    page = SnapshotPage()
    qtbot.addWidget(page)
    page.set_model("Development Mock")
    page.set_captured_frame(np.zeros((12, 16, 3), dtype=np.uint8))
    assert page._run.isEnabled()
    page.apply_state(SnapshotRunState.RUNNING)
    assert not page._run.isEnabled()
    assert page._cancel.isEnabled()

    # Regression: SnapshotPage must use the actual PointCloudCanvas API.
    page._viewport.fit_requested.emit()
    page._viewport.reset_view_clicked.emit()
    assert page._viewport.camera_initialized


def test_snapshot_page_aligns_strided_colors_for_viewport(qtbot, tmp_path, monkeypatch):
    page = SnapshotPage()
    qtbot.addWidget(page)

    height, width, stride = 12, 16, 3
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[..., 0] = np.arange(width, dtype=np.uint8)[None, :]
    rgb[..., 1] = np.arange(height, dtype=np.uint8)[:, None]
    rgb[..., 2] = 200
    depth = np.ones((height, width), dtype=np.float32)
    points = np.zeros((4, 6, 3), dtype=np.float32)
    points[..., 2] = 1.0
    valid_mask = np.ones((height, width), dtype=bool)
    valid_mask[0, 0] = False

    output = Path(tmp_path)
    np.savez_compressed(
        output / "result.npz",
        rgb=rgb,
        depth=depth,
        points=points,
        colors=np.array([]),
        intrinsics=np.array([]),
        valid_mask=valid_mask,
        point_stride=np.asarray([stride], dtype=np.int32),
    )

    captured: dict[str, np.ndarray] = {}

    def _capture_points(point_values: np.ndarray, color_values: np.ndarray) -> None:
        captured["points"] = point_values
        captured["colors"] = color_values

    monkeypatch.setattr(page._canvas, "update_points", _capture_points)
    monkeypatch.setattr(page._canvas, "fit_scene", lambda: None)

    page.show_result(str(output))

    expected_colors = rgb[:, :, ::-1][::stride, ::stride].reshape(-1, 3)[1:]
    assert captured["points"].shape == (23, 3)
    assert captured["colors"].shape == (23, 3)
    assert np.array_equal(captured["colors"], expected_colors)
    assert page._open.isEnabled()

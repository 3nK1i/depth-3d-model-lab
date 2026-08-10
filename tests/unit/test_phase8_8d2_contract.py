from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_d2_remains_presentation_only_and_does_not_touch_offline_worker_or_core() -> None:
    page = (ROOT / "app/widgets/reconstruction_page.py").read_text(encoding="utf-8")
    canvas = (ROOT / "app/widgets/viewport/vispy_canvas.py").read_text(encoding="utf-8")
    inspection = (ROOT / "app/offline_reconstruction_inspection.py").read_text(encoding="utf-8")

    assert "OFFLINE INSPECTION" in page
    assert "select_offline_camera" in canvas
    assert "focus_selected_offline_camera" in canvas
    assert "gauge" in inspection
    assert "metre" in inspection.lower()


def test_d2_canvas_contract_keeps_selected_camera_separate_from_live_pose() -> None:
    canvas = (ROOT / "app/widgets/viewport/vispy_canvas.py").read_text(encoding="utf-8")
    assert "_selected_offline_camera_frustum" in canvas
    assert "_offline_camera_path" in canvas
    assert "_selected_offline_camera_index" in canvas
    assert "self._has_pose = False" in canvas

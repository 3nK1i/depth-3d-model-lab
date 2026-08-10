from __future__ import annotations

import ast
from dataclasses import fields
from pathlib import Path

from core.reconstruction.live_contracts import LiveMapSnapshot, LiveSlamObservation

ROOT = Path(__file__).resolve().parents[2]


def test_c3_native_worker_exports_bounded_sparse_map_samples_without_capture_ownership() -> None:
    text = (ROOT / "native/live_slam_worker/main.cc").read_text(encoding="utf-8")
    assert "kMaxSparseLandmarkSamples = 4096" in text
    assert "kSparseMapPublishIntervalFrames = 10" in text
    assert "get_pos_in_world" in text
    assert '"landmarks_reconstruction"' in text
    assert "VideoCapture" not in text
    assert "rtsp://" not in text
    assert "/dev/video" not in text


def test_c3_preserves_live_observation_and_base_map_snapshot_contracts() -> None:
    assert [field.name for field in fields(LiveSlamObservation)] == [
        "session_id", "frame_id", "sequence_index", "tracking_state", "quality",
        "pose", "is_keyframe", "keyframe_id", "map_snapshot",
    ]
    assert [field.name for field in fields(LiveMapSnapshot)] == [
        "revision", "keyframe_count", "landmark_count",
    ]


def test_c3_ui_sparse_map_toggle_cannot_start_or_stop_reconstruction() -> None:
    source = (ROOT / "app/widgets/reconstruction_page.py").read_text(encoding="utf-8")
    assert 'tooltip="Show or hide live sparse SLAM landmarks"' in source
    canvas_source = (ROOT / "app/widgets/viewport/vispy_canvas.py").read_text(encoding="utf-8")
    tree = ast.parse(canvas_source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ReconstructionCanvas")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "set_sparse_landmarks_visible")
    method_source = ast.get_source_segment(canvas_source, method) or ""
    for forbidden in ("start_selected_live", "request_stop_live", "stella", "pycolmap"):
        assert forbidden not in method_source.lower()


def test_c3_keeps_sparse_map_arbitrary_gauge_and_external_backend_isolation() -> None:
    helper = (ROOT / "app/reconstruction_visualization.py").read_text(encoding="utf-8")
    worker = (ROOT / "native/live_slam_worker/main.cc").read_text(encoding="utf-8")
    assert "arbitrary-gauge" in helper
    assert '"unit_to_metre", nullptr' in worker
    assert '"scale_status", "arbitrary"' in worker
    core_sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (ROOT / "core/reconstruction").glob("*.py")
    )
    assert "import stella_vslam" not in core_sources
    assert "from stella_vslam" not in core_sources

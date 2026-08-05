import numpy as np

from app.stable_video_controller import StableVideoController
from app.widgets.performance_snapshot import PerformanceSnapshot
from core.runtime.messages import InferenceResult


def test_stable_video_controller_accepts_many_frames_without_queue(tmp_path):
    controller = StableVideoController(tmp_path)
    controller.begin(
        output_dir=tmp_path / "run",
        model_id="development-mock",
        model_name="Development Mock",
        temporal_mode="generic",
        smoothing_strength=0.5,
        export_enabled=False,
        export_numeric_depth=False,
        export_fps=20,
    )
    controller.mark_running()
    rgb = np.zeros((12, 16, 3), dtype=np.uint8)
    for frame_id in range(250):
        result = InferenceResult(
            session_id=1,
            frame_id=frame_id,
            model_id="development-mock",
            depth=np.ones((12, 16), dtype=np.float32),
            latency_ms=2.0,
        )
        controller.consume_result(
            result,
            rgb,
            PerformanceSnapshot(status="RUNNING", latency_ms=3.0),
        )
    controller.finish()

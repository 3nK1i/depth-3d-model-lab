import numpy as np

from app.stable_video_controller import StableVideoController, StableVideoRunState
from app.widgets.performance_snapshot import PerformanceSnapshot
from core.runtime.messages import InferenceResult


def test_stable_video_controller_tracks_results_without_export(tmp_path, qtbot):
    controller = StableVideoController(tmp_path)
    seen = []
    controller.result_ready.connect(lambda *args: seen.append(args))
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
    result = InferenceResult(
        session_id=1,
        frame_id=1,
        model_id="development-mock",
        depth=np.ones((8, 10), dtype=np.float32),
        latency_ms=4.0,
    )
    rgb = np.zeros((8, 10, 3), dtype=np.uint8)
    controller.consume_result(
        result,
        rgb,
        PerformanceSnapshot(status="RUNNING", latency_ms=7.0),
    )
    assert len(seen) == 1
    controller.mark_paused()
    assert controller.state is StableVideoRunState.PAUSED
    controller.finish()
    assert controller.state is StableVideoRunState.COMPLETED

from pathlib import Path

import numpy as np

from app.snapshot_controller import SnapshotController, SnapshotRunState


def test_snapshot_controller_mock_job(qtbot, tmp_path):
    root = Path(__file__).resolve().parents[2]
    controller = SnapshotController(root)
    states = []
    controller.state_changed.connect(states.append)
    with qtbot.waitSignal(controller.result_ready, timeout=15000):
        controller.start(
            image=np.zeros((12, 16, 3), dtype=np.uint8),
            model_id="development-mock",
            config={"device": "cpu", "precision": "fp32", "latency_ms": 0},
            output_dir=tmp_path / "snapshot",
        )
    assert SnapshotRunState.RUNNING in states
    assert SnapshotRunState.COMPLETED in states

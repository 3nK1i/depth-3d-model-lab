from __future__ import annotations

from pathlib import Path
from time import monotonic

import numpy as np
import pytest

from app.application_controller import ApplicationController
from app.live_slam_controller import LiveStellaLaunchSettings
from app.reconstruction_controller import ReconstructionController
from app.widgets.original_view import OriginalSourceView
from app.widgets.source_bar import SourceBar
from core.capture.frame import SourceFrame
from core.capture.video_source import VideoSource
from core.reconstruction import (
    DeterministicMockLiveSlamBackend,
    LiveCaptureRuntimeState,
    MOCK_LIVE_SLAM_BACKEND_ID,
)
from tests.reconstruction_live_helpers import LIVE_IMAGE_SIZE, make_calibrated_live_camera


@pytest.fixture
def application_controller(qtbot, monkeypatch):
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    source_bar = SourceBar()
    original_view = OriginalSourceView()
    qtbot.addWidget(source_bar)
    qtbot.addWidget(original_view)
    controller = ApplicationController(source_bar, original_view)
    yield controller
    controller.shutdown()


def _frame(frame_id: int) -> SourceFrame:
    return SourceFrame(
        image=np.full(
            (LIVE_IMAGE_SIZE.height, LIVE_IMAGE_SIZE.width, 3),
            frame_id % 255,
            dtype=np.uint8,
        ),
        frame_id=frame_id,
        timestamp=monotonic() + frame_id * 0.001,
    )


def test_application_signal_to_reconstruction_controller_live_lifecycle(
    application_controller, monkeypatch, tmp_path: Path, qtbot
) -> None:
    video_path = tmp_path / "source.avi"
    video_path.write_bytes(b"phase8.8a2-test-video-identity")
    source = VideoSource(str(video_path))
    source._native_fps = 30.0
    application_controller._active_source = source

    monkeypatch.setattr(
        "app.live_slam_controller.ExternalStellaLiveSlamBackend",
        lambda **_kwargs: DeterministicMockLiveSlamBackend(),
    )
    monkeypatch.setattr(
        "app.live_slam_controller.STELLA_LIVE_BACKEND_ID",
        MOCK_LIVE_SLAM_BACKEND_ID,
    )

    # Match production ownership: MainWindow parents ReconstructionController.
    # Keeping the controller parented across pytest-qt's post-call processEvents()
    # prevents queued worker-thread deliveries from targeting an object whose
    # Python local lifetime ended with the test function.
    controller = ReconstructionController(
        application_controller,
        parent=application_controller,
    )
    observations: list[object] = []
    states: list[LiveCaptureRuntimeState] = []
    controller.live_observation_ready.connect(observations.append)
    controller.live_state_changed.connect(states.append)

    try:
        session_id = controller.start_live(
            camera=make_calibrated_live_camera(),
            settings=LiveStellaLaunchSettings(
                install_prefix=tmp_path / "stella",
                worker_executable=tmp_path / "worker",
                vocabulary_path=tmp_path / "vocab",
            ),
            session_id="00000000-0000-4000-8000-0000000088a2",
        )
        assert session_id == "00000000-0000-4000-8000-0000000088a2"

        runtime = controller.live_slam_controller.runtime
        assert runtime is not None
        assert runtime.wait_for_state(LiveCaptureRuntimeState.RUNNING, timeout_s=2.0)

        for index in range(8):
            application_controller.frame_captured.emit(_frame(index))
            qtbot.wait(15)

        # Runtime callbacks originate on the live worker thread. Qt therefore queues
        # the forwarded controller signal back to the GUI thread; the integration
        # test must pump the Qt event loop instead of blocking it with time.sleep().
        qtbot.waitUntil(lambda: bool(observations), timeout=2_000)

        assert controller.pause_live() is True
        assert controller.live_state is LiveCaptureRuntimeState.PAUSED
        paused_count = len(observations)
        application_controller.frame_captured.emit(_frame(20))
        qtbot.wait(50)
        assert len(observations) == paused_count

        assert controller.resume_live() is True
        application_controller.frame_captured.emit(_frame(21))
        qtbot.waitUntil(lambda: len(observations) > paused_count, timeout=2_000)

        assert controller.request_stop_live() is True
        assert runtime.wait_for_state(LiveCaptureRuntimeState.STOPPED, timeout_s=2.0)
        qtbot.waitUntil(
            lambda: LiveCaptureRuntimeState.STOPPED in states,
            timeout=2_000,
        )
        assert LiveCaptureRuntimeState.STOPPING in states
        assert LiveCaptureRuntimeState.STOPPED in states
        assert controller.live_stats is not None
        assert controller.live_stats.cleanup_complete is True
        assert controller.stop_live(timeout_s=1.0) is True
        assert controller.live_slam_controller.runtime is None

        second_session = controller.start_live(
            camera=make_calibrated_live_camera(),
            settings=LiveStellaLaunchSettings(
                install_prefix=tmp_path / "stella",
                worker_executable=tmp_path / "worker",
                vocabulary_path=tmp_path / "vocab",
            ),
            session_id="00000000-0000-4000-8000-0000000088a3",
        )
        assert second_session == "00000000-0000-4000-8000-0000000088a3"
        runtime2 = controller.live_slam_controller.runtime
        assert runtime2 is not None
        assert runtime2.wait_for_state(LiveCaptureRuntimeState.RUNNING, timeout_s=2.0)
        application_controller.frame_captured.emit(_frame(30))
        qtbot.waitUntil(
            lambda: runtime2.stats.observations >= 1,
            timeout=2_000,
        )
        assert controller.request_stop_live() is True
        assert runtime2.wait_for_state(LiveCaptureRuntimeState.STOPPED, timeout_s=2.0)
        assert controller.stop_live(timeout_s=1.0) is True

    finally:
        # Never leave the Python worker thread or queued Qt deliveries alive on
        # an assertion failure. Production also shuts reconstruction down before
        # its QObject owner is destroyed.
        application_controller._active_source = None
        controller.shutdown()
        qtbot.wait(50)

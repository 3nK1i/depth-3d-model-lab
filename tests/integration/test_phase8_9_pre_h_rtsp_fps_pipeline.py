from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from app.application_controller import ApplicationController
from app.live_slam_controller import LiveSlamCaptureController
from app.widgets.original_view import OriginalSourceView
from app.widgets.source_bar import SourceBar
from core.capture.rtsp_source import RtspSource
from core.reconstruction import (
    LiveSlamRequest,
    ReconstructionQualityProfile,
    STELLA_LIVE_BACKEND_ID,
    stella_monocular_config_text,
)
from tests.reconstruction_live_helpers import make_calibrated_live_camera


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


def test_rtsp_decoder_fps_reaches_generated_stella_config(application_controller) -> None:
    source = RtspSource("rtsp://operator:secret@camera.local/live")
    source._native_fps = 30.0
    application_controller._active_source = source

    bridge = LiveSlamCaptureController(application_controller)
    camera = make_calibrated_live_camera()
    metadata, simulated_live = bridge._build_active_source_metadata(camera)

    assert simulated_live is False
    assert metadata.expected_frame_rate_hz == 30.0
    assert "secret" not in metadata.source_uri

    request = LiveSlamRequest(
        session_id="00000000-0000-4000-8000-0000000089f0",
        source=metadata,
        camera=camera,
        profile=ReconstructionQualityProfile.EFFICIENT,
        backend_id=STELLA_LIVE_BACKEND_ID,
    )
    config = stella_monocular_config_text(request)
    assert "  fps: 30" in config

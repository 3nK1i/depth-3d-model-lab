from __future__ import annotations

from pathlib import Path

import pytest

from app.application_controller import ApplicationController
from app.live_slam_controller import LiveStellaLaunchSettings
from app.reconstruction_controller import ReconstructionController
from app.widgets.original_view import OriginalSourceView
from app.widgets.source_bar import SourceBar
from core.capture.base_source import SourceStatus
from core.reconstruction import LiveCaptureRuntimeState, ReconstructionQualityProfile
from core.reconstruction.errors import ReconstructionValidationError
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


def _settings(tmp_path: Path) -> LiveStellaLaunchSettings:
    return LiveStellaLaunchSettings(
        install_prefix=tmp_path / "stella",
        worker_executable=tmp_path / "worker",
        vocabulary_path=tmp_path / "orb_vocab.fbow",
    )


def test_start_live_delegates_to_existing_bridge_and_records_session(
    application_controller, monkeypatch, tmp_path
) -> None:
    controller = ReconstructionController(application_controller)
    camera = make_calibrated_live_camera()
    settings = _settings(tmp_path)
    calls: list[dict[str, object]] = []

    def fake_start(**kwargs):
        calls.append(kwargs)
        return "recon-a2-session"

    monkeypatch.setattr(
        controller.live_slam_controller,
        "start_stella_for_active_source",
        fake_start,
    )

    session_id = controller.start_live(
        camera=camera,
        settings=settings,
        profile=ReconstructionQualityProfile.EFFICIENT,
        session_id="requested-session",
    )

    assert session_id == "recon-a2-session"
    assert controller.live_session_id == "recon-a2-session"
    assert calls == [
        {
            "camera": camera,
            "settings": settings,
            "profile": ReconstructionQualityProfile.EFFICIENT,
            "session_id": "requested-session",
        }
    ]

    controller.shutdown()


def test_invalid_live_start_preserves_dormant_state_and_session_identity(
    application_controller, tmp_path
) -> None:
    controller = ReconstructionController(application_controller)

    with pytest.raises(
        ReconstructionValidationError,
        match="Connect a camera, RTSP stream, or video first",
    ):
        controller.start_live(
            camera=make_calibrated_live_camera(),
            settings=_settings(tmp_path),
        )

    assert controller.live_session_id is None
    assert controller.live_state is LiveCaptureRuntimeState.IDLE
    assert controller.live_slam_controller.runtime is None

    controller.shutdown()


def test_live_lifecycle_methods_are_thin_delegates(
    application_controller, monkeypatch
) -> None:
    controller = ReconstructionController(application_controller)
    calls: list[object] = []

    monkeypatch.setattr(
        controller.live_slam_controller,
        "pause",
        lambda: calls.append("pause") or True,
    )
    monkeypatch.setattr(
        controller.live_slam_controller,
        "resume",
        lambda: calls.append("resume") or True,
    )
    monkeypatch.setattr(
        controller.live_slam_controller,
        "request_stop",
        lambda: calls.append("request_stop") or True,
    )
    monkeypatch.setattr(
        controller.live_slam_controller,
        "stop",
        lambda *, timeout_s: calls.append(("stop", timeout_s)) or True,
    )

    assert controller.pause_live() is True
    assert controller.resume_live() is True
    assert controller.request_stop_live() is True
    assert controller.stop_live(timeout_s=1.25) is True
    assert calls == ["pause", "resume", "request_stop", ("stop", 1.25)]

    controller.shutdown()


def test_shutdown_blocks_new_live_start_and_ui_lifecycle_actions(
    application_controller, tmp_path
) -> None:
    controller = ReconstructionController(application_controller)
    controller.shutdown()

    with pytest.raises(ReconstructionValidationError, match="shut down"):
        controller.start_live(
            camera=make_calibrated_live_camera(),
            settings=_settings(tmp_path),
        )

    assert controller.pause_live() is False
    assert controller.resume_live() is False
    assert controller.request_stop_live() is False


def test_live_bridge_request_stop_is_nonblocking_delegate(application_controller) -> None:
    controller = ReconstructionController(application_controller)
    bridge = controller.live_slam_controller
    calls: list[str] = []

    class Runtime:
        def request_stop(self) -> bool:
            calls.append("request_stop")
            return True

    bridge._runtime = Runtime()  # type: ignore[assignment]

    assert bridge.request_stop() is True
    assert calls == ["request_stop"]

    bridge._runtime = None
    controller.shutdown()


def test_existing_source_lifecycle_rules_remain_owned_by_live_bridge(
    application_controller,
) -> None:
    controller = ReconstructionController(application_controller)
    bridge = controller.live_slam_controller
    calls: list[str] = []

    class Runtime:
        is_active = True

        def pause(self) -> bool:
            calls.append("pause")
            return True

        def resume(self) -> bool:
            calls.append("resume")
            return True

        def request_stop(self) -> bool:
            calls.append("request_stop")
            return True

    bridge._runtime = Runtime()  # type: ignore[assignment]

    application_controller.source_status_changed.emit(SourceStatus.RECONNECTING)
    application_controller.source_status_changed.emit(SourceStatus.CONNECTED)
    application_controller.source_status_changed.emit(SourceStatus.ERROR)
    bridge._on_source_type_changed(object())

    assert calls == ["pause", "resume", "request_stop", "request_stop"]

    bridge._runtime = None
    controller.shutdown()


def test_a2_keeps_authoritative_live_state_contract(application_controller) -> None:
    controller = ReconstructionController(application_controller)
    states: list[object] = []
    controller.live_state_changed.connect(states.append)

    controller.live_slam_controller.state_changed.emit(LiveCaptureRuntimeState.STARTING)
    controller.live_slam_controller.state_changed.emit(LiveCaptureRuntimeState.RUNNING)
    controller.live_slam_controller.state_changed.emit(LiveCaptureRuntimeState.PAUSED)
    controller.live_slam_controller.state_changed.emit(LiveCaptureRuntimeState.STOPPING)
    controller.live_slam_controller.state_changed.emit(LiveCaptureRuntimeState.STOPPED)

    assert states == [
        LiveCaptureRuntimeState.STARTING,
        LiveCaptureRuntimeState.RUNNING,
        LiveCaptureRuntimeState.PAUSED,
        LiveCaptureRuntimeState.STOPPING,
        LiveCaptureRuntimeState.STOPPED,
    ]

    controller.shutdown()

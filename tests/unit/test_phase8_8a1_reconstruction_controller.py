from __future__ import annotations

from pathlib import Path

import pytest

from app.application_controller import ApplicationController
from app.reconstruction_controller import ReconstructionController
from app.widgets.original_view import OriginalSourceView
from app.widgets.source_bar import SourceBar
from core.reconstruction import LiveCaptureRuntimeState


ROOT = Path(__file__).resolve().parents[2]


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


def test_controller_is_dormant_and_reuses_authoritative_live_state(application_controller) -> None:
    controller = ReconstructionController(application_controller)

    assert controller.live_state is LiveCaptureRuntimeState.IDLE
    assert controller.is_active is False
    assert controller.live_slam_controller.runtime is None
    assert controller.is_shutdown is False

    controller.shutdown()


def test_controller_forwards_live_runtime_contracts_without_translation(application_controller) -> None:
    controller = ReconstructionController(application_controller)
    states: list[object] = []
    observations: list[object] = []
    errors: list[str] = []
    sentinel = object()

    controller.live_state_changed.connect(states.append)
    controller.live_observation_ready.connect(observations.append)
    controller.error_occurred.connect(errors.append)

    controller.live_slam_controller.state_changed.emit(LiveCaptureRuntimeState.PAUSED)
    controller.live_slam_controller.observation_ready.emit(sentinel)
    controller.live_slam_controller.error_occurred.emit("synthetic bridge error")

    assert states == [LiveCaptureRuntimeState.PAUSED]
    assert observations == [sentinel]
    assert errors == ["synthetic bridge error"]

    controller.shutdown()


def test_shutdown_is_idempotent_and_delegates_to_owned_live_bridge(
    application_controller, monkeypatch
) -> None:
    controller = ReconstructionController(application_controller)
    calls: list[str] = []
    monkeypatch.setattr(
        controller.live_slam_controller,
        "shutdown",
        lambda: calls.append("shutdown"),
    )

    controller.shutdown()
    controller.shutdown()

    assert calls == ["shutdown"]
    assert controller.is_shutdown is True


def test_a1_controller_has_no_heavy_backend_imports_or_start_side_effects() -> None:
    text = (ROOT / "app" / "reconstruction_controller.py").read_text(encoding="utf-8")
    lowered = text.lower()

    for forbidden in (
        "import pycolmap",
        "from pycolmap",
        "import gsplat",
        "from gsplat",
        "import nerfstudio",
        "from nerfstudio",
        "import depth_anything_3",
        "from depth_anything_3",
        "import stella_vslam",
        "from stella_vslam",
    ):
        assert forbidden not in lowered

    assert "start_stella_for_active_source(" not in text

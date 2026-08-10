from __future__ import annotations

import pytest

from app.main_window import MainWindow
from app.widgets.mode_tabs import AppMode
from core.runtime.state_machine import ApplicationState


@pytest.fixture
def window(qtbot, monkeypatch):
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    win = MainWindow()
    qtbot.addWidget(win)
    return win


def test_main_window_installs_reconstruction_resource_guards(window) -> None:
    controller = window._reconstruction_controller
    assert controller._live_start_guard is not None
    assert controller._offline_start_guard is not None
    assert controller._live_start_guard() is None


def test_shared_long_running_mode_blocks_reconstruction_start_guard(window, monkeypatch) -> None:
    monkeypatch.setattr(
        type(window._benchmark_controller),
        "is_running",
        property(lambda _self: True),
    )
    reason = window._reconstruction_live_start_guard()
    assert reason is not None
    assert "Benchmark is running" in reason


def test_offline_guard_blocks_busy_inference_then_releases_idle_model(window, monkeypatch) -> None:
    releases: list[bool] = []
    monkeypatch.setattr(
        window._inference_controller,
        "release_model_for_external_job",
        lambda: releases.append(True) or True,
    )
    window._inference_controller._state_machine._state = ApplicationState.RUNNING
    reason = window._reconstruction_offline_start_guard()
    assert reason is not None
    assert "Live 3D inference is busy" in reason
    assert releases == []

    window._inference_controller._state_machine._state = ApplicationState.SOURCE_READY
    assert window._reconstruction_offline_start_guard() is None
    assert releases == [True]


def test_a5_readiness_gate_still_keeps_reconstruction_ui_disabled(window) -> None:
    reconstruction_button = window._mode_tabs._buttons[AppMode.RECONSTRUCTION]
    assert reconstruction_button.isEnabled() is False
    assert window._mode_stack.count() == 6
    with pytest.raises(ValueError, match="Mode is not enabled"):
        window._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)

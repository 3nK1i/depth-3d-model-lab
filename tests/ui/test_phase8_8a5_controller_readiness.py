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


def test_a5_readiness_contract_survives_b5_navigation_enablement(window) -> None:
    reconstruction_button = window._mode_tabs._buttons[AppMode.RECONSTRUCTION]
    assert reconstruction_button.isEnabled() is True
    assert window._mode_stack.count() == 6
    window._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)
    assert window._mode_tabs.active_mode is AppMode.RECONSTRUCTION
    assert window._mode_stack.currentWidget() is window._reconstruction_page
    assert window._reconstruction_controller.is_active is False

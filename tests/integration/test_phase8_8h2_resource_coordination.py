from __future__ import annotations

import pytest

from app.main_window import MainWindow
from app.reconstruction_controller import ReconstructionController
from app.widgets.mode_tabs import AppMode
from core.runtime.state_machine import ApplicationState


@pytest.fixture
def window(qtbot, monkeypatch):
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    win = MainWindow()
    qtbot.addWidget(win)
    yield win
    win.close()


def test_h2_reciprocal_resource_matrix_transitions_from_owned_to_idle(window, monkeypatch, qtbot) -> None:
    active = {"value": True}
    monkeypatch.setattr(
        ReconstructionController,
        "is_active",
        property(lambda _self: active["value"]),
    )
    window._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)
    window._inference_controller._state_machine._state = ApplicationState.DISCONNECTED

    snapshot_calls: list[object] = []
    stable_calls: list[object] = []
    benchmark_calls: list[object] = []
    monkeypatch.setattr(window._snapshot_controller, "start", lambda **kwargs: snapshot_calls.append(kwargs))
    monkeypatch.setattr(window._stable_video_controller, "begin", lambda **kwargs: stable_calls.append(kwargs))
    monkeypatch.setattr(window._benchmark_controller, "start", lambda config: benchmark_calls.append(config))

    window._on_reconstruction_activity_changed()
    window._run_snapshot()
    window._start_stable_video()
    window._on_benchmark_run_requested(object())
    assert snapshot_calls == stable_calls == benchmark_calls == []
    assert window._inspection_jobs_conflict()
    assert not window._mode_tabs._buttons[AppMode.BENCHMARK].isEnabled()

    active["value"] = False
    window._on_reconstruction_activity_changed()
    qtbot.wait(10)
    assert window._reconstruction_conflict_for_application_job("Benchmark") == ""
    assert window._mode_tabs._buttons[AppMode.BENCHMARK].isEnabled()
    assert window._benchmark_page._runtime_safe is True


def test_h2_resource_coordination_never_promotes_static_appearance_to_runtime_owner(window) -> None:
    # H2 ownership is intentionally based on active Reconstruction execution only.
    # Persistent/static UI artifacts do not reserve application resources.
    assert window._reconstruction_controller.is_active is False
    assert window._reconstruction_conflict_for_application_job("Benchmark") == ""

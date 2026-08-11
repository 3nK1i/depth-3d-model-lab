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


def _set_reconstruction_active(monkeypatch, value: bool = True) -> dict[str, bool]:
    state = {"active": bool(value)}
    monkeypatch.setattr(
        ReconstructionController,
        "is_active",
        property(lambda _self: state["active"]),
    )
    return state


def test_h2_active_reconstruction_blocks_snapshot_before_controller_start(window, monkeypatch) -> None:
    _set_reconstruction_active(monkeypatch)
    calls: list[object] = []
    monkeypatch.setattr(window._snapshot_controller, "start", lambda **kwargs: calls.append(kwargs))
    window._run_snapshot()
    assert calls == []
    assert "Reconstruction is active" in window._snapshot_page._log.toPlainText()


def test_h2_active_reconstruction_blocks_stable_video_before_controller_start(window, monkeypatch) -> None:
    _set_reconstruction_active(monkeypatch)
    calls: list[object] = []
    monkeypatch.setattr(window._stable_video_controller, "begin", lambda **kwargs: calls.append(kwargs))
    window._start_stable_video()
    assert calls == []
    assert "Reconstruction is active" in window._stable_video_page._log.toPlainText()


def test_h2_active_reconstruction_blocks_benchmark_before_controller_start(window, monkeypatch) -> None:
    _set_reconstruction_active(monkeypatch)
    calls: list[object] = []
    monkeypatch.setattr(window._benchmark_controller, "start", lambda configuration: calls.append(configuration))
    window._on_benchmark_run_requested(object())
    assert calls == []
    assert "Reconstruction is active" in window._benchmark_page._log.toPlainText()


def test_h2_active_reconstruction_blocks_inspection_analysis_and_recompute(window, monkeypatch) -> None:
    _set_reconstruction_active(monkeypatch)
    assert "Reconstruction is active" in window._inspection_jobs_conflict()
    analysis_calls: list[object] = []
    recompute_calls: list[object] = []
    monkeypatch.setattr(window._inspection_controller, "start_analysis", lambda **kwargs: analysis_calls.append(kwargs))
    monkeypatch.setattr(window._inspection_controller, "recompute", lambda **kwargs: recompute_calls.append(kwargs))
    window._start_inspection_analysis({})
    window._recompute_inspection({})
    assert analysis_calls == []
    assert recompute_calls == []


def test_h2_active_reconstruction_blocks_export_before_file_dialog(window, monkeypatch) -> None:
    _set_reconstruction_active(monkeypatch)
    failures: list[str] = []
    monkeypatch.setattr(window._inspection_page, "show_export_failure", failures.append)
    monkeypatch.setattr(
        "app.main_window.QFileDialog.getExistingDirectory",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("dialog must not open")),
    )
    window._export_inspection("all")
    assert failures and "Reconstruction is active" in failures[-1]


def test_h2_resource_owner_updates_navigation_and_benchmark_stable_readiness(window, monkeypatch, qtbot) -> None:
    state = _set_reconstruction_active(monkeypatch)
    window._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)
    window._inference_controller._state_machine._state = ApplicationState.DISCONNECTED
    window._on_reconstruction_activity_changed()
    qtbot.wait(10)
    assert window._mode_tabs._buttons[AppMode.RECONSTRUCTION].isEnabled()
    assert not window._mode_tabs._buttons[AppMode.BENCHMARK].isEnabled()
    assert window._benchmark_page._runtime_safe is False
    assert "Reconstruction is active" in window._benchmark_page._runtime_block_reason
    assert not window._stable_video_page._start.isEnabled()
    assert "Reconstruction is active" in window._stable_video_page._status.text()

    state["active"] = False
    window._on_reconstruction_activity_changed()
    qtbot.wait(10)
    assert window._mode_tabs._buttons[AppMode.BENCHMARK].isEnabled()
    assert window._benchmark_page._runtime_safe is True
    assert "Reconstruction is active" not in window._stable_video_page._status.text()


def test_h2_live_reconstruction_preserves_shared_inference_without_model_release(window, monkeypatch) -> None:
    releases: list[bool] = []
    monkeypatch.setattr(window, "_reconstruction_shared_job_conflict", lambda: None)
    monkeypatch.setattr(
        window._inference_controller,
        "release_model_for_external_job",
        lambda: releases.append(True) or True,
    )
    window._inference_controller._state_machine._state = ApplicationState.RUNNING
    assert window._reconstruction_live_start_guard() is None
    assert releases == []


def test_h2_offline_reconstruction_still_releases_idle_model_and_blocks_busy_inference(window, monkeypatch) -> None:
    releases: list[bool] = []
    monkeypatch.setattr(window, "_reconstruction_shared_job_conflict", lambda: None)
    monkeypatch.setattr(
        window._inference_controller,
        "release_model_for_external_job",
        lambda: releases.append(True) or True,
    )
    window._inference_controller._state_machine._state = ApplicationState.RUNNING
    assert "Live 3D inference is busy" in window._reconstruction_offline_start_guard()
    assert releases == []
    window._inference_controller._state_machine._state = ApplicationState.SOURCE_READY
    assert window._reconstruction_offline_start_guard() is None
    assert releases == [True]

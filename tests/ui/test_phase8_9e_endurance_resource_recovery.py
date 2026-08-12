from __future__ import annotations

from pathlib import Path

import pytest

from app.offline_reconstruction_controller import OfflineReconstructionRunState
from app.reconstruction_controller import ReconstructionController
from app.widgets.mode_tabs import AppMode
from tests.reconstruction_h3_helpers import offline_configuration, write_h3_worker, write_ready_sequence


@pytest.fixture
def window(qtbot, monkeypatch, tmp_path: Path):
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    worker = write_h3_worker(tmp_path / "phase8_9e_worker.py")
    real_controller = ReconstructionController

    def controller_factory(application_controller, parent=None):
        return real_controller(application_controller, parent, offline_worker_script=worker)

    monkeypatch.setattr("app.main_window.ReconstructionController", controller_factory)
    from app.main_window import MainWindow

    win = MainWindow()
    qtbot.addWidget(win)
    win._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)
    yield win
    win.close()


def _wait_terminal(window, finished: list[dict], expected_count: int, qtbot) -> None:
    controller = window._reconstruction_controller
    qtbot.waitUntil(lambda: len(finished) >= expected_count, timeout=5_000)
    qtbot.waitUntil(lambda: not controller.is_active, timeout=2_000)
    qtbot.wait(15)


def test_e_repeated_cancel_recovery_releases_application_resources_every_cycle(
    window, qtbot, tmp_path: Path
) -> None:
    controller = window._reconstruction_controller
    controller.select_image_sequence(write_ready_sequence(tmp_path))
    finished: list[dict] = []
    controller.offline_run_finished.connect(finished.append)

    for index in range(5):
        controller.start_offline(offline_configuration(tmp_path, f"cancel-e-{index:02d}"))
        qtbot.waitUntil(
            lambda: controller.offline_state is OfflineReconstructionRunState.RUNNING,
            timeout=2_000,
        )
        assert not window._mode_tabs._buttons[AppMode.BENCHMARK].isEnabled()
        assert controller.cancel_offline() is True
        _wait_terminal(window, finished, index + 1, qtbot)
        assert controller.offline_state is OfflineReconstructionRunState.CANCELLED
        assert controller.offline_is_running is False
        assert controller.is_active is False
        assert window._mode_tabs._buttons[AppMode.BENCHMARK].isEnabled()


def test_e_failure_after_repeated_cancellation_still_recovers_and_shutdown_is_clean(
    window, qtbot, tmp_path: Path
) -> None:
    controller = window._reconstruction_controller
    controller.select_image_sequence(write_ready_sequence(tmp_path))
    finished: list[dict] = []
    controller.offline_run_finished.connect(finished.append)

    for index in range(3):
        controller.start_offline(offline_configuration(tmp_path, f"cancel-before-fail-{index:02d}"))
        qtbot.waitUntil(
            lambda: controller.offline_state is OfflineReconstructionRunState.RUNNING,
            timeout=2_000,
        )
        assert controller.cancel_offline()
        _wait_terminal(window, finished, index + 1, qtbot)

    controller.start_offline(offline_configuration(tmp_path, "fail-after-cycles"))
    _wait_terminal(window, finished, 4, qtbot)
    assert controller.offline_state is OfflineReconstructionRunState.FAILED
    assert controller.is_active is False
    assert window._mode_tabs._buttons[AppMode.BENCHMARK].isEnabled()

    controller.start_offline(offline_configuration(tmp_path, "recovered-success"))
    _wait_terminal(window, finished, 5, qtbot)
    assert controller.offline_state is OfflineReconstructionRunState.COMPLETED
    assert controller.is_active is False
    assert window._mode_tabs._buttons[AppMode.BENCHMARK].isEnabled()

    window.close()
    qtbot.wait(20)
    assert controller.is_shutdown
    assert controller.offline_controller.is_shutdown
    assert controller.offline_is_running is False
    assert controller.is_active is False

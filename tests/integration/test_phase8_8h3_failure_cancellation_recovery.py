from __future__ import annotations

from pathlib import Path

import pytest

from app.application_controller import ApplicationController
from app.offline_reconstruction_controller import OfflineReconstructionRunState
from app.reconstruction_controller import ReconstructionController
from app.widgets.original_view import OriginalSourceView
from app.widgets.source_bar import SourceBar
from tests.reconstruction_h3_helpers import offline_configuration, write_h3_worker


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


def _wait_finished(controller: ReconstructionController, finished: list[dict], qtbot) -> None:
    qtbot.waitUntil(lambda: bool(finished), timeout=5_000)
    qtbot.waitUntil(lambda: not controller.is_active, timeout=2_000)


def test_h3_offline_failure_stays_resource_owned_until_run_finished(
    application_controller, tmp_path: Path, qtbot
) -> None:
    worker = write_h3_worker(tmp_path / "h3_worker.py")
    controller = ReconstructionController(
        application_controller,
        parent=application_controller,
        offline_worker_script=worker,
    )
    terminal_active: list[bool] = []
    finished: list[dict] = []

    def on_state(state: OfflineReconstructionRunState) -> None:
        if state is OfflineReconstructionRunState.FAILED:
            terminal_active.append(controller.is_active)

    controller.offline_state_changed.connect(on_state)
    controller.offline_run_finished.connect(finished.append)
    try:
        controller.start_offline(offline_configuration(tmp_path, "fail-run"))
        _wait_finished(controller, finished, qtbot)
        assert controller.offline_state is OfflineReconstructionRunState.FAILED
        assert terminal_active == [True]
        assert controller.is_active is False
        assert finished[-1]["result_accepted"] is False
    finally:
        controller.shutdown()


def test_h3_offline_cancel_stays_resource_owned_until_run_finished(
    application_controller, tmp_path: Path, qtbot
) -> None:
    worker = write_h3_worker(tmp_path / "h3_worker.py")
    controller = ReconstructionController(
        application_controller,
        parent=application_controller,
        offline_worker_script=worker,
    )
    terminal_active: list[bool] = []
    finished: list[dict] = []

    def on_state(state: OfflineReconstructionRunState) -> None:
        if state is OfflineReconstructionRunState.CANCELLED:
            terminal_active.append(controller.is_active)

    controller.offline_state_changed.connect(on_state)
    controller.offline_run_finished.connect(finished.append)
    try:
        controller.start_offline(offline_configuration(tmp_path, "cancel-run"))
        qtbot.waitUntil(
            lambda: controller.offline_state is OfflineReconstructionRunState.RUNNING,
            timeout=2_000,
        )
        assert controller.cancel_offline() is True
        _wait_finished(controller, finished, qtbot)
        assert controller.offline_state is OfflineReconstructionRunState.CANCELLED
        assert terminal_active == [True]
        assert controller.is_active is False
        assert finished[-1]["result_accepted"] is False
    finally:
        controller.shutdown()


def test_h3_shutdown_cancels_active_offline_worker_and_blocks_restart(
    application_controller, tmp_path: Path, qtbot
) -> None:
    worker = write_h3_worker(tmp_path / "h3_worker.py")
    controller = ReconstructionController(
        application_controller,
        parent=application_controller,
        offline_worker_script=worker,
    )
    controller.start_offline(offline_configuration(tmp_path, "shutdown-run"))
    qtbot.waitUntil(
        lambda: controller.offline_state is OfflineReconstructionRunState.RUNNING,
        timeout=2_000,
    )

    controller.shutdown()
    qtbot.wait(25)

    assert controller.is_shutdown
    assert controller.offline_controller.is_shutdown
    assert controller.offline_is_running is False
    assert controller.is_active is False

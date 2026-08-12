from __future__ import annotations

from pathlib import Path

import pytest

from app.offline_reconstruction_controller import OfflineReconstructionRunState
from app.reconstruction_controller import ReconstructionController
from app.reconstruction_failure_recovery import ReconstructionFailureRecoveryState
from app.widgets.mode_tabs import AppMode
from tests.reconstruction_h1_helpers import create_checkpoint_fixture, run_whole_pipeline
from tests.reconstruction_h3_helpers import offline_configuration, write_h3_worker, write_ready_sequence


@pytest.fixture
def window(qtbot, monkeypatch, tmp_path: Path):
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    worker = write_h3_worker(tmp_path / "h3_window_worker.py")
    real_controller = ReconstructionController

    def controller_factory(application_controller, parent=None):
        return real_controller(
            application_controller,
            parent,
            offline_worker_script=worker,
        )

    monkeypatch.setattr("app.main_window.ReconstructionController", controller_factory)
    from app.main_window import MainWindow

    win = MainWindow()
    qtbot.addWidget(win)
    win._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)
    yield win
    win.close()


def _wait_terminal(window, finished: list[dict], qtbot) -> None:
    controller = window._reconstruction_controller
    qtbot.waitUntil(lambda: bool(finished), timeout=5_000)
    qtbot.waitUntil(lambda: not controller.is_active, timeout=2_000)
    qtbot.wait(25)


def test_h3_real_offline_failure_unlocks_resources_and_offers_fresh_retry(
    window, qtbot, tmp_path: Path
) -> None:
    controller = window._reconstruction_controller
    page = window._reconstruction_page
    controller.select_image_sequence(write_ready_sequence(tmp_path))
    finished: list[dict] = []
    controller.offline_run_finished.connect(finished.append)

    controller.start_offline(offline_configuration(tmp_path, "fail-ui"))
    qtbot.waitUntil(
        lambda: controller.offline_state is OfflineReconstructionRunState.RUNNING,
        timeout=2_000,
    )
    assert not window._mode_tabs._buttons[AppMode.BENCHMARK].isEnabled()

    _wait_terminal(window, finished, qtbot)

    assert controller.offline_state is OfflineReconstructionRunState.FAILED
    assert window._mode_tabs._buttons[AppMode.BENCHMARK].isEnabled()
    assert page.failure_recovery_presentation is not None
    assert page.failure_recovery_presentation.state is ReconstructionFailureRecoveryState.RETRY_AVAILABLE
    assert page._session_retry_button.isEnabled()
    assert not page._start_button.isEnabled()


def test_h3_real_offline_cancel_unlocks_resources_and_offers_fresh_retry(
    window, qtbot, tmp_path: Path
) -> None:
    controller = window._reconstruction_controller
    page = window._reconstruction_page
    controller.select_image_sequence(write_ready_sequence(tmp_path))
    finished: list[dict] = []
    controller.offline_run_finished.connect(finished.append)

    controller.start_offline(offline_configuration(tmp_path, "cancel-ui"))
    qtbot.waitUntil(
        lambda: controller.offline_state is OfflineReconstructionRunState.RUNNING,
        timeout=2_000,
    )
    assert controller.cancel_offline() is True
    _wait_terminal(window, finished, qtbot)

    assert controller.offline_state is OfflineReconstructionRunState.CANCELLED
    assert window._mode_tabs._buttons[AppMode.BENCHMARK].isEnabled()
    assert page.failure_recovery_presentation is not None
    assert page.failure_recovery_presentation.state is ReconstructionFailureRecoveryState.CANCELLED_RETRY_AVAILABLE
    assert page._session_retry_button.isEnabled()


def test_h3_failed_same_session_with_scene_checkpoint_routes_to_prepare_resume(
    window, qtbot, tmp_path: Path
) -> None:
    controller = window._reconstruction_controller
    page = window._reconstruction_page
    fixture, result = run_whole_pipeline()
    controller.adopt_pipeline_result(result, (tmp_path / "scene").resolve())
    _checkpoint, prefix = create_checkpoint_fixture(fixture)
    created = controller.create_engine_checkpoint(
        plan=fixture.plan,
        providers=fixture.providers,
        stage_results=prefix,
        output_dir=(tmp_path / "checkpoint").resolve(),
    )
    controller.set_engine_resume_context(
        plan=fixture.plan,
        providers=fixture.providers,
        available_payload_refs=created.payload_refs,
    )
    controller.select_image_sequence(write_ready_sequence(tmp_path))
    finished: list[dict] = []
    controller.offline_run_finished.connect(finished.append)

    controller.start_offline(
        offline_configuration(tmp_path, "fail-preserved"),
        session_id=fixture.request.session_id,
    )
    _wait_terminal(window, finished, qtbot)

    assert page.failure_recovery_presentation is not None
    assert page.failure_recovery_presentation.state is ReconstructionFailureRecoveryState.PRESERVED_EVIDENCE
    assert not page._session_retry_button.isEnabled()
    assert page._session_resume_button.isEnabled()
    assert controller.engine_resume_preparation is None

    page._session_resume_button.click()
    qtbot.wait(25)
    assert controller.engine_resume_preparation is not None
    assert controller.is_active is False
    assert not page._session_retry_button.isEnabled()


def test_h3_main_window_shutdown_stops_active_reconstruction_before_teardown(
    window, qtbot, tmp_path: Path
) -> None:
    controller = window._reconstruction_controller
    controller.select_image_sequence(write_ready_sequence(tmp_path))
    controller.start_offline(offline_configuration(tmp_path, "shutdown-ui"))
    qtbot.waitUntil(
        lambda: controller.offline_state is OfflineReconstructionRunState.RUNNING,
        timeout=2_000,
    )

    window.close()
    qtbot.wait(25)

    assert controller.is_shutdown
    assert controller.offline_controller.is_shutdown
    assert controller.offline_is_running is False
    assert controller.is_active is False

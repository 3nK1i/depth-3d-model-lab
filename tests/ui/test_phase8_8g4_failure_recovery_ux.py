from __future__ import annotations

from pathlib import Path

import pytest

from app.main_window import MainWindow
from app.offline_reconstruction_controller import OfflineReconstructionRunState
from app.reconstruction_controller import ReconstructionSourceSelection, ReconstructionWorkflow
from app.reconstruction_failure_recovery import ReconstructionFailureRecoveryState
from core.reconstruction import LiveCaptureRuntimeState
from tests.reconstruction_g3_helpers import write_checkpoint_fixture


@pytest.fixture
def window(qtbot, monkeypatch):
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    win = MainWindow()
    qtbot.addWidget(win)
    yield win
    win.close()


def _ready_sequence(tmp_path: Path) -> ReconstructionSourceSelection:
    source = tmp_path / "sequence"
    source.mkdir(parents=True, exist_ok=True)
    for index in range(3):
        (source / f"frame-{index}.png").write_bytes(b"g4")
    return ReconstructionSourceSelection(
        workflow=ReconstructionWorkflow.IMAGE_SEQUENCE,
        ready=True,
        display_name="sequence",
        detail="3 supported images • top-level directory",
        path=source,
        item_count=3,
    )


def _apply_selection(window, selection: ReconstructionSourceSelection) -> None:
    controller = window._reconstruction_controller
    controller._source_selection = selection
    controller.source_selection_changed.emit(selection)


def test_g4_initial_recovery_ui_is_passive_and_disabled(window) -> None:
    page = window._reconstruction_page
    assert hasattr(page, "_session_retry_button")
    assert not page._session_retry_button.isEnabled()
    assert page.failure_recovery_presentation is not None
    assert page.failure_recovery_presentation.state is ReconstructionFailureRecoveryState.STABLE
    assert page._session_recovery_state_value.text() == "No recovery action required"


def test_g4_terminal_failure_with_ready_source_enables_fresh_retry(window, qtbot, tmp_path: Path) -> None:
    page = window._reconstruction_page
    _apply_selection(window, _ready_sequence(tmp_path))
    page.set_error("deterministic offline failure")
    page.set_offline_state(OfflineReconstructionRunState.FAILED)
    qtbot.wait(10)

    assert page.failure_recovery_presentation is not None
    assert page.failure_recovery_presentation.state is ReconstructionFailureRecoveryState.RETRY_AVAILABLE
    assert page._session_retry_button.isEnabled()
    assert not page._start_button.isEnabled()
    assert "retry from source" in page._session_recovery_state_value.text().lower()
    assert "deterministic offline failure" in page._session_recovery_state_value.toolTip()


def test_g4_retry_button_reuses_existing_start_path_only(window, qtbot, monkeypatch, tmp_path: Path) -> None:
    page = window._reconstruction_page
    _apply_selection(window, _ready_sequence(tmp_path))
    page.set_offline_state(OfflineReconstructionRunState.FAILED)
    called: list[str] = []
    monkeypatch.setattr(page, "_start_runtime", lambda: called.append("fresh"))

    page._session_retry_button.click()
    qtbot.wait(10)

    assert called == ["fresh"]


def test_g4_live_auto_recovery_is_observed_and_manual_retry_stays_disabled(window, qtbot) -> None:
    page = window._reconstruction_page
    page.set_error("live worker timeout")
    page.set_live_state(LiveCaptureRuntimeState.RECOVERING)
    qtbot.wait(10)

    assert page.failure_recovery_presentation is not None
    assert page.failure_recovery_presentation.state is ReconstructionFailureRecoveryState.AUTO_RECOVERING
    assert page.failure_recovery_presentation.automatic_recovery
    assert not page._session_retry_button.isEnabled()
    assert "Automatic live recovery" in page._session_recovery_state_value.text()


def test_g4_failed_runtime_without_ready_source_fails_closed(window, qtbot) -> None:
    page = window._reconstruction_page
    page.set_error("source disconnected")
    page.set_live_state(LiveCaptureRuntimeState.FAILED)
    qtbot.wait(10)

    assert page.failure_recovery_presentation is not None
    assert page.failure_recovery_presentation.state is ReconstructionFailureRecoveryState.SOURCE_REQUIRED
    assert not page._session_retry_button.isEnabled()


def test_g4_preserved_checkpoint_blocks_ambiguous_fresh_retry(window, qtbot, tmp_path: Path) -> None:
    page = window._reconstruction_page
    controller = window._reconstruction_controller
    _apply_selection(window, _ready_sequence(tmp_path))
    _providers, _request, _plan, _checkpoint, manifest = write_checkpoint_fixture(tmp_path / "checkpoint")
    controller.load_engine_checkpoint(manifest)
    page.set_error("offline worker failed")
    page.set_offline_state(OfflineReconstructionRunState.FAILED)
    qtbot.wait(10)

    assert page.failure_recovery_presentation is not None
    assert page.failure_recovery_presentation.state is ReconstructionFailureRecoveryState.PRESERVED_EVIDENCE
    assert not page._session_retry_button.isEnabled()
    assert "evidence preserved" in page._session_recovery_state_value.text().lower()


def test_g4_cancelled_offline_attempt_can_offer_fresh_retry(window, qtbot, tmp_path: Path) -> None:
    page = window._reconstruction_page
    _apply_selection(window, _ready_sequence(tmp_path))
    page.set_offline_state(OfflineReconstructionRunState.CANCELLED)
    qtbot.wait(10)

    assert page.failure_recovery_presentation is not None
    assert page.failure_recovery_presentation.state is ReconstructionFailureRecoveryState.CANCELLED_RETRY_AVAILABLE
    assert page._session_retry_button.isEnabled()
    assert "Cancelled safely" in page._session_recovery_state_value.text()


def test_g4_recovery_action_remains_separate_from_g3_checkpoint_action(window) -> None:
    page = window._reconstruction_page
    assert page._session_retry_button.text() == "Retry From Source"
    assert page._session_resume_button.text() == "Prepare Resume"
    assert "not checkpoint resume" in page._session_retry_button.toolTip()
    assert "does not launch the resumed engine" in page._session_resume_button.toolTip()

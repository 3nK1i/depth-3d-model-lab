"""I6.8E Qt presentation tests: no reconstruction job is launched."""
from datetime import datetime, timezone

import pytest

from app.main_window import MainWindow
from app.reconstruction_dense_job_controller import DenseJobState
from core.reconstruction.reconstruction_job_telemetry_contract import (
    JobTelemetryObservation, StageWorkCounter,
)


@pytest.fixture
def page(qtbot, monkeypatch):
    monkeypatch.setattr('app.application_controller.CameraEnumerator.start', lambda self: None)
    win = MainWindow()
    qtbot.addWidget(win)
    yield win._reconstruction_page
    win.close()


def test_idle_panel_is_not_a_running_job(page):
    assert 'IDLE' in page._job_telemetry_heading.text()
    assert 'No active' in page._job_telemetry_stage.text()
    assert page._progress_bar.maximum() == 100


def test_fake_observation_from_previous_dense_run_not_displayed(page):
    sid = '11111111-2222-4333-8444-555555555555'
    run = 'aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee'
    page._job_telemetry_panel_state.begin('post_sparse_dense', sid, run)
    good = JobTelemetryObservation(
        source_session_id=sid, job_kind='post_sparse_dense', dense_run_id=run,
        producer='mvs_worker', sequence=1,
        observed_at_utc=datetime.now(timezone.utc).isoformat(),
        stage='dense_fusion', detail='Fusion active',
        counter=StageWorkCounter('fusion_images_started', 1, 87))
    page._on_job_telemetry_received(good)
    assert '1 / 87' in page._job_telemetry_counter.text()
    assert page._dense_progress_bar.maximum() == 100
    page._job_telemetry_panel_state.begin('post_sparse_dense', sid,
                                          'bbbbbbbb-cccc-4ddd-8eee-ffffffffffff')
    page._refresh_job_telemetry_presentation()
    page._on_job_telemetry_received(good)
    assert '1 / 87' not in page._job_telemetry_counter.text()
    page._job_telemetry_panel_state.end('cancelled')
    page._refresh_job_telemetry_presentation()
    assert 'cancelled' in page._job_telemetry_heading.text().lower()


def test_telemetry_details_open_on_demand_without_squeezing_job_controls(page, qtbot):
    assert page._job_telemetry_dialog.isHidden()
    assert page._job_telemetry_stage.parent() is page._job_telemetry_dialog
    assert page._job_telemetry_process.parent() is page._job_telemetry_dialog
    assert page._dense_full_view_button.parent() is page._runtime_card
    assert page._start_button.parent() is page._runtime_card
    page._job_telemetry_details_button.click()
    assert not page._job_telemetry_dialog.isHidden()
    assert 'No active offline reconstruction' in page._job_telemetry_stage.text()
    assert not page._dense_full_view_button.isHidden()
    page._job_telemetry_dialog.hide()
    assert page._job_telemetry_dialog.isHidden()

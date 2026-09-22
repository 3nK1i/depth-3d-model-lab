"""I6.8F: real MainWindow/Qt timer UI path, with explicitly simulated sidecars.

This is a GUI integration check, NOT a reconstruction-worker or sensor test.
The separate I6.8F bounded worker probe qualifies the real offline producer.
"""
from datetime import datetime, timedelta, timezone

import pytest
pytest.importorskip('PyQt6')

from app.main_window import MainWindow
from app.offline_reconstruction_controller import OfflineReconstructionRunState
from app.reconstruction_dense_job_controller import DenseJobState
from core.reconstruction.reconstruction_job_telemetry_contract import (
    JobTelemetryObservation, ResourceReading, StageWorkCounter,
    serialize_job_telemetry_line,
)

SID = '11111111-2222-4333-8444-555555555555'
OTHER_SID = '99999999-2222-4333-8444-555555555555'
DENSE_RUN = 'aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee'
OTHER_RUN = 'bbbbbbbb-cccc-4ddd-8eee-ffffffffffff'


@pytest.fixture
def main_page(qtbot, monkeypatch):
    # No hardware/camera discovery, SfM, MVS, admission, or scene mutation.
    monkeypatch.setattr('app.application_controller.CameraEnumerator.start', lambda self: None)
    window = MainWindow()
    qtbot.addWidget(window)
    page = window._reconstruction_page
    window._mode_stack.setCurrentWidget(page)
    window.show()
    qtbot.waitUntil(lambda: page.isVisible(), timeout=3000)
    try:
        yield window, page, window._reconstruction_controller
    finally:
        window.close()


def event(*, kind='offline_sfm', session=SID, run=None, producer='sfm_worker',
          sequence=1, age_s=0, resource=False):
    if resource:
        readings = (
            ResourceReading('process_tree_cpu_percent', 23.0, 'observed'),
            ResourceReading('process_tree_rss_bytes', 2**30, 'observed'),
            ResourceReading('system_ram_available_bytes', 3 * 2**30, 'observed'),
            ResourceReading('workspace_disk_free_bytes', None, 'unavailable'),
            ResourceReading('gpu_device_util_percent', 45.0, 'observed', 0),
            ResourceReading('gpu_device_vram_used_bytes', 2 * 2**30, 'observed', 0),
        )
    else:
        readings = ()
    stage = 'sfm_mapping' if kind == 'offline_sfm' else 'dense_fusion'
    return JobTelemetryObservation(
        source_session_id=session, job_kind=kind, dense_run_id=run,
        producer=('resource_sampler' if resource else producer), sequence=sequence,
        observed_at_utc=(datetime.now(timezone.utc) - timedelta(seconds=age_s)).isoformat(),
        stage=stage, detail='I6.8F TEST SIMULATION; not a running reconstruction',
        counter=(None if resource else StageWorkCounter(
            'registered_views' if kind == 'offline_sfm' else 'fusion_images_started',
            3, 24)),
        resource_readings=readings,
    )


def _sidecar(directory, name, obs):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(serialize_job_telemetry_line(obs) + '\n', encoding='utf-8')


def test_main_window_offline_qt_timer_displays_progress_resources_and_clears(main_page, qtbot, tmp_path):
    window, page, controller = main_page
    controller._offline_controller._session_id = SID  # test-only identity, no QProcess launch
    controller.offline_state_changed.emit(OfflineReconstructionRunState.STARTING)
    assert page._job_telemetry_panel_state.identity == ('offline_sfm', SID, None)
    relay = controller._offline_telemetry_relay
    source = tmp_path / 'simulated_offline_sidecars'
    relay.begin('offline_sfm', SID, None, source)
    try:
        page._job_telemetry_details_button.click()
        assert page._job_telemetry_dialog.isVisible()
        assert 'awaiting sample' in page._job_telemetry_stage.text()
        _sidecar(source, 'job_progress_observation.txt', event())
        _sidecar(source, 'job_resource_observation.txt', event(resource=True))
        # Production 1500ms Qt poll timer drives the real bound controller signal and page.
        qtbot.waitUntil(lambda: '3 / 24' in page._job_telemetry_counter.text() and
                        '1.00 GiB' in page._job_telemetry_process.text(), timeout=5000)
        assert 'SfM mapping' in page._job_telemetry_stage.text()
        assert 'Host:' in page._job_telemetry_host.text()
        assert 'unavailable' in page._job_telemetry_host.text()
        assert 'GPU 0 device-wide' in page._job_telemetry_gpu.text()
        assert '45.0%' in page._job_telemetry_gpu.text()
        assert 'not overall progress' in page._job_telemetry_counter.text()
        assert page._dense_full_view_button.parent() is page._runtime_card
        assert page._start_button.parent() is page._runtime_card
        # Real terminal-state signal must remove all last-sample values from the page.
        controller.offline_state_changed.emit(OfflineReconstructionRunState.COMPLETED)
        assert 'Live samples ended' in page._job_telemetry_stage.text()
        assert '1.00 GiB' not in page._job_telemetry_process.text()
        assert '45.0%' not in page._job_telemetry_gpu.text()
        # A late sidecar or direct controller signal cannot reactivate a finished page.
        controller.job_telemetry_received.emit(event(sequence=3))
        assert '3 / 24' not in page._job_telemetry_counter.text()
        page._job_telemetry_dialog.hide()
        assert page._job_telemetry_dialog.isHidden()
    finally:
        relay.close()


def test_main_window_rejects_foreign_session_and_stale_values(main_page):
    _, page, controller = main_page
    controller._offline_controller._session_id = SID
    controller.offline_state_changed.emit(OfflineReconstructionRunState.STARTING)
    controller.job_telemetry_received.emit(event(session=OTHER_SID))
    assert '3 / 24' not in page._job_telemetry_counter.text()
    controller.job_telemetry_received.emit(event(age_s=20))
    controller.job_telemetry_received.emit(event(resource=True, age_s=20))
    page._refresh_job_telemetry_presentation()
    assert 'stale' in page._job_telemetry_stage.text()
    assert '3 / 24' not in page._job_telemetry_counter.text()
    assert '1.00 GiB' not in page._job_telemetry_process.text()
    assert '45.0%' not in page._job_telemetry_gpu.text()


def test_main_window_dense_controller_relay_run_identity_and_terminal(main_page, qtbot, tmp_path):
    _, page, controller = main_page
    dense = controller.dense_controller
    dense._session_id = SID  # test-only identity; no MVS QProcess or source is started
    dense._dense_run_id = DENSE_RUN
    controller.dense_state_changed.emit(DenseJobState.STARTING)
    assert page._job_telemetry_panel_state.identity == ('post_sparse_dense', SID, DENSE_RUN)
    source = tmp_path / 'simulated_dense_sidecars'
    relay = dense._telemetry_relay
    relay.begin('post_sparse_dense', SID, DENSE_RUN, source)
    try:
        _sidecar(source, 'job_progress_observation.txt', event(
            kind='post_sparse_dense', run=DENSE_RUN, producer='mvs_worker'))
        qtbot.waitUntil(lambda: '3 / 24' in page._job_telemetry_counter.text(), timeout=5000)
        assert 'Stereo fusion' in page._job_telemetry_stage.text()
        dense._dense_run_id = OTHER_RUN
        controller.dense_state_changed.emit(DenseJobState.STARTING)
        assert '3 / 24' not in page._job_telemetry_counter.text()
        # A late sample for the previous dense run may arrive, but cannot be shown.
        controller.job_telemetry_received.emit(event(
            kind='post_sparse_dense', run=DENSE_RUN, producer='mvs_worker', sequence=2))
        assert '3 / 24' not in page._job_telemetry_counter.text()
        controller.dense_state_changed.emit(DenseJobState.FAILED)
        assert 'Live samples ended' in page._job_telemetry_stage.text()
    finally:
        relay.close()

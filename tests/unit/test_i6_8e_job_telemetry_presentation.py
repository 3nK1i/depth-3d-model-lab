"""I6.8E observed-only UI presentation and cross-run/terminal safety."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from app.reconstruction_job_telemetry_presentation import JobTelemetryPanelState
from core.reconstruction.reconstruction_job_telemetry_contract import (
    JobTelemetryObservation, ResourceReading, StageWorkCounter,
)

SID = '11111111-2222-4333-8444-555555555555'
RUN = 'aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee'
OTHER = '99999999-2222-4333-8444-555555555555'
T = datetime(2026, 9, 21, 17, 0, tzinfo=timezone.utc)


def event(*, resource=False, sid=SID, run=RUN, sequence=1, time=T):
    return JobTelemetryObservation(
        source_session_id=sid, job_kind='post_sparse_dense', dense_run_id=run,
        producer='resource_sampler' if resource else 'mvs_worker', sequence=sequence,
        observed_at_utc=time.isoformat(), stage='dense_fusion',
        detail='Stereo fusion active',
        counter=None if resource else StageWorkCounter('fusion_images_started', 1, 87),
        resource_readings=(
            ResourceReading('process_tree_rss_bytes', 2**30, 'observed'),
            ResourceReading('system_ram_available_bytes', 3 * 2**30, 'observed'),
            ResourceReading('workspace_disk_free_bytes', None, 'unavailable'),
            ResourceReading('gpu_device_util_percent', 92.0, 'observed', 0),
            ResourceReading('gpu_device_vram_used_bytes', 4 * 2**30, 'observed', 0),
        ) if resource else (),
    )


def test_idle_does_not_present_telemetry_as_running():
    p = JobTelemetryPanelState().present(T)
    assert p.status == 'idle' and 'No active' in p.stage
    assert '—' in p.gpu


def test_observed_counter_is_never_overall_percent_or_eta():
    s = JobTelemetryPanelState()
    s.begin('post_sparse_dense', SID, RUN)
    assert s.accept(event())
    p = s.present(T)
    assert 'Stereo fusion' in p.stage
    assert '1 / 87' in p.counter and 'not overall progress' in p.counter
    assert '%' not in p.counter and 'ETA' not in p.counter
    assert 'awaiting sample' in p.gpu


def test_device_scope_and_unavailable_readings_are_explicit():
    s = JobTelemetryPanelState()
    s.begin('post_sparse_dense', SID, RUN)
    assert s.accept(event(resource=True))
    p = s.present(T)
    assert 'Worker tree:' in p.process and '1.00 GiB' in p.process
    assert 'Host:' in p.host and '3.00 GiB' in p.host and 'unavailable' in p.host
    assert 'GPU 0 device-wide:' in p.gpu and '92.0%' in p.gpu
    assert '4.00 GiB' in p.gpu
    assert 'job GPU' not in p.gpu


def test_stale_samples_are_not_live_values_even_when_no_new_signal_arrives():
    s = JobTelemetryPanelState()
    s.begin('post_sparse_dense', SID, RUN)
    s.accept(event())
    s.accept(event(resource=True))
    p = s.present(T + timedelta(seconds=16))
    assert 'stale' in p.stage and 'stale' in p.gpu
    assert '92.0%' not in p.gpu and '1 / 87' not in p.counter


def test_new_job_clears_previous_values_and_rejects_late_old_run():
    s = JobTelemetryPanelState()
    s.begin('post_sparse_dense', SID, RUN)
    s.accept(event())
    s.begin('post_sparse_dense', OTHER, RUN)
    assert not s.accept(event())
    p = s.present(T)
    assert 'awaiting' in p.stage and '1 / 87' not in p.counter
    assert s.accept(event(sid=OTHER))
    assert '1 / 87' in s.present(T).counter


def test_terminal_job_closes_samples_and_refuses_late_observations():
    s = JobTelemetryPanelState()
    s.begin('post_sparse_dense', SID, RUN)
    s.accept(event(resource=True))
    s.end('failed')
    assert not s.accept(event(sequence=2))
    p = s.present(T)
    assert 'failed' in p.heading and '92.0%' not in p.gpu
    assert 'not current' in p.detail


def test_malformed_and_wrong_job_cannot_update_ui():
    s = JobTelemetryPanelState()
    s.begin('offline_sfm', SID, None)
    assert not s.accept(event())
    assert not s.accept({'stage': 'dense_fusion'})
    assert s.present(T).status == 'waiting'

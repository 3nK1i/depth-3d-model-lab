"""I6.8D bounded, strictly run-bound sidecar transport; no reconstruction run."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from core.reconstruction.reconstruction_job_telemetry_contract import (
    JobTelemetryObservation, ResourceReading, serialize_job_telemetry_line,
)
from core.reconstruction.reconstruction_job_telemetry_transport import (
    JobTelemetrySidecarTransport, TELEMETRY_HISTORY_LIMIT,
)

SID = '11111111-2222-4333-8444-555555555555'
RUN = '66666666-7777-4888-9999-aaaaaaaaaaaa'


def event(*, sid=SID, run=RUN, producer='dense_supervisor', sequence=1,
          when=None, stage='dense_patch_match'):
    return JobTelemetryObservation(
        source_session_id=sid, job_kind='post_sparse_dense', dense_run_id=run,
        producer=producer, sequence=sequence,
        observed_at_utc=(when or datetime.now(timezone.utc)).isoformat(),
        stage=stage, detail='Observed stage',
        resource_readings=(ResourceReading('system_ram_available_bytes', 4096, 'observed'),)
        if producer == 'resource_sampler' else (),
    )


def write(path: Path, observation: JobTelemetryObservation):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialize_job_telemetry_line(observation) + '\n', encoding='utf-8')


def transport(tmp_path, run=RUN, clock=None):
    return JobTelemetrySidecarTransport('post_sparse_dense', SID, run, tmp_path,
                                       clock=clock or (lambda: datetime.now(timezone.utc)))


def test_progress_and_resource_transport_latest_only(tmp_path):
    receiver = transport(tmp_path)
    write(tmp_path / 'job_progress_observation.txt', event())
    write(tmp_path / 'job_resource_observation.txt', event(producer='resource_sampler'))
    got = receiver.poll()
    assert len(got) == 2
    assert {x.producer for x in got} == {'dense_supervisor', 'resource_sampler'}
    assert receiver.poll() == ()
    assert len(receiver.history) == 2
    assert receiver.latest('resource_sampler') == got[1]


def test_rejects_foreign_run_session_and_old_sample(tmp_path):
    receiver = transport(tmp_path)
    path = tmp_path / 'job_progress_observation.txt'
    write(path, event(run=str(uuid4())))
    assert not receiver.poll()
    write(path, event(sid=str(uuid4())))
    assert not receiver.poll()
    write(path, event(when=datetime.now(timezone.utc) - timedelta(minutes=2)))
    assert not receiver.poll()
    assert not receiver.history
    write(path, event())
    assert len(receiver.poll()) == 1


def test_rejects_duplicate_and_lower_sequence_and_bounds_history(tmp_path):
    receiver = transport(tmp_path)
    path = tmp_path / 'job_progress_observation.txt'
    for sequence in range(1, TELEMETRY_HISTORY_LIMIT + 20):
        write(path, event(sequence=sequence))
        assert len(receiver.poll()) == 1
    assert len(receiver.history) == TELEMETRY_HISTORY_LIMIT
    assert receiver.history[-1].sequence == TELEMETRY_HISTORY_LIMIT + 19
    write(path, event(sequence=2))
    assert receiver.poll() == ()
    assert receiver.history[-1].sequence == TELEMETRY_HISTORY_LIMIT + 19


def test_stale_latest_marked_unavailable_and_terminal_close(tmp_path):
    now = datetime.now(timezone.utc)
    current = [now]
    receiver = transport(tmp_path, clock=lambda: current[0])
    write(tmp_path / 'job_resource_observation.txt', event(producer='resource_sampler', when=now))
    assert len(receiver.poll()) == 1
    current[0] = now + timedelta(seconds=17)
    assert receiver.latest('resource_sampler') is None
    assert receiver.latest_state('resource_sampler') == 'stale'
    receiver.close()
    write(tmp_path / 'job_resource_observation.txt', event(producer='resource_sampler', sequence=2,
                                                          when=current[0]))
    assert receiver.poll() == ()
    assert receiver.latest_state('resource_sampler') == 'closed'


def test_malformed_oversize_symlink_and_missing_sidecar_are_fail_soft(tmp_path):
    receiver = transport(tmp_path)
    path = tmp_path / 'job_progress_observation.txt'
    assert receiver.poll() == ()
    path.write_text('junk', encoding='utf-8')
    assert receiver.poll() == ()
    path.write_bytes(b'a' * 100000)
    assert receiver.poll() == ()
    path.unlink()
    outside = tmp_path.parent / ('outside-' + uuid4().hex)
    outside.write_text(serialize_job_telemetry_line(event()), encoding='utf-8')
    path.symlink_to(outside)
    try:
        assert receiver.poll() == ()
    finally:
        outside.unlink(missing_ok=True)
    assert not receiver.history


def test_switch_to_new_job_rejects_old_sidecars(tmp_path):
    receiver = transport(tmp_path)
    path = tmp_path / 'job_progress_observation.txt'
    write(path, event())
    assert len(receiver.poll()) == 1
    new_run = str(uuid4())
    receiver.reset('post_sparse_dense', SID, new_run, tmp_path)
    assert not receiver.history
    assert receiver.poll() == ()
    write(path, event(run=new_run))
    assert len(receiver.poll()) == 1


def test_same_job_resource_producer_independent_sequences(tmp_path):
    receiver = transport(tmp_path)
    path = tmp_path / 'job_progress_observation.txt'
    write(path, event(sequence=200))
    assert len(receiver.poll()) == 1
    write(tmp_path / 'job_resource_observation.txt', event(producer='resource_sampler', sequence=1))
    assert len(receiver.poll()) == 1


def test_sfm_binding_with_no_dense_id(tmp_path):
    receiver = JobTelemetrySidecarTransport('offline_sfm', SID, None, tmp_path)
    sfm = JobTelemetryObservation(source_session_id=SID, job_kind='offline_sfm', dense_run_id=None,
                                  producer='sfm_worker', sequence=1,
                                  observed_at_utc=datetime.now(timezone.utc).isoformat(),
                                  stage='sfm_matching', detail='Matching')
    write(tmp_path / 'job_progress_observation.txt', sfm)
    assert receiver.poll() == (sfm,)
    assert receiver.latest_state('sfm_worker') == 'current'


def test_future_dated_samples_rejected_even_with_valid_job_identity(tmp_path):
    now = datetime.now(timezone.utc)
    receiver = transport(tmp_path, clock=lambda: now)
    write(tmp_path / 'job_progress_observation.txt',
          event(when=now + timedelta(seconds=8)))
    assert receiver.poll() == ()
    assert receiver.latest_state('dense_supervisor') == 'unavailable'
    assert receiver.stale_count == 1


def test_one_poll_always_delivers_at_most_two_latest_observations(tmp_path):
    receiver = transport(tmp_path)
    p = tmp_path / 'job_progress_observation.txt'
    r = tmp_path / 'job_resource_observation.txt'
    for sequence in range(1, 50):
        write(p, event(sequence=sequence))
        write(r, event(sequence=sequence, producer='resource_sampler'))
    received = receiver.poll()
    assert len(received) == 2
    assert all(item.sequence == 49 for item in received)
    assert receiver.poll() == ()

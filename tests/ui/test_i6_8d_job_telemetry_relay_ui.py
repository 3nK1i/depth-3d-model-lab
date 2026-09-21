"""I6.8D: workstation Qt event delivery without launching an SfM/MVS worker."""
from datetime import datetime, timezone
from pathlib import Path

import pytest
pytest.importorskip('PyQt6')

from app.reconstruction_job_telemetry_relay import JobTelemetryQtRelay
from core.reconstruction.reconstruction_job_telemetry_contract import (
    JobTelemetryObservation, serialize_job_telemetry_line,
)

SID = '11111111-2222-4333-8444-555555555555'
RUN = '66666666-7777-4888-9999-aaaaaaaaaaaa'


def write(path: Path, sid=SID, run=RUN, sequence=1):
    event = JobTelemetryObservation(
        source_session_id=sid, job_kind='post_sparse_dense', dense_run_id=run,
        producer='dense_supervisor', sequence=sequence,
        observed_at_utc=datetime.now(timezone.utc).isoformat(),
        stage='dense_fusion', detail='Fusion active; elapsed work indeterminate',
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialize_job_telemetry_line(event) + '\n', encoding='utf-8')


def test_qt_relay_run_isolation_and_close(qtbot, tmp_path):
    relay = JobTelemetryQtRelay()
    received = []
    relay.observation_received.connect(received.append)
    relay.begin('post_sparse_dense', SID, RUN, tmp_path)
    path = tmp_path / 'job_progress_observation.txt'
    write(path)
    relay.poll()
    assert len(received) == 1
    assert received[0].stage == 'dense_fusion'
    relay.poll()
    assert len(received) == 1
    relay.close()
    write(path, sequence=2)
    relay.poll()
    assert len(received) == 1
    assert relay.latest_state('dense_supervisor') == 'closed'


def test_qt_relay_resets_on_new_run_without_leaking_old_state(qtbot, tmp_path):
    relay = JobTelemetryQtRelay()
    events = []
    relay.observation_received.connect(events.append)
    path = tmp_path / 'job_progress_observation.txt'
    relay.begin('post_sparse_dense', SID, RUN, tmp_path)
    write(path)
    relay.poll()
    assert len(events) == 1
    new_run = '99999999-aaaa-4bbb-8ccc-dddddddddddd'
    relay.begin('post_sparse_dense', SID, new_run, tmp_path)
    relay.poll()
    assert len(events) == 1
    assert not relay.history
    write(path, run=new_run)
    relay.poll()
    assert len(events) == 2
    assert events[-1].dense_run_id == new_run
    relay.close()


def test_qt_external_sparse_relay_and_dense_controller_forward_without_worker_launch(qtbot, tmp_path):
    from app.reconstruction_dense_job_controller import ExplicitDenseJobController
    from core.reconstruction.reconstruction_job_telemetry_contract import JobTelemetryObservation

    sparse_relay = JobTelemetryQtRelay()
    dense = ExplicitDenseJobController()
    sparse_received = []
    dense_received = []
    sparse_relay.observation_received.connect(sparse_received.append)
    dense.telemetry_received.connect(dense_received.append)

    sparse_relay.begin('offline_sfm', SID, None, tmp_path / 'sparse')
    sparse_observation = JobTelemetryObservation(
        source_session_id=SID, job_kind='offline_sfm', dense_run_id=None,
        producer='sfm_worker', sequence=1,
        observed_at_utc=datetime.now(timezone.utc).isoformat(),
        stage='sfm_matching', detail='Matching observed',
    )
    sparse_file = tmp_path / 'sparse' / 'job_progress_observation.txt'
    sparse_file.parent.mkdir(parents=True)
    sparse_file.write_text(serialize_job_telemetry_line(sparse_observation) + '\n', encoding='utf-8')
    sparse_relay.poll()
    assert sparse_received == [sparse_observation]

    dense._telemetry_relay.begin('post_sparse_dense', SID, RUN, tmp_path / 'dense')
    write(tmp_path / 'dense' / 'job_progress_observation.txt')
    dense._telemetry_relay.poll()
    assert len(dense_received) == 1
    assert dense_received[0].dense_run_id == RUN

    sparse_relay.close()
    dense.shutdown()


def test_qt_timer_delivers_without_explicit_poll_or_reconstruction(qtbot, tmp_path):
    relay = JobTelemetryQtRelay()
    got = []
    relay.observation_received.connect(got.append)
    relay.begin('post_sparse_dense', SID, RUN, tmp_path)
    write(tmp_path / 'job_progress_observation.txt')
    qtbot.waitUntil(lambda: len(got) == 1, timeout=4000)
    assert got[0].dense_run_id == RUN
    relay.close()


def test_reentrant_run_switch_cannot_emit_second_old_run_sample(qtbot, tmp_path):
    relay = JobTelemetryQtRelay()
    new_run = '99999999-aaaa-4bbb-8ccc-dddddddddddd'
    delivered = []

    def on_event(observation):
        delivered.append(observation)
        relay.begin('post_sparse_dense', SID, new_run, tmp_path)

    relay.observation_received.connect(on_event)
    relay.begin('post_sparse_dense', SID, RUN, tmp_path)
    write(tmp_path / 'job_progress_observation.txt')
    resource = JobTelemetryObservation(
        source_session_id=SID, job_kind='post_sparse_dense', dense_run_id=RUN,
        producer='resource_sampler', sequence=1,
        observed_at_utc=datetime.now(timezone.utc).isoformat(),
        stage='dense_fusion', detail='Observed device resources',
    )
    (tmp_path / 'job_resource_observation.txt').write_text(
        serialize_job_telemetry_line(resource) + '\n', encoding='utf-8')
    relay.poll()
    assert len(delivered) == 1
    assert delivered[0].dense_run_id == RUN
    assert not relay.history
    relay.close()

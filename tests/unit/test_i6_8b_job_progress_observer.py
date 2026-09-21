"""I6.8B: source-bound, observational stage progress without invented overall progress."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from core.reconstruction.sfm_worker_protocol import SfmWorkerEvent, SfmWorkerEventType
from core.reconstruction.reconstruction_job_telemetry_contract import (
    JobTelemetryObservation, parse_job_telemetry_line,
)
from tests.i6_7c_helpers import make_snapshot, replace_snapshot, SESSION_ID, DENSE_RUN_ID

from core.reconstruction.reconstruction_job_progress_observer import (
    JobProgressObserver, LatestJobProgressSnapshot, JobProgressObservationError,
)


def sfm_event(stage: str, **metadata):
    return SfmWorkerEvent(
        event_type=SfmWorkerEventType.PROGRESS,
        stage=stage, message='Legacy completion fraction is a workflow estimate.',
        progress_fraction=0.99, metadata=tuple(metadata.items()),
    )


@pytest.mark.parametrize('raw,stage', [
    ('feature_extraction', 'sfm_feature_extraction'),
    ('matching', 'sfm_matching'),
    ('mapping', 'sfm_mapping'),
    ('conversion', 'sfm_finalizing'),
    ('complete', 'sfm_finalizing'),
])
def test_sfm_stage_derived_from_real_worker_events_without_estimate(raw, stage):
    obs = JobProgressObserver('offline_sfm', SESSION_ID)
    event = obs.from_sfm_worker_event(sfm_event(raw, registered_callbacks=6, initial_pair_callbacks=1))
    assert event.stage == stage
    assert event.source_session_id == SESSION_ID
    assert event.producer == 'sfm_worker'
    assert event.sequence == 1
    assert event.counter is None  # callback frequency is not the actual registered-view count
    assert 'progress_fraction' not in event.as_dict()
    assert 'eta_seconds' not in event.as_dict()
    assert 'completed' not in event.as_dict()


def test_sfm_unknown_and_warning_events_do_not_change_sequence():
    obs = JobProgressObserver('offline_sfm', SESSION_ID)
    assert obs.from_sfm_worker_event(sfm_event('internal_unknown')) is None
    warning = SfmWorkerEvent(SfmWorkerEventType.WARNING, 'mapping', 'warning', 0.5)
    assert obs.from_sfm_worker_event(warning) is None
    assert obs.from_sfm_worker_event(sfm_event('mapping')).sequence == 1


def test_verified_final_sparse_statistics_are_separate_from_mapping_callbacks():
    obs = JobProgressObserver('offline_sfm', SESSION_ID)
    sample = obs.from_confirmed_sparse_view_count(registered_views=46, selected_views=59)
    assert sample.stage == 'sfm_mapping'
    assert sample.counter.unit == 'registered_views'
    assert (sample.counter.observed_count, sample.counter.expected_count) == (46, 59)
    assert sample.detail.startswith('Sparse backend reported')
    with pytest.raises(JobProgressObservationError):
        obs.from_confirmed_sparse_view_count(60, 59)
    assert obs.from_confirmed_sparse_view_count(1, 3).sequence == 2


@pytest.mark.parametrize('detail,stage', [
    ('external_worker_launch', 'dense_preparing'),
    ('image_undistortion', 'dense_undistortion'),
    ('patch_match_stereo', 'dense_patch_match'),
    ('stereo_fusion', 'dense_fusion'),
    ('external_worker_complete', 'dense_verification'),
])
def test_dense_progress_uses_observed_stage_from_existing_provider(detail, stage):
    obs = JobProgressObserver('post_sparse_dense', SESSION_ID, DENSE_RUN_ID)
    snapshot = replace_snapshot(make_snapshot(), stage_detail=detail)
    event = obs.from_dense_snapshot(snapshot)
    assert event.stage == stage
    assert event.producer == 'dense_supervisor'
    assert event.dense_run_id == DENSE_RUN_ID
    assert 'progress_fraction' not in event.as_dict()
    assert 'eta_seconds' not in event.as_dict()
    assert 'admitted' not in event.as_dict()
    assert event.resource_readings == ()


def test_patch_match_counter_is_observed_map_file_count_not_duration():
    obs = JobProgressObserver('post_sparse_dense', SESSION_ID, DENSE_RUN_ID)
    snapshot = replace_snapshot(make_snapshot(), stage_detail='patch_match_stereo')
    event = obs.from_dense_snapshot(snapshot)
    assert event.counter.unit == 'geometric_depth_maps_observed'
    assert event.counter.observed_count == 1
    assert event.counter.expected_count == 24
    assert obs.from_dense_snapshot(replace_snapshot(snapshot, revision=2)).sequence == 2


def test_fusion_has_no_fake_per_image_counter_even_if_processed_views_87():
    obs = JobProgressObserver('post_sparse_dense', SESSION_ID, DENSE_RUN_ID)
    snapshot = replace_snapshot(make_snapshot(), stage_detail='stereo_fusion',
                                statistics=replace(make_snapshot().statistics, processed_views=24))
    event = obs.from_dense_snapshot(snapshot)
    assert event.stage == 'dense_fusion'
    assert event.counter is None
    assert '1/87' not in event.detail
    assert 'completion' not in event.detail.lower()


def test_dense_observation_rejects_foreign_identity_and_ignores_nonprovider_status():
    obs = JobProgressObserver('post_sparse_dense', SESSION_ID, DENSE_RUN_ID)
    with pytest.raises(JobProgressObservationError):
        obs.from_dense_snapshot(make_snapshot(session_id='22222222-3333-4444-8555-666666666666'))
    with pytest.raises(JobProgressObservationError):
        obs.from_dense_snapshot(make_snapshot(dense_run_id='99999999-aaaa-4bbb-8ccc-dddddddddddd'))
    assert obs.from_dense_snapshot(make_snapshot()) is None
    assert obs.from_dense_snapshot(replace_snapshot(make_snapshot(), stage_detail='stereo_fusion')).sequence == 1


def test_latest_only_snapshot_is_bound_small_and_does_not_overwrite_authority(tmp_path):
    obs = JobProgressObserver('post_sparse_dense', SESSION_ID, DENSE_RUN_ID)
    sink = LatestJobProgressSnapshot(tmp_path / 'job_progress_observation.txt')
    first = obs.from_dense_snapshot(replace_snapshot(make_snapshot(), stage_detail='patch_match_stereo'))
    second = obs.from_dense_snapshot(replace_snapshot(make_snapshot(revision=2), stage_detail='stereo_fusion'))
    assert sink.publish(first)
    assert sink.publish(second)
    assert sink.path.is_file()
    assert sink.path.stat().st_size < 4096
    current = parse_job_telemetry_line(sink.path.read_text().strip())
    assert current == second
    assert first != current
    assert list(tmp_path.iterdir()) == [sink.path]
    assert not (tmp_path / 'offline_result.json').exists()
    assert not (tmp_path / 'final_dense_admission.json').exists()


def test_latest_only_snapshot_ignores_failure_without_mutating_other_evidence(tmp_path):
    obs = JobProgressObserver('offline_sfm', SESSION_ID)
    original = tmp_path / 'offline_result.json'
    original.write_text('DO NOT CHANGE')
    sink = LatestJobProgressSnapshot(tmp_path / 'nonexistent' / 'job_progress_observation.txt')
    assert sink.publish(obs.from_sfm_worker_event(sfm_event('mapping'))) is False
    assert original.read_text() == 'DO NOT CHANGE'


def test_existing_live_source_hooks_present_without_changing_frozen_admission_files():
    root = Path(__file__).resolve().parents[2]
    sfm = (root / 'core/reconstruction/sfm_pycolmap.py').read_text()
    dense = (root / 'scripts/run_i6_7e2b2a_post_sparse_dense_worker.py').read_text()
    assert 'from_sfm_worker_event(event)' in sfm
    assert 'from_dense_snapshot(snapshot)' in dense
    assert 'job_progress_observation.txt' in sfm and 'job_progress_observation.txt' in dense


def test_sparse_backend_existing_event_remains_authoritative_when_diagnostic_sink_fails(monkeypatch, tmp_path):
    from core.reconstruction.sfm_pycolmap import ExternalPycolmapSfmBackend

    backend = object.__new__(ExternalPycolmapSfmBackend)
    forwarded = []
    monkeypatch.setattr(backend, '_emit', lambda *args, **kwargs: forwarded.append(kwargs), raising=False)
    backend._i68b_progress_observer = JobProgressObserver('offline_sfm', SESSION_ID)
    backend._i68b_progress_sink = LatestJobProgressSnapshot(tmp_path / 'absent' / 'observation.txt')
    backend._emit_worker_event(object(), sfm_event('mapping', registered_callbacks=11))
    assert len(forwarded) == 1
    assert forwarded[0]['progress'] == 0.99  # unchanged legacy workflow fraction
    assert forwarded[0]['metadata']['registered_callbacks'] == 11
    assert not (tmp_path / 'absent').exists()


def test_dense_worker_hook_forwards_original_snapshot_even_if_observation_unavailable(tmp_path, monkeypatch):
    from argparse import Namespace
    from scripts import run_i6_7e2b2a_post_sparse_dense_worker as worker

    args = Namespace(
        session_id=SESSION_ID, dense_run_id=DENSE_RUN_ID, offline_output_dir=tmp_path,
        pycolmap_python=tmp_path / 'unused_python', mvs_worker_script=tmp_path / 'unused_worker',
        provider_id=worker.I43B_PROVIDER_ID, timeout_minutes=1.0,
    )
    monkeypatch.setattr(worker, '_parse_args', lambda _: args)
    stdout = []
    monkeypatch.setattr(worker, 'print', lambda *values, **kw: stdout.extend(values), raising=False)
    def fake_execute(**kwargs):
        kwargs['snapshot_sink'].publish(replace_snapshot(make_snapshot(), stage_detail='patch_match_stereo'))
        return type('Admitted', (), {'as_dict': lambda self: {'admission_test_marker': True}})()
    monkeypatch.setattr(worker, 'execute_explicit_colmap_dense_after_sparse', fake_execute)
    assert worker.main() == 0
    assert any(str(line).startswith('EIA_DENSE_PROGRESSIVE ') for line in stdout)
    assert any('"event":"dense_completed"' in str(line) for line in stdout)
    assert not (tmp_path / 'dense_runs').exists()  # observer never creates fake evidence dirs


def test_dense_worker_hook_reports_real_stage_to_latest_only_sidecar(tmp_path, monkeypatch):
    from argparse import Namespace
    from scripts import run_i6_7e2b2a_post_sparse_dense_worker as worker

    args = Namespace(
        session_id=SESSION_ID, dense_run_id=DENSE_RUN_ID, offline_output_dir=tmp_path,
        pycolmap_python=tmp_path / 'unused_python', mvs_worker_script=tmp_path / 'unused_worker',
        provider_id=worker.I43B_PROVIDER_ID, timeout_minutes=1.0,
    )
    monkeypatch.setattr(worker, '_parse_args', lambda _: args)
    stdout = []
    monkeypatch.setattr(worker, 'print', lambda *values, **kw: stdout.extend(values), raising=False)
    def fake_execute(**kwargs):
        job_root = tmp_path / 'dense_runs' / DENSE_RUN_ID
        job_root.mkdir(parents=True)
        kwargs['snapshot_sink'].publish(replace_snapshot(make_snapshot(), stage_detail='patch_match_stereo'))
        kwargs['snapshot_sink'].publish(replace_snapshot(make_snapshot(revision=2), stage_detail='stereo_fusion'))
        return type('Admitted', (), {'as_dict': lambda self: {'admission_test_marker': True}})()
    monkeypatch.setattr(worker, 'execute_explicit_colmap_dense_after_sparse', fake_execute)
    assert worker.main() == 0
    sidecar = tmp_path / 'dense_runs' / DENSE_RUN_ID / 'job_progress_observation.txt'
    assert sidecar.is_file()
    event = parse_job_telemetry_line(sidecar.read_text().strip())
    assert event.stage == 'dense_fusion'
    assert event.sequence == 2
    assert event.counter is None
    assert len([line for line in stdout if str(line).startswith('EIA_DENSE_PROGRESSIVE ')]) == 2


def test_sparse_backend_hook_writes_actual_worker_stage_in_bound_session(tmp_path, monkeypatch):
    from core.reconstruction.sfm_pycolmap import ExternalPycolmapSfmBackend

    session_root = tmp_path / SESSION_ID
    session_root.mkdir()
    backend = object.__new__(ExternalPycolmapSfmBackend)
    forwarded = []
    monkeypatch.setattr(backend, '_emit', lambda *args, **kwargs: forwarded.append(kwargs), raising=False)
    backend._i68b_progress_observer = JobProgressObserver('offline_sfm', SESSION_ID)
    backend._i68b_progress_sink = LatestJobProgressSnapshot(
        session_root / 'job_progress_observation.txt')
    backend._emit_worker_event(object(), sfm_event('feature_extraction'))
    backend._emit_worker_event(object(), sfm_event('matching'))
    observation = parse_job_telemetry_line(
        (session_root / 'job_progress_observation.txt').read_text().strip())
    assert observation.stage == 'sfm_matching'
    assert observation.sequence == 2
    assert observation.counter is None
    assert len(forwarded) == 2


def test_dense_worker_preserves_terminal_failed_event_on_invalid_telemetry_identity(monkeypatch, tmp_path):
    from argparse import Namespace
    from scripts import run_i6_7e2b2a_post_sparse_dense_worker as worker

    args = Namespace(
        session_id='bad-uuid', dense_run_id=DENSE_RUN_ID, offline_output_dir=tmp_path,
        pycolmap_python=tmp_path / 'unused_python', mvs_worker_script=tmp_path / 'unused_worker',
        provider_id=worker.I43B_PROVIDER_ID, timeout_minutes=1.0,
    )
    monkeypatch.setattr(worker, '_parse_args', lambda _: args)
    output = []
    monkeypatch.setattr(worker, 'print', lambda *values, **kw: output.extend(values), raising=False)
    def fail_original_validator(**kwargs):
        raise ValueError('the original run validator rejects the foreign session')
    monkeypatch.setattr(worker, 'execute_explicit_colmap_dense_after_sparse', fail_original_validator)
    assert worker.main() == 2
    assert any('"event":"dense_failed"' in str(line) for line in output)
    assert not (tmp_path / 'dense_runs').exists()

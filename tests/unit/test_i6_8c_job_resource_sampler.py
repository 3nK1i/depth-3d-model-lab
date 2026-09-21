"""I6.8C: bounded factual resource observations, never admission/control."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.reconstruction.reconstruction_job_telemetry_contract import (
    JobTelemetryObservation, parse_job_telemetry_line,
)
from core.reconstruction.reconstruction_job_resource_sampler import (
    JobResourceSampler, gpu_device_readings, read_available_ram_bytes,
)

SID = '11111111-2222-4333-8444-555555555555'
RID = 'aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee'


class FakeProcess:
    def __init__(self, pid, born=1.0, rss=2048, cpu=2.0, descendants=()):
        self.pid, self.born, self.rss, self.cpu = pid, born, rss, cpu
        self.descendants = descendants
        self.alive = True

    def create_time(self):
        return self.born

    def is_running(self):
        return self.alive

    def status(self):
        return 'running' if self.alive else 'zombie'

    def children(self, recursive=False):
        assert recursive
        return list(self.descendants)

    def memory_info(self):
        return SimpleNamespace(rss=self.rss)

    def cpu_times(self):
        return SimpleNamespace(user=self.cpu, system=0.0)


def sampler(tmp_path, root, *, time_values=None, gpu=None):
    times = iter(time_values or [10.0, 15.0, 20.0])
    return JobResourceSampler(
        job_kind='post_sparse_dense', source_session_id=SID,
        dense_run_id=RID, root_pid=root.pid, workspace=tmp_path,
        process_factory=lambda pid: root if pid == root.pid else None,
        clock=lambda: next(times), gpu_reader=gpu or (lambda: ()),
        ram_reader=lambda: 4 * 1024**3,
        disk_reader=lambda path: 20 * 1024**3,
        interval_seconds=5.0,
    )


def test_requires_observed_stage_and_real_process_identity(tmp_path):
    root = FakeProcess(401)
    observer = sampler(tmp_path, root)
    assert observer.sample_once() is None
    observer.set_stage('dense_patch_match')
    event = observer.sample_once()
    assert isinstance(event, JobTelemetryObservation)
    assert event.producer == 'resource_sampler'
    assert event.stage == 'dense_patch_match'
    assert event.source_session_id == SID and event.dense_run_id == RID
    assert event.counter is None and 'progress_fraction' not in event.as_dict()
    assert len(event.resource_readings) <= 12
    root.born = 2.0  # reused PID must not be sampled as our old job
    assert observer.sample_once() is None


def test_cpu_is_delta_not_instant_guess_and_rss_is_explicit_sum(tmp_path):
    child = FakeProcess(402, rss=4096, cpu=3)
    root = FakeProcess(401, rss=2048, cpu=2, descendants=(child,))
    observer = sampler(tmp_path, root)
    observer.set_stage('dense_fusion')
    one = observer.sample_once()
    first = {r.name: r for r in one.resource_readings}
    assert first['process_tree_rss_bytes'].value == 6144
    assert first['process_tree_cpu_percent'].availability == 'unavailable'
    assert first['process_tree_cpu_percent'].value is None
    assert first['system_ram_available_bytes'].scope == 'system'
    assert first['workspace_disk_free_bytes'].scope == 'system'
    root.cpu += 4.0
    child.cpu += 1.0
    two = observer.sample_once()
    second = {r.name: r for r in two.resource_readings}
    assert second['process_tree_cpu_percent'].value == pytest.approx(100.0)
    assert two.sequence == one.sequence + 1
    assert not any(r.name == 'gpu_process_vram_used_bytes' for r in two.resource_readings)


def test_unavailable_is_null_not_zero_and_no_gpu_is_not_job_gpu(tmp_path):
    root = FakeProcess(401)
    observer = JobResourceSampler(
        job_kind='offline_sfm', source_session_id=SID, dense_run_id=None,
        root_pid=401, workspace=tmp_path, process_factory=lambda pid: root,
        clock=lambda: 10.0, gpu_reader=lambda: (),
        ram_reader=lambda: None, disk_reader=lambda path: None,
    )
    observer.set_stage('sfm_matching')
    event = observer.sample_once()
    reads = {r.name: r for r in event.resource_readings}
    assert event.job_kind == 'offline_sfm' and event.dense_run_id is None
    assert reads['system_ram_available_bytes'].availability == 'unavailable'
    assert reads['system_ram_available_bytes'].value is None
    assert reads['workspace_disk_free_bytes'].value is None
    assert reads['gpu_device_util_percent'].scope == 'gpu_device'
    assert reads['gpu_device_util_percent'].availability == 'unavailable'


def test_latest_resource_file_is_bounded_and_separate_from_progress(tmp_path):
    root = FakeProcess(401)
    observer = sampler(tmp_path, root)
    observer.set_stage('dense_preparing')
    event = observer.sample_once()
    assert observer.publish(event)
    saved = tmp_path / 'job_resource_observation.txt'
    assert saved.stat().st_size <= 4097
    assert parse_job_telemetry_line(saved.read_text().strip()) == event
    assert not (tmp_path / 'job_progress_observation.txt').exists()
    assert not (tmp_path / 'final_dense_admission.json').exists()
    observer.set_stage('dense_fusion')
    assert observer.publish(observer.sample_once())
    assert len(list(tmp_path.iterdir())) == 1


def test_failsoft_bad_sensor_and_absent_directory(tmp_path):
    root = FakeProcess(401)
    observer = JobResourceSampler(
        job_kind='post_sparse_dense', source_session_id=SID, dense_run_id=RID,
        root_pid=401, workspace=tmp_path / 'missing', process_factory=lambda pid: root,
        clock=lambda: 10.0,
        gpu_reader=lambda: (_ for _ in ()).throw(OSError('nvidia-smi failed')),
        ram_reader=lambda: (_ for _ in ()).throw(OSError('RAM probe failed')),
        disk_reader=lambda path: (_ for _ in ()).throw(OSError('disk probe failed')),
    )
    observer.set_stage('dense_preparing')
    event = observer.sample_once()
    assert event is not None
    assert all(r.availability == 'unavailable' for r in event.resource_readings
               if r.scope in {'system', 'gpu_device'})
    assert observer.publish(event) is False
    assert not (tmp_path / 'missing').exists()


def test_gpu_parser_is_device_scoped_bounded_and_rejects_bad_rows():
    vals = gpu_device_readings(((0, 70.0, 512), (1, 0.0, 0),
                                (2, 55.0, 1024), (3, 5.0, 256),
                                (4, 99.0, 2048)))
    assert len(vals) == 8
    assert vals[0].name == 'gpu_device_util_percent' and vals[0].gpu_index == 0
    assert vals[1].value == 512 * 1024**2
    assert all(r.scope == 'gpu_device' for r in vals)
    unavailable = gpu_device_readings(((0, 999, -1), ('bad', 3, 5)))
    assert all(r.availability == 'unavailable' and r.value is None for r in unavailable)


def test_real_linux_memavailable_parser_no_memfree_substitution(tmp_path):
    src = tmp_path / 'meminfo'
    src.write_text('MemTotal: 15000000 kB\nMemFree: 500000 kB\nMemAvailable: 7000000 kB\n')
    assert read_available_ram_bytes(src) == 7000000 * 1024
    src.write_text('MemFree: 500000 kB\n')
    assert read_available_ram_bytes(src) is None


def test_sampling_interval_is_bounded_and_invalid_stage_rejected(tmp_path):
    root = FakeProcess(401)
    with pytest.raises(ValueError):
        JobResourceSampler(job_kind='post_sparse_dense', source_session_id=SID,
                           dense_run_id=RID, root_pid=401, workspace=tmp_path,
                           interval_seconds=.01, process_factory=lambda pid: root)
    obs = sampler(tmp_path, root)
    with pytest.raises(ValueError):
        obs.set_stage('sfm_matching')


def test_resource_hook_does_not_touch_original_progress_protocol():
    root = Path(__file__).resolve().parents[2]
    sfm = (root / 'core/reconstruction/sfm_pycolmap.py').read_text()
    dense = (root / 'scripts/run_i6_7e2b2a_post_sparse_dense_worker.py').read_text()
    assert 'JobResourceSampler' in sfm and 'JobResourceSampler' in dense
    assert 'sampler.set_stage(' in sfm and 'sampler.set_stage(' in dense
    assert 'sampler.stop()' in sfm and 'sampler.stop()' in dense


def test_dense_resource_lifecycle_follows_observed_snapshots_without_new_job(tmp_path, monkeypatch):
    from argparse import Namespace
    from scripts import run_i6_7e2b2a_post_sparse_dense_worker as worker
    from tests.i6_7c_helpers import make_snapshot, replace_snapshot

    events = []

    class FakeSampler:
        def __init__(self, **kwargs):
            events.append(('bind', kwargs['root_pid'], kwargs['dense_run_id']))

        def start(self):
            events.append(('start',))

        def set_stage(self, stage):
            events.append(('stage', stage))

        def stop(self):
            events.append(('stop',))

    args = Namespace(
        session_id=SID, dense_run_id=RID, offline_output_dir=tmp_path,
        pycolmap_python=tmp_path / 'unused_python',
        mvs_worker_script=tmp_path / 'unused_worker',
        provider_id=worker.I43B_PROVIDER_ID, timeout_minutes=1.0,
    )
    monkeypatch.setattr(worker, '_parse_args', lambda _: args)
    monkeypatch.setattr(worker, 'JobResourceSampler', FakeSampler)
    monkeypatch.setattr(worker, 'print', lambda *a, **kw: None, raising=False)

    def fake_execute(**kwargs):
        (tmp_path / 'dense_runs' / RID).mkdir(parents=True)
        kwargs['snapshot_sink'].publish(
            replace_snapshot(make_snapshot(), stage_detail='patch_match_stereo'))
        kwargs['snapshot_sink'].publish(
            replace_snapshot(make_snapshot(revision=2), stage_detail='stereo_fusion'))
        return type('Result', (), {'as_dict': lambda self: {'fake_test_only': True}})()

    monkeypatch.setattr(worker, 'execute_explicit_colmap_dense_after_sparse', fake_execute)
    assert worker.main() == 0
    assert events[0][0] == 'bind' and events[0][2] == RID
    assert events[1:] == [
        ('start',), ('stage', 'dense_preparing'),
        ('stage', 'dense_patch_match'), ('stage', 'dense_fusion'), ('stop',),
    ]


def test_dense_resource_sampler_stops_on_real_worker_failure(tmp_path, monkeypatch):
    from argparse import Namespace
    from scripts import run_i6_7e2b2a_post_sparse_dense_worker as worker

    calls = []

    class FakeSampler:
        def __init__(self, **kw):
            calls.append('bound')
        def start(self):
            calls.append('start')
        def set_stage(self, stage):
            calls.append(stage)
        def stop(self):
            calls.append('stop')

    args = Namespace(
        session_id=SID, dense_run_id=RID, offline_output_dir=tmp_path,
        pycolmap_python=tmp_path / 'unused_python',
        mvs_worker_script=tmp_path / 'unused_worker',
        provider_id=worker.I43B_PROVIDER_ID, timeout_minutes=1.0,
    )
    monkeypatch.setattr(worker, '_parse_args', lambda _: args)
    monkeypatch.setattr(worker, 'JobResourceSampler', FakeSampler)
    stdout = []
    monkeypatch.setattr(worker, 'print', lambda *args, **kw: stdout.extend(args), raising=False)
    monkeypatch.setattr(worker, 'execute_explicit_colmap_dense_after_sparse',
                        lambda **kw: (_ for _ in ()).throw(RuntimeError('test failed worker')))
    assert worker.main() == 2
    assert calls == ['bound', 'start', 'dense_preparing', 'stop']
    assert any('"final_dense_result_admitted":false' in str(line) for line in stdout)

#!/usr/bin/env python3
"""I6.8F bounded telemetry transport/host probe; NEVER starts reconstruction.

Two explicitly separate modes:
* transport: many local diagnostic events, measured poll latency and history bound.
* host: samples one short-lived child owned by this probe and actual host sensors.

Neither mode qualifies real SfM/MVS stage emission or end-to-end reconstruction,
nor does it persist a reconstruction scene or modify any existing session.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
import tracemalloc
from uuid import uuid4

# The script is invoked from the repository root, but its own directory is
# sys.path[0]; keep imports independent of the caller's PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.reconstruction_job_telemetry_presentation import JobTelemetryPanelState
from core.reconstruction.reconstruction_job_progress_observer import LatestJobProgressSnapshot
from core.reconstruction.reconstruction_job_resource_sampler import JobResourceSampler
from core.reconstruction.reconstruction_job_telemetry_contract import JobTelemetryObservation, StageWorkCounter
from core.reconstruction.reconstruction_job_telemetry_transport import (
    JobTelemetrySidecarTransport, TELEMETRY_HISTORY_LIMIT,
)


def _observation(session_id: str, run_id: str, producer: str, sequence: int) -> JobTelemetryObservation:
    return JobTelemetryObservation(
        source_session_id=session_id, job_kind='post_sparse_dense',
        dense_run_id=run_id, producer=producer, sequence=sequence,
        observed_at_utc=datetime.now(timezone.utc).isoformat(),
        stage='dense_patch_match' if producer != 'resource_sampler' else 'dense_preparing',
        detail='Synthetic qualification fixture; NOT a running reconstruction',
        counter=StageWorkCounter('geometric_depth_maps_observed', sequence, None)
        if producer != 'resource_sampler' else None,
    )


def transport_probe(workspace: Path, count: int = 256) -> dict[str, object]:
    """Pure transport microbenchmark; no worker and no resource sampling."""
    if type(count) is not int or not 32 <= count <= 1024:
        raise ValueError('transport count must be 32..1024')
    workspace = Path(workspace)
    if not workspace.is_dir() or workspace.is_symlink():
        raise ValueError('probe workspace must be an existing ordinary directory')
    session_id, run_id = str(uuid4()), str(uuid4())
    relay = JobTelemetrySidecarTransport('post_sparse_dense', session_id, run_id, workspace)
    progress = LatestJobProgressSnapshot(workspace / 'job_progress_observation.txt')
    resources = LatestJobProgressSnapshot(workspace / 'job_resource_observation.txt')
    panel = JobTelemetryPanelState()
    panel.begin('post_sparse_dense', session_id, run_id)
    poll_ms: list[float] = []
    tracemalloc.start()
    try:
        for sequence in range(1, count + 1):
            # These are synthetic, timestamped fixtures, not an actual provider.
            if not progress.publish(_observation(session_id, run_id, 'dense_supervisor', sequence)):
                raise RuntimeError('diagnostic progress publication failed')
            if not resources.publish(_observation(session_id, run_id, 'resource_sampler', sequence)):
                raise RuntimeError('diagnostic resource publication failed')
            start = time.perf_counter_ns()
            accepted = relay.poll()
            poll_ms.append((time.perf_counter_ns() - start) / 1_000_000)
            if len(accepted) != 2 or any(not panel.accept(event) for event in accepted):
                raise RuntimeError('diagnostic observations lost or misattributed')
            if len(relay.history) > TELEMETRY_HISTORY_LIMIT:
                raise RuntimeError('transport history exceeded documented bound')
            if relay.poll():
                raise RuntimeError('duplicate latest-only observations were redelivered')
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
        relay.close()
    if panel.progress is None or panel.progress.sequence != count:
        raise RuntimeError('latest-stage presentation did not advance')
    if panel.resources is None or panel.resources.sequence != count:
        raise RuntimeError('latest-resource presentation did not advance')
    sorted_ms = sorted(poll_ms)
    return {
        'count': count,
        'poll_median_ms': round(statistics.median(poll_ms), 4),
        'poll_p95_ms': round(sorted_ms[math.ceil(.95 * count) - 1], 4),
        'poll_max_ms': round(max(poll_ms), 4),
        'tracemalloc_peak_bytes': peak_bytes,
        'history_length': len(relay.history),
        'workload': 'synthetic sidecar events; no reconstruction or Qt rendering',
    }


# Deliberately bounded to one short-lived child and a small working set; no imports
# from reconstruction workers, no CUDA, no datasets and no project session paths.
_DIAGNOSTIC_CHILD = '''import time\na = bytearray(8 * 1024 * 1024)\nstop = time.monotonic() + 12.0\nx = 0\nwhile time.monotonic() < stop:\n    x = (x * 1664525 + 1013904223) & 0xffffffff\n'''


def host_probe(workspace: Path) -> dict[str, object]:
    """Observe a probe-owned process and real host sensors; NOT an SfM/MVS job."""
    session_id = str(uuid4())
    child = subprocess.Popen(
        [sys.executable, '-c', _DIAGNOSTIC_CHILD],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    sampler = None
    try:
        sampler = JobResourceSampler(
            job_kind='offline_sfm', source_session_id=session_id,
            dense_run_id=None, root_pid=child.pid, workspace=workspace,
            interval_seconds=2.0,
        )
        # Synthetic marker solely to exercise sampler's observed-stage API.
        # It is not an assertion that an actual SfM source-ingestion stage ran.
        sampler.set_stage('source_ingestion')
        receiver = JobTelemetrySidecarTransport('offline_sfm', session_id, None, workspace)
        observed = []
        per_sample_ms = []
        for _ in range(3):
            time.sleep(2.1)
            if child.poll() is not None:
                raise RuntimeError('diagnostic child exited before bounded samples')
            start = time.perf_counter_ns()
            event = sampler.sample_once()
            per_sample_ms.append((time.perf_counter_ns() - start) / 1_000_000)
            if event is None or not sampler.publish(event):
                raise RuntimeError('diagnostic host sample unavailable for publishing')
            accepted = receiver.poll()
            if accepted != (event,):
                raise RuntimeError('diagnostic host sample not delivered to its bound receiver')
            observed.append(event)
        latest = observed[-1]
        resource_by_name = {r.name: r for r in latest.resource_readings}
        cpu = resource_by_name['process_tree_cpu_percent']
        rss = resource_by_name['process_tree_rss_bytes']
        ram = resource_by_name['system_ram_available_bytes']
        disk = resource_by_name['workspace_disk_free_bytes']
        gpu = [r for r in latest.resource_readings if r.scope == 'gpu_device']
        result = {
            'samples': len(observed), 'child_pid': child.pid,
            'sampling_max_ms': round(max(per_sample_ms), 3),
            'worker_tree_cpu_state': cpu.availability,
            'worker_tree_cpu_percent': cpu.value,
            'worker_tree_rss_state': rss.availability,
            'worker_tree_rss_bytes': rss.value,
            'host_ram_state': ram.availability, 'host_ram_available_bytes': ram.value,
            'workspace_disk_state': disk.availability, 'workspace_disk_free_bytes': disk.value,
            'gpu_device_state': ','.join(
                f'{r.name}@{r.gpu_index}:{r.availability}' for r in gpu),
            'history_length': len(receiver.history),
            'workload': 'probe-owned diagnostic CPU process; NOT real SfM/MVS',
        }
        receiver.close()
        return result
    finally:
        if sampler is not None:
            sampler.stop()
        if child.poll() is None:
            child.terminate()  # only the exact child this probe launched
        try:
            child.wait(timeout=3)
        except subprocess.TimeoutExpired:
            child.kill()  # only the exact child this probe launched
            child.wait(timeout=3)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('transport', 'host', 'all'), default='all')
    parser.add_argument('--events', type=int, default=256, help='32..1024 synthetic events')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='depth3d_i6_8f_diagnostic_') as directory:
        workspace = Path(directory)
        if args.mode in ('transport', 'all'):
            print('===== BOUNDED TRANSPORT MICROBENCHMARK =====', flush=True)
            for key, value in transport_probe(workspace, args.events).items():
                print(f'{key.upper()}={value}', flush=True)
        if args.mode in ('host', 'all'):
            print('===== BOUNDED HOST SENSOR PROBE =====', flush=True)
            for key, value in host_probe(workspace).items():
                print(f'{key.upper()}={value}', flush=True)
    print('I6_8F_DIAGNOSTIC_PROBE=PASS')
    print('REAL_RECONSTRUCTION_JOB_TELEMETRY=NOT_QUALIFIED_BY_THIS_PROBE')
    print('I6_8_CLOSURE=DEFERRED_PENDING_REAL_JOB_LIFECYCLE_EVIDENCE')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f'I6_8F_DIAGNOSTIC_PROBE=FAIL: {error}', file=sys.stderr)
        raise SystemExit(1)

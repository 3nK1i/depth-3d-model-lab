"""I6.8C passive, bounded, job-scoped resource observations (Linux desktop).

The sampler never controls a worker, modifies scientific outputs, or guesses job
completion. Process-tree RSS is the sum of resident sizes, NOT unique physical
RAM or GPU process VRAM. CPU percent is measured against ONE logical CPU and
may exceed 100 on a multi-core job. NVIDIA utilization/VRAM is device-wide.
"""
from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import subprocess
from threading import Event, Lock, Thread
from time import monotonic
from typing import Callable, Iterable

from core.reconstruction.reconstruction_job_progress_observer import LatestJobProgressSnapshot
from core.reconstruction.reconstruction_job_telemetry_contract import (
    JobTelemetryObservation, ResourceReading, TelemetryValidationError,
)

SAMPLE_INTERVAL_SECONDS = 5.0
MAX_GPU_DEVICES = 4


def read_available_ram_bytes(path: Path = Path('/proc/meminfo')) -> int | None:
    """Linux MemAvailable, never substitute MemFree or claim allocated GPU VRAM."""
    try:
        for line in path.read_text(encoding='utf-8').splitlines():
            if not line.startswith('MemAvailable:'):
                continue
            words = line.split()
            if len(words) == 3 and words[2] == 'kB' and words[1].isdigit():
                return int(words[1]) * 1024
    except (OSError, UnicodeError, ValueError):
        pass
    return None


def _workspace_free_bytes(path: Path) -> int | None:
    try:
        if not path.is_dir():
            return None
        return int(shutil.disk_usage(path).free)
    except (OSError, ValueError):
        return None


def _query_nvidia_devices() -> tuple[tuple[int, float, int], ...]:
    """One bounded external probe per sample, no GPU allocation or Python CUDA import."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=False, timeout=1.5,
        )
        if result.returncode != 0 or len(result.stdout) > 4096:
            return ()
        rows = []
        for line in result.stdout.splitlines():
            parts = [p.strip() for p in line.split(',')]
            if len(parts) != 3:
                continue
            try:
                index, util, memory = int(parts[0]), float(parts[1]), int(parts[2])
            except ValueError:
                continue
            rows.append((index, util, memory))
        return tuple(rows[:MAX_GPU_DEVICES])
    except (OSError, subprocess.TimeoutExpired, UnicodeError, ValueError):
        return ()


def gpu_device_readings(rows: Iterable[tuple[int, float, int]] | None) -> tuple[ResourceReading, ...]:
    """Only device-wide readings; missing/invalid sensors yield null, never zero."""
    result: list[ResourceReading] = []
    seen: set[int] = set()
    try:
        candidates = iter(rows or ())
    except TypeError:
        candidates = iter(())
    for row in candidates:
        if len(seen) >= MAX_GPU_DEVICES:
            break
        try:
            index, util, memory_mib = row
            if type(index) is not int or not 0 <= index <= 256 or index in seen:
                continue
            seen.add(index)
            valid_util = (type(util) in (float, int) and 0 <= util <= 100)
            valid_mem = (type(memory_mib) is int and 0 <= memory_mib <= 2**32)
            result.append(ResourceReading(
                'gpu_device_util_percent', util if valid_util else None,
                'observed' if valid_util else 'unavailable', gpu_index=index))
            result.append(ResourceReading(
                'gpu_device_vram_used_bytes', memory_mib * 1024**2 if valid_mem else None,
                'observed' if valid_mem else 'unavailable', gpu_index=index))
        except (TypeError, ValueError, TelemetryValidationError):
            continue
    if not result:
        # Index zero indicates the attempted default GPU query, not confirmed
        # NVIDIA device presence. The value remains explicitly unavailable.
        result = [
            ResourceReading('gpu_device_util_percent', None, 'unavailable', gpu_index=0),
            ResourceReading('gpu_device_vram_used_bytes', None, 'unavailable', gpu_index=0),
        ]
    return tuple(result)


def _optional_process_factory(pid: int):
    try:
        import psutil  # optional in the source environment; never mandatory for SfM
        return psutil.Process(pid)
    except (ImportError, OSError, ValueError):
        return None


def _reading(name: str, value: int | float | None) -> ResourceReading:
    try:
        if value is None:
            return ResourceReading(name, None, 'unavailable')
        return ResourceReading(name, value, 'observed')
    except (TypeError, ValueError, TelemetryValidationError):
        return ResourceReading(name, None, 'unavailable')


class JobResourceSampler:
    """Optional passive probe of ONE bound worker/process tree.

    The thread only writes a latest-only diagnostic resource sidecar. I6.8D
    owns Qt transport/session filtering. Missing sensors or sidecar are safe.
    """

    def __init__(
        self, *, job_kind: str, source_session_id: str, dense_run_id: str | None,
        root_pid: int, workspace: Path, interval_seconds: float = SAMPLE_INTERVAL_SECONDS,
        process_factory: Callable[[int], object] = _optional_process_factory,
        clock: Callable[[], float] = monotonic,
        gpu_reader: Callable[[], Iterable[tuple[int, float, int]]] = _query_nvidia_devices,
        ram_reader: Callable[[], int | None] = read_available_ram_bytes,
        disk_reader: Callable[[Path], int | None] = _workspace_free_bytes,
    ) -> None:
        if type(root_pid) is not int or root_pid <= 0:
            raise ValueError('resource sampler requires a valid worker PID')
        if not isinstance(interval_seconds, (float, int)) or not 2.0 <= interval_seconds <= 60.0:
            raise ValueError('resource sampling interval must be within 2–60 seconds')
        if job_kind not in {'offline_sfm', 'post_sparse_dense'}:
            raise ValueError('unknown reconstruction job kind')
        self._source_session_id = source_session_id
        self._job_kind = job_kind
        self._dense_run_id = dense_run_id
        self._root_pid = root_pid
        self._workspace = Path(workspace)
        self._interval = float(interval_seconds)
        self._factory = process_factory
        self._clock = clock
        self._gpu_reader = gpu_reader
        self._ram_reader = ram_reader
        self._disk_reader = disk_reader
        self._sink = LatestJobProgressSnapshot(self._workspace / 'job_resource_observation.txt')
        self._gate = Lock()
        self._stage: str | None = None
        self._sequence = 0
        self._previous_cpu: tuple[float, float] | None = None
        self._stop = Event()
        self._thread: Thread | None = None
        self._born: float | None = None
        try:
            root = self._factory(self._root_pid)
            self._born = float(root.create_time()) if root is not None else None
        except (OSError, AttributeError, ValueError):
            self._born = None
        # Validate job binding eagerly even before the stage is known.
        if job_kind == 'post_sparse_dense' and dense_run_id is None:
            raise ValueError('dense resource sampler requires the run UUID')
        if job_kind == 'offline_sfm' and dense_run_id is not None:
            raise ValueError('SfM resource sampler must not bind a dense run')

    def set_stage(self, stage: str) -> None:
        # Only a stage actually observed from the existing producer is usable.
        candidate = JobTelemetryObservation(
            source_session_id=self._source_session_id,
            job_kind=self._job_kind, dense_run_id=self._dense_run_id,
            producer='resource_sampler', sequence=1,
            observed_at_utc=datetime.now(timezone.utc).isoformat(),
            stage=stage, detail='Observed worker stage; resource sampling pending',
        )
        with self._gate:
            self._stage = candidate.stage

    def _process_readings(self, now: float) -> tuple[ResourceReading, ResourceReading] | None:
        if self._born is None:
            return (_reading('process_tree_rss_bytes', None),
                    _reading('process_tree_cpu_percent', None))
        try:
            root = self._factory(self._root_pid)
            if (root is None or float(root.create_time()) != self._born or
                    not root.is_running() or root.status() == 'zombie'):
                self._stop.set()  # never adopt a reused PID or finished worker
                return None
            # Recursive children of the bound root only, limited for low overhead.
            descendants = list(root.children(recursive=True))
            if len(descendants) > 128:
                # A partial tree sum would be mislabelled as complete job RSS.
                self._previous_cpu = None
                return (_reading('process_tree_rss_bytes', None),
                        _reading('process_tree_cpu_percent', None))
            processes = [root, *descendants]
            rss = 0
            cpu_total = 0.0
            for proc in processes:
                try:
                    if not proc.is_running() or proc.status() == 'zombie':
                        continue
                    rss += max(0, int(proc.memory_info().rss))
                    times = proc.cpu_times()
                    cpu_total += max(0.0, float(times.user) + float(times.system))
                except (OSError, AttributeError, ValueError):
                    continue  # a disappearing child does not erase the bound root
            cpu_percent = None
            if self._previous_cpu is not None:
                previous_time, previous_cpu = self._previous_cpu
                if now > previous_time and cpu_total >= previous_cpu:
                    cpu_percent = 100.0 * (cpu_total - previous_cpu) / (now - previous_time)
            self._previous_cpu = (now, cpu_total)
            return (_reading('process_tree_rss_bytes', rss),
                    _reading('process_tree_cpu_percent', cpu_percent))
        except Exception:
            # psutil may raise NoSuchProcess/AccessDenied outside OSError's
            # hierarchy; unavailable is safer than failing reconstruction.
            self._previous_cpu = None
            return (_reading('process_tree_rss_bytes', None),
                    _reading('process_tree_cpu_percent', None))

    def sample_once(self) -> JobTelemetryObservation | None:
        with self._gate:
            stage = self._stage
        if stage is None or self._stop.is_set():
            return None
        # If the worker vanished/reused its PID, do not emit host-only values
        # under a job identity that may no longer be active.
        now = self._clock()
        process = self._process_readings(now)
        if process is None:
            return None
        try:
            ram = self._ram_reader()
        except Exception:
            ram = None
        try:
            disk = self._disk_reader(self._workspace)
        except Exception:
            disk = None
        try:
            gpu = gpu_device_readings(self._gpu_reader())
        except Exception:
            gpu = gpu_device_readings(())
        # Sampling NVIDIA devices can take longer than a stage transition.
        # Label the resulting sample with the latest observed stage, never an
        # older stage captured before the sensor query.
        with self._gate:
            stage = self._stage
        if stage is None or self._stop.is_set():
            return None
        self._sequence += 1
        return JobTelemetryObservation(
            source_session_id=self._source_session_id,
            job_kind=self._job_kind, dense_run_id=self._dense_run_id,
            producer='resource_sampler', sequence=self._sequence,
            observed_at_utc=datetime.now(timezone.utc).isoformat(),
            stage=stage, detail='Passive resource sample; no completion or ETA claim',
            resource_readings=(*process,
                               _reading('system_ram_available_bytes', ram),
                               _reading('workspace_disk_free_bytes', disk),
                               *gpu),
        )

    def publish(self, observation: JobTelemetryObservation | None) -> bool:
        try:
            return self._sink.publish(observation)
        except Exception:
            return False

    def _run(self) -> None:
        # Wait before first sample: short runs incur no additional nvidia-smi
        # process; at most one active probe per sampler, no unbounded queue.
        while not self._stop.wait(self._interval):
            try:
                event = self.sample_once()
                if event is not None:
                    self.publish(event)
            except Exception:
                pass  # diagnostic sensor failures must never change reconstruction

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = Thread(target=self._run, name='reconstruction-resource-sampler', daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


__all__ = ['JobResourceSampler', 'SAMPLE_INTERVAL_SECONDS', 'MAX_GPU_DEVICES',
           'gpu_device_readings', 'read_available_ram_bytes']

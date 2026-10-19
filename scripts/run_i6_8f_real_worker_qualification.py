#!/usr/bin/env python3
"""I6.8F: one resource-bounded *actual* offline-SfM worker/Qt-relay telemetry probe.

Copies up to 24 existing, small, adjacent real source images into a private temporary
workspace; never alters an existing source or historical reconstruction session.
Does not start MVS, PatchMatch, fusion, the main GUI, or a persistent scene.
An explicit --run is required; default is read-only preflight.
"""
from __future__ import annotations

import argparse
from collections import deque
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
from threading import Thread
from time import monotonic, sleep
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction.reconstruction_job_telemetry_transport import (
    JobTelemetrySidecarTransport, TELEMETRY_HISTORY_LIMIT,
)

DEFAULT_SOURCE = Path('/tmp/depth3d_reconstruction/b0e5e51f-6975-4152-822f-e28602842f58/sfm_workspace/images')
DEFAULT_SFM = Path.home() / 'anaconda3/envs/depth3d-pycolmap-probe/bin/python'
IMAGE_EXTENSIONS = frozenset(('.jpg', '.jpeg', '.png'))
MAX_SOURCE_BYTES = 12 * 1024**2
MAX_TOTAL_SOURCE_BYTES = 80 * 1024**2
MAX_DIMENSION = 1280
IMAGE_COUNT = 24
MIN_IMAGE_COUNT = 8
MIN_AVAILABLE_GIB = 4.0
MIN_DISK_GIB = 10.0
MAX_TREE_RSS_GIB = 5.0
MAX_SCRATCH_BYTES = 1 * 1024**3
MAX_RUN_SECONDS = 120


def natural_key(path: Path):
    return tuple((1, int(piece)) if piece.isdigit() else (0, piece.casefold())
                 for piece in re.split(r'(\d+)', path.name))


def select_source_images(source: Path, *, count: int = IMAGE_COUNT) -> tuple[Path, ...]:
    """Inventory only one specified folder; never scan the user's disk recursively."""
    if type(count) is not int or not MIN_IMAGE_COUNT <= count <= IMAGE_COUNT:
        raise ValueError('source image count must be 8..24')
    source = Path(source).expanduser()
    if not source.is_dir() or source.is_symlink():
        raise ValueError('source image directory must exist and not be a symlink')
    files = sorted((p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS),
                   key=natural_key)
    if len(files) < count:
        raise ValueError(f'need {count} existing real input images; found {len(files)}')
    start = (len(files) - count) // 2
    selected = tuple(files[start:start + count])
    total = 0
    for path in selected:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'source image is not an ordinary file: {path.name}')
        size = path.stat().st_size
        if not 0 < size <= MAX_SOURCE_BYTES:
            raise ValueError(f'image outside safe 0..12 MiB limit: {path.name}')
        total += size
    if total > MAX_TOTAL_SOURCE_BYTES:
        raise ValueError('selected images exceed 80 MiB input bound')
    # Image.open only inspects headers; it does not decode/resample/alter input.
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError('Pillow is required for bounded source-dimension preflight') from exc
    for path in selected:
        with Image.open(path) as img:
            width, height = img.size
        if not (160 <= width <= MAX_DIMENSION and 120 <= height <= MAX_DIMENSION):
            raise ValueError(f'image size outside bounded fixture limits: {path.name} ({width}x{height})')
    return selected


def sfm_runtime_check(python: Path) -> str:
    python = Path(python).expanduser()
    if not python.is_file():
        raise ValueError(f'isolated PyCOLMAP interpreter is absent: {python}')
    command = [str(python), '-c', 'import pycolmap; print(pycolmap.__version__)']
    try:
        result = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True,
                                text=True, timeout=12, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError('isolated PyCOLMAP interpreter probe failed') from exc
    version = result.stdout.strip()
    if result.returncode != 0 or version != '4.1.1':
        raise RuntimeError(f'isolated PyCOLMAP 4.1.1 required; probe returned {version!r}, rc={result.returncode}')
    return version


def mem_available_bytes() -> int:
    from core.reconstruction.reconstruction_job_resource_sampler import read_available_ram_bytes
    return read_available_ram_bytes() or 0


def scratch_bytes(directory: Path) -> int:
    total = 0
    for root, _, files in os.walk(directory, followlinks=False):
        for filename in files:
            path = Path(root) / filename
            if path.is_symlink():
                raise RuntimeError('scratch output contains an unexpected symlink')
            try:
                total += path.stat().st_size
            except FileNotFoundError:
                continue  # live worker can atomically replace a small sidecar
            if total > MAX_SCRATCH_BYTES:
                return total
    return total


def safety_check(directory: Path, process=None) -> tuple[float, float, float]:
    """Independent *hard stop* checks, not passive telemetry sampling."""
    available_gib = mem_available_bytes() / 1024**3
    disk_gib = shutil.disk_usage(directory).free / 1024**3
    tree_rss_gib = 0.0
    if process is not None and process.poll() is None:
        import psutil
        try:
            owner = psutil.Process(process.pid)
            for member in [owner, *owner.children(recursive=True)]:
                try:
                    tree_rss_gib += member.memory_info().rss / 1024**3
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    if available_gib < MIN_AVAILABLE_GIB:
        raise RuntimeError(f'host available RAM below {MIN_AVAILABLE_GIB} GiB: {available_gib:.2f}')
    if disk_gib < MIN_DISK_GIB:
        raise RuntimeError(f'free disk below {MIN_DISK_GIB} GiB: {disk_gib:.2f}')
    if process is not None and scratch_bytes(directory) > MAX_SCRATCH_BYTES:
        raise RuntimeError('owned scratch outputs exceeded 1 GiB disk bound')
    if tree_rss_gib > MAX_TREE_RSS_GIB:
        raise RuntimeError(f'worker tree exceeded {MAX_TREE_RSS_GIB} GiB RSS: {tree_rss_gib:.2f}')
    return available_gib, disk_gib, tree_rss_gib


def terminate_owned_group(child: subprocess.Popen) -> None:
    """Only the unique process group that this harness created can be signalled."""
    if child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=4)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait(timeout=4)


def _start_line_reader(child: subprocess.Popen, tail: deque[str]) -> Thread:
    def read_lines():
        assert child.stdout is not None
        for line in child.stdout:
            tail.append(line.rstrip()[:700])
    thread = Thread(target=read_lines, name='i6-8f-owned-worker-stdout', daemon=True)
    thread.start()
    return thread


def real_worker_probe(images: tuple[Path, ...], sfm_python: Path) -> dict[str, object]:
    """A new disposable session, real production offline worker, real sidecars, Qt relay."""
    import psutil  # hard resource-stop requires process tree enumeration
    from PyQt6.QtCore import QCoreApplication
    from app.reconstruction_job_telemetry_relay import JobTelemetryQtRelay
    from app.reconstruction_job_telemetry_presentation import JobTelemetryPanelState

    app = QCoreApplication.instance() or QCoreApplication([])
    session_id = str(uuid4())
    with tempfile.TemporaryDirectory(prefix='depth3d_i6_8f_real_sfm_') as temp:
        workspace = Path(temp)
        input_dir = workspace / 'input_images'
        output_dir = workspace / session_id
        input_dir.mkdir()
        output_dir.mkdir()
        safety_check(workspace)
        for source in images:
            shutil.copy2(source, input_dir / source.name)
        panel = JobTelemetryPanelState()
        panel.begin('offline_sfm', session_id, None)
        relay = JobTelemetryQtRelay()
        seen = deque(maxlen=128)
        ui_stage_samples = deque(maxlen=128)
        delivery_count = [0]

        def observed(event):
            seen.append(event)
            delivery_count[0] += 1
            if not panel.accept(event):
                raise AssertionError('real relay delivered an event rejected by bound presentation')
            ui_stage_samples.append(panel.present().stage)

        relay.observation_received.connect(observed)
        relay.begin('offline_sfm', session_id, None, output_dir)
        # The direct implementation of production's app-side worker, but no
        # source/controller side effects on existing user projects.
        cmd = [sys.executable, '-u', str(ROOT / 'scripts/run_reconstruction_offline_job.py'),
               '--images-dir', str(input_dir), '--session-id', session_id,
               '--sfm-python', str(sfm_python), '--output-dir', str(output_dir),
               '--profile', 'efficient', '--matching', 'sequential',
               '--max-keyframes', str(len(images)), '--maximum-image-size', '960',
               '--sequential-overlap', '3', '--cpu', '--timeout-minutes', '2']
        tail: deque[str] = deque(maxlen=35)
        child = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, bufsize=1, start_new_session=True)
        thread = _start_line_reader(child, tail)
        started_at = monotonic()
        deadline = started_at + MAX_RUN_SECONDS
        resource_sidecar = output_dir / 'job_resource_observation.txt'
        sidecar_seen_while_alive = False
        first_sidecar_elapsed_s = None
        stop_reason = ''
        max_rss = 0.0
        end_rc = None
        try:
            while True:
                app.processEvents()
                relay.poll()  # extra bounded poll, not a Qt render/FPS benchmark
                if child.poll() is None and resource_sidecar.is_file():
                    sidecar_seen_while_alive = True
                    if first_sidecar_elapsed_s is None:
                        first_sidecar_elapsed_s = monotonic() - started_at
                if child.poll() is not None:
                    end_rc = child.returncode
                    break
                if monotonic() >= deadline:
                    stop_reason = f'exceeded {MAX_RUN_SECONDS}s hard wall-clock bound'
                    break
                try:
                    _, _, rss_gib = safety_check(workspace, child)
                    max_rss = max(max_rss, rss_gib)
                except RuntimeError as exc:
                    stop_reason = str(exc)
                    break
                sleep(0.5)
        finally:
            if child.poll() is None:
                terminate_owned_group(child)
            child.wait(timeout=5)
            worker_finished_at = monotonic()
            # A subprocess could exit while a descendant remains in the unique
            # process group it created. Reap only that group's descendants.
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            thread.join(timeout=3)
            app.processEvents()
            relay.poll()  # last authentic sidecar update before terminal close
            elapsed_s = worker_finished_at - started_at
            final_sidecar_exists = resource_sidecar.is_file()
            final_sidecar_bytes = resource_sidecar.stat().st_size if final_sidecar_exists else 0
            relay.close()
        # A separate wrong-session receiver sees the genuine sidecars but cannot
        # attribute them to itself. No manufactured stage event is used.
        wrong = JobTelemetrySidecarTransport('offline_sfm', str(uuid4()), None, output_dir)
        foreign = wrong.poll()
        wrong.close()
        if foreign:
            raise RuntimeError('real sidecar leaked into another session')
        stages = tuple(dict.fromkeys(event.stage for event in seen if event.producer == 'sfm_worker'))
        resources = [event for event in seen if event.producer == 'resource_sampler']
        proc_observed = any(item.availability == 'observed' and
                            item.name in {'process_tree_rss_bytes', 'process_rss_bytes'}
                            for event in resources for item in event.resource_readings)
        pre_end = panel.present()
        panel.end('stopped' if stop_reason else ('completed' if end_rc == 0 else 'failed'))
        cleared = panel.present().status == 'terminal' and panel.progress is None and panel.resources is None
        if not cleared or relay.history and len(relay.history) > TELEMETRY_HISTORY_LIMIT:
            raise RuntimeError('terminal clearing or history bound failed')
        final_payload = output_dir / 'offline_result.json'
        status = 'not_written'
        if final_payload.is_file():
            try:
                outcome = json.loads(final_payload.read_text(encoding='utf-8'))
                status = 'success' if outcome.get('pass') is True else 'failure_or_unaccepted'
            except (OSError, ValueError):
                status = 'unreadable'
        passed = (not stop_reason and bool(stages) and bool(resources) and proc_observed
                  and any('Observed stage:' in text for text in ui_stage_samples)
                  and cleared and not foreign)
        return {
            'workload': f'REAL production offline SfM worker on {len(images)} copied existing images; CPU-only; no dense',
            'selected_image_count': len(images),
            'worker_wall_seconds': round(elapsed_s, 3),
            'resource_sampler_interval_seconds': 5.0,
            'resource_sidecar_seen_while_worker_alive': sidecar_seen_while_alive,
            'resource_sidecar_first_seen_seconds': round(first_sidecar_elapsed_s, 3) if first_sidecar_elapsed_s is not None else None,
            'resource_sidecar_exists_after_worker': final_sidecar_exists,
            'resource_sidecar_bytes_after_worker': final_sidecar_bytes,
            'session_id': session_id,
            'worker_returncode': end_rc if end_rc is not None else child.returncode,
            'worker_result_status': status,
            'stop_reason': stop_reason or 'none',
            'observed_real_stages': list(stages),
            'qt_relay_deliveries': delivery_count[0],
            'resource_observations': len(resources),
            'job_process_rss_observed': proc_observed,
            'max_seen_process_tree_rss_gib': round(max_rss, 3),
            'last_active_ui_stage': pre_end.stage,
            'terminal_ui_cleared': cleared,
            'wrong_session_deliveries': len(foreign),
            'qt_history_length': len(relay.history),
            'real_worker_transport_presentation': 'PASS' if passed else 'INCOMPLETE',
            'main_application_ui': 'NOT_TESTED_BY_THIS_CLI_PROBE',
            'dense_worker': 'NOT_TESTED',
            'worker_log_tail': [line[:240] for line in list(tail)[-4:]],
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images-dir', type=Path, default=DEFAULT_SOURCE,
                        help='existing real images folder; never modified')
    parser.add_argument('--sfm-python', type=Path,
                        default=Path(os.environ.get('DEPTH3D_RECONSTRUCTION_PYCOLMAP_PYTHON', str(DEFAULT_SFM))))
    parser.add_argument('--image-count', type=int, default=IMAGE_COUNT,
                        help='number of adjacent real source images, 8..24 (default 24; never changes hard limits)')
    parser.add_argument('--run', action='store_true', help='explicitly start ONE bounded real CPU SfM worker')
    args = parser.parse_args()
    try:
        selected = select_source_images(args.images_dir, count=args.image_count)
        # Default mode is read-only. No scratch workspace or worker is started.
        print('I6_8F_REAL_WORKER_PREFLIGHT=PASS')
        print('SOURCE_DIRECTORY=' + str(args.images_dir))
        print('SELECTED_IMAGE_COUNT=' + str(len(selected)))
        print('SELECTED_IMAGES=' + ','.join(item.name for item in selected))
        print('EXTERNAL_PYCOLMAP_VERSION=' + sfm_runtime_check(args.sfm_python))
        available, free, _ = safety_check(Path(tempfile.gettempdir()))
        print(f'HOST_AVAILABLE_RAM_GIB={available:.2f}')
        print(f'TEMP_FREE_DISK_GIB={free:.2f}')
        print('HARD_LIMITS=120s; 5GiB worker tree RSS; 1GiB scratch; 4GiB minimum host available; 10GiB free disk')
        print('EXPECTED_FIRST_RESOURCE_SAMPLE=at least 5s after isolated SfM subprocess starts; may be later')
        if not args.run:
            print('ACTUAL_RECONSTRUCTION_STARTED=NO')
            return 0
        result = real_worker_probe(selected, args.sfm_python)
        for key, value in result.items():
            print(f'{key.upper()}={json.dumps(value, sort_keys=True)}', flush=True)
        return 0 if result['real_worker_transport_presentation'] == 'PASS' else 3
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f'I6_8F_REAL_WORKER_PROBE=STOPPED: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())

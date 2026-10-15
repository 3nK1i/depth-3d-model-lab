#!/usr/bin/env python3
"""Explicit, bounded South Building stereo-fusion recovery; does NOT admit a result.

Never runs undistortion, SfM, PatchMatch, mesh creation, or automatic provider
selection. Consumes the interrupted I6.7F COLMAP workspace read-only and writes
recovery outputs under the interrupted dense run's fusion_recoveries/<new UUID>.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from uuid import UUID

SESSION_ID = '03d5f129-781f-419f-90aa-95673a5306fc'
RUN_ID = '273476cc-4221-4b43-9407-96c0ed6e63e5'
WORKER_SHA256 = '82b5de59687c691fda4959faa411149a3a5f4c8fa10e3d85da513cc254d32689'
REQUEST_SHA256 = '9400e0cd194b6b628f56ccae6835f66fec9c167c9168d739272bda383be6f1a4'
STATE_SHA256 = '37690ad09298cf6ecb41aace7e63e3a28b42dc1e033d15fa64865b1b7904e7bf'
GIB = 1024 ** 3


def require(test: bool, message: str) -> None:
    if not test:
        raise RuntimeError('South Building fusion-only recovery: ' + message)


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def canonical_file(path: Path) -> Path:
    require(path.is_absolute() and path.is_file() and not path.is_symlink(), f'file missing or symlinked: {path}')
    require(path.resolve(strict=True) == path, f'file has symlinked ancestor: {path}')
    return path


def canonical_dir(path: Path) -> Path:
    require(path.is_absolute() and path.is_dir() and not path.is_symlink(), f'directory missing or symlinked: {path}')
    require(path.resolve(strict=True) == path, f'directory has symlinked ancestor: {path}')
    return path


def read_pinned_json(path: Path, expected_sha: str) -> dict:
    canonical_file(path)
    require(path.stat().st_size < 4 * 1024 * 1024, f'oversized metadata: {path}')
    require(digest(path) == expected_sha, f'original evidence SHA256 changed: {path}')
    data = json.loads(path.read_text(encoding='utf-8'))
    require(isinstance(data, dict), f'metadata must be object: {path}')
    return data


def colmap_map_shape(path: Path, channels: int) -> tuple[int, int, int]:
    """Read only the COLMAP dense-map header; validate exact binary payload size."""
    canonical_file(path)
    with path.open('rb') as handle:
        header = handle.read(96)
    parts = header.split(b'&', 3)
    require(len(parts) == 4, f'malformed COLMAP dense-map header: {path.name}')
    require(all(p and p.isascii() and p.isdigit() for p in parts[:3]), f'noninteger dense-map dimensions: {path.name}')
    width, height, actual_channels = map(int, parts[:3])
    require(0 < width <= 16384 and 0 < height <= 16384 and actual_channels == channels,
            f'unsafe/map channel dimensions: {path.name}')
    header_size = sum(len(p) for p in parts[:3]) + 3
    require(path.stat().st_size == header_size + 4 * width * height * channels,
            f'incomplete/oversized dense-map payload: {path.name}')
    return width, height, actual_channels


def inspect_workspace(request: dict, dense: Path) -> dict:
    expected = request['image_names']
    require(isinstance(expected, list) and len(expected) == 87 and len(set(expected)) == 87,
            'the original request must contain exactly 87 distinct registered views')
    require(all(isinstance(name, str) and Path(name).name == name and name.endswith('.jpg') for name in expected),
            'unsafe or noncanonical image names')
    stereo = canonical_dir(dense / 'stereo')
    sizes = {}
    for kind, channels in (('depth_maps', 1), ('normal_maps', 3)):
        folder = canonical_dir(stereo / kind)
        observed = {p.name for p in folder.iterdir() if p.is_file()}
        names = {name + '.' + variant + '.bin' for name in expected for variant in ('geometric', 'photometric')}
        require(observed == names, f'{kind}: expected 174 exact files, missing={len(names-observed)} extra={len(observed-names)}')
        for name in expected:
            for variant in ('geometric', 'photometric'):
                path = folder / (name + '.' + variant + '.bin')
                shape = colmap_map_shape(path, channels)
                key = (name, variant)
                if kind == 'depth_maps':
                    sizes[key] = shape[:2]
                else:
                    require(sizes[key] == shape[:2], f'depth/normal dimension mismatch: {name} {variant}')
    return {'registered_views': len(expected), 'checked_maps': 348, 'map_payloads_complete_by_size': True}


def valid_context(request_path: Path, state_path: Path, worker_path: Path) -> tuple[dict, Path, Path, Path]:
    request = read_pinned_json(request_path, REQUEST_SHA256)
    state = read_pinned_json(state_path, STATE_SHA256)
    require(digest(canonical_file(worker_path)) == WORKER_SHA256, 'baseline worker source changed')
    require(request.get('phase') == 'I4.3B' and request.get('action') == 'colmap_mvs_dense_provider', 'unexpected request protocol')
    require(request.get('schema_version') == 1 and state.get('schema_version') == 1, 'unsupported evidence schema')
    require(state.get('stage') == 'stereo_fusion' and state.get('stereo_fusion_started') is True
            and state.get('stage_completed') is False and state.get('patch_match_started') is True,
            'not the recorded interrupted fusion stage')
    authority = request.get('authority') or {}
    require(authority.get('camera_pose_authority_owner') == 'pycolmap_sfm'
            and authority.get('single_authoritative_sfm_frame') is True
            and authority.get('metric_scale_claimed') is False
            and authority.get('sfm_rerun') is False, 'original SfM authority mismatch')
    config = request.get('config') or {}
    require(request.get('expected_registered_views') == 87
            and config.get('selected_view_count') == 87
            and config.get('all_registered_views_selected') is True
            and config.get('geometric_consistency') is True
            and config.get('max_image_size') == -1
            and config.get('source_neighbors') == 10
            and config.get('quality_thresholds_relaxed') is False
            and config.get('permanent_global_view_cap') is None, 'original scientific configuration changed')
    require(request.get('reconstruction_frame_id') == 'i6_7e2b2:pycolmap_sfm:' + SESSION_ID,
            'wrong camera reference frame')
    dense = canonical_dir(Path(str(request['output_workspace'])))
    run = canonical_dir(dense.parent.parent)
    session = canonical_dir(run.parent.parent)
    require(run.name == RUN_ID and run.parent.name == 'dense_runs' and session.name == SESSION_ID,
            'original dense run or sparse session identity differs')
    require(dense == run / 'provider_colmap_mvs' / 'dense', 'unexpected provider workspace')
    require(request_path == run / 'provider_colmap_mvs' / 'worker' / 'request.json', 'foreign original request')
    require(state_path == request_path.parent / 'execution_state.json', 'foreign state')
    require(Path(str(request['fused_ply'])) == dense / 'fused.ply', 'original fused target changed')
    require(not (dense / 'fused.ply').exists() and not (dense / 'fused.ply.vis').exists(),
            'original run already has fusion output; do not restart')
    require(not (run / 'final_dense_admission.json').exists()
            and not (request_path.parent / 'result.json').exists(),
            'original run has completion evidence; do not recover')
    sparse = canonical_dir(Path(str(request['input_sparse'])))
    images = canonical_dir(Path(str(request['input_images'])))
    require(sparse == session / 'sfm_workspace' / 'sparse' / '0'
            and images == session / 'sfm_workspace' / 'images', 'authoritative staged input paths differ')
    require(request.get('image_sha256') and set(request['image_sha256']) == set(request['image_names']),
            'registered-image hash list differs')
    require(request.get('input_sparse_sha256') and set(request['input_sparse_sha256']) ==
            {'cameras.bin', 'frames.bin', 'images.bin', 'points3D.bin', 'rigs.bin'},
            'authoritative sparse hash list differs')
    require(canonical_file(session / 'sfm_result.json').stat().st_size == 94421034, 'original SfM result size changed')
    require(digest(session / 'sfm_result.json') ==
            '65f6a2b75a89cf6442e8445cdb3df4c4d0ec74269de04aba2a95a42e1e31766d',
            'original authoritative SfM result SHA256 changed')
    for name, sha in request['input_sparse_sha256'].items():
        require(digest(canonical_file(sparse / name)) == sha, f'authoritative sparse checksum mismatch: {name}')
    for name, sha in request['image_sha256'].items():
        require(digest(canonical_file(images / name)) == sha, f'registered image checksum mismatch: {name}')
    return request, dense, run, session


def available_ram_gib() -> float:
    for line in Path('/proc/meminfo').read_text(encoding='utf-8').splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) / (1024 * 1024)
    return 0.0


def resident_gib(pid: int) -> float:
    try:
        for line in Path(f'/proc/{pid}/status').read_text(encoding='utf-8').splitlines():
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) / (1024 * 1024)
    except (FileNotFoundError, ProcessLookupError):
        return 0.0
    return 0.0


def write_new(path: Path, data: dict) -> None:
    raw = (json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf-8')
    with path.open('xb') as out:
        out.write(raw)
        out.flush()
        os.fsync(out.fileno())


def replace_status(path: Path, data: dict) -> None:
    tmp = path.with_name('.' + path.name + '.tmp.' + str(os.getpid()))
    with tmp.open('xb') as out:
        out.write((json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n').encode('utf-8'))
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp, path)


def run_fusion_child(args: argparse.Namespace, request: dict, dense: Path, run: Path, recovery: Path) -> int:
    launch = recovery / 'launch.json'
    require(launch.is_file() and not launch.is_symlink(), 'fusion subprocess has no supervised launch record')
    launch_data = json.loads(launch.read_text(encoding='utf-8'))
    require(launch_data.get('recovery_id') == args.recovery_id
            and launch_data.get('threads_requested') == args.threads
            and launch_data.get('original_request_sha256') == REQUEST_SHA256,
            'fusion subprocess launch record mismatch')
    # The isolated pycolmap interpreter is used ONLY here, after the pinned, read-only preflight.
    import pycolmap
    script_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(script_dir))
    import reconstruction_pycolmap_dense_provider_mvs_worker as baseline

    require(str(getattr(pycolmap, '__version__', '')) == str(request['runtime']['pycolmap_version']),
            'PyCOLMAP version changed since original run')
    require(digest(canonical_file(script_dir / 'reconstruction_pycolmap_dense_provider_mvs_worker.py')) == WORKER_SHA256,
            'baseline worker script changed during launch')
    workspace_check = baseline.validate_dense_sparse(
        pycolmap, Path(request['input_sparse']), dense / 'sparse', request['image_names'])
    require(workspace_check['pass'], f'undistorted authoritative camera/geometry changed: {workspace_check["checks"]}')
    patch_cfg = baseline.parse_patch_match_cfg(
        dense / 'stereo/patch-match.cfg', request['image_names'], int(request['config']['source_neighbors']))
    require(patch_cfg['pass'], f'PatchMatch registered source view contract changed: {patch_cfg["checks"]}')
    map_report = inspect_workspace(request, dense)
    write_new(recovery / 'pre_fusion_evidence.json', {
        'protocol': 'eia.i6_7f.fusion_only_recovery.v1', 'interrupted_run': RUN_ID,
        'session_id': SESSION_ID, 'original_request_sha256': REQUEST_SHA256,
        'original_state_sha256': STATE_SHA256, 'frame_id': request['reconstruction_frame_id'],
        'camera_and_sparse_checks': workspace_check, 'source_view_checks': patch_cfg,
        'stereo_maps': map_report, 'threads_requested': args.threads,
        'cache_disabled': True, 'max_image_size': -1,
        'fusion_scientific_thresholds_modified': False,
        'patch_match_rerun': False, 'sfm_rerun': False,
    })
    opts = pycolmap.StereoFusionOptions()
    opts.cache_size = float(request['config']['stereo_fusion_cache_gib'])
    opts.use_cache = False
    opts.num_threads = args.threads
    opts.max_image_size = int(request['config']['max_image_size'])
    output = recovery / 'fused.ply'
    require(not output.exists() and not (recovery / 'fused.ply.vis').exists(), 'recovery output already exists')
    print('FUSION_ONLY_STARTED=YES', flush=True)
    print(f'FUSION_THREADS_REQUESTED={args.threads}', flush=True)
    t0 = time.monotonic()
    fused = pycolmap.stereo_fusion(
        output_path=output, workspace_path=dense,
        workspace_format='COLMAP', input_type='geometric',
        options=opts, output_type='ply',
    )
    duration = time.monotonic() - t0
    original_count = baseline.ply_vertex_count(output)
    vis = recovery / 'fused.ply.vis'
    visibility_count = baseline.visibility_point_count(vis)
    returned = int(fused.num_points3D()) if hasattr(fused, 'num_points3D') else original_count
    require(original_count > 0 and original_count == returned == visibility_count,
            'recovered fused point count/visibility mismatch')
    # The expensive original input checksums were verified at launch. Repeat the
    # small sparse checks after fusion; report images as unchanged since baseline preflight.
    for name, sha in request['input_sparse_sha256'].items():
        require(digest(canonical_file(Path(request['input_sparse']) / name)) == sha,
                f'authoritative sparse geometry changed during recovery: {name}')
    report = {
        'protocol': 'eia.i6_7f.fusion_only_recovery.v1', 'result': 'fusion_output_verified',
        'session_id': SESSION_ID, 'interrupted_dense_run_id': RUN_ID,
        'recovery_id': args.recovery_id, 'original_request_sha256': REQUEST_SHA256,
        'original_state_sha256': STATE_SHA256, 'reconstruction_frame_id': request['reconstruction_frame_id'],
        'output_ply': str(output), 'output_ply_bytes': output.stat().st_size,
        'output_ply_sha256': digest(output), 'visibility_sha256': digest(vis),
        'output_vis': str(vis), 'point_count': original_count, 'duration_seconds': duration,
        'num_threads_requested': args.threads, 'use_cache': False,
        'scientific_parameters_changed': False, 'original_worker_state_untouched': True,
        'sfm_rerun': False, 'patch_match_rerun': False, 'final_dense_result_admitted': False,
        'persistent_dense_scene_created': False, 'metric_scale_claimed': False,
    }
    write_new(recovery / 'fusion_only_result.json', report)
    print('FUSION_ONLY_RECOVERY=PASS', flush=True)
    print(f'FUSION_POINTS={original_count}', flush=True)
    print(f'FUSION_PLY_SHA256={report["output_ply_sha256"]}', flush=True)
    print('FINAL_DENSE_ADMISSION=NO (separate independent admission required)', flush=True)
    return 0


def terminate_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=8)


def supervise(args: argparse.Namespace, request: dict, dense: Path, run: Path) -> int:
    require(str(UUID(args.recovery_id)) == args.recovery_id, 'noncanonical new recovery UUID')
    require(args.threads in (1, 2, 3, 4), 'explicit thread count must be 1–4')
    require(6.0 <= args.max_rss_gib <= 10.0 and 1.5 <= args.min_available_gib <= 5.0,
            'unsafe memory watchdog settings')
    require(shutil.disk_usage(dense).free >= 15 * GIB, 'less than 15 GiB disk available before recovery')
    require(available_ram_gib() >= 7.0, 'less than 7 GiB RAM available before recovery')
    # Conda's recorded interpreter may itself be a symlink inside its environment.
    py = Path(str(request['runtime']['python_executable']))
    require(py.is_absolute() and py.is_file() and py.resolve(strict=True).is_file(),
            'recorded isolated PyCOLMAP interpreter is missing')
    require(py == Path.home() / 'anaconda3/envs/eia-pycolmap-probe/bin/python3.11',
            'recovery must use recorded isolated PyCOLMAP interpreter')
    recoveries = run / 'fusion_recoveries'
    require(not recoveries.is_symlink(), 'foreign recovery root symlink')
    recoveries.mkdir(mode=0o700, exist_ok=True)
    recovery = recoveries / args.recovery_id
    recovery.mkdir(mode=0o700, exist_ok=False)
    write_new(recovery / 'launch.json', {
        'protocol': 'eia.i6_7f.fusion_only_recovery.v1', 'recovery_id': args.recovery_id,
        'interrupted_run_id': RUN_ID, 'session_id': SESSION_ID,
        'threads_requested': args.threads, 'max_rss_gib': args.max_rss_gib,
        'min_available_gib': args.min_available_gib,
        'original_request_sha256': REQUEST_SHA256, 'original_state_sha256': STATE_SHA256,
        'final_dense_result_admitted': False,
    })
    script = Path(__file__).resolve(strict=True)
    cmd = [str(py), str(script), '--internal-fusion', '--request', str(args.request),
           '--state', str(args.state), '--baseline-worker', str(args.baseline_worker),
           '--recovery-id', args.recovery_id, '--threads', str(args.threads)]
    child_log = recovery / 'fusion_worker.log'
    child_env = dict(os.environ, OMP_NUM_THREADS=str(args.threads))
    start = time.monotonic()
    with child_log.open('xb') as out:
        worker = subprocess.Popen(cmd, cwd=str(script.parents[1]), stdout=out, stderr=subprocess.STDOUT,
                                  env=child_env, start_new_session=True, stdin=subprocess.DEVNULL)
    print(f'RECOVERY_ID={args.recovery_id}', flush=True)
    print(f'RECOVERY_DIR={recovery}', flush=True)
    print(f'FUSION_WORKER_PID={worker.pid}', flush=True)
    print(f'FUSION_WORKER_LOG={child_log}', flush=True)
    reason = ''
    peak_rss = 0.0
    try:
        while worker.poll() is None:
            time.sleep(1.0)
            rss = resident_gib(worker.pid)
            peak_rss = max(peak_rss, rss)
            avail = available_ram_gib()
            disk_free = shutil.disk_usage(dense).free / GIB
            elapsed = (time.monotonic() - start) / 60
            if rss >= args.max_rss_gib or avail < args.min_available_gib or disk_free < 10.0:
                reason = (f'resource_guard: rss_gib={rss:.2f} available_ram_gib={avail:.2f} '
                          f'free_disk_gib={disk_free:.2f}')
                terminate_group(worker)
                break
            if elapsed >= args.timeout_minutes:
                reason = 'time_limit_exceeded'
                terminate_group(worker)
                break
            if int(elapsed * 60) % 30 == 0:
                print(f'HEARTBEAT elapsed_min={elapsed:.1f} child_rss_gib={rss:.2f} '
                      f'ram_available_gib={avail:.2f} free_disk_gib={disk_free:.2f}', flush=True)
    except KeyboardInterrupt:
        reason = 'user_interrupted_supervisor'
        terminate_group(worker)
    code = worker.wait()
    success = code == 0 and (recovery / 'fusion_only_result.json').is_file()
    if not success and not reason:
        reason = f'fusion_worker_exit_code_{code}'
    replace_status(recovery / 'supervisor_result.json', {
        'protocol': 'eia.i6_7f.fusion_only_recovery.v1',
        'status': 'fusion_output_verified' if success else 'failed_no_admission',
        'worker_exit_code': code, 'recovery_id': args.recovery_id,
        'elapsed_seconds': time.monotonic() - start, 'peak_child_rss_gib_observed': peak_rss,
        'reason': reason, 'original_worker_state_untouched': True, 'final_dense_result_admitted': False,
    })
    print('FUSION_ONLY_RECOVERY=' + ('PASS' if success else 'FAIL'), flush=True)
    print('RECOVERY_RESULT_DIR=' + str(recovery), flush=True)
    print('FINAL_DENSE_ADMISSION=NO', flush=True)
    if reason:
        print('FAILURE_REASON=' + reason, flush=True)
    return 0 if success else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--baseline-worker', type=Path, required=True)
    parser.add_argument('--recovery-id', required=True)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--max-rss-gib', type=float, default=8.0)
    parser.add_argument('--min-available-gib', type=float, default=3.0)
    parser.add_argument('--timeout-minutes', type=int, default=240)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--internal-fusion', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    require(str(UUID(args.recovery_id)) == args.recovery_id, 'noncanonical recovery id')
    args.request = args.request.expanduser().absolute()
    args.state = args.state.expanduser().absolute()
    args.baseline_worker = args.baseline_worker.expanduser().absolute()
    require(10 <= args.timeout_minutes <= 600, 'timeout out of bounds')
    request, dense, run, _session = valid_context(args.request, args.state, args.baseline_worker)
    # Structural map checks happen in BOTH the dry run and worker, not in the
    # monitoring parent. Never claim complete data integrity from counts alone.
    summary = inspect_workspace(request, dense)
    if args.check_only:
        require(not (run / 'fusion_recoveries' / args.recovery_id).exists(), 'recovery ID already exists')
        print('FUSION_ONLY_PREFLIGHT=PASS', flush=True)
        print('REGISTERED_VIEWS=' + str(summary['registered_views']), flush=True)
        print('STRUCTURALLY_COMPLETE_MAP_FILES=' + str(summary['checked_maps']), flush=True)
        print('MAP_CONTENT_HASH_VERIFIED=NO (original map content hashes were not recorded)', flush=True)
        print('PREVIOUS_PATCHMATCH_AND_SFM_RERUN=NO', flush=True)
        print('DENSE_PROVIDER_STARTED=NO', flush=True)
        return 0
    require(not args.internal_fusion or Path(sys.executable) == Path(request['runtime']['python_executable']),
            'worker interpreter differs from recorded PyCOLMAP interpreter')
    if args.internal_fusion:
        return run_fusion_child(args, request, dense, run, run / 'fusion_recoveries' / args.recovery_id)
    return supervise(args, request, dense, run)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print('FUSION_ONLY_RECOVERY=FAIL: ' + str(exc), file=sys.stderr, flush=True)
        raise SystemExit(2)

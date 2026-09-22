#!/usr/bin/env python3
"""TEST-ONLY I6.8F dense-worker telemetry fixture; NEVER invokes MVS.

A real, isolated production dense *wrapper* is executed with its costly MVS
executor replaced in this disposable process by a controlled test double.
Only test-sourced stage snapshots are emitted; failure is intentional and no
admission or final dense geometry can be produced. Not production evidence.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
from time import monotonic, sleep
from uuid import UUID

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_i6_7e2b2a_post_sparse_dense_worker as worker
from tests.i6_7c_helpers import make_snapshot

FIXTURE_FAILURE = 'I6.8F_TEST_FIXTURE_NO_MVS_NO_ADMISSION'


def _id(value: str) -> str:
    if str(UUID(value)) != value:
        raise ValueError('canonical UUID required')
    return value


def _fixture_root(value: Path) -> Path:
    # Deliberately narrow: only a caller-owned private temporary folder may
    # receive test sidecars. Historical sessions are never read or written.
    root = value.expanduser().resolve(strict=True)
    temp = Path(tempfile.gettempdir()).resolve(strict=True)
    if (root.parent != temp or not root.name.startswith('i68f_dense_fixture_')
            or root.is_symlink() or not root.is_dir()):
        raise ValueError('fixture requires a dedicated top-level i68f_dense_fixture_ temporary directory')
    if any(root.iterdir()):
        raise ValueError('fixture directory must be empty')
    return root


def _parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test-only-no-mvs', action='store_true', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--session-id', type=_id, required=True)
    parser.add_argument('--dense-run-id', type=_id, required=True)
    args = parser.parse_args(argv)
    args.output_dir = _fixture_root(args.output_dir)
    return args


def run(argv=None) -> int:
    args = _parse(argv)
    run_dir = args.output_dir / 'dense_runs' / args.dense_run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    # The real worker wrapper starts its own *production* passive sampler,
    # writes latest-only sidecars, and emits unaltered I6.7C snapshots.
    # The ONLY substitution is the computational MVS executor.
    def fake_mvs_executor(**kwargs):
        sink = kwargs['snapshot_sink']
        start = monotonic()
        for revision, stage in enumerate((
                'external_worker_launch', 'image_undistortion',
                'patch_match_stereo', 'stereo_fusion'), start=1):
            synthetic = replace(make_snapshot(
                revision=revision, session_id=args.session_id,
                dense_run_id=args.dense_run_id), stage_detail=stage)
            sink.publish(synthetic)
            sleep(0.35)
        # Give the unmodified five-second production sampler a chance to
        # observe this actual fixture-owned subprocess while it is alive.
        while monotonic() - start < 8.0:
            sleep(0.2)
        raise RuntimeError(FIXTURE_FAILURE)

    worker.execute_explicit_colmap_dense_after_sparse = fake_mvs_executor
    return worker.main([
        '--enable-explicit-colmap-mvs',
        '--provider-id', worker.I43B_PROVIDER_ID,
        '--offline-output-dir', str(args.output_dir),
        '--session-id', args.session_id,
        '--dense-run-id', args.dense_run_id,
        '--pycolmap-python', sys.executable,
        '--timeout-minutes', '0.3',
    ])


if __name__ == '__main__':
    raise SystemExit(run())

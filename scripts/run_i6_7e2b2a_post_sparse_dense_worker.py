#!/usr/bin/env python3
"""Explicit post-sparse COLMAP-MVS worker, for later supervised Qt QProcess wiring.

NOT a standalone full application workflow: production cancellation, UI process
ownership, and full-density GPU display require separate qualification. Never
rerun a failed run ID or treat provisional snapshots as final admission.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.reconstruction_dense_production_worker import execute_explicit_colmap_dense_after_sparse
from core.reconstruction.mvs_provider import I43B_PROVIDER_ID
from core.reconstruction.reconstruction_job_progress_observer import (
    JobProgressObserver, LatestJobProgressSnapshot,
)
from core.reconstruction.reconstruction_job_resource_sampler import JobResourceSampler
from core.reconstruction.reconstruction_progressive_dense_transport import ProgressiveDenseTransportLineSink

PROTOCOL = 'eia.reconstruction.dense.post_sparse.v1'


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--enable-explicit-colmap-mvs', action='store_true',
                        help='Required affirmative selection: Auto is not authorized.')
    parser.add_argument('--provider-id', required=True, help='Concrete COLMAP-MVS provider ID only.')
    parser.add_argument('--offline-output-dir', type=Path, required=True)
    parser.add_argument('--session-id', required=True)
    parser.add_argument('--pycolmap-python', type=Path, required=True)
    parser.add_argument('--dense-run-id', required=True)
    parser.add_argument('--timeout-minutes', type=float, default=120.0)
    parser.add_argument('--mvs-worker-script', type=Path,
                        default=PROJECT_ROOT / 'scripts/reconstruction_pycolmap_dense_provider_mvs_worker.py')
    args = parser.parse_args(argv)
    if not args.enable_explicit_colmap_mvs or args.provider_id != I43B_PROVIDER_ID:
        parser.error('This worker requires affirmative explicit COLMAP-MVS selection; Auto is not supported.')
    if not 0.0 < args.timeout_minutes <= 720.0:
        parser.error('timeout-minutes must be in (0, 720].')
    return args


def _emit(event: str, **fields: object) -> None:
    print(json.dumps(dict(protocol=PROTOCOL, event=event, **fields),
                     sort_keys=True, allow_nan=False, separators=(',', ':')), flush=True)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    original_sink = ProgressiveDenseTransportLineSink(lambda line: print(line, flush=True))
    # I6.8B: reuse actual provider snapshots; the new observational sidecar
    # is run-scoped, latest-only and not read by any admission/scene verifier.
    try:
        observer = JobProgressObserver('post_sparse_dense', args.session_id, args.dense_run_id)
        observation_sink = LatestJobProgressSnapshot(
            args.offline_output_dir / 'dense_runs' / args.dense_run_id / 'job_progress_observation.txt')
    except ValueError:
        # Invalid job arguments still reach the original production validator,
        # which owns the terminal error. Diagnostic telemetry is fail-soft.
        observer = None
        observation_sink = None

    sampler = None
    if observer is not None:
        try:
            sampler = JobResourceSampler(
                job_kind='post_sparse_dense', source_session_id=args.session_id,
                dense_run_id=args.dense_run_id, root_pid=os.getpid(),
                workspace=args.offline_output_dir / 'dense_runs' / args.dense_run_id,
            )
            sampler.start()
        except Exception:
            sampler = None  # resources are diagnostic; dense execution must continue

    class _ObservingSnapshotSink:
        def publish(self, snapshot):
            original_sink.publish(snapshot)  # preserve frozen I6.7C transport
            if observer is not None and observation_sink is not None:
                try:
                    observation = observer.from_dense_snapshot(snapshot)
                    observation_sink.publish(observation)
                    if sampler is not None and observation is not None:
                        sampler.set_stage(observation.stage)
                except Exception:
                    pass  # telemetry must never fail an MVS run or promote geometry

    sink = _ObservingSnapshotSink()
    frame_id = 'i6_7e2b2:pycolmap_sfm:' + args.session_id
    try:
        _emit('dense_starting', session_id=args.session_id, dense_run_id=args.dense_run_id,
              provider_id=I43B_PROVIDER_ID, provisional_only=True)
        if sampler is not None:
            try:
                sampler.set_stage('dense_preparing')
            except Exception:
                pass
        admitted = execute_explicit_colmap_dense_after_sparse(
            sparse_output_dir=args.offline_output_dir, sparse_session_id=args.session_id,
            pycolmap_python=args.pycolmap_python, mvs_worker_script=args.mvs_worker_script,
            reconstruction_frame_id=frame_id, dense_run_id=args.dense_run_id,
            snapshot_sink=sink, provider_id=args.provider_id,
            timeout_minutes=args.timeout_minutes)
    except Exception as exc:
        # Worker process reports *failure* without pretending partial artifacts
        # were admitted. A supervisor must preserve this run and its evidence.
        _emit('dense_failed', session_id=args.session_id, dense_run_id=args.dense_run_id,
              error_type=type(exc).__name__, detail=str(exc), final_dense_result_admitted=False)
        return 2
    finally:
        if sampler is not None:
            try:
                sampler.stop()
            except Exception:
                pass
    _emit('dense_completed', session_id=args.session_id, dense_run_id=args.dense_run_id,
          admission=admitted.as_dict(), final_dense_result_admitted=True,
          persistent_dense_scene_created=False, full_density_gpu_viewer_enabled=False,
          metric_scale_claimed=False, production_auto_activated=False)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

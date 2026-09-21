#!/usr/bin/env python3
"""One explicit, Qt-QProcess-supervised COLMAP-MVS attempt from persisted sparse evidence.

Recovery utility, NOT One Start/session rehydration in the GUI. Never reruns SfM.
Keeps this terminal alive until the child AND independent full-artifact verifier finish.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import signal
import sys
from time import monotonic

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

START_FREE_GIB = 35
STOP_FREE_GIB = 10


def _arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--offline-output-dir', type=Path, required=True)
    p.add_argument('--session-id', required=True)
    p.add_argument('--confirm-explicit-colmap-mvs', action='store_true',
                   help='Required affirmative opt-in; this starts a new expensive dense run.')
    p.add_argument('--timeout-minutes', type=float, default=360.0)
    args = p.parse_args(argv)
    if not args.confirm_explicit_colmap_mvs:
        p.error('Starting MVS requires --confirm-explicit-colmap-mvs.')
    if not 0 < args.timeout_minutes <= 720:
        p.error('--timeout-minutes must be in (0, 720].')
    return args


def _free_gib(path: Path) -> float:
    return shutil.disk_usage(path).free / (1024 ** 3)


def main(argv=None) -> int:
    args = _arguments(argv)
    # Keep the GUI closed: this is a standalone recovery job, not a GUI rehydration.
    from PyQt6.QtCore import QCoreApplication, QTimer
    from app.reconstruction_dense_job_controller import DenseJobState, ExplicitDenseJobController
    from app.reconstruction_dense_production_worker import load_admitted_sparse_for_dense
    from core.reconstruction.mvs_provider import I43B_PROVIDER_ID

    handoff, sfm = load_admitted_sparse_for_dense(args.offline_output_dir, args.session_id)
    root = Path(handoff.output_dir)
    original = json.loads((root / 'offline_result.json').read_text(encoding='utf-8'))
    pycolmap_python = Path(original['execution_boundary']['pycolmap_python'])
    if not pycolmap_python.is_file() or pycolmap_python.is_symlink():
        raise RuntimeError('Recorded PyCOLMAP interpreter is missing or is a symlink.')
    if sfm.statistics.registered_keyframes != 87:
        raise RuntimeError('South Building recovery expected exactly 87 admitted views; refusing foreign session.')
    if _free_gib(root) < START_FREE_GIB:
        raise RuntimeError(f'Insufficient space: require at least {START_FREE_GIB} GiB free to launch.')

    logs_root = root / 'dense_recovery_logs'
    if logs_root.is_symlink():
        raise RuntimeError('Recovery log directory must not be a symlink.')
    app = QCoreApplication.instance() or QCoreApplication(sys.argv[:1])
    controller = ExplicitDenseJobController()
    result = {'rc': 2, 'verified': None, 'cancel': False}
    started = monotonic()
    log = None

    def announce(message: str) -> None:
        line = datetime.now(timezone.utc).isoformat(timespec='seconds') + ' ' + message
        print(line, flush=True)
        if log is not None:
            log.write(line + '\n')
            log.flush()

    def on_state(state):
        announce('DENSE_STATE=' + state.value)

    def on_provider_log(message):
        # Preserve diagnostic log messages without printing megabytes of preview transport.
        if log is not None:
            log.write('WORKER_LOG=' + str(message)[:500] + '\n')
            log.flush()
        if any(token in str(message).lower() for token in ('error', 'warning', 'patchmatch', 'fusion', 'dense')):
            print('WORKER_LOG=' + str(message)[:250], flush=True)

    def on_admission(admission):
        result['verified'] = admission
        announce('INDEPENDENT_FINAL_ADMISSION=PASS')
        for key in ('sparse_session_id', 'dense_run_id', 'complete_geometry_vertex_count',
                    'manifest_sha256', 'complete_geometry_sha256'):
            if key in admission:
                announce(key.upper() + '=' + str(admission[key]))
        # Exact verified record is retained locally for the later full-viewer command.
        if log is not None:
            log.write('VERIFIED_ADMISSION_JSON=' + json.dumps(admission, sort_keys=True) + '\n')
            log.flush()

    def on_error(message):
        announce('DENSE_ERROR=' + str(message)[:700])

    def on_finished(info):
        state = info['state']
        complete = (state is DenseJobState.COMPLETED and result['verified'] is not None
                    and info.get('final_dense_result_admitted') is True)
        result['rc'] = 0 if complete else 2
        announce('DENSE_RECOVERY_RESULT=' + ('PASS' if complete else state.value.upper()))
        announce('DENSE_RECOVERY_RC=' + str(result['rc']))
        QTimer.singleShot(0, app.quit)

    controller.state_changed.connect(on_state)
    controller.log_received.connect(on_provider_log)
    controller.admission_ready.connect(on_admission)
    controller.error_occurred.connect(on_error)
    controller.run_finished.connect(on_finished)

    try:
        run_id = controller.start(admitted_sparse=handoff,
                                  pycolmap_python=pycolmap_python,
                                  provider_id=I43B_PROVIDER_ID,
                                  timeout_minutes=args.timeout_minutes)
        logs_root.mkdir(mode=0o700, exist_ok=True)
        log = (logs_root / (run_id + '.log')).open('x', encoding='utf-8', buffering=1)
        announce('SESSION_ID=' + handoff.session_id)
        announce('DENSE_RUN_ID=' + run_id)
        announce('REGISTERED_VIEWS=' + str(sfm.statistics.registered_keyframes))
        announce('RECOVERY_LOG=' + str(log.name))
        announce('COLMAP_MVS_EXPLICIT=YES')
        announce('ONE_START_RERUN=NO')
        announce('SUPERVISION=Qt QProcess isolated process group + independent verifier')
        announce('KEEP_THIS_TERMINAL_OPEN=YES; Ctrl+C requests supervised cancellation')

        # Python SIGINT is serviced even while the Qt event loop is running.
        def request_cancel(_signum, _frame):
            result['cancel'] = True
        signal.signal(signal.SIGINT, request_cancel)
        signal.signal(signal.SIGTERM, request_cancel)
        last_minute = -1

        def heartbeat():
            nonlocal last_minute
            if result['cancel']:
                result['cancel'] = False
                announce('CANCELLATION_REQUESTED=YES')
                controller.cancel()
            if not controller.is_active:
                return
            elapsed_minute = int((monotonic() - started) // 60)
            if elapsed_minute != last_minute:
                last_minute = elapsed_minute
                free = _free_gib(root)
                announce(f'HEARTBEAT elapsed_min={elapsed_minute} state={controller.state.value} free_gib={free:.2f}')
                if free < STOP_FREE_GIB:
                    announce('LOW_DISK_CANCELLATION=YES')
                    controller.cancel()

        timer = QTimer()
        timer.setInterval(1000)
        timer.timeout.connect(heartbeat)
        timer.start()
        heartbeat()
        app.exec()
        return result['rc']
    finally:
        controller.shutdown()
        if log is not None:
            log.close()


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print('DENSE_RECOVERY_START_ERROR=' + str(exc)[:700], file=sys.stderr, flush=True)
        raise SystemExit(2)

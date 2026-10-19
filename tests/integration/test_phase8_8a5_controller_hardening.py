from __future__ import annotations

import os
from pathlib import Path
import sys
import textwrap

import pytest

from app.application_controller import ApplicationController
from app.reconstruction_controller import (
    OfflineReconstructionConfiguration,
    OfflineReconstructionRunState,
    OfflineReconstructionSourceKind,
    ReconstructionController,
    ReconstructionSceneAvailability,
)
from app.widgets.original_view import OriginalSourceView
from app.widgets.source_bar import SourceBar


@pytest.fixture
def application_controller(qtbot, monkeypatch):
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    source_bar = SourceBar()
    original_view = OriginalSourceView()
    qtbot.addWidget(source_bar)
    qtbot.addWidget(original_view)
    controller = ApplicationController(source_bar, original_view)
    yield controller
    controller.shutdown()


def _worker(path: Path) -> None:
    path.write_text(
        textwrap.dedent(
            r'''
            import argparse, json, os, signal, subprocess, sys, time
            from pathlib import Path

            parser = argparse.ArgumentParser(add_help=False)
            parser.add_argument('--session-id', required=True)
            parser.add_argument('--output-dir', required=True)
            args, _ = parser.parse_known_args()
            out = Path(args.output_dir)
            out.mkdir(parents=True, exist_ok=True)
            mode = out.name
            if hasattr(os, 'setpgrp'):
                try:
                    os.setpgrp()
                except OSError:
                    pass

            def write_result(session_id, passed=True):
                payload = {
                    'phase': '8.8A5',
                    'pass': bool(passed),
                    'status': 'completed' if passed else 'failed',
                    'session_id': session_id,
                    'camera_pose_available': bool(passed),
                    'canonical_geometry_authoritative': False,
                    'persistent_scene_ready': False,
                    'certified_metrology': False,
                    'reconstruction_ui_enabled': False,
                    'tests_fixture_used': False,
                }
                (out / 'offline_result.json').write_text(json.dumps(payload), encoding='utf-8')

            if 'mismatch' in mode:
                write_result('00000000-0000-4000-8000-00000000dead', True)
                raise SystemExit(0)
            if 'crash' in mode:
                print('synthetic worker crash', flush=True)
                raise SystemExit(7)
            if 'cancel' in mode:
                def on_term(_sig, _frame):
                    write_result(args.session_id, True)
                    raise SystemExit(0)
                signal.signal(signal.SIGTERM, on_term)
                print('cancel worker ready', flush=True)
                while True:
                    time.sleep(0.05)
            if 'stubborn' in mode:
                child = subprocess.Popen([
                    sys.executable,
                    '-c',
                    'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)',
                ])
                (out / 'child_pid.txt').write_text(str(child.pid), encoding='utf-8')
                signal.signal(signal.SIGTERM, signal.SIG_IGN)
                print('stubborn worker ready', flush=True)
                while True:
                    time.sleep(0.05)

            write_result(args.session_id, True)
            print(json.dumps({
                'protocol': 'depth3d.reconstruction.offline.v1',
                'event': 'completed',
                'stage': 'camera_pose',
                'message': 'mock completed',
                'progress_fraction': 1.0,
            }), flush=True)
            '''
        ).strip()
        + "\n",
        encoding="utf-8",
    )


def _configuration(tmp_path: Path, name: str) -> OfflineReconstructionConfiguration:
    images = tmp_path / "images"
    images.mkdir(exist_ok=True)
    return OfflineReconstructionConfiguration(
        source_kind=OfflineReconstructionSourceKind.IMAGE_DIRECTORY,
        inputs=(images,),
        output_dir=(tmp_path / name).resolve(),
        sfm_python=Path(sys.executable),
        max_keyframes=3,
        use_gpu=False,
    )


def _wait_terminal(controller: ReconstructionController, qtbot) -> None:
    qtbot.waitUntil(
        lambda: controller.offline_state in {
            OfflineReconstructionRunState.COMPLETED,
            OfflineReconstructionRunState.FAILED,
            OfflineReconstructionRunState.CANCELLED,
        },
        timeout=5_000,
    )
    qtbot.wait(25)


def test_failed_cancelled_and_stale_runs_do_not_poison_next_success(
    application_controller, tmp_path: Path, qtbot
) -> None:
    worker = tmp_path / "hardening_worker.py"
    _worker(worker)
    controller = ReconstructionController(
        application_controller,
        parent=application_controller,
        offline_worker_script=worker,
    )
    results: list[dict] = []
    errors: list[str] = []
    finished: list[dict] = []
    controller.offline_result_ready.connect(lambda payload, _out: results.append(payload))
    controller.error_occurred.connect(errors.append)
    controller.offline_run_finished.connect(finished.append)

    try:
        mismatch_sid = "00000000-0000-4000-8000-0000000088a5"
        controller.start_offline(_configuration(tmp_path, "mismatch"), session_id=mismatch_sid)
        _wait_terminal(controller, qtbot)
        assert controller.offline_state is OfflineReconstructionRunState.FAILED
        assert results == []
        assert controller.persistence_controller.offline_result is None
        assert any("session_id does not match" in item for item in errors)
        assert finished[-1]["result_accepted"] is False

        crash_error_index = len(errors)
        controller.start_offline(_configuration(tmp_path, "crash"))
        _wait_terminal(controller, qtbot)
        assert controller.offline_state is OfflineReconstructionRunState.FAILED
        assert results == []
        assert controller.persistence_controller.offline_result is None
        assert any(
            "did not produce a readable offline_result.json" in item
            for item in errors[crash_error_index:]
        )

        controller.start_offline(_configuration(tmp_path, "cancel"))
        qtbot.waitUntil(lambda: controller.offline_state is OfflineReconstructionRunState.RUNNING, timeout=2000)
        assert controller.cancel_offline() is True
        _wait_terminal(controller, qtbot)
        assert controller.offline_state is OfflineReconstructionRunState.CANCELLED
        assert results == []
        assert controller.persistence_controller.offline_result is None
        assert finished[-1]["result_accepted"] is False

        good_sid = controller.start_offline(_configuration(tmp_path, "good"))
        _wait_terminal(controller, qtbot)
        assert controller.offline_state is OfflineReconstructionRunState.COMPLETED
        assert len(results) == 1
        assert results[0]["session_id"] == good_sid
        assert controller.persistence_controller.offline_result is not None
        assert controller.persistence_controller.offline_result["session_id"] == good_sid
        assert controller.scene_availability is ReconstructionSceneAvailability.PARTIAL
        assert finished[-1]["result_accepted"] is True
    finally:
        controller.shutdown()
        qtbot.wait(25)


@pytest.mark.skipif(not hasattr(os, "killpg"), reason="process-group hardening is POSIX-only")
def test_forced_cancel_kills_worker_process_group_without_orphan(
    application_controller, tmp_path: Path, qtbot
) -> None:
    worker = tmp_path / "hardening_worker.py"
    _worker(worker)
    controller = ReconstructionController(
        application_controller,
        parent=application_controller,
        offline_worker_script=worker,
    )
    # Production grace is 8 s. Shorten only the test timer so the same kill-group
    # path is exercised without making the suite slow.
    controller.offline_controller._kill_timer.setInterval(150)
    output = tmp_path / "stubborn"

    try:
        controller.start_offline(_configuration(tmp_path, "stubborn"))
        child_pid_path = output / "child_pid.txt"
        qtbot.waitUntil(child_pid_path.is_file, timeout=2_000)
        child_pid = int(child_pid_path.read_text(encoding="utf-8"))
        assert child_pid > 1
        assert controller.cancel_offline() is True
        _wait_terminal(controller, qtbot)
        assert controller.offline_state is OfflineReconstructionRunState.CANCELLED
        assert controller.offline_is_running is False

        def child_gone_or_zombie() -> bool:
            stat = Path(f"/proc/{child_pid}/stat")
            if not stat.exists():
                return True
            try:
                fields = stat.read_text(encoding="utf-8").split()
            except OSError:
                return True
            return len(fields) > 2 and fields[2] == "Z"

        qtbot.waitUntil(child_gone_or_zombie, timeout=2_000)
    finally:
        controller.shutdown()
        qtbot.wait(25)

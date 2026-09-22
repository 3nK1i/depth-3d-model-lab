"""Bounded actual dense-wrapper/real resource sampler/Qt main-page path.

NOT a real MVS, PatchMatch, fusion, dense point cloud, or final-admission test.
Stage snapshots are intentionally test-generated in a separate owned process;
no historical evidence or user reconstruction is accessed.
"""
from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from time import monotonic
from uuid import uuid4

import pytest
pytest.importorskip('PyQt6')

from app.main_window import MainWindow
from app.reconstruction_dense_job_controller import DenseJobState
from core.reconstruction.reconstruction_job_telemetry_contract import parse_job_telemetry_line

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / 'scripts/run_i6_8f_dense_telemetry_fixture.py'
FAILURE = 'I6.8F_TEST_FIXTURE_NO_MVS_NO_ADMISSION'


def _stop_owned_group(child: subprocess.Popen) -> None:
    if child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        child.wait(timeout=3)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait(timeout=3)


@pytest.fixture
def main_page(qtbot, monkeypatch):
    monkeypatch.setattr('app.application_controller.CameraEnumerator.start', lambda self: None)
    window = MainWindow()
    qtbot.addWidget(window)
    page = window._reconstruction_page
    window._mode_stack.setCurrentWidget(page)
    window.show()
    qtbot.waitUntil(lambda: page.isVisible(), timeout=4000)
    try:
        yield page, window._reconstruction_controller
    finally:
        window.close()


def test_isolated_dense_wrapper_real_resource_sample_reaches_actual_main_page(main_page, qtbot):
    page, controller = main_page
    sid, rid = str(uuid4()), str(uuid4())
    dense = controller.dense_controller
    # Set only the UI/relay identity; NEVER call start(), MVS, or admission.
    dense._session_id, dense._dense_run_id = sid, rid
    controller.dense_state_changed.emit(DenseJobState.STARTING)
    assert page._job_telemetry_panel_state.identity == ('post_sparse_dense', sid, rid)
    relay = dense._telemetry_relay

    with tempfile.TemporaryDirectory(prefix='i68f_dense_fixture_') as temporary:
        root = Path(temporary)
        run_dir = root / 'dense_runs' / rid
        relay.begin('post_sparse_dense', sid, rid, run_dir)
        # Log is outside the fixture output directory; child preflight can
        # still enforce a new, empty, disposable output root.
        log = tempfile.TemporaryFile(mode='w+t', encoding='utf-8')
        child = subprocess.Popen(
            [sys.executable, str(FIXTURE), '--test-only-no-mvs',
             '--output-dir', str(root), '--session-id', sid, '--dense-run-id', rid],
            cwd=str(ROOT), stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        began = monotonic()
        try:
            page._job_telemetry_details_button.click()
            assert page._job_telemetry_dialog.isVisible()
            # Qt's *real* 1500ms poll consumes actual sidecars emitted by the
            # production worker wrapper and resource sampler in the subprocess.
            qtbot.waitUntil(
                lambda: ('Stereo fusion' in page._job_telemetry_stage.text()
                         and page._job_telemetry_panel_state.resources is not None
                         and 'GiB' in page._job_telemetry_process.text()),
                timeout=15000,
            )
            assert child.poll() is None, 'worker must be active at sample reception'
            progress = page._job_telemetry_panel_state.progress
            resource = page._job_telemetry_panel_state.resources
            assert progress is not None and progress.stage == 'dense_fusion'
            assert progress.producer == 'dense_supervisor'
            assert progress.counter is None  # no fabricated fusion percentage
            assert resource is not None and resource.producer == 'resource_sampler'
            assert (resource.source_session_id, resource.dense_run_id) == (sid, rid)
            assert any(r.name == 'process_tree_rss_bytes' and r.availability == 'observed'
                       for r in resource.resource_readings)
            assert 'Host:' in page._job_telemetry_host.text()
            assert ('GPU 0 device-wide' in page._job_telemetry_gpu.text()
                    or 'unavailable' in page._job_telemetry_gpu.text())
            qtbot.waitUntil(lambda: child.poll() is not None, timeout=13000)
            log.seek(0)
            worker_output = log.read(100_000)
            assert child.returncode == 2, worker_output[-1800:]
            assert FAILURE in worker_output and 'dense_failed' in worker_output
            assert 'dense_completed' not in worker_output
            assert monotonic() - began < 24
            assert (run_dir / 'job_progress_observation.txt').is_file()
            assert (run_dir / 'job_resource_observation.txt').is_file()
            observed = parse_job_telemetry_line(
                (run_dir / 'job_resource_observation.txt').read_text().strip())
            assert observed is not None and observed.job_kind == 'post_sparse_dense'
            assert observed.dense_run_id == rid
            assert not (run_dir / 'final_dense_admission.json').exists()
            assert not (root / 'offline_result.json').exists()
            assert dense._verified_admission is None
            # Only the normal owning controller's terminal signal clears the
            # presentation. Closing a fixture must not assert job success.
            relay.close()
            controller.dense_state_changed.emit(DenseJobState.FAILED)
            assert 'Live samples ended' in page._job_telemetry_stage.text()
            assert 'GiB' not in page._job_telemetry_process.text()
            assert 'Stereo fusion' not in page._job_telemetry_stage.text()
            assert not relay.history or len(relay.history) <= 32
            page._job_telemetry_dialog.hide()
        finally:
            relay.close()
            _stop_owned_group(child)
            log.close()

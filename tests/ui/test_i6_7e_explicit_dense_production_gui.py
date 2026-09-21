"""Workstation Qt UI integration; fake nested worker, never run real MVS here."""
from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest

from app.main_window import MainWindow
from app.reconstruction_dense_job_controller import DenseJobState, ExplicitDenseJobController
from tests.unit.test_i6_7e2a_sparse_dense_handoff_gate import fixture_state
from tests.i6_7c_helpers import make_snapshot


@pytest.fixture
def window(qtbot, monkeypatch):
    monkeypatch.setattr('app.application_controller.CameraEnumerator.start', lambda self: None)
    win = MainWindow()
    qtbot.addWidget(win)
    yield win
    win.close()


def test_dense_button_explicit_only_and_no_auto_or_full_viewer_claim(window, monkeypatch, qtbot):
    page = window._reconstruction_page
    controller = window._reconstruction_controller
    assert not page._dense_build_button.isEnabled()
    assert not page._dense_cancel_button.isEnabled()
    launches = []
    monkeypatch.setattr(type(controller), 'dense_ready_for_explicit_start',
                        property(lambda self: True))
    monkeypatch.setattr(controller, 'start_explicit_post_sparse_dense',
                        lambda **kw: launches.append(kw) or 'explicit-run')
    monkeypatch.setattr(page, '_release_appearance_resources_for_heavy_handoff', lambda purpose: None)
    page._refresh_runtime_controls()
    assert page._dense_build_button.isEnabled()
    page._dense_build_button.click()
    assert launches == [{}]  # explicit default provider; never Auto
    page.set_explicit_dense_job_state(DenseJobState.STARTING)
    snapshot = make_snapshot(revision=1, geometry_points=9)
    page._on_explicit_dense_preview(snapshot)
    qtbot.waitUntil(lambda: page._reconstruction_canvas.progressive_dense_revision == 1, timeout=3000)
    assert page._reconstruction_canvas.progressive_dense_display_count == 9
    assert not page._reconstruction_canvas.dense_rgb_points_loaded
    page._on_explicit_dense_admission({'complete_geometry_vertex_count': 1686160})
    page.set_explicit_dense_job_state(DenseJobState.COMPLETED)
    assert '1,686,160' in page._dense_job_value.text()
    assert 'full viewer pending' in page._dense_job_value.text()
    assert not page._reconstruction_canvas.dense_rgb_points_loaded


@pytest.mark.skipif(sys.platform != 'linux', reason='Linux setsid process group contract')
def test_qprocess_cancellation_stops_nested_provider_without_admission(qtbot, tmp_path):
    state, _, root, _ = fixture_state(tmp_path)
    handoff = state['admitted_sparse_handoff']
    worker_script = tmp_path / 'fake_dense_worker.py'
    worker_script.write_text('''\
import argparse, json, pathlib, subprocess, time
p=argparse.ArgumentParser()
p.add_argument('--session-id'); p.add_argument('--dense-run-id'); p.add_argument('--offline-output-dir')
a,_=p.parse_known_args()
child=subprocess.Popen(['sleep','120'])
pathlib.Path(a.offline_output_dir, 'nested_dense_child.pid').write_text(str(child.pid))
print(json.dumps(dict(protocol='eia.reconstruction.dense.post_sparse.v1', event='dense_starting',
                     session_id=a.session_id, dense_run_id=a.dense_run_id)),flush=True)
time.sleep(120)
''', encoding='utf-8')
    controller = ExplicitDenseJobController(worker_script=worker_script)
    completions = []
    admissions = []
    controller.run_finished.connect(completions.append)
    controller.admission_ready.connect(admissions.append)
    controller.start(admitted_sparse=handoff, pycolmap_python=Path(sys.executable))
    pid_file = root / 'nested_dense_child.pid'
    qtbot.waitUntil(lambda: pid_file.is_file(), timeout=5000)
    descendant = int(pid_file.read_text())
    assert controller.state is DenseJobState.RUNNING
    assert controller.cancel() is True
    qtbot.waitUntil(lambda: len(completions) == 1, timeout=12000)
    assert completions[0]['state'] is DenseJobState.CANCELLED
    assert not admissions
    qtbot.waitUntil(lambda: not Path(f'/proc/{descendant}').exists() or
                    (Path(f'/proc/{descendant}/stat').read_text().split()[2] == 'Z'),
                    timeout=4000)
    controller.shutdown()

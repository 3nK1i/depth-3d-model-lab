"""Qt headless double verifies dense activity is separate from sparse 100% progress."""
from __future__ import annotations

import numpy as np
import pytest

from app.main_window import MainWindow
from app.reconstruction_dense_job_controller import DenseJobState
from app.reconstruction_full_dense_viewer import FullDenseViewerFrame


@pytest.fixture
def page(qtbot, monkeypatch):
    monkeypatch.setattr('app.application_controller.CameraEnumerator.start', lambda self: None)
    win = MainWindow()
    qtbot.addWidget(win)
    yield win._reconstruction_page
    win.close()


def test_distinct_indeterminate_dense_bar_not_sparse_percent(page):
    page.set_explicit_dense_job_state(DenseJobState.STARTING)
    assert page._dense_progress_bar.isVisible() or not page.isVisible()
    assert page._dense_progress_bar.minimum() == page._dense_progress_bar.maximum() == 0
    assert page._progress_bar.maximum() == 100  # existing sparse bar remains independent
    page.set_explicit_dense_job_state(DenseJobState.RUNNING)
    assert 'Provider execution' in page._dense_progress_label.text()
    page.set_explicit_dense_job_state(DenseJobState.VERIFYING)
    assert 'Verifying' in page._dense_progress_label.text()
    page.set_explicit_dense_job_state(DenseJobState.COMPLETED)
    assert page._dense_progress_bar.value() == 100
    page.set_explicit_dense_job_state(DenseJobState.FAILED)
    assert page._dense_progress_bar.value() == 0


def test_full_display_is_separate_from_provisional_and_persistent(page):
    page._dense_admission = dict(sparse_session_id='session-a', dense_run_id='run-1',
                                 final_dense_result_admitted=True, manifest_path='/stub')
    page.set_explicit_dense_job_state(DenseJobState.COMPLETED)
    assert page._dense_full_view_button.isEnabled()
    n = 4107
    coords = np.zeros((n, 3), dtype=np.float32)
    rgba = np.ones((n, 4), dtype=np.float32)
    frame = FullDenseViewerFrame(session_id='session-a', dense_run_id='run-1',
        reconstruction_frame_id='i6_7e2b2:pycolmap_sfm:session-a', artifact_sha256='b'*64,
        points_viewer=coords, colors_rgba=rgba, fit_min=(0., 0., 0.),
        fit_max=(1., 1., 1.), preparation_ms=2.0)
    page._on_full_dense_frame(frame, page._dense_load_generation)
    assert page._reconstruction_canvas.admitted_full_dense_display_count == n
    assert page._reconstruction_canvas.admitted_full_dense_gpu_submissions == 1
    assert '4,107 / 4,107' in page._dense_job_value.text()
    assert not page._reconstruction_canvas.dense_rgb_points_loaded
    page._unload_full_dense_view()
    assert page._reconstruction_canvas.admitted_full_dense_display_count == 0


def test_stale_full_geometry_cannot_cross_run(page):
    page._dense_admission = dict(sparse_session_id='session-b', dense_run_id='run-2')
    page.set_explicit_dense_job_state(DenseJobState.COMPLETED)
    frame = FullDenseViewerFrame(session_id='session-a', dense_run_id='run-1',
        reconstruction_frame_id='frame', artifact_sha256='b'*64,
        points_viewer=np.zeros((1, 3), np.float32),
        colors_rgba=np.ones((1, 4), np.float32),
        fit_min=(0.,0.,0.), fit_max=(1.,1.,1.), preparation_ms=0.)
    page._on_full_dense_frame(frame, page._dense_load_generation)
    assert page._reconstruction_canvas.admitted_full_dense_display_count == 0


def test_real_demo_admission_supplies_verified_manifest_vertex_count(tmp_path):
    """Real demo must supply the same count the full-viewer gate checks."""
    import hashlib
    import json
    from scripts.run_i6_7f_full_dense_viewer_demo import build_demo_admission

    run = tmp_path / 'one-start' / 'dense_runs' / 'run-1'
    run.mkdir(parents=True)
    manifest = run / 'final_dense_admission.json'
    manifest.write_text(json.dumps({'complete_geometry_vertex_count': 163827}), encoding='utf-8')
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    admission = build_demo_admission(
        offline_output_dir=tmp_path / 'one-start', session_id='session-a',
        dense_run_id='run-1', manifest_sha256=manifest_sha,
        geometry_sha256='a' * 64)
    assert admission['complete_geometry_vertex_count'] == 163827
    assert admission['manifest_sha256'] == manifest_sha
    assert admission['sparse_session_id'] == 'session-a'


def test_real_demo_admission_rejects_manifest_tampering(tmp_path):
    import hashlib
    import json
    from scripts.run_i6_7f_full_dense_viewer_demo import build_demo_admission

    run = tmp_path / 'one-start' / 'dense_runs' / 'run-1'
    run.mkdir(parents=True)
    manifest = run / 'final_dense_admission.json'
    manifest.write_text(json.dumps({'complete_geometry_vertex_count': 163827}), encoding='utf-8')
    original_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    manifest.write_text(json.dumps({'complete_geometry_vertex_count': 163828}), encoding='utf-8')
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        build_demo_admission(
            offline_output_dir=tmp_path / 'one-start', session_id='session-a',
            dense_run_id='run-1', manifest_sha256=original_sha,
            geometry_sha256='a' * 64)


def test_real_demo_admission_rejects_missing_count(tmp_path):
    import hashlib
    import json
    from scripts.run_i6_7f_full_dense_viewer_demo import build_demo_admission

    run = tmp_path / 'one-start' / 'dense_runs' / 'run-1'
    run.mkdir(parents=True)
    manifest = run / 'final_dense_admission.json'
    manifest.write_text(json.dumps({'schema_version': 1}), encoding='utf-8')
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='vertex count'):
        build_demo_admission(
            offline_output_dir=tmp_path / 'one-start', session_id='session-a',
            dense_run_id='run-1', manifest_sha256=digest,
            geometry_sha256='a' * 64)

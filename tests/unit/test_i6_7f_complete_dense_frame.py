"""I6.7F: full RGB geometry must be non-sampled and bound to its admitted bytes."""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app import reconstruction_full_dense_viewer as full
from tests.unit.test_i6_7e1_dense_artifact_preflight import binary_ply


def _valid(tmp_path, monkeypatch, count=4107):
    run = tmp_path / 'one-start' / 'dense_runs' / 'run-1'
    dense = run / 'provider_colmap_mvs' / 'dense'
    dense.mkdir(parents=True)
    ply = dense / 'fused.ply'
    binary_ply(ply, count)
    digest = sha256(ply.read_bytes()).hexdigest()
    manifest = run / 'final_dense_admission.json'
    manifest.write_text('{}', encoding='utf-8')
    admission = dict(sparse_session_id='session-a', dense_run_id='run-1',
                     manifest_path=str(manifest), manifest_sha256='a' * 64,
                     complete_geometry_sha256=digest, complete_geometry_vertex_count=count,
                     final_dense_result_admitted=True)
    def verified(**kwargs):
        assert kwargs['sparse_session_id'] == 'session-a'
        assert kwargs['dense_run_id'] == 'run-1'
        return SimpleNamespace(vertex_count=count, artifact_sha256=digest,
                               manifest_sha256='a' * 64, geometry_path=str(ply),
                               reconstruction_frame_id='i6_7e2b2:pycolmap_sfm:session-a')
    monkeypatch.setattr(full, 'verify_dense_completion', verified)
    return admission, ply


def test_full_frame_contains_all_vertices_beyond_preview_budget(tmp_path, monkeypatch):
    admission, _ = _valid(tmp_path, monkeypatch)
    frame = full.prepare_full_dense_viewer_frame(admission, expected_session_id='session-a')
    assert frame.point_count == 4107
    assert frame.points_viewer.shape == (4107, 3)
    assert frame.colors_rgba.shape == (4107, 4)
    assert frame.points_viewer.dtype == frame.colors_rgba.dtype == np.float32
    assert frame.points_viewer.flags.c_contiguous and frame.colors_rgba.flags.c_contiguous
    assert np.allclose(frame.points_viewer[4106], (4106.0, -4106.0, -0.5))
    assert np.allclose(frame.colors_rgba[4106], (1.0, 10/255, 20/255, 1.0))
    assert not frame.provisional and not frame.persistent_scene and not frame.metric_scale_claimed


def test_reject_foreign_sparse_session(tmp_path, monkeypatch):
    admission, _ = _valid(tmp_path, monkeypatch)
    with pytest.raises(Exception, match='current sparse session'):
        full.prepare_full_dense_viewer_frame(admission, expected_session_id='foreign')


def test_reject_modified_ply_after_verifier(tmp_path, monkeypatch):
    admission, ply = _valid(tmp_path, monkeypatch, count=17)
    with ply.open('ab') as handle:
        handle.write(b'untrusted-trailing-change')
    with pytest.raises(Exception, match='artifact bytes changed'):
        full.prepare_full_dense_viewer_frame(admission, expected_session_id='session-a')


def test_reject_memory_overflow_without_silent_sampling(tmp_path, monkeypatch):
    admission, _ = _valid(tmp_path, monkeypatch, count=7)
    count = full.MAX_STAGING_BYTES // 28 + 1
    admission['complete_geometry_vertex_count'] = count
    original = full.verify_dense_completion
    def oversized(**kwargs):
        result = original(**kwargs)
        result.vertex_count = count
        return result
    monkeypatch.setattr(full, 'verify_dense_completion', oversized)
    with pytest.raises(Exception, match='memory budget'):
        full.prepare_full_dense_viewer_frame(admission, expected_session_id='session-a')


def test_reject_manifest_identity_mismatch_before_disk_scan(tmp_path, monkeypatch):
    admission, _ = _valid(tmp_path, monkeypatch, count=7)
    admission['manifest_path'] = str(tmp_path / 'outside.json')
    with pytest.raises(Exception, match='admission location'):
        full.prepare_full_dense_viewer_frame(admission, expected_session_id='session-a')

"""I6.7E full result revalidation: no Qt, fake MVS only, no real reconstruction."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from app.reconstruction_dense_terminal_verification import verify_dense_completion
from app.reconstruction_dense_production_worker import execute_explicit_colmap_dense_after_sparse
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.mvs_provider import MvsProductionProviderAdapter
from tests.i4_3b_helpers import FakeWorkerRunner, good_environment, good_resources
from tests.integration.test_i6_7e2b2a_isolated_dense_worker import setup, _PreviewSink


def complete_fake_run(tmp_path):
    state, _, root, run_id, py, worker = setup(tmp_path)
    admission = execute_explicit_colmap_dense_after_sparse(
        sparse_output_dir=root, sparse_session_id=state['source_sfm_result'].session_id,
        pycolmap_python=py, mvs_worker_script=worker, dense_run_id=run_id,
        reconstruction_frame_id='i6_7e2b2:pycolmap_sfm:' + state['source_sfm_result'].session_id,
        snapshot_sink=_PreviewSink(), adapter_factory=lambda context, sink: MvsProductionProviderAdapter(
            context, progress_sink=sink, environment_probe=good_environment(str(py)),
            resource_probe=good_resources(context.output_dir),
            worker_runner=FakeWorkerRunner(ply_points=120, vis_points=120)))
    args = dict(offline_output_dir=root, sparse_session_id=state['source_sfm_result'].session_id,
                dense_run_id=run_id, expected_manifest_sha256=admission.manifest_sha256,
                expected_geometry_sha256=admission.artifact_sha256)
    return admission, args


def test_verified_complete_artifact_from_only_one_sparse_session(tmp_path):
    admission, args = complete_fake_run(tmp_path)
    result = verify_dense_completion(**args)
    assert result == admission
    assert result.vertex_count == 120


def test_rejects_foreign_run_or_wrong_worker_claim(tmp_path):
    _, args = complete_fake_run(tmp_path)
    for field, value in (('dense_run_id', '00000000-0000-4000-8000-000000000001'),
                         ('expected_manifest_sha256', '0'*64),
                         ('expected_geometry_sha256', '0'*64)):
        corrupted = dict(args, **{field:value})
        with pytest.raises((ReconstructionValidationError, ValueError, OSError)):
            verify_dense_completion(**corrupted)


def test_rejects_geometry_or_manifest_tampering(tmp_path):
    admission, args = complete_fake_run(tmp_path)
    ply = Path(admission.geometry_path)
    ply.write_bytes(ply.read_bytes() + b'foreign')
    with pytest.raises((ReconstructionValidationError, ValueError, OSError)):
        verify_dense_completion(**args)
    admission, args = complete_fake_run(tmp_path / 'new')
    path = Path(admission.manifest_path)
    original = json.loads(path.read_text())
    original['metric_scale_claimed'] = True
    path.write_text(json.dumps(original))
    with pytest.raises((ReconstructionValidationError, ValueError, OSError)):
        verify_dense_completion(**args)


def test_isolated_verifier_cli_fails_closed_before_qt_receives_admission(tmp_path):
    admission, args = complete_fake_run(tmp_path)
    script = Path(__file__).resolve().parents[2] / 'scripts/run_i6_7e_dense_completion_verify.py'
    command = [sys.executable, str(script), '--offline-output-dir', str(args['offline_output_dir']),
               '--session-id', args['sparse_session_id'], '--dense-run-id', args['dense_run_id'],
               '--manifest-sha256', args['expected_manifest_sha256'],
               '--geometry-sha256', args['expected_geometry_sha256']]
    proc = subprocess.run(command, cwd=script.parents[1], text=True,
                          capture_output=True, timeout=30)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads(proc.stdout.strip())
    assert data['event'] == 'dense_verified'
    assert data['admission']['complete_geometry_vertex_count'] == 120
    command[-1] = '0'*64
    rejected = subprocess.run(command, cwd=script.parents[1], text=True,
                              capture_output=True, timeout=30)
    assert rejected.returncode != 0
    assert 'dense_verification_failed' in rejected.stdout

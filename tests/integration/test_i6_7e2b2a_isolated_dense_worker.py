"""I6.7E2B2A: source-bound dense worker, no real PatchMatch needed."""
from dataclasses import replace
import json
from pathlib import Path
import shutil
from uuid import uuid4

import pytest

from app.reconstruction_dense_production_worker import (
    load_admitted_sparse_for_dense, execute_explicit_colmap_dense_after_sparse,
)
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.mvs_provider import MvsProductionProviderAdapter
from tests.i4_3b_helpers import FakeWorkerRunner, good_environment, good_resources
from tests.unit.test_i6_7e2a_sparse_dense_handoff_gate import fixture_state


class _PreviewSink:
    def __init__(self): self.events=[]
    def publish(self, item): self.events.append(item)


def setup(tmp_path):
    state, payload, root, _ = fixture_state(tmp_path)
    run_id = state['envelope'].report.dense_run_id
    shutil.rmtree(root / 'dense_runs')
    py = tmp_path / 'pycolmap-python'; py.write_text('', encoding='utf8')
    worker = tmp_path / 'mvs-worker'; worker.write_text('', encoding='utf8')
    return state, payload, root, run_id, py, worker


def test_real_application_sparse_handoff_executes_fake_external_provider_and_admits_only_final(tmp_path):
    state, payload, root, run_id, py, worker = setup(tmp_path)
    view = _PreviewSink()
    def factory(context, sink):
        return MvsProductionProviderAdapter(
            context, progress_sink=sink, environment_probe=good_environment(str(py)),
            resource_probe=good_resources(context.output_dir),
            worker_runner=FakeWorkerRunner(ply_points=120, vis_points=120))
    result = execute_explicit_colmap_dense_after_sparse(
        sparse_output_dir=root, sparse_session_id=state['source_sfm_result'].session_id,
        pycolmap_python=py, mvs_worker_script=worker, dense_run_id=run_id,
        reconstruction_frame_id='i6_7e2b2:pycolmap_sfm:' + state['source_sfm_result'].session_id,
        snapshot_sink=view, adapter_factory=factory)
    assert result.vertex_count == 120
    assert len(view.events) > 0
    assert all(item.final_dense_result_admitted is False for item in view.events)
    assert Path(result.manifest_path).is_file()
    assert Path(result.geometry_path).name == 'fused.ply'
    assert result.sparse_session_id == state['source_sfm_result'].session_id
    with pytest.raises(ReconstructionValidationError, match='already exists|refusing'):
        execute_explicit_colmap_dense_after_sparse(
            sparse_output_dir=root, sparse_session_id=state['source_sfm_result'].session_id,
            pycolmap_python=py, mvs_worker_script=worker, dense_run_id=run_id,
            reconstruction_frame_id='i6_7e2b2:pycolmap_sfm:' + state['source_sfm_result'].session_id,
            snapshot_sink=view, adapter_factory=factory)


def test_no_dense_run_started_with_foreign_or_unadmitted_sparse(tmp_path):
    state, payload, root, run_id, py, worker = setup(tmp_path)
    with pytest.raises(ReconstructionValidationError, match='session'):
        load_admitted_sparse_for_dense(root, str(uuid4()))
    payload['pass'] = False
    (root/'offline_result.json').write_text(json.dumps(payload),encoding='utf8')
    with pytest.raises((ReconstructionValidationError, ValueError), match='completed|passing'):
        load_admitted_sparse_for_dense(root, state['source_sfm_result'].session_id)
    assert not (root/'dense_runs').exists()


def test_rejects_sfm_tamper_and_refuses_auto_or_wrong_provider(tmp_path):
    state, payload, root, run_id, py, worker = setup(tmp_path)
    sfm_path = root/'sfm_result.json'
    sfm = json.loads(sfm_path.read_text()); sfm['session_id']=str(uuid4())
    sfm_path.write_text(json.dumps(sfm))
    with pytest.raises((ReconstructionValidationError, ValueError), match='session|identity'):
        load_admitted_sparse_for_dense(root, state['source_sfm_result'].session_id)
    assert not (root/'dense_runs').exists()
    sfm_path.write_text(json.dumps(state['source_sfm_result'].as_dict()))
    with pytest.raises(ReconstructionValidationError, match='explicit COLMAP'):
        execute_explicit_colmap_dense_after_sparse(
            sparse_output_dir=root, sparse_session_id=state['source_sfm_result'].session_id,
            pycolmap_python=py, mvs_worker_script=worker, dense_run_id=run_id,
            reconstruction_frame_id='i6_7e2b2:pycolmap_sfm:' + state['source_sfm_result'].session_id,
            snapshot_sink=_PreviewSink(), provider_id='auto')
    assert not (root/'dense_runs').exists()


def test_worker_requires_explicit_opt_in_and_blocks_auto_before_execution(tmp_path):
    import subprocess, sys
    script = Path(__file__).resolve().parents[2] / 'scripts/run_i6_7e2b2a_post_sparse_dense_worker.py'
    help_result = subprocess.run([sys.executable, str(script), '--help'],
                                 capture_output=True, text=True, timeout=10)
    assert help_result.returncode == 0
    assert '--enable-explicit-colmap-mvs' in help_result.stdout
    args = ['--offline-output-dir', str(tmp_path), '--session-id', str(uuid4()),
            '--dense-run-id', str(uuid4()), '--pycolmap-python', sys.executable,
            '--provider-id', 'auto']
    rejected = subprocess.run([sys.executable, str(script), '--enable-explicit-colmap-mvs', *args],
                              capture_output=True, text=True, timeout=10)
    assert rejected.returncode == 2
    assert 'Auto is not supported' in rejected.stderr
    assert not (tmp_path / 'dense_runs').exists()


def test_preview_sink_failure_is_fail_soft_but_final_artifact_still_requires_admission(tmp_path):
    state, _, root, run_id, py, worker = setup(tmp_path)
    class _BadPreviewSink:
        def publish(self, _snapshot):
            raise RuntimeError('provisional rendering is unavailable')
    result = execute_explicit_colmap_dense_after_sparse(
        sparse_output_dir=root, sparse_session_id=state['source_sfm_result'].session_id,
        pycolmap_python=py, mvs_worker_script=worker, dense_run_id=run_id,
        reconstruction_frame_id='i6_7e2b2:pycolmap_sfm:' + state['source_sfm_result'].session_id,
        snapshot_sink=_BadPreviewSink(),
        adapter_factory=lambda context, sink: MvsProductionProviderAdapter(
            context, progress_sink=sink, environment_probe=good_environment(str(py)),
            resource_probe=good_resources(context.output_dir),
            worker_runner=FakeWorkerRunner(ply_points=19, vis_points=19)))
    assert result.vertex_count == 19
    assert Path(result.manifest_path).is_file()


def test_rejects_foreign_reconstruction_frame_before_spawning_mvs(tmp_path):
    state, _, root, run_id, py, worker = setup(tmp_path)
    invoked = []
    with pytest.raises(ReconstructionValidationError, match='reconstruction frame'):
        execute_explicit_colmap_dense_after_sparse(
            sparse_output_dir=root, sparse_session_id=state['source_sfm_result'].session_id,
            pycolmap_python=py, mvs_worker_script=worker, dense_run_id=run_id,
            reconstruction_frame_id='i6_7e2b2:pycolmap_sfm:' + str(uuid4()),
            snapshot_sink=_PreviewSink(),
            adapter_factory=lambda _ctx, _sink: invoked.append(True))
    assert invoked == []
    assert not (root / 'dense_runs').exists()


def test_declines_alternate_sfm_python_before_starting_provider(tmp_path):
    state, payload, root, run_id, py, worker = setup(tmp_path)
    other = tmp_path / 'different-pycolmap'; other.write_text('', encoding='utf8')
    payload['execution_boundary'] = {'pycolmap_python': str(py)}
    (root/'offline_result.json').write_text(json.dumps(payload), encoding='utf8')
    with pytest.raises(ReconstructionValidationError, match='PyCOLMAP environment'):
        execute_explicit_colmap_dense_after_sparse(
            sparse_output_dir=root, sparse_session_id=state['source_sfm_result'].session_id,
            pycolmap_python=other, mvs_worker_script=worker, dense_run_id=run_id,
            reconstruction_frame_id='i6_7e2b2:pycolmap_sfm:' + state['source_sfm_result'].session_id,
            snapshot_sink=_PreviewSink())
    assert not (root / 'dense_runs').exists()

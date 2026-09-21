"""Real adapter/wrapper integration with FAKE external worker; no GPU or SfM rerun."""
from dataclasses import replace
from pathlib import Path

from core.reconstruction.mvs_provider import MvsProductionProviderAdapter, MvsProviderRunContext
from core.reconstruction.reconstruction_progressive_dense_emission import execute_dense_provider_with_progressive_emission
from app.reconstruction_dense_final_handoff import prepare_dense_final_handoff
from tests.i4_3b_helpers import FakeWorkerRunner, good_environment, good_resources
from tests.unit.test_i6_7e2a_sparse_dense_handoff_gate import fixture_state


class _Sink:
    def publish(self, snapshot):
        pass


def test_completed_provider_result_produces_complete_candidate_without_provisional_admission(tmp_path: Path):
    state, _, _, _ = fixture_state(tmp_path)
    sfm = state['source_sfm_result']
    root = state['provider_output_root']
    # Fixture built a synthetic PLY; external provider must receive an EMPTY output.
    import shutil
    shutil.rmtree(root)
    py = tmp_path / 'pycolmap-python'; py.write_text('', encoding='utf-8')
    worker = tmp_path / 'worker'; worker.write_text('', encoding='utf-8')
    context = MvsProviderRunContext(
        sfm_result=sfm, reconstruction_frame_id=state['request'].authoritative_sfm_frame_id,
        output_dir=root, pycolmap_python=py, worker_script=worker,
    )
    sink = _Sink()
    envelope = execute_dense_provider_with_progressive_emission(
        adapter_factory=lambda progress_sink: MvsProductionProviderAdapter(
            context, progress_sink=progress_sink,
            environment_probe=good_environment(str(py)),
            resource_probe=good_resources(root),
            worker_runner=FakeWorkerRunner(ply_points=120, vis_points=120),
        ),
        request=state['request'], decision=state['decision'], source_sfm_result=sfm,
        reconstruction_frame_id=state['request'].authoritative_sfm_frame_id,
        snapshot_sink=sink, dense_run_id=state['envelope'].report.dense_run_id,
    )
    import hashlib
    ply = Path(envelope.result.canonical_geometry_ref)
    state.update(envelope=envelope, expected_artifact_sha256=hashlib.sha256(ply.read_bytes()).hexdigest())
    qualified = prepare_dense_final_handoff(**state)
    assert qualified.vertex_count == 120
    assert qualified.as_dict()['candidate_ready'] is True
    assert qualified.as_dict()['final_dense_result_admitted'] is False
    assert envelope.report.provider_execution_result_admitted_by_progressive_path is False

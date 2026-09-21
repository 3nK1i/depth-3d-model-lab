"""Completed adapter -> E2A candidate -> explicit final admission (fake worker)."""
from pathlib import Path
import hashlib
import shutil

from app.reconstruction_dense_final_admission import admit_completed_dense_result, read_admitted_dense_result
from app.reconstruction_dense_final_handoff import prepare_dense_final_handoff
from core.reconstruction.mvs_provider import MvsProductionProviderAdapter, MvsProviderRunContext
from core.reconstruction.reconstruction_progressive_dense_emission import (
    execute_dense_provider_with_progressive_emission,
)
from tests.i4_3b_helpers import FakeWorkerRunner, good_environment, good_resources
from tests.unit.test_i6_7e2a_sparse_dense_handoff_gate import fixture_state


class _NoPreviewSink:
    def publish(self, _snapshot):
        raise RuntimeError('offline preview transport failure must not admit final result')


def test_completed_fake_external_mvs_provider_commits_only_full_artifact(tmp_path: Path):
    state, _payload, root, _ply = fixture_state(tmp_path)
    provider_root = state['provider_output_root']
    shutil.rmtree(provider_root)
    python = tmp_path / 'pycolmap-python'; python.write_text('', encoding='utf-8')
    worker = tmp_path / 'mvs-worker'; worker.write_text('', encoding='utf-8')
    sfm = state['source_sfm_result']
    context = MvsProviderRunContext(
        sfm_result=sfm,
        reconstruction_frame_id=state['request'].authoritative_sfm_frame_id,
        output_dir=provider_root,
        pycolmap_python=python,
        worker_script=worker,
    )
    envelope = execute_dense_provider_with_progressive_emission(
        adapter_factory=lambda progress_sink: MvsProductionProviderAdapter(
            context,
            progress_sink=progress_sink,
            environment_probe=good_environment(str(python)),
            resource_probe=good_resources(provider_root),
            worker_runner=FakeWorkerRunner(ply_points=120, vis_points=120),
        ),
        request=state['request'], decision=state['decision'],
        source_sfm_result=sfm, reconstruction_frame_id=context.reconstruction_frame_id,
        snapshot_sink=_NoPreviewSink(), dense_run_id=state['envelope'].report.dense_run_id,
    )
    # Provider's complete artifact, not the (failed) provisional preview, is the source of truth.
    ply = Path(envelope.result.canonical_geometry_ref)
    state.update(envelope=envelope, expected_artifact_sha256=hashlib.sha256(ply.read_bytes()).hexdigest())
    candidate = prepare_dense_final_handoff(**state)
    admitted = admit_completed_dense_result(candidate=candidate, **state)
    assert admitted.vertex_count == 120
    assert read_admitted_dense_result(admitted, candidate=candidate) == admitted
    assert (root / 'dense_runs' / candidate.dense_run_id / 'final_dense_admission.json').is_file()
    assert envelope.report.provider_execution_result_admitted_by_progressive_path is False

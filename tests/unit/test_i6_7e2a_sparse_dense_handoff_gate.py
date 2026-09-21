"""I6.7E2A: admitted sparse -> explicitly selected completed dense artifact gate."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from uuid import uuid4

import pytest

from app.reconstruction_offline_result_handoff import build_offline_result_handoff
from core.reconstruction.dense_provider_architecture import (
    DenseProviderExecutionResult, DenseProviderImplementation, DenseProviderRunStatus,
    DenseProviderSelectionMode,
)
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.reconstruction_progressive_dense_emission import (
    ProgressiveDenseEmissionReport, ProgressiveDenseExecutionEnvelope,
    canonical_offline_sfm_result_sha256, dense_request_fingerprint_sha256,
)
from app.reconstruction_dense_final_handoff import prepare_dense_final_handoff
from tests.i4_3b_helpers import make_sfm_result, explicit_selection, write_minimal_ply


def fixture_state(tmp_path):
    sfm = make_sfm_result(tmp_path / 'source', view_count=3)
    output = tmp_path / 'offline'; output.mkdir()
    sfm_file = output / 'sfm_result.json'
    sfm_file.write_text(json.dumps(sfm.as_dict()), encoding='utf-8')
    payload = {
        'session_id': sfm.session_id, 'status': 'completed', 'pass': True,
        'camera_pose_available': True, 'canonical_geometry_authoritative': False,
        'persistent_scene_ready': False, 'certified_metrology': False,
        'sfm_summary': {'registered_keyframes': 3, 'gauge': {'scale_status': 'arbitrary'}},
        'artifacts': {'sfm_result_json': str(sfm_file)},
    }
    (output / 'offline_result.json').write_text(json.dumps(payload), encoding='utf-8')
    handoff = build_offline_result_handoff(payload, output,
                                          expected_session_id=sfm.session_id, expected_output_dir=output)
    frame = 'i6_7e2a:pycolmap_sfm:' + sfm.session_id
    request, decision = explicit_selection(3, frame_id=frame)
    run_id = str(uuid4())
    root = output / 'dense_runs' / run_id / 'provider_colmap_mvs'
    ply = root / 'dense' / 'fused.ply'; write_minimal_ply(ply, 7)
    result = DenseProviderExecutionResult(
        provider_id=decision.selected_provider_id,
        implementation=DenseProviderImplementation.COLMAP_MVS,
        status=DenseProviderRunStatus.COMPLETED,
        source_view_count=3, selected_view_count=3, adaptive_view_selection=False,
        view_selection_reason='all_registered_authoritative_sfm_views',
        permanent_global_view_cap=None, canonical_geometry_ref=str(ply),
        canonical_frame_id=frame, native_artifact_refs=(str(ply),),
    )
    sparse_sha = canonical_offline_sfm_result_sha256(sfm)
    report = ProgressiveDenseEmissionReport(
        dense_run_id=run_id, source_sparse_result_sha256=sparse_sha,
        request_fingerprint_sha256=dense_request_fingerprint_sha256(request, decision, sparse_sha, frame),
        emitted_snapshot_count=0, emitted_geometry_snapshot_count=0,
        dropped_status_snapshot_count=0, emission_failure_count=0, failure_records=(),
        final_preview_revision=0, geometry_snapshot_emitted=False, max_preview_points=4096,
    )
    inputs = dict(admitted_sparse_handoff=handoff, source_sfm_result=sfm,
                  request=request, decision=decision,
                  envelope=ProgressiveDenseExecutionEnvelope(result=result, report=report),
                  provider_output_root=root,
                  expected_artifact_sha256=hashlib.sha256(ply.read_bytes()).hexdigest())
    return inputs, payload, output, ply


def test_completed_artifact_bound_to_admitted_sparse_and_explicit_provider(tmp_path):
    inputs, _, _, ply = fixture_state(tmp_path)
    accepted = prepare_dense_final_handoff(**inputs)
    assert accepted.candidate_ready is True
    assert accepted.artifact_sha256 == inputs['expected_artifact_sha256']
    assert accepted.vertex_count == 7
    assert accepted.source_sparse_result_sha256 == canonical_offline_sfm_result_sha256(inputs['source_sfm_result'])
    assert accepted.geometry_path == str(ply)
    assert accepted.as_dict()['final_dense_result_admitted'] is False
    assert accepted.as_dict()['persistent_dense_scene_created'] is False
    assert accepted.as_dict()['full_density_gpu_viewer_enabled'] is False
    assert accepted.as_dict()['metric_scale_claimed'] is False


@pytest.mark.parametrize('mutation, failure', [
    (lambda i: i.update(admitted_sparse_handoff=replace(i['admitted_sparse_handoff'], session_id=str(uuid4()))), 'session'),
    (lambda i: i.update(admitted_sparse_handoff=replace(i['admitted_sparse_handoff'], camera_pose_available=False)), 'camera'),
    (lambda i: i.update(admitted_sparse_handoff=replace(i['admitted_sparse_handoff'], canonical_geometry_authoritative=True)), 'sparse'),
    (lambda i: i.update(admitted_sparse_handoff=replace(i['admitted_sparse_handoff'], availability=i['admitted_sparse_handoff'].availability.__class__.NONE)), 'partial'),
    (lambda i: i.update(request=replace(i['request'], allow_fallback=True)), 'fallback'),
    (lambda i: i.update(request=replace(i['request'], authoritative_sfm_frame_id='foreign-frame')), 'frame'),
    (lambda i: i.update(request=replace(i['request'], available_view_count=4)), 'view'),
    (lambda i: i.update(decision=replace(i['decision'], fallback_used=True, selected_provider_id='foreign-provider', considered_provider_ids=('foreign-provider',))), 'provider'),
    (lambda i: i.update(envelope=replace(i['envelope'], result=replace(i['envelope'].result, canonical_frame_id='foreign-frame'))), 'frame'),
    (lambda i: i.update(envelope=replace(i['envelope'], result=replace(i['envelope'].result, status=DenseProviderRunStatus.DEGRADED, degradation_reasons=('review',)))), 'completed'),
    (lambda i: i.update(envelope=replace(i['envelope'], report=replace(i['envelope'].report, source_sparse_result_sha256='a'*64))), 'sparse'),
    (lambda i: i.update(envelope=replace(i['envelope'], report=replace(i['envelope'].report, request_fingerprint_sha256='a'*64))), 'fingerprint'),
    (lambda i: i.update(envelope=replace(i['envelope'], report=replace(i['envelope'].report, provider_execution_result_admitted_by_progressive_path=True))), 'provisional'),
    (lambda i: i.update(expected_artifact_sha256='a'*64), 'SHA256'),
])
def test_fails_closed_on_identity_or_authority_drift(tmp_path, mutation, failure):
    inputs, _, _, _ = fixture_state(tmp_path)
    mutation(inputs)
    with pytest.raises((ReconstructionValidationError, ValueError), match=failure):
        prepare_dense_final_handoff(**inputs)


def test_rejects_mutated_or_unbound_offline_disk_result(tmp_path):
    inputs, payload, output, _ = fixture_state(tmp_path)
    payload['session_id'] = str(uuid4())
    (output / 'offline_result.json').write_text(json.dumps(payload), encoding='utf-8')
    with pytest.raises(ReconstructionValidationError, match='offline.*session'):
        prepare_dense_final_handoff(**inputs)


def test_rejects_modified_persisted_sfm_result(tmp_path):
    inputs, _, output, _ = fixture_state(tmp_path)
    sfm_file = output / 'sfm_result.json'
    doc = json.loads(sfm_file.read_text(encoding='utf-8'))
    doc['backend_version'] = 'altered'
    sfm_file.write_text(json.dumps(doc), encoding='utf-8')
    with pytest.raises(ReconstructionValidationError, match='SfM.*identity'):
        prepare_dense_final_handoff(**inputs)


def test_rejects_geometry_outside_run_or_symlink(tmp_path):
    inputs, _, output, ply = fixture_state(tmp_path)
    escaped = output / 'foreign.ply'
    escaped.write_bytes(ply.read_bytes())
    inputs['envelope'] = replace(inputs['envelope'], result=replace(inputs['envelope'].result, canonical_geometry_ref=str(escaped)))
    with pytest.raises(ReconstructionValidationError, match='output root'):
        prepare_dense_final_handoff(**inputs)
    inputs, _, _, ply = fixture_state(tmp_path / 'different')
    external = tmp_path / 'external.ply'; external.write_bytes(ply.read_bytes())
    ply.unlink(); ply.symlink_to(external)
    with pytest.raises(ReconstructionValidationError, match='symlink'):
        prepare_dense_final_handoff(**inputs)


def test_rejects_missing_color_and_does_not_write_admission(tmp_path):
    inputs, _, output, ply = fixture_state(tmp_path)
    ply.write_text('ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n0 0 0\n', encoding='ascii')
    inputs['expected_artifact_sha256'] = hashlib.sha256(ply.read_bytes()).hexdigest()
    with pytest.raises(ReconstructionValidationError, match='RGB'):
        prepare_dense_final_handoff(**inputs)
    assert not any(output.glob('**/*admission*'))

"""I6.7E2B1: explicit final-result admission, separate from provisional preview."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from uuid import uuid4

import pytest

from app.reconstruction_dense_final_handoff import prepare_dense_final_handoff
from app.reconstruction_dense_final_admission import (
    admit_completed_dense_result, read_admitted_dense_result,
)
from core.reconstruction.errors import ReconstructionValidationError
from tests.unit.test_i6_7e2a_sparse_dense_handoff_gate import fixture_state


def _valid(tmp_path):
    inputs, _payload, root, ply = fixture_state(tmp_path)
    candidate = prepare_dense_final_handoff(**inputs)
    manifest = root / 'dense_runs' / candidate.dense_run_id / 'final_dense_admission.json'
    return inputs, candidate, manifest, ply


def test_final_admission_commits_complete_artifact_once_and_rechecks_on_read(tmp_path):
    inputs, candidate, manifest, ply = _valid(tmp_path)
    admitted = admit_completed_dense_result(candidate=candidate, **inputs)
    assert manifest.is_file()
    assert admitted.manifest_path == str(manifest)
    assert admitted.vertex_count == candidate.vertex_count == 7
    assert admitted.artifact_sha256 == candidate.artifact_sha256
    assert admitted.sparse_session_id == inputs['source_sfm_result'].session_id
    assert admitted.reconstruction_frame_id == candidate.reconstruction_frame_id
    document = json.loads(manifest.read_text(encoding='utf-8'))
    assert document['final_dense_result_admitted'] is True
    assert document['persistent_dense_scene_created'] is False
    assert document['full_density_gpu_viewer_enabled'] is False
    assert document['metric_scale_claimed'] is False
    assert document['camera_pose_authority_owner'] == 'pycolmap_sfm'
    assert read_admitted_dense_result(admitted, candidate=candidate) == admitted
    assert ply.is_file()


def test_double_admission_rejected_without_overwriting_receipt(tmp_path):
    inputs, candidate, manifest, _ = _valid(tmp_path)
    admit_completed_dense_result(candidate=candidate, **inputs)
    saved = manifest.read_bytes()
    with pytest.raises(ReconstructionValidationError, match='already exists'):
        admit_completed_dense_result(candidate=candidate, **inputs)
    assert manifest.read_bytes() == saved


def test_candidate_forgery_or_foreign_session_is_rejected_before_file_write(tmp_path):
    inputs, candidate, manifest, _ = _valid(tmp_path)
    with pytest.raises(ReconstructionValidationError, match='candidate.*mismatch'):
        admit_completed_dense_result(candidate=replace(candidate, sparse_session_id=str(uuid4())), **inputs)
    assert not manifest.exists()


def test_snapshot_cannot_admit_without_completed_provider_envelope(tmp_path):
    inputs, candidate, manifest, _ = _valid(tmp_path)
    inputs['envelope'] = replace(inputs['envelope'], result=replace(inputs['envelope'].result, status=inputs['envelope'].result.status.__class__.DEGRADED, degradation_reasons=('requires_review',)))
    with pytest.raises(ReconstructionValidationError, match='completed'):
        admit_completed_dense_result(candidate=candidate, **inputs)
    assert not manifest.exists()


def test_changed_artifact_between_candidate_and_admission_is_rejected(tmp_path):
    inputs, candidate, manifest, ply = _valid(tmp_path)
    with ply.open('ab') as handle:
        handle.write(b'changed')
    with pytest.raises(ReconstructionValidationError, match='SHA256'):
        admit_completed_dense_result(candidate=candidate, **inputs)
    assert not manifest.exists()


def test_existing_unsafe_or_symlinked_admission_target_is_not_replaced(tmp_path):
    inputs, candidate, manifest, _ = _valid(tmp_path)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    external = tmp_path / 'outside'; external.write_text('do not touch', encoding='utf-8')
    manifest.symlink_to(external)
    with pytest.raises(ReconstructionValidationError, match='already exists'):
        admit_completed_dense_result(candidate=candidate, **inputs)
    assert external.read_text(encoding='utf-8') == 'do not touch'


def test_reader_rejects_artifact_mutation_after_final_admission(tmp_path):
    inputs, candidate, _, ply = _valid(tmp_path)
    admitted = admit_completed_dense_result(candidate=candidate, **inputs)
    with ply.open('ab') as handle:
        handle.write(b'changed')
    with pytest.raises(ReconstructionValidationError, match='SHA256'):
        read_admitted_dense_result(admitted, candidate=candidate)


def test_reader_rejects_admission_manifest_or_candidate_drift(tmp_path):
    inputs, candidate, manifest, _ = _valid(tmp_path)
    admitted = admit_completed_dense_result(candidate=candidate, **inputs)
    with pytest.raises(ReconstructionValidationError, match='candidate'):
        read_admitted_dense_result(admitted, candidate=replace(candidate, reconstruction_frame_id='foreign'))
    document = json.loads(manifest.read_text(encoding='utf-8'))
    document['metric_scale_claimed'] = True
    manifest.write_text(json.dumps(document), encoding='utf-8')
    with pytest.raises(ReconstructionValidationError, match='manifest'):
        read_admitted_dense_result(admitted, candidate=candidate)


def test_artifact_does_not_need_to_fit_in_preview_transport(tmp_path):
    inputs, candidate, _manifest, _ply = _valid(tmp_path)
    # The full result's count comes from complete PLY, never the 4096-point preview.
    admitted = admit_completed_dense_result(candidate=candidate, **inputs)
    assert admitted.vertex_count == candidate.vertex_count
    assert admitted.as_dict()['preview_geometry_used_for_admission'] is False


def test_reader_rejects_oversized_admission_manifest_without_large_read(tmp_path):
    inputs, candidate, manifest, _ = _valid(tmp_path)
    admitted = admit_completed_dense_result(candidate=candidate, **inputs)
    with manifest.open('ab') as handle:
        handle.truncate(128 * 1024)
    with pytest.raises(ReconstructionValidationError, match='bounded read budget'):
        read_admitted_dense_result(admitted, candidate=candidate)


def test_concurrent_same_run_admissions_have_exactly_one_winner(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    inputs, candidate, manifest, _ = _valid(tmp_path)

    def attempt(_index):
        try:
            return admit_completed_dense_result(candidate=candidate, **inputs)
        except ReconstructionValidationError as exc:
            return str(exc)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(attempt, range(2)))
    assert sum(not isinstance(item, str) for item in results) == 1
    assert any('already exists' in item for item in results if isinstance(item, str))
    assert manifest.is_file()

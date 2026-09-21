"""I6.7E production read-only completion check; run in an isolated child process.

A worker's ``dense_completed`` line is a claim, not a UI admission. This module
rebuilds the E2B1 typed record from on-disk evidence, binds it to the live sparse
session/run, and performs a fresh full-PLY scan outside the GUI thread.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

from app.reconstruction_dense_final_admission import AdmittedDenseResult, read_admitted_dense_result
from app.reconstruction_dense_final_handoff import QualifiedDenseFinalHandoffCandidate
from app.reconstruction_dense_production_worker import load_admitted_sparse_for_dense
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.mvs_provider import I43B_PROVIDER_ID
from core.reconstruction.reconstruction_progressive_dense_emission import canonical_offline_sfm_result_sha256


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ReconstructionValidationError('I6.7E dense completion verification: ' + reason)


def verify_dense_completion(
    *, offline_output_dir: str | Path, sparse_session_id: str,
    dense_run_id: str, expected_manifest_sha256: str,
    expected_geometry_sha256: str,
) -> AdmittedDenseResult:
    """Reverify exactly one completed run; never mutate its manifest/artifact."""
    sparse, sfm = load_admitted_sparse_for_dense(offline_output_dir, sparse_session_id)
    root = Path(sparse.output_dir)
    require(root == Path(offline_output_dir).resolve(strict=True), 'offline output root mismatch')
    run = root / 'dense_runs' / dense_run_id
    require(run.is_dir() and not run.is_symlink(), 'admitted run directory is missing or symlinked')
    manifest_path = run / 'final_dense_admission.json'
    require(manifest_path.is_file() and not manifest_path.is_symlink(), 'no completed admission manifest')
    require(manifest_path.stat().st_size <= 64 * 1024, 'admission manifest is oversized')
    raw = manifest_path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected_manifest_sha256, 'admission SHA256 differs from worker completion')
    payload = json.loads(raw)
    require(isinstance(payload, dict) and payload.get('schema_version') == 1, 'invalid admission schema')
    require(payload.get('phase') == 'I6.7E2B1_EXPLICIT_FULL_DENSE_ADMISSION', 'wrong admission phase')
    require(payload.get('sparse_session_id') == sparse.session_id == sfm.session_id, 'sparse session mismatch')
    require(payload.get('source_sparse_result_sha256') == canonical_offline_sfm_result_sha256(sfm),
            'source SfM fingerprint mismatch')
    require(payload.get('reconstruction_frame_id') == 'i6_7e2b2:pycolmap_sfm:' + sfm.session_id,
            'foreign PyCOLMAP reconstruction frame')
    require(payload.get('dense_run_id') == dense_run_id, 'foreign dense run')
    require(payload.get('provider_id') == I43B_PROVIDER_ID
            and payload.get('implementation') == 'colmap_mvs', 'non-explicit concrete provider')
    geometry = run / 'provider_colmap_mvs' / 'dense' / 'fused.ply'
    require(payload.get('complete_geometry_path') == str(geometry) and geometry.is_file()
            and not geometry.is_symlink(), 'foreign dense geometry path')
    require(payload.get('complete_geometry_sha256') == expected_geometry_sha256,
            'full geometry SHA256 differs from worker completion')
    for flag in ('preview_geometry_used_for_admission', 'persistent_dense_scene_created',
                 'full_density_gpu_viewer_enabled', 'production_auto_activated',
                 'metric_scale_claimed', 'measurement_authoritative',
                 'certified_metrology_claimed', 'surface_mesh_generated'):
        require(payload.get(flag) is False, 'unsupported authority asserted: ' + flag)
    require(payload.get('final_dense_result_admitted') is True, 'final admission is not complete')
    count = payload.get('complete_geometry_vertex_count')
    size = payload.get('complete_geometry_file_bytes')
    require(type(count) is int and count > 0 and type(size) is int and size > 0,
            'invalid full artifact size or count')
    bounds_min, bounds_max = payload.get('bounds_min'), payload.get('bounds_max')
    require(isinstance(bounds_min, list) and len(bounds_min) == 3
            and isinstance(bounds_max, list) and len(bounds_max) == 3,
            'missing artifact bounds')
    candidate = QualifiedDenseFinalHandoffCandidate(
        sparse_session_id=sparse.session_id,
        source_sparse_result_sha256=canonical_offline_sfm_result_sha256(sfm),
        reconstruction_frame_id=payload['reconstruction_frame_id'],
        dense_run_id=dense_run_id, provider_id=I43B_PROVIDER_ID,
        implementation='colmap_mvs', geometry_path=str(geometry),
        artifact_sha256=expected_geometry_sha256, file_bytes=size, vertex_count=count,
        bounds_min=tuple(bounds_min), bounds_max=tuple(bounds_max),
    )
    record = AdmittedDenseResult(
        manifest_path=str(manifest_path), manifest_sha256=expected_manifest_sha256,
        sparse_session_id=sparse.session_id,
        source_sparse_result_sha256=candidate.source_sparse_result_sha256,
        reconstruction_frame_id=candidate.reconstruction_frame_id,
        dense_run_id=dense_run_id, provider_id=I43B_PROVIDER_ID,
        geometry_path=str(geometry), artifact_sha256=expected_geometry_sha256,
        file_bytes=size, vertex_count=count,
    )
    return read_admitted_dense_result(record, candidate=candidate)

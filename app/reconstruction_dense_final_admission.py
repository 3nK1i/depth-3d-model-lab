"""I6.7E2B1: explicit, single-commit completed dense-artifact admission.

Admission is independent of the provisional I6.7B-D transport and is *not*
scene persistence or application-worker wiring. A subsequent production bridge
must invoke this off the GUI thread after an admitted sparse result and a
completed explicitly selected MVS provider envelope exist.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

from app.reconstruction_dense_final_handoff import (
    QualifiedDenseFinalHandoffCandidate, prepare_dense_final_handoff,
)
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.reconstruction_dense_artifact_preflight import inspect_full_dense_ply

I67E2B1_PHASE = 'I6.7E2B1_EXPLICIT_FULL_DENSE_ADMISSION'
_MANIFEST_NAME = 'final_dense_admission.json'
_MAX_MANIFEST_BYTES = 64 * 1024


def _require(condition: bool, detail: str) -> None:
    if not condition:
        raise ReconstructionValidationError('I6.7E2B1 final dense admission: ' + detail)


def _manifest_bytes(candidate: QualifiedDenseFinalHandoffCandidate) -> bytes:
    data = {
        'schema_version': 1, 'phase': I67E2B1_PHASE,
        'sparse_session_id': candidate.sparse_session_id,
        'source_sparse_result_sha256': candidate.source_sparse_result_sha256,
        'reconstruction_frame_id': candidate.reconstruction_frame_id,
        'dense_run_id': candidate.dense_run_id,
        'provider_id': candidate.provider_id,
        'implementation': candidate.implementation,
        'complete_geometry_path': candidate.geometry_path,
        'complete_geometry_sha256': candidate.artifact_sha256,
        'complete_geometry_file_bytes': candidate.file_bytes,
        'complete_geometry_vertex_count': candidate.vertex_count,
        'bounds_min': list(candidate.bounds_min),
        'bounds_max': list(candidate.bounds_max),
        'final_dense_result_admitted': True,
        'preview_geometry_used_for_admission': False,
        'persistent_dense_scene_created': False,
        'full_density_gpu_viewer_enabled': False,
        'production_one_start_dense_wiring': False,
        'camera_pose_authority_owner': 'pycolmap_sfm',
        'metric_scale_claimed': False,
        'measurement_authoritative': False,
        'certified_metrology_claimed': False,
        'surface_mesh_generated': False,
        'production_auto_activated': False,
    }
    return (json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf-8')


@dataclass(frozen=True, slots=True)
class AdmittedDenseResult:
    manifest_path: str
    manifest_sha256: str
    sparse_session_id: str
    source_sparse_result_sha256: str
    reconstruction_frame_id: str
    dense_run_id: str
    provider_id: str
    geometry_path: str
    artifact_sha256: str
    file_bytes: int
    vertex_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            'schema_version': 1, 'phase': I67E2B1_PHASE,
            'manifest_path': self.manifest_path,
            'manifest_sha256': self.manifest_sha256,
            'sparse_session_id': self.sparse_session_id,
            'source_sparse_result_sha256': self.source_sparse_result_sha256,
            'reconstruction_frame_id': self.reconstruction_frame_id,
            'dense_run_id': self.dense_run_id,
            'provider_id': self.provider_id,
            'complete_geometry_path': self.geometry_path,
            'complete_geometry_sha256': self.artifact_sha256,
            'complete_geometry_file_bytes': self.file_bytes,
            'complete_geometry_vertex_count': self.vertex_count,
            'final_dense_result_admitted': True,
            'preview_geometry_used_for_admission': False,
            'persistent_dense_scene_created': False,
            'full_density_gpu_viewer_enabled': False,
            'production_one_start_dense_wiring': False,
            'metric_scale_claimed': False,
            'measurement_authoritative': False,
            'certified_metrology_claimed': False,
        }


def _expected_manifest(candidate: QualifiedDenseFinalHandoffCandidate) -> Path:
    # A caller cannot select an arbitrary destination. The E2A gate has already
    # proven the exact run/provider geometry path and admitted output root.
    ply = Path(candidate.geometry_path)
    _require(ply.is_absolute() and ply.name == 'fused.ply'
             and ply.parent.name == 'dense'
             and ply.parent.parent.name == 'provider_colmap_mvs'
             and ply.parent.parent.parent.name == candidate.dense_run_id
             and ply.parent.parent.parent.parent.name == 'dense_runs',
             'admission destination differs from the validated provider run')
    folder = ply.parent.parent.parent
    _require(folder.is_dir() and not folder.is_symlink(), 'run directory is missing or symlinked')
    return folder / _MANIFEST_NAME


def _record(candidate: QualifiedDenseFinalHandoffCandidate, manifest: Path,
            encoded: bytes) -> AdmittedDenseResult:
    return AdmittedDenseResult(
        manifest_path=str(manifest), manifest_sha256=hashlib.sha256(encoded).hexdigest(),
        sparse_session_id=candidate.sparse_session_id,
        source_sparse_result_sha256=candidate.source_sparse_result_sha256,
        reconstruction_frame_id=candidate.reconstruction_frame_id,
        dense_run_id=candidate.dense_run_id, provider_id=candidate.provider_id,
        geometry_path=candidate.geometry_path, artifact_sha256=candidate.artifact_sha256,
        file_bytes=candidate.file_bytes, vertex_count=candidate.vertex_count,
    )


def admit_completed_dense_result(*, candidate: QualifiedDenseFinalHandoffCandidate,
                                 admitted_sparse_handoff, source_sfm_result, request,
                                 decision, envelope, provider_output_root,
                                 expected_artifact_sha256) -> AdmittedDenseResult:
    """Independently requalify *all* E2A inputs, then atomically admit once.

    Off-Qt-thread CPU/disk work. Does not call an MVS provider, alter a scene or
    notify the viewport. No implicit retry if the destination already exists.
    """
    _require(isinstance(candidate, QualifiedDenseFinalHandoffCandidate)
             and candidate.candidate_ready, 'a qualified candidate is required')
    fresh = prepare_dense_final_handoff(
        admitted_sparse_handoff=admitted_sparse_handoff,
        source_sfm_result=source_sfm_result, request=request, decision=decision,
        envelope=envelope, provider_output_root=provider_output_root,
        expected_artifact_sha256=expected_artifact_sha256,
    )
    _require(candidate == fresh, 'candidate mismatch after independent source revalidation')
    manifest = _expected_manifest(fresh)
    _require(not manifest.exists() and not manifest.is_symlink(),
             'final admission manifest already exists; refusing overwrite/retry')
    geometry = Path(fresh.geometry_path)
    before = geometry.stat()
    encoded = _manifest_bytes(fresh)
    tmp = manifest.with_name('.final_dense_admission.' + uuid4().hex + '.tmp')
    try:
        fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, 'wb') as out:
            out.write(encoded)
            out.flush()
            os.fsync(out.fileno())
        # Atomic, no-clobber commit. os.replace would silently replace an
        # existing admission in concurrent calls, which is forbidden here.
        after = geometry.stat()
        _require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                 == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
                 'complete geometry changed before the admission commit')
        try:
            os.link(tmp, manifest, follow_symlinks=False)
        except FileExistsError as exc:
            raise ReconstructionValidationError(
                'I6.7E2B1 final dense admission: manifest already exists; refusing overwrite/retry') from exc
        # Disk persistence of directory entry before reporting admission.
        directory_fd = os.open(manifest.parent, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        # Never remove a published manifest, including when fsync fails: a
        # post-commit error must be inspected rather than silently retried.
        if tmp.exists():
            tmp.unlink()
    return _record(fresh, manifest, encoded)


def read_admitted_dense_result(admitted: AdmittedDenseResult, *,
                               candidate: QualifiedDenseFinalHandoffCandidate) -> AdmittedDenseResult:
    """Read-only consumption gate: recheck admission bytes and the *full* PLY.

    Run outside the Qt render thread. File content can change after an earlier
    admission: no consumer may use only a stale in-memory admission flag.
    """
    _require(isinstance(admitted, AdmittedDenseResult), 'typed final admission is required')
    _require(isinstance(candidate, QualifiedDenseFinalHandoffCandidate)
             and candidate.candidate_ready, 'original qualified candidate is required')
    manifest = _expected_manifest(candidate)
    expected = _manifest_bytes(candidate)
    _require(admitted == _record(candidate, manifest, expected),
             'admission and candidate identity mismatch')
    _require(manifest.is_file() and not manifest.is_symlink(),
             'admission manifest must be a regular non-symlinked file')
    _require(manifest.stat().st_size <= _MAX_MANIFEST_BYTES,
             'admission manifest exceeds bounded read budget')
    _require(manifest.read_bytes() == expected,
             'admission manifest bytes/authority changed since commit')
    scanned = inspect_full_dense_ply(
        candidate.geometry_path, expected_sha256=candidate.artifact_sha256,
        expected_root=Path(candidate.geometry_path).parent.parent,
        expected_vertex_count=candidate.vertex_count,
    )
    _require(scanned.has_rgb and scanned.file_bytes == candidate.file_bytes
             and scanned.bounds_min == candidate.bounds_min
             and scanned.bounds_max == candidate.bounds_max,
             'complete geometry changed after final admission')
    return admitted


__all__ = [
    'I67E2B1_PHASE', 'AdmittedDenseResult',
    'admit_completed_dense_result', 'read_admitted_dense_result',
]

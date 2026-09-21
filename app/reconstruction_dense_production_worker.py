"""I6.7E2B2A: explicit, isolated post-sparse COLMAP-MVS execution and admission.

This is the *worker-side* production handoff callable. It must run outside the
Qt GUI process. It does not start itself from One Start or update the viewport;
those are separate product/controller integration gates.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable
from uuid import UUID

from app.reconstruction_dense_final_admission import (
    AdmittedDenseResult, admit_completed_dense_result, read_admitted_dense_result,
)
from app.reconstruction_dense_final_handoff import prepare_dense_final_handoff
from app.reconstruction_offline_result_handoff import (
    OfflineReconstructionResultHandoff, build_offline_result_handoff,
)
from core.reconstruction.dense_provider_architecture import (
    DenseProviderImplementation, DenseProviderSelectionDecision,
    DenseProviderSelectionMode, DenseProviderSelectionRequest,
)
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.mvs_provider import (
    I43B_PROVIDER_ID, MvsProductionProviderAdapter, MvsProviderRunContext,
)
from core.reconstruction.reconstruction_progressive_dense_emission import (
    ProgressiveDenseSnapshotSink,
    execute_dense_provider_with_progressive_emission,
)
from core.reconstruction.sfm_contracts import OfflineSfmResult, SfmScaleStatus
from core.reconstruction.sfm_pycolmap import PYCOLMAP_BACKEND_ID
from core.reconstruction.sfm_result_serialization import offline_sfm_result_from_dict


I67E2B2A_PHASE = 'I6.7E2B2A_POST_SPARSE_DENSE_WORKER'
AdapterFactory = Callable[[MvsProviderRunContext, object], MvsProductionProviderAdapter]


def _require(yes: bool, detail: str) -> None:
    if not yes:
        raise ReconstructionValidationError('I6.7E2B2A isolated dense worker: ' + detail)


# Offline result metadata stays at the original 64 MiB limit. A real accepted
# South Building SfM result with 87 registered views is 94,421,034 bytes, so
# the authoritative SfM payload needs a separate, still bounded 128 MiB limit.
_OFFLINE_JSON_MAX_BYTES = 64 * 1024 * 1024
_SFM_JSON_MAX_BYTES = 128 * 1024 * 1024


def _read_json(path: Path, label: str) -> dict:
    _require(path.is_file() and not path.is_symlink(), f'{label} must be an existing non-symlinked file')
    limit = _SFM_JSON_MAX_BYTES if label == 'SfM result' else _OFFLINE_JSON_MAX_BYTES
    _require(path.stat().st_size <= limit, f'{label} exceeds bounded input size')
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError, UnicodeError) as exc:
        raise ReconstructionValidationError(f'I6.7E2B2A invalid {label}: {exc}') from exc
    _require(isinstance(payload, dict), f'{label} must contain one JSON object')
    return payload


def load_admitted_sparse_for_dense(
    sparse_output_dir: str | Path, sparse_session_id: str,
) -> tuple[OfflineReconstructionResultHandoff, OfflineSfmResult]:
    """Reject foreign, failed, stale, or unbound One Start evidence before MVS."""
    root = Path(sparse_output_dir).expanduser()
    _require(root.is_absolute() and root.is_dir() and not root.is_symlink(),
             'admitted offline output root must be a non-symlinked absolute directory')
    root = root.resolve(strict=True)
    try:
        sid = str(UUID(sparse_session_id))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ReconstructionValidationError('I6.7E2B2A invalid sparse session ID') from exc
    _require(sid == sparse_session_id, 'non-canonical sparse session ID')
    payload = _read_json(root / 'offline_result.json', 'offline result')
    _require(payload.get('session_id') == sid, 'offline result session mismatch')
    handoff = build_offline_result_handoff(payload, root,
        expected_session_id=sid, expected_output_dir=root)
    _require(handoff.status == 'completed' and handoff.camera_pose_available
             and not handoff.canonical_geometry_authoritative
             and not handoff.persistent_scene_ready and not handoff.certified_metrology
             and handoff.scale_status == 'arbitrary',
             'admitted partial sparse result without metric/canonical promotion required')
    sfm_file = root / 'sfm_result.json'
    _require(isinstance(payload.get('artifacts'), dict)
             and payload['artifacts'].get('sfm_result_json') == str(sfm_file),
             'offline result must bind the exact persisted sparse SfM file')
    sfm_payload = _read_json(sfm_file, 'SfM result')
    sfm = offline_sfm_result_from_dict(sfm_payload)
    _require(sfm.session_id == sid, 'persisted SfM session mismatch')
    _require(sfm.backend_id == PYCOLMAP_BACKEND_ID and sfm.statistics.registered_keyframes >= 3,
             'authoritative PyCOLMAP sparse SfM with at least three views required')
    _require(sfm.gauge.scale_status is SfmScaleStatus.ARBITRARY
             and sfm.gauge.unit_to_metre is None,
             'raw sparse SfM must not claim metric scale')
    summary = payload.get('sfm_summary')
    _require(isinstance(summary, dict)
             and summary.get('registered_keyframes') == sfm.statistics.registered_keyframes
             and isinstance(summary.get('gauge'), dict)
             and summary['gauge'].get('scale_status') == 'arbitrary',
             'offline sparse summary does not match persisted SfM')
    _require(not (root / 'dense_runs').is_symlink(), 'dense runs root must not be symlinked')
    return handoff, sfm


def _hash_file(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


def execute_explicit_colmap_dense_after_sparse(
    *, sparse_output_dir: str | Path, sparse_session_id: str,
    pycolmap_python: str | Path, mvs_worker_script: str | Path,
    reconstruction_frame_id: str, dense_run_id: str,
    snapshot_sink: ProgressiveDenseSnapshotSink,
    provider_id: str = I43B_PROVIDER_ID,
    timeout_minutes: float = 120.0,
    adapter_factory: AdapterFactory | None = None,
) -> AdmittedDenseResult:
    """Execute one concrete provider for an admitted sparse run; admit *full* PLY.

    Called in a separately supervised worker process, never on the Qt thread.
    A failed/incomplete invocation preserves the per-run folder as negative
    evidence. It must not be silently retried or reused for a different run.
    """
    _require(provider_id == I43B_PROVIDER_ID, 'only explicit COLMAP-MVS is supported; Auto is forbidden')
    _require(callable(getattr(snapshot_sink, 'publish', None)), 'bounded snapshot sink required')
    try:
        run_id = str(UUID(dense_run_id))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ReconstructionValidationError('I6.7E2B2A invalid dense run UUID') from exc
    _require(run_id == dense_run_id, 'dense run UUID is non-canonical')
    _require(isinstance(reconstruction_frame_id, str) and bool(reconstruction_frame_id.strip())
             and reconstruction_frame_id.startswith('i6_7e2b2:pycolmap_sfm:'),
             'explicit PyCOLMAP reconstruction frame required')
    handoff, sfm = load_admitted_sparse_for_dense(sparse_output_dir, sparse_session_id)
    _require(reconstruction_frame_id == 'i6_7e2b2:pycolmap_sfm:' + sfm.session_id,
             'reconstruction frame must bind the admitted sparse session')
    root = Path(handoff.output_dir)
    run_root = root / 'dense_runs' / run_id
    _require(not run_root.exists() and not run_root.is_symlink(),
             'dense run already exists; refusing overwrite/retry')
    py = Path(pycolmap_python).expanduser().resolve(strict=True)
    worker = Path(mvs_worker_script).expanduser().resolve(strict=True)
    _require(py.is_file() and worker.is_file(), 'isolated PyCOLMAP Python and MVS worker required')
    _require(0.0 < float(timeout_minutes) <= 720.0, 'bounded positive provider timeout required')
    evidence = _read_json(root / 'offline_result.json', 'offline result')
    execution = evidence.get('execution_boundary')
    if isinstance(execution, dict) and execution.get('pycolmap_python'):
        _require(py == Path(execution['pycolmap_python']).resolve(),
                 'PyCOLMAP environment differs from admitted sparse execution')
    registered = sfm.statistics.registered_keyframes
    request = DenseProviderSelectionRequest(
        mode=DenseProviderSelectionMode.EXPLICIT_PROVIDER,
        available_view_count=registered, authoritative_sfm_available=True,
        authoritative_sfm_frame_id=reconstruction_frame_id,
        requested_provider_id=I43B_PROVIDER_ID, allow_fallback=False,
    )
    decision = DenseProviderSelectionDecision(
        request_mode=DenseProviderSelectionMode.EXPLICIT_PROVIDER,
        requested_provider_id=I43B_PROVIDER_ID, selected_provider_id=I43B_PROVIDER_ID,
        selected_implementation=DenseProviderImplementation.COLMAP_MVS,
        considered_provider_ids=(I43B_PROVIDER_ID,), automatic=False,
        engineer_override_applied=False, fallback_used=False,
        reason_codes=('explicit_one_start_post_sparse_request',),
    )
    provider_root = run_root / 'provider_colmap_mvs'
    context = MvsProviderRunContext(
        sfm_result=sfm, reconstruction_frame_id=reconstruction_frame_id,
        output_dir=provider_root, pycolmap_python=py, worker_script=worker,
        worker_timeout_s=float(timeout_minutes) * 60.0,
    )
    factory = adapter_factory or (lambda ctx, progress_sink:
        MvsProductionProviderAdapter(ctx, progress_sink=progress_sink))
    envelope = execute_dense_provider_with_progressive_emission(
        adapter_factory=lambda progress_sink: factory(context, progress_sink),
        request=request, decision=decision, source_sfm_result=sfm,
        reconstruction_frame_id=reconstruction_frame_id,
        snapshot_sink=snapshot_sink, dense_run_id=run_id,
    )
    geometry = Path(envelope.result.canonical_geometry_ref)
    _require(geometry == provider_root / 'dense' / 'fused.ply' and geometry.is_file()
             and not geometry.is_symlink(),
             'completed provider must yield its own exact full fused.ply')
    expected_sha = _hash_file(geometry)  # independent of provisional snapshots
    inputs = dict(
        admitted_sparse_handoff=handoff, source_sfm_result=sfm,
        request=request, decision=decision, envelope=envelope,
        provider_output_root=provider_root, expected_artifact_sha256=expected_sha,
    )
    qualified = prepare_dense_final_handoff(**inputs)
    admitted = admit_completed_dense_result(candidate=qualified, **inputs)
    return read_admitted_dense_result(admitted, candidate=qualified)


__all__ = ['I67E2B2A_PHASE', 'load_admitted_sparse_for_dense',
           'execute_explicit_colmap_dense_after_sparse']

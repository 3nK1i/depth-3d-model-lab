"""I6.7E2A independently qualified sparse->dense *candidate* handoff.

No production worker/UI wiring, scene persistence, or automatic final admission.
An eligible candidate is an immutable identity/provenance record, not a canonical
scene. The future caller must separately commit/admit it and recheck the source
artifact before consumption. I6.7B/C/D provisional previews are never inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from uuid import UUID

from app.reconstruction_offline_result_handoff import (
    OfflineReconstructionResultHandoff, OfflineResultHandoffAvailability,
)
from core.reconstruction.dense_provider_architecture import (
    DenseProviderImplementation, DenseProviderRunStatus,
    DenseProviderSelectionDecision, DenseProviderSelectionMode,
    DenseProviderSelectionRequest,
)
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.reconstruction_dense_artifact_preflight import inspect_full_dense_ply
from core.reconstruction.reconstruction_progressive_dense_emission import (
    ProgressiveDenseExecutionEnvelope, canonical_offline_sfm_result_sha256,
    dense_request_fingerprint_sha256,
)
from core.reconstruction.sfm_contracts import OfflineSfmResult, SfmScaleStatus
from core.reconstruction.sfm_pycolmap import PYCOLMAP_BACKEND_ID
from core.reconstruction.sfm_result_serialization import offline_sfm_result_from_dict
from core.reconstruction.validation import require_sha256

I67E2A_PHASE = 'I6.7E2A_SPARSE_DENSE_HANDOFF_CANDIDATE'


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReconstructionValidationError('I6.7E2A final handoff: ' + message)


def _json_regular_file(path: Path, *, label: str) -> dict:
    _require(path.is_file() and not path.is_symlink(), f'{label} must be a regular non-symlinked file')
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ReconstructionValidationError(f'I6.7E2A final handoff: invalid {label}: {exc}') from exc
    _require(isinstance(payload, dict), f'{label} must contain one JSON object')
    return payload


@dataclass(frozen=True, slots=True)
class QualifiedDenseFinalHandoffCandidate:
    """Complete verified artifact bound to one admitted sparse result and explicit run.

    This is an *eligibility* record. It does not admit a production/persistent scene.
    """
    sparse_session_id: str
    source_sparse_result_sha256: str
    reconstruction_frame_id: str
    dense_run_id: str
    provider_id: str
    implementation: str
    geometry_path: str
    artifact_sha256: str
    file_bytes: int
    vertex_count: int
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]
    candidate_ready: bool = True

    def as_dict(self) -> dict[str, object]:
        return {
            'schema_version': 1, 'phase': I67E2A_PHASE,
            'candidate_ready': self.candidate_ready,
            'sparse_session_id': self.sparse_session_id,
            'source_sparse_result_sha256': self.source_sparse_result_sha256,
            'reconstruction_frame_id': self.reconstruction_frame_id,
            'dense_run_id': self.dense_run_id,
            'provider_id': self.provider_id,
            'implementation': self.implementation,
            'complete_geometry_path': self.geometry_path,
            'complete_geometry_sha256': self.artifact_sha256,
            'complete_geometry_file_bytes': self.file_bytes,
            'complete_geometry_vertex_count': self.vertex_count,
            'bounds_min': list(self.bounds_min), 'bounds_max': list(self.bounds_max),
            'final_dense_result_admitted': False,
            'persistent_dense_scene_created': False,
            'full_density_gpu_viewer_enabled': False,
            'production_one_start_dense_wiring': False,
            'metric_scale_claimed': False,
            'measurement_authoritative': False,
            'certified_metrology_claimed': False,
            'camera_pose_authority_owner': 'pycolmap_sfm',
        }


def prepare_dense_final_handoff(
    *,
    admitted_sparse_handoff: OfflineReconstructionResultHandoff,
    source_sfm_result: OfflineSfmResult,
    request: DenseProviderSelectionRequest,
    decision: DenseProviderSelectionDecision,
    envelope: ProgressiveDenseExecutionEnvelope,
    provider_output_root: str | Path,
    expected_artifact_sha256: str,
) -> QualifiedDenseFinalHandoffCandidate:
    """Fail-closed independent identity/artifact validation; no writes or authority promotion.

    A production caller must supply an already accepted I6.2C sparse handoff,
    the matching *disk* SfM source and the completed *concrete* provider envelope.
    Provider output belongs to a per-run directory under that admitted output.
    The expected PLY hash must be captured from the completed provider result,
    independently of the provisional visualization stream.
    """
    _require(isinstance(admitted_sparse_handoff, OfflineReconstructionResultHandoff),
             'an admitted offline sparse handoff is required')
    _require(isinstance(source_sfm_result, OfflineSfmResult), 'typed sparse SfM result is required')
    _require(isinstance(request, DenseProviderSelectionRequest), 'typed provider request is required')
    _require(isinstance(decision, DenseProviderSelectionDecision), 'typed provider decision is required')
    _require(isinstance(envelope, ProgressiveDenseExecutionEnvelope), 'completed provider envelope is required')
    _require(admitted_sparse_handoff.availability is OfflineResultHandoffAvailability.PARTIAL,
             'an admitted partial sparse result is required')
    _require(admitted_sparse_handoff.status == 'completed' and admitted_sparse_handoff.camera_pose_available,
             'admitted sparse camera-pose result is required')
    _require(not admitted_sparse_handoff.canonical_geometry_authoritative
             and not admitted_sparse_handoff.persistent_scene_ready,
             'sparse handoff cannot already assert canonical/persistent dense geometry')
    _require(admitted_sparse_handoff.scale_status == 'arbitrary' and not admitted_sparse_handoff.certified_metrology,
             'only arbitrary-scale, non-metrological sparse handoff is accepted')
    _require(source_sfm_result.gauge.scale_status is SfmScaleStatus.ARBITRARY
             and source_sfm_result.gauge.unit_to_metre is None,
             'source SfM scale requires separate calibration authority')
    _require(source_sfm_result.session_id == admitted_sparse_handoff.session_id,
             'sparse session identity mismatch')
    _require(source_sfm_result.backend_id == PYCOLMAP_BACKEND_ID,
             'PyCOLMAP sparse camera authority required')
    _require(source_sfm_result.statistics.registered_keyframes >= 3,
             'sparse SfM requires at least three registered camera views')

    root = Path(admitted_sparse_handoff.output_dir).expanduser()
    _require(root.is_absolute() and root.is_dir() and not root.is_symlink(),
             'admitted offline output root must be a real directory')
    root = root.resolve(strict=True)
    result_path = root / 'offline_result.json'
    sfm_path = root / 'sfm_result.json'
    _require(admitted_sparse_handoff.result_path == str(result_path),
             'admitted offline result path does not match output root')
    disk_result = _json_regular_file(result_path, label='offline result')
    _require(disk_result.get('session_id') == source_sfm_result.session_id
             and disk_result.get('pass') is True and disk_result.get('status') == 'completed',
             'offline result session/completion mismatch')
    _require(disk_result.get('camera_pose_available') is True
             and disk_result.get('canonical_geometry_authoritative') is False
             and disk_result.get('persistent_scene_ready') is False
             and disk_result.get('certified_metrology') is False,
             'offline sparse result claims unsupported authority')
    artifacts = disk_result.get('artifacts')
    _require(isinstance(artifacts, dict) and artifacts.get('sfm_result_json') == str(sfm_path),
             'offline result must bind its exact SfM result artifact')
    summary = disk_result.get('sfm_summary')
    _require(isinstance(summary, dict)
             and summary.get('registered_keyframes') == source_sfm_result.statistics.registered_keyframes
             and isinstance(summary.get('gauge'), dict)
             and summary['gauge'].get('scale_status') == 'arbitrary',
             'admitted sparse summary/gauge mismatch')
    persisted_sfm = offline_sfm_result_from_dict(_json_regular_file(sfm_path, label='SfM result'))
    source_sha = canonical_offline_sfm_result_sha256(source_sfm_result)
    _require(persisted_sfm.session_id == admitted_sparse_handoff.session_id
             and canonical_offline_sfm_result_sha256(persisted_sfm) == source_sha,
             'SfM persisted identity/content mismatch')

    _require(request.mode is DenseProviderSelectionMode.EXPLICIT_PROVIDER
             and decision.request_mode is DenseProviderSelectionMode.EXPLICIT_PROVIDER
             and request.authoritative_sfm_available is True
             and request.allow_fallback is False,
             'explicit provider without fallback and admitted SfM is required')
    _require(not decision.automatic and not decision.fallback_used
             and not decision.engineer_override_applied,
             'Auto/fallback/override provider selection is not authorized')
    _require(decision.requested_provider_id == request.requested_provider_id
             == decision.selected_provider_id
             and bool(decision.selected_provider_id),
             'selected concrete provider identity mismatch')
    _require(request.available_view_count == source_sfm_result.statistics.registered_keyframes,
             'provider view population differs from admitted sparse SfM')
    _require(request.authoritative_sfm_frame_id
             and request.authoritative_sfm_frame_id == envelope.result.canonical_frame_id,
             'PyCOLMAP reconstruction frame identity mismatch')

    result = envelope.result
    report = envelope.report
    _require(decision.selected_implementation is DenseProviderImplementation.COLMAP_MVS
             and result.implementation is DenseProviderImplementation.COLMAP_MVS,
             'first qualified artifact admission supports COLMAP MVS only')
    _require(result.provider_id == decision.selected_provider_id, 'provider result identity mismatch')
    _require(result.status is DenseProviderRunStatus.COMPLETED and not result.degradation_reasons,
             'completed non-degraded provider result required; degraded needs separate review')
    _require(result.canonical_geometry_authoritative and result.single_authoritative_sfm_frame_preserved
             and result.camera_pose_authority_owner == 'pycolmap_sfm',
             'completed provider must preserve PyCOLMAP camera authority')
    _require(not any((result.metric_scale_claimed, result.measurement_authoritative,
                      result.certified_metrology, result.surface_mesh_generated,
                      result.production_filter_policy_changed, result.provider_promotion_performed)),
             'provider claims unauthorized metric/measurement/mesh/policy authority')
    _require(result.source_view_count == source_sfm_result.statistics.registered_keyframes
             and 0 < result.selected_view_count <= result.source_view_count,
             'completed provider view population mismatch')
    _require(report.source_sparse_result_sha256 == source_sha,
             'progress report sparse source identity mismatch')
    _require(report.request_fingerprint_sha256 == dense_request_fingerprint_sha256(
        request, decision, source_sha, request.authoritative_sfm_frame_id),
        'progress report request fingerprint mismatch')
    _require(not any((report.provider_execution_result_admitted_by_progressive_path,
                      report.persistent_dense_scene_created, report.production_auto_activated,
                      report.metric_scale_claimed, report.measurement_authoritative,
                      report.certified_metrology_claimed, report.surface_mesh_generated)),
             'provisional progress cannot admit or promote final authority')
    try:
        run_uuid = str(UUID(report.dense_run_id))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ReconstructionValidationError('I6.7E2A final handoff: invalid dense run UUID') from exc
    _require(run_uuid == report.dense_run_id, 'dense run ID is not canonical')
    run_root = root / 'dense_runs' / run_uuid / 'provider_colmap_mvs'
    provider_dir = Path(provider_output_root).expanduser()
    _require(provider_dir.is_absolute() and provider_dir == run_root
             and provider_dir.is_dir() and not provider_dir.is_symlink()
             and not (root / 'dense_runs').is_symlink()
             and not (root / 'dense_runs' / run_uuid).is_symlink()
             and provider_dir.resolve(strict=True).is_relative_to(root),
             'provider output root differs from admitted sparse run output root')
    geometry = Path(result.canonical_geometry_ref).expanduser()
    _require(geometry.is_absolute() and geometry == run_root / 'dense' / 'fused.ply',
             'complete dense geometry escaped the selected provider output root')
    _require(not geometry.is_symlink() and not geometry.parent.is_symlink()
             and geometry.is_file(),
             'complete dense geometry must be a regular non-symlinked artifact')
    expected_sha = require_sha256(expected_artifact_sha256,
                                  field_name='expected_artifact_sha256', allow_empty=False)
    scanned = inspect_full_dense_ply(geometry, expected_sha256=expected_sha,
                                     expected_root=run_root,
                                     expected_vertex_count=None)
    _require(scanned.has_rgb, 'complete dense PLY requires RGB for colored dense viewing')
    _require(scanned.vertex_count > 0, 'complete dense geometry contains no vertices')
    return QualifiedDenseFinalHandoffCandidate(
        sparse_session_id=admitted_sparse_handoff.session_id,
        source_sparse_result_sha256=source_sha,
        reconstruction_frame_id=request.authoritative_sfm_frame_id,
        dense_run_id=run_uuid, provider_id=result.provider_id,
        implementation=result.implementation.value, geometry_path=scanned.path,
        artifact_sha256=scanned.sha256, file_bytes=scanned.file_bytes,
        vertex_count=scanned.vertex_count, bounds_min=scanned.bounds_min,
        bounds_max=scanned.bounds_max,
    )


__all__ = ['I67E2A_PHASE', 'QualifiedDenseFinalHandoffCandidate', 'prepare_dense_final_handoff']

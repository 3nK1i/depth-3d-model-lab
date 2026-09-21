"""I6.8B: factual SfM/MVS stage observations from existing worker events.

Observations are passive, run-scoped, and latest-only when written to disk.
Legacy workflow fractions, callback counts, fusion images, estimated time, and
terminal/admission claims are NEVER inferred by this source adapter.
"""
from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import tempfile
from uuid import UUID

from core.reconstruction.reconstruction_job_telemetry_contract import (
    JOB_TELEMETRY_MAX_LINE_BYTES, JobTelemetryObservation, StageWorkCounter,
    serialize_job_telemetry_line,
)
from core.reconstruction.sfm_worker_protocol import SfmWorkerEvent, SfmWorkerEventType
from core.reconstruction.reconstruction_progressive_dense_contract import ProgressiveDenseSnapshot


class JobProgressObservationError(ValueError):
    """An input cannot be claimed as an observed stage for this run."""


_SFM_STAGES = {
    'feature_extraction': ('sfm_feature_extraction', 'PyCOLMAP feature extraction reported'),
    'matching': ('sfm_matching', 'PyCOLMAP feature matching reported'),
    'mapping': ('sfm_mapping', 'PyCOLMAP incremental mapping reported'),
    'conversion': ('sfm_finalizing', 'Sparse model conversion reported; admission pending'),
    'complete': ('sfm_finalizing', 'Sparse worker reported completion; offline result pending'),
}
_DENSE_STAGES = {
    'external_worker_launch': ('dense_preparing', 'Isolated MVS worker launch reported'),
    'image_undistortion': ('dense_undistortion', 'MVS image undistortion reported'),
    'patch_match_stereo': ('dense_patch_match', 'Geometric depth-map files observed in this MVS run'),
    'stereo_fusion': ('dense_fusion', 'Stereo fusion entered; remaining work not quantified'),
    'external_worker_complete': ('dense_verification', 'MVS worker reported completion; independent admission pending'),
}


def _uuid(value: str, label: str) -> str:
    if not isinstance(value, str):
        raise JobProgressObservationError(f'{label} must be a canonical UUID')
    try:
        if str(UUID(value)) != value:
            raise ValueError('non-canonical UUID')
    except ValueError as exc:
        raise JobProgressObservationError(f'{label} must be a canonical UUID') from exc
    return value


class JobProgressObserver:
    """One producer, one job: stage-only observations, not terminal signals."""

    def __init__(self, job_kind: str, source_session_id: str, dense_run_id: str | None = None):
        self.source_session_id = _uuid(source_session_id, 'source_session_id')
        if job_kind not in {'offline_sfm', 'post_sparse_dense'}:
            raise JobProgressObservationError('unknown job kind')
        if job_kind == 'post_sparse_dense':
            self.dense_run_id = _uuid(dense_run_id, 'dense_run_id')
        elif dense_run_id is not None:
            raise JobProgressObservationError('offline_sfm cannot bind a dense_run_id')
        else:
            self.dense_run_id = None
        self.job_kind = job_kind
        self._sequence = 0

    def _event(self, stage: str, detail: str, counter: StageWorkCounter | None = None):
        self._sequence += 1
        return JobTelemetryObservation(
            source_session_id=self.source_session_id, job_kind=self.job_kind,
            dense_run_id=self.dense_run_id,
            producer='sfm_worker' if self.job_kind == 'offline_sfm' else 'dense_supervisor',
            sequence=self._sequence,
            observed_at_utc=datetime.now(timezone.utc).isoformat(),
            stage=stage, detail=detail, counter=counter, resource_readings=(),
        )

    def from_sfm_worker_event(self, event: SfmWorkerEvent) -> JobTelemetryObservation | None:
        if self.job_kind != 'offline_sfm' or not isinstance(event, SfmWorkerEvent):
            raise JobProgressObservationError('SfM worker event requires bound offline_sfm observer')
        if event.event_type is SfmWorkerEventType.WARNING:
            return None  # warnings are not evidence that a stage advanced
        descriptor = _SFM_STAGES.get(event.stage)
        if descriptor is None:
            return None
        # Not using event.progress_fraction, initial_pair_callbacks, or
        # registered_callbacks: those are not confirmed distinct registered views.
        return self._event(*descriptor)

    def from_confirmed_sparse_view_count(self, registered_views: int, selected_views: int) -> JobTelemetryObservation:
        if self.job_kind != 'offline_sfm':
            raise JobProgressObservationError('registered views require an offline_sfm observer')
        if (type(registered_views) is not int or type(selected_views) is not int or
                selected_views < 1 or not 0 <= registered_views <= selected_views):
            raise JobProgressObservationError('confirmed sparse view count exceeds selected views')
        return self._event('sfm_mapping',
                           'Sparse backend reported registered views; offline result admission pending',
                           StageWorkCounter('registered_views', registered_views, selected_views))

    def from_dense_snapshot(self, snapshot: ProgressiveDenseSnapshot) -> JobTelemetryObservation | None:
        if self.job_kind != 'post_sparse_dense' or not isinstance(snapshot, ProgressiveDenseSnapshot):
            raise JobProgressObservationError('dense source requires validated progressive dense snapshot')
        if (snapshot.source_sfm_session_id != self.source_session_id or
                snapshot.dense_run_id != self.dense_run_id):
            raise JobProgressObservationError('foreign dense session or run')
        descriptor = _DENSE_STAGES.get(snapshot.stage_detail)
        if descriptor is None:
            return None
        counter = None
        if snapshot.stage_detail == 'patch_match_stereo':
            # Existing provider polls actual *.geometric.bin files.  It reports
            # their presence, NOT that PatchMatch or total job is N/M complete.
            counter = StageWorkCounter(
                'geometric_depth_maps_observed',
                snapshot.statistics.processed_views,
                snapshot.statistics.selected_views,
            )
        # In particular stereo_fusion has no counter: processed_views may still
        # contain the PatchMatch map count and is NOT a fusion-image count.
        return self._event(*descriptor, counter=counter)


class LatestJobProgressSnapshot:
    """Bounded diagnostic sidecar for a live worker; never admission/persistence.

    Qt transport and stale-session filtering are introduced later in I6.8D.
    Only one bounded observation is retained, not an unbounded event history.
    All filesystem failures are fail-soft: progress cannot invalidate SfM/MVS.
    """

    def __init__(self, path: Path):
        self.path = Path(path)

    def publish(self, event: JobTelemetryObservation | None) -> bool:
        if event is None:
            return False
        tmp = None
        try:
            line = serialize_job_telemetry_line(event) + '\n'
            if len(line.encode('utf-8')) > JOB_TELEMETRY_MAX_LINE_BYTES + 1:
                return False
            # A diagnostic snapshot is written only to a pre-existing run dir.
            if not self.path.parent.is_dir() or self.path.parent.is_symlink():
                return False
            with tempfile.NamedTemporaryFile(
                mode='w', encoding='utf-8', prefix='.i68b_progress_',
                dir=self.path.parent, delete=False,
            ) as handle:
                tmp = Path(handle.name)
                handle.write(line)
            os.replace(tmp, self.path)
            return True
        except OSError:
            return False
        finally:
            if tmp is not None:
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass


__all__ = ['JobProgressObservationError', 'JobProgressObserver', 'LatestJobProgressSnapshot']

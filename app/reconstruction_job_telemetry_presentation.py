"""I6.8E passive job-scoped presentation of *observed* reconstruction telemetry.

Never decides job completion, stage percentage, admission, scale or ETA. Only
approved I6.8A observations for an explicitly bound active job can be shown.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from core.reconstruction.reconstruction_job_telemetry_contract import JobTelemetryObservation
from core.reconstruction.reconstruction_job_telemetry_transport import TELEMETRY_FRESH_SECONDS

FRESH_SECONDS = TELEMETRY_FRESH_SECONDS

_STAGE_TITLES = {
    'source_ingestion': 'Source ingestion',
    'sfm_feature_extraction': 'SfM feature extraction',
    'sfm_matching': 'SfM matching',
    'sfm_mapping': 'SfM mapping',
    'sfm_finalizing': 'SfM finalizing',
    'dense_preparing': 'Dense preparation',
    'dense_undistortion': 'Image undistortion',
    'dense_patch_match': 'PatchMatch stereo',
    'dense_fusion': 'Stereo fusion',
    'dense_verification': 'Final dense verification',
}
_COUNTER_TITLES = {
    'registered_views': 'Registered views',
    'geometric_depth_maps_observed': 'Geometric depth maps observed',
    'fusion_images_started': 'Fusion images started (not completed)',
    'fusion_images_completed': 'Fusion images completed',
}


@dataclass(frozen=True)
class JobTelemetryPanel:
    heading: str
    stage: str
    counter: str
    process: str
    host: str
    gpu: str
    detail: str
    status: str


def _gib(value: int | float) -> str:
    return f'{float(value) / (1024**3):.2f} GiB'


def _metric(readings, name: str, index: int | None = None) -> str:
    for item in readings:
        if item.name == name and item.gpu_index == index:
            if item.availability != 'observed' or item.value is None:
                return 'unavailable'
            return _gib(item.value) if item.unit == 'bytes' else f'{item.value:.1f}%'
    return 'unavailable'


class JobTelemetryPanelState:
    """Small latest-only presentation cache; never stores geometry or history."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.identity: tuple[str, str, str | None] | None = None
        self.progress: JobTelemetryObservation | None = None
        self.resources: JobTelemetryObservation | None = None
        self.terminal: str | None = None

    def begin(self, kind: str, session_id: str, run_id: str | None) -> None:
        self.reset()
        if session_id and (kind == 'offline_sfm' and run_id is None or
                           kind == 'post_sparse_dense' and run_id):
            self.identity = (kind, session_id, run_id)

    def end(self, label: str) -> None:
        self.terminal = label
        self.progress = None
        self.resources = None

    def accept(self, event: object) -> bool:
        if (not isinstance(event, JobTelemetryObservation) or
                self.identity is None or self.terminal is not None or
                (event.job_kind, event.source_session_id, event.dense_run_id) != self.identity):
            return False
        if event.producer == 'resource_sampler':
            if self.resources is not None and event.sequence <= self.resources.sequence:
                return False
            self.resources = event
        else:
            if self.progress is not None and event.sequence <= self.progress.sequence and event.producer == self.progress.producer:
                return False
            self.progress = event
        return True

    @staticmethod
    def _fresh(event: JobTelemetryObservation | None, now: datetime) -> bool:
        if event is None:
            return False
        try:
            instant = datetime.fromisoformat(event.observed_at_utc.replace('Z', '+00:00'))
            age = (now - instant).total_seconds()
            return -5.0 <= age <= FRESH_SECONDS
        except (TypeError, ValueError, OverflowError):
            return False

    def present(self, now: datetime | None = None) -> JobTelemetryPanel:
        now = now or datetime.now(timezone.utc)
        if self.identity is None:
            return JobTelemetryPanel('Job telemetry • idle', 'No active offline reconstruction',
                                     'Observed work: —', 'Job CPU / RAM: —',
                                     'Host RAM / disk: —', 'GPU device: —',
                                     'Telemetry appears only during an active SfM or dense job.', 'idle')
        heading = ('Offline SfM' if self.identity[0] == 'offline_sfm' else 'Dense MVS')
        if self.terminal is not None:
            return JobTelemetryPanel(f'{heading} telemetry • {self.terminal}',
                                     'Live samples ended', 'Observed work: —',
                                     'Job CPU / RAM: —', 'Host RAM / disk: —',
                                     'GPU device: —', 'Past readings are not current.', 'terminal')
        progress = self.progress if self._fresh(self.progress, now) else None
        resources = self.resources if self._fresh(self.resources, now) else None
        if progress:
            stage = 'Observed stage: ' + _STAGE_TITLES.get(progress.stage, 'Unknown stage')
            counter = 'Observed work: —'
            if progress.counter is not None:
                item = progress.counter
                amount = f'{item.observed_count:,}'
                if item.expected_count is not None:
                    amount += f' / {item.expected_count:,}'
                counter = f'{_COUNTER_TITLES[item.unit]}: {amount} (not overall progress)'
            detail = progress.detail or 'Observed stage only; no ETA is inferred.'
        else:
            stage = 'Observed stage: stale' if self.progress else 'Observed stage: awaiting sample'
            counter = 'Observed work: —'
            detail = 'Last stage observation is stale; no live progress claim.' if self.progress else 'Waiting for a real worker observation.'
        if resources:
            metrics = resources.resource_readings
            cpu = _metric(metrics, 'process_tree_cpu_percent')
            rss = _metric(metrics, 'process_tree_rss_bytes')
            if cpu == 'unavailable' and rss == 'unavailable':
                cpu = _metric(metrics, 'process_cpu_percent')
                rss = _metric(metrics, 'process_rss_bytes')
                scope = 'Worker process'
            else:
                scope = 'Worker tree'
            process = f'{scope}: CPU {cpu} • RAM {rss}'
            host = (f'Host: RAM available {_metric(metrics, "system_ram_available_bytes")} • '
                    f'workspace disk free {_metric(metrics, "workspace_disk_free_bytes")}')
            gpu_items = sorted({item.gpu_index for item in metrics if item.scope == 'gpu_device'})
            gpu = 'GPU device: unavailable' if not gpu_items else ' • '.join(
                f'GPU {index} device-wide: {_metric(metrics, "gpu_device_util_percent", index)} util, '
                f'{_metric(metrics, "gpu_device_vram_used_bytes", index)} VRAM used'
                for index in gpu_items)
        else:
            adjective = 'stale' if self.resources else 'awaiting sample'
            process = f'Job CPU / RAM: {adjective}'
            host = f'Host RAM / disk: {adjective}'
            gpu = f'GPU device: {adjective}'
        return JobTelemetryPanel(f'{heading} telemetry • observed only', stage, counter,
                                 process, host, gpu, detail, 'current' if progress or resources else 'waiting')


__all__ = ['JobTelemetryPanel', 'JobTelemetryPanelState', 'FRESH_SECONDS']

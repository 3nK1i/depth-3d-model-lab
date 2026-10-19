"""I6.8A job-scoped observational telemetry contract (not admission or control).

No hardware probes, Qt imports, SfM/geometry changes or process management live
here. Later I6.8 steps bind *observed* stage and resource samples to these
bounded messages; existing worker completion/verifier signals retain authority.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from math import isfinite
from uuid import UUID


JOB_TELEMETRY_SCHEMA = "depth3d.reconstruction.job_telemetry.v1"
JOB_TELEMETRY_PREFIX = "DEPTH3D_JOB_TELEMETRY "
JOB_TELEMETRY_MAX_LINE_BYTES = 4096
JOB_TELEMETRY_MAX_RESOURCES = 12

_JOB_STAGES = {
    "offline_sfm": frozenset({
        "source_ingestion", "sfm_feature_extraction", "sfm_matching",
        "sfm_mapping", "sfm_finalizing",
    }),
    "post_sparse_dense": frozenset({
        "dense_preparing", "dense_undistortion", "dense_patch_match",
        "dense_fusion", "dense_verification",
    }),
}
_PRODUCERS = frozenset({
    "offline_supervisor", "sfm_worker", "dense_supervisor", "mvs_worker",
    "resource_sampler",
})
_COUNTER_STAGES = {
    "registered_views": "sfm_mapping",
    "geometric_depth_maps_observed": "dense_patch_match",
    "fusion_images_started": "dense_fusion",
    "fusion_images_completed": "dense_fusion",
}
_METRICS = {
    "process_rss_bytes": ("process", "bytes"),
    "process_peak_rss_bytes": ("process", "bytes"),
    "process_cpu_percent": ("process", "percent"),
    "process_tree_rss_bytes": ("process_tree", "bytes"),
    "process_tree_cpu_percent": ("process_tree", "percent"),
    "system_ram_available_bytes": ("system", "bytes"),
    "workspace_disk_free_bytes": ("system", "bytes"),
    "gpu_device_vram_used_bytes": ("gpu_device", "bytes"),
    "gpu_device_util_percent": ("gpu_device", "percent"),
    "gpu_process_vram_used_bytes": ("gpu_process", "bytes"),
}
_EVENT_FIELDS = frozenset({
    "schema", "source_session_id", "job_kind", "dense_run_id", "producer",
    "sequence", "observed_at_utc", "stage", "detail", "counter", "resource_readings",
})


class TelemetryValidationError(ValueError):
    """Bad telemetry must be dropped without changing reconstruction authority."""


def _uuid(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise TelemetryValidationError(f"{label} must be a canonical UUID")
    try:
        parsed = str(UUID(value))
    except (ValueError, TypeError, AttributeError) as exc:
        raise TelemetryValidationError(f"{label} must be a canonical UUID") from exc
    if parsed != value:
        raise TelemetryValidationError(f"{label} must be a canonical UUID")
    return value


def _integer(value: object, label: str, *, positive: bool = False) -> int:
    if type(value) is not int or value < (1 if positive else 0) or value > 2**63 - 1:
        raise TelemetryValidationError(f"{label} must be a bounded nonnegative integer")
    return value


def _utc(value: object) -> str:
    if not isinstance(value, str) or len(value) > 45:
        raise TelemetryValidationError("observed_at_utc must be a UTC ISO8601 timestamp")
    try:
        instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise TelemetryValidationError("observed_at_utc must be a UTC ISO8601 timestamp") from exc
    if instant.tzinfo is None or instant.utcoffset() != timedelta(0):
        raise TelemetryValidationError("observed_at_utc must use the UTC timezone")
    return value


def _fields(data: object, allowed: frozenset[str], required: frozenset[str], label: str) -> dict:
    if not isinstance(data, dict) or not required.issubset(data) or set(data) != set(data) & allowed:
        raise TelemetryValidationError(f"{label} has missing or unexpected fields")
    return data


@dataclass(frozen=True, slots=True)
class StageWorkCounter:
    """Observed units; never duration/overall-percent/ETA evidence."""

    unit: str
    observed_count: int
    expected_count: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.unit, str) or self.unit not in _COUNTER_STAGES:
            raise TelemetryValidationError("unknown stage-work unit")
        _integer(self.observed_count, "observed_count")
        if self.expected_count is not None:
            _integer(self.expected_count, "expected_count", positive=True)
            if self.observed_count > self.expected_count:
                raise TelemetryValidationError("observed_count exceeds expected_count")

    def as_dict(self) -> dict:
        return dict(unit=self.unit, observed_count=self.observed_count,
                    expected_count=self.expected_count)

    @classmethod
    def from_dict(cls, payload: object) -> "StageWorkCounter":
        data = _fields(payload, frozenset({"unit", "observed_count", "expected_count"}),
                       frozenset({"unit", "observed_count", "expected_count"}), "counter")
        return cls(**data)


@dataclass(frozen=True, slots=True)
class ResourceReading:
    """An observed metric whose scope is dictated by its name, not an untrusted label."""

    name: str
    value: float | int | None
    availability: str
    gpu_index: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or self.name not in _METRICS:
            raise TelemetryValidationError("unknown resource metric")
        if not isinstance(self.availability, str) or self.availability not in {"observed", "unavailable"}:
            raise TelemetryValidationError("resource availability must be observed or unavailable")
        if self.availability == "unavailable":
            if self.value is not None:
                raise TelemetryValidationError("unavailable resource value must be null, not zero")
        elif isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise TelemetryValidationError("observed resource value must be numeric")
        elif (isinstance(self.value, int) and self.value > 2**63 - 1):
            raise TelemetryValidationError("resource value exceeds telemetry numeric bound")
        elif not isfinite(self.value) or self.value < 0:
            raise TelemetryValidationError("resource value must be finite and nonnegative")
        elif self.name == "gpu_device_util_percent" and self.value > 100:
            raise TelemetryValidationError("GPU utilization must not exceed 100 percent")
        scope, unit = _METRICS[self.name]
        if self.availability == "observed" and unit == "bytes":
            if type(self.value) is not int or self.value > 2**63 - 1:
                raise TelemetryValidationError("observed byte counts must be bounded integers")
        if scope.startswith("gpu_"):
            _integer(self.gpu_index, "gpu_index")
        elif self.gpu_index is not None:
            raise TelemetryValidationError("non-GPU metrics cannot carry a gpu_index")

    @property
    def scope(self) -> str:
        return _METRICS[self.name][0]

    @property
    def unit(self) -> str:
        return _METRICS[self.name][1]

    def as_dict(self) -> dict:
        return dict(name=self.name, value=self.value, availability=self.availability,
                    gpu_index=self.gpu_index, scope=self.scope, unit=self.unit)

    @classmethod
    def from_dict(cls, payload: object) -> "ResourceReading":
        data = _fields(payload,
                       frozenset({"name", "value", "availability", "gpu_index", "scope", "unit"}),
                       frozenset({"name", "value", "availability", "gpu_index", "scope", "unit"}),
                       "resource")
        obj = cls(data["name"], data["value"], data["availability"], data["gpu_index"])
        if data["scope"] != obj.scope or data["unit"] != obj.unit:
            raise TelemetryValidationError("resource scope/unit mismatch")
        return obj


@dataclass(frozen=True, slots=True)
class JobTelemetryObservation:
    """A passive stage/resource snapshot, not a terminal result or admission claim."""

    source_session_id: str
    job_kind: str
    dense_run_id: str | None
    producer: str
    sequence: int
    observed_at_utc: str
    stage: str
    detail: str
    counter: StageWorkCounter | None = None
    resource_readings: tuple[ResourceReading, ...] = ()

    def __post_init__(self) -> None:
        _uuid(self.source_session_id, "source_session_id")
        if not isinstance(self.job_kind, str) or self.job_kind not in _JOB_STAGES:
            raise TelemetryValidationError("unknown job_kind")
        if self.job_kind == "post_sparse_dense":
            _uuid(self.dense_run_id, "dense_run_id")
        elif self.dense_run_id is not None:
            raise TelemetryValidationError("offline_sfm cannot carry a dense_run_id")
        if not isinstance(self.producer, str) or self.producer not in _PRODUCERS:
            raise TelemetryValidationError("unknown producer")
        if (self.job_kind == "offline_sfm" and self.producer in {"dense_supervisor", "mvs_worker"}) or (
                self.job_kind == "post_sparse_dense" and self.producer in {"offline_supervisor", "sfm_worker"}):
            raise TelemetryValidationError("producer is invalid for job_kind")
        _integer(self.sequence, "sequence", positive=True)
        _utc(self.observed_at_utc)
        if not isinstance(self.stage, str) or self.stage not in _JOB_STAGES[self.job_kind]:
            raise TelemetryValidationError("stage is invalid for job_kind")
        if not isinstance(self.detail, str) or len(self.detail) > 160 or "\n" in self.detail or "\r" in self.detail:
            raise TelemetryValidationError("detail must be a short single-line message")
        if self.counter is not None:
            if not isinstance(self.counter, StageWorkCounter) or _COUNTER_STAGES[self.counter.unit] != self.stage:
                raise TelemetryValidationError("counter is invalid for stage")
        if not isinstance(self.resource_readings, tuple) or len(self.resource_readings) > JOB_TELEMETRY_MAX_RESOURCES:
            raise TelemetryValidationError("resource_readings exceeds bounded telemetry capacity")
        keys: set[tuple[str, int | None]] = set()
        for reading in self.resource_readings:
            if not isinstance(reading, ResourceReading):
                raise TelemetryValidationError("resource_readings must contain ResourceReading objects")
            key = (reading.name, reading.gpu_index)
            if key in keys:
                raise TelemetryValidationError("duplicate resource reading")
            keys.add(key)

    def as_dict(self) -> dict:
        return dict(schema=JOB_TELEMETRY_SCHEMA, source_session_id=self.source_session_id,
                    job_kind=self.job_kind, dense_run_id=self.dense_run_id,
                    producer=self.producer, sequence=self.sequence,
                    observed_at_utc=self.observed_at_utc, stage=self.stage,
                    detail=self.detail,
                    counter=None if self.counter is None else self.counter.as_dict(),
                    resource_readings=[x.as_dict() for x in self.resource_readings])

    @classmethod
    def from_dict(cls, payload: object) -> "JobTelemetryObservation":
        data = _fields(payload, _EVENT_FIELDS, _EVENT_FIELDS, "telemetry observation")
        if data["schema"] != JOB_TELEMETRY_SCHEMA:
            raise TelemetryValidationError("unknown job telemetry schema")
        items = data["resource_readings"]
        if not isinstance(items, list) or len(items) > JOB_TELEMETRY_MAX_RESOURCES:
            raise TelemetryValidationError("invalid bounded resource_readings array")
        return cls(
            source_session_id=data["source_session_id"], job_kind=data["job_kind"],
            dense_run_id=data["dense_run_id"], producer=data["producer"],
            sequence=data["sequence"], observed_at_utc=data["observed_at_utc"],
            stage=data["stage"], detail=data["detail"],
            counter=None if data["counter"] is None else StageWorkCounter.from_dict(data["counter"]),
            resource_readings=tuple(ResourceReading.from_dict(item) for item in items),
        )


def serialize_job_telemetry_line(event: JobTelemetryObservation) -> str:
    if not isinstance(event, JobTelemetryObservation):
        raise TelemetryValidationError("telemetry line requires JobTelemetryObservation")
    line = JOB_TELEMETRY_PREFIX + json.dumps(event.as_dict(), sort_keys=True,
                                              separators=(",", ":"), allow_nan=False)
    if len(line.encode("utf-8")) > JOB_TELEMETRY_MAX_LINE_BYTES:
        raise TelemetryValidationError("telemetry line exceeds byte limit")
    return line


def parse_job_telemetry_line(line: str) -> JobTelemetryObservation | None:
    if not isinstance(line, str):
        raise TelemetryValidationError("telemetry input must be a string")
    if not line.startswith(JOB_TELEMETRY_PREFIX):
        return None
    if len(line.encode("utf-8")) > JOB_TELEMETRY_MAX_LINE_BYTES:
        raise TelemetryValidationError("telemetry line exceeds byte limit")
    try:
        payload = json.loads(line[len(JOB_TELEMETRY_PREFIX):])
    except ValueError as exc:
        raise TelemetryValidationError("malformed job telemetry JSON") from exc
    return JobTelemetryObservation.from_dict(payload)


class JobTelemetryStreamGate:
    """Qt-free run binding and per-producer ordering; transport boundedness is I6.8D."""

    def __init__(self, job_kind: str, source_session_id: str, dense_run_id: str | None):
        self.reset(job_kind, source_session_id, dense_run_id)

    def reset(self, job_kind: str, source_session_id: str, dense_run_id: str | None) -> None:
        if not isinstance(job_kind, str) or job_kind not in _JOB_STAGES:
            raise TelemetryValidationError("unknown job_kind")
        _uuid(source_session_id, "source_session_id")
        if job_kind == "post_sparse_dense":
            _uuid(dense_run_id, "dense_run_id")
        elif dense_run_id is not None:
            raise TelemetryValidationError("offline_sfm must not bind dense_run_id")
        self.job_kind = job_kind
        self.source_session_id = source_session_id
        self.dense_run_id = dense_run_id
        self._last_sequence: dict[str, int] = {}
        self.closed = False
        self.accepted_count = 0
        self.rejected_count = 0

    def offer(self, event: JobTelemetryObservation) -> bool:
        if not isinstance(event, JobTelemetryObservation):
            raise TelemetryValidationError("stream gate only accepts validated observations")
        if (self.closed or event.job_kind != self.job_kind or
                event.source_session_id != self.source_session_id or
                event.dense_run_id != self.dense_run_id or
                event.sequence <= self._last_sequence.get(event.producer, 0)):
            self.rejected_count += 1
            return False
        self._last_sequence[event.producer] = event.sequence
        self.accepted_count += 1
        return True

    def close(self) -> None:
        self.closed = True


__all__ = [
    "JOB_TELEMETRY_SCHEMA", "JOB_TELEMETRY_PREFIX", "JOB_TELEMETRY_MAX_LINE_BYTES",
    "JOB_TELEMETRY_MAX_RESOURCES", "TelemetryValidationError", "StageWorkCounter",
    "ResourceReading", "JobTelemetryObservation", "JobTelemetryStreamGate",
    "serialize_job_telemetry_line", "parse_job_telemetry_line",
]

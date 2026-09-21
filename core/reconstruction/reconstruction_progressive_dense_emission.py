"""I6.7B dense-provider execution and provisional dense emission.

This module bridges the already-qualified I4.3 provider adapters into the
I6.7A provisional dense-snapshot contract without modifying provider selection,
Auto policy, provider scientific parameters, final dense admission, persistence,
UI, or rendering.

Key properties:

* provider adapters still own scientific execution and final canonical output;
* existing ``DenseProviderProgressEvent`` values are projected into bounded
  provisional I6.7A snapshots;
* one bounded actual dense geometry preview is emitted after a successful or
  degraded adapter result is constructed but before that result leaves this
  execution wrapper for any later admission/persistence layer;
* snapshot-sink/preview failures are fail-soft and cannot invalidate an
  otherwise valid dense reconstruction;
* revision gaps are legal because failed/dropped visualization emissions do not
  get retried until they pass;
* preview limits are visualization-only and never act as reconstruction caps or
  quality thresholds.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Callable, Protocol
from uuid import uuid4

import numpy as np

from core.reconstruction.dense_provider_architecture import (
    DenseProviderAdapter,
    DenseProviderExecutionResult,
    DenseProviderProgressEvent,
    DenseProviderProgressSink,
    DenseProviderProgressStage,
    DenseProviderRunStatus,
    DenseProviderSelectionDecision,
    DenseProviderSelectionRequest,
)
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.reconstruction_progressive_dense_contract import (
    ProgressiveDenseGeometryPreview,
    ProgressiveDensePointSample,
    ProgressiveDenseSnapshot,
    ProgressiveDenseStage,
    ProgressiveDenseStatistics,
    validate_progressive_dense_snapshot_transition,
)
from core.reconstruction.sfm_contracts import OfflineSfmResult
from core.reconstruction.validation import require_sha256, require_uuid

I67B_PHASE = "I6.7B_DENSE_PROVIDER_EXECUTION_AND_PROGRESSIVE_EMISSION"
I67B_MAX_PREVIEW_POINTS = 4096
I67B_TARGET_STATUS_SNAPSHOTS = 24
I67B_MAX_FAILURE_RECORDS = 16
I67B_MAX_PROVISIONAL_PROGRESS = 0.999999

_PLY_SCALAR_DTYPES: dict[str, str] = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


class ProgressiveDenseSnapshotSink(Protocol):
    def publish(self, snapshot: ProgressiveDenseSnapshot) -> None: ...


DenseProviderAdapterFactory = Callable[[DenseProviderProgressSink], DenseProviderAdapter]


@dataclass(frozen=True, slots=True)
class ProgressiveDenseEmissionPolicy:
    """Visualization-only emission policy.

    These limits never alter provider input views, fusion, filtering, geometry,
    or admission.  They bound only the provisional preview/status stream.
    """

    max_preview_points: int = I67B_MAX_PREVIEW_POINTS
    target_status_snapshots: int = I67B_TARGET_STATUS_SNAPSHOTS
    max_failure_records: int = I67B_MAX_FAILURE_RECORDS

    def __post_init__(self) -> None:
        for name in ("max_preview_points", "target_status_snapshots", "max_failure_records"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or int(value) <= 0:
                raise ReconstructionValidationError(f"{name} must be a positive integer.")
            object.__setattr__(self, name, int(value))


@dataclass(frozen=True, slots=True)
class LoadedDenseGeometryPreview:
    preview: ProgressiveDenseGeometryPreview
    point_count: int
    mean_confidence: float | None
    mean_view_support: float | None


@dataclass(frozen=True, slots=True)
class ProgressiveDenseEmissionReport:
    dense_run_id: str
    source_sparse_result_sha256: str
    request_fingerprint_sha256: str
    emitted_snapshot_count: int
    emitted_geometry_snapshot_count: int
    dropped_status_snapshot_count: int
    emission_failure_count: int
    failure_records: tuple[str, ...]
    final_preview_revision: int
    geometry_snapshot_emitted: bool
    max_preview_points: int
    provider_execution_result_admitted_by_progressive_path: bool = False
    persistent_dense_scene_created: bool = False
    provider_selection_modified: bool = False
    production_auto_activated: bool = False
    metric_scale_claimed: bool = False
    measurement_authoritative: bool = False
    certified_metrology_claimed: bool = False
    surface_mesh_generated: bool = False


@dataclass(frozen=True, slots=True)
class ProgressiveDenseExecutionEnvelope:
    result: DenseProviderExecutionResult
    report: ProgressiveDenseEmissionReport


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_offline_sfm_result_sha256(result: OfflineSfmResult) -> str:
    if not isinstance(result, OfflineSfmResult):
        raise TypeError("result must be OfflineSfmResult.")
    return _canonical_sha256(result.as_dict())


def dense_request_fingerprint_sha256(
    request: DenseProviderSelectionRequest,
    decision: DenseProviderSelectionDecision,
    source_sparse_result_sha256: str,
    reconstruction_frame_id: str,
) -> str:
    if not isinstance(request, DenseProviderSelectionRequest):
        raise TypeError("request must be DenseProviderSelectionRequest.")
    if not isinstance(decision, DenseProviderSelectionDecision):
        raise TypeError("decision must be DenseProviderSelectionDecision.")
    sparse_sha = require_sha256(
        source_sparse_result_sha256,
        field_name="source_sparse_result_sha256",
        allow_empty=False,
    )
    frame_id = str(reconstruction_frame_id).strip()
    if not frame_id:
        raise ReconstructionValidationError("reconstruction_frame_id must not be empty.")
    return _canonical_sha256(
        {
            "phase": I67B_PHASE,
            "request": request.as_dict(),
            "decision": decision.as_dict(),
            "source_sparse_result_sha256": sparse_sha,
            "reconstruction_frame_id": frame_id,
        }
    )


def evenly_sample_indices(count: int, limit: int) -> tuple[int, ...]:
    if isinstance(count, bool) or int(count) != count or int(count) < 0:
        raise ReconstructionValidationError("count must be a non-negative integer.")
    if isinstance(limit, bool) or int(limit) != limit or int(limit) <= 0:
        raise ReconstructionValidationError("limit must be a positive integer.")
    count = int(count)
    limit = int(limit)
    if count == 0:
        return ()
    if count <= limit:
        return tuple(range(count))
    if limit == 1:
        return (0,)
    last = count - 1
    values = tuple((index * last) // (limit - 1) for index in range(limit))
    if len(values) != len(set(values)):
        raise RuntimeError("Deterministic dense-preview sampling produced duplicate indices.")
    return values


def _validate_dense_arrays(
    points: object,
    colors: object | None,
    confidence: object | None,
    support: object | None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    xyz = np.asarray(points, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) <= 0 or not np.isfinite(xyz).all():
        raise ReconstructionValidationError("Dense preview source must contain finite non-empty Nx3 geometry.")

    rgb: np.ndarray | None = None
    if colors is not None:
        raw = np.asarray(colors)
        if raw.shape != xyz.shape or not np.issubdtype(raw.dtype, np.number):
            raise ReconstructionValidationError("Dense RGB preview source must match Nx3 geometry.")
        numeric = np.asarray(raw, dtype=np.float64)
        if not np.isfinite(numeric).all() or np.any(numeric < 0) or np.any(numeric > 255):
            raise ReconstructionValidationError("Dense RGB preview source must be finite within [0,255].")
        rgb = np.asarray(np.rint(numeric), dtype=np.uint8)

    conf: np.ndarray | None = None
    if confidence is not None:
        conf = np.asarray(confidence, dtype=np.float64).reshape(-1)
        if conf.shape != (len(xyz),) or not np.isfinite(conf).all() or np.any(conf < 0.0) or np.any(conf > 1.0):
            raise ReconstructionValidationError("Dense confidence preview source must be finite within [0,1].")

    sup: np.ndarray | None = None
    if support is not None:
        sup = np.asarray(support, dtype=np.float64).reshape(-1)
        if sup.shape != (len(xyz),) or not np.isfinite(sup).all() or np.any(sup < 0.0):
            raise ReconstructionValidationError("Dense view-support preview source must be finite and non-negative.")

    return xyz, rgb, conf, sup


def _samples_from_arrays(
    points: np.ndarray,
    colors: np.ndarray | None,
    confidence: np.ndarray | None,
    support: np.ndarray | None,
    *,
    max_preview_points: int,
) -> ProgressiveDenseGeometryPreview:
    samples: list[ProgressiveDensePointSample] = []
    for index in evenly_sample_indices(len(points), max_preview_points):
        color = None if colors is None else tuple(int(v) for v in colors[index])
        conf = None if confidence is None else float(confidence[index])
        view_support = None
        if support is not None and support[index] > 0:
            view_support = int(round(float(support[index])))
            if view_support <= 0:
                view_support = None
        samples.append(
            ProgressiveDensePointSample(
                sample_index=index,
                position_reconstruction=points[index],
                color_rgb=color,  # type: ignore[arg-type]
                confidence=conf,
                view_support=view_support,
            )
        )
    return ProgressiveDenseGeometryPreview(point_samples=tuple(samples))


def _load_npz_dense_geometry(path: Path, *, max_preview_points: int) -> LoadedDenseGeometryPreview:
    try:
        with np.load(path, allow_pickle=False) as data:
            if "points_reconstruction" not in data:
                raise ReconstructionValidationError("Dense NPZ is missing points_reconstruction.")
            points = np.asarray(data["points_reconstruction"])
            colors = np.asarray(data["colors_rgb"]) if "colors_rgb" in data else None
            confidence = np.asarray(data["mean_confidence"]) if "mean_confidence" in data else None
            support = np.asarray(data["view_observation_counts"]) if "view_observation_counts" in data else None
    except (OSError, ValueError) as exc:
        raise ReconstructionValidationError(f"Could not read dense NPZ preview source {path}: {exc}") from exc
    xyz, rgb, conf, sup = _validate_dense_arrays(points, colors, confidence, support)
    return LoadedDenseGeometryPreview(
        preview=_samples_from_arrays(
            xyz,
            rgb,
            conf,
            sup,
            max_preview_points=max_preview_points,
        ),
        point_count=len(xyz),
        mean_confidence=None if conf is None else float(np.mean(conf)),
        mean_view_support=None if sup is None else float(np.mean(sup)),
    )


def _parse_ply_header(handle) -> tuple[str, int, list[tuple[str, str]]]:
    first = handle.readline()
    if first != b"ply\n":
        raise ReconstructionValidationError("Dense PLY must begin with a canonical ply header.")
    fmt = ""
    vertex_count: int | None = None
    current_element = ""
    vertex_properties: list[tuple[str, str]] = []
    while True:
        line = handle.readline()
        if not line:
            raise ReconstructionValidationError("Dense PLY header is truncated.")
        text = line.decode("ascii", errors="strict").strip()
        if text == "end_header":
            break
        if not text or text.startswith("comment") or text.startswith("obj_info"):
            continue
        parts = text.split()
        if parts[0] == "format" and len(parts) >= 2:
            fmt = parts[1]
        elif parts[0] == "element" and len(parts) == 3:
            current_element = parts[1]
            if current_element == "vertex":
                vertex_count = int(parts[2])
        elif parts[0] == "property" and current_element == "vertex":
            if len(parts) != 3 or parts[1] == "list":
                raise ReconstructionValidationError("Dense PLY vertex properties must be scalar.")
            vertex_properties.append((parts[2], parts[1]))
    if fmt not in {"binary_little_endian", "binary_big_endian", "ascii"}:
        raise ReconstructionValidationError(f"Dense PLY format is unsupported: {fmt!r}.")
    if vertex_count is None or vertex_count <= 0:
        raise ReconstructionValidationError("Dense PLY must contain one or more vertices.")
    names = {name for name, _ in vertex_properties}
    required = {"x", "y", "z"}
    if not required.issubset(names):
        raise ReconstructionValidationError("Dense PLY is missing x/y/z vertex properties.")
    for _, scalar_type in vertex_properties:
        if scalar_type not in _PLY_SCALAR_DTYPES:
            raise ReconstructionValidationError(f"Dense PLY uses unsupported scalar type: {scalar_type}.")
    return fmt, vertex_count, vertex_properties


def _load_ply_dense_geometry(path: Path, *, max_preview_points: int) -> LoadedDenseGeometryPreview:
    with path.open("rb") as handle:
        fmt, vertex_count, properties = _parse_ply_header(handle)
        indices = evenly_sample_indices(vertex_count, max_preview_points)
        wanted = set(indices)
        names = [name for name, _ in properties]
        has_rgb = {"red", "green", "blue"}.issubset(names)
        rows: dict[int, dict[str, float]] = {}
        if fmt == "ascii":
            for index in range(vertex_count):
                line = handle.readline()
                if not line:
                    raise ReconstructionValidationError("Dense PLY ASCII vertex payload is truncated.")
                if index not in wanted:
                    continue
                values = line.decode("ascii", errors="strict").split()
                if len(values) != len(properties):
                    raise ReconstructionValidationError("Dense PLY ASCII vertex row width is invalid.")
                rows[index] = {name: float(values[pos]) for pos, (name, _) in enumerate(properties)}
        else:
            endian = "<" if fmt == "binary_little_endian" else ">"
            dtype = np.dtype([(name, endian + _PLY_SCALAR_DTYPES[scalar_type]) for name, scalar_type in properties])
            base = handle.tell()
            for index in indices:
                handle.seek(base + index * dtype.itemsize)
                raw = handle.read(dtype.itemsize)
                if len(raw) != dtype.itemsize:
                    raise ReconstructionValidationError("Dense PLY binary vertex payload is truncated.")
                record = np.frombuffer(raw, dtype=dtype, count=1)[0]
                rows[index] = {name: float(record[name]) for name, _ in properties}

    samples: list[ProgressiveDensePointSample] = []
    for index in indices:
        row = rows.get(index)
        if row is None:
            raise ReconstructionValidationError("Dense PLY preview sampling did not recover every selected vertex.")
        xyz = np.asarray([row["x"], row["y"], row["z"]], dtype=np.float64)
        if not np.isfinite(xyz).all():
            raise ReconstructionValidationError("Dense PLY contains non-finite sampled geometry.")
        color = None
        if has_rgb:
            raw = tuple(int(round(row[name])) for name in ("red", "green", "blue"))
            if any(value < 0 or value > 255 for value in raw):
                raise ReconstructionValidationError("Dense PLY sampled RGB is outside [0,255].")
            color = raw
        samples.append(
            ProgressiveDensePointSample(
                sample_index=index,
                position_reconstruction=xyz,
                color_rgb=color,
            )
        )
    return LoadedDenseGeometryPreview(
        preview=ProgressiveDenseGeometryPreview(point_samples=tuple(samples)),
        point_count=vertex_count,
        mean_confidence=None,
        mean_view_support=None,
    )


def load_dense_geometry_preview(
    path: str | Path,
    *,
    max_preview_points: int = I67B_MAX_PREVIEW_POINTS,
) -> LoadedDenseGeometryPreview:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ReconstructionValidationError(f"Dense geometry preview source is missing: {source}")
    if isinstance(max_preview_points, bool) or int(max_preview_points) != max_preview_points or int(max_preview_points) <= 0:
        raise ReconstructionValidationError("max_preview_points must be a positive integer.")
    suffix = source.suffix.lower()
    if suffix == ".npz":
        return _load_npz_dense_geometry(source, max_preview_points=int(max_preview_points))
    if suffix == ".ply":
        return _load_ply_dense_geometry(source, max_preview_points=int(max_preview_points))
    raise ReconstructionValidationError(
        f"Unsupported dense preview geometry format {suffix!r}; expected .npz or .ply."
    )


class ProgressiveDenseEmissionSession(DenseProviderProgressSink):
    """Provider-progress sink that emits bounded I6.7A provisional snapshots."""

    def __init__(
        self,
        *,
        source_sfm_result: OfflineSfmResult,
        request: DenseProviderSelectionRequest,
        decision: DenseProviderSelectionDecision,
        reconstruction_frame_id: str,
        snapshot_sink: ProgressiveDenseSnapshotSink,
        dense_run_id: str | None = None,
        policy: ProgressiveDenseEmissionPolicy = ProgressiveDenseEmissionPolicy(),
        utc_now: Callable[[], str] | None = None,
        monotonic_ns: Callable[[], int] | None = None,
    ) -> None:
        if not isinstance(source_sfm_result, OfflineSfmResult):
            raise TypeError("source_sfm_result must be OfflineSfmResult.")
        if not isinstance(request, DenseProviderSelectionRequest):
            raise TypeError("request must be DenseProviderSelectionRequest.")
        if not isinstance(decision, DenseProviderSelectionDecision):
            raise TypeError("decision must be DenseProviderSelectionDecision.")
        if decision.selected_implementation is None or not decision.selected_provider_id:
            raise ReconstructionValidationError("I6.7B requires one explicitly resolved concrete provider decision.")
        if request.authoritative_sfm_available is not True:
            raise ReconstructionValidationError("I6.7B requires an admitted authoritative sparse SfM result.")
        frame_id = str(reconstruction_frame_id).strip()
        if not frame_id or request.authoritative_sfm_frame_id != frame_id:
            raise ReconstructionValidationError("I6.7B reconstruction frame must match the selection request.")
        if source_sfm_result.session_id != source_sfm_result.workspace.session_id:
            raise ReconstructionValidationError("I6.7B sparse SfM session/workspace identity drifted.")
        if not isinstance(policy, ProgressiveDenseEmissionPolicy):
            raise TypeError("policy must be ProgressiveDenseEmissionPolicy.")
        if not hasattr(snapshot_sink, "publish"):
            raise TypeError("snapshot_sink must expose publish(snapshot).")

        run_id = str(dense_run_id or uuid4())
        require_uuid(run_id, field_name="dense_run_id")
        sparse_sha = canonical_offline_sfm_result_sha256(source_sfm_result)

        self.source_sfm_result = source_sfm_result
        self.request = request
        self.decision = decision
        self.reconstruction_frame_id = frame_id
        self.snapshot_sink = snapshot_sink
        self.dense_run_id = run_id
        self.policy = policy
        self.source_sparse_result_sha256 = sparse_sha
        self.request_fingerprint_sha256 = dense_request_fingerprint_sha256(
            request,
            decision,
            sparse_sha,
            frame_id,
        )
        self._utc_now = utc_now or (lambda: datetime.now(timezone.utc).isoformat())
        self._monotonic_ns = monotonic_ns or time.monotonic_ns
        self._started_ns = int(self._monotonic_ns())
        self._revision = 0
        self._emitted_snapshot_count = 0
        self._emitted_geometry_snapshot_count = 0
        self._dropped_status_snapshot_count = 0
        self._emission_failure_count = 0
        self._failure_records: list[str] = []
        self._last_stage: DenseProviderProgressStage | None = None
        self._last_progress_bucket = -1
        self._final_preview_revision = 0
        self._last_emitted_snapshot: ProgressiveDenseSnapshot | None = None

    @property
    def dropped_status_snapshot_count(self) -> int:
        return self._dropped_status_snapshot_count

    @property
    def emission_failure_count(self) -> int:
        return self._emission_failure_count

    @property
    def failure_records(self) -> tuple[str, ...]:
        return tuple(self._failure_records)

    def _record_failure(self, message: str) -> None:
        self._emission_failure_count += 1
        if len(self._failure_records) < self.policy.max_failure_records:
            self._failure_records.append(str(message).strip() or "progressive_dense_emission_failure")

    def _next_revision(self) -> int:
        self._revision += 1
        return self._revision

    def _stage(self, event: DenseProviderProgressEvent) -> ProgressiveDenseStage:
        return ProgressiveDenseStage(event.stage.value)

    def _elapsed_ns(self) -> int:
        return max(0, int(self._monotonic_ns()) - self._started_ns)

    def _emit(self, snapshot: ProgressiveDenseSnapshot, *, geometry: bool) -> bool:
        if self._last_emitted_snapshot is not None:
            validate_progressive_dense_snapshot_transition(
                self._last_emitted_snapshot,
                snapshot,
            )
        try:
            self.snapshot_sink.publish(snapshot)
        except Exception as exc:  # visualization path is intentionally fail-soft
            self._record_failure(f"snapshot_sink:{type(exc).__name__}:{exc}")
            return False
        self._last_emitted_snapshot = snapshot
        self._emitted_snapshot_count += 1
        if geometry:
            self._emitted_geometry_snapshot_count += 1
            self._final_preview_revision = snapshot.revision
        return True

    def _should_emit_status(self, event: DenseProviderProgressEvent) -> bool:
        stage_changed = self._last_stage is None or event.stage is not self._last_stage
        bucket = min(
            self.policy.target_status_snapshots,
            int(float(event.progress_fraction) * self.policy.target_status_snapshots),
        )
        if stage_changed or bucket > self._last_progress_bucket:
            self._last_stage = event.stage
            self._last_progress_bucket = bucket
            return True
        self._dropped_status_snapshot_count += 1
        return False

    def publish(self, event: DenseProviderProgressEvent) -> None:
        if not isinstance(event, DenseProviderProgressEvent):
            raise TypeError("event must be DenseProviderProgressEvent.")
        if event.provider_id != self.decision.selected_provider_id:
            raise ReconstructionValidationError("I6.7B provider progress changed the selected provider identity.")
        if event.implementation is not self.decision.selected_implementation:
            raise ReconstructionValidationError("I6.7B provider progress changed the selected implementation identity.")
        if event.total_views <= 0 or event.total_views > self.request.available_view_count:
            raise ReconstructionValidationError("I6.7B provider progress total_views is outside the admitted source population.")
        if not self._should_emit_status(event):
            return

        geometry = None
        dense_points = 0
        mean_confidence = None
        mean_support = None
        if event.provisional_geometry_available:
            try:
                loaded = load_dense_geometry_preview(
                    event.provisional_geometry_ref,
                    max_preview_points=self.policy.max_preview_points,
                )
                geometry = loaded.preview
                dense_points = loaded.point_count
                mean_confidence = loaded.mean_confidence
                mean_support = loaded.mean_view_support
            except Exception as exc:
                self._record_failure(f"progress_geometry:{type(exc).__name__}:{exc}")

        warnings = tuple(
            value
            for value in (event.degraded_reason, event.fallback_reason)
            if str(value).strip()
        )
        progress = min(float(event.progress_fraction), I67B_MAX_PROVISIONAL_PROGRESS)
        revision = self._next_revision()
        snapshot = ProgressiveDenseSnapshot(
            source_sfm_session_id=self.source_sfm_result.session_id,
            dense_run_id=self.dense_run_id,
            revision=revision,
            timestamp_utc=str(self._utc_now()),
            elapsed_ns=self._elapsed_ns(),
            provider_id=event.provider_id,
            implementation=event.implementation,
            stage=self._stage(event),
            stage_detail=event.stage_detail,
            request_fingerprint_sha256=self.request_fingerprint_sha256,
            source_sparse_result_sha256=self.source_sparse_result_sha256,
            reconstruction_frame_id=self.reconstruction_frame_id,
            statistics=ProgressiveDenseStatistics(
                input_views=self.request.available_view_count,
                selected_views=event.total_views,
                processed_views=event.processed_views,
                current_dense_points=dense_points,
                mean_confidence=mean_confidence,
                mean_view_support=mean_support,
            ),
            progress_fraction=progress,
            geometry=geometry,
            warnings=warnings,
        )
        self._emit(snapshot, geometry=geometry is not None)

    def emit_final_geometry(self, result: DenseProviderExecutionResult) -> int:
        if not isinstance(result, DenseProviderExecutionResult):
            raise TypeError("result must be DenseProviderExecutionResult.")
        if result.provider_id != self.decision.selected_provider_id or result.implementation is not self.decision.selected_implementation:
            raise ReconstructionValidationError("I6.7B result provider identity changed during execution.")
        if result.status not in {DenseProviderRunStatus.COMPLETED, DenseProviderRunStatus.DEGRADED}:
            return 0
        if result.canonical_frame_id != self.reconstruction_frame_id:
            raise ReconstructionValidationError("I6.7B final dense geometry changed the authoritative reconstruction frame.")
        try:
            loaded = load_dense_geometry_preview(
                result.canonical_geometry_ref,
                max_preview_points=self.policy.max_preview_points,
            )
        except Exception as exc:
            self._record_failure(f"final_geometry:{type(exc).__name__}:{exc}")
            return 0

        revision = self._next_revision()
        snapshot = ProgressiveDenseSnapshot(
            source_sfm_session_id=self.source_sfm_result.session_id,
            dense_run_id=self.dense_run_id,
            revision=revision,
            timestamp_utc=str(self._utc_now()),
            elapsed_ns=self._elapsed_ns(),
            provider_id=result.provider_id,
            implementation=result.implementation,
            stage=ProgressiveDenseStage.FINALIZATION,
            stage_detail="bounded_canonical_geometry_before_external_admission",
            request_fingerprint_sha256=self.request_fingerprint_sha256,
            source_sparse_result_sha256=self.source_sparse_result_sha256,
            reconstruction_frame_id=self.reconstruction_frame_id,
            statistics=ProgressiveDenseStatistics(
                input_views=result.source_view_count,
                selected_views=result.selected_view_count,
                processed_views=result.selected_view_count,
                current_dense_points=loaded.point_count,
                mean_confidence=loaded.mean_confidence,
                mean_view_support=loaded.mean_view_support,
            ),
            progress_fraction=I67B_MAX_PROVISIONAL_PROGRESS,
            geometry=loaded.preview,
            warnings=tuple(result.degradation_reasons),
        )
        return revision if self._emit(snapshot, geometry=True) else 0

    def report(self) -> ProgressiveDenseEmissionReport:
        return ProgressiveDenseEmissionReport(
            dense_run_id=self.dense_run_id,
            source_sparse_result_sha256=self.source_sparse_result_sha256,
            request_fingerprint_sha256=self.request_fingerprint_sha256,
            emitted_snapshot_count=self._emitted_snapshot_count,
            emitted_geometry_snapshot_count=self._emitted_geometry_snapshot_count,
            dropped_status_snapshot_count=self._dropped_status_snapshot_count,
            emission_failure_count=self._emission_failure_count,
            failure_records=tuple(self._failure_records),
            final_preview_revision=self._final_preview_revision,
            geometry_snapshot_emitted=self._emitted_geometry_snapshot_count > 0,
            max_preview_points=self.policy.max_preview_points,
        )


def execute_dense_provider_with_progressive_emission(
    *,
    adapter_factory: DenseProviderAdapterFactory,
    request: DenseProviderSelectionRequest,
    decision: DenseProviderSelectionDecision,
    source_sfm_result: OfflineSfmResult,
    reconstruction_frame_id: str,
    snapshot_sink: ProgressiveDenseSnapshotSink,
    dense_run_id: str | None = None,
    policy: ProgressiveDenseEmissionPolicy = ProgressiveDenseEmissionPolicy(),
) -> ProgressiveDenseExecutionEnvelope:
    """Execute one already-selected concrete provider with provisional emission.

    The wrapper does not choose a provider.  ``adapter_factory`` receives the
    I6.7B progress sink so existing I4.3 adapters can publish their unchanged
    progress events.  A successful/degraded result is sampled once for a final
    bounded provisional geometry snapshot before the result is returned to any
    later admission/persistence layer.
    """

    session = ProgressiveDenseEmissionSession(
        source_sfm_result=source_sfm_result,
        request=request,
        decision=decision,
        reconstruction_frame_id=reconstruction_frame_id,
        snapshot_sink=snapshot_sink,
        dense_run_id=dense_run_id,
        policy=policy,
    )
    adapter = adapter_factory(session)
    descriptor = adapter.descriptor
    if descriptor.provider_id != decision.selected_provider_id:
        raise ReconstructionValidationError("I6.7B adapter provider_id does not match the selected decision.")
    if descriptor.implementation is not decision.selected_implementation:
        raise ReconstructionValidationError("I6.7B adapter implementation does not match the selected decision.")

    result = adapter.execute(request, decision)
    revision = session.emit_final_geometry(result)
    if revision > 0:
        result = replace(result, final_preview_revision=revision)
    return ProgressiveDenseExecutionEnvelope(result=result, report=session.report())


__all__ = [
    "I67B_PHASE",
    "I67B_MAX_PREVIEW_POINTS",
    "I67B_TARGET_STATUS_SNAPSHOTS",
    "I67B_MAX_FAILURE_RECORDS",
    "I67B_MAX_PROVISIONAL_PROGRESS",
    "ProgressiveDenseSnapshotSink",
    "ProgressiveDenseEmissionPolicy",
    "LoadedDenseGeometryPreview",
    "ProgressiveDenseEmissionReport",
    "ProgressiveDenseExecutionEnvelope",
    "canonical_offline_sfm_result_sha256",
    "dense_request_fingerprint_sha256",
    "evenly_sample_indices",
    "load_dense_geometry_preview",
    "ProgressiveDenseEmissionSession",
    "execute_dense_provider_with_progressive_emission",
]

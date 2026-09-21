"""Main-environment adapter for the isolated external PyCOLMAP worker.

This module intentionally never imports PyCOLMAP. Native COLMAP/PyCOLMAP objects
stay in the external worker process; only JSON-safe data crosses the boundary.
"""

from __future__ import annotations

from collections import deque
import hashlib
import json
from pathlib import Path
import queue
import subprocess
from threading import Lock, Thread
from time import monotonic, monotonic_ns, sleep
from typing import Callable, Mapping, Sequence

import numpy as np

from core.contracts import ImageSize
from core.reconstruction.backends import (
    BackendCapabilities,
    BackendOperationContext,
    ReconstructionBackendKind,
)
from core.reconstruction.contracts import InputSourceKind
from core.reconstruction.errors import (
    ReconstructionCancelledError,
    ReconstructionValidationError,
    SfmBackendUnavailableError,
    SfmWorkerExecutionError,
    SfmWorkerProtocolError,
)
from core.reconstruction.events import (
    EventSeverity,
    ReconstructionEventRecorder,
    ReconstructionEventType,
    ReconstructionStage,
)
from core.reconstruction.profiles import ReconstructionQualityProfile
from core.reconstruction.sfm_contracts import (
    OfflineSfmRequest,
    OfflineSfmResult,
    ReconstructionGauge,
    SfmCameraEstimate,
    SfmCameraModelEstimate,
    SfmLandmarkEstimate,
    SfmObservation,
    SfmStageStatistics,
    SfmWorkspaceImage,
)
from core.reconstruction.sfm_coordinates import (
    colmap_camera_from_reconstruction_to_internal_pose,
    validate_rigid_matrix,
)
from core.reconstruction.sfm_quality import assess_sfm_quality
from core.reconstruction.sfm_environment import (
    PYCOLMAP_RECOMMENDED_VERSION,
    PycolmapEnvironmentProbe,
    probe_pycolmap_environment,
)
from core.reconstruction.sfm_workspace import validate_sfm_workspace_integrity
from core.reconstruction.reconstruction_progressive_sfm_contract import (
    ProgressiveSfmSnapshot,
    validate_progressive_sfm_snapshot_transition,
)
from core.reconstruction.reconstruction_progressive_sfm_transport import (
    parse_progressive_sfm_worker_line,
)
from core.reconstruction.reconstruction_job_progress_observer import (
    JobProgressObserver, LatestJobProgressSnapshot,
)
from core.reconstruction.sfm_worker_protocol import (
    SFM_WORKER_SCHEMA_VERSION,
    SfmWorkerEvent,
    SfmWorkerEventType,
    parse_worker_event_line,
)

PYCOLMAP_BACKEND_ID = "pycolmap-offline-sfm"
PYCOLMAP_BACKEND_VERSION = "1"
_DEFAULT_TIMEOUT_S = 60.0 * 60.0
_PROFILE_MAX_IMAGE_SIZE = {
    ReconstructionQualityProfile.EFFICIENT: 1600,
    ReconstructionQualityProfile.BALANCED: 2400,
    ReconstructionQualityProfile.HIGH_QUALITY: 3200,
}
_PROFILE_SEQUENTIAL_OVERLAP = {
    ReconstructionQualityProfile.EFFICIENT: 5,
    ReconstructionQualityProfile.BALANCED: 10,
    ReconstructionQualityProfile.HIGH_QUALITY: 15,
}


def _default_worker_script() -> Path:
    return Path(__file__).resolve().parents[2] / "scripts" / "reconstruction_pycolmap_worker.py"


def _safe_workspace_child(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ReconstructionValidationError(
            f"SfM worker path escapes the workspace: {relative}"
        ) from exc
    return candidate


def _worker_request_payload(request: OfflineSfmRequest) -> dict[str, object]:
    root = Path(request.workspace.workspace_root).expanduser().resolve()
    image_records: list[dict[str, object]] = []
    image_names: list[str] = []
    for item in request.workspace.images:
        staged = Path(item.staged_image_path)
        if staged.parent.as_posix() != request.workspace.images_directory:
            raise ReconstructionValidationError(
                "Phase 8.3D requires materialized SfM images directly inside images_directory."
            )
        image_names.append(staged.name)
        image_records.append(
            {
                "name": staged.name,
                "order": item.order,
                "sha256": item.staged_sha256,
                "keyframe_id": item.keyframe_id,
                "frame_id": item.frame_id,
                "timestamp_ns": item.timestamp_ns,
            }
        )
    if len(image_names) < 3:
        raise ReconstructionValidationError(
            "Real PyCOLMAP sparse reconstruction requires at least three materialized keyframes."
        )

    configured_size = request.configuration.maximum_image_size
    maximum_image_size = configured_size or _PROFILE_MAX_IMAGE_SIZE[request.profile]
    configured_overlap = request.configuration.sequential_overlap
    sequential_overlap = configured_overlap or _PROFILE_SEQUENTIAL_OVERLAP[request.profile]
    sequential_overlap = min(max(1, sequential_overlap), max(1, len(image_names) - 1))
    payload: dict[str, object] = {
        "schema_version": SFM_WORKER_SCHEMA_VERSION,
        "session_id": request.session_id,
        "workspace_root": str(root),
        "images_directory": request.workspace.images_directory,
        "database_path": request.workspace.database_path,
        "sparse_directory": request.workspace.sparse_directory,
        "checkpoints_directory": request.workspace.checkpoints_directory,
        "source_fingerprint": request.workspace.source.fingerprint,
        "profile": request.profile.value,
        "image_names": image_names,
        "image_records": image_records,
        "configuration": {
            "feature_strategy": request.configuration.feature_strategy.value,
            "matching_strategy": request.configuration.matching_strategy.value,
            "use_gpu": request.configuration.use_gpu,
            "maximum_image_size": maximum_image_size,
            "sequential_overlap": sequential_overlap,
            "camera_mode": "single",
            "camera_model": "SIMPLE_RADIAL",
            "bundle_adjustment_gpu": False,
            "resume_policy": request.configuration.resume_policy.value,
        },
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    payload["request_fingerprint"] = hashlib.sha256(canonical).hexdigest()
    return payload


def _read_json_object(path: Path, *, description: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SfmWorkerProtocolError(f"Could not read {description}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SfmWorkerProtocolError(f"{description} must contain a JSON object.")
    return payload


def _mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise SfmWorkerProtocolError(f"{field_name} must be a JSON object.")
    return value


def _sequence(value: object, field_name: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise SfmWorkerProtocolError(f"{field_name} must be a JSON array.")
    return value


def _convert_worker_payload(
    request: OfflineSfmRequest,
    payload: Mapping[str, object],
    *,
    elapsed_ns: int,
) -> OfflineSfmResult:
    if int(payload.get("schema_version", 0)) != SFM_WORKER_SCHEMA_VERSION:
        raise SfmWorkerProtocolError(
            f"Unsupported SfM worker result schema: {payload.get('schema_version')}."
        )
    if str(payload.get("status", "")) != "ok":
        raise SfmWorkerExecutionError(str(payload.get("error", "SfM worker failed.")))
    if str(payload.get("session_id", "")) != request.session_id:
        raise SfmWorkerProtocolError("SfM worker result session_id does not match the request.")

    backend = _mapping(payload.get("backend", {}), "backend")
    pycolmap_version = str(backend.get("pycolmap_version", "")).strip()
    if pycolmap_version != PYCOLMAP_RECOMMENDED_VERSION:
        raise SfmWorkerProtocolError(
            f"Unexpected PyCOLMAP result version: {pycolmap_version or 'missing'}."
        )

    expected_worker_request = _worker_request_payload(request)
    expected_fingerprint = str(expected_worker_request["request_fingerprint"])
    actual_fingerprint = str(payload.get("request_fingerprint", "")).strip()
    if actual_fingerprint != expected_fingerprint:
        raise SfmWorkerProtocolError(
            "SfM worker result fingerprint does not match the current request/workspace."
        )

    workspace_by_name = {
        Path(item.staged_image_path).name: item for item in request.workspace.images
    }
    camera_models: list[SfmCameraModelEstimate] = []
    for raw_camera in _sequence(payload.get("camera_models", []), "camera_models"):
        camera = _mapping(raw_camera, "camera model")
        backend_camera_id = int(camera["camera_id"])
        parameters = tuple(
            float(value) for value in _sequence(camera.get("params", []), "camera.params")
        )
        camera_models.append(
            SfmCameraModelEstimate(
                camera_id=f"pycolmap-camera-{backend_camera_id}",
                backend_camera_id=backend_camera_id,
                backend_model_name=str(camera["model_name"]),
                image_size=ImageSize(
                    width=int(camera["width"]),
                    height=int(camera["height"]),
                ),
                parameters=parameters,
            )
        )
    camera_models.sort(key=lambda item: item.backend_camera_id)

    cameras: list[SfmCameraEstimate] = []
    image_id_to_workspace: dict[int, SfmWorkspaceImage] = {}
    for raw_item in _sequence(payload.get("registered_images", []), "registered_images"):
        item = _mapping(raw_item, "registered image")
        image_name = str(item.get("name", ""))
        workspace_image = workspace_by_name.get(image_name)
        if workspace_image is None:
            raise SfmWorkerProtocolError(
                f"Worker registered an image outside the workspace manifest: {image_name}"
            )
        image_id = int(item["image_id"])
        matrix34 = np.asarray(item["camera_from_reconstruction"], dtype=np.float64)
        if matrix34.shape != (3, 4):
            raise SfmWorkerProtocolError(
                f"COLMAP camera transform for {image_name} must be 3x4."
            )
        matrix44 = np.eye(4, dtype=np.float64)
        matrix44[:3, :] = matrix34
        matrix44 = validate_rigid_matrix(
            matrix44, field_name="T_camera_from_reconstruction"
        )
        internal_pose = colmap_camera_from_reconstruction_to_internal_pose(matrix44)
        cameras.append(
            SfmCameraEstimate(
                estimate_id=f"pycolmap-pose-{image_id}",
                camera_id=f"pycolmap-camera-{int(item['camera_id'])}",
                keyframe_id=workspace_image.keyframe_id,
                frame_id=workspace_image.frame_id,
                backend_image_id=image_id,
                T_reconstruction_from_camera=internal_pose,
            )
        )
        image_id_to_workspace[image_id] = workspace_image

    cameras.sort(key=lambda item: next(
        workspace.order
        for workspace in request.workspace.images
        if workspace.keyframe_id == item.keyframe_id
    ))

    landmarks: list[SfmLandmarkEstimate] = []
    dropped_short_tracks = 0
    for raw_point in _sequence(payload.get("points3D", []), "points3D"):
        point = _mapping(raw_point, "point3D")
        point_id = int(point["point3D_id"])
        observations: list[SfmObservation] = []
        seen_keyframes: set[str] = set()
        for raw_track in _sequence(point.get("track", []), "point3D.track"):
            track = _mapping(raw_track, "track element")
            image_id = int(track["image_id"])
            workspace_image = image_id_to_workspace.get(image_id)
            if workspace_image is None or workspace_image.keyframe_id in seen_keyframes:
                continue
            xy = _sequence(track.get("xy", []), "track.xy")
            if len(xy) != 2:
                raise SfmWorkerProtocolError("SfM track xy must contain two coordinates.")
            point2d_index = int(track["point2D_idx"])
            observations.append(
                SfmObservation(
                    observation_id=f"pycolmap-observation-{point_id}-{image_id}-{point2d_index}",
                    landmark_id=f"pycolmap-landmark-{point_id}",
                    keyframe_id=workspace_image.keyframe_id,
                    feature_xy=(float(xy[0]), float(xy[1])),
                    feature_index=point2d_index,
                    backend_point2d_index=point2d_index,
                )
            )
            seen_keyframes.add(workspace_image.keyframe_id)
        if len(observations) < 2:
            dropped_short_tracks += 1
            continue
        color_raw = point.get("color")
        color = None
        if color_raw is not None:
            color_values = _sequence(color_raw, "point3D.color")
            if len(color_values) == 3:
                color = tuple(int(value) for value in color_values)
        landmarks.append(
            SfmLandmarkEstimate(
                landmark_id=f"pycolmap-landmark-{point_id}",
                track_id=f"pycolmap-track-{point_id}",
                position_reconstruction=np.asarray(point["xyz"], dtype=np.float64),
                observations=tuple(observations),
                color_rgb=color,
            )
        )

    statistics_raw = _mapping(payload.get("statistics", {}), "statistics")
    mean_error = statistics_raw.get("mean_reprojection_error_px")
    statistics = SfmStageStatistics(
        input_keyframes=len(request.workspace.images),
        registered_keyframes=len(cameras),
        landmarks=len(landmarks),
        observations=sum(len(item.observations) for item in landmarks),
        elapsed_ns=max(0, int(elapsed_ns)),
        mean_reprojection_error_px=(None if mean_error is None else float(mean_error)),
    )
    warnings = [str(item) for item in _sequence(payload.get("warnings", []), "warnings")]
    if dropped_short_tracks:
        warnings.append(
            f"Dropped {dropped_short_tracks} sparse points whose converted tracks had fewer than two keyframes."
        )
    models_raw = _sequence(payload.get("models", []), "models")
    candidate_models = [dict(_mapping(item, "model summary")) for item in models_raw]
    if len(candidate_models) > 1:
        warnings.append(
            f"PyCOLMAP produced {len(candidate_models)} sparse models; the best model was selected deterministically."
        )
    if not cameras:
        raise SfmWorkerExecutionError("PyCOLMAP produced no registered cameras.")
    if not landmarks:
        raise SfmWorkerExecutionError("PyCOLMAP produced no valid sparse landmarks.")

    quality = assess_sfm_quality(statistics)
    if quality.status.value != "accepted":
        warnings.append(
            f"Sparse reconstruction quality classified as {quality.status.value}: "
            + "; ".join(quality.reasons)
        )

    resolved_configuration = _mapping(
        payload.get("resolved_configuration", {}), "resolved_configuration"
    )
    resume_from_stage = str(payload.get("resume_from_stage", "none"))

    return OfflineSfmResult(
        session_id=request.session_id,
        backend_id=PYCOLMAP_BACKEND_ID,
        backend_version=PYCOLMAP_BACKEND_VERSION,
        gauge=ReconstructionGauge(),
        cameras=tuple(cameras),
        landmarks=tuple(landmarks),
        statistics=statistics,
        workspace=request.workspace,
        camera_models=tuple(camera_models),
        warnings=tuple(warnings),
        metadata={
            "real_sfm": True,
            "pycolmap_version": pycolmap_version,
            "selected_model_id": int(payload.get("selected_model_id", -1)),
            "candidate_models": candidate_models,
            "matching_strategy": request.configuration.matching_strategy.value,
            "feature_strategy": request.configuration.feature_strategy.value,
            "use_gpu": request.configuration.use_gpu,
            "scale_status": "arbitrary",
            "source_fingerprint": request.workspace.source.fingerprint,
            "worker_request_fingerprint": actual_fingerprint,
            "resume_policy": request.configuration.resume_policy.value,
            "resume_from_stage": resume_from_stage,
            "quality_assessment": quality.as_dict(),
            "resolved_configuration": dict(resolved_configuration),
            "external_python_executable": str(backend.get("python_executable", "")),
            "external_cuda_devices": int(backend.get("cuda_devices", 0)),
            "worker_elapsed_s": float(payload.get("worker_elapsed_s", 0.0)),
        },
    )


class ExternalPycolmapSfmBackend:
    """Run PyCOLMAP 4.1.1 in a dedicated external Python interpreter."""

    def __init__(
        self,
        python_executable: str,
        *,
        worker_script: str | Path | None = None,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
        probe_function: Callable[[str], PycolmapEnvironmentProbe] = probe_pycolmap_environment,
        progressive_snapshot_callback: Callable[[ProgressiveSfmSnapshot], None] | None = None,
    ) -> None:
        self._python_executable = str(python_executable).strip()
        if not self._python_executable:
            raise ReconstructionValidationError("python_executable must not be empty.")
        self._worker_script = Path(worker_script or _default_worker_script()).resolve()
        timeout = float(timeout_s)
        if timeout <= 0.0:
            raise ReconstructionValidationError("timeout_s must be positive.")
        self._timeout_s = timeout
        self._probe_function = probe_function
        if progressive_snapshot_callback is not None and not callable(progressive_snapshot_callback):
            raise TypeError("progressive_snapshot_callback must be callable or None.")
        self._progressive_snapshot_callback = progressive_snapshot_callback
        self._probe: PycolmapEnvironmentProbe | None = None
        self._events: ReconstructionEventRecorder | None = None
        self._started_ns = 0
        self._process_lock = Lock()
        self._active_process: subprocess.Popen[str] | None = None

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend_id=PYCOLMAP_BACKEND_ID,
            display_name="PyCOLMAP Offline SfM",
            backend_version=PYCOLMAP_BACKEND_VERSION,
            kinds=(ReconstructionBackendKind.OFFLINE_SFM,),
            source_kinds=(InputSourceKind.IMAGE_SEQUENCE, InputSourceKind.RECORDED_VIDEO),
            quality_profiles=(
                ReconstructionQualityProfile.EFFICIENT,
                ReconstructionQualityProfile.BALANCED,
                ReconstructionQualityProfile.HIGH_QUALITY,
            ),
            supports_cancellation=True,
            supports_pause=False,
            supports_checkpoint=True,
            supports_resume=True,
            supports_metric_scale=False,
            supports_cpu=True,
            requires_cuda=False,
            optional_dependencies=(f"pycolmap-cuda12=={PYCOLMAP_RECOMMENDED_VERSION}",),
        )

    @property
    def environment_probe(self) -> PycolmapEnvironmentProbe | None:
        return self._probe

    def _emit(
        self,
        context: BackendOperationContext,
        *,
        event_type: ReconstructionEventType,
        message: str,
        progress: float | None = None,
        severity: EventSeverity = EventSeverity.INFO,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        if self._events is None:
            self._events = ReconstructionEventRecorder(context.session_id)
        event = self._events.emit(
            event_type=event_type,
            stage=ReconstructionStage.SPARSE_RECONSTRUCTION,
            message=message,
            elapsed_ns=max(0, monotonic_ns() - self._started_ns),
            severity=severity,
            progress_fraction=progress,
            metadata=metadata or {},
        )
        context.emit_event(event)

    def _emit_worker_event(self, context: BackendOperationContext, event: SfmWorkerEvent) -> None:
        event_type = ReconstructionEventType.STATUS
        severity = EventSeverity.INFO
        if event.event_type is SfmWorkerEventType.PROGRESS:
            event_type = ReconstructionEventType.PROGRESS
        elif event.event_type is SfmWorkerEventType.WARNING:
            event_type = ReconstructionEventType.WARNING
            severity = EventSeverity.WARNING
        self._emit(
            context,
            event_type=event_type,
            message=event.message,
            progress=event.progress_fraction,
            severity=severity,
            metadata={"worker_stage": event.stage, **dict(event.metadata)},
        )
        # I6.8B: passive worker stage source, entirely outside result/admission.
        # Never allow diagnostic observation to alter the existing worker path.
        observer = getattr(self, '_i68b_progress_observer', None)
        sink = getattr(self, '_i68b_progress_sink', None)
        if observer is not None and sink is not None:
            try:
                sink.publish(observer.from_sfm_worker_event(event))
            except Exception:
                pass


    def prepare(self, context: BackendOperationContext) -> None:
        if context.control.cancellation_requested:
            raise ReconstructionCancelledError("Reconstruction cancellation requested.")
        if context.stage is not ReconstructionStage.SPARSE_RECONSTRUCTION:
            raise ReconstructionValidationError(
                "PyCOLMAP backend requires the sparse_reconstruction stage."
            )
        if not self._worker_script.is_file():
            raise SfmBackendUnavailableError(
                f"External PyCOLMAP worker script is missing: {self._worker_script}"
            )
        probe = self._probe_function(self._python_executable)
        self._probe = probe
        if not probe.backend_ready:
            raise SfmBackendUnavailableError(
                f"External PyCOLMAP environment is not ready: {probe.error or probe.as_dict()}"
            )
        if not probe.recommended_version:
            raise SfmBackendUnavailableError(
                f"Phase 8.3C requires PyCOLMAP {PYCOLMAP_RECOMMENDED_VERSION}; "
                f"external environment reports {probe.pycolmap_version}."
            )
        self._started_ns = monotonic_ns()
        self._events = ReconstructionEventRecorder(context.session_id)
        self._emit(
            context,
            event_type=ReconstructionEventType.STATUS,
            message="External PyCOLMAP environment validated.",
            metadata={
                "python_executable": probe.python_executable,
                "pycolmap_version": probe.pycolmap_version,
                "cuda_devices": probe.cuda_devices,
            },
        )

    @staticmethod
    def _terminate_process(process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5.0)

    def run_sparse_reconstruction(
        self,
        request: OfflineSfmRequest,
        context: BackendOperationContext,
    ) -> OfflineSfmResult:
        if request.session_id != context.session_id:
            raise ReconstructionValidationError(
                "SfM request and backend context session IDs must match."
            )
        # Re-prepare for every run so event sequencing, elapsed time, stage
        # validation, and the external-environment probe are fresh even when
        # one backend instance is reused after a previous attempt.
        self.prepare(context)
        assert self._probe is not None
        if request.configuration.use_gpu and not self._probe.cuda_ready:
            raise SfmBackendUnavailableError(
                "GPU SfM was requested but the external PyCOLMAP environment has no ready CUDA device."
            )
        if context.control.cancellation_requested:
            raise ReconstructionCancelledError("Reconstruction cancellation requested.")

        validate_sfm_workspace_integrity(request.workspace)
        root = Path(request.workspace.workspace_root).expanduser().resolve()
        # Optional diagnostic sidecar for the normal One Start session layout.
        # Other callers and test workspaces do not get new output files.
        self._i68b_progress_observer = None
        self._i68b_progress_sink = None
        if root.name == 'sfm_workspace' and root.parent.name == request.session_id:
            try:
                self._i68b_progress_observer = JobProgressObserver('offline_sfm', request.session_id)
                self._i68b_progress_sink = LatestJobProgressSnapshot(
                    root.parent / 'job_progress_observation.txt')
            except ValueError:
                pass

        checkpoints_dir = _safe_workspace_child(root, request.workspace.checkpoints_directory)
        logs_dir = _safe_workspace_child(root, request.workspace.logs_directory)
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        request_path = checkpoints_dir / "pycolmap_worker_request.json"
        result_path = checkpoints_dir / "pycolmap_worker_result.json"
        stderr_path = logs_dir / "pycolmap_worker_stderr.log"
        worker_request_payload = _worker_request_payload(request)
        request_path.write_text(
            json.dumps(worker_request_payload, indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        expected_request_fingerprint = str(worker_request_payload["request_fingerprint"])
        result_path.unlink(missing_ok=True)
        stderr_path.unlink(missing_ok=True)

        command = [
            self._python_executable,
            "-u",
            str(self._worker_script),
            "--request",
            str(request_path),
            "--result",
            str(result_path),
        ]
        self._emit(
            context,
            event_type=ReconstructionEventType.STATUS,
            message="Starting isolated PyCOLMAP sparse reconstruction worker.",
            metadata={"worker_script": str(self._worker_script)},
        )

        stdout_queue: queue.Queue[str] = queue.Queue()
        stderr_tail: deque[str] = deque(maxlen=80)

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        with self._process_lock:
            self._active_process = process

        def read_stdout() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                stdout_queue.put(line)

        def read_stderr() -> None:
            assert process.stderr is not None
            with stderr_path.open("a", encoding="utf-8") as handle:
                for line in process.stderr:
                    stderr_tail.append(line.rstrip())
                    handle.write(line)
                    handle.flush()

        stdout_thread = Thread(target=read_stdout, name="pycolmap-worker-stdout", daemon=True)
        stderr_thread = Thread(target=read_stderr, name="pycolmap-worker-stderr", daemon=True)
        stdout_thread.start()
        stderr_thread.start()
        started = monotonic()
        last_progressive_snapshot: ProgressiveSfmSnapshot | None = None

        def consume_stdout_line(line: str) -> None:
            nonlocal last_progressive_snapshot
            event = parse_worker_event_line(line)
            if event is not None:
                self._emit_worker_event(context, event)
                return
            try:
                snapshot = parse_progressive_sfm_worker_line(line)
            except SfmWorkerProtocolError as exc:
                self._emit(
                    context,
                    event_type=ReconstructionEventType.WARNING,
                    message="Rejected malformed progressive sparse-SfM worker snapshot; final SfM continues.",
                    severity=EventSeverity.WARNING,
                    metadata={"progressive_snapshot_rejection": str(exc)},
                )
                return
            if snapshot is None:
                return
            if snapshot.session_id != request.session_id:
                self._emit(
                    context,
                    event_type=ReconstructionEventType.WARNING,
                    message="Rejected cross-session progressive sparse-SfM snapshot; final SfM continues.",
                    severity=EventSeverity.WARNING,
                    metadata={"progressive_snapshot_rejection": "session_id_mismatch"},
                )
                return
            if snapshot.request_fingerprint_sha256 != expected_request_fingerprint:
                self._emit(
                    context,
                    event_type=ReconstructionEventType.WARNING,
                    message="Rejected mismatched-request progressive sparse-SfM snapshot; final SfM continues.",
                    severity=EventSeverity.WARNING,
                    metadata={"progressive_snapshot_rejection": "request_fingerprint_mismatch"},
                )
                return
            if last_progressive_snapshot is not None:
                try:
                    validate_progressive_sfm_snapshot_transition(
                        last_progressive_snapshot, snapshot
                    )
                except ReconstructionValidationError as exc:
                    self._emit(
                        context,
                        event_type=ReconstructionEventType.WARNING,
                        message="Rejected stale progressive sparse-SfM snapshot; final SfM continues.",
                        severity=EventSeverity.WARNING,
                        metadata={"progressive_snapshot_rejection": str(exc)},
                    )
                    return
            callback = self._progressive_snapshot_callback
            if callback is not None:
                try:
                    callback(snapshot)
                except Exception as exc:
                    self._emit(
                        context,
                        event_type=ReconstructionEventType.WARNING,
                        message="Progressive sparse-SfM transport callback failed; final SfM continues.",
                        severity=EventSeverity.WARNING,
                        metadata={
                            "progressive_snapshot_transport_error": f"{type(exc).__name__}: {exc}"
                        },
                    )
                    return
            last_progressive_snapshot = snapshot

        try:
            while process.poll() is None:
                if context.control.cancellation_requested:
                    self._terminate_process(process)
                    self._emit(
                        context,
                        event_type=ReconstructionEventType.CANCELLATION,
                        message="PyCOLMAP sparse reconstruction cancelled.",
                    )
                    raise ReconstructionCancelledError(
                        "Reconstruction cancellation requested."
                    )
                while True:
                    try:
                        line = stdout_queue.get_nowait()
                    except queue.Empty:
                        break
                    consume_stdout_line(line)
                if monotonic() - started > self._timeout_s:
                    self._terminate_process(process)
                    raise SfmWorkerExecutionError(
                        f"PyCOLMAP worker exceeded timeout of {self._timeout_s:.1f} seconds."
                    )
                sleep(0.02)

            stdout_thread.join(timeout=2.0)
            stderr_thread.join(timeout=2.0)
            while True:
                try:
                    line = stdout_queue.get_nowait()
                except queue.Empty:
                    break
                consume_stdout_line(line)

            if process.returncode != 0:
                detail = "\n".join(stderr_tail).strip()
                worker_error = ""
                if result_path.is_file():
                    try:
                        failure_payload = _read_json_object(
                            result_path, description="PyCOLMAP worker failure result"
                        )
                        worker_error = str(failure_payload.get("error", "")).strip()
                    except SfmWorkerProtocolError:
                        worker_error = ""
                summary = worker_error or detail or "No worker error detail was captured."
                raise SfmWorkerExecutionError(
                    f"PyCOLMAP worker exited with code {process.returncode}: {summary}"
                )
            if not result_path.is_file():
                raise SfmWorkerProtocolError(
                    "PyCOLMAP worker completed without writing its result payload."
                )
            payload = _read_json_object(result_path, description="PyCOLMAP worker result")
            result = _convert_worker_payload(
                request,
                payload,
                elapsed_ns=max(0, monotonic_ns() - self._started_ns),
            )
            self._emit(
                context,
                event_type=ReconstructionEventType.PROGRESS,
                message="Real PyCOLMAP sparse reconstruction converted to EIA contracts.",
                progress=1.0,
                metadata={
                    "registered_keyframes": result.statistics.registered_keyframes,
                    "landmarks": result.statistics.landmarks,
                },
            )
            # The returned typed SfM result supplies actual distinct registered
            # views; mapping callback counts are never used as that statistic.
            observer = getattr(self, '_i68b_progress_observer', None)
            sink = getattr(self, '_i68b_progress_sink', None)
            if observer is not None and sink is not None:
                try:
                    sink.publish(observer.from_confirmed_sparse_view_count(
                        result.statistics.registered_keyframes, len(request.workspace.images)))
                except Exception:
                    pass  # observational failure cannot invalidate a valid SfM result
            return result
        finally:
            with self._process_lock:
                self._active_process = None
            if process.poll() is None:
                self._terminate_process(process)

    def shutdown(self) -> None:
        with self._process_lock:
            process = self._active_process
        if process is not None:
            self._terminate_process(process)
        self._probe = None
        self._events = None
        self._started_ns = 0

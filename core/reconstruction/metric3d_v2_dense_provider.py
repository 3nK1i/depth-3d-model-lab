"""I4.3E1 canonical production adapter for Metric3D v2.

Metric3D v2 already exists as a mature image-depth model adapter in the
application. I4.3E1 binds that existing runtime to the I4.3A offline dense
provider contract without changing its interactive behavior.

Each registered SfM view is inferred independently in a worker process. Native
Metric3D metric-depth, virtual/estimated intrinsics, normals, and confidence are
retained as model evidence only. Canonical projection always uses the existing
bundle-adjusted PyCOLMAP camera poses and SfM camera model. Per-view robust
scale-only alignment maps native depth into the arbitrary reconstruction gauge;
it does not establish physical metric scale.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from math import isfinite
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Protocol

import cv2
import numpy as np

from core.contracts.validation import JsonValue, metadata_as_dict, normalize_metadata, require_nonempty
from core.models.catalog import METRIC3D_V2_MODEL_ID
from core.reconstruction.camera_geometry import camera_geometry_from_estimated_camera
from core.reconstruction.dense_fusion import (
    DenseDepthUnit,
    DenseFusionFrame,
    DenseFusionPolicy,
    DenseSceneProfile,
    fuse_dense_frames,
)
from core.reconstruction.dense_provider_architecture import (
    I43A_AUTHORITATIVE_FRAME_OWNER,
    DenseProviderDescriptor,
    DenseProviderExecutionResult,
    DenseProviderImplementation,
    DenseProviderProgressEvent,
    DenseProviderProgressSink,
    DenseProviderProgressStage,
    DenseProviderQualificationState,
    DenseProviderRunStatus,
    DenseProviderSelectionDecision,
    DenseProviderSelectionMode,
    DenseProviderSelectionRequest,
)
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.orchestration import ReconstructionProviderExecution
from core.reconstruction.scale_reference import arbitrary_scale_reference
from core.reconstruction.sfm_contracts import OfflineSfmResult, SfmWorkspaceImage
from core.reconstruction.sfm_dense_alignment import align_sfm_keyframe_dense_depth
from core.reconstruction.sfm_dense_projection import reconstruction_camera_from_sfm_model
from core.reconstruction.sfm_pycolmap import PYCOLMAP_BACKEND_ID

I43E1_SCHEMA_VERSION = 1
I43E1_PHASE = "I4.3E1"
I43E1_ADAPTER_VERSION = "i4.3e1-v1"
I43E1_PROVIDER_ID = "metric3d-v2-single-image"
I43E1_DEFAULT_SOURCE_DIR = "third_party/Metric3D"
I43E1_DEFAULT_READY_MARKER = "third_party/Metric3D/.eia_metric3d_v2_ready.json"
I43E1_DEFAULT_CHECKPOINT = "checkpoints/metric3d-v2/metric_depth_vit_small_800k.pth"
I43E1_DEFAULT_MINIMUM_CORRESPONDENCES = 8
I43E1_DEFAULT_MAXIMUM_RELATIVE_RESIDUAL = 0.35
I43E1_DEFAULT_VOXEL_SCENE_FRACTION = 0.004
I43E1_DEFAULT_PIXEL_STRIDE = 2
I43E1_DEFAULT_MAX_VOXELS = 1_500_000
I43E1_DEFAULT_MINIMUM_ACCEPTED_VIEWS = 2


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError(f"I4.3E1 could not read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"I4.3E1 JSON root must be an object: {path}")
    return payload


def _robust_scene_diagonal(sfm_result: OfflineSfmResult) -> float:
    if len(sfm_result.landmarks) < 4:
        raise ReconstructionValidationError("I4.3E1 requires sparse SfM landmarks for scene-relative fusion scale.")
    points = np.vstack([np.asarray(item.position_reconstruction, dtype=np.float64) for item in sfm_result.landmarks])
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ReconstructionValidationError("I4.3E1 sparse SfM landmarks are invalid.")
    low = np.quantile(points, 0.02, axis=0)
    high = np.quantile(points, 0.98, axis=0)
    diagonal = float(np.linalg.norm(high - low))
    if not isfinite(diagonal) or diagonal <= 0.0:
        raise ReconstructionValidationError("I4.3E1 robust sparse scene diagonal is invalid.")
    return diagonal


def _sfm_authority_digest(sfm_result: OfflineSfmResult) -> str:
    digest = hashlib.sha256()
    for camera in sorted(sfm_result.cameras, key=lambda item: item.keyframe_id):
        digest.update(camera.keyframe_id.encode("utf-8"))
        digest.update(camera.camera_id.encode("utf-8"))
        digest.update(np.asarray(camera.T_reconstruction_from_camera, dtype="<f8").tobytes())
    for landmark in sorted(sfm_result.landmarks, key=lambda item: item.landmark_id):
        digest.update(landmark.landmark_id.encode("utf-8"))
        digest.update(np.asarray(landmark.position_reconstruction, dtype="<f8").tobytes())
    return digest.hexdigest()


def _write_binary_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> Path:
    xyz = np.asarray(points, dtype=np.float32)
    rgb = np.asarray(colors, dtype=np.uint8)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or rgb.shape != (len(xyz), 3):
        raise ReconstructionValidationError("I4.3E1 PLY geometry must be aligned Nx3 XYZ/RGB arrays.")
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(xyz)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    rows = np.empty(len(xyz), dtype=dtype)
    rows["x"], rows["y"], rows["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rows["r"], rows["g"], rows["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with path.open("wb") as handle:
        handle.write(header)
        handle.write(rows.tobytes())
    return path


@dataclass(frozen=True, slots=True)
class Metric3DV2EnvironmentProbe:
    python_executable: str
    project_root: str
    source_directory: str
    ready_marker: str
    checkpoint_path: str
    source_revision: str
    source_ready: bool
    checkpoint_ready: bool
    adapter_importable: bool
    torch_available: bool
    cuda_available: bool
    device_names: tuple[str, ...] = ()
    error: str = ""

    @property
    def available(self) -> bool:
        return bool(
            self.source_ready
            and self.checkpoint_ready
            and self.adapter_importable
            and self.torch_available
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "python_executable": self.python_executable,
            "project_root": self.project_root,
            "source_directory": self.source_directory,
            "ready_marker": self.ready_marker,
            "checkpoint_path": self.checkpoint_path,
            "source_revision": self.source_revision,
            "source_ready": self.source_ready,
            "checkpoint_ready": self.checkpoint_ready,
            "adapter_importable": self.adapter_importable,
            "torch_available": self.torch_available,
            "cuda_available": self.cuda_available,
            "device_names": list(self.device_names),
            "available": self.available,
            "error": self.error,
        }


def probe_metric3d_v2_environment(
    *,
    project_root: Path | None = None,
    python_executable: Path | None = None,
    checkpoint_path: Path | None = None,
) -> Metric3DV2EnvironmentProbe:
    root = Path(project_root or Path.cwd()).expanduser().resolve()
    python = Path(python_executable or sys.executable).expanduser().resolve()
    source = root / I43E1_DEFAULT_SOURCE_DIR
    marker = root / I43E1_DEFAULT_READY_MARKER
    checkpoint = Path(checkpoint_path or (root / I43E1_DEFAULT_CHECKPOINT)).expanduser().resolve()
    source_ready = source.is_dir() and marker.is_file()
    checkpoint_ready = checkpoint.is_file()
    revision = ""
    errors: list[str] = []
    if marker.is_file():
        try:
            payload = _read_json(marker)
            revision = str(payload.get("commit") or payload.get("ref") or "").strip()
        except ReconstructionValidationError as exc:
            errors.append(str(exc))
    if not revision:
        errors.append("Metric3D ready marker has no pinned commit/ref.")

    adapter_importable = False
    torch_available = False
    cuda_available = False
    device_names: tuple[str, ...] = ()
    try:
        from core.models.adapters.metric3d_v2_adapter import Metric3DV2GeometryModel
        adapter_importable = Metric3DV2GeometryModel is not None
    except Exception as exc:  # noqa: BLE001 - structured runtime probe
        errors.append(f"adapter_import_failed:{exc}")
    try:
        import torch
        torch_available = True
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            device_names = tuple(str(torch.cuda.get_device_name(index)) for index in range(torch.cuda.device_count()))
    except Exception as exc:  # noqa: BLE001
        errors.append(f"torch_import_failed:{exc}")
    if not source_ready:
        errors.append("Metric3D source/ready marker missing.")
    if not checkpoint_ready:
        errors.append(f"Metric3D checkpoint missing: {checkpoint}")
    if not python.is_file():
        errors.append(f"Python executable missing: {python}")

    return Metric3DV2EnvironmentProbe(
        python_executable=str(python),
        project_root=str(root),
        source_directory=str(source),
        ready_marker=str(marker),
        checkpoint_path=str(checkpoint),
        source_revision=revision,
        source_ready=source_ready,
        checkpoint_ready=checkpoint_ready,
        adapter_importable=adapter_importable,
        torch_available=torch_available,
        cuda_available=cuda_available,
        device_names=device_names,
        error="; ".join(errors),
    )


def metric3d_v2_dense_provider_descriptor(probe: Metric3DV2EnvironmentProbe) -> DenseProviderDescriptor:
    if not isinstance(probe, Metric3DV2EnvironmentProbe):
        raise TypeError("probe must be Metric3DV2EnvironmentProbe.")
    return DenseProviderDescriptor(
        provider_id=I43E1_PROVIDER_ID,
        implementation=DenseProviderImplementation.METRIC3D_V2,
        execution=ReconstructionProviderExecution.EXTERNAL_PROCESS,
        backend_version=probe.source_revision or "unresolved",
        adapter_version=I43E1_ADAPTER_VERSION,
        qualification_state=DenseProviderQualificationState.CANDIDATE,
        available=probe.available,
        unavailable_reason="" if probe.available else (probe.error or "metric3d_v2_runtime_unavailable"),
        metadata={
            "native_output": "depth_evidence",
            "frame_binding": "align_to_authoritative_sfm",
            "native_depth_scale": "metric",
            "native_intrinsics_source": "estimated",
            "native_confidence_kind": "confidence",
            "native_confidence_range": "unbounded_score",
            "native_confidence_consumed_as_point_confidence": False,
            "single_image_stateful_sequence": False,
            "all_registered_views_available_by_default": True,
            "adaptive_view_selection_deferred_to": "I4.3F",
            "permanent_global_view_cap": None,
            "model_metric_depth_promoted_to_canonical_metric_scale": False,
            "model_estimated_intrinsics_consumed_for_canonical_projection": False,
            "camera_pose_authority_owner": I43A_AUTHORITATIVE_FRAME_OWNER,
        },
    )


@dataclass(frozen=True, slots=True)
class Metric3DV2ProductionConfig:
    minimum_correspondences: int = I43E1_DEFAULT_MINIMUM_CORRESPONDENCES
    maximum_relative_residual: float = I43E1_DEFAULT_MAXIMUM_RELATIVE_RESIDUAL
    voxel_scene_fraction: float = I43E1_DEFAULT_VOXEL_SCENE_FRACTION
    pixel_stride: int = I43E1_DEFAULT_PIXEL_STRIDE
    max_voxels: int = I43E1_DEFAULT_MAX_VOXELS
    minimum_accepted_views: int = I43E1_DEFAULT_MINIMUM_ACCEPTED_VIEWS

    def __post_init__(self) -> None:
        if int(self.minimum_correspondences) < 3:
            raise ReconstructionValidationError("I4.3E1 minimum_correspondences must be at least 3.")
        residual = float(self.maximum_relative_residual)
        if not isfinite(residual) or not 0.0 < residual < 1.0:
            raise ReconstructionValidationError("I4.3E1 maximum_relative_residual must be finite in (0,1).")
        fraction = float(self.voxel_scene_fraction)
        if not isfinite(fraction) or not 0.0 < fraction < 0.1:
            raise ReconstructionValidationError("I4.3E1 voxel_scene_fraction must be finite in (0,0.1).")
        if int(self.pixel_stride) <= 0 or int(self.max_voxels) <= 0:
            raise ReconstructionValidationError("I4.3E1 pixel_stride/max_voxels must be positive.")
        if int(self.minimum_accepted_views) < 2:
            raise ReconstructionValidationError("I4.3E1 minimum_accepted_views must be at least 2.")
        object.__setattr__(self, "minimum_correspondences", int(self.minimum_correspondences))
        object.__setattr__(self, "maximum_relative_residual", residual)
        object.__setattr__(self, "voxel_scene_fraction", fraction)
        object.__setattr__(self, "pixel_stride", int(self.pixel_stride))
        object.__setattr__(self, "max_voxels", int(self.max_voxels))
        object.__setattr__(self, "minimum_accepted_views", int(self.minimum_accepted_views))

    def as_dict(self) -> dict[str, Any]:
        return {
            "minimum_correspondences": self.minimum_correspondences,
            "maximum_relative_residual": self.maximum_relative_residual,
            "voxel_scene_fraction": self.voxel_scene_fraction,
            "pixel_stride": self.pixel_stride,
            "max_voxels": self.max_voxels,
            "minimum_accepted_views": self.minimum_accepted_views,
            "all_registered_views_selected": True,
            "adaptive_view_selection_deferred_to": "I4.3F",
            "permanent_global_view_cap": None,
        }


@dataclass(frozen=True, slots=True)
class Metric3DV2ProviderRunContext:
    sfm_result: OfflineSfmResult
    reconstruction_frame_id: str
    output_dir: Path
    python_executable: Path
    worker_script: Path
    runtime_config: Mapping[str, JsonValue]
    device: str = "cuda:0"
    precision: str = "fp16"
    config: Metric3DV2ProductionConfig = Metric3DV2ProductionConfig()

    def __post_init__(self) -> None:
        if not isinstance(self.sfm_result, OfflineSfmResult):
            raise TypeError("sfm_result must be OfflineSfmResult.")
        if self.sfm_result.backend_id != PYCOLMAP_BACKEND_ID:
            raise ReconstructionValidationError("I4.3E1 requires the authoritative PyCOLMAP offline SfM result.")
        if self.sfm_result.statistics.registered_keyframes < 2:
            raise ReconstructionValidationError("I4.3E1 requires at least two registered SfM views.")
        if not self.sfm_result.camera_models:
            raise ReconstructionValidationError("I4.3E1 requires SfM camera-model estimates.")
        frame_id = require_nonempty(self.reconstruction_frame_id, field_name="reconstruction_frame_id")
        object.__setattr__(self, "reconstruction_frame_id", frame_id)
        for name in ("output_dir", "python_executable", "worker_script"):
            object.__setattr__(self, name, Path(getattr(self, name)).expanduser().resolve())
        object.__setattr__(self, "runtime_config", normalize_metadata(self.runtime_config))
        if not isinstance(self.config, Metric3DV2ProductionConfig):
            raise TypeError("config must be Metric3DV2ProductionConfig.")
        device = require_nonempty(self.device, field_name="device")
        precision = str(self.precision).strip().lower()
        if precision not in {"fp32", "fp16"}:
            raise ReconstructionValidationError("I4.3E1 precision must be fp32 or fp16.")
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "precision", precision)


@dataclass(frozen=True, slots=True)
class Metric3DV2WorkerResult:
    manifest_path: Path
    metadata: Mapping[str, Any]


class Metric3DV2WorkerRunner(Protocol):
    def run(self, manifest_path: Path, output_dir: Path) -> Metric3DV2WorkerResult: ...


class SubprocessMetric3DV2WorkerRunner:
    def __init__(self, *, python_executable: Path, worker_script: Path, project_root: Path) -> None:
        self.python_executable = Path(python_executable).expanduser().resolve()
        self.worker_script = Path(worker_script).expanduser().resolve()
        self.project_root = Path(project_root).expanduser().resolve()

    def run(self, manifest_path: Path, output_dir: Path) -> Metric3DV2WorkerResult:
        output_dir = Path(output_dir).expanduser().resolve()
        completed = subprocess.run(
            [
                str(self.python_executable),
                str(self.worker_script),
                "--manifest",
                str(Path(manifest_path).expanduser().resolve()),
                "--output-dir",
                str(output_dir),
            ],
            cwd=str(self.project_root),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode != 0:
            raise ReconstructionValidationError(
                "I4.3E1 Metric3D worker failed: "
                f"exit={completed.returncode} stderr={completed.stderr.strip()[:2000]}"
            )
        result_path = output_dir / "single_image_depth_result_manifest.json"
        if not result_path.is_file():
            raise ReconstructionValidationError("I4.3E1 worker did not produce its result manifest.")
        return Metric3DV2WorkerResult(result_path, _read_json(result_path))


class Metric3DV2ProductionProviderAdapter:
    def __init__(
        self,
        *,
        context: Metric3DV2ProviderRunContext,
        environment: Metric3DV2EnvironmentProbe,
        runner: Metric3DV2WorkerRunner | None = None,
        progress_sink: DenseProviderProgressSink | None = None,
    ) -> None:
        if not isinstance(context, Metric3DV2ProviderRunContext):
            raise TypeError("context must be Metric3DV2ProviderRunContext.")
        if not isinstance(environment, Metric3DV2EnvironmentProbe):
            raise TypeError("environment must be Metric3DV2EnvironmentProbe.")
        self.context = context
        self.environment = environment
        self._runner = runner or SubprocessMetric3DV2WorkerRunner(
            python_executable=context.python_executable,
            worker_script=context.worker_script,
            project_root=Path(environment.project_root),
        )
        self._progress_sink = progress_sink
        self._progress_event_count = 0

    @property
    def descriptor(self) -> DenseProviderDescriptor:
        return metric3d_v2_dense_provider_descriptor(self.environment)

    def _publish(
        self,
        *,
        stage: DenseProviderProgressStage,
        detail: str,
        progress: float,
        processed: int,
        total: int,
    ) -> None:
        self._progress_event_count += 1
        if self._progress_sink is None:
            return
        self._progress_sink.publish(
            DenseProviderProgressEvent(
                provider_id=I43E1_PROVIDER_ID,
                implementation=DenseProviderImplementation.METRIC3D_V2,
                stage=stage,
                stage_detail=detail,
                progress_fraction=progress,
                processed_views=processed,
                total_views=total,
            )
        )

    def _validate_selection(
        self,
        request: DenseProviderSelectionRequest,
        decision: DenseProviderSelectionDecision,
    ) -> None:
        if not isinstance(request, DenseProviderSelectionRequest) or not isinstance(decision, DenseProviderSelectionDecision):
            raise TypeError("request/decision must use I4.3A dense-provider contracts.")
        if request.mode is not DenseProviderSelectionMode.EXPLICIT_PROVIDER:
            raise ReconstructionValidationError("I4.3E1 Metric3D v2 is explicit-provider only; Auto remains unauthorized here.")
        if request.requested_provider_id != I43E1_PROVIDER_ID or decision.selected_provider_id != I43E1_PROVIDER_ID:
            raise ReconstructionValidationError("I4.3E1 concrete provider ID does not match Metric3D v2.")
        if decision.selected_implementation is not DenseProviderImplementation.METRIC3D_V2:
            raise ReconstructionValidationError("I4.3E1 selected implementation must be Metric3D v2.")
        if decision.fallback_used or request.allow_fallback:
            raise ReconstructionValidationError("I4.3E1 forbids silent provider fallback during explicit qualification.")
        if not request.authoritative_sfm_available or request.authoritative_sfm_frame_id != self.context.reconstruction_frame_id:
            raise ReconstructionValidationError("I4.3E1 refuses canonical binding without the requested authoritative SfM frame.")
        if request.available_view_count != self.context.sfm_result.statistics.registered_keyframes:
            raise ReconstructionValidationError("I4.3E1 available_view_count must match registered SfM views.")

    def execute(
        self,
        request: DenseProviderSelectionRequest,
        decision: DenseProviderSelectionDecision,
    ) -> DenseProviderExecutionResult:
        self._validate_selection(request, decision)
        context = self.context
        if not self.environment.available:
            raise ReconstructionValidationError(
                "I4.3E1 Metric3D v2 runtime is unavailable: " + (self.environment.error or "not_ready")
            )
        if not context.python_executable.is_file():
            raise ReconstructionValidationError(f"I4.3E1 Python executable is missing: {context.python_executable}")
        if not context.worker_script.is_file():
            raise ReconstructionValidationError(f"I4.3E1 worker script is missing: {context.worker_script}")

        root = Path(context.sfm_result.workspace.workspace_root).expanduser().resolve()
        images = tuple(sorted(context.sfm_result.workspace.images, key=lambda item: item.order))
        if len(images) != context.sfm_result.statistics.registered_keyframes:
            raise ReconstructionValidationError("I4.3E1 workspace/registered-view count drifted.")
        camera_by_keyframe = {item.keyframe_id: item for item in context.sfm_result.cameras}
        model_by_id = {item.camera_id: item for item in context.sfm_result.camera_models}
        camera_ids = {camera_by_keyframe[item.keyframe_id].camera_id for item in images}
        if len(camera_ids) != 1:
            raise ReconstructionValidationError(
                "I4.3E1 canonical fusion currently requires one authoritative SfM camera model per provider run."
            )
        camera_id = next(iter(camera_ids))
        camera_model = model_by_id.get(camera_id)
        if camera_model is None:
            raise ReconstructionValidationError("I4.3E1 authoritative SfM camera model is unavailable.")

        paths: list[Path] = []
        before_hashes: dict[str, str] = {}
        for item in images:
            path = (root / item.staged_image_path).resolve()
            if not path.is_file():
                raise ReconstructionValidationError(f"I4.3E1 staged image is missing: {path}")
            digest = _sha256_file(path)
            if digest != item.staged_sha256:
                raise ReconstructionValidationError(f"I4.3E1 staged image checksum drifted before execution: {item.keyframe_id}")
            paths.append(path)
            before_hashes[item.keyframe_id] = digest

        context.output_dir.mkdir(parents=True, exist_ok=True)
        provider_dir = context.output_dir / "provider"
        provider_dir.mkdir(parents=True, exist_ok=False)
        input_manifest = provider_dir / "input_manifest.json"
        _write_json(
            input_manifest,
            {
                "schema_version": I43E1_SCHEMA_VERSION,
                "phase": I43E1_PHASE,
                "request_id": f"i4-3e1-{context.sfm_result.session_id}",
                "session_id": context.sfm_result.session_id,
                "authoritative_sfm_frame_id": context.reconstruction_frame_id,
                "model_id": METRIC3D_V2_MODEL_ID,
                "source_revision": self.environment.source_revision,
                "device": context.device,
                "precision": context.precision,
                "runtime_config": metadata_as_dict(context.runtime_config),  # type: ignore[arg-type]
                "all_registered_views_selected": True,
                "permanent_global_view_cap": None,
                "sequence": [
                    {
                        "order": index,
                        "keyframe_id": item.keyframe_id,
                        "frame_id": item.frame_id,
                        "source_sequence_index": item.source_sequence_index,
                        "timestamp_ns": item.timestamp_ns,
                        "image_path": str(path),
                        "image_sha256": item.staged_sha256,
                        "width": item.image_size.width,
                        "height": item.image_size.height,
                    }
                    for index, (item, path) in enumerate(zip(images, paths, strict=True))
                ],
                "authority": {
                    "camera_pose_source": I43A_AUTHORITATIVE_FRAME_OWNER,
                    "model_estimated_intrinsics_are_canonical_authority": False,
                    "model_metric_depth_establishes_metric_scale": False,
                },
            },
        )

        authority_before = _sfm_authority_digest(context.sfm_result)
        total_views = len(images)
        self._progress_event_count = 0
        self._publish(stage=DenseProviderProgressStage.PREPARING, detail="validated_metric3d_sfm_inputs", progress=0.02, processed=0, total=total_views)
        self._publish(stage=DenseProviderProgressStage.VIEW_SELECTION, detail="all_registered_views_selected_i4_3e1", progress=0.05, processed=0, total=total_views)
        self._publish(stage=DenseProviderProgressStage.PROVIDER_EXECUTION, detail="external_metric3d_single_image_inference", progress=0.10, processed=0, total=total_views)
        worker = self._runner.run(input_manifest, provider_dir / "worker_output")
        payload = dict(worker.metadata)
        if payload.get("status") != "pass" or payload.get("phase") != I43E1_PHASE:
            raise ReconstructionValidationError("I4.3E1 worker result did not report a passing I4.3E1 run.")
        if payload.get("model_id") != METRIC3D_V2_MODEL_ID:
            raise ReconstructionValidationError("I4.3E1 worker model identity drifted.")
        if str(payload.get("source_revision", "")) != self.environment.source_revision:
            raise ReconstructionValidationError("I4.3E1 worker source revision does not match the probed runtime.")
        records = payload.get("frame_records")
        if not isinstance(records, list) or len(records) != total_views:
            raise ReconstructionValidationError("I4.3E1 worker must return one frame record per registered view.")
        self._publish(stage=DenseProviderProgressStage.PROVIDER_EXECUTION, detail="metric3d_depth_evidence_ready", progress=0.35, processed=total_views, total=total_views)

        scene_scale = _robust_scene_diagonal(context.sfm_result)
        voxel_size = max(scene_scale * context.config.voxel_scene_fraction, 1e-8)
        reconstruction_camera = reconstruction_camera_from_sfm_model(
            camera_model,
            keyframe_id=images[0].keyframe_id,
            reconstruction_frame_id=context.reconstruction_frame_id,
        )
        camera_profile = camera_geometry_from_estimated_camera(
            profile_id=f"i43e1-{camera_id}-sfm-estimated",
            camera=reconstruction_camera,
            focal_length_relative_uncertainty=0.20,
            notes="I4.3E1 camera geometry from authoritative PyCOLMAP SfM; arbitrary gauge only.",
        )
        scale_profile = arbitrary_scale_reference(
            profile_id=f"i43e1-{context.sfm_result.session_id[:12]}-arbitrary-sfm-gauge",
            notes=(
                "I4.3E1 aligns Metric3D v2 depth into the arbitrary SfM gauge; native metric depth "
                "and estimated intrinsics are not promoted to physical/camera authority."
            ),
        )

        frames: list[DenseFusionFrame] = []
        alignment_records: list[dict[str, Any]] = []
        rejected: list[str] = []
        worker_dir = worker.manifest_path.parent
        for index, (item, record) in enumerate(zip(images, records, strict=True)):
            if not isinstance(record, Mapping):
                raise ReconstructionValidationError("I4.3E1 worker frame records must be objects.")
            if record.get("keyframe_id") != item.keyframe_id or int(record.get("order", -1)) != index:
                raise ReconstructionValidationError("I4.3E1 worker frame ordering/keyframe identity drifted.")
            if record.get("native_depth_scale") != "metric" or record.get("native_is_metric") is not True:
                raise ReconstructionValidationError("I4.3E1 worker changed Metric3D native metric-depth semantics.")
            if record.get("intrinsics_source_native") != "estimated":
                raise ReconstructionValidationError("I4.3E1 worker changed Metric3D estimated-intrinsics semantics.")
            if record.get("confidence_kind_native") != "confidence":
                raise ReconstructionValidationError("I4.3E1 worker changed Metric3D confidence semantics.")
            evidence_path = Path(str(record.get("evidence_npz_path", ""))).expanduser()
            if not evidence_path.is_absolute():
                evidence_path = (worker_dir / evidence_path).resolve()
            if not evidence_path.is_file():
                raise ReconstructionValidationError(f"I4.3E1 worker evidence file is missing: {evidence_path}")
            try:
                with np.load(evidence_path, allow_pickle=False) as archive:
                    depth = np.asarray(archive["depth"], dtype=np.float32)
                    confidence = np.asarray(archive["confidence"], dtype=np.float32)
                    valid_mask = np.asarray(archive["valid_mask"], dtype=bool)
            except Exception as exc:  # noqa: BLE001
                raise ReconstructionValidationError(f"I4.3E1 could not read worker evidence {evidence_path}: {exc}") from exc
            expected_hw = (item.image_size.height, item.image_size.width)
            if depth.shape != expected_hw or confidence.shape != expected_hw or valid_mask.shape != expected_hw:
                raise ReconstructionValidationError("I4.3E1 worker evidence must remain on each authoritative SfM pixel grid.")
            if not np.isfinite(depth[valid_mask]).all() or np.any(depth[valid_mask] <= 0.0):
                raise ReconstructionValidationError("I4.3E1 valid Metric3D depth evidence must be finite and positive.")
            if not np.isfinite(confidence).all():
                raise ReconstructionValidationError("I4.3E1 Metric3D native confidence must be finite.")

            evidence = align_sfm_keyframe_dense_depth(
                context.sfm_result,
                keyframe_id=item.keyframe_id,
                dense_depth_source_units=depth,
                depth_provenance=(
                    f"Metric3D-v2 source_revision={self.environment.source_revision};I4.3E1;"
                    "single_image=true;native_metric=true;authoritative_pose=pycolmap_sfm;"
                    "native_metric_depth_not_promoted=true;estimated_intrinsics_not_canonical=true"
                ),
                minimum_correspondences=context.config.minimum_correspondences,
                maximum_relative_residual=context.config.maximum_relative_residual,
            )
            alignment_record = evidence.as_dict()
            alignment_record["native_metric_depth_promoted_to_canonical_metric_scale"] = False
            alignment_record["estimated_model_intrinsics_consumed_for_canonical_projection"] = False
            alignment_record["native_confidence_consumed_as_point_confidence"] = False
            alignment_record["native_confidence_range"] = "unbounded_score"
            alignment_record["native_confidence_min"] = float(np.min(confidence))
            alignment_record["native_confidence_max"] = float(np.max(confidence))
            alignment_records.append(alignment_record)
            if not evidence.is_accepted or evidence.alignment is None:
                rejected.append(item.keyframe_id)
            else:
                bgr = cv2.imread(str(paths[index]), cv2.IMREAD_COLOR)
                if bgr is None:
                    raise ReconstructionValidationError(f"I4.3E1 could not decode staged RGB: {paths[index]}")
                rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)
                frames.append(
                    DenseFusionFrame(
                        frame_id=camera_by_keyframe[item.keyframe_id].frame_id,
                        timestamp_ns=item.timestamp_ns,
                        rgb=rgb,
                        depth=depth,
                        T_reconstruction_from_camera=camera_by_keyframe[item.keyframe_id].T_reconstruction_from_camera,
                        depth_unit=DenseDepthUnit.SOURCE_MODEL_UNITS,
                        depth_alignment=evidence.alignment,
                        # Metric3D v2 exposes an unbounded native confidence score.
                        # Preserve it in worker/alignment evidence, but do not invent a
                        # normalization mapping for canonical [0,1] fusion weights.
                        confidence=None,
                        valid_mask=valid_mask,
                        pose_layer="offline_sfm_bundle_adjusted",
                    )
                )
            self._publish(
                stage=DenseProviderProgressStage.ALIGNMENT,
                detail="align_metric3d_depth_to_authoritative_sfm",
                progress=0.40 + 0.35 * ((index + 1) / float(total_views)),
                processed=index + 1,
                total=total_views,
            )

        if len(frames) < context.config.minimum_accepted_views:
            raise ReconstructionValidationError(
                "I4.3E1 insufficient Metric3D/SfM alignment-accepted views: "
                f"accepted={len(frames)} required={context.config.minimum_accepted_views}."
            )

        self._publish(stage=DenseProviderProgressStage.FUSION_FILTERING, detail="qualified_metric3d_voxel_fusion", progress=0.82, processed=len(frames), total=total_views)
        fusion = fuse_dense_frames(
            frames,
            session_id=f"{context.sfm_result.session_id}:i4.3e1-metric3d-v2",
            camera_profile=camera_profile,
            scale_profile=scale_profile,
            policy=DenseFusionPolicy(
                scene_profile=DenseSceneProfile.SPATIAL_DIGITAL_TWIN,
                voxel_size_reconstruction_units=voxel_size,
                pixel_stride=context.config.pixel_stride,
                minimum_view_observations=1,
                minimum_confidence=0.0,
                max_voxels=context.config.max_voxels,
            ),
            reconstruction_frame_id=context.reconstruction_frame_id,
        )
        cloud = fusion.point_cloud
        canonical_npz = context.output_dir / "canonical_metric3d_v2_fused.npz"
        np.savez_compressed(
            canonical_npz,
            points_reconstruction=cloud.points_reconstruction,
            colors_rgb=cloud.colors_rgb,
            view_observation_counts=cloud.view_observation_counts,
            mean_confidence=cloud.mean_confidence,
            voxel_size_reconstruction_units=np.asarray(cloud.voxel_size_reconstruction_units, dtype=np.float64),
        )
        canonical_ply = _write_binary_ply(
            context.output_dir / "canonical_metric3d_v2_fused.ply",
            cloud.points_reconstruction,
            cloud.colors_rgb,
        )

        for item, path in zip(images, paths, strict=True):
            if _sha256_file(path) != before_hashes[item.keyframe_id]:
                raise ReconstructionValidationError("I4.3E1 registered staged images changed during provider execution.")
        if _sfm_authority_digest(context.sfm_result) != authority_before:
            raise ReconstructionValidationError("I4.3E1 authoritative SfM state changed during provider execution.")

        degradation_reasons: list[str] = []
        if rejected:
            degradation_reasons.append(f"alignment_rejected_views:{len(rejected)}")
        status = DenseProviderRunStatus.DEGRADED if degradation_reasons else DenseProviderRunStatus.COMPLETED
        self._publish(stage=DenseProviderProgressStage.FINALIZATION, detail="persist_metric3d_canonical_candidate", progress=1.0, processed=len(frames), total=total_views)

        result = DenseProviderExecutionResult(
            provider_id=I43E1_PROVIDER_ID,
            implementation=DenseProviderImplementation.METRIC3D_V2,
            status=status,
            source_view_count=total_views,
            selected_view_count=total_views,
            adaptive_view_selection=False,
            view_selection_reason="all_registered_views_i4_3e1_adaptive_selection_deferred_i4_3f",
            permanent_global_view_cap=None,
            canonical_geometry_ref=str(canonical_npz.resolve()),
            canonical_frame_id=context.reconstruction_frame_id,
            native_artifact_refs=(str(worker.manifest_path.resolve()), str(canonical_ply.resolve())),
            degradation_reasons=tuple(degradation_reasons),
            progress_event_count=self._progress_event_count,
            camera_pose_authority_owner=I43A_AUTHORITATIVE_FRAME_OWNER,
            metric_scale_claimed=False,
            measurement_authoritative=False,
            certified_metrology=False,
            surface_mesh_generated=False,
            production_filter_policy_changed=False,
            provider_promotion_performed=False,
        )
        _write_json(
            context.output_dir / "i4_3e1_metric3d_v2_execution_summary.json",
            {
                "schema_version": I43E1_SCHEMA_VERSION,
                "phase": I43E1_PHASE,
                "status": "pass",
                "provider": self.descriptor.as_dict(),
                "execution": result.as_dict(),
                "worker": payload,
                "config": context.config.as_dict(),
                "alignment": {
                    "accepted_views": len(frames),
                    "rejected_views": rejected,
                    "records": alignment_records,
                    "robust_sparse_scene_diagonal_reconstruction_units": scene_scale,
                },
                "integrity": {
                    "authoritative_sfm_digest_before": authority_before,
                    "authoritative_sfm_digest_after": _sfm_authority_digest(context.sfm_result),
                    "registered_image_hashes_preserved": True,
                    "canonical_pose_layer": "offline_sfm_bundle_adjusted",
                },
                "authority": {
                    "camera_pose_authority_owner": I43A_AUTHORITATIVE_FRAME_OWNER,
                    "native_metric_depth_promoted_to_canonical_metric_scale": False,
                    "estimated_model_intrinsics_consumed_for_canonical_projection": False,
                    "native_confidence_consumed_as_point_confidence": False,
                    "native_confidence_normalization_invented": False,
                    "metric_scale_claimed": False,
                    "measurement_authoritative": False,
                    "certified_metrology": False,
                    "surface_mesh_generated": False,
                    "provider_promoted": False,
                },
            },
        )
        return result


def validate_i4_3e1_binding(
    *,
    i4_3a_snapshot: Mapping[str, Any],
    i4_3d3_snapshot: Mapping[str, Any],
    environment: Metric3DV2EnvironmentProbe,
) -> dict[str, Any]:
    if not isinstance(i4_3a_snapshot, Mapping) or not isinstance(i4_3d3_snapshot, Mapping):
        raise TypeError("I4.3E1 binding snapshots must be mappings.")
    if i4_3a_snapshot.get("phase") != "I4.3A" or i4_3a_snapshot.get("status") != "pass" or i4_3a_snapshot.get("pass") is not True:
        raise ReconstructionValidationError("I4.3E1 requires the passing I4.3A architecture snapshot.")
    architecture = i4_3a_snapshot.get("architecture")
    if not isinstance(architecture, Mapping):
        raise ReconstructionValidationError("I4.3E1 I4.3A snapshot is missing architecture.")
    if architecture.get("authoritative_camera_frame_owner") != I43A_AUTHORITATIVE_FRAME_OWNER:
        raise ReconstructionValidationError("I4.3E1 I4.3A camera-frame authority drifted.")
    contracts = architecture.get("provider_implementation_contracts")
    if not isinstance(contracts, list):
        raise ReconstructionValidationError("I4.3E1 I4.3A implementation contracts are missing.")
    match = next((item for item in contracts if isinstance(item, Mapping) and item.get("implementation") == "metric3d_v2"), None)
    if not isinstance(match, Mapping):
        raise ReconstructionValidationError("I4.3E1 Metric3D v2 implementation contract is missing.")
    if match.get("integration_stage") != "I4.3E" or match.get("family") != "single_image_depth":
        raise ReconstructionValidationError("I4.3E1 Metric3D v2 implementation contract drifted.")
    if match.get("frame_binding") != "align_to_authoritative_sfm" or match.get("native_output") != "depth_evidence":
        raise ReconstructionValidationError("I4.3E1 single-image depth frame/output contract drifted.")

    if i4_3d3_snapshot.get("phase") != "I4.3D3" or i4_3d3_snapshot.get("status") != "pass" or i4_3d3_snapshot.get("pass") is not True:
        raise ReconstructionValidationError("I4.3E1 requires the passing I4.3D3 temporal-provider closure snapshot.")
    decision = i4_3d3_snapshot.get("decision")
    authority = i4_3d3_snapshot.get("authority")
    if not isinstance(decision, Mapping) or not isinstance(authority, Mapping):
        raise ReconstructionValidationError("I4.3E1 I4.3D3 closure snapshot is incomplete.")
    if decision.get("i4_3d_temporal_provider_architecture_closed") is not True or decision.get("i4_3e_authorized") is not True:
        raise ReconstructionValidationError("I4.3E1 is not authorized by the I4.3D3 closure decision.")
    if any(
        authority.get(key) is True
        for key in ("metric_scale_claimed", "measurement_authoritative", "certified_metrology", "surface_mesh_generated")
    ):
        raise ReconstructionValidationError("I4.3E1 refuses an I4.3D3 snapshot with promoted authority claims.")

    descriptor = metric3d_v2_dense_provider_descriptor(environment)
    if not descriptor.available:
        raise ReconstructionValidationError("I4.3E1 Metric3D v2 runtime is not ready for binding.")
    return {
        "schema_version": I43E1_SCHEMA_VERSION,
        "phase": I43E1_PHASE,
        "status": "pass",
        "pass": True,
        "provider": descriptor.as_dict(),
        "architecture": {
            "family": "single_image_depth",
            "implementation": "metric3d_v2",
            "frame_binding": "align_to_authoritative_sfm",
            "native_output": "depth_evidence",
            "native_depth_scale": "metric",
            "native_intrinsics_source": "estimated",
            "native_confidence_kind": "confidence",
            "native_confidence_range": "unbounded_score",
            "native_confidence_consumed_as_point_confidence": False,
            "native_confidence_normalization_invented": False,
            "per_view_alignment": "sparse_correspondence_scale_to_authoritative_sfm",
            "all_registered_views_available_by_default": True,
            "permanent_global_view_cap": None,
            "adaptive_view_selection_deferred_to": "I4.3F",
        },
        "authority": {
            "camera_pose_authority_owner": I43A_AUTHORITATIVE_FRAME_OWNER,
            "model_metric_depth_promoted_to_canonical_metric_scale": False,
            "estimated_model_intrinsics_consumed_for_canonical_projection": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
            "provider_promoted": False,
        },
        "qualification": {
            "state": "candidate",
            "universal_provider_winner_assumed": False,
            "cross_provider_qualification_pending_i4_3g": True,
            "remaining_i4_3e_providers_pending": [
                "unidepth_v2",
                "moge_2",
                "depth_pro",
                "depth_anything_v2",
                "marigold",
            ],
        },
        "environment": environment.as_dict(),
        "next": "I4.3E2 UniDepthV2 production single-image-depth provider adapter.",
    }


__all__ = [
    "I43E1_ADAPTER_VERSION",
    "I43E1_PHASE",
    "I43E1_PROVIDER_ID",
    "Metric3DV2EnvironmentProbe",
    "Metric3DV2ProductionConfig",
    "Metric3DV2ProductionProviderAdapter",
    "Metric3DV2ProviderRunContext",
    "Metric3DV2WorkerResult",
    "SubprocessMetric3DV2WorkerRunner",
    "metric3d_v2_dense_provider_descriptor",
    "probe_metric3d_v2_environment",
    "validate_i4_3e1_binding",
]

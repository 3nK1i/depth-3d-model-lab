"""I4.3D1 canonical production adapter for Video Depth Anything.

Video Depth Anything (VDA) is bound as a temporal-depth provider behind the
I4.3A dense-provider contract.  The existing stateful VDA model adapter remains
the native depth runtime.  I4.3D1 executes that runtime in an isolated worker
process, preserves source temporal order, aligns each temporal depth map into the
existing authoritative PyCOLMAP SfM gauge, and projects/fuses only with the
bundle-adjusted PyCOLMAP poses.

The model's own metric-depth / estimated-intrinsics semantics are native model
evidence only.  They are deliberately *not* promoted to physical metric scale or
camera authority in the canonical reconstruction.  I4.3D1 uses every registered
SfM view and introduces no permanent view cap; adaptive selection belongs to
I4.3F and temporal cross-provider quality qualification belongs to I4.3G/I5.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from math import isfinite
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Mapping, Protocol, Sequence

import cv2
import numpy as np

from core.contracts.validation import JsonValue, metadata_as_dict, normalize_metadata, require_nonempty
from core.models.catalog import VIDEO_DEPTH_ANYTHING_MODEL_ID
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
    DenseProviderSelectionRequest,
)
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.orchestration import ReconstructionProviderExecution
from core.reconstruction.scale_reference import arbitrary_scale_reference
from core.reconstruction.sfm_contracts import OfflineSfmResult, SfmWorkspaceImage
from core.reconstruction.sfm_dense_alignment import align_sfm_keyframe_dense_depth
from core.reconstruction.sfm_dense_projection import reconstruction_camera_from_sfm_model
from core.reconstruction.sfm_pycolmap import PYCOLMAP_BACKEND_ID

I43D1_SCHEMA_VERSION = 1
I43D1_PHASE = "I4.3D1"
I43D1_ADAPTER_VERSION = "i4.3d1-v1"
I43D1_PROVIDER_ID = "video-depth-anything-temporal"
I43D1_DEFAULT_VOXEL_SCENE_FRACTION = 0.004
I43D1_DEFAULT_MINIMUM_CORRESPONDENCES = 8
I43D1_DEFAULT_MAXIMUM_RELATIVE_RESIDUAL = 0.35
I43D1_DEFAULT_PIXEL_STRIDE = 2
I43D1_DEFAULT_MAX_VOXELS = 1_500_000
I43D1_DEFAULT_MINIMUM_ACCEPTED_VIEWS = 2
I43D1_DEFAULT_SOURCE_DIR = "third_party/Video-Depth-Anything"
I43D1_DEFAULT_READY_MARKER = "third_party/Video-Depth-Anything/.depth3d_video_depth_anything_ready.json"
I43D1_DEFAULT_CHECKPOINT = "checkpoints/video-depth-anything/metric_video_depth_anything_vits.pth"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError(f"I4.3D1 could not read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"I4.3D1 JSON must contain an object: {path}")
    return payload


def _require_clean_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise ReconstructionValidationError(
            f"I4.3D1 output directory must be empty or absent; refusing to overwrite provider evidence: {path}"
        )


def _write_binary_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> Path:
    xyz = np.asarray(points, dtype=np.float32)
    rgb = np.asarray(colors, dtype=np.uint8)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or rgb.shape != (len(xyz), 3):
        raise ReconstructionValidationError("I4.3D1 PLY geometry must be aligned Nx3 XYZ/RGB arrays.")
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


def _robust_scene_diagonal(sfm_result: OfflineSfmResult) -> float:
    if len(sfm_result.landmarks) < 4:
        raise ReconstructionValidationError("I4.3D1 requires sparse SfM landmarks for scene-relative fusion scale.")
    points = np.vstack([np.asarray(item.position_reconstruction, dtype=np.float64) for item in sfm_result.landmarks])
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ReconstructionValidationError("I4.3D1 sparse SfM landmarks are invalid.")
    low = np.quantile(points, 0.02, axis=0)
    high = np.quantile(points, 0.98, axis=0)
    diagonal = float(np.linalg.norm(high - low))
    if not isfinite(diagonal) or diagonal <= 0.0:
        raise ReconstructionValidationError("I4.3D1 robust sparse scene diagonal is invalid.")
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


def _ordered_registered_inputs(sfm_result: OfflineSfmResult) -> tuple[tuple[SfmWorkspaceImage, ...], tuple[Path, ...]]:
    if sfm_result.backend_id != PYCOLMAP_BACKEND_ID:
        raise ReconstructionValidationError("I4.3D1 requires the authoritative PyCOLMAP SfM result.")
    registered = {camera.keyframe_id for camera in sfm_result.cameras}
    if not registered:
        raise ReconstructionValidationError("I4.3D1 requires registered SfM cameras.")
    images = tuple(item for item in sfm_result.workspace.images if item.keyframe_id in registered)
    if len(images) != len(registered):
        raise ReconstructionValidationError("I4.3D1 workspace/registered SfM image coverage is incomplete.")
    ordered = tuple(sorted(images, key=lambda item: (item.source_sequence_index, item.timestamp_ns, item.order)))
    sequence_indices = [item.source_sequence_index for item in ordered]
    if len(set(sequence_indices)) != len(sequence_indices):
        raise ReconstructionValidationError("I4.3D1 temporal provider requires unique source_sequence_index values.")
    if sequence_indices != sorted(sequence_indices):
        raise ReconstructionValidationError("I4.3D1 temporal source ordering is invalid.")
    timestamps = [item.timestamp_ns for item in ordered]
    if any(b <= a for a, b in zip(timestamps, timestamps[1:])):
        raise ReconstructionValidationError("I4.3D1 temporal provider requires strictly increasing canonical timestamps.")
    root = Path(sfm_result.workspace.workspace_root).expanduser().resolve()
    paths = tuple((root / item.staged_image_path).resolve() for item in ordered)
    for item, path in zip(ordered, paths, strict=True):
        if not path.is_file():
            raise ReconstructionValidationError(f"I4.3D1 staged registered image is missing: {path}")
        if _sha256_file(path) != item.staged_sha256:
            raise ReconstructionValidationError(f"I4.3D1 staged image hash changed before provider execution: {item.keyframe_id}")
    return ordered, paths


@dataclass(frozen=True, slots=True)
class VideoDepthAnythingEnvironmentProbe:
    python_executable: str
    project_root: str
    source_directory: str
    ready_marker: str
    checkpoint_path: str
    source_revision: str
    source_ready: bool
    checkpoint_ready: bool
    adapter_importable: bool
    upstream_importable: bool
    torch_available: bool
    cuda_available: bool
    device_names: tuple[str, ...] = ()
    error: str = ""
    schema_version: int = I43D1_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if int(self.schema_version) != I43D1_SCHEMA_VERSION:
            raise ReconstructionValidationError("Unsupported I4.3D1 environment-probe schema version.")
        revision = str(self.source_revision).strip()
        if revision and (len(revision) != 40 or any(ch not in "0123456789abcdef" for ch in revision.lower())):
            raise ReconstructionValidationError("I4.3D1 VDA source revision must be a 40-character git digest when present.")
        object.__setattr__(self, "source_revision", revision.lower())
        object.__setattr__(self, "device_names", tuple(str(value).strip() for value in self.device_names if str(value).strip()))
        object.__setattr__(self, "error", str(self.error).strip())
        object.__setattr__(self, "schema_version", I43D1_SCHEMA_VERSION)

    @property
    def available(self) -> bool:
        return bool(
            self.source_ready
            and bool(self.source_revision)
            and self.checkpoint_ready
            and self.adapter_importable
            and self.upstream_importable
            and self.torch_available
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "python_executable": self.python_executable,
            "project_root": self.project_root,
            "source_directory": self.source_directory,
            "ready_marker": self.ready_marker,
            "checkpoint_path": self.checkpoint_path,
            "source_revision": self.source_revision,
            "source_ready": self.source_ready,
            "checkpoint_ready": self.checkpoint_ready,
            "adapter_importable": self.adapter_importable,
            "upstream_importable": self.upstream_importable,
            "torch_available": self.torch_available,
            "cuda_available": self.cuda_available,
            "device_names": list(self.device_names),
            "available": self.available,
            "error": self.error,
        }


def probe_video_depth_anything_environment(
    project_root: str | Path,
    *,
    python_executable: str | Path | None = None,
    checkpoint: str | Path = I43D1_DEFAULT_CHECKPOINT,
) -> VideoDepthAnythingEnvironmentProbe:
    root = Path(project_root).expanduser().resolve()
    python_path = Path(python_executable or sys.executable).expanduser().resolve()
    source = (root / I43D1_DEFAULT_SOURCE_DIR).resolve()
    marker = (root / I43D1_DEFAULT_READY_MARKER).resolve()
    checkpoint_path = Path(checkpoint).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = (root / checkpoint_path).resolve()
    source_ready = (source / "video_depth_anything" / "video_depth_stream.py").is_file() and marker.is_file()
    source_revision = ""
    errors: list[str] = []
    if marker.is_file():
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                source_revision = str(payload.get("commit", "")).strip().lower()
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"ready_marker_invalid:{exc}")
    if not source_ready:
        errors.append("official_source_not_ready")
    if not source_revision:
        errors.append("source_revision_missing")
    checkpoint_ready = checkpoint_path.is_file()
    if not checkpoint_ready:
        errors.append("metric_checkpoint_missing")

    adapter_importable = False
    upstream_importable = False
    torch_available = False
    cuda_available = False
    device_names: tuple[str, ...] = ()
    if python_path.is_file():
        script = """
import json, sys
payload = {'adapter': False, 'upstream': False, 'torch': False, 'cuda': False, 'devices': []}
try:
    from core.models.adapters.video_depth_anything_adapter import VideoDepthAnythingGeometryModel
    payload['adapter'] = VideoDepthAnythingGeometryModel is not None
except Exception as exc:
    payload['adapter_error'] = repr(exc)
try:
    from pathlib import Path
    source = Path(r'''SOURCE_DIR''').resolve()
    sys.path.insert(0, str(source))
    from video_depth_anything.video_depth_stream import VideoDepthAnything
    payload['upstream'] = VideoDepthAnything is not None
except Exception as exc:
    payload['upstream_error'] = repr(exc)
try:
    import torch
    payload['torch'] = True
    payload['cuda'] = bool(torch.cuda.is_available())
    if payload['cuda']:
        payload['devices'] = [str(torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]
except Exception as exc:
    payload['torch_error'] = repr(exc)
print(json.dumps(payload))
"""
        script = script.replace("SOURCE_DIR", str(source))
        try:
            completed = subprocess.run(
                [str(python_path), "-c", script],
                cwd=str(root),
                text=True,
                capture_output=True,
                check=False,
                timeout=60,
            )
            if completed.returncode == 0 and completed.stdout.strip():
                payload = json.loads(completed.stdout.strip().splitlines()[-1])
                adapter_importable = bool(payload.get("adapter"))
                upstream_importable = bool(payload.get("upstream"))
                torch_available = bool(payload.get("torch"))
                cuda_available = bool(payload.get("cuda"))
                device_names = tuple(str(value) for value in payload.get("devices", ()))
                if not adapter_importable:
                    errors.append("adapter_import_failed")
                if not upstream_importable:
                    errors.append("upstream_import_failed")
                if not torch_available:
                    errors.append("torch_import_failed")
            else:
                errors.append(f"runtime_probe_failed:{completed.stderr.strip() or completed.returncode}")
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
            errors.append(f"runtime_probe_failed:{exc}")
    else:
        errors.append("python_executable_missing")

    return VideoDepthAnythingEnvironmentProbe(
        python_executable=str(python_path),
        project_root=str(root),
        source_directory=str(source),
        ready_marker=str(marker),
        checkpoint_path=str(checkpoint_path),
        source_revision=source_revision,
        source_ready=source_ready,
        checkpoint_ready=checkpoint_ready,
        adapter_importable=adapter_importable,
        upstream_importable=upstream_importable,
        torch_available=torch_available,
        cuda_available=cuda_available,
        device_names=device_names,
        error=";".join(errors),
    )


def video_depth_anything_dense_provider_descriptor(probe: VideoDepthAnythingEnvironmentProbe) -> DenseProviderDescriptor:
    if not isinstance(probe, VideoDepthAnythingEnvironmentProbe):
        raise TypeError("probe must be VideoDepthAnythingEnvironmentProbe.")
    reason = "" if probe.available else (probe.error or "video_depth_anything_runtime_unavailable")
    return DenseProviderDescriptor(
        provider_id=I43D1_PROVIDER_ID,
        implementation=DenseProviderImplementation.VIDEO_DEPTH_ANYTHING,
        execution=ReconstructionProviderExecution.EXTERNAL_PROCESS,
        backend_version=probe.source_revision[:12] if probe.source_revision else "unknown",
        adapter_version=I43D1_ADAPTER_VERSION,
        qualification_state=DenseProviderQualificationState.CANDIDATE,
        available=probe.available,
        unavailable_reason=reason,
        supports_engineer_override=True,
        metadata={
            "phase": I43D1_PHASE,
            "native_output": "depth_evidence",
            "frame_binding": "align_to_authoritative_sfm",
            "temporal_mode": "official_streaming_cache",
            "stateful_sequence": True,
            "sequence_order_source": "sfm_workspace.source_sequence_index",
            "canonical_projection_pose_source": I43A_AUTHORITATIVE_FRAME_OWNER,
            "model_metric_depth_promoted_to_canonical_metric_scale": False,
            "estimated_model_intrinsics_consumed_for_canonical_projection": False,
            "all_registered_views_used_in_i4_3d1": True,
            "adaptive_view_selection_deferred_to": "I4.3F",
            "permanent_global_view_cap": None,
            "temporal_quality_comparison_pending": "I4.3G/I5",
            "provider_promotion_performed": False,
        },
    )


@dataclass(frozen=True, slots=True)
class VideoDepthAnythingProductionConfig:
    minimum_correspondences: int = I43D1_DEFAULT_MINIMUM_CORRESPONDENCES
    maximum_relative_residual: float = I43D1_DEFAULT_MAXIMUM_RELATIVE_RESIDUAL
    voxel_scene_fraction: float = I43D1_DEFAULT_VOXEL_SCENE_FRACTION
    pixel_stride: int = I43D1_DEFAULT_PIXEL_STRIDE
    max_voxels: int = I43D1_DEFAULT_MAX_VOXELS
    minimum_accepted_views: int = I43D1_DEFAULT_MINIMUM_ACCEPTED_VIEWS

    def __post_init__(self) -> None:
        if int(self.minimum_correspondences) < 3:
            raise ReconstructionValidationError("I4.3D1 minimum_correspondences must be >= 3.")
        residual = float(self.maximum_relative_residual)
        if not isfinite(residual) or not 0.0 < residual < 1.0:
            raise ReconstructionValidationError("I4.3D1 maximum_relative_residual must be in (0,1).")
        fraction = float(self.voxel_scene_fraction)
        if not isfinite(fraction) or fraction <= 0.0:
            raise ReconstructionValidationError("I4.3D1 voxel_scene_fraction must be positive.")
        if int(self.pixel_stride) <= 0 or int(self.max_voxels) <= 0 or int(self.minimum_accepted_views) <= 0:
            raise ReconstructionValidationError("I4.3D1 pixel_stride/max_voxels/minimum_accepted_views must be positive.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "minimum_correspondences": int(self.minimum_correspondences),
            "maximum_relative_residual": float(self.maximum_relative_residual),
            "voxel_scene_fraction": float(self.voxel_scene_fraction),
            "pixel_stride": int(self.pixel_stride),
            "max_voxels": int(self.max_voxels),
            "minimum_accepted_views": int(self.minimum_accepted_views),
        }


@dataclass(frozen=True, slots=True)
class VideoDepthAnythingProviderRunContext:
    sfm_result: OfflineSfmResult
    reconstruction_frame_id: str
    output_dir: Path
    python_executable: Path
    worker_script: Path
    runtime_config: Mapping[str, JsonValue] | Sequence[tuple[str, JsonValue]]
    device: str
    precision: str
    config: VideoDepthAnythingProductionConfig = field(default_factory=VideoDepthAnythingProductionConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.sfm_result, OfflineSfmResult):
            raise TypeError("sfm_result must be OfflineSfmResult.")
        frame = require_nonempty(self.reconstruction_frame_id, field_name="reconstruction_frame_id")
        device = require_nonempty(self.device, field_name="device")
        precision = require_nonempty(self.precision, field_name="precision").lower()
        if precision not in {"fp16", "fp32"}:
            raise ReconstructionValidationError("I4.3D1 precision must be fp16 or fp32.")
        if not isinstance(self.config, VideoDepthAnythingProductionConfig):
            raise TypeError("config must be VideoDepthAnythingProductionConfig.")
        object.__setattr__(self, "reconstruction_frame_id", frame)
        object.__setattr__(self, "output_dir", Path(self.output_dir).expanduser().resolve())
        object.__setattr__(self, "python_executable", Path(self.python_executable).expanduser().resolve())
        object.__setattr__(self, "worker_script", Path(self.worker_script).expanduser().resolve())
        object.__setattr__(self, "runtime_config", normalize_metadata(self.runtime_config))
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "precision", precision)


@dataclass(frozen=True, slots=True)
class VideoDepthAnythingWorkerResult:
    manifest_path: Path
    depth_npz_path: Path
    depth: np.ndarray
    metadata: Mapping[str, Any]


class VideoDepthAnythingWorkerRunner(Protocol):
    def run(self, manifest_path: Path, output_dir: Path) -> VideoDepthAnythingWorkerResult: ...


class ExternalVideoDepthAnythingWorkerRunner:
    def __init__(self, *, python_executable: Path, worker_script: Path, project_root: Path) -> None:
        self.python_executable = Path(python_executable).expanduser().resolve()
        self.worker_script = Path(worker_script).expanduser().resolve()
        self.project_root = Path(project_root).expanduser().resolve()

    def run(self, manifest_path: Path, output_dir: Path) -> VideoDepthAnythingWorkerResult:
        output_dir.mkdir(parents=True, exist_ok=False)
        completed = subprocess.run(
            [
                str(self.python_executable),
                str(self.worker_script),
                "--input-manifest",
                str(manifest_path),
                "--output-dir",
                str(output_dir),
            ],
            cwd=str(self.project_root),
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise ReconstructionValidationError(
                "I4.3D1 isolated VDA worker failed: "
                f"code={completed.returncode} stderr={completed.stderr.strip() or '<empty>'}"
            )
        result_manifest = output_dir / "vda_result_manifest.json"
        if not result_manifest.is_file():
            raise ReconstructionValidationError("I4.3D1 isolated VDA worker did not produce its result manifest.")
        payload = _read_json(result_manifest)
        npz_path = Path(str(payload.get("depth_npz_path", ""))).expanduser().resolve()
        if not npz_path.is_file():
            raise ReconstructionValidationError("I4.3D1 VDA worker result NPZ is missing.")
        try:
            with np.load(npz_path, allow_pickle=False) as archive:
                depth = np.asarray(archive["depth"], dtype=np.float32)
        except (OSError, KeyError, ValueError) as exc:
            raise ReconstructionValidationError(f"I4.3D1 could not read VDA depth NPZ: {exc}") from exc
        return VideoDepthAnythingWorkerResult(
            manifest_path=result_manifest,
            depth_npz_path=npz_path,
            depth=depth,
            metadata=payload,
        )


class VideoDepthAnythingProductionProviderAdapter:
    def __init__(
        self,
        *,
        context: VideoDepthAnythingProviderRunContext,
        environment: VideoDepthAnythingEnvironmentProbe,
        runner: VideoDepthAnythingWorkerRunner | None = None,
        progress_sink: DenseProviderProgressSink | None = None,
    ) -> None:
        if not isinstance(context, VideoDepthAnythingProviderRunContext):
            raise TypeError("context must be VideoDepthAnythingProviderRunContext.")
        if not isinstance(environment, VideoDepthAnythingEnvironmentProbe):
            raise TypeError("environment must be VideoDepthAnythingEnvironmentProbe.")
        self.context = context
        self.environment = environment
        self._progress_sink = progress_sink
        self._progress_event_count = 0
        root = Path(environment.project_root)
        self._runner = runner or ExternalVideoDepthAnythingWorkerRunner(
            python_executable=context.python_executable,
            worker_script=context.worker_script,
            project_root=root,
        )

    @property
    def descriptor(self) -> DenseProviderDescriptor:
        return video_depth_anything_dense_provider_descriptor(self.environment)

    def _publish(self, *, stage: DenseProviderProgressStage, detail: str, progress: float, processed: int, total: int) -> None:
        event = DenseProviderProgressEvent(
            provider_id=self.descriptor.provider_id,
            implementation=DenseProviderImplementation.VIDEO_DEPTH_ANYTHING,
            stage=stage,
            stage_detail=detail,
            progress_fraction=max(0.0, min(1.0, float(progress))),
            processed_views=int(processed),
            total_views=int(total),
        )
        self._progress_event_count += 1
        if self._progress_sink is not None:
            self._progress_sink.publish(event)

    def _validate_selection(self, request: DenseProviderSelectionRequest, decision: DenseProviderSelectionDecision, *, views: int) -> None:
        if request.authoritative_sfm_available is not True:
            raise ReconstructionValidationError("I4.3D1 refuses temporal canonical binding without authoritative SfM.")
        if request.authoritative_sfm_frame_id != self.context.reconstruction_frame_id:
            raise ReconstructionValidationError("I4.3D1 selection frame ID does not match provider context.")
        if request.available_view_count != views:
            raise ReconstructionValidationError("I4.3D1 available_view_count must equal registered-view count.")
        if decision.selected_provider_id != self.descriptor.provider_id:
            raise ReconstructionValidationError("I4.3D1 adapter may execute only when its concrete provider ID was selected.")
        if decision.selected_implementation is not DenseProviderImplementation.VIDEO_DEPTH_ANYTHING:
            raise ReconstructionValidationError("I4.3D1 selection implementation must be Video Depth Anything.")

    def execute(self, request: DenseProviderSelectionRequest, decision: DenseProviderSelectionDecision) -> DenseProviderExecutionResult:
        context = self.context
        if not self.environment.available:
            raise ReconstructionValidationError(f"I4.3D1 VDA runtime is unavailable: {self.environment.error}")
        if not context.python_executable.is_file():
            raise ReconstructionValidationError(f"I4.3D1 Python executable is missing: {context.python_executable}")
        if not context.worker_script.is_file():
            raise ReconstructionValidationError(f"I4.3D1 worker script is missing: {context.worker_script}")
        images, paths = _ordered_registered_inputs(context.sfm_result)
        total_views = len(images)
        self._validate_selection(request, decision, views=total_views)
        _require_clean_output(context.output_dir)

        camera_by_keyframe = {item.keyframe_id: item for item in context.sfm_result.cameras}
        model_by_id = {item.camera_id: item for item in context.sfm_result.camera_models}
        camera_ids = {camera_by_keyframe[item.keyframe_id].camera_id for item in images}
        if len(camera_ids) != 1:
            raise ReconstructionValidationError(
                "I4.3D1 currently requires one shared SfM camera model; mixed-camera temporal fusion is not silently approximated."
            )
        camera_id = next(iter(camera_ids))
        camera_model = model_by_id.get(camera_id)
        if camera_model is None:
            raise ReconstructionValidationError("I4.3D1 registered SfM camera has no camera-model estimate.")
        expected_hw = (camera_model.image_size.height, camera_model.image_size.width)
        if any((item.image_size.height, item.image_size.width) != expected_hw for item in images):
            raise ReconstructionValidationError("I4.3D1 temporal sequence must remain on one stable SfM pixel grid.")

        context.output_dir.mkdir(parents=True, exist_ok=True)
        provider_dir = context.output_dir / "provider"
        provider_dir.mkdir(parents=True, exist_ok=False)
        input_manifest = provider_dir / "input_manifest.json"
        _write_json(
            input_manifest,
            {
                "schema_version": I43D1_SCHEMA_VERSION,
                "phase": I43D1_PHASE,
                "request_id": f"i4-3d1-{context.sfm_result.session_id}",
                "session_id": context.sfm_result.session_id,
                "authoritative_sfm_frame_id": context.reconstruction_frame_id,
                "model_id": VIDEO_DEPTH_ANYTHING_MODEL_ID,
                "device": context.device,
                "precision": context.precision,
                "runtime_config": metadata_as_dict(context.runtime_config),  # type: ignore[arg-type]
                "source_revision": self.environment.source_revision,
                "checkpoint_path": self.environment.checkpoint_path,
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
        images_before = {item.keyframe_id: item.staged_sha256 for item in images}
        self._progress_event_count = 0
        self._publish(stage=DenseProviderProgressStage.PREPARING, detail="validated_temporal_sfm_inputs", progress=0.02, processed=0, total=total_views)
        self._publish(stage=DenseProviderProgressStage.VIEW_SELECTION, detail="all_registered_temporal_views_selected_i4_3d1", progress=0.05, processed=0, total=total_views)
        self._publish(stage=DenseProviderProgressStage.PROVIDER_EXECUTION, detail="isolated_video_depth_anything_stream", progress=0.10, processed=0, total=total_views)
        worker = self._runner.run(input_manifest, provider_dir / "worker_output")
        depth_stack = np.asarray(worker.depth, dtype=np.float32)
        if depth_stack.ndim != 3 or depth_stack.shape[0] != total_views:
            raise ReconstructionValidationError("I4.3D1 VDA depth evidence must be NxHxW for all registered views.")
        if depth_stack.shape[1:] != expected_hw:
            raise ReconstructionValidationError(
                "I4.3D1 VDA worker must restore depth to the authoritative SfM pixel grid; implicit provider-side grid reinterpretation is forbidden."
            )
        if not np.isfinite(depth_stack).all() or np.any(depth_stack < 0.0):
            raise ReconstructionValidationError("I4.3D1 VDA depth evidence must be finite and non-negative.")
        if str(worker.metadata.get("source_revision", "")) != self.environment.source_revision:
            raise ReconstructionValidationError("I4.3D1 VDA worker source revision does not match the probed runtime.")
        if str(worker.metadata.get("model_id", "")) != VIDEO_DEPTH_ANYTHING_MODEL_ID:
            raise ReconstructionValidationError("I4.3D1 VDA worker model identity drifted.")
        self._publish(stage=DenseProviderProgressStage.PROVIDER_EXECUTION, detail="temporal_depth_evidence_ready", progress=0.35, processed=total_views, total=total_views)

        scene_scale = _robust_scene_diagonal(context.sfm_result)
        voxel_size = max(scene_scale * context.config.voxel_scene_fraction, 1e-8)
        reconstruction_camera = reconstruction_camera_from_sfm_model(
            camera_model,
            keyframe_id=images[0].keyframe_id,
            reconstruction_frame_id=context.reconstruction_frame_id,
        )
        camera_profile = camera_geometry_from_estimated_camera(
            profile_id=f"i43d1-{camera_id}-sfm-estimated",
            camera=reconstruction_camera,
            focal_length_relative_uncertainty=0.20,
            notes="I4.3D1 camera geometry from authoritative PyCOLMAP SfM; arbitrary gauge only.",
        )
        scale_profile = arbitrary_scale_reference(
            profile_id=f"i43d1-{context.sfm_result.session_id[:12]}-arbitrary-sfm-gauge",
            notes="I4.3D1 aligns VDA temporal depth into the arbitrary SfM gauge; VDA metric output is not promoted to physical scale.",
        )

        frames: list[DenseFusionFrame] = []
        alignment_records: list[dict[str, Any]] = []
        rejected: list[str] = []
        for index, item in enumerate(images):
            camera = camera_by_keyframe[item.keyframe_id]
            model = model_by_id.get(camera.camera_id)
            if model is None or model.camera_id != camera_model.camera_id:
                raise ReconstructionValidationError("I4.3D1 camera-model identity drifted during temporal alignment.")
            depth = np.ascontiguousarray(depth_stack[index], dtype=np.float32)
            evidence = align_sfm_keyframe_dense_depth(
                context.sfm_result,
                keyframe_id=item.keyframe_id,
                dense_depth_source_units=depth,
                depth_provenance=(
                    f"Video Depth Anything source_revision={self.environment.source_revision};"
                    "I4.3D1;temporal_stream=true;authoritative_pose=pycolmap_sfm;"
                    "model_metric_depth_not_promoted=true"
                ),
                minimum_correspondences=context.config.minimum_correspondences,
                maximum_relative_residual=context.config.maximum_relative_residual,
            )
            record = evidence.as_dict()
            record["source_sequence_index"] = item.source_sequence_index
            record["temporal_order_index"] = index
            record["estimated_model_intrinsics_consumed_for_canonical_projection"] = False
            record["model_metric_depth_promoted_to_canonical_metric_scale"] = False
            alignment_records.append(record)
            if not evidence.is_accepted or evidence.alignment is None:
                rejected.append(item.keyframe_id)
            else:
                bgr = cv2.imread(str(paths[index]), cv2.IMREAD_COLOR)
                if bgr is None:
                    raise ReconstructionValidationError(f"I4.3D1 could not decode staged RGB: {paths[index]}")
                rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)
                if rgb.shape != (expected_hw[0], expected_hw[1], 3):
                    raise ReconstructionValidationError("I4.3D1 decoded RGB does not match authoritative SfM grid.")
                frames.append(
                    DenseFusionFrame(
                        frame_id=camera.frame_id,
                        timestamp_ns=item.timestamp_ns,
                        rgb=rgb,
                        depth=depth,
                        T_reconstruction_from_camera=camera.T_reconstruction_from_camera,
                        depth_unit=DenseDepthUnit.SOURCE_MODEL_UNITS,
                        depth_alignment=evidence.alignment,
                        confidence=None,
                        valid_mask=np.isfinite(depth) & (depth > 0.0),
                        pose_layer="offline_sfm_bundle_adjusted",
                    )
                )
            self._publish(
                stage=DenseProviderProgressStage.ALIGNMENT,
                detail="align_temporal_depth_to_authoritative_sfm",
                progress=0.40 + 0.35 * ((index + 1) / float(total_views)),
                processed=index + 1,
                total=total_views,
            )

        if len(frames) < context.config.minimum_accepted_views:
            raise ReconstructionValidationError(
                "I4.3D1 insufficient VDA/SfM alignment-accepted views: "
                f"accepted={len(frames)} required={context.config.minimum_accepted_views}."
            )

        self._publish(stage=DenseProviderProgressStage.FUSION_FILTERING, detail="qualified_temporal_depth_voxel_fusion", progress=0.80, processed=len(frames), total=total_views)
        fusion = fuse_dense_frames(
            frames,
            session_id=f"{context.sfm_result.session_id}:i4.3d1-vda",
            camera_profile=camera_profile,
            scale_profile=scale_profile,
            policy=DenseFusionPolicy(
                scene_profile=DenseSceneProfile.SPATIAL_DIGITAL_TWIN,
                voxel_size_reconstruction_units=voxel_size,
                pixel_stride=context.config.pixel_stride,
                minimum_view_observations=1,
                max_voxels=context.config.max_voxels,
            ),
            reconstruction_frame_id=context.reconstruction_frame_id,
        )
        cloud = fusion.point_cloud
        canonical_npz = context.output_dir / "canonical_vda_fused.npz"
        np.savez_compressed(
            canonical_npz,
            points_reconstruction=cloud.points_reconstruction,
            colors_rgb=cloud.colors_rgb,
            view_observation_counts=cloud.view_observation_counts,
            mean_confidence=cloud.mean_confidence,
            voxel_size_reconstruction_units=np.asarray(cloud.voxel_size_reconstruction_units, dtype=np.float64),
            source_frame_ids=np.asarray(fusion.source_frame_ids, dtype="U160"),
            source_pose_layers=np.asarray(fusion.source_pose_layers, dtype="U160"),
        )
        canonical_ply = _write_binary_ply(context.output_dir / "canonical_vda_fused.ply", cloud.points_reconstruction, cloud.colors_rgb)

        authority_after = _sfm_authority_digest(context.sfm_result)
        images_after = {item.keyframe_id: _sha256_file(path) for item, path in zip(images, paths, strict=True)}
        if authority_after != authority_before:
            raise ReconstructionValidationError("I4.3D1 authoritative SfM cameras/landmarks changed during VDA execution.")
        if images_after != images_before:
            raise ReconstructionValidationError("I4.3D1 registered staged images changed during VDA execution.")
        if any(layer != "offline_sfm_bundle_adjusted" for layer in fusion.source_pose_layers):
            raise ReconstructionValidationError("I4.3D1 canonical fusion consumed a non-SfM pose layer.")

        degradation_reasons: list[str] = []
        if rejected:
            degradation_reasons.append(f"alignment_rejected_views:{len(rejected)}")
        if fusion.statistics.dropped_new_voxels_capacity:
            degradation_reasons.append(f"fusion_capacity_drops:{fusion.statistics.dropped_new_voxels_capacity}")
        status = DenseProviderRunStatus.DEGRADED if degradation_reasons else DenseProviderRunStatus.COMPLETED
        self._publish(stage=DenseProviderProgressStage.FINALIZATION, detail="canonical_temporal_geometry_ready", progress=1.0, processed=total_views, total=total_views)

        result = DenseProviderExecutionResult(
            provider_id=self.descriptor.provider_id,
            implementation=DenseProviderImplementation.VIDEO_DEPTH_ANYTHING,
            status=status,
            source_view_count=total_views,
            selected_view_count=total_views,
            adaptive_view_selection=False,
            view_selection_reason="all_registered_authoritative_sfm_views_in_temporal_order",
            permanent_global_view_cap=None,
            canonical_geometry_ref=str(canonical_npz),
            canonical_frame_id=context.reconstruction_frame_id,
            native_artifact_refs=(str(worker.manifest_path), str(worker.depth_npz_path), str(canonical_ply)),
            degradation_reasons=tuple(degradation_reasons),
            progress_event_count=self._progress_event_count,
            final_preview_revision=0,
            canonical_geometry_authoritative=True,
            single_authoritative_sfm_frame_preserved=True,
            camera_pose_authority_owner=I43A_AUTHORITATIVE_FRAME_OWNER,
            metric_scale_claimed=False,
            measurement_authoritative=False,
            certified_metrology=False,
            surface_mesh_generated=False,
            production_filter_policy_changed=False,
            provider_promotion_performed=False,
        )

        summary = {
            "schema_version": I43D1_SCHEMA_VERSION,
            "phase": I43D1_PHASE,
            "status": "degraded" if status is DenseProviderRunStatus.DEGRADED else "pass",
            "pass": True,
            "provider": self.descriptor.as_dict(),
            "selection_request": request.as_dict(),
            "selection_decision": decision.as_dict(),
            "execution_result": result.as_dict(),
            "config": context.config.as_dict(),
            "runtime_config": metadata_as_dict(context.runtime_config),  # type: ignore[arg-type]
            "runtime": self.environment.as_dict(),
            "worker": dict(worker.metadata),
            "temporal_sequence": {
                "ordered_keyframe_ids": [item.keyframe_id for item in images],
                "ordered_source_sequence_indices": [item.source_sequence_index for item in images],
                "ordered_timestamps_ns": [item.timestamp_ns for item in images],
                "stateful_sequence_required": True,
                "sequence_reordered_by": "source_sequence_index_then_timestamp",
            },
            "alignment": {
                "accepted_views": len(frames),
                "rejected_views": len(rejected),
                "rejected_keyframe_ids": rejected,
                "records": alignment_records,
            },
            "fusion": {
                "robust_sparse_scene_diagonal_reconstruction_units": scene_scale,
                "voxel_size_reconstruction_units": voxel_size,
                "statistics": fusion.statistics.as_dict(),
                "point_cloud": cloud.as_dict(),
                "canonical_npz": str(canonical_npz),
                "canonical_ply": str(canonical_ply),
            },
            "integrity": {
                "authoritative_sfm_unchanged": True,
                "registered_images_unchanged": True,
                "authoritative_sfm_digest": authority_before,
                "registered_image_sha256": images_before,
                "canonical_pose_layer": "offline_sfm_bundle_adjusted",
            },
            "authority": {
                "camera_pose_authority_owner": I43A_AUTHORITATIVE_FRAME_OWNER,
                "estimated_model_intrinsics_consumed_for_canonical_projection": False,
                "model_metric_depth_promoted_to_canonical_metric_scale": False,
                "metric_scale_claimed": False,
                "measurement_authoritative": False,
                "certified_metrology": False,
                "surface_mesh_generated": False,
                "provider_promotion_performed": False,
            },
        }
        _write_json(context.output_dir / "i4_3d1_vda_execution_summary.json", summary)
        return result


def validate_i4_3d1_binding(*, i4_3a_snapshot: Mapping[str, Any], environment: VideoDepthAnythingEnvironmentProbe) -> dict[str, Any]:
    if not isinstance(i4_3a_snapshot, Mapping):
        raise TypeError("i4_3a_snapshot must be a mapping.")
    if i4_3a_snapshot.get("phase") != "I4.3A" or i4_3a_snapshot.get("status") != "pass" or i4_3a_snapshot.get("pass") is not True:
        raise ReconstructionValidationError("I4.3D1 requires the passing I4.3A architecture snapshot.")
    architecture = i4_3a_snapshot.get("architecture")
    if not isinstance(architecture, Mapping):
        raise ReconstructionValidationError("I4.3D1 I4.3A snapshot is missing architecture.")
    if architecture.get("authoritative_camera_frame_owner") != I43A_AUTHORITATIVE_FRAME_OWNER:
        raise ReconstructionValidationError("I4.3D1 refuses changed camera-frame authority.")
    if architecture.get("permanent_global_view_cap") is not None:
        raise ReconstructionValidationError("I4.3D1 refuses a permanent global dense-view cap.")
    contracts = architecture.get("provider_implementation_contracts")
    if not isinstance(contracts, list):
        raise ReconstructionValidationError("I4.3D1 I4.3A implementation contracts are missing.")
    match = next((item for item in contracts if isinstance(item, Mapping) and item.get("implementation") == "video_depth_anything"), None)
    if not isinstance(match, Mapping):
        raise ReconstructionValidationError("I4.3D1 I4.3A Video Depth Anything implementation contract is missing.")
    if match.get("integration_stage") != "I4.3D" or match.get("family") != "temporal_depth":
        raise ReconstructionValidationError("I4.3D1 VDA implementation is assigned to another stage/family.")
    if match.get("frame_binding") != "align_to_authoritative_sfm" or match.get("native_output") != "depth_evidence":
        raise ReconstructionValidationError("I4.3D1 VDA frame/native-output contract drifted.")
    descriptor = video_depth_anything_dense_provider_descriptor(environment)
    if not descriptor.available:
        raise ReconstructionValidationError(f"I4.3D1 VDA runtime is not ready: {descriptor.unavailable_reason}")
    return {
        "schema_version": I43D1_SCHEMA_VERSION,
        "phase": I43D1_PHASE,
        "status": "pass",
        "pass": True,
        "provider": descriptor.as_dict(),
        "architecture": {
            "camera_pose_authority_owner": I43A_AUTHORITATIVE_FRAME_OWNER,
            "frame_binding": "align_to_authoritative_sfm",
            "native_output": "depth_evidence",
            "temporal_mode": "official_streaming_cache",
            "stateful_sequence": True,
            "all_registered_views_default_i4_3d1": True,
            "permanent_global_view_cap": None,
            "adaptive_view_selection_deferred_to": "I4.3F",
            "progressive_status_supported": True,
        },
        "authority": {
            "backend_started": False,
            "provider_selected": False,
            "provider_promotion_performed": False,
            "estimated_model_intrinsics_consumed_for_canonical_projection": False,
            "model_metric_depth_promoted_to_canonical_metric_scale": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
        },
        "qualification": {
            "qualification_state": "candidate",
            "rolling_depth_binding_pending_i4_3d2": True,
            "temporal_cross_provider_comparison_pending_i4_3g_i5": True,
            "universal_provider_winner_assumed": False,
        },
        "next": "I4.3D2 RollingDepth isolated runtime + canonical temporal-depth provider binding.",
    }


__all__ = [
    "I43D1_SCHEMA_VERSION",
    "I43D1_PHASE",
    "I43D1_ADAPTER_VERSION",
    "I43D1_PROVIDER_ID",
    "VideoDepthAnythingEnvironmentProbe",
    "VideoDepthAnythingProductionConfig",
    "VideoDepthAnythingProviderRunContext",
    "VideoDepthAnythingWorkerResult",
    "VideoDepthAnythingWorkerRunner",
    "ExternalVideoDepthAnythingWorkerRunner",
    "VideoDepthAnythingProductionProviderAdapter",
    "probe_video_depth_anything_environment",
    "video_depth_anything_dense_provider_descriptor",
    "validate_i4_3d1_binding",
]

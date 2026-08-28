"""I4.3D2 canonical production adapter for RollingDepth.

RollingDepth is bound as a temporal-depth provider behind the I4.3A dense
provider contract.  The official RollingDepth runtime stays isolated in its own
Python environment and source checkout.  It consumes the complete ordered
sequence, performs its native multi-dilation snippet inference/co-alignment,
and returns one temporally co-aligned relative-depth map per registered view.

The provider does *not* fit each frame independently.  One robust affine mapping
is estimated for the whole RollingDepth sequence from sparse observations in the
authoritative PyCOLMAP SfM reconstruction::

    z_sfm ~= sequence_scale * depth_rolling + sequence_shift

That single mapping is then applied to every frame before canonical projection
with bundle-adjusted PyCOLMAP poses.  The mapping only places RollingDepth depth
in the arbitrary reconstruction gauge; it never establishes physical metric
scale, camera-pose authority, measurement authority, certified metrology, or a
surface mesh.

I4.3D2 uses every registered SfM view and introduces no permanent view cap.
Adaptive view selection belongs to I4.3F.  RollingDepth vs Video Depth Anything
quality/robustness comparisons remain I4.3G/I5 work.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from math import isfinite
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping, Protocol, Sequence

import cv2
import numpy as np

from core.contracts.validation import JsonValue, metadata_as_dict, normalize_metadata, require_nonempty
from core.reconstruction.camera_geometry import camera_geometry_from_estimated_camera
from core.reconstruction.dense_fusion import (
    DenseDepthUnit,
    DenseFusionFrame,
    DenseFusionPolicy,
    DenseSceneProfile,
    fuse_dense_frames,
    reconstruction_unit_depth_alignment,
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
from core.reconstruction.sfm_dense_projection import reconstruction_camera_from_sfm_model

I43D2_SCHEMA_VERSION = 1
I43D2_PHASE = "I4.3D2"
I43D2_ADAPTER_VERSION = "i4.3d2-v1"
I43D2_PROVIDER_ID = "rolling-depth-temporal"
I43D2_RUNTIME_LOCK_NAME = ".eia_rolling_depth_runtime_lock.json"
I43D2_DEFAULT_CHECKPOINT_RELATIVE = "checkpoint/rollingdepth-v1-0"
I43D2_DEFAULT_VOXEL_SCENE_FRACTION = 0.004
I43D2_DEFAULT_MINIMUM_SEQUENCE_CORRESPONDENCES = 24
I43D2_DEFAULT_MINIMUM_CONTRIBUTING_VIEWS = 2
I43D2_DEFAULT_MAXIMUM_RELATIVE_RESIDUAL = 0.35
I43D2_DEFAULT_PIXEL_STRIDE = 2
I43D2_DEFAULT_MAX_VOXELS = 1_500_000
I43D2_DEFAULT_MINIMUM_ACCEPTED_VIEWS = 2
I43D2_DEFAULT_PROCESSING_RESOLUTION = 512
I43D2_DEFAULT_DILATIONS = (1, 25)
I43D2_DEFAULT_SNIPPET_LENGTHS = (3,)
I43D2_DEFAULT_REFINE_STEP = 0
I43D2_DEFAULT_MAX_VAE_BATCH_SIZE = 1
I43D2_DEFAULT_SEED = 42


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
        raise ReconstructionValidationError(f"I4.3D2 could not read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"I4.3D2 JSON must contain an object: {path}")
    return payload


def _require_clean_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise ReconstructionValidationError(
            f"I4.3D2 output directory must be empty or absent; refusing to overwrite provider evidence: {path}"
        )


def _write_binary_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> Path:
    xyz = np.asarray(points, dtype=np.float32)
    rgb = np.asarray(colors, dtype=np.uint8)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or rgb.shape != (len(xyz), 3):
        raise ReconstructionValidationError("I4.3D2 PLY geometry must be aligned Nx3 XYZ/RGB arrays.")
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
        raise ReconstructionValidationError("I4.3D2 requires sparse SfM landmarks for scene-relative fusion scale.")
    points = np.vstack([np.asarray(item.position_reconstruction, dtype=np.float64) for item in sfm_result.landmarks])
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ReconstructionValidationError("I4.3D2 sparse SfM landmarks are invalid.")
    low = np.quantile(points, 0.02, axis=0)
    high = np.quantile(points, 0.98, axis=0)
    diagonal = float(np.linalg.norm(high - low))
    if not isfinite(diagonal) or diagonal <= 0.0:
        raise ReconstructionValidationError("I4.3D2 robust sparse scene diagonal is invalid.")
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
    registered = {item.keyframe_id for item in sfm_result.cameras}
    images = tuple(item for item in sfm_result.workspace.images if item.keyframe_id in registered)
    if len(images) != len(sfm_result.cameras) or len(images) < 2:
        raise ReconstructionValidationError("I4.3D2 requires every registered SfM camera to have a staged temporal image.")
    ordered = tuple(sorted(images, key=lambda item: (item.source_sequence_index, item.timestamp_ns, item.order)))
    indices = [item.source_sequence_index for item in ordered]
    if len(set(indices)) != len(indices):
        raise ReconstructionValidationError("I4.3D2 temporal provider requires unique source_sequence_index values.")
    timestamps = [item.timestamp_ns for item in ordered]
    if any(b <= a for a, b in zip(timestamps, timestamps[1:])):
        raise ReconstructionValidationError("I4.3D2 temporal provider requires strictly increasing canonical timestamps.")
    root = Path(sfm_result.workspace.workspace_root).expanduser().resolve()
    paths = tuple((root / item.staged_image_path).resolve() for item in ordered)
    for item, path in zip(ordered, paths, strict=True):
        if not path.is_file():
            raise ReconstructionValidationError(f"I4.3D2 staged registered image is missing: {path}")
        if _sha256_file(path) != item.staged_sha256:
            raise ReconstructionValidationError(f"I4.3D2 staged image hash changed before provider execution: {item.keyframe_id}")
    return ordered, paths


def _bilinear_finite_value(image: np.ndarray, *, x: float, y: float) -> float | None:
    height, width = image.shape
    if not (0.0 <= x <= width - 1 and 0.0 <= y <= height - 1):
        return None
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, width - 1)
    y1 = min(y0 + 1, height - 1)
    values = np.asarray([image[y0, x0], image[y0, x1], image[y1, x0], image[y1, x1]], dtype=np.float64)
    if not np.isfinite(values).all():
        return None
    wx = float(x - x0)
    wy = float(y - y0)
    top = (1.0 - wx) * values[0] + wx * values[1]
    bottom = (1.0 - wx) * values[2] + wx * values[3]
    return float((1.0 - wy) * top + wy * bottom)


@dataclass(frozen=True, slots=True)
class RollingDepthSequenceAlignmentEvidence:
    accepted: bool
    scale_to_reconstruction: float | None
    shift_to_reconstruction: float | None
    initial_correspondences: int
    retained_correspondences: int
    contributing_views: int
    median_relative_residual: float | None
    p90_relative_residual: float | None
    robust_inlier_threshold: float | None
    view_correspondence_counts: tuple[tuple[str, int], ...]
    message: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "scale_to_reconstruction": self.scale_to_reconstruction,
            "shift_to_reconstruction": self.shift_to_reconstruction,
            "initial_correspondences": self.initial_correspondences,
            "retained_correspondences": self.retained_correspondences,
            "contributing_views": self.contributing_views,
            "median_relative_residual": self.median_relative_residual,
            "p90_relative_residual": self.p90_relative_residual,
            "robust_inlier_threshold": self.robust_inlier_threshold,
            "view_correspondence_counts": dict(self.view_correspondence_counts),
            "message": self.message,
            "alignment_scope": "one_global_affine_for_entire_temporal_sequence",
            "physical_metric_scale_claimed": False,
        }


def fit_rolling_depth_sequence_to_sfm(
    sfm_result: OfflineSfmResult,
    *,
    ordered_images: Sequence[SfmWorkspaceImage],
    depth_stack: np.ndarray,
    minimum_correspondences: int = I43D2_DEFAULT_MINIMUM_SEQUENCE_CORRESPONDENCES,
    minimum_contributing_views: int = I43D2_DEFAULT_MINIMUM_CONTRIBUTING_VIEWS,
    maximum_relative_residual: float = I43D2_DEFAULT_MAXIMUM_RELATIVE_RESIDUAL,
) -> RollingDepthSequenceAlignmentEvidence:
    """Fit one robust affine mapping for the complete co-aligned sequence.

    The source is RollingDepth's already globally co-aligned relative depth.  A
    single scale+shift is deliberately shared by every frame so sparse SfM
    anchoring cannot reintroduce frame-wise scale flicker.
    """

    depth = np.asarray(depth_stack, dtype=np.float64)
    if depth.ndim != 3 or depth.shape[0] != len(ordered_images):
        raise ReconstructionValidationError("I4.3D2 sequence alignment requires NxHxW depth matching ordered images.")
    if minimum_correspondences < 6 or minimum_contributing_views < 2:
        raise ReconstructionValidationError("I4.3D2 sequence alignment thresholds are too small.")
    if not 0.0 < float(maximum_relative_residual) < 1.0:
        raise ReconstructionValidationError("I4.3D2 maximum_relative_residual must be in (0,1).")

    image_index = {item.keyframe_id: index for index, item in enumerate(ordered_images)}
    camera_by_keyframe = {item.keyframe_id: item for item in sfm_result.cameras}
    source_values: list[float] = []
    target_values: list[float] = []
    keyframes: list[str] = []
    counts: dict[str, int] = {item.keyframe_id: 0 for item in ordered_images}

    for landmark in sfm_result.landmarks:
        point_r = np.append(np.asarray(landmark.position_reconstruction, dtype=np.float64), 1.0)
        for observation in landmark.observations:
            stack_index = image_index.get(observation.keyframe_id)
            camera = camera_by_keyframe.get(observation.keyframe_id)
            if stack_index is None or camera is None:
                continue
            T_camera_from_reconstruction = np.linalg.inv(np.asarray(camera.T_reconstruction_from_camera, dtype=np.float64))
            point_camera = T_camera_from_reconstruction @ point_r
            camera_z = float(point_camera[2])
            if not isfinite(camera_z) or camera_z <= 1e-8:
                continue
            sampled = _bilinear_finite_value(
                depth[stack_index], x=float(observation.feature_xy[0]), y=float(observation.feature_xy[1])
            )
            if sampled is None:
                continue
            source_values.append(sampled)
            target_values.append(camera_z)
            keyframes.append(observation.keyframe_id)
            counts[observation.keyframe_id] += 1

    initial_count = len(source_values)
    contributing = sum(value > 0 for value in counts.values())
    if initial_count < minimum_correspondences or contributing < minimum_contributing_views:
        return RollingDepthSequenceAlignmentEvidence(
            False, None, None, initial_count, 0, contributing, None, None, None,
            tuple(sorted(counts.items())),
            f"insufficient_sequence_correspondences:{initial_count};contributing_views:{contributing}",
        )

    src = np.asarray(source_values, dtype=np.float64)
    target = np.asarray(target_values, dtype=np.float64)
    if float(np.ptp(src)) <= 1e-9:
        return RollingDepthSequenceAlignmentEvidence(
            False, None, None, initial_count, 0, contributing, None, None, None,
            tuple(sorted(counts.items())), "rolling_depth_sparse_samples_have_no_dynamic_range",
        )

    design = np.column_stack([src, np.ones_like(src)])
    fit, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    if rank < 2:
        return RollingDepthSequenceAlignmentEvidence(
            False, None, None, initial_count, 0, contributing, None, None, None,
            tuple(sorted(counts.items())), "sequence_affine_fit_rank_deficient",
        )
    scale, shift = float(fit[0]), float(fit[1])
    if not isfinite(scale) or not isfinite(shift) or scale <= 0.0:
        return RollingDepthSequenceAlignmentEvidence(
            False, scale if isfinite(scale) else None, shift if isfinite(shift) else None,
            initial_count, 0, contributing, None, None, None,
            tuple(sorted(counts.items())), "sequence_affine_fit_requires_positive_depth_scale",
        )

    predicted = scale * src + shift
    relative = np.abs(predicted - target) / np.maximum(np.abs(target), 1e-8)
    median = float(np.median(relative))
    mad = float(np.median(np.abs(relative - median)))
    robust_limit = min(float(maximum_relative_residual), max(0.03, median + 3.5 * 1.4826 * mad))
    inliers = np.isfinite(relative) & (relative <= robust_limit)
    retained = int(np.count_nonzero(inliers))
    retained_views = len({keyframes[index] for index in np.flatnonzero(inliers)})
    if retained < minimum_correspondences or retained_views < minimum_contributing_views:
        return RollingDepthSequenceAlignmentEvidence(
            False, scale, shift, initial_count, retained, retained_views, median,
            float(np.quantile(relative[np.isfinite(relative)], 0.90)), robust_limit,
            tuple(sorted(counts.items())), "robust_sequence_affine_inliers_insufficient",
        )

    fit2, _, rank2, _ = np.linalg.lstsq(design[inliers], target[inliers], rcond=None)
    if rank2 < 2:
        return RollingDepthSequenceAlignmentEvidence(
            False, scale, shift, initial_count, retained, retained_views, median, None, robust_limit,
            tuple(sorted(counts.items())), "robust_sequence_affine_refit_rank_deficient",
        )
    scale2, shift2 = float(fit2[0]), float(fit2[1])
    if not isfinite(scale2) or not isfinite(shift2) or scale2 <= 0.0:
        return RollingDepthSequenceAlignmentEvidence(
            False, scale2 if isfinite(scale2) else None, shift2 if isfinite(shift2) else None,
            initial_count, retained, retained_views, median, None, robust_limit,
            tuple(sorted(counts.items())), "robust_sequence_affine_refit_requires_positive_depth_scale",
        )
    final_pred = scale2 * src[inliers] + shift2
    final_relative = np.abs(final_pred - target[inliers]) / np.maximum(np.abs(target[inliers]), 1e-8)
    final_median = float(np.median(final_relative))
    final_p90 = float(np.quantile(final_relative, 0.90))
    accepted = bool(final_p90 <= float(maximum_relative_residual))
    message = "accepted_global_sequence_affine_to_authoritative_sfm" if accepted else "sequence_affine_residual_exceeds_limit"
    return RollingDepthSequenceAlignmentEvidence(
        accepted, scale2, shift2, initial_count, retained, retained_views,
        final_median, final_p90, robust_limit, tuple(sorted(counts.items())), message,
    )


@dataclass(frozen=True, slots=True)
class PreparedRollingDepthVideo:
    video_path: Path
    frame_count: int
    width: int
    height: int
    fps: float
    sha256: str
    codec: str = "png"

    def as_dict(self) -> dict[str, Any]:
        return {
            "video_path": str(self.video_path),
            "frame_count": self.frame_count,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "sha256": self.sha256,
            "codec": self.codec,
            "lossless": True,
        }


class RollingDepthSequenceBuilder(Protocol):
    def build(
        self,
        *,
        images: Sequence[SfmWorkspaceImage],
        paths: Sequence[Path],
        output_dir: Path,
    ) -> PreparedRollingDepthVideo: ...


class FfmpegLosslessRollingDepthSequenceBuilder:
    def __init__(self, *, ffmpeg_executable: str = "ffmpeg") -> None:
        self.ffmpeg_executable = ffmpeg_executable

    def build(
        self,
        *,
        images: Sequence[SfmWorkspaceImage],
        paths: Sequence[Path],
        output_dir: Path,
    ) -> PreparedRollingDepthVideo:
        if len(images) != len(paths) or len(images) < 2:
            raise ReconstructionValidationError("I4.3D2 lossless temporal video requires matching image/path sequences.")
        ffmpeg = shutil.which(self.ffmpeg_executable)
        if not ffmpeg:
            raise ReconstructionValidationError("I4.3D2 requires ffmpeg to prepare the lossless RollingDepth input video.")
        output_dir.mkdir(parents=True, exist_ok=False)
        frames_dir = output_dir / "frames"
        frames_dir.mkdir()
        expected_hw = (images[0].image_size.height, images[0].image_size.width)
        for index, path in enumerate(paths):
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None or bgr.shape[:2] != expected_hw:
                raise ReconstructionValidationError(f"I4.3D2 could not decode stable-grid staged RGB: {path}")
            out = frames_dir / f"{index:06d}.png"
            if not cv2.imwrite(str(out), bgr):
                raise ReconstructionValidationError(f"I4.3D2 could not prepare lossless frame: {out}")
        deltas = np.diff(np.asarray([item.timestamp_ns for item in images], dtype=np.float64))
        median_delta = float(np.median(deltas)) if len(deltas) else (1e9 / 30.0)
        fps = 1e9 / median_delta if median_delta > 0.0 else 30.0
        fps = max(1.0, min(120.0, fps))
        video_path = output_dir / "rolling_depth_sequence.mkv"
        completed = subprocess.run(
            [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                "-framerate", f"{fps:.9f}",
                "-i", str(frames_dir / "%06d.png"),
                "-an", "-c:v", "png", "-pix_fmt", "rgb24", str(video_path),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0 or not video_path.is_file():
            raise ReconstructionValidationError(
                "I4.3D2 ffmpeg lossless input-video preparation failed: "
                f"{completed.stderr.strip() or completed.returncode}"
            )
        return PreparedRollingDepthVideo(
            video_path=video_path.resolve(),
            frame_count=len(images),
            width=expected_hw[1],
            height=expected_hw[0],
            fps=fps,
            sha256=_sha256_file(video_path),
        )


@dataclass(frozen=True, slots=True)
class RollingDepthEnvironmentProbe:
    python_executable: str
    repository_root: str
    runtime_lock_path: str
    checkpoint_path: str
    source_revision: str
    python_version: str
    torch_version: str
    diffusers_version: str
    torch_cuda_version: str
    source_ready: bool
    checkpoint_ready: bool
    runtime_lock_valid: bool
    upstream_importable: bool
    modified_diffusers_ready: bool
    torch_available: bool
    ffmpeg_available: bool
    cuda_available: bool
    device_names: tuple[str, ...] = ()
    error: str = ""
    schema_version: int = I43D2_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if int(self.schema_version) != I43D2_SCHEMA_VERSION:
            raise ReconstructionValidationError("Unsupported I4.3D2 environment-probe schema version.")
        revision = str(self.source_revision).strip().lower()
        if revision and (len(revision) != 40 or any(ch not in "0123456789abcdef" for ch in revision)):
            raise ReconstructionValidationError("I4.3D2 RollingDepth source revision must be a 40-character git digest when present.")
        object.__setattr__(self, "source_revision", revision)
        object.__setattr__(self, "device_names", tuple(str(v).strip() for v in self.device_names if str(v).strip()))
        object.__setattr__(self, "error", str(self.error).strip())
        object.__setattr__(self, "schema_version", I43D2_SCHEMA_VERSION)

    @property
    def available(self) -> bool:
        return bool(
            self.source_ready
            and bool(self.source_revision)
            and self.checkpoint_ready
            and self.runtime_lock_valid
            and self.upstream_importable
            and self.modified_diffusers_ready
            and self.torch_available
            and self.ffmpeg_available
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "python_executable": self.python_executable,
            "repository_root": self.repository_root,
            "runtime_lock_path": self.runtime_lock_path,
            "checkpoint_path": self.checkpoint_path,
            "source_revision": self.source_revision,
            "python_version": self.python_version,
            "torch_version": self.torch_version,
            "diffusers_version": self.diffusers_version,
            "torch_cuda_version": self.torch_cuda_version,
            "source_ready": self.source_ready,
            "checkpoint_ready": self.checkpoint_ready,
            "runtime_lock_valid": self.runtime_lock_valid,
            "upstream_importable": self.upstream_importable,
            "modified_diffusers_ready": self.modified_diffusers_ready,
            "torch_available": self.torch_available,
            "ffmpeg_available": self.ffmpeg_available,
            "cuda_available": self.cuda_available,
            "device_names": list(self.device_names),
            "available": self.available,
            "error": self.error,
        }


def probe_rolling_depth_environment(
    *,
    python_executable: str | Path,
    repository_root: str | Path,
    checkpoint_path: str | Path | None = None,
) -> RollingDepthEnvironmentProbe:
    python_path = Path(python_executable).expanduser().resolve()
    repo = Path(repository_root).expanduser().resolve()
    lock_path = repo / I43D2_RUNTIME_LOCK_NAME
    checkpoint = Path(checkpoint_path).expanduser() if checkpoint_path is not None else repo / I43D2_DEFAULT_CHECKPOINT_RELATIVE
    if not checkpoint.is_absolute():
        checkpoint = (repo / checkpoint).resolve()
    else:
        checkpoint = checkpoint.resolve()
    errors: list[str] = []
    source_ready = (repo / "rollingdepth" / "__init__.py").is_file() and (repo / "run_video.py").is_file()
    if not source_ready:
        errors.append("official_source_not_ready")
    checkpoint_ready = checkpoint.is_dir() and (checkpoint / "model_index.json").is_file()
    if not checkpoint_ready:
        errors.append("local_checkpoint_missing_or_incomplete")
    ffmpeg_available = shutil.which("ffmpeg") is not None
    if not ffmpeg_available:
        errors.append("ffmpeg_missing")

    source_revision = ""
    if repo.is_dir():
        try:
            completed = subprocess.run(
                ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True, capture_output=True, check=False, timeout=10
            )
            if completed.returncode == 0:
                source_revision = completed.stdout.strip().lower()
        except (OSError, subprocess.SubprocessError):
            pass
    if not source_revision:
        errors.append("source_revision_missing")

    lock_payload: dict[str, Any] = {}
    if lock_path.is_file():
        try:
            payload = json.loads(lock_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                lock_payload = payload
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"runtime_lock_invalid:{exc}")
    else:
        errors.append("runtime_lock_missing")
    lock_revision = str(lock_payload.get("source_revision", "")).strip().lower()
    runtime_lock_valid = bool(source_revision and lock_revision == source_revision)
    if not runtime_lock_valid:
        errors.append("runtime_lock_revision_mismatch")

    python_version = ""
    torch_version = ""
    diffusers_version = ""
    torch_cuda_version = ""
    upstream_importable = False
    modified_diffusers_ready = False
    torch_available = False
    cuda_available = False
    device_names: tuple[str, ...] = ()
    if python_path.is_file():
        probe_script = r'''
import json, pathlib, sys
payload = {"upstream": False, "modified_diffusers": False, "torch": False, "cuda": False, "devices": []}
repo = pathlib.Path(r"REPO_PATH").resolve()
sys.path.insert(0, str(repo))
try:
    from rollingdepth import RollingDepthPipeline
    payload["upstream"] = RollingDepthPipeline is not None
except Exception as exc:
    payload["upstream_error"] = repr(exc)
try:
    import diffusers
    payload["diffusers_version"] = str(getattr(diffusers, "__version__", ""))
    root = pathlib.Path(diffusers.__file__).resolve().parent
    marker = "Modified in RollingDepth"
    payload["modified_diffusers"] = any(marker in p.read_text(encoding="utf-8", errors="ignore") for p in root.rglob("*.py"))
except Exception as exc:
    payload["diffusers_error"] = repr(exc)
try:
    import torch
    payload["torch"] = True
    payload["torch_version"] = str(torch.__version__)
    payload["torch_cuda_version"] = str(torch.version.cuda or "")
    payload["cuda"] = bool(torch.cuda.is_available())
    if payload["cuda"]:
        payload["devices"] = [str(torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]
except Exception as exc:
    payload["torch_error"] = repr(exc)
payload["python_version"] = sys.version.split()[0]
print(json.dumps(payload))
'''.replace("REPO_PATH", str(repo))
        try:
            completed = subprocess.run(
                [str(python_path), "-c", probe_script], cwd=str(repo) if repo.is_dir() else None,
                text=True, capture_output=True, check=False, timeout=90,
            )
            if completed.returncode == 0 and completed.stdout.strip():
                payload = json.loads(completed.stdout.strip().splitlines()[-1])
                upstream_importable = bool(payload.get("upstream"))
                modified_diffusers_ready = bool(payload.get("modified_diffusers"))
                torch_available = bool(payload.get("torch"))
                cuda_available = bool(payload.get("cuda"))
                device_names = tuple(str(v) for v in payload.get("devices", ()))
                python_version = str(payload.get("python_version", ""))
                torch_version = str(payload.get("torch_version", ""))
                diffusers_version = str(payload.get("diffusers_version", ""))
                torch_cuda_version = str(payload.get("torch_cuda_version", ""))
                if not upstream_importable:
                    errors.append("upstream_import_failed")
                if not modified_diffusers_ready:
                    errors.append("modified_diffusers_marker_missing")
                if not torch_available:
                    errors.append("torch_import_failed")
            else:
                errors.append(f"runtime_probe_failed:{completed.stderr.strip() or completed.returncode}")
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
            errors.append(f"runtime_probe_failed:{exc}")
    else:
        errors.append("python_executable_missing")

    return RollingDepthEnvironmentProbe(
        python_executable=str(python_path),
        repository_root=str(repo),
        runtime_lock_path=str(lock_path.resolve()),
        checkpoint_path=str(checkpoint),
        source_revision=source_revision,
        python_version=python_version,
        torch_version=torch_version,
        diffusers_version=diffusers_version,
        torch_cuda_version=torch_cuda_version,
        source_ready=source_ready,
        checkpoint_ready=checkpoint_ready,
        runtime_lock_valid=runtime_lock_valid,
        upstream_importable=upstream_importable,
        modified_diffusers_ready=modified_diffusers_ready,
        torch_available=torch_available,
        ffmpeg_available=ffmpeg_available,
        cuda_available=cuda_available,
        device_names=device_names,
        error=";".join(dict.fromkeys(errors)),
    )


def rolling_depth_dense_provider_descriptor(probe: RollingDepthEnvironmentProbe) -> DenseProviderDescriptor:
    if not isinstance(probe, RollingDepthEnvironmentProbe):
        raise TypeError("probe must be RollingDepthEnvironmentProbe.")
    reason = "" if probe.available else (probe.error or "rolling_depth_runtime_unavailable")
    return DenseProviderDescriptor(
        provider_id=I43D2_PROVIDER_ID,
        implementation=DenseProviderImplementation.ROLLING_DEPTH,
        execution=ReconstructionProviderExecution.EXTERNAL_PROCESS,
        backend_version=probe.source_revision[:12] if probe.source_revision else "unknown",
        adapter_version=I43D2_ADAPTER_VERSION,
        qualification_state=DenseProviderQualificationState.CANDIDATE,
        available=probe.available,
        unavailable_reason=reason,
        supports_engineer_override=True,
        metadata={
            "phase": I43D2_PHASE,
            "native_output": "depth_evidence",
            "frame_binding": "align_to_authoritative_sfm",
            "temporal_mode": "multi_dilation_snippet_coalignment",
            "stateful_sequence": True,
            "sequence_order_source": "sfm_workspace.source_sequence_index",
            "sequence_level_alignment": "global_affine_to_authoritative_sfm",
            "per_frame_alignment": False,
            "relative_depth_affine_not_metric": True,
            "canonical_projection_pose_source": I43A_AUTHORITATIVE_FRAME_OWNER,
            "all_registered_views_used_in_i4_3d2": True,
            "adaptive_view_selection_deferred_to": "I4.3F",
            "permanent_global_view_cap": None,
            "temporal_quality_comparison_pending": "I4.3G/I5",
            "provider_promotion_performed": False,
        },
    )


@dataclass(frozen=True, slots=True)
class RollingDepthProductionConfig:
    minimum_sequence_correspondences: int = I43D2_DEFAULT_MINIMUM_SEQUENCE_CORRESPONDENCES
    minimum_contributing_views: int = I43D2_DEFAULT_MINIMUM_CONTRIBUTING_VIEWS
    maximum_relative_residual: float = I43D2_DEFAULT_MAXIMUM_RELATIVE_RESIDUAL
    voxel_scene_fraction: float = I43D2_DEFAULT_VOXEL_SCENE_FRACTION
    pixel_stride: int = I43D2_DEFAULT_PIXEL_STRIDE
    max_voxels: int = I43D2_DEFAULT_MAX_VOXELS
    minimum_accepted_views: int = I43D2_DEFAULT_MINIMUM_ACCEPTED_VIEWS
    processing_resolution: int = I43D2_DEFAULT_PROCESSING_RESOLUTION
    dilations: tuple[int, ...] = I43D2_DEFAULT_DILATIONS
    snippet_lengths: tuple[int, ...] = I43D2_DEFAULT_SNIPPET_LENGTHS
    refine_step: int = I43D2_DEFAULT_REFINE_STEP
    max_vae_batch_size: int = I43D2_DEFAULT_MAX_VAE_BATCH_SIZE
    unload_snippet: bool = True
    restore_resolution: bool = True
    seed: int = I43D2_DEFAULT_SEED

    def __post_init__(self) -> None:
        if int(self.minimum_sequence_correspondences) < 6 or int(self.minimum_contributing_views) < 2:
            raise ReconstructionValidationError("I4.3D2 sequence correspondence/view thresholds are too small.")
        residual = float(self.maximum_relative_residual)
        if not isfinite(residual) or not 0.0 < residual < 1.0:
            raise ReconstructionValidationError("I4.3D2 maximum_relative_residual must be in (0,1).")
        fraction = float(self.voxel_scene_fraction)
        if not isfinite(fraction) or fraction <= 0.0:
            raise ReconstructionValidationError("I4.3D2 voxel_scene_fraction must be positive.")
        positive_ints = (
            self.pixel_stride, self.max_voxels, self.minimum_accepted_views,
            self.processing_resolution, self.max_vae_batch_size,
        )
        if any(int(v) <= 0 for v in positive_ints):
            raise ReconstructionValidationError("I4.3D2 positive integer configuration contains a non-positive value.")
        dilations = tuple(int(v) for v in self.dilations)
        snippets = tuple(int(v) for v in self.snippet_lengths)
        if not dilations or any(v <= 0 for v in dilations):
            raise ReconstructionValidationError("I4.3D2 dilations must be positive.")
        if not snippets or any(v not in {2, 3, 4} for v in snippets):
            raise ReconstructionValidationError("I4.3D2 snippet lengths must contain 2, 3, or 4.")
        if int(self.refine_step) < 0 or int(self.seed) < 0:
            raise ReconstructionValidationError("I4.3D2 refine_step/seed must be non-negative.")
        object.__setattr__(self, "dilations", dilations)
        object.__setattr__(self, "snippet_lengths", snippets)

    def as_dict(self) -> dict[str, Any]:
        return {
            "minimum_sequence_correspondences": int(self.minimum_sequence_correspondences),
            "minimum_contributing_views": int(self.minimum_contributing_views),
            "maximum_relative_residual": float(self.maximum_relative_residual),
            "voxel_scene_fraction": float(self.voxel_scene_fraction),
            "pixel_stride": int(self.pixel_stride),
            "max_voxels": int(self.max_voxels),
            "minimum_accepted_views": int(self.minimum_accepted_views),
            "processing_resolution": int(self.processing_resolution),
            "dilations": list(self.dilations),
            "snippet_lengths": list(self.snippet_lengths),
            "refine_step": int(self.refine_step),
            "max_vae_batch_size": int(self.max_vae_batch_size),
            "unload_snippet": bool(self.unload_snippet),
            "restore_resolution": bool(self.restore_resolution),
            "seed": int(self.seed),
        }


@dataclass(frozen=True, slots=True)
class RollingDepthProviderRunContext:
    sfm_result: OfflineSfmResult
    reconstruction_frame_id: str
    output_dir: Path
    rolling_depth_python: Path
    rolling_depth_repository: Path
    worker_script: Path
    device: str = "cuda:0"
    precision: str = "fp16"
    config: RollingDepthProductionConfig = field(default_factory=RollingDepthProductionConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.sfm_result, OfflineSfmResult):
            raise TypeError("sfm_result must be OfflineSfmResult.")
        if not isinstance(self.config, RollingDepthProductionConfig):
            raise TypeError("config must be RollingDepthProductionConfig.")
        frame = require_nonempty(self.reconstruction_frame_id, field_name="reconstruction_frame_id")
        device = require_nonempty(self.device, field_name="device")
        precision = require_nonempty(self.precision, field_name="precision").lower()
        if precision not in {"fp16", "fp32"}:
            raise ReconstructionValidationError("I4.3D2 precision must be fp16 or fp32.")
        object.__setattr__(self, "reconstruction_frame_id", frame)
        object.__setattr__(self, "output_dir", Path(self.output_dir).expanduser().resolve())
        object.__setattr__(self, "rolling_depth_python", Path(self.rolling_depth_python).expanduser().resolve())
        object.__setattr__(self, "rolling_depth_repository", Path(self.rolling_depth_repository).expanduser().resolve())
        object.__setattr__(self, "worker_script", Path(self.worker_script).expanduser().resolve())
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "precision", precision)


@dataclass(frozen=True, slots=True)
class RollingDepthWorkerResult:
    manifest_path: Path
    depth_npz_path: Path
    depth: np.ndarray
    metadata: Mapping[str, Any]


class RollingDepthWorkerRunner(Protocol):
    def run(self, manifest_path: Path, output_dir: Path) -> RollingDepthWorkerResult: ...


class ExternalRollingDepthWorkerRunner:
    def __init__(self, *, python_executable: Path, worker_script: Path, repository_root: Path) -> None:
        self.python_executable = Path(python_executable).expanduser().resolve()
        self.worker_script = Path(worker_script).expanduser().resolve()
        self.repository_root = Path(repository_root).expanduser().resolve()

    def run(self, manifest_path: Path, output_dir: Path) -> RollingDepthWorkerResult:
        output_dir.mkdir(parents=True, exist_ok=False)
        completed = subprocess.run(
            [str(self.python_executable), str(self.worker_script), "--input-manifest", str(manifest_path), "--output-dir", str(output_dir)],
            cwd=str(self.repository_root), text=True, capture_output=True, check=False,
        )
        if completed.returncode != 0:
            raise ReconstructionValidationError(
                "I4.3D2 isolated RollingDepth worker failed: "
                f"code={completed.returncode} stderr={completed.stderr.strip() or '<empty>'}"
            )
        result_manifest = output_dir / "rolling_depth_result_manifest.json"
        if not result_manifest.is_file():
            raise ReconstructionValidationError("I4.3D2 isolated RollingDepth worker did not produce its result manifest.")
        payload = _read_json(result_manifest)
        npz_path = Path(str(payload.get("depth_npz_path", ""))).expanduser().resolve()
        if not npz_path.is_file():
            raise ReconstructionValidationError("I4.3D2 RollingDepth worker result NPZ is missing.")
        try:
            with np.load(npz_path, allow_pickle=False) as archive:
                depth = np.asarray(archive["depth"], dtype=np.float32)
        except (OSError, KeyError, ValueError) as exc:
            raise ReconstructionValidationError(f"I4.3D2 could not read RollingDepth NPZ: {exc}") from exc
        return RollingDepthWorkerResult(result_manifest, npz_path, depth, payload)


class RollingDepthProductionProviderAdapter:
    def __init__(
        self,
        *,
        context: RollingDepthProviderRunContext,
        environment: RollingDepthEnvironmentProbe,
        runner: RollingDepthWorkerRunner | None = None,
        sequence_builder: RollingDepthSequenceBuilder | None = None,
        progress_sink: DenseProviderProgressSink | None = None,
    ) -> None:
        if not isinstance(context, RollingDepthProviderRunContext):
            raise TypeError("context must be RollingDepthProviderRunContext.")
        if not isinstance(environment, RollingDepthEnvironmentProbe):
            raise TypeError("environment must be RollingDepthEnvironmentProbe.")
        self.context = context
        self.environment = environment
        self._progress_sink = progress_sink
        self._progress_event_count = 0
        self._runner = runner or ExternalRollingDepthWorkerRunner(
            python_executable=context.rolling_depth_python,
            worker_script=context.worker_script,
            repository_root=context.rolling_depth_repository,
        )
        self._sequence_builder = sequence_builder or FfmpegLosslessRollingDepthSequenceBuilder()

    @property
    def descriptor(self) -> DenseProviderDescriptor:
        return rolling_depth_dense_provider_descriptor(self.environment)

    def _publish(self, *, stage: DenseProviderProgressStage, detail: str, progress: float, processed: int, total: int) -> None:
        event = DenseProviderProgressEvent(
            provider_id=self.descriptor.provider_id,
            implementation=DenseProviderImplementation.ROLLING_DEPTH,
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
            raise ReconstructionValidationError("I4.3D2 refuses temporal canonical binding without authoritative SfM.")
        if request.authoritative_sfm_frame_id != self.context.reconstruction_frame_id:
            raise ReconstructionValidationError("I4.3D2 selection frame ID does not match provider context.")
        if request.available_view_count != views:
            raise ReconstructionValidationError("I4.3D2 available_view_count must equal registered-view count.")
        if decision.selected_provider_id != self.descriptor.provider_id:
            raise ReconstructionValidationError("I4.3D2 adapter may execute only when its concrete provider ID was selected.")
        if decision.selected_implementation is not DenseProviderImplementation.ROLLING_DEPTH:
            raise ReconstructionValidationError("I4.3D2 selection implementation must be RollingDepth.")

    def execute(self, request: DenseProviderSelectionRequest, decision: DenseProviderSelectionDecision) -> DenseProviderExecutionResult:
        context = self.context
        if not self.environment.available:
            raise ReconstructionValidationError(f"I4.3D2 RollingDepth runtime is unavailable: {self.environment.error}")
        if not context.rolling_depth_python.is_file():
            raise ReconstructionValidationError(f"I4.3D2 Python executable is missing: {context.rolling_depth_python}")
        if not context.worker_script.is_file():
            raise ReconstructionValidationError(f"I4.3D2 worker script is missing: {context.worker_script}")
        if context.rolling_depth_repository != Path(self.environment.repository_root).resolve():
            raise ReconstructionValidationError("I4.3D2 provider context/runtime repository roots differ.")
        images, paths = _ordered_registered_inputs(context.sfm_result)
        total_views = len(images)
        self._validate_selection(request, decision, views=total_views)
        _require_clean_output(context.output_dir)

        camera_by_keyframe = {item.keyframe_id: item for item in context.sfm_result.cameras}
        model_by_id = {item.camera_id: item for item in context.sfm_result.camera_models}
        camera_ids = {camera_by_keyframe[item.keyframe_id].camera_id for item in images}
        if len(camera_ids) != 1:
            raise ReconstructionValidationError(
                "I4.3D2 currently requires one shared SfM camera model; mixed-camera temporal fusion is not silently approximated."
            )
        camera_id = next(iter(camera_ids))
        camera_model = model_by_id.get(camera_id)
        if camera_model is None:
            raise ReconstructionValidationError("I4.3D2 registered SfM camera has no camera-model estimate.")
        expected_hw = (camera_model.image_size.height, camera_model.image_size.width)
        if any((item.image_size.height, item.image_size.width) != expected_hw for item in images):
            raise ReconstructionValidationError("I4.3D2 temporal sequence must remain on one stable SfM pixel grid.")

        context.output_dir.mkdir(parents=True, exist_ok=True)
        provider_dir = context.output_dir / "provider"
        provider_dir.mkdir(parents=True, exist_ok=False)
        self._progress_event_count = 0
        self._publish(stage=DenseProviderProgressStage.PREPARING, detail="validated_temporal_sfm_inputs", progress=0.02, processed=0, total=total_views)
        self._publish(stage=DenseProviderProgressStage.VIEW_SELECTION, detail="all_registered_temporal_views_selected_i4_3d2", progress=0.05, processed=0, total=total_views)
        prepared = self._sequence_builder.build(images=images, paths=paths, output_dir=provider_dir / "lossless_input")
        if prepared.frame_count != total_views or (prepared.height, prepared.width) != expected_hw:
            raise ReconstructionValidationError("I4.3D2 prepared lossless video does not match the authoritative temporal sequence/grid.")

        input_manifest = provider_dir / "input_manifest.json"
        _write_json(
            input_manifest,
            {
                "schema_version": I43D2_SCHEMA_VERSION,
                "phase": I43D2_PHASE,
                "request_id": f"i4-3d2-{context.sfm_result.session_id}",
                "session_id": context.sfm_result.session_id,
                "authoritative_sfm_frame_id": context.reconstruction_frame_id,
                "rolling_depth_repository": str(context.rolling_depth_repository),
                "checkpoint_path": self.environment.checkpoint_path,
                "source_revision": self.environment.source_revision,
                "device": context.device,
                "precision": context.precision,
                "prepared_video": prepared.as_dict(),
                "expected_frame_count": total_views,
                "expected_height": expected_hw[0],
                "expected_width": expected_hw[1],
                "config": context.config.as_dict(),
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
                    "rolling_depth_camera_pose_authority": False,
                    "rolling_depth_relative_depth_establishes_metric_scale": False,
                    "sequence_alignment": "one_global_affine_to_authoritative_sfm",
                    "per_frame_alignment": False,
                },
            },
        )

        authority_before = _sfm_authority_digest(context.sfm_result)
        images_before = {item.keyframe_id: item.staged_sha256 for item in images}
        self._publish(stage=DenseProviderProgressStage.PROVIDER_EXECUTION, detail="isolated_rolling_depth_sequence_inference", progress=0.10, processed=0, total=total_views)
        worker = self._runner.run(input_manifest, provider_dir / "worker_output")
        depth_stack = np.asarray(worker.depth, dtype=np.float32)
        if depth_stack.ndim != 3 or depth_stack.shape != (total_views, expected_hw[0], expected_hw[1]):
            raise ReconstructionValidationError("I4.3D2 RollingDepth evidence must be NxHxW on the authoritative SfM pixel grid.")
        if not np.isfinite(depth_stack).all():
            raise ReconstructionValidationError("I4.3D2 RollingDepth depth evidence must be finite.")
        if float(np.ptp(depth_stack)) <= 1e-9:
            raise ReconstructionValidationError("I4.3D2 RollingDepth sequence has no usable relative-depth dynamic range.")
        if str(worker.metadata.get("source_revision", "")) != self.environment.source_revision:
            raise ReconstructionValidationError("I4.3D2 RollingDepth worker source revision does not match the probed runtime.")
        if str(worker.metadata.get("depth_semantics", "")) != "temporally_coaligned_relative_depth":
            raise ReconstructionValidationError("I4.3D2 worker depth semantics drifted from temporally co-aligned relative depth.")
        if bool(worker.metadata.get("per_frame_alignment", True)):
            raise ReconstructionValidationError("I4.3D2 refuses worker-side per-frame alignment.")
        self._publish(stage=DenseProviderProgressStage.PROVIDER_EXECUTION, detail="rolling_depth_coaligned_sequence_ready", progress=0.38, processed=total_views, total=total_views)

        alignment = fit_rolling_depth_sequence_to_sfm(
            context.sfm_result,
            ordered_images=images,
            depth_stack=depth_stack,
            minimum_correspondences=context.config.minimum_sequence_correspondences,
            minimum_contributing_views=context.config.minimum_contributing_views,
            maximum_relative_residual=context.config.maximum_relative_residual,
        )
        if not alignment.accepted or alignment.scale_to_reconstruction is None or alignment.shift_to_reconstruction is None:
            raise ReconstructionValidationError(f"I4.3D2 global sequence/SfM affine alignment rejected: {alignment.message}")
        aligned_stack = alignment.scale_to_reconstruction * depth_stack.astype(np.float64) + alignment.shift_to_reconstruction
        valid_stack = np.isfinite(aligned_stack) & (aligned_stack > 1e-8)
        aligned_stack = np.asarray(aligned_stack, dtype=np.float32)
        self._publish(stage=DenseProviderProgressStage.ALIGNMENT, detail="global_affine_temporal_sequence_to_authoritative_sfm", progress=0.70, processed=total_views, total=total_views)

        scene_scale = _robust_scene_diagonal(context.sfm_result)
        voxel_size = max(scene_scale * context.config.voxel_scene_fraction, 1e-8)
        reconstruction_camera = reconstruction_camera_from_sfm_model(
            camera_model, keyframe_id=images[0].keyframe_id, reconstruction_frame_id=context.reconstruction_frame_id
        )
        camera_profile = camera_geometry_from_estimated_camera(
            profile_id=f"i43d2-{camera_id}-sfm-estimated",
            camera=reconstruction_camera,
            focal_length_relative_uncertainty=0.20,
            notes="I4.3D2 camera geometry from authoritative PyCOLMAP SfM; arbitrary gauge only.",
        )
        scale_profile = arbitrary_scale_reference(
            profile_id=f"i43d2-{context.sfm_result.session_id[:12]}-arbitrary-sfm-gauge",
            notes="I4.3D2 one global affine maps RollingDepth relative depth into the arbitrary SfM gauge; not physical metric scale.",
        )
        depth_alignment = reconstruction_unit_depth_alignment(
            provenance=(
                "RollingDepth I4.3D2 global sequence affine already applied;"
                f"source_revision={self.environment.source_revision};"
                f"scale={alignment.scale_to_reconstruction:.12g};shift={alignment.shift_to_reconstruction:.12g};"
                "physical_metric_scale=false"
            )
        )

        frames: list[DenseFusionFrame] = []
        rejected_views: list[str] = []
        for index, item in enumerate(images):
            valid = np.asarray(valid_stack[index], dtype=bool)
            if int(np.count_nonzero(valid)) < 16:
                rejected_views.append(item.keyframe_id)
                continue
            bgr = cv2.imread(str(paths[index]), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ReconstructionValidationError(f"I4.3D2 could not decode staged RGB: {paths[index]}")
            rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)
            if rgb.shape != (expected_hw[0], expected_hw[1], 3):
                raise ReconstructionValidationError("I4.3D2 decoded RGB does not match authoritative SfM grid.")
            camera = camera_by_keyframe[item.keyframe_id]
            frames.append(
                DenseFusionFrame(
                    frame_id=camera.frame_id,
                    timestamp_ns=item.timestamp_ns,
                    rgb=rgb,
                    depth=np.ascontiguousarray(aligned_stack[index], dtype=np.float32),
                    T_reconstruction_from_camera=camera.T_reconstruction_from_camera,
                    depth_unit=DenseDepthUnit.RECONSTRUCTION_UNITS,
                    depth_alignment=depth_alignment,
                    confidence=None,
                    valid_mask=valid,
                    pose_layer="offline_sfm_bundle_adjusted",
                )
            )
        if len(frames) < context.config.minimum_accepted_views:
            raise ReconstructionValidationError(
                "I4.3D2 insufficient positive globally-aligned views: "
                f"accepted={len(frames)} required={context.config.minimum_accepted_views}."
            )

        self._publish(stage=DenseProviderProgressStage.FUSION_FILTERING, detail="qualified_rolling_depth_voxel_fusion", progress=0.82, processed=len(frames), total=total_views)
        fusion = fuse_dense_frames(
            frames,
            session_id=f"{context.sfm_result.session_id}:i4.3d2-rolling-depth",
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
        canonical_npz = context.output_dir / "canonical_rolling_depth_fused.npz"
        np.savez_compressed(
            canonical_npz,
            points_reconstruction=cloud.points_reconstruction,
            colors_rgb=cloud.colors_rgb,
            view_observation_counts=cloud.view_observation_counts,
            mean_confidence=cloud.mean_confidence,
            voxel_size_reconstruction_units=np.asarray(cloud.voxel_size_reconstruction_units, dtype=np.float64),
            source_frame_ids=np.asarray(fusion.source_frame_ids, dtype="U160"),
            source_pose_layers=np.asarray(fusion.source_pose_layers, dtype="U160"),
            rolling_depth_sequence_scale=np.asarray(alignment.scale_to_reconstruction, dtype=np.float64),
            rolling_depth_sequence_shift=np.asarray(alignment.shift_to_reconstruction, dtype=np.float64),
        )
        canonical_ply = _write_binary_ply(context.output_dir / "canonical_rolling_depth_fused.ply", cloud.points_reconstruction, cloud.colors_rgb)

        authority_after = _sfm_authority_digest(context.sfm_result)
        images_after = {item.keyframe_id: _sha256_file(path) for item, path in zip(images, paths, strict=True)}
        if authority_after != authority_before:
            raise ReconstructionValidationError("I4.3D2 authoritative SfM cameras/landmarks changed during RollingDepth execution.")
        if images_after != images_before:
            raise ReconstructionValidationError("I4.3D2 registered staged images changed during RollingDepth execution.")
        if any(layer != "offline_sfm_bundle_adjusted" for layer in fusion.source_pose_layers):
            raise ReconstructionValidationError("I4.3D2 canonical fusion consumed a non-SfM pose layer.")

        degradation_reasons: list[str] = []
        if rejected_views:
            degradation_reasons.append(f"nonpositive_aligned_views:{len(rejected_views)}")
        if fusion.statistics.dropped_new_voxels_capacity:
            degradation_reasons.append(f"fusion_capacity_drops:{fusion.statistics.dropped_new_voxels_capacity}")
        status = DenseProviderRunStatus.DEGRADED if degradation_reasons else DenseProviderRunStatus.COMPLETED
        self._publish(stage=DenseProviderProgressStage.FINALIZATION, detail="canonical_rolling_depth_geometry_ready", progress=1.0, processed=total_views, total=total_views)

        result = DenseProviderExecutionResult(
            provider_id=self.descriptor.provider_id,
            implementation=DenseProviderImplementation.ROLLING_DEPTH,
            status=status,
            source_view_count=total_views,
            selected_view_count=total_views,
            adaptive_view_selection=False,
            view_selection_reason="all_registered_authoritative_sfm_views_in_temporal_order",
            permanent_global_view_cap=None,
            canonical_geometry_ref=str(canonical_npz),
            canonical_frame_id=context.reconstruction_frame_id,
            native_artifact_refs=(str(prepared.video_path), str(worker.manifest_path), str(worker.depth_npz_path), str(canonical_ply)),
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
            "schema_version": I43D2_SCHEMA_VERSION,
            "phase": I43D2_PHASE,
            "status": "degraded" if status is DenseProviderRunStatus.DEGRADED else "pass",
            "pass": True,
            "provider": self.descriptor.as_dict(),
            "selection_request": request.as_dict(),
            "selection_decision": decision.as_dict(),
            "execution_result": result.as_dict(),
            "config": context.config.as_dict(),
            "runtime": self.environment.as_dict(),
            "prepared_video": prepared.as_dict(),
            "worker": dict(worker.metadata),
            "temporal_sequence": {
                "ordered_keyframe_ids": [item.keyframe_id for item in images],
                "ordered_source_sequence_indices": [item.source_sequence_index for item in images],
                "ordered_timestamps_ns": [item.timestamp_ns for item in images],
                "sequence_reordered_by": "source_sequence_index_then_timestamp",
                "native_temporal_coalignment_preserved": True,
            },
            "alignment": alignment.as_dict(),
            "fusion": {
                "accepted_views": len(frames),
                "rejected_views": rejected_views,
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
                "rolling_depth_pose_authority_promoted": False,
                "sequence_affine_establishes_physical_metric_scale": False,
                "per_frame_alignment_performed": False,
                "metric_scale_claimed": False,
                "measurement_authoritative": False,
                "certified_metrology": False,
                "surface_mesh_generated": False,
                "provider_promotion_performed": False,
            },
        }
        _write_json(context.output_dir / "i4_3d2_rolling_depth_execution_summary.json", summary)
        return result


def validate_i4_3d2_binding(
    *,
    i4_3a_snapshot: Mapping[str, Any],
    i4_3d1_snapshot: Mapping[str, Any],
    environment: RollingDepthEnvironmentProbe,
) -> dict[str, Any]:
    if not isinstance(i4_3a_snapshot, Mapping) or i4_3a_snapshot.get("phase") != "I4.3A" or i4_3a_snapshot.get("pass") is not True:
        raise ReconstructionValidationError("I4.3D2 requires the passing I4.3A architecture snapshot.")
    architecture = i4_3a_snapshot.get("architecture")
    if not isinstance(architecture, Mapping):
        raise ReconstructionValidationError("I4.3D2 I4.3A snapshot is missing architecture.")
    if architecture.get("authoritative_camera_frame_owner") != I43A_AUTHORITATIVE_FRAME_OWNER:
        raise ReconstructionValidationError("I4.3D2 refuses changed camera-frame authority.")
    if architecture.get("permanent_global_view_cap") is not None:
        raise ReconstructionValidationError("I4.3D2 refuses a permanent global dense-view cap.")
    contracts = architecture.get("provider_implementation_contracts")
    if not isinstance(contracts, list):
        raise ReconstructionValidationError("I4.3D2 I4.3A implementation contracts are missing.")
    match = next((item for item in contracts if isinstance(item, Mapping) and item.get("implementation") == "rolling_depth"), None)
    if not isinstance(match, Mapping):
        raise ReconstructionValidationError("I4.3D2 I4.3A RollingDepth implementation contract is missing.")
    if match.get("integration_stage") != "I4.3D" or match.get("family") != "temporal_depth":
        raise ReconstructionValidationError("I4.3D2 RollingDepth implementation is assigned to another stage/family.")
    if match.get("frame_binding") != "align_to_authoritative_sfm" or match.get("native_output") != "depth_evidence":
        raise ReconstructionValidationError("I4.3D2 RollingDepth frame/native-output contract drifted.")
    if match.get("existing_adapter_expected") is not False:
        raise ReconstructionValidationError("I4.3D2 expected I4.3A to record RollingDepth as a new isolated adapter integration.")

    if not isinstance(i4_3d1_snapshot, Mapping) or i4_3d1_snapshot.get("phase") != "I4.3D1" or i4_3d1_snapshot.get("pass") is not True:
        raise ReconstructionValidationError("I4.3D2 requires the passing I4.3D1 VDA binding snapshot.")
    d1_provider = i4_3d1_snapshot.get("provider")
    if not isinstance(d1_provider, Mapping) or d1_provider.get("implementation") != "video_depth_anything":
        raise ReconstructionValidationError("I4.3D2 I4.3D1 snapshot does not retain Video Depth Anything.")
    d1_authority = i4_3d1_snapshot.get("authority")
    if not isinstance(d1_authority, Mapping) or d1_authority.get("metric_scale_claimed") is not False:
        raise ReconstructionValidationError("I4.3D2 refuses an I4.3D1 snapshot with changed authority limits.")

    descriptor = rolling_depth_dense_provider_descriptor(environment)
    if not descriptor.available:
        raise ReconstructionValidationError(f"I4.3D2 RollingDepth runtime is not ready: {descriptor.unavailable_reason}")
    return {
        "schema_version": I43D2_SCHEMA_VERSION,
        "phase": I43D2_PHASE,
        "status": "pass",
        "pass": True,
        "provider": descriptor.as_dict(),
        "architecture": {
            "camera_pose_authority_owner": I43A_AUTHORITATIVE_FRAME_OWNER,
            "frame_binding": "align_to_authoritative_sfm",
            "native_output": "depth_evidence",
            "temporal_mode": "multi_dilation_snippet_coalignment",
            "sequence_level_alignment": "global_affine_to_authoritative_sfm",
            "per_frame_alignment": False,
            "all_registered_views_default_i4_3d2": True,
            "permanent_global_view_cap": None,
            "adaptive_view_selection_deferred_to": "I4.3F",
            "progressive_status_supported": True,
        },
        "authority": {
            "backend_started": False,
            "provider_selected": False,
            "provider_promotion_performed": False,
            "rolling_depth_pose_authority_promoted": False,
            "sequence_affine_establishes_physical_metric_scale": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
        },
        "qualification": {
            "qualification_state": "candidate",
            "video_depth_anything_binding_retained": True,
            "temporal_provider_closure_pending_i4_3d3": True,
            "temporal_cross_provider_comparison_pending_i4_3g_i5": True,
            "universal_provider_winner_assumed": False,
        },
        "next": "I4.3D3 temporal-provider architecture closure. No provider promotion before I4.3G/I5 evidence.",
    }


__all__ = [
    "I43D2_SCHEMA_VERSION",
    "I43D2_PHASE",
    "I43D2_ADAPTER_VERSION",
    "I43D2_PROVIDER_ID",
    "I43D2_RUNTIME_LOCK_NAME",
    "RollingDepthSequenceAlignmentEvidence",
    "fit_rolling_depth_sequence_to_sfm",
    "PreparedRollingDepthVideo",
    "RollingDepthSequenceBuilder",
    "FfmpegLosslessRollingDepthSequenceBuilder",
    "RollingDepthEnvironmentProbe",
    "probe_rolling_depth_environment",
    "rolling_depth_dense_provider_descriptor",
    "RollingDepthProductionConfig",
    "RollingDepthProviderRunContext",
    "RollingDepthWorkerResult",
    "RollingDepthWorkerRunner",
    "ExternalRollingDepthWorkerRunner",
    "RollingDepthProductionProviderAdapter",
    "validate_i4_3d2_binding",
]

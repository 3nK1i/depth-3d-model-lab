"""I4.3E6 canonical production adapter for Marigold Depth v1.1.

Marigold predicts affine-invariant relative depth (or disparity) and can expose
ensemble uncertainty on higher-quality profiles. I4.3E6 keeps this as model
evidence, aligns each registered view to the authoritative PyCOLMAP SfM gauge
in the correct native affine domain, and projects only with bundle-adjusted SfM
cameras. The alignment establishes reconstruction gauge only, never physical
metric scale or metrology authority.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
from importlib import metadata as importlib_metadata
import json
from math import isfinite
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping, Protocol

import cv2
import numpy as np

from core.contracts.validation import JsonValue, metadata_as_dict, normalize_metadata, require_nonempty
from core.models.catalog import MARIGOLD_MODEL_ID
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
    DenseProviderSelectionMode,
    DenseProviderSelectionRequest,
)
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.orchestration import ReconstructionProviderExecution
from core.reconstruction.scale_reference import arbitrary_scale_reference
from core.reconstruction.sfm_contracts import OfflineSfmResult
from core.reconstruction.sfm_dense_projection import reconstruction_camera_from_sfm_model
from core.reconstruction.sfm_pycolmap import PYCOLMAP_BACKEND_ID
from core.reconstruction.sfm_coordinates import invert_rigid_matrix

I43E6_SCHEMA_VERSION = 1
I43E6_PHASE = "I4.3E6"
I43E6_ADAPTER_VERSION = "i4.3e6-v1"
I43E6_PROVIDER_ID = "marigold-single-image"
I43E6_CHECKPOINT_REPOSITORY = "prs-eth/marigold-depth-v1-1"
I43E6_DEFAULT_CHECKPOINT_DIR = "checkpoints/marigold/depth-v1-1"
I43E6_DEFAULT_MINIMUM_CORRESPONDENCES = 8
I43E6_DEFAULT_MAXIMUM_RELATIVE_RESIDUAL = 0.35
I43E6_DEFAULT_VOXEL_SCENE_FRACTION = 0.004
I43E6_DEFAULT_PIXEL_STRIDE = 2
I43E6_DEFAULT_MAX_VOXELS = 1_500_000
I43E6_DEFAULT_MINIMUM_ACCEPTED_VIEWS = 2

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
        raise ReconstructionValidationError(f"I4.3E6 could not read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"I4.3E6 JSON root must be an object: {path}")
    return payload


def _robust_scene_diagonal(sfm_result: OfflineSfmResult) -> float:
    if len(sfm_result.landmarks) < 4:
        raise ReconstructionValidationError("I4.3E6 requires sparse SfM landmarks for scene-relative fusion scale.")
    points = np.vstack([np.asarray(item.position_reconstruction, dtype=np.float64) for item in sfm_result.landmarks])
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ReconstructionValidationError("I4.3E6 sparse SfM landmarks are invalid.")
    low = np.quantile(points, 0.02, axis=0)
    high = np.quantile(points, 0.98, axis=0)
    diagonal = float(np.linalg.norm(high - low))
    if not isfinite(diagonal) or diagonal <= 0.0:
        raise ReconstructionValidationError("I4.3E6 robust sparse scene diagonal is invalid.")
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
        raise ReconstructionValidationError("I4.3E6 PLY geometry must be aligned Nx3 XYZ/RGB arrays.")
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
class MarigoldEnvironmentProbe:
    python_executable: str
    project_root: str
    distribution_name: str
    distribution_version: str
    distribution_root: str
    official_runtime_importable: bool
    official_runtime_source_matches_distribution: bool
    runtime_requirement_exact_pinned: bool
    profile: str
    checkpoint_repository: str
    checkpoint_requested_revision: str
    checkpoint_resolved_revision: str
    checkpoint_revision_pinned: bool
    checkpoint_path: str
    checkpoint_marker_sha256: str
    model_index_sha256: str
    checkpoint_file_count: int
    checkpoint_size_bytes: int
    checkpoint_ready: bool
    adapter_importable: bool
    torch_available: bool
    cuda_available: bool
    device_names: tuple[str, ...] = ()
    error: str = ""

    @property
    def available(self) -> bool:
        return bool(
            self.official_runtime_importable
            and self.official_runtime_source_matches_distribution
            and self.checkpoint_revision_pinned
            and self.checkpoint_ready
            and self.adapter_importable
            and self.torch_available
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "python_executable": self.python_executable,
            "project_root": self.project_root,
            "distribution_name": self.distribution_name,
            "distribution_version": self.distribution_version,
            "distribution_root": self.distribution_root,
            "official_runtime_importable": self.official_runtime_importable,
            "official_runtime_source_matches_distribution": self.official_runtime_source_matches_distribution,
            "runtime_requirement_exact_pinned": self.runtime_requirement_exact_pinned,
            "profile": self.profile,
            "checkpoint_repository": self.checkpoint_repository,
            "checkpoint_requested_revision": self.checkpoint_requested_revision,
            "checkpoint_resolved_revision": self.checkpoint_resolved_revision,
            "checkpoint_revision_pinned": self.checkpoint_revision_pinned,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_marker_sha256": self.checkpoint_marker_sha256,
            "model_index_sha256": self.model_index_sha256,
            "checkpoint_file_count": self.checkpoint_file_count,
            "checkpoint_size_bytes": self.checkpoint_size_bytes,
            "checkpoint_ready": self.checkpoint_ready,
            "adapter_importable": self.adapter_importable,
            "torch_available": self.torch_available,
            "cuda_available": self.cuda_available,
            "device_names": list(self.device_names),
            "available": self.available,
            "error": self.error,
        }


def _marigold_requirement_exact_pinned(requirement_path: Path) -> bool:
    if not requirement_path.is_file():
        return False
    try:
        text = requirement_path.read_text(encoding="utf-8")
    except OSError:
        return False
    return bool(re.search(r"(?im)^\s*diffusers==[^\s#]+", text))


def _auto_marigold_profile(torch_module: Any | None = None) -> str:
    try:
        torch = torch_module
        if torch is None:
            import torch  # type: ignore
        if not torch.cuda.is_available():
            return "efficient"
        total_gib = float(torch.cuda.get_device_properties(0).total_memory) / 1024**3
    except Exception:  # noqa: BLE001
        return "efficient"
    if total_gib >= 20.0:
        return "ultra"
    if total_gib >= 10.0:
        return "balanced"
    return "efficient"


def _checkpoint_tree_stats(directory: Path) -> tuple[int, int]:
    count = 0
    total = 0
    for path in directory.rglob("*"):
        if not path.is_file():
            continue
        try:
            total += int(path.stat().st_size)
            count += 1
        except OSError:
            continue
    return count, total


def probe_marigold_environment(
    *,
    project_root: Path | None = None,
    python_executable: Path | None = None,
    profile: str = "auto",
    checkpoint_dir: Path | None = None,
) -> MarigoldEnvironmentProbe:
    root = Path(project_root or Path.cwd()).expanduser().resolve()
    python = Path(python_executable or sys.executable).expanduser().resolve()
    selected = str(profile).strip().lower()
    if selected == "auto":
        selected = _auto_marigold_profile()
    if selected not in {"efficient", "balanced", "ultra"}:
        raise ValueError(f"Unsupported Marigold profile: {profile!r}.")
    checkpoint = Path(
        checkpoint_dir or (root / "checkpoints/marigold/depth-v1-1")
    ).expanduser().resolve()
    marker_path = checkpoint / ".ready"
    model_index = checkpoint / "model_index.json"
    errors: list[str] = []

    adapter_importable = False
    try:
        from core.models.adapters.marigold_adapter import MarigoldGeometryModel
        adapter_importable = MarigoldGeometryModel is not None
    except Exception as exc:  # noqa: BLE001
        errors.append(f"adapter_import_failed:{exc}")

    distribution_name = "diffusers"
    distribution_version = ""
    distribution_root = ""
    distribution_root_path: Path | None = None
    try:
        distribution = importlib_metadata.distribution("diffusers")
        distribution_name = str(distribution.metadata.get("Name") or "diffusers")
        distribution_version = str(distribution.version or "")
        distribution_root_path = Path(distribution.locate_file("diffusers")).expanduser().resolve()
        distribution_root = str(distribution_root_path)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"distribution_probe_failed:{exc}")

    official_runtime_importable = False
    runtime_matches_distribution = False
    try:
        from diffusers import MarigoldDepthPipeline
        official_runtime_importable = MarigoldDepthPipeline is not None
        runtime_file = Path(inspect.getfile(MarigoldDepthPipeline)).expanduser().resolve()
        if distribution_root_path is not None:
            runtime_matches_distribution = (
                runtime_file == distribution_root_path or distribution_root_path in runtime_file.parents
            )
        if not runtime_matches_distribution:
            errors.append(
                "MarigoldDepthPipeline does not resolve to the installed Diffusers distribution: "
                f"{runtime_file}"
            )
    except Exception as exc:  # noqa: BLE001
        errors.append(f"official_runtime_import_failed:{exc}")

    checkpoint_repository = ""
    checkpoint_requested_revision = ""
    checkpoint_resolved_revision = ""
    checkpoint_revision_pinned = False
    checkpoint_marker_sha = ""
    model_index_sha = ""
    checkpoint_file_count = 0
    checkpoint_size = 0
    checkpoint_ready = False
    if marker_path.is_file() and model_index.is_file():
        try:
            marker = _read_json(marker_path)
            checkpoint_repository = str(marker.get("repository") or "").strip()
            checkpoint_requested_revision = str(marker.get("requested_revision") or "").strip()
            checkpoint_resolved_revision = str(marker.get("resolved_revision") or "").strip()
            checkpoint_revision_pinned = bool(
                checkpoint_repository == "prs-eth/marigold-depth-v1-1"
                and re.fullmatch(r"[0-9a-fA-F]{40}", checkpoint_resolved_revision)
            )
            checkpoint_marker_sha = _sha256_file(marker_path)
            model_index_sha = _sha256_file(model_index)
            checkpoint_file_count, checkpoint_size = _checkpoint_tree_stats(checkpoint)
            checkpoint_ready = bool(
                checkpoint_revision_pinned
                and checkpoint_file_count >= 3
                and checkpoint_size >= 100 * 1024 * 1024
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"checkpoint_probe_failed:{exc}")
    if not checkpoint_ready:
        errors.append(
            "Marigold pinned local snapshot is missing, incomplete, or lacks an exact resolved Hugging Face revision: "
            f"{checkpoint}"
        )

    torch_available = False
    cuda_available = False
    device_names: tuple[str, ...] = ()
    try:
        import torch
        torch_available = True
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            device_names = tuple(
                str(torch.cuda.get_device_name(index)) for index in range(torch.cuda.device_count())
            )
    except Exception as exc:  # noqa: BLE001
        errors.append(f"torch_import_failed:{exc}")
    if not python.is_file():
        errors.append(f"Python executable missing: {python}")

    return MarigoldEnvironmentProbe(
        python_executable=str(python),
        project_root=str(root),
        distribution_name=distribution_name,
        distribution_version=distribution_version,
        distribution_root=distribution_root,
        official_runtime_importable=official_runtime_importable,
        official_runtime_source_matches_distribution=runtime_matches_distribution,
        runtime_requirement_exact_pinned=_marigold_requirement_exact_pinned(
            root / "requirements/models-marigold.txt"
        ),
        profile=selected,
        checkpoint_repository=checkpoint_repository,
        checkpoint_requested_revision=checkpoint_requested_revision,
        checkpoint_resolved_revision=checkpoint_resolved_revision,
        checkpoint_revision_pinned=checkpoint_revision_pinned,
        checkpoint_path=str(checkpoint),
        checkpoint_marker_sha256=checkpoint_marker_sha,
        model_index_sha256=model_index_sha,
        checkpoint_file_count=checkpoint_file_count,
        checkpoint_size_bytes=checkpoint_size,
        checkpoint_ready=checkpoint_ready,
        adapter_importable=adapter_importable,
        torch_available=torch_available,
        cuda_available=cuda_available,
        device_names=device_names,
        error="; ".join(errors),
    )


def marigold_dense_provider_descriptor(probe: MarigoldEnvironmentProbe) -> DenseProviderDescriptor:
    if not isinstance(probe, MarigoldEnvironmentProbe):
        raise TypeError("probe must be MarigoldEnvironmentProbe.")
    backend_version = probe.checkpoint_resolved_revision or probe.distribution_version or "unresolved"
    return DenseProviderDescriptor(
        provider_id=I43E6_PROVIDER_ID,
        implementation=DenseProviderImplementation.MARIGOLD,
        execution=ReconstructionProviderExecution.EXTERNAL_PROCESS,
        backend_version=backend_version,
        adapter_version=I43E6_ADAPTER_VERSION,
        qualification_state=DenseProviderQualificationState.CANDIDATE,
        available=probe.available,
        unavailable_reason="" if probe.available else (probe.error or "marigold_runtime_unavailable"),
        metadata={
            "native_output": "depth_evidence",
            "frame_binding": "align_to_authoritative_sfm",
            "native_depth_scale": "relative",
            "native_depth_representation": "affine_invariant_relative_depth_or_disparity",
            "native_intrinsics_source": "estimated",
            "native_confidence_kind": "uncertainty_optional",
            "native_valid_mask_available": True,
            "native_direct_points_available": False,
            "native_normals_available": False,
            "adapter_projected_points_discarded": True,
            "single_image_stateful_sequence": False,
            "all_registered_views_available_by_default": True,
            "adaptive_view_selection_deferred_to": "I4.3F",
            "permanent_global_view_cap": None,
            "per_view_alignment": "affine_domain_according_to_prediction_type_to_authoritative_sfm",
            "relative_depth_promoted_to_metric_scale": False,
            "estimated_model_intrinsics_consumed_for_canonical_projection": False,
            "uncertainty_semantics": "lower_is_better_when_available",
            "uncertainty_to_fusion_confidence": "1/(1+uncertainty) when available",
            "runtime_distribution": probe.distribution_name,
            "runtime_distribution_version": probe.distribution_version,
            "runtime_requirement_exact_pinned": probe.runtime_requirement_exact_pinned,
            "profile": probe.profile,
            "checkpoint_repository": probe.checkpoint_repository,
            "checkpoint_requested_revision": probe.checkpoint_requested_revision,
            "checkpoint_resolved_revision": probe.checkpoint_resolved_revision,
            "checkpoint_revision_pinned": probe.checkpoint_revision_pinned,
            "checkpoint_path": probe.checkpoint_path,
            "checkpoint_marker_sha256": probe.checkpoint_marker_sha256,
            "model_index_sha256": probe.model_index_sha256,
            "license_status": "review_required",
            "camera_pose_authority_owner": I43A_AUTHORITATIVE_FRAME_OWNER,
        },
    )


@dataclass(frozen=True, slots=True)
class MarigoldAffineAlignmentFit:
    domain: str
    slope: float
    intercept: float
    input_correspondences: int
    valid_correspondences: int
    inlier_correspondences: int
    robust_relative_residual_limit: float
    median_relative_residual: float
    p90_relative_residual: float

    def __post_init__(self) -> None:
        if self.domain not in {"depth", "inverse_depth"}:
            raise ReconstructionValidationError("I4.3E6 Marigold alignment domain is invalid.")
        if not isfinite(float(self.slope)) or float(self.slope) <= 0.0:
            raise ReconstructionValidationError("I4.3E6 Marigold alignment slope must be finite and positive.")
        if not isfinite(float(self.intercept)):
            raise ReconstructionValidationError("I4.3E6 Marigold alignment intercept must be finite.")
        for name in ("input_correspondences", "valid_correspondences", "inlier_correspondences"):
            if int(getattr(self, name)) < 0:
                raise ReconstructionValidationError(f"I4.3E6 {name} must be non-negative.")
        for name in ("robust_relative_residual_limit", "median_relative_residual", "p90_relative_residual"):
            value = float(getattr(self, name))
            if not isfinite(value) or value < 0.0:
                raise ReconstructionValidationError(f"I4.3E6 {name} must be finite and non-negative.")

    @property
    def method(self) -> str:
        return (
            "affine_depth_to_authoritative_sfm"
            if self.domain == "depth"
            else "affine_inverse_depth_to_authoritative_sfm"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "domain": self.domain,
            "slope": float(self.slope),
            "intercept": float(self.intercept),
            "input_correspondences": int(self.input_correspondences),
            "valid_correspondences": int(self.valid_correspondences),
            "inlier_correspondences": int(self.inlier_correspondences),
            "robust_relative_residual_limit": float(self.robust_relative_residual_limit),
            "median_relative_residual": float(self.median_relative_residual),
            "p90_relative_residual": float(self.p90_relative_residual),
            "establishes_physical_metric_scale": False,
        }


_OUT_OF_BOUNDS = object()


def _bilinear_positive_value(array: np.ndarray, *, x: float, y: float) -> float | None | object:
    height, width = array.shape
    if x < 0.0 or y < 0.0 or x > width - 1 or y > height - 1:
        return _OUT_OF_BOUNDS
    x0 = int(np.floor(x)); y0 = int(np.floor(y))
    x1 = min(x0 + 1, width - 1); y1 = min(y0 + 1, height - 1)
    values = np.asarray([array[y0, x0], array[y0, x1], array[y1, x0], array[y1, x1]], dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        return None
    wx = float(x) - x0; wy = float(y) - y0
    top = values[0] * (1.0 - wx) + values[1] * wx
    bottom = values[2] * (1.0 - wx) + values[3] * wx
    sampled = float(top * (1.0 - wy) + bottom * wy)
    return sampled if isfinite(sampled) and sampled > 0.0 else None


def align_marigold_relative_depth_to_sfm(
    sfm_result: OfflineSfmResult,
    *,
    keyframe_id: str,
    relative_depth: np.ndarray,
    prediction_type: str,
    source_valid_mask: np.ndarray | None = None,
    minimum_correspondences: int = I43E6_DEFAULT_MINIMUM_CORRESPONDENCES,
    maximum_relative_residual: float = I43E6_DEFAULT_MAXIMUM_RELATIVE_RESIDUAL,
) -> tuple[np.ndarray, np.ndarray, MarigoldAffineAlignmentFit]:
    """Align affine-invariant Marigold evidence to one authoritative SfM camera gauge.

    Native Marigold ``depth`` is affine-invariant in depth space. Native
    ``disparity`` is affine-invariant in inverse-depth space; the mature app
    adapter converts that disparity to monotonic relative distance, so taking
    its reciprocal restores the affine disparity domain for alignment.
    """
    depth = np.asarray(relative_depth, dtype=np.float64)
    if depth.ndim != 2:
        raise ReconstructionValidationError("I4.3E6 relative depth must be a 2D array.")
    mode = str(prediction_type).strip().lower()
    if mode not in {"depth", "disparity"}:
        raise ReconstructionValidationError("I4.3E6 Marigold prediction_type must be depth or disparity.")
    domain = "depth" if mode == "depth" else "inverse_depth"
    minimum = int(minimum_correspondences)
    if minimum < 3:
        raise ReconstructionValidationError("I4.3E6 minimum_correspondences must be at least 3.")
    maximum = float(maximum_relative_residual)
    if not isfinite(maximum) or not 0.0 < maximum < 1.0:
        raise ReconstructionValidationError("I4.3E6 maximum_relative_residual must be finite in (0,1).")

    camera = next((item for item in sfm_result.cameras if item.keyframe_id == keyframe_id), None)
    if camera is None:
        raise ReconstructionValidationError("I4.3E6 keyframe is not registered in authoritative SfM.")
    camera_model = next((item for item in sfm_result.camera_models if item.camera_id == camera.camera_id), None)
    workspace_image = next((item for item in sfm_result.workspace.images if item.keyframe_id == keyframe_id), None)
    if camera_model is None or workspace_image is None:
        raise ReconstructionValidationError("I4.3E6 authoritative SfM camera/image metadata is unavailable.")
    expected_hw = (workspace_image.image_size.height, workspace_image.image_size.width)
    if depth.shape != expected_hw or (camera_model.image_size.height, camera_model.image_size.width) != expected_hw:
        raise ReconstructionValidationError(
            "I4.3E6 relative depth, SfM camera model, and staged image must share one pixel grid."
        )

    base_valid = np.isfinite(depth) & (depth > 0.0)
    if source_valid_mask is not None:
        mask = np.asarray(source_valid_mask, dtype=bool)
        if mask.shape != depth.shape:
            raise ReconstructionValidationError("I4.3E6 source valid mask pixel grid drifted.")
        base_valid &= mask

    T_camera_from_reconstruction = invert_rigid_matrix(camera.T_reconstruction_from_camera)
    source_values: list[float] = []
    target_values: list[float] = []
    candidate = 0
    for landmark in sfm_result.landmarks:
        observation = next((item for item in landmark.observations if item.keyframe_id == keyframe_id), None)
        if observation is None:
            continue
        candidate += 1
        position_camera = T_camera_from_reconstruction[:3, :3] @ landmark.position_reconstruction + T_camera_from_reconstruction[:3, 3]
        camera_z = float(position_camera[2])
        if not isfinite(camera_z) or camera_z <= 0.0:
            continue
        x_px, y_px = observation.feature_xy
        sampled = _bilinear_positive_value(depth, x=float(x_px), y=float(y_px))
        if sampled is _OUT_OF_BOUNDS or sampled is None:
            continue
        x0 = min(max(int(np.floor(float(x_px))), 0), depth.shape[1] - 1)
        y0 = min(max(int(np.floor(float(y_px))), 0), depth.shape[0] - 1)
        x1 = min(x0 + 1, depth.shape[1] - 1); y1 = min(y0 + 1, depth.shape[0] - 1)
        if not bool(base_valid[y0, x0] and base_valid[y0, x1] and base_valid[y1, x0] and base_valid[y1, x1]):
            continue
        if domain == "depth":
            source_values.append(float(sampled)); target_values.append(camera_z)
        else:
            source_values.append(1.0 / float(sampled)); target_values.append(1.0 / camera_z)

    x = np.asarray(source_values, dtype=np.float64)
    y = np.asarray(target_values, dtype=np.float64)
    valid_count = int(len(x))
    if valid_count < minimum:
        raise ReconstructionValidationError(
            f"I4.3E6 insufficient Marigold/SfM {domain} correspondences: {valid_count} < {minimum}."
        )
    qx = np.quantile(x, [0.10, 0.90]); qy = np.quantile(y, [0.10, 0.90])
    dx = float(qx[1] - qx[0])
    if not isfinite(dx) or abs(dx) <= 1e-12:
        raise ReconstructionValidationError(f"I4.3E6 Marigold {domain} correspondences are degenerate.")
    slope0 = float((qy[1] - qy[0]) / dx)
    if not isfinite(slope0) or slope0 <= 0.0:
        raise ReconstructionValidationError(f"I4.3E6 Marigold {domain} ordering is incompatible with SfM depth.")
    intercept0 = float(np.median(y - slope0 * x))
    pred0 = slope0 * x + intercept0
    positive0 = pred0 > 1e-12
    residual0 = np.full_like(y, np.inf)
    residual0[positive0] = np.abs(y[positive0] - pred0[positive0]) / np.maximum(np.abs(y[positive0]), 1e-12)
    finite_residual = residual0[np.isfinite(residual0)]
    if len(finite_residual) < minimum:
        raise ReconstructionValidationError(f"I4.3E6 Marigold {domain} affine initialization is invalid.")
    median0 = float(np.median(finite_residual)); mad0 = float(np.median(np.abs(finite_residual - median0)))
    robust_limit = min(maximum, max(0.03, median0 + 3.5 * 1.4826 * mad0))
    inliers = np.isfinite(residual0) & (residual0 <= robust_limit)
    inlier_count = int(np.count_nonzero(inliers))
    if inlier_count < minimum:
        raise ReconstructionValidationError(f"I4.3E6 Marigold {domain} affine fit rejected too many correspondences.")

    design = np.column_stack((x[inliers], np.ones(inlier_count, dtype=np.float64)))
    slope, intercept = np.linalg.lstsq(design, y[inliers], rcond=None)[0]
    slope = float(slope); intercept = float(intercept)
    if not isfinite(slope) or slope <= 0.0 or not isfinite(intercept):
        raise ReconstructionValidationError(f"I4.3E6 fitted Marigold {domain} affine is invalid.")
    prediction = slope * x[inliers] + intercept
    positive = prediction > 1e-12
    positive_count = int(np.count_nonzero(positive))
    if positive_count < minimum:
        raise ReconstructionValidationError(f"I4.3E6 fitted Marigold {domain} affine predicts non-positive depth domain.")
    final_residual = np.abs(y[inliers][positive] - prediction[positive]) / np.maximum(np.abs(y[inliers][positive]), 1e-12)
    median_final = float(np.median(final_residual)); p90_final = float(np.quantile(final_residual, 0.90))
    if p90_final > maximum:
        raise ReconstructionValidationError(f"I4.3E6 Marigold {domain} affine residual exceeds the accepted bound.")

    aligned_depth = np.zeros_like(depth, dtype=np.float32)
    if domain == "depth":
        aligned_domain = slope * depth + intercept
        aligned_valid = base_valid & np.isfinite(aligned_domain) & (aligned_domain > 1e-12)
        aligned_depth[aligned_valid] = aligned_domain[aligned_valid].astype(np.float32)
    else:
        source_inverse_map = np.zeros_like(depth, dtype=np.float64)
        source_inverse_map[base_valid] = 1.0 / depth[base_valid]
        aligned_inverse = slope * source_inverse_map + intercept
        aligned_valid = base_valid & np.isfinite(aligned_inverse) & (aligned_inverse > 1e-12)
        aligned_depth[aligned_valid] = (1.0 / aligned_inverse[aligned_valid]).astype(np.float32)
    aligned_valid &= np.isfinite(aligned_depth) & (aligned_depth > 0.0)

    fit = MarigoldAffineAlignmentFit(
        domain=domain,
        slope=slope,
        intercept=intercept,
        input_correspondences=candidate,
        valid_correspondences=valid_count,
        inlier_correspondences=positive_count,
        robust_relative_residual_limit=robust_limit,
        median_relative_residual=median_final,
        p90_relative_residual=p90_final,
    )
    return np.ascontiguousarray(aligned_depth), np.ascontiguousarray(aligned_valid), fit


@dataclass(frozen=True, slots=True)
class MarigoldProductionConfig:
    minimum_correspondences: int = I43E6_DEFAULT_MINIMUM_CORRESPONDENCES
    maximum_relative_residual: float = I43E6_DEFAULT_MAXIMUM_RELATIVE_RESIDUAL
    voxel_scene_fraction: float = I43E6_DEFAULT_VOXEL_SCENE_FRACTION
    pixel_stride: int = I43E6_DEFAULT_PIXEL_STRIDE
    max_voxels: int = I43E6_DEFAULT_MAX_VOXELS
    minimum_accepted_views: int = I43E6_DEFAULT_MINIMUM_ACCEPTED_VIEWS

    def __post_init__(self) -> None:
        if int(self.minimum_correspondences) < 3:
            raise ReconstructionValidationError("I4.3E6 minimum_correspondences must be at least 3.")
        residual = float(self.maximum_relative_residual)
        if not isfinite(residual) or not 0.0 < residual < 1.0:
            raise ReconstructionValidationError("I4.3E6 maximum_relative_residual must be finite in (0,1).")
        fraction = float(self.voxel_scene_fraction)
        if not isfinite(fraction) or not 0.0 < fraction < 0.1:
            raise ReconstructionValidationError("I4.3E6 voxel_scene_fraction must be finite in (0,0.1).")
        if int(self.pixel_stride) <= 0 or int(self.max_voxels) <= 0:
            raise ReconstructionValidationError("I4.3E6 pixel_stride/max_voxels must be positive.")
        if int(self.minimum_accepted_views) < 2:
            raise ReconstructionValidationError("I4.3E6 minimum_accepted_views must be at least 2.")
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
class MarigoldProviderRunContext:
    sfm_result: OfflineSfmResult
    reconstruction_frame_id: str
    output_dir: Path
    python_executable: Path
    worker_script: Path
    runtime_config: Mapping[str, JsonValue]
    device: str = "cuda:0"
    precision: str = "fp16"
    config: MarigoldProductionConfig = MarigoldProductionConfig()

    def __post_init__(self) -> None:
        if not isinstance(self.sfm_result, OfflineSfmResult):
            raise TypeError("sfm_result must be OfflineSfmResult.")
        if self.sfm_result.backend_id != PYCOLMAP_BACKEND_ID:
            raise ReconstructionValidationError("I4.3E6 requires the authoritative PyCOLMAP offline SfM result.")
        if self.sfm_result.statistics.registered_keyframes < 2:
            raise ReconstructionValidationError("I4.3E6 requires at least two registered SfM views.")
        if not self.sfm_result.camera_models:
            raise ReconstructionValidationError("I4.3E6 requires SfM camera-model estimates.")
        frame_id = require_nonempty(self.reconstruction_frame_id, field_name="reconstruction_frame_id")
        object.__setattr__(self, "reconstruction_frame_id", frame_id)
        for name in ("output_dir", "python_executable", "worker_script"):
            object.__setattr__(self, name, Path(getattr(self, name)).expanduser().resolve())
        object.__setattr__(self, "runtime_config", normalize_metadata(self.runtime_config))
        if not isinstance(self.config, MarigoldProductionConfig):
            raise TypeError("config must be MarigoldProductionConfig.")
        device = require_nonempty(self.device, field_name="device")
        precision = str(self.precision).strip().lower()
        if precision not in {"fp32", "fp16"}:
            raise ReconstructionValidationError("I4.3E6 precision must be fp32 or fp16.")
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "precision", precision)


@dataclass(frozen=True, slots=True)
class MarigoldWorkerResult:
    manifest_path: Path
    metadata: Mapping[str, Any]


class MarigoldWorkerRunner(Protocol):
    def run(self, manifest_path: Path, output_dir: Path) -> MarigoldWorkerResult: ...


class SubprocessMarigoldWorkerRunner:
    def __init__(self, *, python_executable: Path, worker_script: Path, project_root: Path) -> None:
        self.python_executable = Path(python_executable).expanduser().resolve()
        self.worker_script = Path(worker_script).expanduser().resolve()
        self.project_root = Path(project_root).expanduser().resolve()

    def run(self, manifest_path: Path, output_dir: Path) -> MarigoldWorkerResult:
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
                "I4.3E6 Marigold worker failed: "
                f"exit={completed.returncode} stderr={completed.stderr.strip()[:2000]}"
            )
        result_path = output_dir / "marigold_result_manifest.json"
        if not result_path.is_file():
            raise ReconstructionValidationError("I4.3E6 worker did not produce its result manifest.")
        return MarigoldWorkerResult(result_path, _read_json(result_path))


class MarigoldProductionProviderAdapter:
    def __init__(
        self,
        *,
        context: MarigoldProviderRunContext,
        environment: MarigoldEnvironmentProbe,
        runner: MarigoldWorkerRunner | None = None,
        progress_sink: DenseProviderProgressSink | None = None,
    ) -> None:
        if not isinstance(context, MarigoldProviderRunContext):
            raise TypeError("context must be MarigoldProviderRunContext.")
        if not isinstance(environment, MarigoldEnvironmentProbe):
            raise TypeError("environment must be MarigoldEnvironmentProbe.")
        self.context = context
        self.environment = environment
        self._runner = runner or SubprocessMarigoldWorkerRunner(
            python_executable=context.python_executable,
            worker_script=context.worker_script,
            project_root=Path(environment.project_root),
        )
        self._progress_sink = progress_sink
        self._progress_event_count = 0

    @property
    def descriptor(self) -> DenseProviderDescriptor:
        return marigold_dense_provider_descriptor(self.environment)

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
                provider_id=I43E6_PROVIDER_ID,
                implementation=DenseProviderImplementation.MARIGOLD,
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
            raise ReconstructionValidationError("I4.3E6 Marigold is explicit-provider only; Auto remains unauthorized here.")
        if request.requested_provider_id != I43E6_PROVIDER_ID or decision.selected_provider_id != I43E6_PROVIDER_ID:
            raise ReconstructionValidationError("I4.3E6 concrete provider ID does not match Marigold.")
        if decision.selected_implementation is not DenseProviderImplementation.MARIGOLD:
            raise ReconstructionValidationError("I4.3E6 selected implementation must be Marigold.")
        if decision.fallback_used or request.allow_fallback:
            raise ReconstructionValidationError("I4.3E6 forbids silent provider fallback during explicit qualification.")
        if not request.authoritative_sfm_available or request.authoritative_sfm_frame_id != self.context.reconstruction_frame_id:
            raise ReconstructionValidationError("I4.3E6 refuses canonical binding without the requested authoritative SfM frame.")
        if request.available_view_count != self.context.sfm_result.statistics.registered_keyframes:
            raise ReconstructionValidationError("I4.3E6 available_view_count must match registered SfM views.")

    def execute(
        self,
        request: DenseProviderSelectionRequest,
        decision: DenseProviderSelectionDecision,
    ) -> DenseProviderExecutionResult:
        self._validate_selection(request, decision)
        context = self.context
        if not self.environment.available:
            raise ReconstructionValidationError(
                "I4.3E6 Marigold runtime is unavailable: "
                + (self.environment.error or "not_ready")
            )
        if not context.python_executable.is_file():
            raise ReconstructionValidationError(
                f"I4.3E6 Python executable is missing: {context.python_executable}"
            )
        if not context.worker_script.is_file():
            raise ReconstructionValidationError(
                f"I4.3E6 worker script is missing: {context.worker_script}"
            )

        runtime_payload = metadata_as_dict(context.runtime_config)  # type: ignore[arg-type]
        configured_checkpoint = Path(
            str(runtime_payload.get("checkpoint_dir", I43E6_DEFAULT_CHECKPOINT_DIR))
        ).expanduser()
        if not configured_checkpoint.is_absolute():
            configured_checkpoint = (Path(self.environment.project_root) / configured_checkpoint).resolve()
        else:
            configured_checkpoint = configured_checkpoint.resolve()
        if configured_checkpoint != Path(self.environment.checkpoint_path).resolve():
            raise ReconstructionValidationError(
                "I4.3E6 runtime checkpoint directory does not match the qualified Marigold snapshot."
            )
        runtime_payload["checkpoint_dir"] = str(configured_checkpoint)
        configured_repository = str(
            runtime_payload.get("repository", I43E6_CHECKPOINT_REPOSITORY)
        ).strip()
        if configured_repository != self.environment.checkpoint_repository:
            raise ReconstructionValidationError(
                "I4.3E6 runtime repository does not match the qualified Marigold snapshot."
            )
        runtime_payload["repository"] = configured_repository
        configured_profile = str(runtime_payload.get("profile", self.environment.profile)).strip().lower()
        if configured_profile != self.environment.profile:
            raise ReconstructionValidationError(
                "I4.3E6 runtime profile does not match the qualified hardware profile."
            )
        runtime_payload["profile"] = configured_profile

        root = Path(context.sfm_result.workspace.workspace_root).expanduser().resolve()
        images = tuple(sorted(context.sfm_result.workspace.images, key=lambda item: item.order))
        if len(images) != context.sfm_result.statistics.registered_keyframes:
            raise ReconstructionValidationError("I4.3E6 workspace/registered-view count drifted.")
        camera_by_keyframe = {item.keyframe_id: item for item in context.sfm_result.cameras}
        model_by_id = {item.camera_id: item for item in context.sfm_result.camera_models}
        camera_ids = {camera_by_keyframe[item.keyframe_id].camera_id for item in images}
        if len(camera_ids) != 1:
            raise ReconstructionValidationError(
                "I4.3E6 canonical fusion currently requires one authoritative SfM camera model per provider run."
            )
        camera_id = next(iter(camera_ids))
        camera_model = model_by_id.get(camera_id)
        if camera_model is None:
            raise ReconstructionValidationError("I4.3E6 authoritative SfM camera model is unavailable.")

        paths: list[Path] = []
        before_hashes: dict[str, str] = {}
        for item in images:
            path = (root / item.staged_image_path).resolve()
            if not path.is_file():
                raise ReconstructionValidationError(f"I4.3E6 staged image is missing: {path}")
            digest = _sha256_file(path)
            if digest != item.staged_sha256:
                raise ReconstructionValidationError(
                    f"I4.3E6 staged image checksum drifted before execution: {item.keyframe_id}"
                )
            paths.append(path)
            before_hashes[item.keyframe_id] = digest

        context.output_dir.mkdir(parents=True, exist_ok=True)
        provider_dir = context.output_dir / "provider"
        provider_dir.mkdir(parents=True, exist_ok=False)
        input_manifest = provider_dir / "input_manifest.json"
        _write_json(
            input_manifest,
            {
                "schema_version": I43E6_SCHEMA_VERSION,
                "phase": I43E6_PHASE,
                "request_id": f"i4-3e6-{context.sfm_result.session_id}",
                "session_id": context.sfm_result.session_id,
                "authoritative_sfm_frame_id": context.reconstruction_frame_id,
                "model_id": MARIGOLD_MODEL_ID,
                "runtime_distribution": self.environment.distribution_name,
                "runtime_distribution_version": self.environment.distribution_version,
                "profile": self.environment.profile,
                "checkpoint_repository": self.environment.checkpoint_repository,
                "checkpoint_requested_revision": self.environment.checkpoint_requested_revision,
                "checkpoint_resolved_revision": self.environment.checkpoint_resolved_revision,
                "checkpoint_path": self.environment.checkpoint_path,
                "checkpoint_marker_sha256": self.environment.checkpoint_marker_sha256,
                "model_index_sha256": self.environment.model_index_sha256,
                "checkpoint_size_bytes": self.environment.checkpoint_size_bytes,
                "device": context.device,
                "precision": context.precision,
                "runtime_config": runtime_payload,
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
                    "estimated_model_intrinsics_are_canonical_authority": False,
                    "adapter_projected_points_are_canonical_geometry": False,
                    "relative_depth_establishes_metric_scale": False,
                },
            },
        )

        authority_before = _sfm_authority_digest(context.sfm_result)
        total_views = len(images)
        self._progress_event_count = 0
        self._publish(
            stage=DenseProviderProgressStage.PREPARING,
            detail="validated_marigold_sfm_inputs",
            progress=0.02,
            processed=0,
            total=total_views,
        )
        self._publish(
            stage=DenseProviderProgressStage.VIEW_SELECTION,
            detail="all_registered_views_selected_i4_3e6",
            progress=0.05,
            processed=0,
            total=total_views,
        )
        self._publish(
            stage=DenseProviderProgressStage.PROVIDER_EXECUTION,
            detail="external_marigold_single_image_inference",
            progress=0.10,
            processed=0,
            total=total_views,
        )
        worker = self._runner.run(input_manifest, provider_dir / "worker_output")
        payload = dict(worker.metadata)
        if payload.get("status") != "pass" or payload.get("phase") != I43E6_PHASE:
            raise ReconstructionValidationError("I4.3E6 worker result did not report a passing I4.3E6 run.")
        if payload.get("model_id") != MARIGOLD_MODEL_ID:
            raise ReconstructionValidationError("I4.3E6 worker model identity drifted.")
        if str(payload.get("runtime_distribution_version", "")) != self.environment.distribution_version:
            raise ReconstructionValidationError("I4.3E6 worker Diffusers runtime version drifted.")
        if str(payload.get("checkpoint_path", "")) != self.environment.checkpoint_path:
            raise ReconstructionValidationError("I4.3E6 worker checkpoint path drifted.")
        if str(payload.get("checkpoint_resolved_revision", "")) != self.environment.checkpoint_resolved_revision:
            raise ReconstructionValidationError("I4.3E6 worker checkpoint revision drifted.")
        if str(payload.get("checkpoint_marker_sha256", "")) != self.environment.checkpoint_marker_sha256:
            raise ReconstructionValidationError("I4.3E6 worker checkpoint marker checksum drifted.")
        if str(payload.get("model_index_sha256", "")) != self.environment.model_index_sha256:
            raise ReconstructionValidationError("I4.3E6 worker model-index checksum drifted.")
        worker_authority = payload.get("authority")
        if not isinstance(worker_authority, Mapping):
            raise ReconstructionValidationError("I4.3E6 worker authority record is missing.")
        if any(
            worker_authority.get(key) is True
            for key in (
                "estimated_model_intrinsics_are_canonical_authority",
                "adapter_projected_points_are_canonical_geometry",
                "relative_depth_establishes_metric_scale",
            )
        ):
            raise ReconstructionValidationError("I4.3E6 worker attempted to promote Marigold evidence authority.")
        records = payload.get("frame_records")
        if not isinstance(records, list) or len(records) != total_views:
            raise ReconstructionValidationError("I4.3E6 worker must return one frame record per registered view.")
        self._publish(
            stage=DenseProviderProgressStage.PROVIDER_EXECUTION,
            detail="marigold_relative_depth_evidence_ready",
            progress=0.35,
            processed=total_views,
            total=total_views,
        )

        scene_scale = _robust_scene_diagonal(context.sfm_result)
        voxel_size = max(scene_scale * context.config.voxel_scene_fraction, 1e-8)
        reconstruction_camera = reconstruction_camera_from_sfm_model(
            camera_model,
            keyframe_id=images[0].keyframe_id,
            reconstruction_frame_id=context.reconstruction_frame_id,
        )
        camera_profile = camera_geometry_from_estimated_camera(
            profile_id=f"i43e6-{camera_id}-sfm-estimated",
            camera=reconstruction_camera,
            focal_length_relative_uncertainty=0.20,
            notes="I4.3E6 camera geometry from authoritative PyCOLMAP SfM; arbitrary gauge only.",
        )
        scale_profile = arbitrary_scale_reference(
            profile_id=f"i43e6-{context.sfm_result.session_id[:12]}-arbitrary-sfm-gauge",
            notes=(
                "I4.3E6 aligns Marigold affine-invariant relative evidence into the arbitrary "
                "SfM gauge; the alignment does not establish physical metric scale and model-estimated "
                "intrinsics are never canonical projection authority."
            ),
        )

        frames: list[DenseFusionFrame] = []
        alignment_records: list[dict[str, Any]] = []
        rejected: list[str] = []
        worker_dir = worker.manifest_path.parent
        for index, (item, record) in enumerate(zip(images, records, strict=True)):
            if not isinstance(record, Mapping):
                raise ReconstructionValidationError("I4.3E6 worker frame records must be objects.")
            if record.get("keyframe_id") != item.keyframe_id or int(record.get("order", -1)) != index:
                raise ReconstructionValidationError("I4.3E6 worker frame ordering/keyframe identity drifted.")
            if record.get("native_depth_scale") != "relative" or record.get("native_is_metric") is not False:
                raise ReconstructionValidationError("I4.3E6 worker changed Marigold relative-depth semantics.")
            if record.get("depth_representation_native") != "affine_invariant_relative_depth_or_disparity":
                raise ReconstructionValidationError("I4.3E6 worker changed Marigold affine-invariant representation semantics.")
            prediction_type = str(record.get("prediction_type", "")).strip().lower()
            if prediction_type not in {"depth", "disparity"}:
                raise ReconstructionValidationError("I4.3E6 worker prediction_type must be depth or disparity.")
            if record.get("intrinsics_source_native") != "estimated":
                raise ReconstructionValidationError("I4.3E6 worker changed Marigold estimated-intrinsics semantics.")
            confidence_kind = str(record.get("confidence_kind_native", "none"))
            if confidence_kind not in {"none", "uncertainty"}:
                raise ReconstructionValidationError("I4.3E6 worker changed Marigold uncertainty semantics.")
            uncertainty_available = bool(record.get("uncertainty_available", False))
            if uncertainty_available != (confidence_kind == "uncertainty"):
                raise ReconstructionValidationError("I4.3E6 worker uncertainty availability/kind drifted.")
            if record.get("direct_points_native") is not False or record.get("normals_native") is not False:
                raise ReconstructionValidationError("I4.3E6 worker fabricated unsupported native Marigold geometry evidence.")
            if record.get("adapter_projected_points_discarded") is not True:
                raise ReconstructionValidationError("I4.3E6 worker did not explicitly discard adapter-projected points.")
            if any(
                record.get(key) is True
                for key in (
                    "estimated_model_intrinsics_are_canonical_authority",
                    "adapter_projected_points_are_canonical_geometry",
                    "relative_depth_establishes_metric_scale",
                )
            ):
                raise ReconstructionValidationError("I4.3E6 worker frame attempted to promote Marigold evidence authority.")
            evidence_path = Path(str(record.get("evidence_npz_path", ""))).expanduser()
            if not evidence_path.is_absolute():
                evidence_path = (worker_dir / evidence_path).resolve()
            if not evidence_path.is_file():
                raise ReconstructionValidationError(f"I4.3E6 worker evidence file is missing: {evidence_path}")
            try:
                with np.load(evidence_path, allow_pickle=False) as archive:
                    relative_depth = np.asarray(archive["relative_depth"], dtype=np.float32)
                    valid_mask = np.asarray(archive["valid_mask"], dtype=bool)
                    estimated_intrinsics = np.asarray(archive["estimated_intrinsics"], dtype=np.float32)
                    uncertainty = (
                        np.asarray(archive["uncertainty"], dtype=np.float32)
                        if "uncertainty" in archive.files
                        else None
                    )
            except Exception as exc:  # noqa: BLE001
                raise ReconstructionValidationError(
                    f"I4.3E6 could not read worker evidence {evidence_path}: {exc}"
                ) from exc
            expected_hw = (item.image_size.height, item.image_size.width)
            if relative_depth.shape != expected_hw or valid_mask.shape != expected_hw:
                raise ReconstructionValidationError(
                    "I4.3E6 relative-depth/mask evidence must remain on each authoritative SfM pixel grid."
                )
            if not np.isfinite(relative_depth[valid_mask]).all() or np.any(relative_depth[valid_mask] <= 0.0):
                raise ReconstructionValidationError(
                    "I4.3E6 valid Marigold relative-depth evidence must be finite and positive."
                )
            if estimated_intrinsics.shape != (3, 3) or not np.isfinite(estimated_intrinsics).all():
                raise ReconstructionValidationError("I4.3E6 Marigold estimated-intrinsics evidence is invalid.")
            if uncertainty_available:
                if uncertainty is None or uncertainty.shape != expected_hw or not np.isfinite(uncertainty).all():
                    raise ReconstructionValidationError("I4.3E6 Marigold uncertainty evidence is missing or invalid.")
                if np.any(uncertainty < 0.0):
                    raise ReconstructionValidationError("I4.3E6 Marigold uncertainty must be non-negative.")
            elif uncertainty is not None:
                raise ReconstructionValidationError("I4.3E6 worker persisted undeclared Marigold uncertainty.")

            try:
                aligned_depth, aligned_valid, fit = align_marigold_relative_depth_to_sfm(
                    context.sfm_result,
                    keyframe_id=item.keyframe_id,
                    relative_depth=relative_depth,
                    prediction_type=prediction_type,
                    source_valid_mask=valid_mask,
                    minimum_correspondences=context.config.minimum_correspondences,
                    maximum_relative_residual=context.config.maximum_relative_residual,
                )
            except ReconstructionValidationError as exc:
                rejected.append(item.keyframe_id)
                alignment_records.append({
                    "keyframe_id": item.keyframe_id,
                    "status": "rejected",
                    "reason": str(exc),
                    "prediction_type": prediction_type,
                    "native_depth_scale": "relative",
                    "establishes_physical_metric_scale": False,
                    "estimated_model_intrinsics_consumed_for_canonical_projection": False,
                    "adapter_projected_points_consumed_for_canonical_geometry": False,
                })
            else:
                alignment_record = fit.as_dict()
                alignment_record.update({
                    "keyframe_id": item.keyframe_id,
                    "status": "accepted",
                    "prediction_type": prediction_type,
                    "native_depth_scale": "relative",
                    "estimated_model_intrinsics_consumed_for_canonical_projection": False,
                    "adapter_projected_points_consumed_for_canonical_geometry": False,
                    "native_uncertainty_available": uncertainty_available,
                    "uncertainty_semantics": "lower_is_better" if uncertainty_available else "none",
                })
                alignment_records.append(alignment_record)
                bgr = cv2.imread(str(paths[index]), cv2.IMREAD_COLOR)
                if bgr is None:
                    raise ReconstructionValidationError(f"I4.3E6 could not decode staged RGB: {paths[index]}")
                rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)
                fusion_confidence = None
                if uncertainty is not None:
                    fusion_confidence = np.ascontiguousarray(
                        1.0 / (1.0 + np.maximum(uncertainty, 0.0)), dtype=np.float32
                    )
                    fusion_confidence[~aligned_valid] = 0.0
                frames.append(
                    DenseFusionFrame(
                        frame_id=camera_by_keyframe[item.keyframe_id].frame_id,
                        timestamp_ns=item.timestamp_ns,
                        rgb=rgb,
                        depth=aligned_depth,
                        T_reconstruction_from_camera=camera_by_keyframe[item.keyframe_id].T_reconstruction_from_camera,
                        depth_unit=DenseDepthUnit.RECONSTRUCTION_UNITS,
                        depth_alignment=reconstruction_unit_depth_alignment(
                            provenance=(
                                f"I4.3E6 Marigold {fit.method};"
                                f"checkpoint_revision={self.environment.checkpoint_resolved_revision};"
                                f"diffusers={self.environment.distribution_version};"
                                "physical_metric_scale=false"
                            )
                        ),
                        confidence=fusion_confidence,
                        valid_mask=aligned_valid,
                        pose_layer="offline_sfm_bundle_adjusted",
                    )
                )
            self._publish(
                stage=DenseProviderProgressStage.ALIGNMENT,
                detail=f"align_marigold_{prediction_type}_to_authoritative_sfm",
                progress=0.40 + 0.35 * ((index + 1) / float(total_views)),
                processed=index + 1,
                total=total_views,
            )

        if len(frames) < context.config.minimum_accepted_views:
            raise ReconstructionValidationError(
                "I4.3E6 insufficient Marigold/SfM alignment-accepted views: "
                f"accepted={len(frames)} required={context.config.minimum_accepted_views}."
            )

        self._publish(
            stage=DenseProviderProgressStage.FUSION_FILTERING,
            detail="qualified_marigold_voxel_fusion",
            progress=0.82,
            processed=len(frames),
            total=total_views,
        )
        fusion = fuse_dense_frames(
            frames,
            session_id=f"{context.sfm_result.session_id}:i4.3e6-marigold",
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
        canonical_npz = context.output_dir / "canonical_marigold_fused.npz"
        np.savez_compressed(
            canonical_npz,
            points_reconstruction=cloud.points_reconstruction,
            colors_rgb=cloud.colors_rgb,
            view_observation_counts=cloud.view_observation_counts,
            mean_confidence=cloud.mean_confidence,
            voxel_size_reconstruction_units=np.asarray(
                cloud.voxel_size_reconstruction_units, dtype=np.float64
            ),
        )
        canonical_ply = _write_binary_ply(
            context.output_dir / "canonical_marigold_fused.ply",
            cloud.points_reconstruction,
            cloud.colors_rgb,
        )

        for item, path in zip(images, paths, strict=True):
            if _sha256_file(path) != before_hashes[item.keyframe_id]:
                raise ReconstructionValidationError(
                    "I4.3E6 registered staged images changed during provider execution."
                )
        if _sfm_authority_digest(context.sfm_result) != authority_before:
            raise ReconstructionValidationError(
                "I4.3E6 authoritative SfM state changed during provider execution."
            )

        degradation_reasons: list[str] = []
        if rejected:
            degradation_reasons.append(f"alignment_rejected_views:{len(rejected)}")
        status = DenseProviderRunStatus.DEGRADED if degradation_reasons else DenseProviderRunStatus.COMPLETED
        self._publish(
            stage=DenseProviderProgressStage.FINALIZATION,
            detail="persist_marigold_canonical_candidate",
            progress=1.0,
            processed=len(frames),
            total=total_views,
        )

        result = DenseProviderExecutionResult(
            provider_id=I43E6_PROVIDER_ID,
            implementation=DenseProviderImplementation.MARIGOLD,
            status=status,
            source_view_count=total_views,
            selected_view_count=total_views,
            adaptive_view_selection=False,
            view_selection_reason="all_registered_views_i4_3e6_adaptive_selection_deferred_i4_3f",
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
            context.output_dir / "i4_3e6_marigold_execution_summary.json",
            {
                "schema_version": I43E6_SCHEMA_VERSION,
                "phase": I43E6_PHASE,
                "status": "pass",
                "provider": self.descriptor.as_dict(),
                "execution": result.as_dict(),
                "worker": payload,
                "config": context.config.as_dict(),
                "alignment": {
                    "accepted_views": len(frames),
                    "rejected_views": rejected,
                    "records": alignment_records,
                    "method": "affine_domain_according_to_prediction_type_to_authoritative_sfm",
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
                    "relative_depth_promoted_to_metric_scale": False,
                    "estimated_model_intrinsics_consumed_for_canonical_projection": False,
                    "adapter_projected_points_consumed_for_canonical_geometry": False,
                    "metric_scale_claimed": False,
                    "measurement_authoritative": False,
                    "certified_metrology": False,
                    "surface_mesh_generated": False,
                    "provider_promoted": False,
                },
            },
        )
        return result


def validate_i4_3e6_binding(
    *,
    i4_3a_snapshot: Mapping[str, Any],
    i4_3d3_snapshot: Mapping[str, Any],
    i4_3e1_snapshot: Mapping[str, Any],
    i4_3e2_snapshot: Mapping[str, Any],
    i4_3e3_snapshot: Mapping[str, Any],
    i4_3e4_snapshot: Mapping[str, Any],
    i4_3e5_snapshot: Mapping[str, Any],
    environment: MarigoldEnvironmentProbe,
) -> dict[str, Any]:
    snapshots = (
        i4_3a_snapshot,
        i4_3d3_snapshot,
        i4_3e1_snapshot,
        i4_3e2_snapshot,
        i4_3e3_snapshot,
        i4_3e4_snapshot,
        i4_3e5_snapshot,
    )
    if not all(isinstance(item, Mapping) for item in snapshots):
        raise TypeError("I4.3E6 binding snapshots must be mappings.")
    if (
        i4_3a_snapshot.get("phase") != "I4.3A"
        or i4_3a_snapshot.get("status") != "pass"
        or i4_3a_snapshot.get("pass") is not True
    ):
        raise ReconstructionValidationError("I4.3E6 requires the passing I4.3A architecture snapshot.")
    architecture = i4_3a_snapshot.get("architecture")
    if not isinstance(architecture, Mapping):
        raise ReconstructionValidationError("I4.3E6 I4.3A snapshot is missing architecture.")
    if architecture.get("authoritative_camera_frame_owner") != I43A_AUTHORITATIVE_FRAME_OWNER:
        raise ReconstructionValidationError("I4.3E6 I4.3A camera-frame authority drifted.")
    contracts = architecture.get("provider_implementation_contracts")
    if not isinstance(contracts, list):
        raise ReconstructionValidationError("I4.3E6 I4.3A implementation contracts are missing.")
    match = next(
        (
            item
            for item in contracts
            if isinstance(item, Mapping) and item.get("implementation") == "marigold"
        ),
        None,
    )
    if not isinstance(match, Mapping):
        raise ReconstructionValidationError("I4.3E6 Marigold implementation contract is missing.")
    if match.get("integration_stage") != "I4.3E" or match.get("family") != "single_image_depth":
        raise ReconstructionValidationError("I4.3E6 Marigold implementation contract drifted.")
    if match.get("frame_binding") != "align_to_authoritative_sfm" or match.get("native_output") != "depth_evidence":
        raise ReconstructionValidationError("I4.3E6 single-image depth frame/output contract drifted.")

    if (
        i4_3d3_snapshot.get("phase") != "I4.3D3"
        or i4_3d3_snapshot.get("status") != "pass"
        or i4_3d3_snapshot.get("pass") is not True
    ):
        raise ReconstructionValidationError("I4.3E6 requires the passing I4.3D3 temporal-provider closure snapshot.")
    decision = i4_3d3_snapshot.get("decision")
    authority = i4_3d3_snapshot.get("authority")
    if not isinstance(decision, Mapping) or not isinstance(authority, Mapping):
        raise ReconstructionValidationError("I4.3E6 I4.3D3 closure snapshot is incomplete.")
    if decision.get("i4_3d_temporal_provider_architecture_closed") is not True or decision.get("i4_3e_authorized") is not True:
        raise ReconstructionValidationError("I4.3E6 is not authorized by the I4.3D3 closure decision.")
    if any(
        authority.get(key) is True
        for key in ("metric_scale_claimed", "measurement_authoritative", "certified_metrology", "surface_mesh_generated")
    ):
        raise ReconstructionValidationError("I4.3E6 refuses an I4.3D3 snapshot with promoted authority claims.")

    for phase, implementation, snapshot in (
        ("I4.3E1", "metric3d_v2", i4_3e1_snapshot),
        ("I4.3E2", "unidepth_v2", i4_3e2_snapshot),
        ("I4.3E3", "moge_2", i4_3e3_snapshot),
        ("I4.3E4", "depth_pro", i4_3e4_snapshot),
        ("I4.3E5", "depth_anything_v2", i4_3e5_snapshot),
    ):
        if snapshot.get("phase") != phase or snapshot.get("status") != "pass" or snapshot.get("pass") is not True:
            raise ReconstructionValidationError(f"I4.3E6 requires the passing {phase} predecessor binding snapshot.")
        provider = snapshot.get("provider")
        predecessor_authority = snapshot.get("authority")
        if not isinstance(provider, Mapping) or not isinstance(predecessor_authority, Mapping):
            raise ReconstructionValidationError(f"I4.3E6 {phase} predecessor snapshot is incomplete.")
        if provider.get("implementation") != implementation or provider.get("qualification_state") != "candidate":
            raise ReconstructionValidationError(f"I4.3E6 requires the closed candidate {phase} provider state.")
        if predecessor_authority.get("provider_promoted") is True or any(
            predecessor_authority.get(key) is True
            for key in ("metric_scale_claimed", "measurement_authoritative", "certified_metrology", "surface_mesh_generated")
        ):
            raise ReconstructionValidationError(f"I4.3E6 refuses a {phase} predecessor with promoted authority claims.")

    descriptor = marigold_dense_provider_descriptor(environment)
    if not descriptor.available:
        raise ReconstructionValidationError("I4.3E6 Marigold runtime is not ready for binding.")
    return {
        "schema_version": I43E6_SCHEMA_VERSION,
        "phase": I43E6_PHASE,
        "status": "pass",
        "pass": True,
        "provider": descriptor.as_dict(),
        "architecture": {
            "family": "single_image_depth",
            "implementation": "marigold",
            "frame_binding": "align_to_authoritative_sfm",
            "native_output": "depth_evidence",
            "native_depth_scale": "relative",
            "native_depth_representation": "affine_invariant_relative_depth_or_disparity",
            "native_intrinsics_source": "estimated",
            "native_confidence_kind": "uncertainty_optional",
            "native_direct_points_available": False,
            "native_normals_available": False,
            "native_valid_mask_available": True,
            "adapter_projected_points_discarded": True,
            "per_view_alignment": "affine_domain_according_to_prediction_type_to_authoritative_sfm",
            "all_registered_views_available_by_default": True,
            "permanent_global_view_cap": None,
            "adaptive_view_selection_deferred_to": "I4.3F",
        },
        "authority": {
            "camera_pose_authority_owner": I43A_AUTHORITATIVE_FRAME_OWNER,
            "relative_depth_promoted_to_metric_scale": False,
            "estimated_model_intrinsics_consumed_for_canonical_projection": False,
            "adapter_projected_points_consumed_for_canonical_geometry": False,
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
            "predecessor_i4_3e1_verified": True,
            "predecessor_i4_3e2_verified": True,
            "predecessor_i4_3e3_verified": True,
            "predecessor_i4_3e4_verified": True,
            "predecessor_i4_3e5_verified": True,
            "remaining_i4_3e_providers_pending": [],
            "i4_3e_provider_adapters_complete": True,
        },
        "environment": environment.as_dict(),
        "next": "I4.3F adaptive frame/view selection.",
    }


__all__ = [
    "I43E6_ADAPTER_VERSION",
    "I43E6_PHASE",
    "I43E6_PROVIDER_ID",
    "MarigoldEnvironmentProbe",
    "MarigoldAffineAlignmentFit",
    "MarigoldProductionConfig",
    "MarigoldProductionProviderAdapter",
    "MarigoldProviderRunContext",
    "MarigoldWorkerResult",
    "SubprocessMarigoldWorkerRunner",
    "align_marigold_relative_depth_to_sfm",
    "marigold_dense_provider_descriptor",
    "probe_marigold_environment",
    "validate_i4_3e6_binding",
]

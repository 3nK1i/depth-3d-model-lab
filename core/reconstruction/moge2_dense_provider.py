"""I4.3E3 canonical production adapter for MoGe-2.

MoGe-2 already exists as a mature image-depth model adapter in the
application. I4.3E3 binds that existing runtime to the I4.3A offline dense
provider contract without changing its interactive behavior.

Each registered SfM view is inferred independently in a worker process. Native
MoGe-2 metric depth, model-predicted intrinsics, direct points, normals, and
validity mask are retained as model evidence only. MoGe-2 exposes no native
confidence map, so I4.3E3 does not synthesize one. Canonical projection always uses the existing
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
import importlib.metadata as importlib_metadata
import inspect
import os
import re
import subprocess
import sys
from typing import Any, Mapping, Protocol

import cv2
import numpy as np

from core.contracts.validation import JsonValue, metadata_as_dict, normalize_metadata, require_nonempty
from core.models.catalog import MOGE2_MODEL_ID
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

I43E3_SCHEMA_VERSION = 1
I43E3_PHASE = "I4.3E3"
I43E3_ADAPTER_VERSION = "i4.3e3-v1"
I43E3_PROVIDER_ID = "moge-2-single-image"
I43E3_OFFICIAL_SOURCE_REPOSITORY = "https://github.com/microsoft/MoGe.git"
I43E3_DEFAULT_MINIMUM_CORRESPONDENCES = 8
I43E3_DEFAULT_MAXIMUM_RELATIVE_RESIDUAL = 0.35
I43E3_DEFAULT_VOXEL_SCENE_FRACTION = 0.004
I43E3_DEFAULT_PIXEL_STRIDE = 2
I43E3_DEFAULT_MAX_VOXELS = 1_500_000
I43E3_DEFAULT_MINIMUM_ACCEPTED_VIEWS = 2


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
        raise ReconstructionValidationError(f"I4.3E3 could not read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"I4.3E3 JSON root must be an object: {path}")
    return payload


def _robust_scene_diagonal(sfm_result: OfflineSfmResult) -> float:
    if len(sfm_result.landmarks) < 4:
        raise ReconstructionValidationError("I4.3E3 requires sparse SfM landmarks for scene-relative fusion scale.")
    points = np.vstack([np.asarray(item.position_reconstruction, dtype=np.float64) for item in sfm_result.landmarks])
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ReconstructionValidationError("I4.3E3 sparse SfM landmarks are invalid.")
    low = np.quantile(points, 0.02, axis=0)
    high = np.quantile(points, 0.98, axis=0)
    diagonal = float(np.linalg.norm(high - low))
    if not isfinite(diagonal) or diagonal <= 0.0:
        raise ReconstructionValidationError("I4.3E3 robust sparse scene diagonal is invalid.")
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
        raise ReconstructionValidationError("I4.3E3 PLY geometry must be aligned Nx3 XYZ/RGB arrays.")
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
class MoGe2EnvironmentProbe:
    python_executable: str
    project_root: str
    distribution_name: str
    distribution_version: str
    distribution_root: str
    source_repository: str
    source_revision: str
    source_revision_pinned: bool
    official_runtime_importable: bool
    official_runtime_source_matches_distribution: bool
    variant: str
    checkpoint_repository: str
    checkpoint_cache_root: str
    checkpoint_revision: str
    checkpoint_ready: bool
    adapter_importable: bool
    torch_available: bool
    cuda_available: bool
    device_names: tuple[str, ...] = ()
    error: str = ""

    @property
    def available(self) -> bool:
        return bool(
            self.source_revision_pinned
            and self.official_runtime_importable
            and self.official_runtime_source_matches_distribution
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
            "source_repository": self.source_repository,
            "source_revision": self.source_revision,
            "source_revision_pinned": self.source_revision_pinned,
            "official_runtime_importable": self.official_runtime_importable,
            "official_runtime_source_matches_distribution": self.official_runtime_source_matches_distribution,
            "variant": self.variant,
            "checkpoint_repository": self.checkpoint_repository,
            "checkpoint_cache_root": self.checkpoint_cache_root,
            "checkpoint_revision": self.checkpoint_revision,
            "checkpoint_ready": self.checkpoint_ready,
            "adapter_importable": self.adapter_importable,
            "torch_available": self.torch_available,
            "cuda_available": self.cuda_available,
            "device_names": list(self.device_names),
            "available": self.available,
            "error": self.error,
        }


_MOGE_VARIANTS: dict[str, str] = {
    "small": "Ruicheng/moge-2-vits-normal",
    "base": "Ruicheng/moge-2-vitb-normal",
    "large": "Ruicheng/moge-2-vitl-normal",
}


def _select_moge_variant(*, cuda_available: bool, total_vram_gib: float | None) -> str:
    if not cuda_available or total_vram_gib is None:
        return "small"
    if total_vram_gib >= 20.0:
        return "large"
    if total_vram_gib >= 8.0:
        return "base"
    return "small"


def _huggingface_hub_root() -> Path:
    explicit = os.environ.get("HF_HUB_CACHE", "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    hf_home = os.environ.get("HF_HOME", "").strip()
    if hf_home:
        return (Path(hf_home).expanduser() / "hub").resolve()
    return (Path.home() / ".cache" / "huggingface" / "hub").resolve()


def _probe_hf_checkpoint(repository: str) -> tuple[Path, str, bool]:
    repo_dir = _huggingface_hub_root() / f"models--{repository.replace('/', '--')}"
    revision = ""
    refs_main = repo_dir / "refs" / "main"
    if refs_main.is_file():
        try:
            revision = refs_main.read_text(encoding="utf-8").strip()
        except OSError:
            revision = ""
    snapshot = repo_dir / "snapshots" / revision if revision else Path()
    ready = bool(
        revision
        and re.fullmatch(r"[0-9a-fA-F]{40,64}", revision)
        and snapshot.is_dir()
        and any(path.is_file() for path in snapshot.rglob("*"))
    )
    return repo_dir, revision, ready


def probe_moge_2_environment(
    *,
    project_root: Path | None = None,
    python_executable: Path | None = None,
    variant: str = "auto",
) -> MoGe2EnvironmentProbe:
    root = Path(project_root or Path.cwd()).expanduser().resolve()
    python = Path(python_executable or sys.executable).expanduser().resolve()
    errors: list[str] = []

    adapter_importable = False
    official_runtime_importable = False
    runtime_matches_distribution = False
    distribution_name = "moge"
    distribution_version = ""
    distribution_root = ""
    source_repository = ""
    source_revision = ""
    source_revision_pinned = False

    try:
        from core.models.adapters.moge2_adapter import Moge2GeometryModel
        adapter_importable = Moge2GeometryModel is not None
    except Exception as exc:  # noqa: BLE001
        errors.append(f"adapter_import_failed:{exc}")

    distribution = None
    try:
        distribution = importlib_metadata.distribution("moge")
        distribution_name = str(distribution.metadata.get("Name") or "moge")
        distribution_version = str(distribution.version or "")
        distribution_root_path = Path(distribution.locate_file("moge")).expanduser().resolve()
        distribution_root = str(distribution_root_path)
        direct_url_raw = distribution.read_text("direct_url.json")
        direct_url = json.loads(direct_url_raw) if direct_url_raw else {}
        if not isinstance(direct_url, dict):
            direct_url = {}
        source_repository = str(direct_url.get("url") or "").strip()
        vcs_info = direct_url.get("vcs_info")
        if isinstance(vcs_info, Mapping) and str(vcs_info.get("vcs") or "").lower() == "git":
            source_revision = str(vcs_info.get("commit_id") or "").strip()
        repository_norm = source_repository.removeprefix("git+").rstrip("/").removesuffix(".git").lower()
        expected_norm = I43E3_OFFICIAL_SOURCE_REPOSITORY.rstrip("/").removesuffix(".git").lower()
        source_revision_pinned = bool(
            repository_norm == expected_norm
            and re.fullmatch(r"[0-9a-fA-F]{40}", source_revision)
        )
        if not source_revision_pinned:
            errors.append(
                "MoGe-2 installed distribution does not expose an exact official microsoft/MoGe VCS commit."
            )
    except Exception as exc:  # noqa: BLE001
        errors.append(f"distribution_probe_failed:{exc}")
        distribution_root_path = None

    try:
        from moge.model.v2 import MoGeModel as OfficialMoGeModel
        official_runtime_importable = OfficialMoGeModel is not None
        class_file = Path(inspect.getfile(OfficialMoGeModel)).expanduser().resolve()
        if distribution_root_path is not None:
            runtime_matches_distribution = (
                class_file == distribution_root_path or distribution_root_path in class_file.parents
            )
        if not runtime_matches_distribution:
            errors.append(
                "MoGe-2 imported runtime does not resolve to the installed moge distribution: "
                f"{class_file}"
            )
    except Exception as exc:  # noqa: BLE001
        errors.append(f"official_runtime_import_failed:{exc}")

    torch_available = False
    cuda_available = False
    device_names: tuple[str, ...] = ()
    total_vram_gib: float | None = None
    try:
        import torch
        torch_available = True
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            device_names = tuple(
                str(torch.cuda.get_device_name(index)) for index in range(torch.cuda.device_count())
            )
            total_vram_gib = float(torch.cuda.get_device_properties(0).total_memory) / 1024**3
    except Exception as exc:  # noqa: BLE001
        errors.append(f"torch_import_failed:{exc}")

    selected = str(variant).strip().lower()
    if selected == "auto":
        selected = _select_moge_variant(cuda_available=cuda_available, total_vram_gib=total_vram_gib)
    if selected not in _MOGE_VARIANTS:
        raise ReconstructionValidationError(f"I4.3E3 unsupported MoGe-2 variant: {variant!r}.")
    checkpoint_repository = _MOGE_VARIANTS[selected]
    checkpoint_cache_root, checkpoint_revision, checkpoint_ready = _probe_hf_checkpoint(checkpoint_repository)
    if not checkpoint_ready:
        errors.append(
            f"MoGe-2 {selected} checkpoint is not cached with a resolved refs/main snapshot: {checkpoint_cache_root}"
        )
    if not python.is_file():
        errors.append(f"Python executable missing: {python}")

    return MoGe2EnvironmentProbe(
        python_executable=str(python),
        project_root=str(root),
        distribution_name=distribution_name,
        distribution_version=distribution_version,
        distribution_root=distribution_root,
        source_repository=source_repository,
        source_revision=source_revision,
        source_revision_pinned=source_revision_pinned,
        official_runtime_importable=official_runtime_importable,
        official_runtime_source_matches_distribution=runtime_matches_distribution,
        variant=selected,
        checkpoint_repository=checkpoint_repository,
        checkpoint_cache_root=str(checkpoint_cache_root),
        checkpoint_revision=checkpoint_revision,
        checkpoint_ready=checkpoint_ready,
        adapter_importable=adapter_importable,
        torch_available=torch_available,
        cuda_available=cuda_available,
        device_names=device_names,
        error="; ".join(errors),
    )

def moge_2_dense_provider_descriptor(probe: MoGe2EnvironmentProbe) -> DenseProviderDescriptor:
    if not isinstance(probe, MoGe2EnvironmentProbe):
        raise TypeError("probe must be MoGe2EnvironmentProbe.")
    return DenseProviderDescriptor(
        provider_id=I43E3_PROVIDER_ID,
        implementation=DenseProviderImplementation.MOGE_2,
        execution=ReconstructionProviderExecution.EXTERNAL_PROCESS,
        backend_version=probe.source_revision or "unresolved",
        adapter_version=I43E3_ADAPTER_VERSION,
        qualification_state=DenseProviderQualificationState.CANDIDATE,
        available=probe.available,
        unavailable_reason="" if probe.available else (probe.error or "moge_2_runtime_unavailable"),
        metadata={
            "native_output": "depth_evidence",
            "frame_binding": "align_to_authoritative_sfm",
            "native_depth_scale": "metric",
            "native_intrinsics_source": "model_predicted",
            "native_confidence_kind": "none",
            "native_direct_points_available": True,
            "native_normals_available": True,
            "native_valid_mask_available": True,
            "single_image_stateful_sequence": False,
            "all_registered_views_available_by_default": True,
            "adaptive_view_selection_deferred_to": "I4.3F",
            "permanent_global_view_cap": None,
            "model_metric_depth_promoted_to_canonical_metric_scale": False,
            "model_predicted_intrinsics_consumed_for_canonical_projection": False,
            "model_direct_points_consumed_for_canonical_geometry": False,
            "model_normals_consumed_for_canonical_geometry": False,
            "checkpoint_repository": probe.checkpoint_repository,
            "checkpoint_revision": probe.checkpoint_revision,
            "license_status": "review_required",
            "camera_pose_authority_owner": I43A_AUTHORITATIVE_FRAME_OWNER,
        },
    )


@dataclass(frozen=True, slots=True)
class MoGe2ProductionConfig:
    minimum_correspondences: int = I43E3_DEFAULT_MINIMUM_CORRESPONDENCES
    maximum_relative_residual: float = I43E3_DEFAULT_MAXIMUM_RELATIVE_RESIDUAL
    voxel_scene_fraction: float = I43E3_DEFAULT_VOXEL_SCENE_FRACTION
    pixel_stride: int = I43E3_DEFAULT_PIXEL_STRIDE
    max_voxels: int = I43E3_DEFAULT_MAX_VOXELS
    minimum_accepted_views: int = I43E3_DEFAULT_MINIMUM_ACCEPTED_VIEWS

    def __post_init__(self) -> None:
        if int(self.minimum_correspondences) < 3:
            raise ReconstructionValidationError("I4.3E3 minimum_correspondences must be at least 3.")
        residual = float(self.maximum_relative_residual)
        if not isfinite(residual) or not 0.0 < residual < 1.0:
            raise ReconstructionValidationError("I4.3E3 maximum_relative_residual must be finite in (0,1).")
        fraction = float(self.voxel_scene_fraction)
        if not isfinite(fraction) or not 0.0 < fraction < 0.1:
            raise ReconstructionValidationError("I4.3E3 voxel_scene_fraction must be finite in (0,0.1).")
        if int(self.pixel_stride) <= 0 or int(self.max_voxels) <= 0:
            raise ReconstructionValidationError("I4.3E3 pixel_stride/max_voxels must be positive.")
        if int(self.minimum_accepted_views) < 2:
            raise ReconstructionValidationError("I4.3E3 minimum_accepted_views must be at least 2.")
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
class MoGe2ProviderRunContext:
    sfm_result: OfflineSfmResult
    reconstruction_frame_id: str
    output_dir: Path
    python_executable: Path
    worker_script: Path
    runtime_config: Mapping[str, JsonValue]
    device: str = "cuda:0"
    precision: str = "fp16"
    config: MoGe2ProductionConfig = MoGe2ProductionConfig()

    def __post_init__(self) -> None:
        if not isinstance(self.sfm_result, OfflineSfmResult):
            raise TypeError("sfm_result must be OfflineSfmResult.")
        if self.sfm_result.backend_id != PYCOLMAP_BACKEND_ID:
            raise ReconstructionValidationError("I4.3E3 requires the authoritative PyCOLMAP offline SfM result.")
        if self.sfm_result.statistics.registered_keyframes < 2:
            raise ReconstructionValidationError("I4.3E3 requires at least two registered SfM views.")
        if not self.sfm_result.camera_models:
            raise ReconstructionValidationError("I4.3E3 requires SfM camera-model estimates.")
        frame_id = require_nonempty(self.reconstruction_frame_id, field_name="reconstruction_frame_id")
        object.__setattr__(self, "reconstruction_frame_id", frame_id)
        for name in ("output_dir", "python_executable", "worker_script"):
            object.__setattr__(self, name, Path(getattr(self, name)).expanduser().resolve())
        object.__setattr__(self, "runtime_config", normalize_metadata(self.runtime_config))
        if not isinstance(self.config, MoGe2ProductionConfig):
            raise TypeError("config must be MoGe2ProductionConfig.")
        device = require_nonempty(self.device, field_name="device")
        precision = str(self.precision).strip().lower()
        if precision not in {"fp32", "fp16"}:
            raise ReconstructionValidationError("I4.3E3 precision must be fp32 or fp16.")
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "precision", precision)


@dataclass(frozen=True, slots=True)
class MoGe2WorkerResult:
    manifest_path: Path
    metadata: Mapping[str, Any]


class MoGe2WorkerRunner(Protocol):
    def run(self, manifest_path: Path, output_dir: Path) -> MoGe2WorkerResult: ...


class SubprocessMoGe2WorkerRunner:
    def __init__(self, *, python_executable: Path, worker_script: Path, project_root: Path) -> None:
        self.python_executable = Path(python_executable).expanduser().resolve()
        self.worker_script = Path(worker_script).expanduser().resolve()
        self.project_root = Path(project_root).expanduser().resolve()

    def run(self, manifest_path: Path, output_dir: Path) -> MoGe2WorkerResult:
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
                "I4.3E3 MoGe-2 worker failed: "
                f"exit={completed.returncode} stderr={completed.stderr.strip()[:2000]}"
            )
        result_path = output_dir / "moge2_result_manifest.json"
        if not result_path.is_file():
            raise ReconstructionValidationError("I4.3E3 worker did not produce its result manifest.")
        return MoGe2WorkerResult(result_path, _read_json(result_path))


class MoGe2ProductionProviderAdapter:
    def __init__(
        self,
        *,
        context: MoGe2ProviderRunContext,
        environment: MoGe2EnvironmentProbe,
        runner: MoGe2WorkerRunner | None = None,
        progress_sink: DenseProviderProgressSink | None = None,
    ) -> None:
        if not isinstance(context, MoGe2ProviderRunContext):
            raise TypeError("context must be MoGe2ProviderRunContext.")
        if not isinstance(environment, MoGe2EnvironmentProbe):
            raise TypeError("environment must be MoGe2EnvironmentProbe.")
        self.context = context
        self.environment = environment
        self._runner = runner or SubprocessMoGe2WorkerRunner(
            python_executable=context.python_executable,
            worker_script=context.worker_script,
            project_root=Path(environment.project_root),
        )
        self._progress_sink = progress_sink
        self._progress_event_count = 0

    @property
    def descriptor(self) -> DenseProviderDescriptor:
        return moge_2_dense_provider_descriptor(self.environment)

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
                provider_id=I43E3_PROVIDER_ID,
                implementation=DenseProviderImplementation.MOGE_2,
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
            raise ReconstructionValidationError("I4.3E3 MoGe-2 is explicit-provider only; Auto remains unauthorized here.")
        if request.requested_provider_id != I43E3_PROVIDER_ID or decision.selected_provider_id != I43E3_PROVIDER_ID:
            raise ReconstructionValidationError("I4.3E3 concrete provider ID does not match MoGe-2.")
        if decision.selected_implementation is not DenseProviderImplementation.MOGE_2:
            raise ReconstructionValidationError("I4.3E3 selected implementation must be MoGe-2.")
        if decision.fallback_used or request.allow_fallback:
            raise ReconstructionValidationError("I4.3E3 forbids silent provider fallback during explicit qualification.")
        if not request.authoritative_sfm_available or request.authoritative_sfm_frame_id != self.context.reconstruction_frame_id:
            raise ReconstructionValidationError("I4.3E3 refuses canonical binding without the requested authoritative SfM frame.")
        if request.available_view_count != self.context.sfm_result.statistics.registered_keyframes:
            raise ReconstructionValidationError("I4.3E3 available_view_count must match registered SfM views.")

    def execute(
        self,
        request: DenseProviderSelectionRequest,
        decision: DenseProviderSelectionDecision,
    ) -> DenseProviderExecutionResult:
        self._validate_selection(request, decision)
        context = self.context
        if not self.environment.available:
            raise ReconstructionValidationError(
                "I4.3E3 MoGe-2 runtime is unavailable: " + (self.environment.error or "not_ready")
            )
        if not context.python_executable.is_file():
            raise ReconstructionValidationError(f"I4.3E3 Python executable is missing: {context.python_executable}")
        if not context.worker_script.is_file():
            raise ReconstructionValidationError(f"I4.3E3 worker script is missing: {context.worker_script}")

        runtime_payload = metadata_as_dict(context.runtime_config)  # type: ignore[arg-type]
        configured_repository = str(runtime_payload.get("pretrained", "")).strip()
        if configured_repository != self.environment.checkpoint_repository:
            raise ReconstructionValidationError(
                "I4.3E3 runtime pretrained repository does not match the qualified MoGe-2 variant."
            )
        # Freeze this offline provider run to the checkpoint already qualified in
        # the local Hugging Face cache; never allow a network-side main update to
        # change evidence during execution.
        runtime_payload["local_files_only"] = True

        root = Path(context.sfm_result.workspace.workspace_root).expanduser().resolve()
        images = tuple(sorted(context.sfm_result.workspace.images, key=lambda item: item.order))
        if len(images) != context.sfm_result.statistics.registered_keyframes:
            raise ReconstructionValidationError("I4.3E3 workspace/registered-view count drifted.")
        camera_by_keyframe = {item.keyframe_id: item for item in context.sfm_result.cameras}
        model_by_id = {item.camera_id: item for item in context.sfm_result.camera_models}
        camera_ids = {camera_by_keyframe[item.keyframe_id].camera_id for item in images}
        if len(camera_ids) != 1:
            raise ReconstructionValidationError(
                "I4.3E3 canonical fusion currently requires one authoritative SfM camera model per provider run."
            )
        camera_id = next(iter(camera_ids))
        camera_model = model_by_id.get(camera_id)
        if camera_model is None:
            raise ReconstructionValidationError("I4.3E3 authoritative SfM camera model is unavailable.")

        paths: list[Path] = []
        before_hashes: dict[str, str] = {}
        for item in images:
            path = (root / item.staged_image_path).resolve()
            if not path.is_file():
                raise ReconstructionValidationError(f"I4.3E3 staged image is missing: {path}")
            digest = _sha256_file(path)
            if digest != item.staged_sha256:
                raise ReconstructionValidationError(f"I4.3E3 staged image checksum drifted before execution: {item.keyframe_id}")
            paths.append(path)
            before_hashes[item.keyframe_id] = digest

        context.output_dir.mkdir(parents=True, exist_ok=True)
        provider_dir = context.output_dir / "provider"
        provider_dir.mkdir(parents=True, exist_ok=False)
        input_manifest = provider_dir / "input_manifest.json"
        _write_json(
            input_manifest,
            {
                "schema_version": I43E3_SCHEMA_VERSION,
                "phase": I43E3_PHASE,
                "request_id": f"i4-3e3-{context.sfm_result.session_id}",
                "session_id": context.sfm_result.session_id,
                "authoritative_sfm_frame_id": context.reconstruction_frame_id,
                "model_id": MOGE2_MODEL_ID,
                "source_revision": self.environment.source_revision,
                "checkpoint_repository": self.environment.checkpoint_repository,
                "checkpoint_revision": self.environment.checkpoint_revision,
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
                    "model_predicted_intrinsics_are_canonical_authority": False,
                    "model_direct_points_are_canonical_geometry": False,
                    "model_normals_are_canonical_geometry": False,
                    "model_metric_depth_establishes_metric_scale": False,
                },
            },
        )

        authority_before = _sfm_authority_digest(context.sfm_result)
        total_views = len(images)
        self._progress_event_count = 0
        self._publish(stage=DenseProviderProgressStage.PREPARING, detail="validated_moge2_sfm_inputs", progress=0.02, processed=0, total=total_views)
        self._publish(stage=DenseProviderProgressStage.VIEW_SELECTION, detail="all_registered_views_selected_i4_3e3", progress=0.05, processed=0, total=total_views)
        self._publish(stage=DenseProviderProgressStage.PROVIDER_EXECUTION, detail="external_moge2_single_image_inference", progress=0.10, processed=0, total=total_views)
        worker = self._runner.run(input_manifest, provider_dir / "worker_output")
        payload = dict(worker.metadata)
        if payload.get("status") != "pass" or payload.get("phase") != I43E3_PHASE:
            raise ReconstructionValidationError("I4.3E3 worker result did not report a passing I4.3E3 run.")
        if payload.get("model_id") != MOGE2_MODEL_ID:
            raise ReconstructionValidationError("I4.3E3 worker model identity drifted.")
        if str(payload.get("source_revision", "")) != self.environment.source_revision:
            raise ReconstructionValidationError("I4.3E3 worker source revision does not match the probed runtime.")
        if str(payload.get("checkpoint_repository", "")) != self.environment.checkpoint_repository:
            raise ReconstructionValidationError("I4.3E3 worker checkpoint repository drifted.")
        if str(payload.get("checkpoint_revision", "")) != self.environment.checkpoint_revision:
            raise ReconstructionValidationError("I4.3E3 worker checkpoint revision drifted.")
        worker_authority = payload.get("authority")
        if not isinstance(worker_authority, Mapping):
            raise ReconstructionValidationError("I4.3E3 worker authority record is missing.")
        if any(
            worker_authority.get(key) is True
            for key in (
                "model_intrinsics_are_canonical_authority",
                "model_direct_points_are_canonical_geometry",
                "model_normals_are_canonical_geometry",
                "model_metric_depth_establishes_metric_scale",
            )
        ):
            raise ReconstructionValidationError("I4.3E3 worker attempted to promote native MoGe-2 authority.")
        records = payload.get("frame_records")
        if not isinstance(records, list) or len(records) != total_views:
            raise ReconstructionValidationError("I4.3E3 worker must return one frame record per registered view.")
        self._publish(stage=DenseProviderProgressStage.PROVIDER_EXECUTION, detail="moge2_depth_evidence_ready", progress=0.35, processed=total_views, total=total_views)

        scene_scale = _robust_scene_diagonal(context.sfm_result)
        voxel_size = max(scene_scale * context.config.voxel_scene_fraction, 1e-8)
        reconstruction_camera = reconstruction_camera_from_sfm_model(
            camera_model,
            keyframe_id=images[0].keyframe_id,
            reconstruction_frame_id=context.reconstruction_frame_id,
        )
        camera_profile = camera_geometry_from_estimated_camera(
            profile_id=f"i43e3-{camera_id}-sfm-estimated",
            camera=reconstruction_camera,
            focal_length_relative_uncertainty=0.20,
            notes="I4.3E3 camera geometry from authoritative PyCOLMAP SfM; arbitrary gauge only.",
        )
        scale_profile = arbitrary_scale_reference(
            profile_id=f"i43e3-{context.sfm_result.session_id[:12]}-arbitrary-sfm-gauge",
            notes=(
                "I4.3E3 aligns MoGe-2 depth into the arbitrary SfM gauge; native metric depth "
                "while model-predicted intrinsics/direct points/normals are not promoted to canonical authority."
            ),
        )

        frames: list[DenseFusionFrame] = []
        alignment_records: list[dict[str, Any]] = []
        rejected: list[str] = []
        worker_dir = worker.manifest_path.parent
        for index, (item, record) in enumerate(zip(images, records, strict=True)):
            if not isinstance(record, Mapping):
                raise ReconstructionValidationError("I4.3E3 worker frame records must be objects.")
            if record.get("keyframe_id") != item.keyframe_id or int(record.get("order", -1)) != index:
                raise ReconstructionValidationError("I4.3E3 worker frame ordering/keyframe identity drifted.")
            if record.get("native_depth_scale") != "metric" or record.get("native_is_metric") is not True:
                raise ReconstructionValidationError("I4.3E3 worker changed MoGe-2 native metric-depth semantics.")
            if record.get("intrinsics_source_native") != "model_predicted":
                raise ReconstructionValidationError("I4.3E3 worker changed MoGe-2 model-predicted-intrinsics semantics.")
            if record.get("confidence_kind_native") != "none":
                raise ReconstructionValidationError("I4.3E3 worker changed MoGe-2 no-confidence semantics.")
            if record.get("direct_points_native") is not True or record.get("normals_native") is not True:
                raise ReconstructionValidationError("I4.3E3 worker dropped native MoGe-2 geometry evidence.")
            if any(
                record.get(key) is True
                for key in (
                    "model_intrinsics_are_canonical_authority",
                    "model_direct_points_are_canonical_geometry",
                    "model_normals_are_canonical_geometry",
                    "model_metric_depth_establishes_metric_scale",
                )
            ):
                raise ReconstructionValidationError("I4.3E3 worker frame attempted to promote native MoGe-2 authority.")
            evidence_path = Path(str(record.get("evidence_npz_path", ""))).expanduser()
            if not evidence_path.is_absolute():
                evidence_path = (worker_dir / evidence_path).resolve()
            if not evidence_path.is_file():
                raise ReconstructionValidationError(f"I4.3E3 worker evidence file is missing: {evidence_path}")
            try:
                with np.load(evidence_path, allow_pickle=False) as archive:
                    depth = np.asarray(archive["depth"], dtype=np.float32)
                    valid_mask = np.asarray(archive["valid_mask"], dtype=bool)
                    direct_points = np.asarray(archive["direct_points_model"], dtype=np.float32)
                    normals = np.asarray(archive["normals_model"], dtype=np.float32)
                    model_intrinsics = np.asarray(archive["model_intrinsics"], dtype=np.float32)
            except Exception as exc:  # noqa: BLE001
                raise ReconstructionValidationError(f"I4.3E3 could not read worker evidence {evidence_path}: {exc}") from exc
            expected_hw = (item.image_size.height, item.image_size.width)
            if depth.shape != expected_hw or valid_mask.shape != expected_hw:
                raise ReconstructionValidationError("I4.3E3 depth/mask evidence must remain on each authoritative SfM pixel grid.")
            if not np.isfinite(depth[valid_mask]).all() or np.any(depth[valid_mask] <= 0.0):
                raise ReconstructionValidationError("I4.3E3 valid MoGe-2 depth evidence must be finite and positive.")
            if model_intrinsics.shape != (3, 3) or not np.isfinite(model_intrinsics).all():
                raise ReconstructionValidationError("I4.3E3 MoGe-2 intrinsics evidence is invalid.")
            if direct_points.ndim != 3 or direct_points.shape[-1] != 3 or not np.isfinite(direct_points).all():
                raise ReconstructionValidationError("I4.3E3 MoGe-2 direct-points evidence is invalid.")
            if normals.ndim != 3 or normals.shape[-1] != 3 or not np.isfinite(normals).all():
                raise ReconstructionValidationError("I4.3E3 MoGe-2 normals evidence is invalid.")

            evidence = align_sfm_keyframe_dense_depth(
                context.sfm_result,
                keyframe_id=item.keyframe_id,
                dense_depth_source_units=depth,
                depth_provenance=(
                    f"MoGe-2 source_revision={self.environment.source_revision};"
                    f"checkpoint_revision={self.environment.checkpoint_revision};I4.3E3;"
                    "single_image=true;native_metric=true;authoritative_pose=pycolmap_sfm;"
                    "native_metric_depth_not_promoted=true;model_intrinsics_not_canonical=true;"
                    "direct_points_not_canonical=true;normals_not_canonical=true"
                ),
                minimum_correspondences=context.config.minimum_correspondences,
                maximum_relative_residual=context.config.maximum_relative_residual,
            )
            alignment_record = evidence.as_dict()
            alignment_record["native_metric_depth_promoted_to_canonical_metric_scale"] = False
            alignment_record["model_predicted_intrinsics_consumed_for_canonical_projection"] = False
            alignment_record["model_direct_points_consumed_for_canonical_geometry"] = False
            alignment_record["model_normals_consumed_for_canonical_geometry"] = False
            alignment_record["native_confidence_available"] = False
            alignment_records.append(alignment_record)
            if not evidence.is_accepted or evidence.alignment is None:
                rejected.append(item.keyframe_id)
            else:
                bgr = cv2.imread(str(paths[index]), cv2.IMREAD_COLOR)
                if bgr is None:
                    raise ReconstructionValidationError(f"I4.3E3 could not decode staged RGB: {paths[index]}")
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
                        confidence=None,
                        valid_mask=valid_mask,
                        pose_layer="offline_sfm_bundle_adjusted",
                    )
                )
            self._publish(
                stage=DenseProviderProgressStage.ALIGNMENT,
                detail="align_moge2_depth_to_authoritative_sfm",
                progress=0.40 + 0.35 * ((index + 1) / float(total_views)),
                processed=index + 1,
                total=total_views,
            )

        if len(frames) < context.config.minimum_accepted_views:
            raise ReconstructionValidationError(
                "I4.3E3 insufficient MoGe-2/SfM alignment-accepted views: "
                f"accepted={len(frames)} required={context.config.minimum_accepted_views}."
            )

        self._publish(stage=DenseProviderProgressStage.FUSION_FILTERING, detail="qualified_moge2_voxel_fusion", progress=0.82, processed=len(frames), total=total_views)
        fusion = fuse_dense_frames(
            frames,
            session_id=f"{context.sfm_result.session_id}:i4.3e3-moge-2",
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
        canonical_npz = context.output_dir / "canonical_moge_2_fused.npz"
        np.savez_compressed(
            canonical_npz,
            points_reconstruction=cloud.points_reconstruction,
            colors_rgb=cloud.colors_rgb,
            view_observation_counts=cloud.view_observation_counts,
            mean_confidence=cloud.mean_confidence,
            voxel_size_reconstruction_units=np.asarray(cloud.voxel_size_reconstruction_units, dtype=np.float64),
        )
        canonical_ply = _write_binary_ply(
            context.output_dir / "canonical_moge_2_fused.ply",
            cloud.points_reconstruction,
            cloud.colors_rgb,
        )

        for item, path in zip(images, paths, strict=True):
            if _sha256_file(path) != before_hashes[item.keyframe_id]:
                raise ReconstructionValidationError("I4.3E3 registered staged images changed during provider execution.")
        if _sfm_authority_digest(context.sfm_result) != authority_before:
            raise ReconstructionValidationError("I4.3E3 authoritative SfM state changed during provider execution.")

        degradation_reasons: list[str] = []
        if rejected:
            degradation_reasons.append(f"alignment_rejected_views:{len(rejected)}")
        status = DenseProviderRunStatus.DEGRADED if degradation_reasons else DenseProviderRunStatus.COMPLETED
        self._publish(stage=DenseProviderProgressStage.FINALIZATION, detail="persist_moge2_canonical_candidate", progress=1.0, processed=len(frames), total=total_views)

        result = DenseProviderExecutionResult(
            provider_id=I43E3_PROVIDER_ID,
            implementation=DenseProviderImplementation.MOGE_2,
            status=status,
            source_view_count=total_views,
            selected_view_count=total_views,
            adaptive_view_selection=False,
            view_selection_reason="all_registered_views_i4_3e3_adaptive_selection_deferred_i4_3f",
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
            context.output_dir / "i4_3e3_moge_2_execution_summary.json",
            {
                "schema_version": I43E3_SCHEMA_VERSION,
                "phase": I43E3_PHASE,
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
                    "model_predicted_intrinsics_consumed_for_canonical_projection": False,
                    "model_direct_points_consumed_for_canonical_geometry": False,
                    "model_normals_consumed_for_canonical_geometry": False,
                    "metric_scale_claimed": False,
                    "measurement_authoritative": False,
                    "certified_metrology": False,
                    "surface_mesh_generated": False,
                    "provider_promoted": False,
                },
            },
        )
        return result


def validate_i4_3e3_binding(
    *,
    i4_3a_snapshot: Mapping[str, Any],
    i4_3d3_snapshot: Mapping[str, Any],
    i4_3e1_snapshot: Mapping[str, Any],
    i4_3e2_snapshot: Mapping[str, Any],
    environment: MoGe2EnvironmentProbe,
) -> dict[str, Any]:
    if not all(
        isinstance(item, Mapping)
        for item in (i4_3a_snapshot, i4_3d3_snapshot, i4_3e1_snapshot, i4_3e2_snapshot)
    ):
        raise TypeError("I4.3E3 binding snapshots must be mappings.")
    if i4_3a_snapshot.get("phase") != "I4.3A" or i4_3a_snapshot.get("status") != "pass" or i4_3a_snapshot.get("pass") is not True:
        raise ReconstructionValidationError("I4.3E3 requires the passing I4.3A architecture snapshot.")
    architecture = i4_3a_snapshot.get("architecture")
    if not isinstance(architecture, Mapping):
        raise ReconstructionValidationError("I4.3E3 I4.3A snapshot is missing architecture.")
    if architecture.get("authoritative_camera_frame_owner") != I43A_AUTHORITATIVE_FRAME_OWNER:
        raise ReconstructionValidationError("I4.3E3 I4.3A camera-frame authority drifted.")
    contracts = architecture.get("provider_implementation_contracts")
    if not isinstance(contracts, list):
        raise ReconstructionValidationError("I4.3E3 I4.3A implementation contracts are missing.")
    match = next((item for item in contracts if isinstance(item, Mapping) and item.get("implementation") == "moge_2"), None)
    if not isinstance(match, Mapping):
        raise ReconstructionValidationError("I4.3E3 MoGe-2 implementation contract is missing.")
    if match.get("integration_stage") != "I4.3E" or match.get("family") != "single_image_depth":
        raise ReconstructionValidationError("I4.3E3 MoGe-2 implementation contract drifted.")
    if match.get("frame_binding") != "align_to_authoritative_sfm" or match.get("native_output") != "depth_evidence":
        raise ReconstructionValidationError("I4.3E3 single-image depth frame/output contract drifted.")

    if i4_3d3_snapshot.get("phase") != "I4.3D3" or i4_3d3_snapshot.get("status") != "pass" or i4_3d3_snapshot.get("pass") is not True:
        raise ReconstructionValidationError("I4.3E3 requires the passing I4.3D3 temporal-provider closure snapshot.")
    decision = i4_3d3_snapshot.get("decision")
    authority = i4_3d3_snapshot.get("authority")
    if not isinstance(decision, Mapping) or not isinstance(authority, Mapping):
        raise ReconstructionValidationError("I4.3E3 I4.3D3 closure snapshot is incomplete.")
    if decision.get("i4_3d_temporal_provider_architecture_closed") is not True or decision.get("i4_3e_authorized") is not True:
        raise ReconstructionValidationError("I4.3E3 is not authorized by the I4.3D3 closure decision.")
    if any(
        authority.get(key) is True
        for key in ("metric_scale_claimed", "measurement_authoritative", "certified_metrology", "surface_mesh_generated")
    ):
        raise ReconstructionValidationError("I4.3E3 refuses an I4.3D3 snapshot with promoted authority claims.")


    for phase, implementation, snapshot in (
        ("I4.3E1", "metric3d_v2", i4_3e1_snapshot),
        ("I4.3E2", "unidepth_v2", i4_3e2_snapshot),
    ):
        if snapshot.get("phase") != phase or snapshot.get("status") != "pass" or snapshot.get("pass") is not True:
            raise ReconstructionValidationError(f"I4.3E3 requires the passing {phase} predecessor binding snapshot.")
        provider = snapshot.get("provider")
        predecessor_authority = snapshot.get("authority")
        if not isinstance(provider, Mapping) or not isinstance(predecessor_authority, Mapping):
            raise ReconstructionValidationError(f"I4.3E3 {phase} predecessor snapshot is incomplete.")
        if provider.get("implementation") != implementation or provider.get("qualification_state") != "candidate":
            raise ReconstructionValidationError(f"I4.3E3 requires the closed candidate {phase} provider state.")
        if predecessor_authority.get("provider_promoted") is True or any(
            predecessor_authority.get(key) is True
            for key in ("metric_scale_claimed", "measurement_authoritative", "certified_metrology", "surface_mesh_generated")
        ):
            raise ReconstructionValidationError(f"I4.3E3 refuses a {phase} predecessor with promoted authority claims.")

    descriptor = moge_2_dense_provider_descriptor(environment)
    if not descriptor.available:
        raise ReconstructionValidationError("I4.3E3 MoGe-2 runtime is not ready for binding.")
    return {
        "schema_version": I43E3_SCHEMA_VERSION,
        "phase": I43E3_PHASE,
        "status": "pass",
        "pass": True,
        "provider": descriptor.as_dict(),
        "architecture": {
            "family": "single_image_depth",
            "implementation": "moge_2",
            "frame_binding": "align_to_authoritative_sfm",
            "native_output": "depth_evidence",
            "native_depth_scale": "metric",
            "native_intrinsics_source": "model_predicted",
            "native_confidence_kind": "none",
            "native_direct_points_available": True,
            "native_normals_available": True,
            "native_valid_mask_available": True,
            "per_view_alignment": "sparse_correspondence_scale_to_authoritative_sfm",
            "all_registered_views_available_by_default": True,
            "permanent_global_view_cap": None,
            "adaptive_view_selection_deferred_to": "I4.3F",
        },
        "authority": {
            "camera_pose_authority_owner": I43A_AUTHORITATIVE_FRAME_OWNER,
            "model_metric_depth_promoted_to_canonical_metric_scale": False,
            "model_predicted_intrinsics_consumed_for_canonical_projection": False,
            "model_direct_points_consumed_for_canonical_geometry": False,
            "model_normals_consumed_for_canonical_geometry": False,
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
            "remaining_i4_3e_providers_pending": [
                "depth_pro",
                "depth_anything_v2",
                "marigold",
            ],
        },
        "environment": environment.as_dict(),
        "next": "I4.3E4 Depth Pro production single-image-depth provider adapter.",
    }


__all__ = [
    "I43E3_ADAPTER_VERSION",
    "I43E3_PHASE",
    "I43E3_PROVIDER_ID",
    "MoGe2EnvironmentProbe",
    "MoGe2ProductionConfig",
    "MoGe2ProductionProviderAdapter",
    "MoGe2ProviderRunContext",
    "MoGe2WorkerResult",
    "SubprocessMoGe2WorkerRunner",
    "moge_2_dense_provider_descriptor",
    "probe_moge_2_environment",
    "validate_i4_3e3_binding",
]

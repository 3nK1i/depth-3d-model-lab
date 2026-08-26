"""I4.3C canonical production adapter for Depth Anything 3.

This adapter binds the already-qualified isolated DA3 runtime behind the I4.3A
provider contract.  DA3 remains a multi-view AI-depth provider: its depth and
confidence outputs are native provider evidence, while canonical geometry is
constructed only after per-view depth alignment into the existing authoritative
PyCOLMAP SfM gauge and projection with the authoritative SfM camera poses.

DA3-predicted extrinsics are validated and preserved as native evidence but are
never used as the canonical camera trajectory.  I4.3C uses every registered SfM
view and deliberately introduces no permanent view cap.  Adaptive view selection
and hardware-aware batching belong to I4.3F / later robustness work.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from math import isfinite
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

import cv2
import numpy as np

from core.contracts.validation import metadata_as_dict, require_nonempty
from core.reconstruction.camera_geometry import camera_geometry_from_estimated_camera
from core.reconstruction.da3 import (
    DA3_PROVIDER_ID,
    DA3EnvironmentProbe,
    DA3ResultManifest,
    probe_da3_environment,
    validate_da3_npz_result,
)
from core.reconstruction.da3_provider import (
    DA3ExternalWorkerRunner,
    DA3InputManifest,
    DA3ProviderProfile,
    write_da3_input_manifest,
)
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

I43C_SCHEMA_VERSION = 1
I43C_PHASE = "I4.3C"
I43C_ADAPTER_VERSION = "i4.3c-v1"
I43C_DEFAULT_VOXEL_SCENE_FRACTION = 0.004
I43C_DEFAULT_MINIMUM_CORRESPONDENCES = 8
I43C_DEFAULT_MAXIMUM_RELATIVE_RESIDUAL = 0.35
I43C_DEFAULT_PIXEL_STRIDE = 2
I43C_DEFAULT_MAX_VOXELS = 1_500_000
I43C_DEFAULT_MINIMUM_ACCEPTED_VIEWS = 2
I43C_DEFAULT_MAX_PIXEL_SCALE_ANISOTROPY = 0.02


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


def _require_clean_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise ReconstructionValidationError(
            f"I4.3C output directory must be empty or absent; refusing to overwrite provider evidence: {path}"
        )


def _default_image_loader(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ReconstructionValidationError(f"I4.3C could not decode registered RGB keyframe: {path}")
    return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)


def _adapt_grid_to_sfm(
    values: object,
    *,
    sfm_width: int,
    sfm_height: int,
    maximum_scale_anisotropy: float,
    field_name: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] <= 0 or array.shape[1] <= 0:
        raise ReconstructionValidationError(f"I4.3C {field_name} must be a non-empty HxW array.")
    if not np.isfinite(array).all():
        raise ReconstructionValidationError(f"I4.3C {field_name} must be finite.")
    if field_name == "depth" and np.any(array <= 0.0):
        raise ReconstructionValidationError("I4.3C DA3 depth must be strictly positive.")
    if field_name == "confidence" and np.any(array < 0.0):
        raise ReconstructionValidationError("I4.3C DA3 confidence must be non-negative.")
    width = int(sfm_width)
    height = int(sfm_height)
    if width <= 0 or height <= 0:
        raise ReconstructionValidationError("I4.3C SfM image dimensions must be positive.")
    maximum = float(maximum_scale_anisotropy)
    if not isfinite(maximum) or not 0.0 <= maximum < 1.0:
        raise ReconstructionValidationError("I4.3C maximum_scale_anisotropy must be finite in [0,1).")

    source_height, source_width = array.shape
    if source_width == width and source_height == height:
        result = np.ascontiguousarray(array.copy())
        result.setflags(write=False)
        return result, {
            "mode": "identity",
            "source_size": [source_width, source_height],
            "target_sfm_size": [width, height],
            "relative_scale_anisotropy": 0.0,
            "interpolation": "none",
        }

    scale_x = float(source_width) / float(width)
    scale_y = float(source_height) / float(height)
    anisotropy = abs(scale_x - scale_y) / max(scale_x, scale_y, 1e-12)
    if anisotropy > maximum:
        raise ReconstructionValidationError(
            "I4.3C DA3 output pixel scaling is not sufficiently uniform to align safely "
            f"to the SfM grid (anisotropy={anisotropy:.6f}, limit={maximum:.6f})."
        )
    resized = cv2.resize(array, (width, height), interpolation=cv2.INTER_LINEAR)
    resized = np.asarray(resized, dtype=np.float32)
    if resized.shape != (height, width) or not np.isfinite(resized).all():
        raise ReconstructionValidationError(f"I4.3C {field_name} resize produced invalid evidence.")
    if field_name == "depth" and np.any(resized <= 0.0):
        raise ReconstructionValidationError("I4.3C resized depth became non-positive.")
    if field_name == "confidence" and np.any(resized < 0.0):
        raise ReconstructionValidationError("I4.3C resized confidence became negative.")
    result = np.ascontiguousarray(resized.copy())
    result.setflags(write=False)
    return result, {
        "mode": "aspect_preserving_resize_to_sfm_grid",
        "source_size": [source_width, source_height],
        "target_sfm_size": [width, height],
        "scale_x": scale_x,
        "scale_y": scale_y,
        "relative_scale_anisotropy": anisotropy,
        "maximum_allowed_scale_anisotropy": maximum,
        "interpolation": "opencv_linear",
        "authority_note": "pixel-grid adaptation only; no metric or camera-pose authority is created",
    }


def _robust_scene_diagonal(sfm_result: OfflineSfmResult) -> float:
    if len(sfm_result.landmarks) < 4:
        raise ReconstructionValidationError("I4.3C requires sparse SfM landmarks to establish a scene-relative fusion scale.")
    points = np.vstack([np.asarray(item.position_reconstruction, dtype=np.float64) for item in sfm_result.landmarks])
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ReconstructionValidationError("I4.3C sparse SfM landmarks are invalid.")
    low = np.quantile(points, 0.02, axis=0)
    high = np.quantile(points, 0.98, axis=0)
    diagonal = float(np.linalg.norm(high - low))
    if not isfinite(diagonal) or diagonal <= 0.0:
        raise ReconstructionValidationError("I4.3C robust sparse scene diagonal is invalid.")
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
        raise ReconstructionValidationError("I4.3C PLY geometry must be aligned Nx3 XYZ/RGB arrays.")
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
class DA3ProductionConfig:
    minimum_correspondences: int = I43C_DEFAULT_MINIMUM_CORRESPONDENCES
    maximum_relative_residual: float = I43C_DEFAULT_MAXIMUM_RELATIVE_RESIDUAL
    voxel_scene_fraction: float = I43C_DEFAULT_VOXEL_SCENE_FRACTION
    pixel_stride: int = I43C_DEFAULT_PIXEL_STRIDE
    max_voxels: int = I43C_DEFAULT_MAX_VOXELS
    minimum_accepted_views: int = I43C_DEFAULT_MINIMUM_ACCEPTED_VIEWS
    maximum_pixel_scale_anisotropy: float = I43C_DEFAULT_MAX_PIXEL_SCALE_ANISOTROPY

    def __post_init__(self) -> None:
        if int(self.minimum_correspondences) < 4:
            raise ReconstructionValidationError("I4.3C minimum_correspondences must be at least 4.")
        residual = float(self.maximum_relative_residual)
        if not isfinite(residual) or not 0.0 < residual < 1.0:
            raise ReconstructionValidationError("I4.3C maximum_relative_residual must be finite in (0,1).")
        fraction = float(self.voxel_scene_fraction)
        if not isfinite(fraction) or not 0.0 < fraction < 0.1:
            raise ReconstructionValidationError("I4.3C voxel_scene_fraction must be finite in (0,0.1).")
        if int(self.pixel_stride) <= 0:
            raise ReconstructionValidationError("I4.3C pixel_stride must be positive.")
        if int(self.max_voxels) <= 0:
            raise ReconstructionValidationError("I4.3C max_voxels must be positive.")
        if int(self.minimum_accepted_views) < 2:
            raise ReconstructionValidationError("I4.3C minimum_accepted_views must be at least 2.")
        anisotropy = float(self.maximum_pixel_scale_anisotropy)
        if not isfinite(anisotropy) or not 0.0 <= anisotropy < 1.0:
            raise ReconstructionValidationError("I4.3C maximum_pixel_scale_anisotropy must be finite in [0,1).")
        object.__setattr__(self, "minimum_correspondences", int(self.minimum_correspondences))
        object.__setattr__(self, "maximum_relative_residual", residual)
        object.__setattr__(self, "voxel_scene_fraction", fraction)
        object.__setattr__(self, "pixel_stride", int(self.pixel_stride))
        object.__setattr__(self, "max_voxels", int(self.max_voxels))
        object.__setattr__(self, "minimum_accepted_views", int(self.minimum_accepted_views))
        object.__setattr__(self, "maximum_pixel_scale_anisotropy", anisotropy)

    def as_dict(self) -> dict[str, Any]:
        return {
            "minimum_correspondences": self.minimum_correspondences,
            "maximum_relative_residual": self.maximum_relative_residual,
            "voxel_scene_fraction": self.voxel_scene_fraction,
            "pixel_stride": self.pixel_stride,
            "max_voxels": self.max_voxels,
            "minimum_accepted_views": self.minimum_accepted_views,
            "maximum_pixel_scale_anisotropy": self.maximum_pixel_scale_anisotropy,
            "all_registered_views_selected": True,
            "adaptive_view_selection_deferred_to": "I4.3F",
            "permanent_global_view_cap": None,
        }


@dataclass(frozen=True, slots=True)
class DA3ProviderRunContext:
    sfm_result: OfflineSfmResult
    reconstruction_frame_id: str
    output_dir: Path
    da3_python: Path
    da3_repository: Path
    worker_script: Path
    profile: DA3ProviderProfile = DA3ProviderProfile()
    config: DA3ProductionConfig = DA3ProductionConfig()

    def __post_init__(self) -> None:
        if not isinstance(self.sfm_result, OfflineSfmResult):
            raise TypeError("sfm_result must be OfflineSfmResult.")
        if self.sfm_result.backend_id != PYCOLMAP_BACKEND_ID:
            raise ReconstructionValidationError("I4.3C requires the authoritative PyCOLMAP offline SfM result.")
        if self.sfm_result.statistics.registered_keyframes < 2:
            raise ReconstructionValidationError("I4.3C requires at least two registered SfM views.")
        if not self.sfm_result.camera_models:
            raise ReconstructionValidationError("I4.3C requires returned SfM camera-model estimates.")
        frame_id = require_nonempty(self.reconstruction_frame_id, field_name="reconstruction_frame_id")
        object.__setattr__(self, "reconstruction_frame_id", frame_id)
        for name in ("output_dir", "da3_python", "da3_repository", "worker_script"):
            path = Path(getattr(self, name)).expanduser().resolve()
            object.__setattr__(self, name, path)
        if not isinstance(self.profile, DA3ProviderProfile):
            raise TypeError("profile must be DA3ProviderProfile.")
        if not isinstance(self.config, DA3ProductionConfig):
            raise TypeError("config must be DA3ProductionConfig.")


@dataclass(frozen=True, slots=True)
class _RegisteredDA3Inputs:
    images: tuple[SfmWorkspaceImage, ...]
    image_paths: tuple[Path, ...]
    image_sha256: tuple[tuple[str, str], ...]

    @property
    def image_hash_map(self) -> dict[str, str]:
        return dict(self.image_sha256)


def _resolve_registered_inputs(result: OfflineSfmResult) -> _RegisteredDA3Inputs:
    root = Path(result.workspace.workspace_root).expanduser().resolve()
    by_keyframe = {item.keyframe_id: item for item in result.workspace.images}
    registered: list[SfmWorkspaceImage] = []
    paths: list[Path] = []
    hashes: list[tuple[str, str]] = []
    for camera in sorted(result.cameras, key=lambda item: by_keyframe[item.keyframe_id].order):
        workspace = by_keyframe.get(camera.keyframe_id)
        if workspace is None:
            raise ReconstructionValidationError("I4.3C registered camera is absent from the materialized SfM workspace.")
        path = (root / workspace.staged_image_path).resolve()
        if not path.is_file():
            raise ReconstructionValidationError(f"I4.3C registered staged image is missing: {path}")
        actual = _sha256_file(path)
        if actual != workspace.staged_sha256:
            raise ReconstructionValidationError(f"I4.3C staged image checksum changed: {path.name}")
        registered.append(workspace)
        paths.append(path)
        hashes.append((workspace.keyframe_id, actual))
    if len(registered) != result.statistics.registered_keyframes:
        raise ReconstructionValidationError("I4.3C registered-view set is incomplete.")
    return _RegisteredDA3Inputs(tuple(registered), tuple(paths), tuple(hashes))


def da3_dense_provider_descriptor(probe: DA3EnvironmentProbe) -> DenseProviderDescriptor:
    if not isinstance(probe, DA3EnvironmentProbe):
        raise TypeError("probe must be DA3EnvironmentProbe.")
    return DenseProviderDescriptor(
        provider_id=DA3_PROVIDER_ID,
        implementation=DenseProviderImplementation.DA3,
        execution=ReconstructionProviderExecution.EXTERNAL_PROCESS,
        backend_version=probe.source_revision[:12] if probe.source_revision else "unqualified",
        adapter_version=I43C_ADAPTER_VERSION,
        qualification_state=DenseProviderQualificationState.CANDIDATE,
        available=probe.available,
        unavailable_reason="" if probe.available else (probe.error or "da3_environment_not_qualified"),
        supports_engineer_override=True,
        metadata={
            "phase": I43C_PHASE,
            "native_output": "depth_evidence",
            "frame_binding": "align_to_authoritative_sfm",
            "legacy_da3_runtime_reused": True,
            "da3_predicted_pose_is_canonical_authority": False,
            "canonical_projection_pose_source": "pycolmap_sfm",
            "all_registered_views_used_in_i4_3c": True,
            "adaptive_view_selection_deferred_to": "I4.3F",
            "permanent_global_view_cap": None,
            "progressive_status_supported": True,
            "provider_promotion_performed": False,
        },
    )


class DA3WorkerRunner(Protocol):
    def run(self, input_manifest_path: str | Path, output_dir: str | Path) -> tuple[Path, DA3ResultManifest]: ...


class DA3ProductionProviderAdapter:
    """Canonical I4.3C adapter around the isolated DA3 worker."""

    def __init__(
        self,
        context: DA3ProviderRunContext,
        *,
        progress_sink: DenseProviderProgressSink | None = None,
        environment_probe: DA3EnvironmentProbe | None = None,
        worker_runner: DA3WorkerRunner | None = None,
        image_loader: Callable[[Path], np.ndarray] | None = None,
    ) -> None:
        if not isinstance(context, DA3ProviderRunContext):
            raise TypeError("context must be DA3ProviderRunContext.")
        self._context = context
        self._progress_sink = progress_sink
        self._environment = environment_probe or probe_da3_environment(context.da3_python, context.da3_repository)
        self._runner = worker_runner or DA3ExternalWorkerRunner(
            python_executable=context.da3_python,
            worker_script=context.worker_script,
            profile=context.profile,
            timeout_s=600.0,
        )
        self._image_loader = image_loader or _default_image_loader
        self._descriptor = da3_dense_provider_descriptor(self._environment)
        self._progress_event_count = 0

    @property
    def descriptor(self) -> DenseProviderDescriptor:
        return self._descriptor

    @property
    def context(self) -> DA3ProviderRunContext:
        return self._context

    def _publish(self, *, stage: DenseProviderProgressStage, detail: str, progress: float, processed: int, total: int) -> None:
        event = DenseProviderProgressEvent(
            provider_id=self.descriptor.provider_id,
            implementation=DenseProviderImplementation.DA3,
            stage=stage,
            stage_detail=detail,
            progress_fraction=max(0.0, min(1.0, float(progress))),
            processed_views=max(0, min(int(processed), int(total))),
            total_views=int(total),
        )
        self._progress_event_count += 1
        if self._progress_sink is not None:
            self._progress_sink.publish(event)

    def _validate_selection(self, request: DenseProviderSelectionRequest, decision: DenseProviderSelectionDecision, *, views: int) -> None:
        if not isinstance(request, DenseProviderSelectionRequest) or not isinstance(decision, DenseProviderSelectionDecision):
            raise TypeError("I4.3C execute requires I4.3A request/decision contracts.")
        if not self.descriptor.available:
            raise ReconstructionValidationError(f"I4.3C DA3 provider is unavailable: {self.descriptor.unavailable_reason}")
        if request.authoritative_sfm_available is not True:
            raise ReconstructionValidationError("I4.3C refuses DA3 canonical binding without authoritative SfM.")
        if request.authoritative_sfm_frame_id != self.context.reconstruction_frame_id:
            raise ReconstructionValidationError("I4.3C selection frame ID does not match the provider run context.")
        if request.available_view_count != views:
            raise ReconstructionValidationError("I4.3C available_view_count must equal the authoritative registered-view count.")
        if decision.selected_provider_id != self.descriptor.provider_id:
            raise ReconstructionValidationError("I4.3C adapter may execute only when its concrete provider ID was selected.")
        if decision.selected_implementation is not DenseProviderImplementation.DA3:
            raise ReconstructionValidationError("I4.3C selection implementation must be DA3.")

    def execute(self, request: DenseProviderSelectionRequest, decision: DenseProviderSelectionDecision) -> DenseProviderExecutionResult:
        context = self.context
        inputs = _resolve_registered_inputs(context.sfm_result)
        total_views = len(inputs.images)
        self._validate_selection(request, decision, views=total_views)
        _require_clean_output(context.output_dir)
        if not context.da3_python.is_file():
            raise ReconstructionValidationError(f"I4.3C DA3 Python executable is missing: {context.da3_python}")
        if not context.da3_repository.is_dir():
            raise ReconstructionValidationError(f"I4.3C DA3 repository is missing: {context.da3_repository}")
        if not context.worker_script.is_file():
            raise ReconstructionValidationError(f"I4.3C DA3 worker script is missing: {context.worker_script}")

        camera_by_keyframe = {item.keyframe_id: item for item in context.sfm_result.cameras}
        model_by_id = {item.camera_id: item for item in context.sfm_result.camera_models}
        camera_ids = {camera_by_keyframe[item.keyframe_id].camera_id for item in inputs.images}
        if len(camera_ids) != 1:
            raise ReconstructionValidationError(
                "I4.3C currently requires one shared SfM camera model for the qualified fusion kernel; mixed-camera support is not silently approximated."
            )
        camera_id = next(iter(camera_ids))
        camera_model = model_by_id.get(camera_id)
        if camera_model is None:
            raise ReconstructionValidationError("I4.3C registered SfM camera has no camera-model estimate.")

        context.output_dir.mkdir(parents=True, exist_ok=True)
        provider_dir = context.output_dir / "provider"
        provider_dir.mkdir(parents=True, exist_ok=False)
        input_manifest = write_da3_input_manifest(
            provider_dir / "input_manifest.json",
            DA3InputManifest(
                request_id=f"i4-3c-{context.sfm_result.session_id}",
                image_paths=tuple(str(path) for path in inputs.image_paths),
                metadata={
                    "phase": I43C_PHASE,
                    "source_sfm_session_id": context.sfm_result.session_id,
                    "authoritative_sfm_frame_id": context.reconstruction_frame_id,
                    "all_registered_views_selected": True,
                    "permanent_global_view_cap": None,
                    "da3_pose_authority": False,
                },
            ),
        )

        authority_before = _sfm_authority_digest(context.sfm_result)
        images_before = dict(inputs.image_sha256)
        self._progress_event_count = 0
        self._publish(stage=DenseProviderProgressStage.PREPARING, detail="validated_authoritative_sfm_inputs", progress=0.02, processed=0, total=total_views)
        self._publish(stage=DenseProviderProgressStage.VIEW_SELECTION, detail="all_registered_sfm_views_selected_i4_3c", progress=0.05, processed=0, total=total_views)
        self._publish(stage=DenseProviderProgressStage.PROVIDER_EXECUTION, detail="isolated_da3_multi_view_inference", progress=0.10, processed=0, total=total_views)

        manifest_path, manifest = self._runner.run(input_manifest, provider_dir / "worker_output")
        arrays = validate_da3_npz_result(manifest)
        if manifest.request_id != f"i4-3c-{context.sfm_result.session_id}":
            raise ReconstructionValidationError("I4.3C DA3 result request_id does not match the provider request.")
        if manifest.source_revision != self._environment.source_revision:
            raise ReconstructionValidationError("I4.3C DA3 result source revision does not match the qualified runtime.")
        if manifest.model_id != context.profile.model_id:
            raise ReconstructionValidationError("I4.3C DA3 result model identity does not match the selected provider profile.")
        if context.profile.model_revision and manifest.model_revision != context.profile.model_revision:
            raise ReconstructionValidationError("I4.3C DA3 result model revision does not match the pinned provider profile.")
        if manifest.frame_count != total_views or arrays["depth"].shape[0] != total_views:
            raise ReconstructionValidationError("I4.3C DA3 result view count does not match the registered SfM view count.")
        self._publish(stage=DenseProviderProgressStage.PROVIDER_EXECUTION, detail="da3_depth_evidence_ready", progress=0.35, processed=total_views, total=total_views)

        scene_scale = _robust_scene_diagonal(context.sfm_result)
        voxel_size = max(scene_scale * context.config.voxel_scene_fraction, 1e-8)
        reconstruction_camera = reconstruction_camera_from_sfm_model(
            camera_model,
            keyframe_id=inputs.images[0].keyframe_id,
            reconstruction_frame_id=context.reconstruction_frame_id,
        )
        camera_profile = camera_geometry_from_estimated_camera(
            profile_id=f"i43c-{camera_id}-sfm-estimated",
            camera=reconstruction_camera,
            focal_length_relative_uncertainty=0.20,
            notes="I4.3C camera geometry from authoritative PyCOLMAP SfM; arbitrary gauge only.",
        )
        scale_profile = arbitrary_scale_reference(
            profile_id=f"i43c-{context.sfm_result.session_id[:12]}-arbitrary-sfm-gauge",
            notes="I4.3C preserves the arbitrary authoritative SfM gauge; no physical metric scale is created.",
        )
        frames: list[DenseFusionFrame] = []
        alignments: list[dict[str, Any]] = []
        rejected: list[str] = []
        pixel_adapters: list[dict[str, Any]] = []

        for index, workspace in enumerate(inputs.images):
            camera = camera_by_keyframe[workspace.keyframe_id]
            model = model_by_id.get(camera.camera_id)
            if model is None or model.camera_id != camera_model.camera_id:
                raise ReconstructionValidationError("I4.3C camera-model identity drifted during DA3 alignment.")
            depth, pixel_adapter = _adapt_grid_to_sfm(
                arrays["depth"][index],
                sfm_width=model.image_size.width,
                sfm_height=model.image_size.height,
                maximum_scale_anisotropy=context.config.maximum_pixel_scale_anisotropy,
                field_name="depth",
            )
            # Confidence is preserved/adapted for evidence only. Its numerical scale
            # is not treated as calibrated [0,1] fusion weight in I4.3C.
            _, confidence_adapter = _adapt_grid_to_sfm(
                arrays["confidence"][index],
                sfm_width=model.image_size.width,
                sfm_height=model.image_size.height,
                maximum_scale_anisotropy=context.config.maximum_pixel_scale_anisotropy,
                field_name="confidence",
            )
            pixel_adapter = dict(pixel_adapter)
            pixel_adapter["confidence"] = confidence_adapter
            pixel_adapters.append(pixel_adapter)
            evidence = align_sfm_keyframe_dense_depth(
                context.sfm_result,
                keyframe_id=workspace.keyframe_id,
                dense_depth_source_units=depth,
                depth_provenance=(
                    f"Depth Anything 3 model={manifest.model_id} revision={manifest.model_revision};"
                    f"source_revision={manifest.source_revision};I4.3C;authoritative_pose=pycolmap_sfm"
                ),
                minimum_correspondences=context.config.minimum_correspondences,
                maximum_relative_residual=context.config.maximum_relative_residual,
            )
            record = evidence.as_dict()
            record["workspace_order"] = int(workspace.order)
            record["da3_predicted_pose_consumed_for_canonical_geometry"] = False
            alignments.append(record)
            if not evidence.is_accepted or evidence.alignment is None:
                rejected.append(workspace.keyframe_id)
            else:
                rgb = self._image_loader(inputs.image_paths[index])
                expected_shape = (model.image_size.height, model.image_size.width, 3)
                if rgb.shape != expected_shape or rgb.dtype != np.uint8:
                    raise ReconstructionValidationError("I4.3C decoded RGB does not match the authoritative SfM pixel grid.")
                frames.append(
                    DenseFusionFrame(
                        frame_id=camera.frame_id,
                        timestamp_ns=workspace.timestamp_ns,
                        rgb=rgb,
                        depth=depth,
                        T_reconstruction_from_camera=camera.T_reconstruction_from_camera,
                        depth_unit=DenseDepthUnit.SOURCE_MODEL_UNITS,
                        depth_alignment=evidence.alignment,
                        confidence=None,
                        pose_layer="offline_sfm_bundle_adjusted",
                    )
                )
            fraction = 0.40 + 0.35 * ((index + 1) / float(total_views))
            self._publish(stage=DenseProviderProgressStage.ALIGNMENT, detail="align_da3_depth_to_authoritative_sfm", progress=fraction, processed=index + 1, total=total_views)

        if len(frames) < context.config.minimum_accepted_views:
            raise ReconstructionValidationError(
                "I4.3C insufficient DA3/SfM alignment-accepted views: "
                f"accepted={len(frames)} required={context.config.minimum_accepted_views}."
            )

        self._publish(stage=DenseProviderProgressStage.FUSION_FILTERING, detail="qualified_voxel_fusion_in_authoritative_sfm_frame", progress=0.80, processed=len(frames), total=total_views)
        fusion = fuse_dense_frames(
            frames,
            session_id=f"{context.sfm_result.session_id}:i4.3c-da3",
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
        canonical_npz = context.output_dir / "canonical_da3_fused.npz"
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
        canonical_ply = _write_binary_ply(context.output_dir / "canonical_da3_fused.ply", cloud.points_reconstruction, cloud.colors_rgb)

        authority_after = _sfm_authority_digest(context.sfm_result)
        images_after = {item.keyframe_id: _sha256_file(path) for item, path in zip(inputs.images, inputs.image_paths, strict=True)}
        if authority_after != authority_before:
            raise ReconstructionValidationError("I4.3C authoritative SfM cameras/landmarks changed during DA3 execution.")
        if images_after != images_before:
            raise ReconstructionValidationError("I4.3C registered staged images changed during DA3 execution.")
        if any(layer != "offline_sfm_bundle_adjusted" for layer in fusion.source_pose_layers):
            raise ReconstructionValidationError("I4.3C canonical fusion consumed a non-SfM pose layer.")

        degradation_reasons: list[str] = []
        if rejected:
            degradation_reasons.append(f"alignment_rejected_views:{len(rejected)}")
        if fusion.statistics.dropped_new_voxels_capacity:
            degradation_reasons.append(f"fusion_capacity_drops:{fusion.statistics.dropped_new_voxels_capacity}")
        status = DenseProviderRunStatus.DEGRADED if degradation_reasons else DenseProviderRunStatus.COMPLETED
        self._publish(stage=DenseProviderProgressStage.FINALIZATION, detail="canonical_da3_geometry_ready", progress=1.0, processed=total_views, total=total_views)

        result = DenseProviderExecutionResult(
            provider_id=self.descriptor.provider_id,
            implementation=DenseProviderImplementation.DA3,
            status=status,
            source_view_count=total_views,
            selected_view_count=total_views,
            adaptive_view_selection=False,
            view_selection_reason="all_registered_authoritative_sfm_views",
            permanent_global_view_cap=None,
            canonical_geometry_ref=str(canonical_npz),
            canonical_frame_id=context.reconstruction_frame_id,
            native_artifact_refs=(str(manifest_path), manifest.output_npz_path, str(canonical_ply)),
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
            "schema_version": I43C_SCHEMA_VERSION,
            "phase": I43C_PHASE,
            "status": "degraded" if status is DenseProviderRunStatus.DEGRADED else "pass",
            "pass": True,
            "provider": self.descriptor.as_dict(),
            "selection_request": request.as_dict(),
            "selection_decision": decision.as_dict(),
            "execution_result": result.as_dict(),
            "config": context.config.as_dict(),
            "profile": context.profile.as_dict(),
            "runtime": self._environment.as_dict(),
            "da3_manifest": manifest.as_dict(),
            "alignment": {
                "accepted_views": len(frames),
                "rejected_views": len(rejected),
                "rejected_keyframe_ids": rejected,
                "records": alignments,
                "pixel_adapters": pixel_adapters,
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
            },
            "authority": {
                "camera_pose_authority_owner": I43A_AUTHORITATIVE_FRAME_OWNER,
                "single_authoritative_sfm_frame_preserved": True,
                "da3_predicted_extrinsics_validated": True,
                "da3_predicted_extrinsics_consumed_for_canonical_projection": False,
                "canonical_projection_pose_source": "pycolmap_sfm",
                "sfm_rerun": False,
                "bundle_adjustment_rerun": False,
                "pose_refinement": False,
                "production_filter_policy_changed": False,
                "provider_promotion_performed": False,
                "metric_scale_claimed": False,
                "measurement_authoritative": False,
                "certified_metrology": False,
                "surface_mesh_generated": False,
            },
            "qualification": {
                "qualification_state": "candidate",
                "house_prior_da3_evidence_retained": True,
                "mvs_vs_da3_production_comparison_pending_i4_3g": True,
                "universal_provider_winner_assumed": False,
            },
            "next": "I4.3D temporal depth providers. DA3/MVS comparative production qualification remains I4.3G.",
        }
        _write_json(context.output_dir / "i4_3c_summary.json", summary)
        return result


def validate_i4_3c_binding(
    *,
    i4_3a_snapshot: Mapping[str, Any],
    environment: DA3EnvironmentProbe,
) -> dict[str, Any]:
    """Fail-closed, execution-free binding check for the real DA3 runtime."""
    architecture = dict(i4_3a_snapshot)
    if architecture.get("phase") != "I4.3A" or architecture.get("status") != "pass":
        raise ReconstructionValidationError("I4.3C requires a passing I4.3A architecture snapshot.")
    details = architecture.get("architecture")
    if not isinstance(details, Mapping):
        raise ReconstructionValidationError("I4.3C I4.3A architecture snapshot is malformed.")
    if details.get("authoritative_camera_frame_owner") != I43A_AUTHORITATIVE_FRAME_OWNER:
        raise ReconstructionValidationError("I4.3C refuses changed I4.3A camera-frame authority.")
    contracts = details.get("provider_implementation_contracts")
    if not isinstance(contracts, list):
        raise ReconstructionValidationError("I4.3C I4.3A provider implementation contracts are missing.")
    da3_contract = next((item for item in contracts if isinstance(item, Mapping) and item.get("implementation") == "da3"), None)
    if da3_contract is None or da3_contract.get("integration_stage") != "I4.3C":
        raise ReconstructionValidationError("I4.3C DA3 implementation contract is absent or assigned to another stage.")
    descriptor = da3_dense_provider_descriptor(environment)
    if not descriptor.available:
        raise ReconstructionValidationError(f"I4.3C DA3 runtime is unavailable: {descriptor.unavailable_reason}")
    return {
        "schema_version": I43C_SCHEMA_VERSION,
        "phase": I43C_PHASE,
        "status": "pass",
        "pass": True,
        "provider_bound": True,
        "provider": descriptor.as_dict(),
        "architecture": {
            "camera_pose_authority_owner": I43A_AUTHORITATIVE_FRAME_OWNER,
            "single_authoritative_sfm_frame_required": True,
            "native_output": "depth_evidence",
            "frame_binding": "align_to_authoritative_sfm",
            "all_registered_views_default_i4_3c": True,
            "adaptive_view_selection_deferred_to": "I4.3F",
            "permanent_global_view_cap": None,
            "progressive_status_supported": True,
        },
        "authority": {
            "backend_started": False,
            "provider_selected": False,
            "provider_promotion_performed": False,
            "da3_pose_authority_promoted": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
        },
        "qualification": {
            "qualification_state": "candidate",
            "house_prior_da3_evidence_retained": True,
            "mvs_vs_da3_production_comparison_pending_i4_3g": True,
            "universal_provider_winner_assumed": False,
        },
        "next": "I4.3D temporal depth providers. Do not promote DA3 or MVS as a universal winner before I4.3G/I4.3H.",
    }


__all__ = [
    "DA3ProductionConfig",
    "DA3ProviderRunContext",
    "DA3ProductionProviderAdapter",
    "I43C_ADAPTER_VERSION",
    "I43C_PHASE",
    "I43C_SCHEMA_VERSION",
    "da3_dense_provider_descriptor",
    "validate_i4_3c_binding",
]

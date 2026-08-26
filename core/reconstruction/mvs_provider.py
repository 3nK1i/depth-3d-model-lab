"""I4.3B canonical production adapter for the qualified COLMAP/PyCOLMAP MVS path.

This module binds the already-qualified MVS methodology behind the I4.3A
provider contract without changing SfM authority, production filtering policy,
or application/UI wiring.  PyCOLMAP remains isolated in an external worker.

I4.3B deliberately consumes an existing authoritative ``OfflineSfmResult`` and
uses every registered SfM view.  Adaptive view selection belongs to I4.3F; no
permanent frame/view cap is introduced here.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from math import isfinite
from pathlib import Path
import subprocess
import time
from typing import Any, Callable, Mapping, Protocol

from core.contracts.validation import metadata_as_dict, require_nonempty
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
from core.reconstruction.mvs_environment import (
    MVS_HOUSE_DEFAULT_MIN_AVAILABLE_RAM_GIB,
    MVS_HOUSE_DEFAULT_MIN_FREE_DISK_GIB,
    MVS_HOUSE_DEFAULT_MIN_FREE_VRAM_GIB,
    MvsHostResourceProbe,
    PycolmapMvsEnvironmentProbe,
    probe_mvs_host_resources,
    probe_pycolmap_mvs_environment,
)
from core.reconstruction.orchestration import ReconstructionProviderExecution
from core.reconstruction.sfm_contracts import OfflineSfmResult
from core.reconstruction.sfm_pycolmap import PYCOLMAP_BACKEND_ID

I43B_SCHEMA_VERSION = 1
I43B_PHASE = "I4.3B"
I43B_PROVIDER_ID = "colmap-mvs-external"
I43B_ADAPTER_VERSION = "i4.3b-v1"
I43B_DEFAULT_SOURCE_NEIGHBORS = 10
I43B_DEFAULT_PATCH_MATCH_CACHE_GIB = 2.0
I43B_DEFAULT_STEREO_FUSION_CACHE_GIB = 2.0
I43B_DEFAULT_GPU_INDEX = "0"
I43B_DEFAULT_WORKER_TIMEOUT_S = 4.0 * 60.0 * 60.0
I43B_DEFAULT_PROGRESS_POLL_S = 0.25
_REQUIRED_SPARSE_FILES = ("cameras.bin", "images.bin", "points3D.bin")
_OPTIONAL_SPARSE_FILES = ("frames.bin", "rigs.bin")


def _positive_finite(value: float, *, field_name: str) -> float:
    number = float(value)
    if not isfinite(number) or number <= 0.0:
        raise ReconstructionValidationError(f"{field_name} must be finite and positive.")
    return number


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError(f"Could not read {label}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"{label} must contain a JSON object.")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return path


def _require_clean_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise ReconstructionValidationError(
            f"I4.3B output directory must be empty or absent; refusing to overwrite provider evidence: {path}"
        )


def _ply_vertex_count(path: Path) -> int:
    if not path.is_file():
        raise ReconstructionValidationError(f"I4.3B fused PLY is missing: {path}")
    with path.open("rb") as handle:
        header = handle.read(1024 * 1024)
    marker = b"end_header"
    if marker not in header:
        raise ReconstructionValidationError("I4.3B fused PLY header is missing end_header.")
    text = header.split(marker, 1)[0].decode("ascii", errors="strict")
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) == 3 and parts[0] == "element" and parts[1] == "vertex":
            return int(parts[2])
    raise ReconstructionValidationError("I4.3B fused PLY header does not declare a vertex count.")


def _visibility_point_count(path: Path) -> int:
    import struct

    if not path.is_file():
        raise ReconstructionValidationError(f"I4.3B native MVS visibility sidecar is missing: {path}")
    raw = path.read_bytes()[:8]
    if len(raw) != 8:
        raise ReconstructionValidationError("I4.3B native MVS visibility header is truncated.")
    return int(struct.unpack("<Q", raw)[0])


@dataclass(frozen=True, slots=True)
class MvsProductionConfig:
    """Production adapter settings; no setting is a permanent frame/view cap."""

    source_neighbors: int = I43B_DEFAULT_SOURCE_NEIGHBORS
    patch_match_cache_gib: float = I43B_DEFAULT_PATCH_MATCH_CACHE_GIB
    stereo_fusion_cache_gib: float = I43B_DEFAULT_STEREO_FUSION_CACHE_GIB
    gpu_index: str = I43B_DEFAULT_GPU_INDEX
    max_image_size: int = -1
    geometric_consistency: bool = True

    def __post_init__(self) -> None:
        neighbors = int(self.source_neighbors)
        if neighbors < 1:
            raise ReconstructionValidationError("I4.3B source_neighbors must be positive.")
        object.__setattr__(self, "source_neighbors", neighbors)
        object.__setattr__(
            self,
            "patch_match_cache_gib",
            _positive_finite(self.patch_match_cache_gib, field_name="patch_match_cache_gib"),
        )
        object.__setattr__(
            self,
            "stereo_fusion_cache_gib",
            _positive_finite(self.stereo_fusion_cache_gib, field_name="stereo_fusion_cache_gib"),
        )
        gpu_index = str(self.gpu_index).strip()
        if not gpu_index or not all(part.strip().isdigit() for part in gpu_index.split(",")):
            raise ReconstructionValidationError(
                "I4.3B gpu_index must contain comma-separated non-negative indices."
            )
        object.__setattr__(self, "gpu_index", gpu_index)
        maximum = int(self.max_image_size)
        if maximum == 0 or maximum < -1:
            raise ReconstructionValidationError("I4.3B max_image_size must be -1 or a positive integer.")
        object.__setattr__(self, "max_image_size", maximum)
        if self.geometric_consistency is not True:
            raise ReconstructionValidationError(
                "I4.3B geometric consistency is locked ON; do not weaken the qualified MVS path."
            )

    def resolved(self, *, selected_view_count: int) -> dict[str, Any]:
        views = int(selected_view_count)
        if views < 2:
            raise ReconstructionValidationError("I4.3B MVS requires at least two registered views.")
        neighbors = min(self.source_neighbors, views - 1)
        return {
            "source_neighbors": neighbors,
            "source_selection": f"__auto__, {neighbors}",
            "all_registered_views_selected": True,
            "selected_view_count": views,
            "patch_match_cache_gib": self.patch_match_cache_gib,
            "stereo_fusion_cache_gib": self.stereo_fusion_cache_gib,
            "gpu_index": self.gpu_index,
            "max_image_size": self.max_image_size,
            "geometric_consistency": True,
            "quality_thresholds_relaxed": False,
            "permanent_global_view_cap": None,
        }


@dataclass(frozen=True, slots=True)
class MvsProviderRunContext:
    """Inputs that are deliberately outside the I4.3A selection request."""

    sfm_result: OfflineSfmResult
    reconstruction_frame_id: str
    output_dir: Path
    pycolmap_python: Path
    worker_script: Path
    config: MvsProductionConfig = MvsProductionConfig()
    min_free_disk_gib: float = MVS_HOUSE_DEFAULT_MIN_FREE_DISK_GIB
    min_available_ram_gib: float = MVS_HOUSE_DEFAULT_MIN_AVAILABLE_RAM_GIB
    min_free_vram_gib: float = MVS_HOUSE_DEFAULT_MIN_FREE_VRAM_GIB
    worker_timeout_s: float = I43B_DEFAULT_WORKER_TIMEOUT_S
    progress_poll_s: float = I43B_DEFAULT_PROGRESS_POLL_S

    def __post_init__(self) -> None:
        if not isinstance(self.sfm_result, OfflineSfmResult):
            raise TypeError("sfm_result must be OfflineSfmResult.")
        frame_id = require_nonempty(self.reconstruction_frame_id, field_name="reconstruction_frame_id")
        if self.sfm_result.backend_id != PYCOLMAP_BACKEND_ID:
            raise ReconstructionValidationError(
                "I4.3B requires the authoritative PyCOLMAP offline SfM result; another pose backend cannot be substituted."
            )
        if self.sfm_result.statistics.registered_keyframes < 2:
            raise ReconstructionValidationError("I4.3B requires at least two registered SfM views.")
        if not isinstance(self.config, MvsProductionConfig):
            raise TypeError("config must be MvsProductionConfig.")
        output = Path(self.output_dir).expanduser().resolve()
        executable = Path(self.pycolmap_python).expanduser().resolve()
        worker = Path(self.worker_script).expanduser().resolve()
        object.__setattr__(self, "reconstruction_frame_id", frame_id)
        object.__setattr__(self, "output_dir", output)
        object.__setattr__(self, "pycolmap_python", executable)
        object.__setattr__(self, "worker_script", worker)
        for name in ("min_free_disk_gib", "min_available_ram_gib", "min_free_vram_gib"):
            object.__setattr__(self, name, _positive_finite(getattr(self, name), field_name=name))
        object.__setattr__(
            self,
            "worker_timeout_s",
            _positive_finite(self.worker_timeout_s, field_name="worker_timeout_s"),
        )
        object.__setattr__(
            self,
            "progress_poll_s",
            _positive_finite(self.progress_poll_s, field_name="progress_poll_s"),
        )


@dataclass(frozen=True, slots=True)
class _ResolvedMvsInputs:
    workspace_root: Path
    images_dir: Path
    sparse_dir: Path
    image_names: tuple[str, ...]
    image_sha256: tuple[tuple[str, str], ...]
    sparse_sha256: tuple[tuple[str, str], ...]
    selected_model_id: int

    @property
    def image_hash_map(self) -> dict[str, str]:
        return dict(self.image_sha256)

    @property
    def sparse_hash_map(self) -> dict[str, str]:
        return dict(self.sparse_sha256)


def _resolve_registered_sfm_inputs(result: OfflineSfmResult) -> _ResolvedMvsInputs:
    metadata = metadata_as_dict(result.metadata)  # type: ignore[arg-type]
    try:
        selected_model_id = int(metadata.get("selected_model_id", -1))
    except (TypeError, ValueError) as exc:
        raise ReconstructionValidationError("I4.3B PyCOLMAP selected_model_id is malformed.") from exc
    if selected_model_id < 0:
        raise ReconstructionValidationError("I4.3B requires the selected PyCOLMAP sparse-model ID.")

    root = Path(result.workspace.workspace_root).expanduser().resolve()
    images_dir = (root / result.workspace.images_directory).resolve()
    sparse_root = (root / result.workspace.sparse_directory).resolve()
    sparse_dir = (sparse_root / str(selected_model_id)).resolve()
    if not images_dir.is_dir() or not sparse_dir.is_dir():
        raise ReconstructionValidationError("I4.3B authoritative SfM workspace/images or selected sparse model is missing.")

    by_keyframe = {item.keyframe_id: item for item in result.workspace.images}
    registered = []
    for camera in result.cameras:
        item = by_keyframe.get(camera.keyframe_id)
        if item is None:
            raise ReconstructionValidationError("I4.3B registered camera is absent from the materialized SfM workspace.")
        registered.append(item)
    registered.sort(key=lambda item: item.order)
    names = tuple(Path(item.staged_image_path).name for item in registered)
    if len(names) != result.statistics.registered_keyframes or len(set(names)) != len(names):
        raise ReconstructionValidationError("I4.3B registered image-name set is incomplete or duplicated.")

    image_hashes: list[tuple[str, str]] = []
    for item, name in zip(registered, names, strict=True):
        path = images_dir / name
        if not path.is_file():
            raise ReconstructionValidationError(f"I4.3B registered staged image is missing: {name}")
        actual = _sha256_file(path)
        if actual != item.staged_sha256:
            raise ReconstructionValidationError(f"I4.3B staged image checksum changed: {name}")
        image_hashes.append((name, actual))

    sparse_hashes: list[tuple[str, str]] = []
    for filename in _REQUIRED_SPARSE_FILES:
        path = sparse_dir / filename
        if not path.is_file():
            raise ReconstructionValidationError(f"I4.3B selected sparse model is incomplete: {filename}")
        sparse_hashes.append((filename, _sha256_file(path)))
    for filename in _OPTIONAL_SPARSE_FILES:
        path = sparse_dir / filename
        if path.is_file():
            sparse_hashes.append((filename, _sha256_file(path)))

    return _ResolvedMvsInputs(
        workspace_root=root,
        images_dir=images_dir,
        sparse_dir=sparse_dir,
        image_names=names,
        image_sha256=tuple(image_hashes),
        sparse_sha256=tuple(sorted(sparse_hashes)),
        selected_model_id=selected_model_id,
    )


def _runtime_ready(probe: PycolmapMvsEnvironmentProbe) -> bool:
    return bool(
        probe.mvs_api_ready
        and probe.patchmatch_cuda_ready
        and probe.recommended_version
        and probe.cuda_distribution
    )


def _launch_checks(
    resources: MvsHostResourceProbe,
    config: MvsProductionConfig,
    *,
    min_free_disk_gib: float,
    min_available_ram_gib: float,
    min_free_vram_gib: float,
) -> dict[str, bool]:
    requested = {int(part.strip()) for part in config.gpu_index.split(",")}
    gpu_by_index = {gpu.index: gpu for gpu in resources.gpus}
    requested_visible = requested.issubset(gpu_by_index)
    requested_vram_ready = requested_visible and all(
        gpu_by_index[index].free_vram_gib >= min_free_vram_gib for index in requested
    )
    ram_min = max(
        min_available_ram_gib,
        config.patch_match_cache_gib + 2.0,
        config.stereo_fusion_cache_gib + 2.0,
    )
    checks = {
        "disk_ready": resources.disk_free_gib >= min_free_disk_gib,
        "ram_ready": resources.ram_available_gib >= ram_min,
        "requested_gpu_visible": requested_visible,
        "vram_ready": requested_vram_ready,
    }
    checks["ready"] = all(checks.values())
    return checks


def colmap_mvs_provider_descriptor(probe: PycolmapMvsEnvironmentProbe) -> DenseProviderDescriptor:
    if not isinstance(probe, PycolmapMvsEnvironmentProbe):
        raise TypeError("probe must be PycolmapMvsEnvironmentProbe.")
    ready = _runtime_ready(probe)
    reason = "" if ready else (probe.error or "pycolmap_mvs_runtime_not_qualified")
    return DenseProviderDescriptor(
        provider_id=I43B_PROVIDER_ID,
        implementation=DenseProviderImplementation.COLMAP_MVS,
        execution=ReconstructionProviderExecution.EXTERNAL_PROCESS,
        backend_version=probe.pycolmap_version or "unqualified",
        adapter_version=I43B_ADAPTER_VERSION,
        qualification_state=DenseProviderQualificationState.CANDIDATE,
        available=ready,
        unavailable_reason=reason,
        supports_engineer_override=True,
        metadata={
            "phase": I43B_PHASE,
            "direct_authoritative_sfm_frame_geometry": True,
            "all_registered_views_used_in_i4_3b": True,
            "adaptive_view_selection_deferred_to": "I4.3F",
            "permanent_global_view_cap": None,
            "progressive_status_supported": True,
            "production_qualification_pending": "I4.3G/I4.3H",
            "provider_promotion_performed": False,
        },
    )


class MvsWorkerRunner(Protocol):
    def run(
        self,
        *,
        command: tuple[str, ...],
        result_path: Path,
        state_path: Path,
        dense_workspace: Path,
        total_views: int,
        timeout_s: float,
        poll_s: float,
        progress_callback: Callable[[str, int, Mapping[str, Any]], None],
    ) -> int: ...


class ExternalMvsWorkerRunner:
    """Launch the isolated worker and expose dependency-light progress polling."""

    def run(
        self,
        *,
        command: tuple[str, ...],
        result_path: Path,
        state_path: Path,
        dense_workspace: Path,
        total_views: int,
        timeout_s: float,
        poll_s: float,
        progress_callback: Callable[[str, int, Mapping[str, Any]], None],
    ) -> int:
        started = time.monotonic()
        try:
            process = subprocess.Popen(command)
        except OSError as exc:
            raise ReconstructionValidationError(f"I4.3B external MVS worker could not start: {exc}") from exc

        last_signature: tuple[str, int] | None = None
        while True:
            return_code = process.poll()
            state: dict[str, Any] = {}
            if state_path.is_file():
                try:
                    state = _read_json(state_path, label="I4.3B execution state")
                except ReconstructionValidationError:
                    state = {}
            depth_count = 0
            depth_dir = dense_workspace / "stereo/depth_maps"
            if depth_dir.is_dir():
                depth_count = min(total_views, len(tuple(depth_dir.glob("*.geometric.bin"))))
            stage = str(state.get("stage", "worker_launch"))
            signature = (stage, depth_count)
            if signature != last_signature:
                progress_callback(stage, depth_count, state)
                last_signature = signature
            if return_code is not None:
                return int(return_code)
            if time.monotonic() - started > timeout_s:
                process.kill()
                process.wait(timeout=10.0)
                return 124
            time.sleep(poll_s)


def _resource_mib(state: Mapping[str, Any]) -> tuple[float | None, float | None]:
    resources = state.get("resources")
    if not isinstance(resources, Mapping):
        return None, None
    ram = resources.get("process_peak_rss_gib")
    ram_mib = None if ram is None else max(0.0, float(ram) * 1024.0)
    gpu_mib: float | None = None
    gpus = resources.get("gpus")
    if isinstance(gpus, list) and gpus:
        used = [float(item.get("used_vram_gib", 0.0)) for item in gpus if isinstance(item, Mapping)]
        if used:
            gpu_mib = max(0.0, max(used) * 1024.0)
    return gpu_mib, ram_mib


class MvsProductionProviderAdapter:
    """Canonical I4.3B adapter around the isolated COLMAP/PyCOLMAP MVS worker."""

    def __init__(
        self,
        context: MvsProviderRunContext,
        *,
        progress_sink: DenseProviderProgressSink | None = None,
        environment_probe: PycolmapMvsEnvironmentProbe | None = None,
        resource_probe: MvsHostResourceProbe | None = None,
        worker_runner: MvsWorkerRunner | None = None,
    ) -> None:
        if not isinstance(context, MvsProviderRunContext):
            raise TypeError("context must be MvsProviderRunContext.")
        self._context = context
        self._progress_sink = progress_sink
        self._environment = environment_probe or probe_pycolmap_mvs_environment(str(context.pycolmap_python))
        self._resources = resource_probe or probe_mvs_host_resources(context.output_dir)
        self._runner = worker_runner or ExternalMvsWorkerRunner()
        self._descriptor = colmap_mvs_provider_descriptor(self._environment)
        self._progress_event_count = 0

    @property
    def descriptor(self) -> DenseProviderDescriptor:
        return self._descriptor

    @property
    def context(self) -> MvsProviderRunContext:
        return self._context

    def _publish(
        self,
        *,
        stage: DenseProviderProgressStage,
        detail: str,
        progress: float,
        processed: int,
        total: int,
        state: Mapping[str, Any] | None = None,
    ) -> None:
        state = state or {}
        gpu_mib, ram_mib = _resource_mib(state)
        event = DenseProviderProgressEvent(
            provider_id=self.descriptor.provider_id,
            implementation=DenseProviderImplementation.COLMAP_MVS,
            stage=stage,
            stage_detail=detail,
            progress_fraction=max(0.0, min(1.0, float(progress))),
            processed_views=max(0, min(int(processed), int(total))),
            total_views=int(total),
            gpu_vram_used_mib=gpu_mib,
            process_ram_used_mib=ram_mib,
        )
        self._progress_event_count += 1
        if self._progress_sink is not None:
            self._progress_sink.publish(event)

    def _validate_selection(
        self,
        request: DenseProviderSelectionRequest,
        decision: DenseProviderSelectionDecision,
        *,
        registered_views: int,
    ) -> None:
        if not isinstance(request, DenseProviderSelectionRequest) or not isinstance(decision, DenseProviderSelectionDecision):
            raise TypeError("I4.3B execute requires I4.3A request/decision contracts.")
        if not self.descriptor.available:
            raise ReconstructionValidationError(
                f"I4.3B COLMAP MVS provider is unavailable: {self.descriptor.unavailable_reason}"
            )
        if request.authoritative_sfm_available is not True:
            raise ReconstructionValidationError("I4.3B refuses MVS without authoritative SfM.")
        if request.authoritative_sfm_frame_id != self.context.reconstruction_frame_id:
            raise ReconstructionValidationError("I4.3B selection frame ID does not match the provider run context.")
        if request.available_view_count != registered_views:
            raise ReconstructionValidationError("I4.3B available_view_count must equal the authoritative registered-view count.")
        if decision.selected_provider_id != self.descriptor.provider_id:
            raise ReconstructionValidationError("I4.3B adapter may execute only when its concrete provider ID was selected.")
        if decision.selected_implementation is not DenseProviderImplementation.COLMAP_MVS:
            raise ReconstructionValidationError("I4.3B selection implementation must be COLMAP_MVS.")

    def execute(
        self,
        request: DenseProviderSelectionRequest,
        decision: DenseProviderSelectionDecision,
    ) -> DenseProviderExecutionResult:
        context = self.context
        inputs = _resolve_registered_sfm_inputs(context.sfm_result)
        total_views = len(inputs.image_names)
        self._validate_selection(request, decision, registered_views=total_views)
        _require_clean_output(context.output_dir)
        if not context.pycolmap_python.is_file():
            raise ReconstructionValidationError(f"I4.3B PyCOLMAP Python executable is missing: {context.pycolmap_python}")
        if not context.worker_script.is_file():
            raise ReconstructionValidationError(f"I4.3B worker script is missing: {context.worker_script}")

        checks = _launch_checks(
            self._resources,
            context.config,
            min_free_disk_gib=context.min_free_disk_gib,
            min_available_ram_gib=context.min_available_ram_gib,
            min_free_vram_gib=context.min_free_vram_gib,
        )
        if not checks["ready"]:
            raise ReconstructionValidationError(
                f"I4.3B resource launch guardrails are not satisfied: checks={checks} resources={self._resources.as_dict()}"
            )

        context.output_dir.mkdir(parents=True, exist_ok=True)
        worker_dir = context.output_dir / "worker"
        dense_dir = context.output_dir / "dense"
        worker_dir.mkdir(parents=True, exist_ok=False)
        resolved_config = context.config.resolved(selected_view_count=total_views)
        request_payload = {
            "schema_version": I43B_SCHEMA_VERSION,
            "phase": I43B_PHASE,
            "action": "colmap_mvs_dense_provider",
            "input_images": str(inputs.images_dir),
            "input_sparse": str(inputs.sparse_dir),
            "selected_model_id": inputs.selected_model_id,
            "image_names": list(inputs.image_names),
            "image_sha256": inputs.image_hash_map,
            "input_sparse_sha256": inputs.sparse_hash_map,
            "expected_registered_views": total_views,
            "reconstruction_frame_id": context.reconstruction_frame_id,
            "output_workspace": str(dense_dir),
            "fused_ply": str(dense_dir / "fused.ply"),
            "config": resolved_config,
            "runtime": self._environment.as_dict(),
            "launch_resources": self._resources.as_dict(),
            "launch_checks": checks,
            "authority": {
                "camera_pose_authority_owner": I43A_AUTHORITATIVE_FRAME_OWNER,
                "single_authoritative_sfm_frame": True,
                "sfm_rerun": False,
                "bundle_adjustment_rerun": False,
                "pose_refinement": False,
                "metric_scale_claimed": False,
                "measurement_authoritative": False,
                "certified_metrology": False,
                "surface_mesh_generated": False,
                "production_filter_policy_changed": False,
                "provider_promotion_performed": False,
                "permanent_global_view_cap": None,
            },
        }
        request_path = _write_json(worker_dir / "request.json", request_payload)
        result_path = worker_dir / "result.json"
        state_path = worker_dir / "execution_state.json"

        self._progress_event_count = 0
        self._publish(
            stage=DenseProviderProgressStage.PREPARING,
            detail="validated_authoritative_sfm_inputs",
            progress=0.02,
            processed=0,
            total=total_views,
        )
        self._publish(
            stage=DenseProviderProgressStage.VIEW_SELECTION,
            detail="all_registered_sfm_views_selected_i4_3b",
            progress=0.05,
            processed=0,
            total=total_views,
        )

        def progress_callback(stage: str, depth_count: int, state: Mapping[str, Any]) -> None:
            if stage in {"undistort", "worker_launch"}:
                generic_stage = DenseProviderProgressStage.PROVIDER_EXECUTION
                detail = "image_undistortion" if stage == "undistort" else "external_worker_launch"
                fraction = 0.10
            elif stage == "patch_match":
                generic_stage = DenseProviderProgressStage.PROVIDER_EXECUTION
                detail = "patch_match_stereo"
                fraction = 0.15 + 0.60 * (depth_count / float(max(1, total_views)))
            elif stage == "stereo_fusion":
                generic_stage = DenseProviderProgressStage.FUSION_FILTERING
                detail = "stereo_fusion"
                fraction = 0.85
            elif stage == "failed":
                generic_stage = DenseProviderProgressStage.FINALIZATION
                detail = "worker_failed"
                fraction = 0.95
            else:
                generic_stage = DenseProviderProgressStage.PROVIDER_EXECUTION
                detail = stage or "provider_execution"
                fraction = 0.12
            self._publish(
                stage=generic_stage,
                detail=detail,
                progress=fraction,
                processed=depth_count,
                total=total_views,
                state=state,
            )

        return_code = self._runner.run(
            command=(
                str(context.pycolmap_python),
                str(context.worker_script),
                "--request",
                str(request_path),
                "--result",
                str(result_path),
                "--state",
                str(state_path),
            ),
            result_path=result_path,
            state_path=state_path,
            dense_workspace=dense_dir,
            total_views=total_views,
            timeout_s=context.worker_timeout_s,
            poll_s=context.progress_poll_s,
            progress_callback=progress_callback,
        )

        worker_result = _read_json(result_path, label="I4.3B worker result") if result_path.is_file() else {}
        success = return_code == 0 and worker_result.get("status") == "pass"
        failure_reason = ""
        if not success:
            if return_code == 124:
                failure_reason = f"I4.3B external MVS worker timed out after {context.worker_timeout_s:.3f}s."
            else:
                failure_reason = str(worker_result.get("error", "")).strip() or f"I4.3B worker exited with code {return_code}."

        post_sparse = {name: _sha256_file(inputs.sparse_dir / name) for name in inputs.sparse_hash_map}
        post_images = {name: _sha256_file(inputs.images_dir / name) for name in inputs.image_names}
        if post_sparse != inputs.sparse_hash_map:
            raise ReconstructionValidationError("I4.3B authoritative sparse model changed during MVS execution.")
        if post_images != inputs.image_hash_map:
            raise ReconstructionValidationError("I4.3B registered staged images changed during MVS execution.")

        fused_ply = dense_dir / "fused.ply"
        visibility = Path(f"{fused_ply}.vis")
        if success:
            vertices = _ply_vertex_count(fused_ply)
            visibility_points = _visibility_point_count(visibility)
            if vertices <= 0 or visibility_points != vertices:
                raise ReconstructionValidationError(
                    "I4.3B fused geometry/native visibility correspondence is invalid."
                )
            self._publish(
                stage=DenseProviderProgressStage.FINALIZATION,
                detail="canonical_mvs_geometry_ready",
                progress=1.0,
                processed=total_views,
                total=total_views,
            )
            result = DenseProviderExecutionResult(
                provider_id=self.descriptor.provider_id,
                implementation=DenseProviderImplementation.COLMAP_MVS,
                status=DenseProviderRunStatus.COMPLETED,
                source_view_count=total_views,
                selected_view_count=total_views,
                adaptive_view_selection=False,
                view_selection_reason="all_registered_authoritative_sfm_views",
                permanent_global_view_cap=None,
                canonical_geometry_ref=str(fused_ply),
                canonical_frame_id=context.reconstruction_frame_id,
                native_artifact_refs=(str(fused_ply), str(visibility), str(dense_dir)),
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
        else:
            result = DenseProviderExecutionResult(
                provider_id=self.descriptor.provider_id,
                implementation=DenseProviderImplementation.COLMAP_MVS,
                status=DenseProviderRunStatus.FAILED,
                source_view_count=total_views,
                selected_view_count=total_views,
                adaptive_view_selection=False,
                view_selection_reason="all_registered_authoritative_sfm_views",
                permanent_global_view_cap=None,
                canonical_geometry_ref="",
                canonical_frame_id="",
                native_artifact_refs=(),
                failure_reason=failure_reason,
                progress_event_count=self._progress_event_count,
                final_preview_revision=0,
                canonical_geometry_authoritative=False,
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
            "schema_version": I43B_SCHEMA_VERSION,
            "phase": I43B_PHASE,
            "status": "pass" if success else "fail",
            "pass": bool(success),
            "provider": self.descriptor.as_dict(),
            "selection_request": request.as_dict(),
            "selection_decision": decision.as_dict(),
            "execution_result": result.as_dict(),
            "resolved_config": resolved_config,
            "runtime": self._environment.as_dict(),
            "launch_resources": self._resources.as_dict(),
            "launch_checks": checks,
            "worker": worker_result,
            "integrity": {
                "authoritative_sparse_unchanged": True,
                "registered_images_unchanged": True,
                "input_sparse_sha256": inputs.sparse_hash_map,
                "registered_image_sha256": inputs.image_hash_map,
            },
            "authority": {
                "camera_pose_authority_owner": I43A_AUTHORITATIVE_FRAME_OWNER,
                "single_authoritative_sfm_frame_preserved": True,
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
            "next": (
                "I4.3C DA3 production provider adapter. Cross-provider production qualification remains I4.3G."
                if success
                else "Inspect I4.3B provider evidence; do not weaken MVS/authority gates to obtain PASS."
            ),
        }
        _write_json(context.output_dir / "i4_3b_summary.json", summary)
        return result


def validate_existing_i4_2_mvs_binding(
    *,
    i4_2f_closure: Mapping[str, Any],
    i4_3a_snapshot: Mapping[str, Any],
    environment: PycolmapMvsEnvironmentProbe,
) -> dict[str, Any]:
    """Read-only I4.3B smoke: prove the adapter contract binds qualified MVS evidence.

    This intentionally does not rerun PatchMatch/Stereo Fusion.  I4.2 already
    qualified the real MVS methodology; I4.3B's heavy execution surface is
    covered by the adapter/worker contracts and is re-exercised in later real
    scenario qualification.
    """

    if i4_2f_closure.get("phase") != "I4.2F" or i4_2f_closure.get("status") != "pass":
        raise ReconstructionValidationError("I4.3B binding smoke requires passing I4.2F closure evidence.")
    decision = i4_2f_closure.get("decision")
    if not isinstance(decision, Mapping) or decision.get("i4_3_provider_architecture_authorized") is not True:
        raise ReconstructionValidationError("I4.3B binding smoke requires I4.3 authorization from I4.2F.")
    if i4_3a_snapshot.get("phase") != "I4.3A" or i4_3a_snapshot.get("status") != "pass":
        raise ReconstructionValidationError("I4.3B binding smoke requires passing I4.3A snapshot evidence.")
    architecture = i4_3a_snapshot.get("architecture")
    if not isinstance(architecture, Mapping):
        raise ReconstructionValidationError("I4.3A architecture payload is missing.")
    if architecture.get("authoritative_camera_frame_owner") != I43A_AUTHORITATIVE_FRAME_OWNER:
        raise ReconstructionValidationError("I4.3B binding smoke refuses changed camera-frame authority.")
    if architecture.get("permanent_global_view_cap") is not None:
        raise ReconstructionValidationError("I4.3B binding smoke refuses a permanent global view cap.")
    implementations = architecture.get("provider_implementation_contracts")
    if not isinstance(implementations, list):
        raise ReconstructionValidationError("I4.3A provider implementation contracts are missing.")
    matches = [item for item in implementations if isinstance(item, Mapping) and item.get("implementation") == "colmap_mvs"]
    if len(matches) != 1 or matches[0].get("integration_stage") != "I4.3B":
        raise ReconstructionValidationError("I4.3A does not authorize the expected I4.3B COLMAP MVS binding.")

    evidence = i4_2f_closure.get("evidence")
    if not isinstance(evidence, Mapping):
        raise ReconstructionValidationError("I4.2F frozen MVS evidence is missing.")
    upstream = evidence.get("upstream")
    source_sha = evidence.get("source_artifact_sha256")
    if not isinstance(upstream, Mapping) or not isinstance(source_sha, Mapping):
        raise ReconstructionValidationError("I4.2F upstream/source-artifact evidence is incomplete.")
    i4_2d_path = Path(str(upstream.get("i4_2d_summary", ""))).expanduser().resolve()
    i4_2d = _read_json(i4_2d_path, label="I4.2D summary")
    if i4_2d.get("phase") != "I4.2D" or i4_2d.get("status") != "pass" or i4_2d.get("pass") is not True:
        raise ReconstructionValidationError("I4.3B requires the passing I4.2D raw MVS evidence.")
    authority = i4_2d.get("authority")
    if not isinstance(authority, Mapping):
        raise ReconstructionValidationError("I4.2D authority evidence is missing.")
    forbidden = (
        "sfm_rerun", "bundle_adjustment_rerun", "pose_refinement",
        "held_out_ground_truth_consumed_by_i4_2d", "production_filter_policy_changed",
        "production_manifold_applied", "dense_provider_promoted", "metric_scale_claimed",
        "measurement_authoritative", "certified_metrology", "surface_mesh_generated",
    )
    if any(bool(authority.get(name)) for name in forbidden):
        raise ReconstructionValidationError("I4.2D evidence violates I4.3B authority requirements.")
    if authority.get("single_authoritative_sfm_frame") is not True or authority.get("raw_mvs_dense_reconstruction") is not True:
        raise ReconstructionValidationError("I4.2D MVS/SfM-frame evidence is not qualified.")

    artifacts = i4_2d.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ReconstructionValidationError("I4.2D artifact evidence is missing.")
    fused = Path(str(artifacts.get("raw_fused_ply", ""))).expanduser().resolve()
    visibility = Path(f"{fused}.vis")
    if _sha256_file(fused) != str(source_sha.get("raw_fused_ply", "")):
        raise ReconstructionValidationError("I4.2D raw fused PLY changed after I4.2F evidence freeze.")
    if _sha256_file(visibility) != str(source_sha.get("native_visibility", "")):
        raise ReconstructionValidationError("I4.2D native visibility changed after I4.2F evidence freeze.")
    vertices = _ply_vertex_count(fused)
    if vertices <= 0 or _visibility_point_count(visibility) != vertices:
        raise ReconstructionValidationError("I4.3B frozen MVS PLY/VIS correspondence is invalid.")

    descriptor = colmap_mvs_provider_descriptor(environment)
    if not descriptor.available:
        raise ReconstructionValidationError(
            f"I4.3B runtime is not currently available: {descriptor.unavailable_reason}"
        )
    return {
        "schema_version": I43B_SCHEMA_VERSION,
        "phase": I43B_PHASE,
        "status": "pass",
        "pass": True,
        "provider_bound": True,
        "provider": descriptor.as_dict(),
        "frozen_mvs_evidence": {
            "raw_fused_ply": str(fused),
            "raw_fused_ply_sha256": _sha256_file(fused),
            "native_visibility": str(visibility),
            "native_visibility_sha256": _sha256_file(visibility),
            "fused_points": vertices,
        },
        "architecture": {
            "camera_pose_authority_owner": I43A_AUTHORITATIVE_FRAME_OWNER,
            "single_authoritative_sfm_frame_required": True,
            "all_registered_views_default_i4_3b": True,
            "adaptive_view_selection_deferred_to": "I4.3F",
            "permanent_global_view_cap": None,
            "progressive_status_supported": True,
        },
        "authority": {
            "backend_started": False,
            "provider_selected": False,
            "provider_promotion_performed": False,
            "production_filter_policy_changed": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
        },
        "qualification": {
            "house_positive_control_retained": True,
            "temple_ring_detail_qualification_retained": True,
            "production_operating_envelope_pending_i4_3g": True,
            "universal_provider_winner_assumed": False,
        },
        "next": "I4.3C DA3 production provider adapter. Do not promote MVS as a universal winner before I4.3G/I4.3H.",
    }


__all__ = [
    "I43B_SCHEMA_VERSION",
    "I43B_PHASE",
    "I43B_PROVIDER_ID",
    "I43B_ADAPTER_VERSION",
    "MvsProductionConfig",
    "MvsProviderRunContext",
    "MvsWorkerRunner",
    "ExternalMvsWorkerRunner",
    "MvsProductionProviderAdapter",
    "colmap_mvs_provider_descriptor",
    "validate_existing_i4_2_mvs_binding",
]

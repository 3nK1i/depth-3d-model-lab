"""Bounded COLMAP MVS execution for the I4.1.9A House positive control.

This stage consumes the already-validated A1 68-view camera bridge.  It may
undistort images, run PatchMatch Stereo, and run Stereo Fusion, but it never
runs SfM, bundle adjustment, pose refinement, calibration import, metric-scale
estimation, or mesh generation.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from math import isfinite
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping, Sequence

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.mvs_environment import (
    MVS_HOUSE_DEFAULT_MIN_AVAILABLE_RAM_GIB,
    MVS_HOUSE_DEFAULT_MIN_FREE_DISK_GIB,
    MVS_HOUSE_DEFAULT_MIN_FREE_VRAM_GIB,
    MvsHostResourceProbe,
    probe_mvs_host_resources,
    probe_pycolmap_mvs_environment,
)

MVS_EXECUTION_SCHEMA_VERSION = 1
MVS_HOUSE_EXPECTED_REGISTERED_VIEWS = 68
MVS_HOUSE_DEFAULT_SOURCE_NEIGHBORS = 10
MVS_HOUSE_DEFAULT_PATCH_MATCH_CACHE_GIB = 2.0
MVS_HOUSE_DEFAULT_STEREO_FUSION_CACHE_GIB = 2.0
MVS_HOUSE_DEFAULT_GPU_INDEX = "0"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError(f"Could not read A2 JSON evidence {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"A2 JSON evidence must contain an object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_finite(value: float, *, field_name: str) -> float:
    number = float(value)
    if not isfinite(number) or number <= 0.0:
        raise ReconstructionValidationError(f"{field_name} must be finite and positive.")
    return number


def _require_clean_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise ReconstructionValidationError(
            f"A2 output directory must be empty or absent; refusing to overwrite MVS evidence: {path}"
        )


@dataclass(frozen=True, slots=True)
class MvsExecutionConfig:
    source_neighbors: int = MVS_HOUSE_DEFAULT_SOURCE_NEIGHBORS
    patch_match_cache_gib: float = MVS_HOUSE_DEFAULT_PATCH_MATCH_CACHE_GIB
    stereo_fusion_cache_gib: float = MVS_HOUSE_DEFAULT_STEREO_FUSION_CACHE_GIB
    gpu_index: str = MVS_HOUSE_DEFAULT_GPU_INDEX
    max_image_size: int = -1
    geometric_consistency: bool = True

    def __post_init__(self) -> None:
        neighbors = int(self.source_neighbors)
        if neighbors < 1 or neighbors >= MVS_HOUSE_EXPECTED_REGISTERED_VIEWS:
            raise ReconstructionValidationError(
                f"source_neighbors must be between 1 and {MVS_HOUSE_EXPECTED_REGISTERED_VIEWS - 1}."
            )
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
            raise ReconstructionValidationError("gpu_index must contain one or more comma-separated non-negative indices.")
        object.__setattr__(self, "gpu_index", gpu_index)
        maximum = int(self.max_image_size)
        if maximum == 0 or maximum < -1:
            raise ReconstructionValidationError("max_image_size must be -1 or a positive integer.")
        object.__setattr__(self, "max_image_size", maximum)
        if self.geometric_consistency is not True:
            raise ReconstructionValidationError(
                "The first House MVS positive control requires geometric consistency; do not weaken this gate."
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_neighbors": self.source_neighbors,
            "source_selection": f"__auto__, {self.source_neighbors}",
            "all_reference_views_available": MVS_HOUSE_EXPECTED_REGISTERED_VIEWS,
            "patch_match_cache_gib": self.patch_match_cache_gib,
            "stereo_fusion_cache_gib": self.stereo_fusion_cache_gib,
            "gpu_index": self.gpu_index,
            "max_image_size": self.max_image_size,
            "geometric_consistency": self.geometric_consistency,
            "quality_thresholds_relaxed": False,
            "permanent_global_view_cap": False,
        }


@dataclass(frozen=True, slots=True)
class A1MvsAuthority:
    workspace: Path
    images: Path
    sparse: Path
    summary: Path
    manifest: Path
    image_names: tuple[str, ...]
    camera_intrinsics_fingerprint: str
    camera_pose_fingerprint_68: str


def load_a1_mvs_authority(a1_dir: str | Path) -> A1MvsAuthority:
    root = Path(a1_dir).expanduser().resolve()
    if not root.is_dir():
        raise ReconstructionValidationError(f"A1 camera-bridge workspace is missing: {root}")
    summary_path = root / "i4_1_9a_a1_summary.json"
    manifest_path = root / "evidence/a1_camera_authority_manifest.json"
    summary = _read_json(summary_path)
    manifest = _read_json(manifest_path)
    if summary.get("phase") != "I4.1.9A-A1" or summary.get("status") != "pass":
        raise ReconstructionValidationError("A2 requires an A1 PASS summary.")
    if manifest.get("phase") != "I4.1.9A-A1":
        raise ReconstructionValidationError("A1 authority manifest phase is invalid.")
    counts = summary.get("counts", {})
    if not isinstance(counts, Mapping) or int(counts.get("total_registered_views", -1)) != MVS_HOUSE_EXPECTED_REGISTERED_VIEWS:
        raise ReconstructionValidationError("A2 requires exactly 68 A1 registered views.")
    authority = summary.get("authority", {})
    if not isinstance(authority, Mapping):
        raise ReconstructionValidationError("A1 authority payload is missing.")
    forbidden_true = (
        "sfm_rerun",
        "bundle_adjustment_rerun",
        "pose_refinement",
        "camera_geometry_modified",
        "camera_intrinsics_modified",
        "house_k_txt_consumed",
        "metric_scale_claimed",
        "measurement_authoritative",
        "certified_metrology",
        "surface_mesh_generated",
        "heavy_mvs_started",
        "image_undistortion_started",
        "patch_match_started",
        "stereo_fusion_started",
    )
    if any(bool(authority.get(key)) for key in forbidden_true):
        raise ReconstructionValidationError("A1 authority evidence is inconsistent with a pre-MVS bridge.")
    if authority.get("single_authoritative_sfm_frame") is not True:
        raise ReconstructionValidationError("A2 requires the single authoritative SfM frame lock.")

    images = root / "images"
    sparse = root / "sparse"
    if not images.is_dir() or not sparse.is_dir():
        raise ReconstructionValidationError("A1 staged images or sparse model are missing.")
    for filename in ("cameras.txt", "images.txt", "points3D.txt"):
        if not (sparse / filename).is_file():
            raise ReconstructionValidationError(f"A1 sparse model is incomplete: {filename}")

    views = manifest.get("views", [])
    if not isinstance(views, Sequence) or isinstance(views, (str, bytes, bytearray)) or len(views) != MVS_HOUSE_EXPECTED_REGISTERED_VIEWS:
        raise ReconstructionValidationError("A1 authority manifest must describe all 68 views.")
    ordered = sorted(views, key=lambda item: int(item["source_sequence_index"]))
    names = tuple(str(item["image_name"]) for item in ordered)
    if len(set(names)) != MVS_HOUSE_EXPECTED_REGISTERED_VIEWS:
        raise ReconstructionValidationError("A1 image names must be unique across all 68 views.")
    for item in ordered:
        name = str(item["image_name"])
        path = images / name
        if not path.is_file():
            raise ReconstructionValidationError(f"A1 staged image disappeared: {name}")
        expected = str(item.get("staged_sha256", ""))
        if not expected or sha256_file(path) != expected:
            raise ReconstructionValidationError(f"A1 staged image checksum changed: {name}")

    intrinsics = str(manifest.get("camera_intrinsics_fingerprint", "")).strip()
    poses = str(manifest.get("camera_pose_fingerprint_68", "")).strip()
    if len(intrinsics) != 64 or len(poses) != 64:
        raise ReconstructionValidationError("A1 camera authority fingerprints are missing or malformed.")
    return A1MvsAuthority(
        workspace=root,
        images=images,
        sparse=sparse,
        summary=summary_path,
        manifest=manifest_path,
        image_names=names,
        camera_intrinsics_fingerprint=intrinsics,
        camera_pose_fingerprint_68=poses,
    )


def _validate_resource_launch(
    resources: MvsHostResourceProbe,
    config: MvsExecutionConfig,
    *,
    min_free_disk_gib: float,
    min_available_ram_gib: float,
    min_free_vram_gib: float,
) -> dict[str, bool]:
    disk_min = _positive_finite(min_free_disk_gib, field_name="min_free_disk_gib")
    ram_min = max(
        _positive_finite(min_available_ram_gib, field_name="min_available_ram_gib"),
        config.patch_match_cache_gib + 2.0,
        config.stereo_fusion_cache_gib + 2.0,
    )
    vram_min = _positive_finite(min_free_vram_gib, field_name="min_free_vram_gib")
    gpu_by_index = {gpu.index: gpu for gpu in resources.gpus}
    requested = {int(part.strip()) for part in config.gpu_index.split(",")}
    requested_visible = requested.issubset(gpu_by_index)
    requested_vram_ready = requested_visible and all(
        gpu_by_index[index].free_vram_gib >= vram_min for index in requested
    )
    checks = {
        "disk_ready": resources.disk_free_gib >= disk_min,
        "ram_ready": resources.ram_available_gib >= ram_min,
        "vram_ready": requested_vram_ready,
        "requested_gpu_visible": requested_visible,
    }
    checks["ready"] = all(checks.values())
    return checks


def build_house_mvs_execution_request(
    *,
    a1: A1MvsAuthority,
    output_dir: Path,
    config: MvsExecutionConfig,
    resources: MvsHostResourceProbe,
) -> dict[str, Any]:
    sparse_hashes = {
        name: sha256_file(a1.sparse / name)
        for name in ("cameras.txt", "images.txt", "points3D.txt")
    }
    return {
        "schema_version": MVS_EXECUTION_SCHEMA_VERSION,
        "phase": "I4.1.9A-A2",
        "action": "house_mvs_positive_control",
        "a1_workspace": str(a1.workspace),
        "input_images": str(a1.images),
        "input_sparse": str(a1.sparse),
        "image_names": list(a1.image_names),
        "expected_registered_views": MVS_HOUSE_EXPECTED_REGISTERED_VIEWS,
        "output_workspace": str(output_dir / "dense"),
        "fused_ply": str(output_dir / "dense/fused.ply"),
        "camera_intrinsics_fingerprint": a1.camera_intrinsics_fingerprint,
        "camera_pose_fingerprint_68": a1.camera_pose_fingerprint_68,
        "input_sparse_sha256": sparse_hashes,
        "config": config.as_dict(),
        "launch_resources": resources.as_dict(),
        "authority": {
            "single_authoritative_sfm_frame": True,
            "sfm_rerun": False,
            "bundle_adjustment_rerun": False,
            "pose_refinement": False,
            "house_k_txt_consumed": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
        },
    }


def run_house_mvs_execution(
    *,
    a1_dir: str | Path,
    output_dir: str | Path,
    pycolmap_python: str | Path,
    worker_script: str | Path,
    config: MvsExecutionConfig = MvsExecutionConfig(),
    min_free_disk_gib: float = MVS_HOUSE_DEFAULT_MIN_FREE_DISK_GIB,
    min_available_ram_gib: float = MVS_HOUSE_DEFAULT_MIN_AVAILABLE_RAM_GIB,
    min_free_vram_gib: float = MVS_HOUSE_DEFAULT_MIN_FREE_VRAM_GIB,
    worker_timeout_s: float = 21600.0,
) -> dict[str, Any]:
    """Run one immutable A2 MVS attempt and persist success/failure evidence."""

    output = Path(output_dir).expanduser().resolve()
    _require_clean_output(output)
    a1 = load_a1_mvs_authority(a1_dir)
    executable = Path(pycolmap_python).expanduser().resolve()
    worker = Path(worker_script).expanduser().resolve()
    if not executable.is_file():
        raise ReconstructionValidationError(f"PyCOLMAP Python executable is missing: {executable}")
    if not worker.is_file():
        raise ReconstructionValidationError(f"A2 MVS worker script is missing: {worker}")
    timeout = _positive_finite(worker_timeout_s, field_name="worker_timeout_s")

    environment = probe_pycolmap_mvs_environment(str(executable))
    if not (
        environment.mvs_api_ready
        and environment.patchmatch_cuda_ready
        and environment.recommended_version
        and environment.cuda_distribution
    ):
        raise ReconstructionValidationError(f"A2 PyCOLMAP MVS runtime is not qualified: {environment.as_dict()}")
    resources = probe_mvs_host_resources(output)
    launch_checks = _validate_resource_launch(
        resources,
        config,
        min_free_disk_gib=min_free_disk_gib,
        min_available_ram_gib=min_available_ram_gib,
        min_free_vram_gib=min_free_vram_gib,
    )
    if not launch_checks["ready"]:
        raise ReconstructionValidationError(
            f"A2 resource launch guardrails are not satisfied: checks={launch_checks} resources={resources.as_dict()}"
        )

    output.mkdir(parents=True, exist_ok=True)
    evidence_dir = output / "evidence"
    worker_dir = output / "worker"
    frozen_dir = evidence_dir / "frozen_authority"
    frozen_dir.mkdir(parents=True, exist_ok=False)
    worker_dir.mkdir(parents=True, exist_ok=False)

    frozen_summary = shutil.copy2(a1.summary, frozen_dir / "i4_1_9a_a1_summary.json")
    frozen_manifest = shutil.copy2(a1.manifest, frozen_dir / "a1_camera_authority_manifest.json")
    request = build_house_mvs_execution_request(a1=a1, output_dir=output, config=config, resources=resources)
    request["runtime"] = environment.as_dict()
    request["launch_checks"] = launch_checks
    request["frozen_authority_sha256"] = {
        "a1_summary": sha256_file(Path(frozen_summary)),
        "a1_manifest": sha256_file(Path(frozen_manifest)),
    }
    request_path = _write_json(worker_dir / "request.json", request)
    result_path = worker_dir / "result.json"
    state_path = worker_dir / "execution_state.json"

    worker_error = ""
    return_code = -1
    try:
        completed = subprocess.run(
            [
                str(executable),
                str(worker),
                "--request",
                str(request_path),
                "--result",
                str(result_path),
                "--state",
                str(state_path),
            ],
            check=False,
            timeout=timeout,
        )
        return_code = int(completed.returncode)
        if return_code != 0:
            worker_error = f"A2 external MVS worker failed with code {return_code}."
    except subprocess.TimeoutExpired as exc:
        worker_error = f"A2 external MVS worker timed out after {timeout:.3f}s."
        return_code = 124
        timeout_exc: Exception | None = exc
    else:
        timeout_exc = None

    result = _read_json(result_path) if result_path.is_file() else {}
    state = _read_json(state_path) if state_path.is_file() else {}
    success = return_code == 0 and result.get("status") == "pass"
    if not success and not worker_error:
        worker_error = str(result.get("error", "A2 worker did not report PASS."))

    post_sparse_hashes = {
        name: sha256_file(a1.sparse / name)
        for name in ("cameras.txt", "images.txt", "points3D.txt")
    }
    sparse_unchanged = post_sparse_hashes == request["input_sparse_sha256"]
    if not sparse_unchanged:
        success = False
        worker_error = "A1 authoritative sparse input changed during A2 execution."

    summary = {
        "schema_version": MVS_EXECUTION_SCHEMA_VERSION,
        "phase": "I4.1.9A-A2",
        "status": "pass" if success else "fail",
        "error": worker_error,
        "return_code": return_code,
        "config": config.as_dict(),
        "runtime": environment.as_dict(),
        "launch_resources": resources.as_dict(),
        "launch_checks": launch_checks,
        "a1_authority": {
            "camera_intrinsics_fingerprint": a1.camera_intrinsics_fingerprint,
            "camera_pose_fingerprint_68": a1.camera_pose_fingerprint_68,
            "input_sparse_unchanged": sparse_unchanged,
            "input_sparse_sha256_before": request["input_sparse_sha256"],
            "input_sparse_sha256_after": post_sparse_hashes,
        },
        "worker": result,
        "last_execution_state": state,
        "artifacts": {
            "workspace": str(output),
            "dense_workspace": str(output / "dense"),
            "fused_ply": str(output / "dense/fused.ply"),
            "request": str(request_path),
            "result": str(result_path),
            "state": str(state_path),
        },
        "authority": {
            "single_authoritative_sfm_frame": True,
            "sfm_rerun": False,
            "bundle_adjustment_rerun": False,
            "pose_refinement": False,
            "authoritative_camera_geometry_modified": False,
            "dense_workspace_camera_representation_transformed_by_undistortion": bool(
                result.get("image_undistortion_completed", False)
            ),
            "house_k_txt_consumed": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
            "heavy_mvs_started": bool(result.get("heavy_mvs_started", False) or state.get("heavy_mvs_started", False)),
            "image_undistortion_started": bool(result.get("image_undistortion_started", False) or state.get("image_undistortion_started", False)),
            "patch_match_started": bool(result.get("patch_match_started", False) or state.get("patch_match_started", False)),
            "stereo_fusion_started": bool(result.get("stereo_fusion_started", False) or state.get("stereo_fusion_started", False)),
        },
        "next": (
            "A2 PASS authorizes visual A/B inspection of this MVS fused cloud against the existing DA3 result. "
            "It does not authorize metric scale, measurement authority, mesh generation, or camera re-estimation."
        ),
    }
    _write_json(output / "i4_1_9a_a2_summary.json", summary)
    if timeout_exc is not None:
        raise ReconstructionValidationError(worker_error) from timeout_exc
    if not success:
        raise ReconstructionValidationError(worker_error)
    return summary

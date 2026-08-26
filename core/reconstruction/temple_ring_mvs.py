"""I4.2D raw TempleRing COLMAP MVS reconstruction.

This stage consumes the unchanged I4.2B authoritative sparse reconstruction
only after I4.2C held-out trajectory validation has passed.  It runs image
undistortion, PatchMatch Stereo, and geometric Stereo Fusion for all 47
canonical TempleRing views.  It deliberately does *not* run SfM/BA, consume
held-out benchmark poses, apply the production manifold filter, or promote a
dense provider.
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

TEMPLE_RING_MVS_SCHEMA_VERSION = 1
TEMPLE_RING_MVS_PHASE = "I4.2D"
TEMPLE_RING_EXPECTED_VIEWS = 47
TEMPLE_RING_EXPECTED_NAMES = tuple(f"templeR{index:04d}.png" for index in range(1, 48))
TEMPLE_RING_DEFAULT_SOURCE_NEIGHBORS = 10
TEMPLE_RING_DEFAULT_PATCH_MATCH_CACHE_GIB = 2.0
TEMPLE_RING_DEFAULT_STEREO_FUSION_CACHE_GIB = 2.0
TEMPLE_RING_DEFAULT_GPU_INDEX = "0"
TEMPLE_RING_DEFAULT_WORKER_TIMEOUT_S = 21600.0
_AUTHORITATIVE_SPARSE_FILES = ("cameras.bin", "images.bin", "points3D.bin", "frames.bin", "rigs.bin")
_HASHED_SPARSE_FILES = ("cameras.bin", "images.bin", "points3D.bin")
_FORBIDDEN_GT_TOKENS = (
    "ground_truth_validation_manifest",
    "camera_records",
    "R_camera_from_world",
    "t_camera_from_world",
    "camera_center_world",
    "templeR_ang",
)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError(f"Could not read {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"{label} must contain a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def sha256_file(path: str | Path) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    try:
        with source.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ReconstructionValidationError(f"Could not hash I4.2D artifact {source}: {exc}") from exc
    return digest.hexdigest()


def _positive_finite(value: float, *, field_name: str) -> float:
    number = float(value)
    if not isfinite(number) or number <= 0.0:
        raise ReconstructionValidationError(f"{field_name} must be finite and positive.")
    return number


def _require_clean_output(path: str | Path) -> Path:
    output = Path(path).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ReconstructionValidationError(
            f"I4.2D output must be empty or absent; refusing to overwrite raw MVS evidence: {output}"
        )
    return output


def _require_false(mapping: Mapping[str, Any], key: str, *, label: str) -> None:
    if bool(mapping.get(key)):
        raise ReconstructionValidationError(f"I4.2D refuses {label} state {key}=True.")


def _sparse_hashes(sparse: Path, *, include_auxiliary: bool = False) -> dict[str, str]:
    required = _AUTHORITATIVE_SPARSE_FILES if include_auxiliary else _HASHED_SPARSE_FILES
    hashes: dict[str, str] = {}
    for filename in required:
        path = sparse / filename
        if not path.is_file() or path.stat().st_size <= 0:
            raise ReconstructionValidationError(f"I4.2D authoritative sparse model is incomplete: {filename}")
        hashes[filename] = sha256_file(path)
    return hashes


@dataclass(frozen=True, slots=True)
class TempleRingMvsConfig:
    source_neighbors: int = TEMPLE_RING_DEFAULT_SOURCE_NEIGHBORS
    patch_match_cache_gib: float = TEMPLE_RING_DEFAULT_PATCH_MATCH_CACHE_GIB
    stereo_fusion_cache_gib: float = TEMPLE_RING_DEFAULT_STEREO_FUSION_CACHE_GIB
    gpu_index: str = TEMPLE_RING_DEFAULT_GPU_INDEX
    max_image_size: int = -1
    geometric_consistency: bool = True

    def __post_init__(self) -> None:
        neighbors = int(self.source_neighbors)
        if neighbors < 1 or neighbors >= TEMPLE_RING_EXPECTED_VIEWS:
            raise ReconstructionValidationError(
                f"source_neighbors must be between 1 and {TEMPLE_RING_EXPECTED_VIEWS - 1}."
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
            raise ReconstructionValidationError(
                "gpu_index must contain one or more comma-separated non-negative indices."
            )
        object.__setattr__(self, "gpu_index", gpu_index)
        maximum = int(self.max_image_size)
        if maximum == 0 or maximum < -1:
            raise ReconstructionValidationError("max_image_size must be -1 or a positive integer.")
        object.__setattr__(self, "max_image_size", maximum)
        if self.geometric_consistency is not True:
            raise ReconstructionValidationError(
                "I4.2D requires geometric consistency; do not weaken the raw TempleRing MVS control."
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_neighbors": self.source_neighbors,
            "source_selection": f"__auto__, {self.source_neighbors}",
            "all_reference_views_available": TEMPLE_RING_EXPECTED_VIEWS,
            "patch_match_cache_gib": self.patch_match_cache_gib,
            "stereo_fusion_cache_gib": self.stereo_fusion_cache_gib,
            "gpu_index": self.gpu_index,
            "max_image_size": self.max_image_size,
            "geometric_consistency": self.geometric_consistency,
            "quality_thresholds_relaxed": False,
            "production_manifold_applied": False,
            "permanent_global_view_cap": False,
        }


@dataclass(frozen=True, slots=True)
class TempleRingMvsInputs:
    i4_2b_root: Path
    i4_2c_root: Path
    i4_2b_summary: Path
    i4_2c_summary: Path
    i4_2b_worker_request: Path
    images: Path
    authoritative_sparse: Path
    image_names: tuple[str, ...]
    image_sha256: tuple[tuple[str, str], ...]
    authoritative_sparse_sha256: tuple[tuple[str, str], ...]
    i4_2b_summary_sha256: str
    i4_2c_summary_sha256: str

    @property
    def image_hash_map(self) -> dict[str, str]:
        return dict(self.image_sha256)

    @property
    def sparse_hash_map(self) -> dict[str, str]:
        return dict(self.authoritative_sparse_sha256)


def load_i4_2d_inputs(*, i4_2b_dir: str | Path, i4_2c_dir: str | Path) -> TempleRingMvsInputs:
    b_root = Path(i4_2b_dir).expanduser().resolve()
    c_root = Path(i4_2c_dir).expanduser().resolve()
    if not b_root.is_dir() or not c_root.is_dir():
        raise ReconstructionValidationError("I4.2D requires existing I4.2B and I4.2C evidence directories.")

    b_summary_path = b_root / "i4_2b_summary.json"
    c_summary_path = c_root / "i4_2c_summary.json"
    b_request_path = b_root / "worker/request.json"
    b_summary = _read_json(b_summary_path, label="I4.2B summary")
    c_summary = _read_json(c_summary_path, label="I4.2C summary")
    b_request = _read_json(b_request_path, label="I4.2B worker request")

    if b_summary.get("phase") != "I4.2B" or b_summary.get("status") != "pass" or b_summary.get("pass") is not True:
        raise ReconstructionValidationError("I4.2D requires I4.2B PASS.")
    validation = b_summary.get("validation")
    authority_b = b_summary.get("authority")
    if not isinstance(validation, Mapping) or not isinstance(authority_b, Mapping):
        raise ReconstructionValidationError("I4.2B validation/authority evidence is missing.")
    if validation.get("all_47_registered") is not True or int(validation.get("registered_images", -1)) != 47:
        raise ReconstructionValidationError("I4.2D requires the complete 47-view I4.2B SfM model.")
    if validation.get("fixed_intrinsics_preserved") is not True or validation.get("single_shared_camera") is not True:
        raise ReconstructionValidationError("I4.2D requires preserved shared TempleRing intrinsics.")
    if authority_b.get("single_authoritative_sfm_frame") is not True or authority_b.get("intrinsics_fixed") is not True:
        raise ReconstructionValidationError("I4.2D requires the locked I4.2B authoritative SfM frame.")
    for key in (
        "ground_truth_extrinsics_consumed",
        "ground_truth_camera_centers_consumed",
        "ground_truth_angles_consumed",
        "mvs_started",
        "da3_started",
        "production_filter_policy_changed",
        "dense_provider_promoted",
        "i4_3_provider_architecture_authorized",
        "metric_scale_claimed",
        "measurement_authoritative",
        "certified_metrology",
        "surface_mesh_generated",
    ):
        _require_false(authority_b, key, label="I4.2B")

    if c_summary.get("phase") != "I4.2C" or c_summary.get("status") != "pass" or c_summary.get("pass") is not True:
        raise ReconstructionValidationError("I4.2D requires I4.2C PASS.")
    trajectory = c_summary.get("trajectory_quality")
    authority_c = c_summary.get("authority")
    evidence_c = c_summary.get("evidence")
    if not isinstance(trajectory, Mapping) or trajectory.get("accepted") is not True:
        raise ReconstructionValidationError("I4.2D requires accepted held-out trajectory quality.")
    if not isinstance(authority_c, Mapping) or not isinstance(evidence_c, Mapping):
        raise ReconstructionValidationError("I4.2C authority/evidence is missing.")
    if authority_c.get("authoritative_sparse_modified") is not False:
        raise ReconstructionValidationError("I4.2D refuses a modified I4.2B sparse model.")
    if authority_c.get("single_authoritative_sfm_frame_preserved") is not True:
        raise ReconstructionValidationError("I4.2D requires the I4.2C authoritative-frame preservation lock.")
    for key in (
        "bundle_adjustment_rerun",
        "da3_started",
        "dense_provider_promoted",
        "i4_3_provider_architecture_authorized",
        "measurement_authoritative",
        "metric_scale_claimed",
        "mvs_started",
        "pose_refinement",
        "production_filter_policy_changed",
        "sfm_rerun",
        "similarity_applied_to_authoritative_sparse_model",
        "surface_mesh_generated",
    ):
        _require_false(authority_c, key, label="I4.2C")

    b_summary_hash = sha256_file(b_summary_path)
    source_sha = evidence_c.get("source_sha256")
    if not isinstance(source_sha, Mapping) or str(source_sha.get("i4_2b_summary", "")) != b_summary_hash:
        raise ReconstructionValidationError("I4.2C does not checksum-bind to this exact I4.2B summary.")

    artifacts = b_summary.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ReconstructionValidationError("I4.2B artifact paths are missing.")
    images = Path(str(artifacts.get("images", ""))).expanduser().resolve()
    sparse = Path(str(artifacts.get("authoritative_sparse", ""))).expanduser().resolve()
    if not images.is_dir() or not sparse.is_dir():
        raise ReconstructionValidationError("I4.2B staged images or authoritative sparse model disappeared.")

    expected_hashes = validation.get("authoritative_sparse_sha256")
    if not isinstance(expected_hashes, Mapping):
        raise ReconstructionValidationError("I4.2B authoritative sparse hashes are missing.")
    actual_primary_hashes = _sparse_hashes(sparse)
    expected_primary_hashes = {name: str(expected_hashes.get(name, "")) for name in _HASHED_SPARSE_FILES}
    if actual_primary_hashes != expected_primary_hashes:
        raise ReconstructionValidationError("I4.2B authoritative sparse binaries changed before I4.2D.")
    c_before = evidence_c.get("authoritative_sparse_sha256_before")
    c_after = evidence_c.get("authoritative_sparse_sha256_after")
    if c_before != expected_primary_hashes or c_after != expected_primary_hashes:
        raise ReconstructionValidationError("I4.2C sparse hash evidence does not match I4.2B.")
    all_sparse_hashes = _sparse_hashes(sparse, include_auxiliary=True)

    if b_request.get("phase") != "I4.2B" or b_request.get("action") != "authoritative_temple_ring_sfm":
        raise ReconstructionValidationError("I4.2D I4.2B worker request identity is invalid.")
    image_names_raw = b_request.get("image_names")
    image_hashes_raw = b_request.get("image_sha256")
    if not isinstance(image_names_raw, Sequence) or isinstance(image_names_raw, (str, bytes, bytearray)):
        raise ReconstructionValidationError("I4.2B image-name evidence is malformed.")
    image_names = tuple(str(name) for name in image_names_raw)
    if image_names != TEMPLE_RING_EXPECTED_NAMES:
        raise ReconstructionValidationError("I4.2D requires canonical TempleRing image order 1..47.")
    if not isinstance(image_hashes_raw, Mapping) or set(image_hashes_raw) != set(TEMPLE_RING_EXPECTED_NAMES):
        raise ReconstructionValidationError("I4.2B image checksum evidence is incomplete.")
    image_hashes: list[tuple[str, str]] = []
    for name in TEMPLE_RING_EXPECTED_NAMES:
        path = images / name
        expected = str(image_hashes_raw.get(name, ""))
        if len(expected) != 64 or not path.is_file() or sha256_file(path) != expected:
            raise ReconstructionValidationError(f"I4.2B staged image changed before I4.2D: {name}")
        image_hashes.append((name, expected))

    serialized_request = json.dumps(b_request, sort_keys=True)
    if any(token in serialized_request for token in _FORBIDDEN_GT_TOKENS):
        raise ReconstructionValidationError("I4.2B worker request unexpectedly contains held-out GT payload.")

    return TempleRingMvsInputs(
        i4_2b_root=b_root,
        i4_2c_root=c_root,
        i4_2b_summary=b_summary_path,
        i4_2c_summary=c_summary_path,
        i4_2b_worker_request=b_request_path,
        images=images,
        authoritative_sparse=sparse,
        image_names=image_names,
        image_sha256=tuple(image_hashes),
        authoritative_sparse_sha256=tuple(sorted(all_sparse_hashes.items())),
        i4_2b_summary_sha256=b_summary_hash,
        i4_2c_summary_sha256=sha256_file(c_summary_path),
    )


def _validate_resource_launch(
    resources: MvsHostResourceProbe,
    config: TempleRingMvsConfig,
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


def build_temple_ring_mvs_request(
    *,
    inputs: TempleRingMvsInputs,
    output_dir: Path,
    config: TempleRingMvsConfig,
    resources: MvsHostResourceProbe,
) -> dict[str, Any]:
    payload = {
        "schema_version": TEMPLE_RING_MVS_SCHEMA_VERSION,
        "phase": TEMPLE_RING_MVS_PHASE,
        "action": "temple_ring_raw_mvs_dense_reconstruction",
        "input_images": str(inputs.images),
        "input_sparse": str(inputs.authoritative_sparse),
        "image_names": list(inputs.image_names),
        "image_sha256": inputs.image_hash_map,
        "expected_registered_views": TEMPLE_RING_EXPECTED_VIEWS,
        "output_workspace": str(output_dir / "dense"),
        "fused_ply": str(output_dir / "dense/fused.ply"),
        "input_sparse_sha256": inputs.sparse_hash_map,
        "config": config.as_dict(),
        "launch_resources": resources.as_dict(),
        "authority": {
            "single_authoritative_sfm_frame": True,
            "sfm_rerun": False,
            "bundle_adjustment_rerun": False,
            "pose_refinement": False,
            "held_out_ground_truth_consumed": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
            "production_filter_policy_changed": False,
            "production_manifold_applied": False,
            "dense_provider_promoted": False,
            "i4_3_provider_architecture_authorized": False,
        },
    }
    serialized = json.dumps(payload, sort_keys=True)
    for token in _FORBIDDEN_GT_TOKENS:
        if token in serialized:
            raise ReconstructionValidationError(f"I4.2D worker request leaked held-out GT token: {token}")
    return payload


def _validate_worker_result(result: Mapping[str, Any], *, inputs: TempleRingMvsInputs) -> None:
    if result.get("phase") != TEMPLE_RING_MVS_PHASE or result.get("status") != "pass":
        raise ReconstructionValidationError(
            f"I4.2D worker did not return a valid PASS result: {result.get('error', 'unknown error')}"
        )
    for key in (
        "sfm_rerun",
        "bundle_adjustment_rerun",
        "pose_refinement",
        "held_out_ground_truth_consumed",
        "metric_scale_claimed",
        "measurement_authoritative",
        "certified_metrology",
        "surface_mesh_generated",
        "production_filter_policy_changed",
        "production_manifold_applied",
        "dense_provider_promoted",
        "i4_3_provider_architecture_authorized",
    ):
        if bool(result.get(key)):
            raise ReconstructionValidationError(f"I4.2D worker violated locked authority flag {key}=True.")
    for key in (
        "heavy_mvs_started",
        "image_undistortion_started",
        "image_undistortion_completed",
        "patch_match_started",
        "patch_match_completed",
        "stereo_fusion_started",
        "stereo_fusion_completed",
    ):
        if result.get(key) is not True:
            raise ReconstructionValidationError(f"I4.2D worker did not complete required stage flag {key}.")

    dense_validation = result.get("dense_sparse_validation")
    cfg_validation = result.get("patch_match_config_validation")
    patch_outputs = result.get("patch_match_outputs")
    fusion = result.get("fusion")
    if not isinstance(dense_validation, Mapping) or dense_validation.get("pass") is not True:
        raise ReconstructionValidationError("I4.2D worker dense-sparse authority validation did not pass.")
    if not isinstance(cfg_validation, Mapping) or cfg_validation.get("pass") is not True:
        raise ReconstructionValidationError("I4.2D worker PatchMatch reference/source validation did not pass.")
    if not isinstance(patch_outputs, Mapping):
        raise ReconstructionValidationError("I4.2D worker PatchMatch output evidence is missing.")
    if int(patch_outputs.get("geometric_depth_maps", -1)) != TEMPLE_RING_EXPECTED_VIEWS:
        raise ReconstructionValidationError("I4.2D worker did not produce 47 geometric depth maps.")
    if int(patch_outputs.get("geometric_normal_maps", -1)) != TEMPLE_RING_EXPECTED_VIEWS:
        raise ReconstructionValidationError("I4.2D worker did not produce 47 geometric normal maps.")
    if not isinstance(fusion, Mapping) or int(fusion.get("ply_vertex_count", 0)) <= 0:
        raise ReconstructionValidationError("I4.2D worker fused PLY is empty or missing.")
    if fusion.get("input_type") != "geometric" or fusion.get("production_manifold_applied") is not False:
        raise ReconstructionValidationError("I4.2D worker fusion is not the locked raw geometric MVS result.")
    if result.get("detail_qualification_pending") is not True:
        raise ReconstructionValidationError("I4.2D worker must leave detail qualification pending for I4.2E.")
    worker_sparse = result.get("authoritative_sparse_sha256_after")
    if worker_sparse != inputs.sparse_hash_map:
        raise ReconstructionValidationError("I4.2D worker sparse checksum evidence does not match the frozen input.")


def run_temple_ring_mvs(
    *,
    i4_2b_dir: str | Path,
    i4_2c_dir: str | Path,
    output_dir: str | Path,
    pycolmap_python: str | Path,
    worker_script: str | Path,
    config: TempleRingMvsConfig = TempleRingMvsConfig(),
    min_free_disk_gib: float = MVS_HOUSE_DEFAULT_MIN_FREE_DISK_GIB,
    min_available_ram_gib: float = MVS_HOUSE_DEFAULT_MIN_AVAILABLE_RAM_GIB,
    min_free_vram_gib: float = MVS_HOUSE_DEFAULT_MIN_FREE_VRAM_GIB,
    worker_timeout_s: float = TEMPLE_RING_DEFAULT_WORKER_TIMEOUT_S,
) -> dict[str, Any]:
    """Run one immutable raw TempleRing MVS attempt after I4.2C PASS."""

    output = _require_clean_output(output_dir)
    inputs = load_i4_2d_inputs(i4_2b_dir=i4_2b_dir, i4_2c_dir=i4_2c_dir)
    executable = Path(pycolmap_python).expanduser().resolve()
    worker = Path(worker_script).expanduser().resolve()
    if not executable.is_file():
        raise ReconstructionValidationError(f"I4.2D PyCOLMAP Python executable is missing: {executable}")
    if not worker.is_file():
        raise ReconstructionValidationError(f"I4.2D MVS worker script is missing: {worker}")
    timeout = _positive_finite(worker_timeout_s, field_name="worker_timeout_s")

    environment = probe_pycolmap_mvs_environment(str(executable))
    if not (
        environment.mvs_api_ready
        and environment.patchmatch_cuda_ready
        and environment.recommended_version
        and environment.cuda_distribution
    ):
        raise ReconstructionValidationError(f"I4.2D PyCOLMAP MVS runtime is not qualified: {environment.as_dict()}")
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
            f"I4.2D resource launch guardrails are not satisfied: checks={launch_checks} resources={resources.as_dict()}"
        )

    output.mkdir(parents=True, exist_ok=True)
    evidence_dir = output / "evidence"
    worker_dir = output / "worker"
    evidence_dir.mkdir(parents=True, exist_ok=False)
    worker_dir.mkdir(parents=True, exist_ok=False)

    frozen_b = shutil.copy2(inputs.i4_2b_summary, evidence_dir / "i4_2b_summary.json")
    frozen_c = shutil.copy2(inputs.i4_2c_summary, evidence_dir / "i4_2c_summary.json")
    frozen_b_request = shutil.copy2(inputs.i4_2b_worker_request, evidence_dir / "i4_2b_worker_request.json")

    request = build_temple_ring_mvs_request(inputs=inputs, output_dir=output, config=config, resources=resources)
    request["runtime"] = environment.as_dict()
    request["launch_checks"] = launch_checks
    request["frozen_evidence_sha256"] = {
        "i4_2b_summary": sha256_file(frozen_b),
        "i4_2c_summary": sha256_file(frozen_c),
        "i4_2b_worker_request": sha256_file(frozen_b_request),
    }
    request_path = _write_json(worker_dir / "request.json", request)
    result_path = worker_dir / "result.json"
    state_path = worker_dir / "execution_state.json"

    return_code = -1
    worker_error = ""
    timeout_exc: Exception | None = None
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
            worker_error = f"I4.2D external MVS worker failed with code {return_code}."
    except subprocess.TimeoutExpired as exc:
        worker_error = f"I4.2D external MVS worker timed out after {timeout:.3f}s."
        return_code = 124
        timeout_exc = exc
    except OSError as exc:
        worker_error = f"I4.2D external MVS worker could not start: {exc}"

    result = _read_json(result_path, label="I4.2D worker result") if result_path.is_file() else {}
    state = _read_json(state_path, label="I4.2D execution state") if state_path.is_file() else {}
    success = False
    if return_code == 0 and result:
        try:
            _validate_worker_result(result, inputs=inputs)
        except ReconstructionValidationError as exc:
            worker_error = str(exc)
        else:
            success = True
    if not success and not worker_error:
        worker_error = str(result.get("error", "I4.2D worker did not report PASS."))

    post_sparse_hashes = _sparse_hashes(inputs.authoritative_sparse, include_auxiliary=True)
    sparse_unchanged = post_sparse_hashes == inputs.sparse_hash_map
    if not sparse_unchanged:
        success = False
        worker_error = "I4.2B authoritative sparse input changed during I4.2D execution."
    post_image_hashes = {name: sha256_file(inputs.images / name) for name in inputs.image_names}
    images_unchanged = post_image_hashes == inputs.image_hash_map
    if not images_unchanged:
        success = False
        worker_error = "I4.2B staged images changed during I4.2D execution."

    summary = {
        "schema_version": TEMPLE_RING_MVS_SCHEMA_VERSION,
        "phase": TEMPLE_RING_MVS_PHASE,
        "status": "pass" if success else "fail",
        "pass": bool(success),
        "error": worker_error,
        "return_code": return_code,
        "dataset": {
            "identity": "Middlebury_MVS_TempleRing",
            "reference_views": TEMPLE_RING_EXPECTED_VIEWS,
            "image_size": [640, 480],
        },
        "config": config.as_dict(),
        "runtime": environment.as_dict(),
        "launch_resources": resources.as_dict(),
        "launch_checks": launch_checks,
        "input_authority": {
            "i4_2b_summary_sha256": inputs.i4_2b_summary_sha256,
            "i4_2c_summary_sha256": inputs.i4_2c_summary_sha256,
            "authoritative_sparse_unchanged": sparse_unchanged,
            "authoritative_sparse_sha256_before": inputs.sparse_hash_map,
            "authoritative_sparse_sha256_after": post_sparse_hashes,
            "staged_images_unchanged": images_unchanged,
            "held_out_trajectory_validation_passed": True,
        },
        "worker": result,
        "last_execution_state": state,
        "artifacts": {
            "workspace": str(output),
            "dense_workspace": str(output / "dense"),
            "raw_fused_ply": str(output / "dense/fused.ply"),
            "request": str(request_path),
            "result": str(result_path),
            "state": str(state_path),
            "frozen_i4_2b_summary": str(frozen_b),
            "frozen_i4_2c_summary": str(frozen_c),
            "frozen_i4_2b_worker_request": str(frozen_b_request),
        },
        "authority": {
            "single_authoritative_sfm_frame": True,
            "sfm_rerun": False,
            "bundle_adjustment_rerun": False,
            "pose_refinement": False,
            "authoritative_camera_geometry_modified": False,
            "held_out_ground_truth_consumed_by_i4_2d": False,
            "mvs_started": bool(result.get("heavy_mvs_started", False) or state.get("heavy_mvs_started", False)),
            "raw_mvs_dense_reconstruction": bool(success),
            "da3_started": False,
            "production_filter_policy_changed": False,
            "production_manifold_applied": False,
            "dense_provider_promoted": False,
            "i4_3_provider_architecture_authorized": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
        },
        "next": (
            "I4.2E thin-structure/detail-preservation qualification on the raw TempleRing MVS result, including a controlled "
            "comparison against the existing production d12_sparse manifold cleanup. Do not change production policy in I4.2D."
            if success else
            "I4.2D did not produce a qualified raw MVS result; inspect evidence without relaxing quality or authority gates."
        ),
    }
    _write_json(output / "i4_2d_summary.json", summary)
    if timeout_exc is not None:
        raise ReconstructionValidationError(worker_error) from timeout_exc
    if not success:
        raise ReconstructionValidationError(worker_error)
    return summary

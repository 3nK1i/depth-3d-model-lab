"""I4.2B authoritative TempleRing sparse SfM orchestration.

This benchmark stage reconstructs the canonical 47-view Middlebury TempleRing
image set with the benchmark-provided *shared intrinsics only*.  Per-view
benchmark R/t, camera centres, angles, laser ground truth, and any visual hull
remain quarantined and are forbidden worker inputs.

PyCOLMAP owns the single authoritative reconstruction frame.  Intrinsics are
held fixed during mapping/BA; extrinsics and sparse structure are estimated
from image evidence.  This stage makes no metric-scale, measurement, mesh, or
provider-promotion claim.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from math import isfinite
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Mapping, Sequence

import numpy as np

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.sfm_contracts import SfmStageStatistics
from core.reconstruction.sfm_environment import (
    PYCOLMAP_CUDA_PACKAGE,
    PYCOLMAP_RECOMMENDED_VERSION,
    probe_pycolmap_environment,
)
from core.reconstruction.sfm_quality import assess_sfm_quality

TEMPLE_RING_SFM_SCHEMA_VERSION = 1
TEMPLE_RING_SFM_PHASE = "I4.2B"
TEMPLE_RING_EXPECTED_VIEWS = 47
TEMPLE_RING_EXPECTED_WIDTH = 640
TEMPLE_RING_EXPECTED_HEIGHT = 480
TEMPLE_RING_CAMERA_MODEL = "PINHOLE"
TEMPLE_RING_MATCHING_STRATEGY = "exhaustive"
TEMPLE_RING_DEFAULT_MAX_IMAGE_SIZE = 640
TEMPLE_RING_DEFAULT_WORKER_TIMEOUT_S = 7200.0
TEMPLE_RING_EXPECTED_NAMES = tuple(f"templeR{index:04d}.png" for index in range(1, 48))
_FORBIDDEN_GT_TOKENS = (
    "R_camera_from_world",
    "t_camera_from_world",
    "camera_center_world",
    "ground_truth_validation_manifest",
    "templeR_ang.txt",
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
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def sha256_file(path: str | Path) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    try:
        with source.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ReconstructionValidationError(f"Could not hash I4.2B artifact {source}: {exc}") from exc
    return digest.hexdigest()


def _require_clean_output(path: str | Path) -> Path:
    output = Path(path).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ReconstructionValidationError(
            f"I4.2B output must be empty or absent; use a new immutable attempt directory: {output}"
        )
    return output


def _require_false(mapping: Mapping[str, Any], key: str) -> None:
    if bool(mapping.get(key)):
        raise ReconstructionValidationError(f"I4.2B refuses forbidden upstream state {key}=True.")


def _finite_positive(value: object, *, field_name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ReconstructionValidationError(f"{field_name} must be numeric.") from exc
    if not isfinite(result) or result <= 0.0:
        raise ReconstructionValidationError(f"{field_name} must be finite and positive.")
    return result


def _validate_shared_k(value: object) -> np.ndarray:
    K = np.asarray(value, dtype=np.float64)
    if K.shape != (3, 3) or np.any(~np.isfinite(K)):
        raise ReconstructionValidationError("I4.2B shared K must be a finite 3x3 matrix.")
    if K[0, 0] <= 0.0 or K[1, 1] <= 0.0:
        raise ReconstructionValidationError("I4.2B shared K focal lengths must be positive.")
    if not np.allclose(K[2], [0.0, 0.0, 1.0], atol=1e-12, rtol=0.0):
        raise ReconstructionValidationError("I4.2B shared K bottom row is invalid.")
    if abs(float(K[0, 1])) > 1e-12 or abs(float(K[1, 0])) > 1e-12:
        raise ReconstructionValidationError("I4.2B PINHOLE benchmark expects zero skew.")
    return K


@dataclass(frozen=True, slots=True)
class TempleRingSfmInputs:
    preflight_dir: Path
    preflight_summary: Path
    reconstruction_manifest: Path
    image_records: tuple[dict[str, Any], ...]
    shared_K: np.ndarray
    reconstruction_manifest_sha256: str


@dataclass(frozen=True, slots=True)
class TempleRingSfmConfig:
    use_gpu: bool = True
    maximum_image_size: int = TEMPLE_RING_DEFAULT_MAX_IMAGE_SIZE
    random_seed: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.use_gpu, bool):
            raise TypeError("use_gpu must be a bool.")
        maximum = int(self.maximum_image_size)
        if maximum != TEMPLE_RING_DEFAULT_MAX_IMAGE_SIZE:
            raise ReconstructionValidationError(
                "I4.2B is a fixed full-resolution 640px benchmark; maximum_image_size must remain 640."
            )
        object.__setattr__(self, "maximum_image_size", maximum)
        seed = int(self.random_seed)
        if seed < 0:
            raise ReconstructionValidationError("random_seed must be non-negative.")
        object.__setattr__(self, "random_seed", seed)

    def as_dict(self) -> dict[str, Any]:
        return {
            "use_gpu": self.use_gpu,
            "maximum_image_size": self.maximum_image_size,
            "matching_strategy": TEMPLE_RING_MATCHING_STRATEGY,
            "camera_mode": "single",
            "camera_model": TEMPLE_RING_CAMERA_MODEL,
            "intrinsics_fixed": True,
            "ba_refine_focal_length": False,
            "ba_refine_principal_point": False,
            "ba_refine_extra_params": False,
            "abs_pose_refine_focal_length": False,
            "abs_pose_refine_extra_params": False,
            "multiple_models": False,
            "permanent_global_view_cap": False,
            "random_seed": self.random_seed,
        }


def load_i4_2a_reconstruction_inputs(i4_2a_dir: str | Path) -> TempleRingSfmInputs:
    root = Path(i4_2a_dir).expanduser().resolve()
    if not root.is_dir():
        raise ReconstructionValidationError(f"I4.2A preflight directory is missing: {root}")
    summary_path = root / "i4_2a_summary.json"
    manifest_path = root / "i4_2a_reconstruction_input_manifest.json"
    summary = _read_json(summary_path, label="I4.2A summary")
    manifest = _read_json(manifest_path, label="I4.2A reconstruction manifest")

    if summary.get("phase") != "I4.2A" or summary.get("status") != "pass" or summary.get("pass") is not True:
        raise ReconstructionValidationError("I4.2B requires a passing I4.2A summary.")
    dataset = summary.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ReconstructionValidationError("I4.2A dataset evidence is missing.")
    required_dataset = {
        "identity": "Middlebury_MVS_TempleRing",
        "expected_views": TEMPLE_RING_EXPECTED_VIEWS,
        "validated_views": TEMPLE_RING_EXPECTED_VIEWS,
        "shared_intrinsics": True,
    }
    for key, expected in required_dataset.items():
        if dataset.get(key) != expected:
            raise ReconstructionValidationError(f"I4.2B upstream dataset field {key} is not {expected!r}.")
    if dataset.get("image_size") != [TEMPLE_RING_EXPECTED_WIDTH, TEMPLE_RING_EXPECTED_HEIGHT]:
        raise ReconstructionValidationError("I4.2B requires canonical 640x480 TempleRing imagery.")
    if list(dataset.get("cross_dataset_contamination", [])):
        raise ReconstructionValidationError("I4.2B refuses cross-dataset Temple contamination.")

    authority = summary.get("authority")
    if not isinstance(authority, Mapping):
        raise ReconstructionValidationError("I4.2A authority evidence is missing.")
    for key in (
        "sfm_started",
        "bundle_adjustment_started",
        "mvs_started",
        "da3_started",
        "production_filtering_rerun",
        "production_filter_policy_changed",
        "dense_provider_promoted",
        "i4_3_provider_architecture_authorized",
        "metric_scale_claimed",
        "measurement_authoritative",
        "certified_metrology",
        "surface_mesh_generated",
    ):
        _require_false(authority, key)

    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ReconstructionValidationError("I4.2A artifact evidence is missing.")
    expected_manifest_hash = str(artifacts.get("reconstruction_input_manifest_sha256", ""))
    actual_manifest_hash = sha256_file(manifest_path)
    if not expected_manifest_hash or actual_manifest_hash != expected_manifest_hash:
        raise ReconstructionValidationError("I4.2A reconstruction manifest checksum changed.")

    manifest_text = json.dumps(manifest, sort_keys=True)
    for token in _FORBIDDEN_GT_TOKENS:
        if token in manifest_text:
            raise ReconstructionValidationError(
                f"I4.2B reconstruction manifest contains forbidden held-out evidence token: {token}"
            )
    if manifest.get("phase") != "I4.2A" or int(manifest.get("image_count", -1)) != TEMPLE_RING_EXPECTED_VIEWS:
        raise ReconstructionValidationError("I4.2B requires the exact 47-view I4.2A reconstruction manifest.")
    for key in (
        "ground_truth_extrinsics_included",
        "ground_truth_camera_centers_included",
        "ground_truth_angles_included",
        "laser_ground_truth_consumed",
        "visual_hull_consumed",
    ):
        if manifest.get(key) is not False:
            raise ReconstructionValidationError(f"I4.2B requires {key}=False.")
    if manifest.get("all_47_views_available") is not True:
        raise ReconstructionValidationError("I4.2B requires all 47 TempleRing views.")
    if manifest.get("permanent_global_view_cap") is not False:
        raise ReconstructionValidationError("I4.2B refuses a permanent global view cap.")

    K = _validate_shared_k(manifest.get("shared_intrinsics_K"))
    images = manifest.get("images")
    if not isinstance(images, Sequence) or isinstance(images, (str, bytes, bytearray)) or len(images) != 47:
        raise ReconstructionValidationError("I4.2B requires exactly 47 image records.")
    ordered: list[dict[str, Any]] = []
    for index, raw in enumerate(images):
        if not isinstance(raw, Mapping):
            raise ReconstructionValidationError("I4.2A image record must be an object.")
        item = dict(raw)
        expected_name = TEMPLE_RING_EXPECTED_NAMES[index]
        if item.get("name") != expected_name:
            raise ReconstructionValidationError(
                f"I4.2B canonical image order changed at index {index}: expected {expected_name}."
            )
        if int(item.get("width", -1)) != 640 or int(item.get("height", -1)) != 480:
            raise ReconstructionValidationError(f"I4.2B image dimensions changed: {expected_name}")
        path = Path(str(item.get("path", ""))).expanduser().resolve()
        if not path.is_file():
            raise ReconstructionValidationError(f"I4.2B source image disappeared: {expected_name}")
        digest = str(item.get("sha256", ""))
        if len(digest) != 64 or sha256_file(path) != digest:
            raise ReconstructionValidationError(f"I4.2B source image checksum changed: {expected_name}")
        item["path"] = str(path)
        ordered.append(item)

    return TempleRingSfmInputs(
        preflight_dir=root,
        preflight_summary=summary_path,
        reconstruction_manifest=manifest_path,
        image_records=tuple(ordered),
        shared_K=K,
        reconstruction_manifest_sha256=actual_manifest_hash,
    )


def stage_temple_ring_images(inputs: TempleRingSfmInputs, images_dir: Path) -> tuple[dict[str, Any], ...]:
    images_dir.mkdir(parents=True, exist_ok=False)
    staged: list[dict[str, Any]] = []
    for item in inputs.image_records:
        source = Path(str(item["path"]))
        destination = images_dir / str(item["name"])
        shutil.copy2(source, destination)
        digest = sha256_file(destination)
        if digest != str(item["sha256"]):
            raise ReconstructionValidationError(f"I4.2B staged image checksum mismatch: {destination.name}")
        staged.append({
            "name": destination.name,
            "path": str(destination),
            "sha256": digest,
            "width": 640,
            "height": 480,
        })
    return tuple(staged)


def build_temple_ring_sfm_worker_request(
    *,
    inputs: TempleRingSfmInputs,
    workspace: Path,
    staged_images: Sequence[Mapping[str, Any]],
    config: TempleRingSfmConfig,
) -> dict[str, Any]:
    K = inputs.shared_K
    payload = {
        "schema_version": TEMPLE_RING_SFM_SCHEMA_VERSION,
        "phase": TEMPLE_RING_SFM_PHASE,
        "action": "authoritative_temple_ring_sfm",
        "workspace": str(workspace),
        "images_directory": str(workspace / "images"),
        "database_path": str(workspace / "database.db"),
        "mapping_output": str(workspace / "mapping_models"),
        "authoritative_sparse": str(workspace / "authoritative_sparse"),
        "image_names": [str(item["name"]) for item in staged_images],
        "image_sha256": {str(item["name"]): str(item["sha256"]) for item in staged_images},
        "shared_intrinsics": {
            "model": TEMPLE_RING_CAMERA_MODEL,
            "width": TEMPLE_RING_EXPECTED_WIDTH,
            "height": TEMPLE_RING_EXPECTED_HEIGHT,
            "fx": float(K[0, 0]),
            "fy": float(K[1, 1]),
            "cx": float(K[0, 2]),
            "cy": float(K[1, 2]),
        },
        "config": config.as_dict(),
        "authority": {
            "single_authoritative_sfm_frame": True,
            "ground_truth_extrinsics_consumed": False,
            "ground_truth_camera_centers_consumed": False,
            "ground_truth_angles_consumed": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
            "mvs_started": False,
            "da3_started": False,
            "dense_provider_promoted": False,
            "i4_3_provider_architecture_authorized": False,
        },
    }
    serialized = json.dumps(payload, sort_keys=True)
    for token in _FORBIDDEN_GT_TOKENS:
        if token in serialized:
            raise ReconstructionValidationError(f"I4.2B worker request leaked forbidden token: {token}")
    return payload


def _validate_worker_result(
    result: Mapping[str, Any], *, inputs: TempleRingSfmInputs, workspace: Path
) -> tuple[dict[str, Any], str]:
    if result.get("phase") != TEMPLE_RING_SFM_PHASE or result.get("status") != "ok":
        raise ReconstructionValidationError(f"I4.2B worker failed: {result.get('error', 'unknown error')}")
    authority = result.get("authority")
    if not isinstance(authority, Mapping):
        raise ReconstructionValidationError("I4.2B worker authority evidence is missing.")
    if authority.get("single_authoritative_sfm_frame") is not True:
        raise ReconstructionValidationError("I4.2B worker did not preserve one authoritative SfM frame.")
    for key in (
        "ground_truth_extrinsics_consumed",
        "ground_truth_camera_centers_consumed",
        "ground_truth_angles_consumed",
        "metric_scale_claimed",
        "measurement_authoritative",
        "certified_metrology",
        "surface_mesh_generated",
        "mvs_started",
        "da3_started",
        "dense_provider_promoted",
        "i4_3_provider_architecture_authorized",
    ):
        _require_false(authority, key)

    camera = result.get("camera")
    if not isinstance(camera, Mapping) or camera.get("model") != TEMPLE_RING_CAMERA_MODEL:
        raise ReconstructionValidationError("I4.2B requires one shared PINHOLE camera.")
    if int(camera.get("camera_count", -1)) != 1:
        raise ReconstructionValidationError("I4.2B requires exactly one shared camera record.")
    expected = np.array([
        inputs.shared_K[0, 0], inputs.shared_K[1, 1], inputs.shared_K[0, 2], inputs.shared_K[1, 2]
    ], dtype=np.float64)
    actual = np.asarray(camera.get("params", []), dtype=np.float64)
    if actual.shape != (4,) or not np.allclose(actual, expected, atol=1e-9, rtol=0.0):
        raise ReconstructionValidationError("I4.2B fixed benchmark intrinsics changed during SfM/BA.")
    if camera.get("intrinsics_fixed") is not True:
        raise ReconstructionValidationError("I4.2B worker did not report fixed intrinsics.")

    registered_names = tuple(str(name) for name in result.get("registered_image_names", []))
    if len(registered_names) != len(set(registered_names)):
        raise ReconstructionValidationError("I4.2B worker returned duplicate registered image names.")
    if not set(registered_names).issubset(set(TEMPLE_RING_EXPECTED_NAMES)):
        raise ReconstructionValidationError("I4.2B worker registered an image outside TempleRing.")

    stats_raw = result.get("statistics")
    if not isinstance(stats_raw, Mapping):
        raise ReconstructionValidationError("I4.2B worker statistics are missing.")
    elapsed_s = _finite_positive(result.get("worker_elapsed_s", 0.0), field_name="worker_elapsed_s")
    stats = SfmStageStatistics(
        input_keyframes=47,
        registered_keyframes=int(stats_raw.get("registered_images", -1)),
        landmarks=int(stats_raw.get("landmarks", -1)),
        observations=int(stats_raw.get("observations", -1)),
        elapsed_ns=max(1, int(elapsed_s * 1_000_000_000)),
        mean_reprojection_error_px=(
            None if stats_raw.get("mean_reprojection_error_px") is None
            else float(stats_raw.get("mean_reprojection_error_px"))
        ),
    )
    quality = assess_sfm_quality(stats)
    all_47_registered = stats.registered_keyframes == 47 and set(registered_names) == set(TEMPLE_RING_EXPECTED_NAMES)
    status = "pass" if all_47_registered and quality.status.value == "accepted" else "degraded"

    sparse = workspace / "authoritative_sparse"
    hashes: dict[str, str] = {}
    for filename in ("cameras.bin", "images.bin", "points3D.bin"):
        path = sparse / filename
        if not path.is_file() or path.stat().st_size <= 0:
            raise ReconstructionValidationError(f"I4.2B authoritative sparse model is incomplete: {filename}")
        hashes[filename] = sha256_file(path)

    validated = {
        "all_47_registered": all_47_registered,
        "registered_images": stats.registered_keyframes,
        "input_images": 47,
        "landmarks": stats.landmarks,
        "observations": stats.observations,
        "mean_reprojection_error_px": stats.mean_reprojection_error_px,
        "generic_sfm_quality": quality.as_dict(),
        "fixed_intrinsics_preserved": True,
        "single_shared_camera": True,
        "authoritative_sparse_sha256": hashes,
    }
    return validated, status


def run_temple_ring_authoritative_sfm(
    *,
    i4_2a_dir: str | Path,
    output_dir: str | Path,
    pycolmap_python: str | Path,
    worker_script: str | Path,
    config: TempleRingSfmConfig = TempleRingSfmConfig(),
    worker_timeout_s: float = TEMPLE_RING_DEFAULT_WORKER_TIMEOUT_S,
) -> dict[str, Any]:
    output = _require_clean_output(output_dir)
    inputs = load_i4_2a_reconstruction_inputs(i4_2a_dir)
    executable = Path(pycolmap_python).expanduser().resolve()
    worker = Path(worker_script).expanduser().resolve()
    if not executable.is_file():
        raise ReconstructionValidationError(f"I4.2B PyCOLMAP Python executable is missing: {executable}")
    if not worker.is_file():
        raise ReconstructionValidationError(f"I4.2B worker script is missing: {worker}")
    timeout = _finite_positive(worker_timeout_s, field_name="worker_timeout_s")

    environment = probe_pycolmap_environment(str(executable))
    if not environment.backend_ready or not environment.recommended_version:
        raise ReconstructionValidationError(f"I4.2B PyCOLMAP runtime is not qualified: {environment.as_dict()}")
    if config.use_gpu and not environment.cuda_ready:
        raise ReconstructionValidationError(f"I4.2B GPU SfM requested but CUDA is not ready: {environment.as_dict()}")
    if environment.distribution not in {PYCOLMAP_CUDA_PACKAGE, "pycolmap"}:
        raise ReconstructionValidationError(f"I4.2B unexpected PyCOLMAP distribution: {environment.distribution}")

    output.mkdir(parents=True, exist_ok=True)
    workspace = output / "workspace"
    worker_dir = output / "worker"
    evidence_dir = output / "evidence"
    workspace.mkdir(parents=True, exist_ok=False)
    worker_dir.mkdir(parents=True, exist_ok=False)
    evidence_dir.mkdir(parents=True, exist_ok=False)

    frozen_summary = shutil.copy2(inputs.preflight_summary, evidence_dir / "i4_2a_summary.json")
    frozen_manifest = shutil.copy2(inputs.reconstruction_manifest, evidence_dir / "i4_2a_reconstruction_input_manifest.json")
    staged = stage_temple_ring_images(inputs, workspace / "images")
    request = build_temple_ring_sfm_worker_request(
        inputs=inputs, workspace=workspace, staged_images=staged, config=config
    )
    request["runtime"] = environment.as_dict()
    request["frozen_i4_2a_sha256"] = {
        "summary": sha256_file(frozen_summary),
        "reconstruction_manifest": sha256_file(frozen_manifest),
    }
    request_path = _write_json(worker_dir / "request.json", request)
    result_path = worker_dir / "result.json"

    started = time.monotonic()
    return_code = -1
    launch_error = ""
    try:
        completed = subprocess.run(
            [str(executable), str(worker), "--request", str(request_path), "--result", str(result_path)],
            check=False,
            timeout=timeout,
        )
        return_code = int(completed.returncode)
    except (OSError, subprocess.SubprocessError) as exc:
        launch_error = f"{type(exc).__name__}: {exc}"
    elapsed_s = max(0.0, time.monotonic() - started)

    worker_result: dict[str, Any] = {}
    validation: dict[str, Any] = {}
    status = "fail"
    error = launch_error
    if result_path.is_file():
        worker_result = _read_json(result_path, label="I4.2B worker result")
        if not error and return_code == 0:
            try:
                validation, status = _validate_worker_result(worker_result, inputs=inputs, workspace=workspace)
            except ReconstructionValidationError as exc:
                error = str(exc)
        elif not error:
            error = str(worker_result.get("error", f"worker exited with code {return_code}"))
    elif not error:
        error = f"I4.2B worker exited with code {return_code} without result.json"

    summary = {
        "schema_version": TEMPLE_RING_SFM_SCHEMA_VERSION,
        "phase": TEMPLE_RING_SFM_PHASE,
        "status": status,
        "pass": status == "pass",
        "error": error,
        "return_code": return_code,
        "elapsed_s": elapsed_s,
        "dataset": {
            "identity": "Middlebury_MVS_TempleRing",
            "input_views": 47,
            "image_size": [640, 480],
            "shared_K": inputs.shared_K.tolist(),
            "camera_model": TEMPLE_RING_CAMERA_MODEL,
        },
        "configuration": config.as_dict(),
        "validation": validation,
        "authority": {
            "single_authoritative_sfm_frame": status in {"pass", "degraded"},
            "sfm_started": True,
            "bundle_adjustment_started": True,
            "known_shared_intrinsics_consumed": True,
            "intrinsics_fixed": True,
            "ground_truth_extrinsics_consumed": False,
            "ground_truth_camera_centers_consumed": False,
            "ground_truth_angles_consumed": False,
            "mvs_started": False,
            "da3_started": False,
            "production_filter_policy_changed": False,
            "dense_provider_promoted": False,
            "i4_3_provider_architecture_authorized": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
        },
        "artifacts": {
            "workspace": str(workspace),
            "images": str(workspace / "images"),
            "database": str(workspace / "database.db"),
            "authoritative_sparse": str(workspace / "authoritative_sparse"),
            "worker_request": str(request_path),
            "worker_result": str(result_path),
            "frozen_i4_2a_summary": str(frozen_summary),
            "frozen_i4_2a_reconstruction_manifest": str(frozen_manifest),
        },
        "runtime": environment.as_dict(),
        "next": (
            "I4.2C held-out trajectory validation against TempleRing benchmark R/t using a similarity alignment only after "
            "I4.2B PASS. Do not start MVS/DA3/provider architecture yet."
            if status == "pass" else
            "I4.2B did not qualify for held-out trajectory validation; inspect evidence without relaxing gates."
        ),
    }
    _write_json(output / "i4_2b_summary.json", summary)
    return summary

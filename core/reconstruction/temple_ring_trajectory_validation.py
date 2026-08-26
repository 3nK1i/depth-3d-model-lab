"""I4.2C held-out TempleRing trajectory validation.

This stage opens the Middlebury per-view R/t evidence *only after* I4.2B has
completed a passing, independently estimated 47-view PyCOLMAP reconstruction.
It never feeds benchmark poses back into SfM or BA.

The authoritative I4.2B camera centres are compared with the held-out
Middlebury camera centres after one proper Sim(3) fit (Umeyama, no reflection)
using all 47 corresponding camera centres.  The fitted similarity exists only
inside validation evidence: it is never applied to the authoritative sparse
model.  Camera rotations are then evaluated independently after expressing the
SfM camera orientations in the held-out world frame induced by that same
centre-only similarity.

No MVS, DA3, production filtering, provider promotion, metric-scale claim,
measurement authority, or mesh generation is performed here.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from math import acos, degrees, isfinite
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Mapping, Sequence

import numpy as np

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.sfm_environment import (
    PYCOLMAP_CUDA_PACKAGE,
    probe_pycolmap_environment,
)

TEMPLE_RING_TRAJECTORY_SCHEMA_VERSION = 1
TEMPLE_RING_TRAJECTORY_PHASE = "I4.2C"
TEMPLE_RING_EXPECTED_VIEWS = 47
TEMPLE_RING_EXPECTED_NAMES = tuple(f"templeR{index:04d}.png" for index in range(1, 48))
TEMPLE_RING_POSE_EXPORT_WORKER_TIMEOUT_S = 600.0


# These are I4.2 engineering qualification gates, not official Middlebury
# benchmark thresholds and not metrology claims.  They are intentionally
# declared before the real held-out result is inspected.
@dataclass(frozen=True, slots=True)
class TempleRingTrajectoryThresholds:
    max_center_rmse_fraction_of_median_ring_radius: float = 0.02
    max_center_p95_fraction_of_median_ring_radius: float = 0.03
    max_center_max_fraction_of_median_ring_radius: float = 0.06
    max_rotation_median_deg: float = 1.0
    max_rotation_p95_deg: float = 2.0
    max_rotation_max_deg: float = 5.0

    def __post_init__(self) -> None:
        for field_name in (
            "max_center_rmse_fraction_of_median_ring_radius",
            "max_center_p95_fraction_of_median_ring_radius",
            "max_center_max_fraction_of_median_ring_radius",
            "max_rotation_median_deg",
            "max_rotation_p95_deg",
            "max_rotation_max_deg",
        ):
            value = float(getattr(self, field_name))
            if not isfinite(value) or value <= 0.0:
                raise ReconstructionValidationError(f"I4.2C threshold {field_name} must be finite and positive.")
            object.__setattr__(self, field_name, value)
        if self.max_center_p95_fraction_of_median_ring_radius < self.max_center_rmse_fraction_of_median_ring_radius:
            raise ReconstructionValidationError("I4.2C centre p95 threshold must not be tighter than RMSE threshold.")
        if self.max_center_max_fraction_of_median_ring_radius < self.max_center_p95_fraction_of_median_ring_radius:
            raise ReconstructionValidationError("I4.2C centre max threshold must not be tighter than p95 threshold.")
        if self.max_rotation_p95_deg < self.max_rotation_median_deg:
            raise ReconstructionValidationError("I4.2C rotation p95 threshold must not be tighter than median threshold.")
        if self.max_rotation_max_deg < self.max_rotation_p95_deg:
            raise ReconstructionValidationError("I4.2C rotation max threshold must not be tighter than p95 threshold.")

    def as_dict(self) -> dict[str, float]:
        return {
            "max_center_rmse_fraction_of_median_ring_radius": self.max_center_rmse_fraction_of_median_ring_radius,
            "max_center_p95_fraction_of_median_ring_radius": self.max_center_p95_fraction_of_median_ring_radius,
            "max_center_max_fraction_of_median_ring_radius": self.max_center_max_fraction_of_median_ring_radius,
            "max_rotation_median_deg": self.max_rotation_median_deg,
            "max_rotation_p95_deg": self.max_rotation_p95_deg,
            "max_rotation_max_deg": self.max_rotation_max_deg,
        }


@dataclass(frozen=True, slots=True)
class TempleRingTrajectoryInputs:
    i4_2a_dir: Path
    i4_2b_dir: Path
    i4_2a_summary: Path
    ground_truth_manifest: Path
    i4_2b_summary: Path
    authoritative_sparse: Path
    ground_truth_records: tuple[dict[str, Any], ...]
    shared_K: np.ndarray
    expected_sparse_sha256: dict[str, str]
    source_sha256: dict[str, str]


@dataclass(frozen=True, slots=True)
class SimilarityTransform:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray

    def __post_init__(self) -> None:
        scale = float(self.scale)
        rotation = np.asarray(self.rotation, dtype=np.float64)
        translation = np.asarray(self.translation, dtype=np.float64).reshape(-1)
        if not isfinite(scale) or scale <= 0.0:
            raise ReconstructionValidationError("I4.2C similarity scale must be finite and positive.")
        if rotation.shape != (3, 3) or translation.shape != (3,):
            raise ReconstructionValidationError("I4.2C similarity R/t shapes are invalid.")
        if np.any(~np.isfinite(rotation)) or np.any(~np.isfinite(translation)):
            raise ReconstructionValidationError("I4.2C similarity values must be finite.")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8, rtol=0.0):
            raise ReconstructionValidationError("I4.2C similarity rotation is not orthonormal.")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-8, rtol=0.0):
            raise ReconstructionValidationError("I4.2C similarity fit forbids reflections.")
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation", translation)

    def apply(self, points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 3 or np.any(~np.isfinite(values)):
            raise ReconstructionValidationError("I4.2C similarity points must be finite Nx3 values.")
        return self.scale * (values @ self.rotation.T) + self.translation

    def as_dict(self) -> dict[str, Any]:
        return {
            "convention": "x_ground_truth = scale * R_ground_truth_from_sfm @ x_sfm + translation_ground_truth",
            "scale": self.scale,
            "rotation_ground_truth_from_sfm": self.rotation.tolist(),
            "translation_ground_truth": self.translation.tolist(),
            "rotation_determinant": float(np.linalg.det(self.rotation)),
            "reflection_allowed": False,
        }


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
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ReconstructionValidationError(f"Could not hash I4.2C artifact {source}: {exc}") from exc
    return digest.hexdigest()


def _require_clean_output(path: str | Path) -> Path:
    output = Path(path).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ReconstructionValidationError(
            f"I4.2C output must be empty or absent; use a new immutable validation attempt: {output}"
        )
    return output


def _require_false(mapping: Mapping[str, Any], key: str, *, label: str) -> None:
    if bool(mapping.get(key)):
        raise ReconstructionValidationError(f"I4.2C refuses {label} state {key}=True.")


def _finite_matrix(value: object, shape: tuple[int, ...], *, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != shape or np.any(~np.isfinite(matrix)):
        raise ReconstructionValidationError(f"I4.2C {label} must be a finite {shape} matrix/vector.")
    return matrix


def _validate_rotation(value: object, *, label: str, atol: float = 2e-4) -> np.ndarray:
    rotation = _finite_matrix(value, (3, 3), label=label)
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=atol, rtol=0.0):
        raise ReconstructionValidationError(f"I4.2C {label} is not orthonormal.")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=atol, rtol=0.0):
        raise ReconstructionValidationError(f"I4.2C {label} is not a proper rotation.")
    return rotation


def camera_center_from_camera_from_world(matrix34: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = _finite_matrix(matrix34, (3, 4), label="camera_from_world")
    rotation = _validate_rotation(matrix[:, :3], label="camera_from_world rotation", atol=1e-6)
    translation = matrix[:, 3]
    center = -(rotation.T @ translation)
    if np.any(~np.isfinite(center)):
        raise ReconstructionValidationError("I4.2C derived SfM camera centre is non-finite.")
    return center, rotation, translation


def fit_similarity_umeyama(source_points: object, target_points: object) -> SimilarityTransform:
    """Fit proper Sim(3) mapping ``source`` to ``target`` with no reflection."""
    source = np.asarray(source_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3 or source.shape[0] < 3:
        raise ReconstructionValidationError("I4.2C Sim(3) requires corresponding Nx3 trajectories with N>=3.")
    if np.any(~np.isfinite(source)) or np.any(~np.isfinite(target)):
        raise ReconstructionValidationError("I4.2C Sim(3) trajectories contain non-finite values.")

    source_mean = np.mean(source, axis=0)
    target_mean = np.mean(target, axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    if np.linalg.matrix_rank(source_centered, tol=1e-12) < 2:
        raise ReconstructionValidationError("I4.2C SfM camera trajectory is degenerate for a stable Sim(3) fit.")
    if np.linalg.matrix_rank(target_centered, tol=1e-12) < 2:
        raise ReconstructionValidationError("I4.2C held-out camera trajectory is degenerate for a stable Sim(3) fit.")

    variance = float(np.mean(np.sum(source_centered * source_centered, axis=1)))
    if not isfinite(variance) or variance <= 1e-15:
        raise ReconstructionValidationError("I4.2C SfM trajectory variance is too small for Sim(3).")

    covariance = (target_centered.T @ source_centered) / float(source.shape[0])
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.eye(3, dtype=np.float64)
    if np.linalg.det(u @ vt) < 0.0:
        correction[2, 2] = -1.0
    rotation = u @ correction @ vt
    scale = float(np.sum(singular_values * np.diag(correction)) / variance)
    translation = target_mean - scale * (rotation @ source_mean)
    return SimilarityTransform(scale=scale, rotation=rotation, translation=translation)


def rotation_error_degrees(estimated_camera_from_gt_world: object, ground_truth_camera_from_world: object) -> float:
    estimated = _validate_rotation(estimated_camera_from_gt_world, label="aligned SfM camera rotation", atol=1e-6)
    ground_truth = _validate_rotation(ground_truth_camera_from_world, label="held-out camera rotation")
    delta = ground_truth @ estimated.T
    cosine = float(np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0))
    angle = degrees(acos(cosine))
    if not isfinite(angle):
        raise ReconstructionValidationError("I4.2C rotation error is non-finite.")
    return float(angle)


def _canonical_gt_records(manifest: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    raw = manifest.get("camera_records")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)) or len(raw) != 47:
        raise ReconstructionValidationError("I4.2C requires exactly 47 held-out camera records.")
    records: list[dict[str, Any]] = []
    names: list[str] = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise ReconstructionValidationError("I4.2C held-out camera record is malformed.")
        name = str(item.get("image_name", ""))
        names.append(name)
        R = _validate_rotation(item.get("R_camera_from_world"), label=f"GT rotation {name}")
        t = _finite_matrix(item.get("t_camera_from_world"), (3,), label=f"GT translation {name}")
        center = _finite_matrix(item.get("camera_center_world"), (3,), label=f"GT camera centre {name}")
        expected_center = -(R.T @ t)
        if not np.allclose(center, expected_center, atol=1e-9, rtol=0.0):
            raise ReconstructionValidationError(f"I4.2C held-out centre/R/t inconsistency for {name}.")
        records.append({
            "image_name": name,
            "R_camera_from_world": R,
            "t_camera_from_world": t,
            "camera_center_world": center,
        })
    if tuple(names) != TEMPLE_RING_EXPECTED_NAMES or len(set(names)) != 47:
        raise ReconstructionValidationError("I4.2C held-out camera names/order must be canonical TempleRing 1..47.")
    return tuple(records)


def _validate_shared_k(value: object) -> np.ndarray:
    K = _finite_matrix(value, (3, 3), label="shared K")
    if K[0, 0] <= 0.0 or K[1, 1] <= 0.0 or not np.allclose(K[2], [0.0, 0.0, 1.0], atol=1e-12, rtol=0.0):
        raise ReconstructionValidationError("I4.2C shared K is invalid.")
    return K


def load_i4_2c_inputs(*, i4_2a_dir: str | Path, i4_2b_dir: str | Path) -> TempleRingTrajectoryInputs:
    a_root = Path(i4_2a_dir).expanduser().resolve()
    b_root = Path(i4_2b_dir).expanduser().resolve()
    if not a_root.is_dir() or not b_root.is_dir():
        raise ReconstructionValidationError("I4.2C requires existing I4.2A and I4.2B evidence directories.")

    a_summary_path = a_root / "i4_2a_summary.json"
    gt_path = a_root / "i4_2a_ground_truth_validation_manifest.json"
    b_summary_path = b_root / "i4_2b_summary.json"
    b_frozen_a_summary = b_root / "evidence/i4_2a_summary.json"
    b_frozen_reconstruction_manifest = b_root / "evidence/i4_2a_reconstruction_input_manifest.json"
    a_reconstruction_manifest = a_root / "i4_2a_reconstruction_input_manifest.json"

    a_summary = _read_json(a_summary_path, label="I4.2A summary")
    gt_manifest = _read_json(gt_path, label="I4.2A held-out ground-truth manifest")
    b_summary = _read_json(b_summary_path, label="I4.2B summary")

    if a_summary.get("phase") != "I4.2A" or a_summary.get("status") != "pass" or a_summary.get("pass") is not True:
        raise ReconstructionValidationError("I4.2C requires a passing I4.2A preflight.")
    artifacts = a_summary.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ReconstructionValidationError("I4.2C I4.2A artifact evidence is missing.")
    expected_gt_hash = str(artifacts.get("ground_truth_validation_manifest_sha256", ""))
    expected_reconstruction_hash = str(artifacts.get("reconstruction_input_manifest_sha256", ""))
    if not expected_gt_hash or sha256_file(gt_path) != expected_gt_hash:
        raise ReconstructionValidationError("I4.2C held-out ground-truth manifest checksum changed since I4.2A.")
    if not expected_reconstruction_hash or sha256_file(a_reconstruction_manifest) != expected_reconstruction_hash:
        raise ReconstructionValidationError("I4.2C I4.2A reconstruction manifest checksum changed.")

    if not b_frozen_a_summary.is_file() or sha256_file(b_frozen_a_summary) != sha256_file(a_summary_path):
        raise ReconstructionValidationError("I4.2C I4.2B did not use this exact I4.2A summary.")
    if not b_frozen_reconstruction_manifest.is_file() or sha256_file(b_frozen_reconstruction_manifest) != sha256_file(a_reconstruction_manifest):
        raise ReconstructionValidationError("I4.2C I4.2B did not use this exact I4.2A reconstruction manifest.")

    if gt_manifest.get("phase") != "I4.2A" or gt_manifest.get("dataset") != "Middlebury TempleRing":
        raise ReconstructionValidationError("I4.2C held-out manifest identity is invalid.")
    if gt_manifest.get("purpose") != "held-out validation evidence only; forbidden as SfM pose input":
        raise ReconstructionValidationError("I4.2C held-out manifest purpose/role changed.")
    if gt_manifest.get("sfm_pose_input_authorized") is not False or gt_manifest.get("bundle_adjustment_pose_prior_authorized") is not False:
        raise ReconstructionValidationError("I4.2C requires GT poses to have remained forbidden SfM/BA inputs.")
    for key in ("metric_scale_claimed", "measurement_authoritative", "certified_metrology"):
        _require_false(gt_manifest, key, label="held-out manifest")
    gt_records = _canonical_gt_records(gt_manifest)
    K_gt = _validate_shared_k(gt_manifest.get("shared_intrinsics_K"))

    if b_summary.get("phase") != "I4.2B" or b_summary.get("status") != "pass" or b_summary.get("pass") is not True:
        raise ReconstructionValidationError("I4.2C requires a passing I4.2B authoritative SfM result.")
    validation = b_summary.get("validation")
    authority = b_summary.get("authority")
    if not isinstance(validation, Mapping) or not isinstance(authority, Mapping):
        raise ReconstructionValidationError("I4.2C I4.2B validation/authority evidence is missing.")
    if validation.get("all_47_registered") is not True or int(validation.get("registered_images", -1)) != 47:
        raise ReconstructionValidationError("I4.2C requires 47/47 I4.2B camera registration.")
    if validation.get("fixed_intrinsics_preserved") is not True or validation.get("single_shared_camera") is not True:
        raise ReconstructionValidationError("I4.2C requires fixed shared benchmark intrinsics from I4.2B.")
    quality = validation.get("generic_sfm_quality")
    if not isinstance(quality, Mapping) or quality.get("status") != "accepted" or quality.get("usable") is not True:
        raise ReconstructionValidationError("I4.2C requires accepted I4.2B sparse quality.")
    if authority.get("single_authoritative_sfm_frame") is not True or authority.get("intrinsics_fixed") is not True:
        raise ReconstructionValidationError("I4.2C authoritative-frame/fixed-intrinsics lock is missing.")
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
        _require_false(authority, key, label="I4.2B authority")

    K_b = _validate_shared_k(b_summary.get("dataset", {}).get("shared_K"))
    if not np.allclose(K_gt, K_b, atol=1e-12, rtol=0.0):
        raise ReconstructionValidationError("I4.2C GT and I4.2B shared K do not match exactly.")

    sparse = Path(str(b_summary.get("artifacts", {}).get("authoritative_sparse", ""))).expanduser().resolve()
    expected_sparse_raw = validation.get("authoritative_sparse_sha256")
    if not sparse.is_dir() or not isinstance(expected_sparse_raw, Mapping):
        raise ReconstructionValidationError("I4.2C authoritative sparse model/hash evidence is missing.")
    expected_sparse = {str(name): str(value) for name, value in expected_sparse_raw.items()}
    if set(expected_sparse) != {"cameras.bin", "images.bin", "points3D.bin"}:
        raise ReconstructionValidationError("I4.2C authoritative sparse checksum set is incomplete.")
    for filename, expected in expected_sparse.items():
        path = sparse / filename
        if not path.is_file() or sha256_file(path) != expected:
            raise ReconstructionValidationError(f"I4.2C authoritative sparse artifact changed since I4.2B: {filename}")

    source_hashes = {
        "i4_2a_summary": sha256_file(a_summary_path),
        "i4_2a_reconstruction_manifest": sha256_file(a_reconstruction_manifest),
        "i4_2a_ground_truth_validation_manifest": sha256_file(gt_path),
        "i4_2b_summary": sha256_file(b_summary_path),
        **{f"i4_2b_authoritative_sparse_{name}": value for name, value in sorted(expected_sparse.items())},
    }
    return TempleRingTrajectoryInputs(
        i4_2a_dir=a_root,
        i4_2b_dir=b_root,
        i4_2a_summary=a_summary_path,
        ground_truth_manifest=gt_path,
        i4_2b_summary=b_summary_path,
        authoritative_sparse=sparse,
        ground_truth_records=gt_records,
        shared_K=K_b,
        expected_sparse_sha256=expected_sparse,
        source_sha256=source_hashes,
    )


def build_pose_export_request(inputs: TempleRingTrajectoryInputs) -> dict[str, Any]:
    return {
        "schema_version": TEMPLE_RING_TRAJECTORY_SCHEMA_VERSION,
        "phase": "I4.2C-pose-export",
        "authoritative_sparse": str(inputs.authoritative_sparse),
        "expected_image_names": list(TEMPLE_RING_EXPECTED_NAMES),
        "expected_sparse_sha256": dict(inputs.expected_sparse_sha256),
        "read_only": True,
        "ground_truth_available_to_worker": False,
    }


def _validate_pose_export_result(result: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    if result.get("phase") != "I4.2C-pose-export" or result.get("status") != "ok":
        raise ReconstructionValidationError(f"I4.2C pose-export worker failed: {result.get('error', '')}")
    if result.get("read_only") is not True or result.get("authoritative_sparse_modified") is not False:
        raise ReconstructionValidationError("I4.2C pose exporter did not preserve read-only authority.")
    raw = result.get("registered_images")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)) or len(raw) != 47:
        raise ReconstructionValidationError("I4.2C pose exporter must return exactly 47 registered images.")
    by_name: dict[str, dict[str, Any]] = {}
    for item in raw:
        if not isinstance(item, Mapping):
            raise ReconstructionValidationError("I4.2C pose-export record is malformed.")
        name = str(item.get("name", ""))
        if name in by_name:
            raise ReconstructionValidationError(f"I4.2C pose exporter returned duplicate image {name}.")
        center, R, t = camera_center_from_camera_from_world(item.get("camera_from_reconstruction"))
        by_name[name] = {
            "name": name,
            "image_id": int(item.get("image_id", -1)),
            "camera_id": int(item.get("camera_id", -1)),
            "R_camera_from_sfm_world": R,
            "t_camera_from_sfm_world": t,
            "camera_center_sfm_world": center,
        }
    if tuple(sorted(by_name)) != TEMPLE_RING_EXPECTED_NAMES:
        raise ReconstructionValidationError("I4.2C pose-export image set does not match canonical TempleRing names.")
    return tuple(by_name[name] for name in TEMPLE_RING_EXPECTED_NAMES)


def evaluate_trajectory(
    *,
    sfm_records: Sequence[Mapping[str, Any]],
    ground_truth_records: Sequence[Mapping[str, Any]],
    thresholds: TempleRingTrajectoryThresholds = TempleRingTrajectoryThresholds(),
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], str]:
    if len(sfm_records) != 47 or len(ground_truth_records) != 47:
        raise ReconstructionValidationError("I4.2C trajectory evaluation requires exactly 47 corresponding views.")
    sfm_names = tuple(str(item.get("name", "")) for item in sfm_records)
    gt_names = tuple(str(item.get("image_name", "")) for item in ground_truth_records)
    if sfm_names != TEMPLE_RING_EXPECTED_NAMES or gt_names != TEMPLE_RING_EXPECTED_NAMES:
        raise ReconstructionValidationError("I4.2C trajectory correspondence order changed.")

    sfm_centers = np.stack([
        _finite_matrix(item.get("camera_center_sfm_world"), (3,), label=f"SfM centre {name}")
        for item, name in zip(sfm_records, sfm_names)
    ], axis=0)
    gt_centers = np.stack([
        _finite_matrix(item.get("camera_center_world"), (3,), label=f"GT centre {name}")
        for item, name in zip(ground_truth_records, gt_names)
    ], axis=0)

    similarity = fit_similarity_umeyama(sfm_centers, gt_centers)
    aligned_centers = similarity.apply(sfm_centers)
    center_errors = np.linalg.norm(aligned_centers - gt_centers, axis=1)
    gt_centroid = np.mean(gt_centers, axis=0)
    gt_radii = np.linalg.norm(gt_centers - gt_centroid, axis=1)
    median_radius = float(np.median(gt_radii))
    if not isfinite(median_radius) or median_radius <= 1e-12:
        raise ReconstructionValidationError("I4.2C held-out trajectory has invalid median ring radius.")
    center_fractions = center_errors / median_radius

    rotation_errors: list[float] = []
    per_view: list[dict[str, Any]] = []
    for index, (sfm, gt, name) in enumerate(zip(sfm_records, ground_truth_records, sfm_names)):
        R_sfm = _validate_rotation(sfm.get("R_camera_from_sfm_world"), label=f"SfM rotation {name}", atol=1e-6)
        R_gt = _validate_rotation(gt.get("R_camera_from_world"), label=f"GT rotation {name}")
        # x_gt = s*A*x_sfm+b -> x_sfm=A^T(x_gt-b)/s.  Thus the
        # SfM camera rotation expressed against GT world axes is R_sfm*A^T.
        R_camera_from_gt_world_est = R_sfm @ similarity.rotation.T
        rotation_error = rotation_error_degrees(R_camera_from_gt_world_est, R_gt)
        rotation_errors.append(rotation_error)
        per_view.append({
            "image_name": name,
            "camera_center_sfm_world": sfm_centers[index].tolist(),
            "camera_center_ground_truth": gt_centers[index].tolist(),
            "camera_center_sfm_aligned_to_ground_truth": aligned_centers[index].tolist(),
            "center_error_benchmark_units": float(center_errors[index]),
            "center_error_fraction_of_median_ring_radius": float(center_fractions[index]),
            "rotation_error_deg": float(rotation_error),
        })

    rotation_array = np.asarray(rotation_errors, dtype=np.float64)
    center_rmse = float(np.sqrt(np.mean(center_errors ** 2)))
    metrics = {
        "view_count": 47,
        "ground_truth_median_ring_radius_benchmark_units": median_radius,
        "center_rmse_benchmark_units": center_rmse,
        "center_median_benchmark_units": float(np.median(center_errors)),
        "center_p95_benchmark_units": float(np.percentile(center_errors, 95)),
        "center_max_benchmark_units": float(np.max(center_errors)),
        "center_rmse_fraction_of_median_ring_radius": float(center_rmse / median_radius),
        "center_median_fraction_of_median_ring_radius": float(np.median(center_fractions)),
        "center_p95_fraction_of_median_ring_radius": float(np.percentile(center_fractions, 95)),
        "center_max_fraction_of_median_ring_radius": float(np.max(center_fractions)),
        "rotation_median_deg": float(np.median(rotation_array)),
        "rotation_mean_deg": float(np.mean(rotation_array)),
        "rotation_p95_deg": float(np.percentile(rotation_array, 95)),
        "rotation_max_deg": float(np.max(rotation_array)),
    }
    checks = {
        "center_rmse_within_gate": metrics["center_rmse_fraction_of_median_ring_radius"] <= thresholds.max_center_rmse_fraction_of_median_ring_radius,
        "center_p95_within_gate": metrics["center_p95_fraction_of_median_ring_radius"] <= thresholds.max_center_p95_fraction_of_median_ring_radius,
        "center_max_within_gate": metrics["center_max_fraction_of_median_ring_radius"] <= thresholds.max_center_max_fraction_of_median_ring_radius,
        "rotation_median_within_gate": metrics["rotation_median_deg"] <= thresholds.max_rotation_median_deg,
        "rotation_p95_within_gate": metrics["rotation_p95_deg"] <= thresholds.max_rotation_p95_deg,
        "rotation_max_within_gate": metrics["rotation_max_deg"] <= thresholds.max_rotation_max_deg,
    }
    status = "pass" if all(checks.values()) else "degraded"
    quality = {
        "status": status,
        "accepted": status == "pass",
        "thresholds": thresholds.as_dict(),
        "checks": checks,
        "metrics": metrics,
        "threshold_role": "I4.2 engineering qualification gates; not official Middlebury scoring and not metrology",
    }
    alignment = {
        **similarity.as_dict(),
        "method": "Umeyama least-squares Sim(3) on all 47 corresponding camera centres",
        "fit_correspondences": 47,
        "fit_uses_camera_centers_only": True,
        "ground_truth_rotations_used_for_alignment": False,
        "applied_to_authoritative_sparse_model": False,
        "validation_only": True,
    }
    return alignment, quality, per_view, status


def run_temple_ring_heldout_trajectory_validation(
    *,
    i4_2a_dir: str | Path,
    i4_2b_dir: str | Path,
    output_dir: str | Path,
    pycolmap_python: str | Path,
    pose_export_worker_script: str | Path,
    thresholds: TempleRingTrajectoryThresholds = TempleRingTrajectoryThresholds(),
    worker_timeout_s: float = TEMPLE_RING_POSE_EXPORT_WORKER_TIMEOUT_S,
) -> dict[str, Any]:
    output = _require_clean_output(output_dir)
    inputs = load_i4_2c_inputs(i4_2a_dir=i4_2a_dir, i4_2b_dir=i4_2b_dir)
    executable = Path(pycolmap_python).expanduser().resolve()
    worker = Path(pose_export_worker_script).expanduser().resolve()
    if not executable.is_file() or not worker.is_file():
        raise ReconstructionValidationError("I4.2C PyCOLMAP Python or read-only pose-export worker is missing.")
    try:
        timeout = float(worker_timeout_s)
    except (TypeError, ValueError) as exc:
        raise ReconstructionValidationError("I4.2C worker timeout must be numeric.") from exc
    if not isfinite(timeout) or timeout <= 0.0:
        raise ReconstructionValidationError("I4.2C worker timeout must be finite and positive.")

    environment = probe_pycolmap_environment(str(executable))
    if not environment.backend_ready or not environment.recommended_version:
        raise ReconstructionValidationError(f"I4.2C PyCOLMAP runtime is not qualified: {environment.as_dict()}")
    if environment.distribution not in {PYCOLMAP_CUDA_PACKAGE, "pycolmap"}:
        raise ReconstructionValidationError(f"I4.2C unexpected PyCOLMAP distribution: {environment.distribution}")

    output.mkdir(parents=True, exist_ok=True)
    evidence_dir = output / "evidence"
    worker_dir = output / "worker"
    evidence_dir.mkdir(parents=True, exist_ok=False)
    worker_dir.mkdir(parents=True, exist_ok=False)

    frozen_a_summary = shutil.copy2(inputs.i4_2a_summary, evidence_dir / "i4_2a_summary.json")
    frozen_gt = shutil.copy2(inputs.ground_truth_manifest, evidence_dir / "i4_2a_ground_truth_validation_manifest.json")
    frozen_b_summary = shutil.copy2(inputs.i4_2b_summary, evidence_dir / "i4_2b_summary.json")

    sparse_hashes_before = {name: sha256_file(inputs.authoritative_sparse / name) for name in inputs.expected_sparse_sha256}
    request = build_pose_export_request(inputs)
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

    sparse_hashes_after = {name: sha256_file(inputs.authoritative_sparse / name) for name in inputs.expected_sparse_sha256}
    if sparse_hashes_before != inputs.expected_sparse_sha256 or sparse_hashes_after != inputs.expected_sparse_sha256:
        raise ReconstructionValidationError("I4.2C detected mutation of the authoritative I4.2B sparse model.")

    worker_result: dict[str, Any] = {}
    alignment: dict[str, Any] = {}
    quality: dict[str, Any] = {}
    per_view: list[dict[str, Any]] = []
    status = "fail"
    error = launch_error
    if result_path.is_file():
        worker_result = _read_json(result_path, label="I4.2C pose-export result")
        if not error and return_code == 0:
            try:
                sfm_records = _validate_pose_export_result(worker_result)
                alignment, quality, per_view, status = evaluate_trajectory(
                    sfm_records=sfm_records,
                    ground_truth_records=inputs.ground_truth_records,
                    thresholds=thresholds,
                )
            except ReconstructionValidationError as exc:
                error = str(exc)
        elif not error:
            error = str(worker_result.get("error", f"pose-export worker exited with code {return_code}"))
    elif not error:
        error = f"I4.2C pose-export worker exited with code {return_code} without result.json"

    if status in {"pass", "degraded"}:
        _write_json(output / "i4_2c_similarity_alignment.json", alignment)
        _write_json(output / "i4_2c_per_view_residuals.json", {
            "schema_version": TEMPLE_RING_TRAJECTORY_SCHEMA_VERSION,
            "phase": TEMPLE_RING_TRAJECTORY_PHASE,
            "views": per_view,
        })

    summary = {
        "schema_version": TEMPLE_RING_TRAJECTORY_SCHEMA_VERSION,
        "phase": TEMPLE_RING_TRAJECTORY_PHASE,
        "status": status,
        "pass": status == "pass",
        "error": error,
        "return_code": return_code,
        "elapsed_s": elapsed_s,
        "dataset": {
            "identity": "Middlebury_MVS_TempleRing",
            "views": 47,
            "shared_K": inputs.shared_K.tolist(),
        },
        "alignment": alignment,
        "trajectory_quality": quality,
        "authority": {
            "single_authoritative_sfm_frame_preserved": status in {"pass", "degraded"},
            "authoritative_sparse_modified": False,
            "authoritative_camera_poses_rewritten": False,
            "similarity_applied_to_authoritative_sparse_model": False,
            "similarity_validation_only": True,
            "sfm_rerun": False,
            "bundle_adjustment_rerun": False,
            "pose_refinement": False,
            "ground_truth_extrinsics_consumed_for_validation": status in {"pass", "degraded"},
            "ground_truth_camera_centers_consumed_for_validation": status in {"pass", "degraded"},
            "ground_truth_rotations_used_for_similarity_fit": False,
            "ground_truth_angles_consumed": False,
            "ground_truth_pose_input_to_sfm": False,
            "benchmark_similarity_scale_is_metric_authority": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "mvs_started": False,
            "da3_started": False,
            "production_filter_policy_changed": False,
            "dense_provider_promoted": False,
            "i4_3_provider_architecture_authorized": False,
            "surface_mesh_generated": False,
        },
        "evidence": {
            "source_sha256": dict(inputs.source_sha256),
            "frozen_i4_2a_summary": str(frozen_a_summary),
            "frozen_ground_truth_manifest": str(frozen_gt),
            "frozen_i4_2b_summary": str(frozen_b_summary),
            "authoritative_sparse_sha256_before": sparse_hashes_before,
            "authoritative_sparse_sha256_after": sparse_hashes_after,
            "pose_export_request": str(request_path),
            "pose_export_result": str(result_path),
            "similarity_alignment": str(output / "i4_2c_similarity_alignment.json") if status in {"pass", "degraded"} else "",
            "per_view_residuals": str(output / "i4_2c_per_view_residuals.json") if status in {"pass", "degraded"} else "",
        },
        "runtime": environment.as_dict(),
        "next": (
            "I4.2D TempleRing MVS dense reconstruction from the unchanged authoritative I4.2B sparse model. "
            "Do not start production manifold/detail qualification until the raw MVS result exists."
            if status == "pass" else
            "I4.2C trajectory validation did not qualify for I4.2D. Inspect held-out residual evidence without changing SfM poses or thresholds."
        ),
    }
    _write_json(output / "i4_2c_summary.json", summary)
    return summary

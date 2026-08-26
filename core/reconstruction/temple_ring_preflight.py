"""I4.2A TempleRing dataset and evaluation-contract preflight.

I4.2A is intentionally read-only with respect to reconstruction geometry.  It
validates the canonical Middlebury TempleRing package, freezes the exact 47
image/calibration inputs, and quarantines benchmark ground-truth extrinsics so
later SfM cannot accidentally consume them.

The provided shared camera intrinsics may be carried into the bounded I4.2 SfM
benchmark as an explicit known-intrinsics constraint.  Per-view R/t, camera
centres, optional gantry angles, laser ground truth, metric scale, and any
visual hull are validation-only evidence and are forbidden reconstruction
inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from math import isfinite
from pathlib import Path
import re
import struct
from typing import Any, Mapping, Sequence

import numpy as np

from core.reconstruction.errors import ReconstructionValidationError

TEMPLE_RING_PREFLIGHT_SCHEMA_VERSION = 1
TEMPLE_RING_EXPECTED_VIEWS = 47
TEMPLE_RING_EXPECTED_WIDTH = 640
TEMPLE_RING_EXPECTED_HEIGHT = 480
TEMPLE_RING_PARAMETER_FILE = "templeR_par.txt"
TEMPLE_RING_ANGLE_FILE = "templeR_ang.txt"
TEMPLE_RING_IMAGE_PATTERN = re.compile(r"^templeR\d{4}\.png$")
TEMPLE_RING_REVIEW_DIMENSIONS = (
    "thin_vertical_member_separation",
    "small_gap_and_opening_preservation",
    "edge_definition",
    "small_protrusion_preservation",
    "surface_thickness_and_false_bridging",
    "local_completeness_vs_noise",
    "production_manifold_detail_loss",
)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError(f"Could not read {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"{label} must contain a JSON object: {path}")
    return payload


def write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ReconstructionValidationError(f"Could not hash I4.2A input {path}: {exc}") from exc
    return digest.hexdigest()


def require_clean_output(path: str | Path) -> Path:
    output = Path(path).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ReconstructionValidationError(
            f"I4.2A output must be empty or absent; refusing to overwrite frozen preflight evidence: {output}"
        )
    return output


def _require_false(mapping: Mapping[str, Any], key: str) -> None:
    if bool(mapping.get(key)):
        raise ReconstructionValidationError(f"I4.2A refuses forbidden upstream authority state {key}=True.")


def validate_i4_1_9a_closure(path: str | Path) -> dict[str, Any]:
    closure_path = Path(path).expanduser().resolve()
    closure = _read_json(closure_path, label="I4.1.9A closure")
    if closure.get("phase") != "I4.1.9A-A3.1" or closure.get("status") != "pass" or closure.get("pass") is not True:
        raise ReconstructionValidationError("I4.2A requires a passing I4.1.9A-A3.1 closure.")
    decision = closure.get("decision")
    if not isinstance(decision, Mapping):
        raise ReconstructionValidationError("I4.2A upstream decision evidence is missing.")
    required_true = (
        "house_mvs_positive_control_closed",
        "i4_1_9a_closed",
        "i4_2_detail_preservation_authorized",
        "mvs_static_object_orbit_provider_candidate_supported",
    )
    if any(decision.get(key) is not True for key in required_true):
        raise ReconstructionValidationError("I4.2A is not authorized by the House MVS closure.")
    if decision.get("i4_3_provider_architecture_authorized") is not False:
        raise ReconstructionValidationError("I4.2A requires I4.3 provider architecture to remain unauthorized.")
    if decision.get("random_dense_threshold_tuning_authorized") is not False:
        raise ReconstructionValidationError("I4.2A refuses random dense-threshold tuning authorization.")

    authority = closure.get("authority")
    if not isinstance(authority, Mapping):
        raise ReconstructionValidationError("I4.2A upstream authority evidence is missing.")
    for key in (
        "camera_geometry_modified",
        "sfm_rerun",
        "bundle_adjustment_rerun",
        "pose_refinement",
        "dense_provider_promoted",
        "metric_scale_claimed",
        "measurement_authoritative",
        "certified_metrology",
        "surface_mesh_generated",
    ):
        _require_false(authority, key)
    if authority.get("single_authoritative_sfm_frame") is not True:
        raise ReconstructionValidationError("I4.2A requires the single-authoritative-SfM-frame lock.")
    return {
        "path": str(closure_path),
        "sha256": sha256_file(closure_path),
        "verdict": str(closure.get("manual_review", {}).get("verdict", "")),
        "classification": str(closure.get("manual_review", {}).get("classification", "")),
    }


def read_png_size(path: str | Path) -> tuple[int, int]:
    image = Path(path)
    try:
        header = image.read_bytes()[:24]
    except OSError as exc:
        raise ReconstructionValidationError(f"Could not read TempleRing image header {image}: {exc}") from exc
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ReconstructionValidationError(f"TempleRing input must be PNG with an IHDR header: {image.name}")
    width, height = struct.unpack(">II", header[16:24])
    if width <= 0 or height <= 0:
        raise ReconstructionValidationError(f"TempleRing image has invalid dimensions: {image.name}")
    return int(width), int(height)


@dataclass(frozen=True, slots=True)
class TempleRingCalibrationRecord:
    image_name: str
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    camera_center: np.ndarray

    def __post_init__(self) -> None:
        if TEMPLE_RING_IMAGE_PATTERN.fullmatch(str(self.image_name)) is None:
            raise ReconstructionValidationError(f"Unexpected TempleRing image name in calibration: {self.image_name}")
        K = np.asarray(self.K, dtype=np.float64)
        R = np.asarray(self.R, dtype=np.float64)
        t = np.asarray(self.t, dtype=np.float64).reshape(-1)
        center = np.asarray(self.camera_center, dtype=np.float64).reshape(-1)
        if K.shape != (3, 3) or R.shape != (3, 3) or t.shape != (3,) or center.shape != (3,):
            raise ReconstructionValidationError("TempleRing K/R/t/camera-center shapes are invalid.")
        if np.any(~np.isfinite(K)) or np.any(~np.isfinite(R)) or np.any(~np.isfinite(t)) or np.any(~np.isfinite(center)):
            raise ReconstructionValidationError("TempleRing calibration values must be finite.")
        if K[0, 0] <= 0.0 or K[1, 1] <= 0.0 or not np.allclose(K[2], [0.0, 0.0, 1.0], atol=1e-8, rtol=0.0):
            raise ReconstructionValidationError("TempleRing intrinsic matrix is invalid.")
        if abs(float(np.linalg.det(K))) <= 1e-12:
            raise ReconstructionValidationError("TempleRing intrinsic matrix must be invertible.")
        if not np.allclose(R.T @ R, np.eye(3), atol=2e-4, rtol=0.0) or not np.isclose(np.linalg.det(R), 1.0, atol=2e-4, rtol=0.0):
            raise ReconstructionValidationError("TempleRing ground-truth rotation must be a proper rigid rotation.")
        expected_center = -(R.T @ t)
        if not np.allclose(center, expected_center, atol=1e-9, rtol=0.0):
            raise ReconstructionValidationError("TempleRing camera center must equal -R^T t.")
        object.__setattr__(self, "K", K)
        object.__setattr__(self, "R", R)
        object.__setattr__(self, "t", t)
        object.__setattr__(self, "camera_center", center)


def parse_calibration_line(line: str, *, line_number: int) -> TempleRingCalibrationRecord:
    tokens = str(line).split()
    if len(tokens) != 22:
        raise ReconstructionValidationError(
            f"TempleRing calibration line {line_number} must contain filename + 21 numeric parameters."
        )
    name = tokens[0]
    try:
        values = np.asarray([float(value) for value in tokens[1:]], dtype=np.float64)
    except ValueError as exc:
        raise ReconstructionValidationError(f"TempleRing calibration line {line_number} contains a non-numeric value.") from exc
    if np.any(~np.isfinite(values)):
        raise ReconstructionValidationError(f"TempleRing calibration line {line_number} contains non-finite values.")
    K = values[:9].reshape(3, 3)
    R = values[9:18].reshape(3, 3)
    t = values[18:21]
    return TempleRingCalibrationRecord(image_name=name, K=K, R=R, t=t, camera_center=-(R.T @ t))


def _parse_temple_ring_calibration_file(
    path: str | Path,
) -> tuple[tuple[TempleRingCalibrationRecord, ...], dict[str, Any]]:
    """Parse canonical Middlebury TempleRing calibration with strict count validation.

    The official ``templeR_par.txt`` begins with a single integer declaring the
    number of camera records (47), followed by one ``filename + 21 values`` row
    per image.  Earlier synthetic I4.2A fixtures were headerless, so the parser
    remains backward-compatible with that exact 47-row form while preferring
    and explicitly validating the canonical count header when present.
    """

    calibration_path = Path(path)
    try:
        raw_lines = tuple(
            (line_number, line.strip())
            for line_number, line in enumerate(
                calibration_path.read_text(encoding="utf-8").splitlines(),
                start=1,
            )
            if line.strip()
        )
    except OSError as exc:
        raise ReconstructionValidationError(f"Could not read TempleRing calibration file {calibration_path}: {exc}") from exc
    if not raw_lines:
        raise ReconstructionValidationError("TempleRing calibration file is empty.")

    declared_view_count: int | None = None
    count_header_present = False
    calibration_rows = raw_lines
    first_line_number, first_line = raw_lines[0]
    first_tokens = first_line.split()
    if len(first_tokens) == 1 and re.fullmatch(r"[+-]?\d+", first_tokens[0]):
        count_header_present = True
        declared_view_count = int(first_tokens[0])
        if declared_view_count != TEMPLE_RING_EXPECTED_VIEWS:
            raise ReconstructionValidationError(
                "TempleRing calibration count header must declare exactly "
                f"{TEMPLE_RING_EXPECTED_VIEWS} views; found {declared_view_count} on line {first_line_number}."
            )
        calibration_rows = raw_lines[1:]

    if len(calibration_rows) != TEMPLE_RING_EXPECTED_VIEWS:
        suffix = " after the count header" if count_header_present else ""
        raise ReconstructionValidationError(
            f"TempleRing requires exactly {TEMPLE_RING_EXPECTED_VIEWS} calibration rows{suffix}; "
            f"found {len(calibration_rows)}."
        )

    records = tuple(
        parse_calibration_line(line, line_number=line_number)
        for line_number, line in calibration_rows
    )
    names = tuple(record.image_name for record in records)
    if len(set(names)) != TEMPLE_RING_EXPECTED_VIEWS:
        raise ReconstructionValidationError("TempleRing calibration contains duplicate image names.")

    expected_names = tuple(f"templeR{index:04d}.png" for index in range(1, TEMPLE_RING_EXPECTED_VIEWS + 1))
    if set(names) != set(expected_names):
        missing = sorted(set(expected_names) - set(names))
        unexpected = sorted(set(names) - set(expected_names))
        raise ReconstructionValidationError(
            "TempleRing calibration must contain the exact canonical templeR0001.png..templeR0047.png set: "
            f"missing={missing} unexpected={unexpected}"
        )

    ordered = tuple(sorted(records, key=lambda item: item.image_name))
    reference = ordered[0].K
    max_spread = max(float(np.max(np.abs(record.K - reference))) for record in ordered)
    if max_spread > 1e-6:
        raise ReconstructionValidationError(
            "I4.2A requires the TempleRing benchmark's single shared intrinsic matrix; per-view K changed."
        )
    metadata = {
        "count_header_present": count_header_present,
        "declared_view_count": declared_view_count,
        "validated_calibration_rows": len(ordered),
    }
    return ordered, metadata


def load_temple_ring_calibration(path: str | Path) -> tuple[TempleRingCalibrationRecord, ...]:
    records, _metadata = _parse_temple_ring_calibration_file(path)
    return records


def _find_cross_dataset_contamination(root: Path, expected_names: set[str]) -> tuple[str, ...]:
    contamination: list[str] = []
    for path in root.iterdir():
        if not path.is_file() or path.suffix.lower() != ".png":
            continue
        name = path.name
        if name in expected_names:
            continue
        if name.lower().startswith("temple"):
            contamination.append(name)
    return tuple(sorted(contamination))


def inspect_temple_ring_dataset(dataset_dir: str | Path) -> dict[str, Any]:
    root = Path(dataset_dir).expanduser().resolve()
    if not root.is_dir():
        raise ReconstructionValidationError(f"TempleRing dataset directory does not exist: {root}")
    calibration_path = root / TEMPLE_RING_PARAMETER_FILE
    if not calibration_path.is_file():
        raise ReconstructionValidationError(f"TempleRing calibration file is missing: {calibration_path}")
    records, calibration_metadata = _parse_temple_ring_calibration_file(calibration_path)
    expected_names = {record.image_name for record in records}
    images = tuple(sorted(root.glob("templeR*.png"), key=lambda path: path.name))
    image_names = {path.name for path in images}
    if len(images) != TEMPLE_RING_EXPECTED_VIEWS or image_names != expected_names:
        missing = sorted(expected_names - image_names)
        unexpected = sorted(image_names - expected_names)
        raise ReconstructionValidationError(
            f"TempleRing image/calibration set mismatch: images={len(images)} missing={missing} unexpected={unexpected}"
        )
    contamination = _find_cross_dataset_contamination(root, expected_names)
    if contamination:
        raise ReconstructionValidationError(
            "TempleRing directory contains cross-dataset Temple PNGs that must not enter this benchmark: "
            + ", ".join(contamination)
        )

    image_records: list[dict[str, Any]] = []
    for path in images:
        width, height = read_png_size(path)
        if (width, height) != (TEMPLE_RING_EXPECTED_WIDTH, TEMPLE_RING_EXPECTED_HEIGHT):
            raise ReconstructionValidationError(
                f"TempleRing image dimensions changed for {path.name}: {width}x{height}; "
                f"expected {TEMPLE_RING_EXPECTED_WIDTH}x{TEMPLE_RING_EXPECTED_HEIGHT}."
            )
        image_records.append(
            {
                "name": path.name,
                "path": str(path),
                "sha256": sha256_file(path),
                "width": width,
                "height": height,
            }
        )

    K = records[0].K
    camera_centers = np.stack([record.camera_center for record in records], axis=0)
    angle_path = root / TEMPLE_RING_ANGLE_FILE
    optional_angles = {
        "present": angle_path.is_file(),
        "path": str(angle_path) if angle_path.is_file() else "",
        "sha256": sha256_file(angle_path) if angle_path.is_file() else "",
    }
    return {
        "root": str(root),
        "images": image_records,
        "calibration_path": str(calibration_path),
        "calibration_sha256": sha256_file(calibration_path),
        "calibration_records": records,
        "calibration_count_header_present": bool(calibration_metadata["count_header_present"]),
        "calibration_declared_view_count": calibration_metadata["declared_view_count"],
        "validated_calibration_rows": int(calibration_metadata["validated_calibration_rows"]),
        "shared_K": K,
        "shared_K_max_abs_spread": max(float(np.max(np.abs(record.K - K))) for record in records),
        "camera_center_bounds_min": np.min(camera_centers, axis=0),
        "camera_center_bounds_max": np.max(camera_centers, axis=0),
        "optional_angles": optional_angles,
        "cross_dataset_contamination": list(contamination),
    }


def build_i4_2a_evidence(
    *,
    dataset: Mapping[str, Any],
    upstream_closure: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    images = dataset.get("images")
    records = dataset.get("calibration_records")
    if not isinstance(images, Sequence) or isinstance(images, (str, bytes, bytearray)) or len(images) != TEMPLE_RING_EXPECTED_VIEWS:
        raise ReconstructionValidationError("I4.2A requires exactly 47 validated image records.")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes, bytearray)) or len(records) != TEMPLE_RING_EXPECTED_VIEWS:
        raise ReconstructionValidationError("I4.2A requires exactly 47 validated calibration records.")
    K = np.asarray(dataset.get("shared_K"), dtype=np.float64)
    if K.shape != (3, 3) or np.any(~np.isfinite(K)):
        raise ReconstructionValidationError("I4.2A shared intrinsics are invalid.")

    reconstruction_manifest = {
        "schema_version": TEMPLE_RING_PREFLIGHT_SCHEMA_VERSION,
        "phase": "I4.2A",
        "dataset": "Middlebury TempleRing",
        "purpose": "future bounded I4.2 SfM reconstruction inputs only",
        "image_count": TEMPLE_RING_EXPECTED_VIEWS,
        "images": [
            {
                "name": str(item["name"]),
                "path": str(item["path"]),
                "sha256": str(item["sha256"]),
                "width": int(item["width"]),
                "height": int(item["height"]),
            }
            for item in images
        ],
        "shared_intrinsics_K": K.tolist(),
        "intrinsics_role": "benchmark_provided_known_intrinsics_candidate_for_i4_2_sfm",
        "radial_distortion_reestimation_required": False,
        "ground_truth_extrinsics_included": False,
        "ground_truth_camera_centers_included": False,
        "ground_truth_angles_included": False,
        "laser_ground_truth_consumed": False,
        "visual_hull_consumed": False,
        "all_47_views_available": True,
        "permanent_global_view_cap": False,
    }
    forbidden_pose_keys = {"R_camera_from_world", "t_camera_from_world", "camera_center_world"}
    if forbidden_pose_keys.intersection(reconstruction_manifest):
        raise ReconstructionValidationError("I4.2A reconstruction manifest leaked held-out ground-truth extrinsics.")

    ground_truth_manifest = {
        "schema_version": TEMPLE_RING_PREFLIGHT_SCHEMA_VERSION,
        "phase": "I4.2A",
        "dataset": "Middlebury TempleRing",
        "purpose": "held-out validation evidence only; forbidden as SfM pose input",
        "calibration_file": str(dataset["calibration_path"]),
        "calibration_sha256": str(dataset["calibration_sha256"]),
        "shared_intrinsics_K": K.tolist(),
        "camera_records": [
            {
                "image_name": record.image_name,
                "R_camera_from_world": record.R.tolist(),
                "t_camera_from_world": record.t.tolist(),
                "camera_center_world": record.camera_center.tolist(),
            }
            for record in records
        ],
        "camera_center_bounds_min": np.asarray(dataset["camera_center_bounds_min"], dtype=float).tolist(),
        "camera_center_bounds_max": np.asarray(dataset["camera_center_bounds_max"], dtype=float).tolist(),
        "optional_angle_file": dict(dataset["optional_angles"]),
        "sfm_pose_input_authorized": False,
        "bundle_adjustment_pose_prior_authorized": False,
        "metric_scale_claimed": False,
        "measurement_authoritative": False,
        "certified_metrology": False,
    }

    summary = {
        "schema_version": TEMPLE_RING_PREFLIGHT_SCHEMA_VERSION,
        "phase": "I4.2A",
        "status": "pass",
        "pass": True,
        "dataset": {
            "identity": "Middlebury_MVS_TempleRing",
            "root": str(dataset["root"]),
            "expected_views": TEMPLE_RING_EXPECTED_VIEWS,
            "validated_views": TEMPLE_RING_EXPECTED_VIEWS,
            "image_size": [TEMPLE_RING_EXPECTED_WIDTH, TEMPLE_RING_EXPECTED_HEIGHT],
            "shared_intrinsics": True,
            "shared_K_max_abs_spread": float(dataset["shared_K_max_abs_spread"]),
            "calibration_count_header_present": bool(dataset["calibration_count_header_present"]),
            "calibration_declared_view_count": dataset["calibration_declared_view_count"],
            "validated_calibration_rows": int(dataset["validated_calibration_rows"]),
            "cross_dataset_contamination": list(dataset["cross_dataset_contamination"]),
        },
        "upstream_house_mvs_closure": dict(upstream_closure),
        "evaluation_contract": {
            "goal": "thin_structure_and_detail_preservation_before_I4.3_provider_architecture",
            "review_dimensions": list(TEMPLE_RING_REVIEW_DIMENSIONS),
            "preserved_house_mvs_baseline_required": True,
            "production_manifold_policy_under_test": "d12_sparse",
            "room_scale_manifold_policy_assumed_safe_for_object_detail": False,
            "manifold_policy_change_authorized": False,
            "random_threshold_tuning_authorized": False,
            "laser_ground_truth_mesh_consumed": False,
            "manual_visual_review_required_later": True,
        },
        "authority": {
            "dataset_preflight_only": True,
            "sfm_started": False,
            "bundle_adjustment_started": False,
            "mvs_started": False,
            "da3_started": False,
            "production_filtering_rerun": False,
            "production_filter_policy_changed": False,
            "dense_provider_promoted": False,
            "i4_3_provider_architecture_authorized": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
        },
        "next": (
            "I4.2B: bounded TempleRing SfM using all 47 images and the provided shared K while keeping per-view "
            "ground-truth R/t fully held out for post-SfM validation."
        ),
    }
    return reconstruction_manifest, ground_truth_manifest, summary


def run_i4_2a_preflight(
    *,
    dataset_dir: str | Path,
    i4_1_9a_closure: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    output = require_clean_output(output_dir)
    upstream = validate_i4_1_9a_closure(i4_1_9a_closure)
    dataset = inspect_temple_ring_dataset(dataset_dir)
    reconstruction, ground_truth, summary = build_i4_2a_evidence(dataset=dataset, upstream_closure=upstream)

    output.mkdir(parents=True, exist_ok=True)
    reconstruction_path = write_json(output / "i4_2a_reconstruction_input_manifest.json", reconstruction)
    ground_truth_path = write_json(output / "i4_2a_ground_truth_validation_manifest.json", ground_truth)
    summary["artifacts"] = {
        "reconstruction_input_manifest": str(reconstruction_path),
        "reconstruction_input_manifest_sha256": sha256_file(reconstruction_path),
        "ground_truth_validation_manifest": str(ground_truth_path),
        "ground_truth_validation_manifest_sha256": sha256_file(ground_truth_path),
    }
    summary_path = write_json(output / "i4_2a_summary.json", summary)
    return {**summary, "summary_path": str(summary_path)}

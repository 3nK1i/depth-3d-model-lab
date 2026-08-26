#!/usr/bin/env python3
"""D2 three-view overlap and geometric-consistency diagnostics.

D2 consumes an already successful D1 raw three-view accumulation.  It does not
rerun PyCOLMAP or DA3 and it does not fuse, filter, persist, or promote geometry.
Instead it asks the one question D1 intentionally left open: when two registered
views observe the same scene region, do their independently aligned dense
surfaces agree in the common offline-SfM reconstruction frame?

The primary diagnostic is bidirectional reprojection depth consistency.  Points
from view A are transformed into view B's SfM camera, projected through B's
actual camera model, and compared against B's already-produced aligned dense
depth.  Points hidden behind B's nearer surface are explicitly classified as
occluded rather than treated as registration failures.  A bounded spatial
nearest-neighbour diagnostic is recorded as secondary evidence only.

All values remain in the arbitrary SfM reconstruction gauge.  No metric scale,
canonical persistence/authority, measurement authority, or certified metrology
is introduced here.
"""

from __future__ import annotations

import argparse
import json
from math import isfinite
from pathlib import Path
import sys
from typing import Any, Iterable

import cv2
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction import reconstruction_camera_from_sfm_model  # noqa: E402
from core.reconstruction.camera import CameraDistortionModel, ReconstructionCamera  # noqa: E402
from core.reconstruction.da3 import read_da3_result_manifest, validate_da3_npz_result  # noqa: E402
from core.reconstruction.errors import ReconstructionValidationError  # noqa: E402
from scripts.run_b2_sfm_dense_real_alignment import _adapt_da3_depth_to_sfm_pixel_space  # noqa: E402
from scripts.run_d1_three_view_canonical_accumulation import _sfm_result_from_payload  # noqa: E402

_D2_VIEW_COUNT = 3
_D2_PAIRS = ((0, 1), (1, 2), (0, 2))
_CLASS_CONSISTENT = np.uint8(0)
_CLASS_FRONT_CONFLICT = np.uint8(1)
_CLASS_BACK_CONFLICT = np.uint8(2)
_CLASS_OCCLUDED = np.uint8(3)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--d1-dir",
        type=Path,
        default=Path("/tmp/eia_d1_three_view_accumulation"),
        help="Successful D1 output directory to diagnose.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/eia_d2_overlap_diagnostics"),
    )
    parser.add_argument(
        "--max-points-per-direction",
        type=int,
        default=120_000,
        help="Deterministic bound for each directed reprojection diagnostic.",
    )
    parser.add_argument(
        "--nn-max-points-per-view",
        type=int,
        default=60_000,
        help="Bounded sample used only by the secondary spatial-NN diagnostic.",
    )
    parser.add_argument(
        "--consistency-relative-tolerance",
        type=float,
        default=0.12,
        help="Relative depth residual considered cross-view consistent.",
    )
    parser.add_argument(
        "--occlusion-relative-margin",
        type=float,
        default=0.20,
        help="Positive relative residual above which the source point is treated as hidden behind target geometry.",
    )
    parser.add_argument("--minimum-comparable-samples", type=int, default=1_000)
    parser.add_argument("--minimum-consistent-ratio", type=float, default=0.50)
    parser.add_argument("--maximum-median-relative-residual", type=float, default=0.10)
    parser.add_argument("--maximum-p90-relative-residual", type=float, default=0.25)
    parser.add_argument("--maximum-front-conflict-ratio", type=float, default=0.25)
    parser.add_argument(
        "--nn-support-scene-fraction",
        type=float,
        default=0.05,
        help="Spatial-NN support radius as a fraction of robust sparse-scene diagonal.",
    )
    args = parser.parse_args()
    if args.max_points_per_direction <= 0 or args.nn_max_points_per_view <= 0:
        parser.error("D2 point budgets must be positive.")
    if args.minimum_comparable_samples <= 0:
        parser.error("--minimum-comparable-samples must be positive.")
    for name in (
        "consistency_relative_tolerance",
        "occlusion_relative_margin",
        "minimum_consistent_ratio",
        "maximum_median_relative_residual",
        "maximum_p90_relative_residual",
        "maximum_front_conflict_ratio",
        "nn_support_scene_fraction",
    ):
        value = float(getattr(args, name))
        if not isfinite(value) or value <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive.")
    if args.consistency_relative_tolerance >= args.occlusion_relative_margin:
        parser.error("--consistency-relative-tolerance must be smaller than --occlusion-relative-margin.")
    if args.minimum_consistent_ratio > 1.0 or args.maximum_front_conflict_ratio > 1.0:
        parser.error("Ratio thresholds must not exceed 1.0.")
    return args


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError(f"Unable to read required D2 JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"Required D2 JSON root must be an object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def _deterministic_sample(points: np.ndarray, maximum_points: int) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1:] != (3,):
        raise ReconstructionValidationError("D2 point samples must be Nx3 arrays.")
    finite = np.isfinite(values).all(axis=1)
    values = values[finite]
    if values.shape[0] <= maximum_points:
        return np.ascontiguousarray(values)
    indices = np.linspace(0, values.shape[0] - 1, maximum_points, dtype=np.int64)
    return np.ascontiguousarray(values[indices])


def _robust_scene_scale(
    sparse_landmarks: np.ndarray,
    fallback_clouds: Iterable[np.ndarray],
) -> tuple[float, str, list[float], list[float]]:
    sparse = np.asarray(sparse_landmarks, dtype=np.float64)
    sparse = sparse[np.isfinite(sparse).all(axis=1)] if sparse.ndim == 2 and sparse.shape[1:] == (3,) else np.empty((0, 3))
    if sparse.shape[0] >= 32:
        source = sparse
        source_name = "sparse_sfm_landmarks"
    else:
        parts = [_deterministic_sample(points, 50_000) for points in fallback_clouds]
        source = np.concatenate(parts, axis=0) if parts else np.empty((0, 3), dtype=np.float64)
        source_name = "dense_view_fallback"
    if source.shape[0] < 8:
        raise ReconstructionValidationError("D2 cannot establish a robust scene scale from available geometry.")
    low = np.quantile(source, 0.01, axis=0)
    high = np.quantile(source, 0.99, axis=0)
    diagonal = float(np.linalg.norm(high - low))
    if not isfinite(diagonal) or diagonal <= 1e-9:
        raise ReconstructionValidationError("D2 robust scene diagonal is degenerate.")
    return diagonal, source_name, low.tolist(), high.tolist()


def _bilinear_sample(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    values = np.asarray(image, dtype=np.float64)
    if values.ndim != 2:
        raise ReconstructionValidationError("D2 bilinear sampling requires a 2D depth image.")
    uu = np.asarray(u, dtype=np.float64)
    vv = np.asarray(v, dtype=np.float64)
    if uu.shape != vv.shape:
        raise ReconstructionValidationError("D2 bilinear u/v coordinates must have matching shapes.")
    result = np.full(uu.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(uu) & np.isfinite(vv)
    inside = finite & (uu >= 0.0) & (vv >= 0.0) & (uu <= values.shape[1] - 1) & (vv <= values.shape[0] - 1)
    if not np.any(inside):
        return result
    x = uu[inside]
    y = vv[inside]
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, values.shape[1] - 1)
    y1 = np.minimum(y0 + 1, values.shape[0] - 1)
    wx = x - x0
    wy = y - y0
    a = values[y0, x0]
    b = values[y0, x1]
    c = values[y1, x0]
    d = values[y1, x1]
    sampled = (
        a * (1.0 - wx) * (1.0 - wy)
        + b * wx * (1.0 - wy)
        + c * (1.0 - wx) * wy
        + d * wx * wy
    )
    result[inside] = sampled
    return result


def _project_camera_points_to_pixels(
    camera: ReconstructionCamera,
    points_camera: np.ndarray,
) -> np.ndarray:
    points = np.asarray(points_camera, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise ReconstructionValidationError("D2 camera projection requires Nx3 camera-space points.")
    matrix = np.asarray(camera.intrinsics.matrix, dtype=np.float64)
    if camera.distortion_model is CameraDistortionModel.NONE:
        z = points[:, 2]
        x = points[:, 0] / z
        y = points[:, 1] / z
        return np.ascontiguousarray(
            np.column_stack((matrix[0, 0] * x + matrix[0, 2], matrix[1, 1] * y + matrix[1, 2])),
            dtype=np.float64,
        )
    object_points = points.reshape(-1, 1, 3).astype(np.float64)
    zero = np.zeros((3, 1), dtype=np.float64)
    coefficients = np.asarray(camera.distortion_coefficients, dtype=np.float64)
    if camera.distortion_model is CameraDistortionModel.BROWN_CONRADY:
        pixels, _ = cv2.projectPoints(object_points, zero, zero, matrix, coefficients)
    elif camera.distortion_model is CameraDistortionModel.OPENCV_FISHEYE:
        pixels, _ = cv2.fisheye.projectPoints(object_points, zero, zero, matrix, coefficients)
    else:
        raise ReconstructionValidationError(
            "D2 cannot diagnose an unsupported backend-specific camera distortion model."
        )
    return np.ascontiguousarray(pixels.reshape(-1, 2), dtype=np.float64)


def _world_to_camera(points_reconstruction: np.ndarray, T_reconstruction_from_camera: np.ndarray) -> np.ndarray:
    points = np.asarray(points_reconstruction, dtype=np.float64)
    transform = np.asarray(T_reconstruction_from_camera, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,) or transform.shape != (4, 4):
        raise ReconstructionValidationError("D2 world-to-camera conversion received invalid geometry.")
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return np.ascontiguousarray((points - translation) @ rotation, dtype=np.float64)


def _format_optional(value: object, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    parsed = float(value)
    return f"{parsed:.{digits}f}" if isfinite(parsed) else "n/a"


def _safe_quantile(values: np.ndarray, q: float) -> float | None:
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return None
    return float(np.quantile(data, q))


def _directed_reprojection_diagnostic(
    source_points_reconstruction: np.ndarray,
    *,
    target_camera: ReconstructionCamera,
    target_T_reconstruction_from_camera: np.ndarray,
    target_aligned_depth: np.ndarray,
    maximum_points: int,
    consistency_relative_tolerance: float,
    occlusion_relative_margin: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    source = _deterministic_sample(source_points_reconstruction, maximum_points)
    camera_points = _world_to_camera(source, target_T_reconstruction_from_camera)
    positive = np.isfinite(camera_points).all(axis=1) & (camera_points[:, 2] > 1e-9)
    source = source[positive]
    camera_points = camera_points[positive]
    if source.shape[0] == 0:
        raise ReconstructionValidationError("D2 directed reprojection has no points in front of the target camera.")
    pixels = _project_camera_points_to_pixels(target_camera, camera_points)
    width = target_camera.image_size.width
    height = target_camera.image_size.height
    in_frame = (
        np.isfinite(pixels).all(axis=1)
        & (pixels[:, 0] >= 0.0)
        & (pixels[:, 0] <= width - 1)
        & (pixels[:, 1] >= 0.0)
        & (pixels[:, 1] <= height - 1)
    )
    projected_source = source[in_frame]
    projected_camera = camera_points[in_frame]
    projected_pixels = pixels[in_frame]
    sampled_target_depth = _bilinear_sample(
        target_aligned_depth,
        projected_pixels[:, 0],
        projected_pixels[:, 1],
    )
    valid_target = np.isfinite(sampled_target_depth) & (sampled_target_depth > 0.0)
    projected_source = projected_source[valid_target]
    projected_camera = projected_camera[valid_target]
    sampled_target_depth = sampled_target_depth[valid_target]
    if projected_source.shape[0] == 0:
        raise ReconstructionValidationError("D2 directed reprojection found no valid target-depth samples.")

    signed_relative = (projected_camera[:, 2] - sampled_target_depth) / np.maximum(sampled_target_depth, 1e-12)
    absolute_relative = np.abs(signed_relative)
    consistent = absolute_relative <= consistency_relative_tolerance
    occluded = signed_relative > occlusion_relative_margin
    front_conflict = signed_relative < -consistency_relative_tolerance
    back_conflict = (
        (signed_relative > consistency_relative_tolerance)
        & (signed_relative <= occlusion_relative_margin)
    )
    comparable = ~occluded
    comparable_residuals = absolute_relative[comparable]
    comparable_count = int(np.count_nonzero(comparable))
    consistent_count = int(np.count_nonzero(consistent & comparable))
    front_conflict_count = int(np.count_nonzero(front_conflict & comparable))
    back_conflict_count = int(np.count_nonzero(back_conflict & comparable))
    occluded_count = int(np.count_nonzero(occluded))
    classes = np.full(projected_source.shape[0], _CLASS_BACK_CONFLICT, dtype=np.uint8)
    classes[consistent] = _CLASS_CONSISTENT
    classes[front_conflict] = _CLASS_FRONT_CONFLICT
    classes[occluded] = _CLASS_OCCLUDED

    denominator = max(comparable_count, 1)
    metrics = {
        "source_sampled_points": int(_deterministic_sample(source_points_reconstruction, maximum_points).shape[0]),
        "source_points_in_front_of_target_camera": int(source.shape[0]),
        "projected_in_frame": int(projected_pixels.shape[0]),
        "valid_target_depth_samples": int(projected_source.shape[0]),
        "comparable_samples": comparable_count,
        "consistent_samples": consistent_count,
        "front_conflict_samples": front_conflict_count,
        "back_conflict_samples": back_conflict_count,
        "occluded_samples": occluded_count,
        "consistent_ratio_of_comparable": float(consistent_count / denominator),
        "front_conflict_ratio_of_comparable": float(front_conflict_count / denominator),
        "back_conflict_ratio_of_comparable": float(back_conflict_count / denominator),
        "occluded_ratio_of_valid_target": float(occluded_count / max(projected_source.shape[0], 1)),
        "median_absolute_relative_depth_residual": _safe_quantile(comparable_residuals, 0.5),
        "p90_absolute_relative_depth_residual": _safe_quantile(comparable_residuals, 0.9),
        "p95_absolute_relative_depth_residual": _safe_quantile(comparable_residuals, 0.95),
        "consistency_relative_tolerance": float(consistency_relative_tolerance),
        "occlusion_relative_margin": float(occlusion_relative_margin),
    }
    diagnostics = {
        "points_reconstruction": np.ascontiguousarray(projected_source, dtype=np.float64),
        "classification": np.ascontiguousarray(classes, dtype=np.uint8),
        "signed_relative_depth_residual": np.ascontiguousarray(signed_relative, dtype=np.float32),
        "absolute_relative_depth_residual": np.ascontiguousarray(absolute_relative, dtype=np.float32),
    }
    return metrics, diagnostics


def _spatial_nn_diagnostic(
    points_a: np.ndarray,
    points_b: np.ndarray,
    *,
    scene_scale: float,
    maximum_points: int,
    support_scene_fraction: float,
) -> dict[str, Any]:
    a = _deterministic_sample(points_a, maximum_points)
    b = _deterministic_sample(points_b, maximum_points)
    if a.shape[0] == 0 or b.shape[0] == 0:
        raise ReconstructionValidationError("D2 spatial-NN diagnostic requires non-empty clouds.")
    support_radius = float(scene_scale * support_scene_fraction)
    tree_b = cKDTree(b)
    tree_a = cKDTree(a)
    d_ab, _ = tree_b.query(a, k=1, workers=1)
    d_ba, _ = tree_a.query(b, k=1, workers=1)

    def directed(values: np.ndarray) -> dict[str, Any]:
        supported = values <= support_radius
        supported_values = values[supported]
        return {
            "sample_count": int(values.size),
            "support_count": int(np.count_nonzero(supported)),
            "support_ratio": float(np.mean(supported)),
            "median_nn_scene_fraction_all": float(np.median(values) / scene_scale),
            "p90_nn_scene_fraction_all": float(np.quantile(values, 0.9) / scene_scale),
            "median_nn_scene_fraction_supported": (
                None if supported_values.size == 0 else float(np.median(supported_values) / scene_scale)
            ),
            "p90_nn_scene_fraction_supported": (
                None if supported_values.size == 0 else float(np.quantile(supported_values, 0.9) / scene_scale)
            ),
        }

    return {
        "support_radius_reconstruction_units": support_radius,
        "support_scene_fraction": float(support_scene_fraction),
        "a_to_b": directed(np.asarray(d_ab, dtype=np.float64)),
        "b_to_a": directed(np.asarray(d_ba, dtype=np.float64)),
    }


def _assess_direction(
    metrics: dict[str, Any],
    *,
    minimum_comparable_samples: int,
    minimum_consistent_ratio: float,
    maximum_median_relative_residual: float,
    maximum_p90_relative_residual: float,
    maximum_front_conflict_ratio: float,
) -> tuple[str, list[str]]:
    if int(metrics["comparable_samples"]) < minimum_comparable_samples:
        return "inconclusive", ["insufficient_comparable_overlap"]
    failures: list[str] = []
    consistent_ratio = float(metrics["consistent_ratio_of_comparable"])
    front_ratio = float(metrics["front_conflict_ratio_of_comparable"])
    median = metrics["median_absolute_relative_depth_residual"]
    p90 = metrics["p90_absolute_relative_depth_residual"]
    if consistent_ratio < minimum_consistent_ratio:
        failures.append("low_consistent_overlap_ratio")
    if median is None or float(median) > maximum_median_relative_residual:
        failures.append("median_depth_disagreement")
    if p90 is None or float(p90) > maximum_p90_relative_residual:
        failures.append("p90_depth_disagreement")
    if front_ratio > maximum_front_conflict_ratio:
        failures.append("front_surface_conflict")
    return ("pass" if not failures else "degraded"), failures


def _pair_status(status_a_to_b: str, status_b_to_a: str) -> str:
    values = {status_a_to_b, status_b_to_a}
    if "degraded" in values:
        return "degraded"
    if "inconclusive" in values:
        return "inconclusive"
    return "pass"


def _load_view_cloud(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as payload:
            points = np.asarray(payload["points_reconstruction"], dtype=np.float64)
            colors = np.asarray(payload["colors_rgb"], dtype=np.uint8)
            transform = np.asarray(payload["T_reconstruction_from_camera"], dtype=np.float64)
    except (OSError, ValueError, KeyError) as exc:
        raise ReconstructionValidationError(f"Unable to read D1 view cloud {path}: {exc}") from exc
    if points.ndim != 2 or points.shape[1:] != (3,) or colors.shape != points.shape or transform.shape != (4, 4):
        raise ReconstructionValidationError(f"Invalid D1 view cloud arrays: {path}")
    return points, colors, transform


def _load_aligned_depth(
    view_summary: dict[str, Any],
    *,
    sfm_width: int,
    sfm_height: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    manifest_path = Path(str(view_summary["da3_result_manifest"])).expanduser().resolve()
    manifest = read_da3_result_manifest(manifest_path)
    arrays = validate_da3_npz_result(manifest)
    if manifest.frame_count != 1 or arrays["depth"].shape[0] != 1:
        raise ReconstructionValidationError("D2 expects each retained D1 DA3 artifact to contain exactly one frame.")
    depth_sfm, adapter = _adapt_da3_depth_to_sfm_pixel_space(
        arrays["depth"][0],
        sfm_width=sfm_width,
        sfm_height=sfm_height,
    )
    scale = float(view_summary["scale_to_reconstruction_units"])
    if not isfinite(scale) or scale <= 0.0:
        raise ReconstructionValidationError("D2 found an invalid D1 depth-alignment scale.")
    return np.ascontiguousarray(depth_sfm.astype(np.float64) * scale), adapter


def main() -> int:
    args = _parse_args()
    d1_dir = args.d1_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    d1_summary = _read_json(d1_dir / "d1_summary.json")
    if not bool(d1_summary.get("pass")) or int(d1_summary.get("view_count", 0)) != _D2_VIEW_COUNT:
        raise ReconstructionValidationError("D2 requires a successful three-view D1 result.")
    raw = d1_summary.get("raw_accumulation", {})
    if bool(raw.get("spatial_merge_applied")) or bool(raw.get("voxel_fusion_applied")) or bool(raw.get("multi_view_filtering_applied")):
        raise ReconstructionValidationError("D2 must diagnose the unfused/unfiltered D1 accumulation.")
    views_payload = d1_summary.get("views")
    if not isinstance(views_payload, list) or len(views_payload) != _D2_VIEW_COUNT:
        raise ReconstructionValidationError("D2 requires exactly three D1 view summaries.")

    source_b2_dir = Path(str(d1_summary["source_b2_dir"])).expanduser().resolve()
    sfm_result = _sfm_result_from_payload(_read_json(source_b2_dir / "sfm_result.json"))
    cameras_by_keyframe = {item.keyframe_id: item for item in sfm_result.cameras}
    camera_models_by_id = {item.camera_id: item for item in sfm_result.camera_models}

    combined_npz = Path(str(d1_summary["artifacts"]["combined_npz"])).expanduser().resolve()
    try:
        with np.load(combined_npz, allow_pickle=False) as payload:
            sparse_landmarks = np.asarray(
                payload.get("sparse_landmarks_reconstruction", np.empty((0, 3))), dtype=np.float64
            )
    except (OSError, ValueError) as exc:
        raise ReconstructionValidationError(f"Unable to read D1 combined cloud: {exc}") from exc

    view_records: list[dict[str, Any]] = []
    view_points: list[np.ndarray] = []
    for index, item in enumerate(views_payload):
        if int(item.get("view_index", -1)) != index:
            raise ReconstructionValidationError("D2 requires D1 view indices 0,1,2 in order.")
        keyframe_id = str(item["keyframe_id"])
        camera_estimate = cameras_by_keyframe.get(keyframe_id)
        if camera_estimate is None:
            raise ReconstructionValidationError(f"D2 cannot find SfM camera estimate for {keyframe_id}.")
        camera_model = camera_models_by_id.get(camera_estimate.camera_id)
        if camera_model is None:
            raise ReconstructionValidationError(f"D2 cannot find SfM camera model {camera_estimate.camera_id}.")
        camera = reconstruction_camera_from_sfm_model(
            camera_model,
            keyframe_id=keyframe_id,
            reconstruction_frame_id="offline-sfm-reconstruction",
        )
        cloud_path = Path(str(item["cloud_npz"])).expanduser().resolve()
        points, colors, transform = _load_view_cloud(cloud_path)
        if not np.allclose(transform, camera_estimate.T_reconstruction_from_camera, rtol=0.0, atol=1e-9):
            raise ReconstructionValidationError("D2 detected camera-pose drift between D1 cloud and SfM result.")
        aligned_depth, pixel_adapter = _load_aligned_depth(
            item,
            sfm_width=camera_model.image_size.width,
            sfm_height=camera_model.image_size.height,
        )
        view_points.append(points)
        view_records.append(
            {
                "view_index": index,
                "keyframe_id": keyframe_id,
                "workspace_order": int(item["workspace_order"]),
                "scale_to_reconstruction_units": float(item["scale_to_reconstruction_units"]),
                "relative_scale_uncertainty": float(item["relative_scale_uncertainty"]),
                "camera_center_reconstruction": np.asarray(item["camera_center_reconstruction"], dtype=np.float64),
                "camera_estimate": camera_estimate,
                "camera": camera,
                "aligned_depth": aligned_depth,
                "points": points,
                "colors": colors,
                "pixel_adapter": pixel_adapter,
            }
        )

    scene_scale, scene_scale_source, scene_low, scene_high = _robust_scene_scale(
        sparse_landmarks, view_points
    )
    thresholds = {
        "consistency_relative_tolerance": float(args.consistency_relative_tolerance),
        "occlusion_relative_margin": float(args.occlusion_relative_margin),
        "minimum_comparable_samples": int(args.minimum_comparable_samples),
        "minimum_consistent_ratio": float(args.minimum_consistent_ratio),
        "maximum_median_relative_residual": float(args.maximum_median_relative_residual),
        "maximum_p90_relative_residual": float(args.maximum_p90_relative_residual),
        "maximum_front_conflict_ratio": float(args.maximum_front_conflict_ratio),
        "nn_support_scene_fraction": float(args.nn_support_scene_fraction),
    }

    pair_summaries: list[dict[str, Any]] = []
    for a_index, b_index in _D2_PAIRS:
        a = view_records[a_index]
        b = view_records[b_index]
        a_to_b, diag_a_to_b = _directed_reprojection_diagnostic(
            a["points"],
            target_camera=b["camera"],
            target_T_reconstruction_from_camera=b["camera_estimate"].T_reconstruction_from_camera,
            target_aligned_depth=b["aligned_depth"],
            maximum_points=int(args.max_points_per_direction),
            consistency_relative_tolerance=float(args.consistency_relative_tolerance),
            occlusion_relative_margin=float(args.occlusion_relative_margin),
        )
        b_to_a, diag_b_to_a = _directed_reprojection_diagnostic(
            b["points"],
            target_camera=a["camera"],
            target_T_reconstruction_from_camera=a["camera_estimate"].T_reconstruction_from_camera,
            target_aligned_depth=a["aligned_depth"],
            maximum_points=int(args.max_points_per_direction),
            consistency_relative_tolerance=float(args.consistency_relative_tolerance),
            occlusion_relative_margin=float(args.occlusion_relative_margin),
        )
        status_ab, failures_ab = _assess_direction(
            a_to_b,
            minimum_comparable_samples=int(args.minimum_comparable_samples),
            minimum_consistent_ratio=float(args.minimum_consistent_ratio),
            maximum_median_relative_residual=float(args.maximum_median_relative_residual),
            maximum_p90_relative_residual=float(args.maximum_p90_relative_residual),
            maximum_front_conflict_ratio=float(args.maximum_front_conflict_ratio),
        )
        status_ba, failures_ba = _assess_direction(
            b_to_a,
            minimum_comparable_samples=int(args.minimum_comparable_samples),
            minimum_consistent_ratio=float(args.minimum_consistent_ratio),
            maximum_median_relative_residual=float(args.maximum_median_relative_residual),
            maximum_p90_relative_residual=float(args.maximum_p90_relative_residual),
            maximum_front_conflict_ratio=float(args.maximum_front_conflict_ratio),
        )
        pair_status = _pair_status(status_ab, status_ba)
        role = "adjacent_required" if abs(a_index - b_index) == 1 else "endpoint_informational"
        baseline = float(np.linalg.norm(a["camera_center_reconstruction"] - b["camera_center_reconstruction"]))
        nn = _spatial_nn_diagnostic(
            a["points"],
            b["points"],
            scene_scale=scene_scale,
            maximum_points=int(args.nn_max_points_per_view),
            support_scene_fraction=float(args.nn_support_scene_fraction),
        )
        diagnostic_path = output_dir / f"d2_pair_{a_index}{b_index}_diagnostic.npz"
        np.savez_compressed(
            diagnostic_path,
            pair=np.asarray([a_index, b_index], dtype=np.uint8),
            keyframe_ids=np.asarray([a["keyframe_id"], b["keyframe_id"]], dtype="U160"),
            a_to_b_points_reconstruction=diag_a_to_b["points_reconstruction"],
            a_to_b_classification=diag_a_to_b["classification"],
            a_to_b_signed_relative_depth_residual=diag_a_to_b["signed_relative_depth_residual"],
            a_to_b_absolute_relative_depth_residual=diag_a_to_b["absolute_relative_depth_residual"],
            b_to_a_points_reconstruction=diag_b_to_a["points_reconstruction"],
            b_to_a_classification=diag_b_to_a["classification"],
            b_to_a_signed_relative_depth_residual=diag_b_to_a["signed_relative_depth_residual"],
            b_to_a_absolute_relative_depth_residual=diag_b_to_a["absolute_relative_depth_residual"],
            sparse_landmarks_reconstruction=sparse_landmarks,
        )
        pair_summaries.append(
            {
                "pair": [a_index, b_index],
                "pair_label": f"{a_index}-{b_index}",
                "role": role,
                "status": pair_status,
                "camera_baseline_reconstruction_units": baseline,
                "camera_baseline_scene_fraction": float(baseline / scene_scale),
                "scale_ratio_max_to_min": float(
                    max(a["scale_to_reconstruction_units"], b["scale_to_reconstruction_units"])
                    / min(a["scale_to_reconstruction_units"], b["scale_to_reconstruction_units"])
                ),
                "a_to_b": {**a_to_b, "status": status_ab, "failure_reasons": failures_ab},
                "b_to_a": {**b_to_a, "status": status_ba, "failure_reasons": failures_ba},
                "spatial_nearest_neighbour_secondary": nn,
                "diagnostic_npz": str(diagnostic_path.resolve()),
            }
        )

    required = [item for item in pair_summaries if item["role"] == "adjacent_required"]
    required_statuses = {str(item["status"]) for item in required}
    if "degraded" in required_statuses:
        overall_status = "degraded"
        passed = False
        exit_code = 2
    elif "inconclusive" in required_statuses:
        overall_status = "inconclusive"
        passed = False
        exit_code = 3
    else:
        overall_status = "pass"
        passed = True
        exit_code = 0

    summary = {
        "phase": "D2",
        "pass": passed,
        "status": overall_status,
        "source_d1_dir": str(d1_dir),
        "source_d1_pass": True,
        "source_sfm_session_id": str(d1_summary["source_sfm_session_id"]),
        "view_count": _D2_VIEW_COUNT,
        "pair_count": len(pair_summaries),
        "scene_scale": {
            "robust_diagonal_reconstruction_units": scene_scale,
            "source": scene_scale_source,
            "quantile_bounds_min": scene_low,
            "quantile_bounds_max": scene_high,
            "physical_unit_claimed": False,
        },
        "thresholds": thresholds,
        "pairs": pair_summaries,
        "decision": {
            "required_pairs": ["0-1", "1-2"],
            "informational_pairs": ["0-2"],
            "rule": "Both adjacent required pairs must pass bidirectional reprojection-depth consistency. Endpoint pair is diagnostic only because temporal-third views may have limited direct overlap.",
            "status": overall_status,
        },
        "authority": {
            "sfm_scale_status": str(d1_summary["authority"]["sfm_scale_status"]),
            "sfm_world_alignment_status": str(d1_summary["authority"]["sfm_world_alignment_status"]),
            "metric_scale_claimed": False,
            "canonical_geometry_authoritative": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "spatial_merge_applied": False,
            "voxel_fusion_applied": False,
            "multi_view_filtering_applied": False,
            "note": "D2 validation-only overlap diagnostics; raw D1 geometry remains unchanged.",
        },
    }
    summary_path = _write_json(output_dir / "d2_summary.json", summary)

    print(f"\nD2 three-view overlap diagnostics: {overall_status.upper()}")
    print(
        f"scene_scale={scene_scale:.6f} reconstruction_units source={scene_scale_source} "
        "physical_unit_claimed=False"
    )
    for item in pair_summaries:
        a_to_b = item["a_to_b"]
        b_to_a = item["b_to_a"]
        print(
            f"pair[{item['pair_label']}] role={item['role']} status={item['status']} "
            f"baseline_scene_fraction={item['camera_baseline_scene_fraction']:.4f} "
            f"scale_ratio={item['scale_ratio_max_to_min']:.4f}"
        )
        print(
            f"  {item['pair'][0]}->{item['pair'][1]} status={a_to_b['status']} "
            f"comparable={a_to_b['comparable_samples']} "
            f"consistent={a_to_b['consistent_ratio_of_comparable']:.3f} "
            f"front_conflict={a_to_b['front_conflict_ratio_of_comparable']:.3f} "
            f"median={_format_optional(a_to_b['median_absolute_relative_depth_residual'])} "
            f"p90={_format_optional(a_to_b['p90_absolute_relative_depth_residual'])} "
            f"occluded={a_to_b['occluded_ratio_of_valid_target']:.3f}"
        )
        print(
            f"  {item['pair'][1]}->{item['pair'][0]} status={b_to_a['status']} "
            f"comparable={b_to_a['comparable_samples']} "
            f"consistent={b_to_a['consistent_ratio_of_comparable']:.3f} "
            f"front_conflict={b_to_a['front_conflict_ratio_of_comparable']:.3f} "
            f"median={_format_optional(b_to_a['median_absolute_relative_depth_residual'])} "
            f"p90={_format_optional(b_to_a['p90_absolute_relative_depth_residual'])} "
            f"occluded={b_to_a['occluded_ratio_of_valid_target']:.3f}"
        )
        print(f"  diagnostic={item['diagnostic_npz']}")
    print("spatial_merge_applied=False voxel_fusion_applied=False multi_view_filtering_applied=False")
    print("sfm_scale_status=arbitrary metric_scale_claimed=False")
    print("canonical_geometry_authoritative=False measurement_authoritative=False certified_metrology=False")
    print(f"summary={summary_path}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

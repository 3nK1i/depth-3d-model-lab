#!/usr/bin/env python3
"""E3 qualified scene-fusion expansion from the trusted E2 ten-view seed.

E3 grows the already-qualified E2 seed outward in registered SfM workspace
order.  New views are never admitted by temporal order alone: each candidate
receives isolated DA3 depth, SfM-depth scale alignment, canonical projection,
and D2-style bidirectional reprojection/depth-consistency qualification.

A monotonic pass-path is selected independently on the left and right of the
seed.  Small skip bridges are allowed so a known-bad adjacent transition (for
example 15<->16) does not force a globally useful view to be discarded.  The
selected expanded chain is then fused with the existing DenseVoxelFusion
kernel using the exact E1 voxel size.

E3 expands coverage only.  Formal confidence/artifact filtering remains Phase
F; canonical persistence remains Phase G; UI integration remains Phase H.
Metric scale, measurement authority, canonical persistence authority, mesh
authority, and certified metrology remain explicitly unpromoted.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from math import isfinite
from pathlib import Path
import sys
from typing import Any, Iterable

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction import (  # noqa: E402
    DenseDepthUnit,
    DenseFusionFrame,
    DenseFusionPolicy,
    DenseSceneProfile,
    SfmScaleStatus,
    align_sfm_keyframe_dense_depth,
    arbitrary_scale_reference,
    camera_geometry_from_estimated_camera,
    fuse_dense_frames,
    project_sfm_keyframe_dense_rgb,
    reconstruction_camera_from_sfm_model,
    reconstruction_unit_depth_alignment,
)
from core.reconstruction.da3 import (  # noqa: E402
    DA3_EFFICIENT_MODEL_ID,
    probe_da3_environment,
    validate_da3_npz_result,
)
from core.reconstruction.da3_provider import (  # noqa: E402
    DA3ExternalWorkerRunner,
    DA3InputManifest,
    DA3ProviderProfile,
    write_da3_input_manifest,
)
from core.reconstruction.errors import ReconstructionValidationError  # noqa: E402
from scripts.run_b2_sfm_dense_real_alignment import _adapt_da3_depth_to_sfm_pixel_space  # noqa: E402
from scripts.run_d1_three_view_canonical_accumulation import (  # noqa: E402
    _all_sparse_landmarks,
    _observation_counts,
    _sfm_result_from_payload,
)
from scripts.run_d2_three_view_overlap_diagnostics import (  # noqa: E402
    _assess_direction,
    _directed_reprojection_diagnostic,
    _pair_status,
    _spatial_nn_diagnostic,
)
from scripts.run_e1_qualified_ten_view_voxel_fusion import (  # noqa: E402
    _save_cloud_npz,
    _support_histogram,
    _write_binary_ply,
)

_DEFAULT_E2_DIR = Path("/tmp/eia_e2_consistency_aware_fusion_refinement")
_DEFAULT_OUTPUT_DIR = Path("/tmp/eia_e3_qualified_scene_fusion_expansion")
_DEFAULT_DA3_PYTHON = Path.home() / "anaconda3/envs/eia-da3/bin/python"
_DEFAULT_DA3_REPO = Path.home() / "Desktop/eia-da3-backends/depth-anything-3"
_DA3_WORKER = ROOT / "scripts/da3_workers/da3_worker.py"

# The numerical gates are intentionally the unchanged D4.1/D2 gates.  At
# runtime E3 validates that the D4.3 summary contains these exact values and
# then uses the values from that evidence rather than silently inventing new
# qualification thresholds.
_REQUIRED_DENSE_THRESHOLD_KEYS = (
    "consistency_relative_tolerance",
    "occlusion_relative_margin",
    "minimum_comparable_samples",
    "minimum_consistent_ratio",
    "maximum_median_relative_residual",
    "maximum_p90_relative_residual",
    "maximum_front_conflict_ratio",
    "nn_support_scene_fraction",
)


@dataclass(frozen=True)
class GrowthPath:
    anchor_order: int
    orders_anchor_outward: tuple[int, ...]
    worst_consistency: float
    worst_median_residual: float

    @property
    def added_orders(self) -> tuple[int, ...]:
        return self.orders_anchor_outward[1:]



def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e2-dir", type=Path, default=_DEFAULT_E2_DIR)
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR)
    parser.add_argument("--da3-python", type=Path, default=_DEFAULT_DA3_PYTHON)
    parser.add_argument("--da3-repo", type=Path, default=_DEFAULT_DA3_REPO)
    parser.add_argument("--da3-model", default="")
    parser.add_argument("--da3-model-revision", default="")
    parser.add_argument("--process-res", type=int, default=392)
    parser.add_argument("--da3-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--minimum-correspondences", type=int, default=8)
    parser.add_argument("--maximum-relative-residual", type=float, default=0.35)
    parser.add_argument("--left-candidate-count", type=int, default=5)
    parser.add_argument("--right-candidate-count", type=int, default=8)
    parser.add_argument("--max-bridge-gap", type=int, default=3)
    parser.add_argument("--minimum-new-views", type=int, default=8)
    parser.add_argument("--minimum-new-views-per-side", type=int, default=2)
    parser.add_argument("--max-points-per-direction", type=int, default=120_000)
    parser.add_argument("--nn-max-points-per-view", type=int, default=60_000)
    parser.add_argument("--max-voxels", type=int, default=1_500_000)
    parser.add_argument("--minimum-supported-views", type=int, default=2)
    parser.add_argument("--high-support-views", type=int, default=3)
    parser.add_argument("--minimum-supported-voxels", type=int, default=1_000)
    parser.add_argument("--sfm-focal-relative-uncertainty", type=float, default=0.20)
    args = parser.parse_args()
    if args.process_res <= 0:
        parser.error("--process-res must be positive.")
    if args.da3_timeout_seconds <= 0.0:
        parser.error("--da3-timeout-seconds must be positive.")
    if args.minimum_correspondences < 3:
        parser.error("--minimum-correspondences must be at least 3.")
    if not 0.0 < args.maximum_relative_residual < 1.0:
        parser.error("--maximum-relative-residual must be in (0,1).")
    if args.left_candidate_count < 0 or args.right_candidate_count < 0:
        parser.error("Candidate counts must be non-negative.")
    if not 1 <= args.max_bridge_gap <= 6:
        parser.error("--max-bridge-gap must be in [1,6].")
    if args.minimum_new_views < 1:
        parser.error("--minimum-new-views must be positive.")
    if args.minimum_new_views_per_side < 0:
        parser.error("--minimum-new-views-per-side must be non-negative.")
    if args.max_points_per_direction <= 0 or args.nn_max_points_per_view <= 0:
        parser.error("Diagnostic point budgets must be positive.")
    if args.max_voxels <= 0:
        parser.error("--max-voxels must be positive.")
    if not 2 <= args.minimum_supported_views <= 40:
        parser.error("--minimum-supported-views must be in [2,40].")
    if not args.minimum_supported_views <= args.high_support_views <= 40:
        parser.error("--high-support-views must be >= minimum-supported-views and <= 40.")
    if args.minimum_supported_voxels <= 0:
        parser.error("--minimum-supported-voxels must be positive.")
    if not 0.0 <= args.sfm_focal_relative_uncertainty < 1.0:
        parser.error("--sfm-focal-relative-uncertainty must be in [0,1).")
    return args



def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))



def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path



def _validate_e2_summary(summary: dict[str, Any]) -> None:
    if str(summary.get("phase", "")) != "E2":
        raise ReconstructionValidationError("E3 requires an E2 summary.")
    if not bool(summary.get("pass")) or str(summary.get("status", "")) != "pass":
        raise ReconstructionValidationError("E3 requires E2 status=pass.")
    decision = summary.get("decision")
    if not isinstance(decision, dict) or not bool(decision.get("authorizes_next_scene_fusion_expansion_milestone")):
        raise ReconstructionValidationError("E3 requires E2 scene-expansion authorization.")
    orders = tuple(int(v) for v in summary.get("selected_workspace_orders", ()))
    keyframes = tuple(str(v) for v in summary.get("selected_keyframe_ids", ()))
    if len(orders) != 10 or len(keyframes) != 10 or len(set(orders)) != 10 or len(set(keyframes)) != 10:
        raise ReconstructionValidationError("E3 requires the exact ten-view E2 seed.")
    if tuple(sorted(orders)) != orders:
        raise ReconstructionValidationError("E3 E2 seed orders must be strictly increasing.")



def _validate_thresholds(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ReconstructionValidationError("E3 requires D4.3 unchanged D4.1/D2 threshold evidence.")
    missing = [key for key in _REQUIRED_DENSE_THRESHOLD_KEYS if key not in payload]
    if missing:
        raise ReconstructionValidationError("E3 missing dense qualification thresholds: " + ", ".join(missing))
    result = {key: payload[key] for key in _REQUIRED_DENSE_THRESHOLD_KEYS}
    if float(result["consistency_relative_tolerance"]) >= float(result["occlusion_relative_margin"]):
        raise ReconstructionValidationError("E3 threshold evidence is internally inconsistent.")
    return result



def _candidate_workspace_orders(
    registered_orders: Iterable[int],
    seed_orders: Iterable[int],
    *,
    left_count: int,
    right_count: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    registered = tuple(sorted(set(int(v) for v in registered_orders)))
    seed = tuple(sorted(set(int(v) for v in seed_orders)))
    if not seed:
        raise ReconstructionValidationError("E3 seed orders cannot be empty.")
    left = tuple(v for v in registered if v < seed[0])
    right = tuple(v for v in registered if v > seed[-1])
    if left_count:
        left = left[-int(left_count):]
    else:
        left = ()
    if right_count:
        right = right[:int(right_count)]
    else:
        right = ()
    return left, right



def _pair_key(order_a: int, order_b: int) -> tuple[int, int]:
    a, b = int(order_a), int(order_b)
    return (a, b) if a < b else (b, a)



def _pair_quality(pair: dict[str, Any]) -> tuple[float, float]:
    if str(pair.get("status")) != "pass":
        return 0.0, float("inf")
    directions = (pair["a_to_b"], pair["b_to_a"])
    worst_consistency = min(float(item["consistent_ratio_of_comparable"]) for item in directions)
    medians = [item["median_absolute_relative_depth_residual"] for item in directions]
    worst_median = max(float(value) for value in medians if value is not None)
    return worst_consistency, worst_median



def _best_growth_path(
    *,
    anchor_order: int,
    candidate_orders: Iterable[int],
    pair_summaries: dict[tuple[int, int], dict[str, Any]],
    direction: str,
) -> GrowthPath:
    """Return the longest monotonic pass-path from anchor toward one side.

    View count is primary, then spatial/order span, then worst-case dense
    consistency, then worst median residual.  Only pair status=pass edges can
    enter a path; degraded/inconclusive edges are never softened.
    """

    if direction not in {"left", "right"}:
        raise ReconstructionValidationError("E3 growth direction must be left or right.")
    anchor = int(anchor_order)
    candidates = sorted(set(int(v) for v in candidate_orders), reverse=(direction == "left"))
    if any((v >= anchor if direction == "left" else v <= anchor) for v in candidates):
        raise ReconstructionValidationError("E3 candidate orders are on the wrong side of the anchor.")

    normalized_pairs = {
        _pair_key(key[0], key[1]): value for key, value in pair_summaries.items()
    }
    best_to: dict[int, tuple[tuple[int, ...], float, float]] = {
        anchor: ((anchor,), 1.0, 0.0)
    }
    seen: list[int] = [anchor]
    for node in candidates:
        options: list[tuple[tuple[int, ...], float, float]] = []
        for predecessor in seen:
            if predecessor not in best_to:
                continue
            if direction == "right" and predecessor >= node:
                continue
            if direction == "left" and predecessor <= node:
                continue
            pair = normalized_pairs.get(_pair_key(predecessor, node))
            if pair is None or str(pair.get("status")) != "pass":
                continue
            edge_consistency, edge_median = _pair_quality(pair)
            path, current_worst_consistency, current_worst_median = best_to[predecessor]
            options.append(
                (
                    path + (node,),
                    min(current_worst_consistency, edge_consistency),
                    max(current_worst_median, edge_median),
                )
            )
        if options:
            best_to[node] = max(
                options,
                key=lambda item: (
                    len(item[0]),
                    abs(item[0][-1] - anchor),
                    item[1],
                    -item[2],
                ),
            )
        seen.append(node)

    best = max(
        best_to.values(),
        key=lambda item: (
            len(item[0]),
            abs(item[0][-1] - anchor),
            item[1],
            -item[2],
        ),
    )
    return GrowthPath(
        anchor_order=anchor,
        orders_anchor_outward=tuple(int(v) for v in best[0]),
        worst_consistency=float(best[1]),
        worst_median_residual=float(best[2]),
    )





def _expanded_support_masks(
    counts: object,
    *,
    minimum_supported_views: int,
    high_support_views: int,
    maximum_views: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(counts, dtype=np.int64)
    maximum = int(maximum_views)
    minimum = int(minimum_supported_views)
    high = int(high_support_views)
    if values.ndim != 1 or values.size == 0 or np.any(values <= 0):
        raise ReconstructionValidationError("E3 view-observation counts must be a non-empty positive vector.")
    if maximum < 2 or not 2 <= minimum <= maximum or not minimum <= high <= maximum:
        raise ReconstructionValidationError("E3 support thresholds must fit the selected expanded view count.")
    if np.any(values > maximum):
        raise ReconstructionValidationError("E3 fused support count exceeds the selected view count.")
    return values >= minimum, values >= high


def _robust_diagonal(points: object) -> float:
    xyz = np.asarray(points, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) == 0:
        raise ReconstructionValidationError("E3 robust scene span requires non-empty Nx3 points.")
    low = np.quantile(xyz, 0.02, axis=0)
    high = np.quantile(xyz, 0.98, axis=0)
    value = float(np.linalg.norm(high - low))
    if not isfinite(value) or value <= 0.0:
        raise ReconstructionValidationError("E3 robust scene span must be finite and positive.")
    return value



def _load_cloud_record(
    *,
    order: int,
    keyframe_id: str,
    cloud_npz: Path,
    sfm_result: Any,
    scale_to_reconstruction_units: float | None,
    source: str,
) -> dict[str, Any]:
    estimates = {item.keyframe_id: item for item in sfm_result.cameras}
    models = {item.camera_id: item for item in sfm_result.camera_models}
    workspaces = {item.keyframe_id: item for item in sfm_result.workspace.images}
    estimate = estimates.get(keyframe_id)
    workspace = workspaces.get(keyframe_id)
    if estimate is None or workspace is None:
        raise ReconstructionValidationError(f"E3 missing SfM/workspace evidence for {keyframe_id}.")
    model = models.get(estimate.camera_id)
    if model is None:
        raise ReconstructionValidationError(f"E3 missing SfM camera model {estimate.camera_id}.")
    path = Path(cloud_npz).expanduser().resolve()
    try:
        with np.load(path, allow_pickle=False) as payload:
            points = np.asarray(payload["points_reconstruction"], dtype=np.float64)
            colors = np.asarray(payload["colors_rgb"], dtype=np.uint8)
            aligned_depth = np.asarray(payload["aligned_depth_reconstruction_units"], dtype=np.float64)
            transform = np.asarray(payload["T_reconstruction_from_camera"], dtype=np.float64)
    except (OSError, ValueError, KeyError) as exc:
        raise ReconstructionValidationError(f"E3 unable to read source cloud {path}: {exc}") from exc
    if points.ndim != 2 or points.shape[1] != 3 or colors.shape != points.shape:
        raise ReconstructionValidationError(f"E3 source cloud arrays are invalid: {path}")
    if aligned_depth.shape != (model.image_size.height, model.image_size.width):
        raise ReconstructionValidationError(f"E3 aligned depth shape disagrees with SfM camera for order {order}.")
    if transform.shape != (4, 4) or not np.allclose(
        transform, estimate.T_reconstruction_from_camera, rtol=0.0, atol=1e-9
    ):
        raise ReconstructionValidationError(f"E3 detected pose drift for workspace order {order}.")
    camera = reconstruction_camera_from_sfm_model(
        model,
        keyframe_id=keyframe_id,
        reconstruction_frame_id="offline-sfm-reconstruction",
    )
    return {
        "workspace_order": int(order),
        "keyframe_id": str(keyframe_id),
        "frame_id": str(estimate.frame_id),
        "camera_id": str(estimate.camera_id),
        "camera_model_name": str(model.backend_model_name),
        "source": str(source),
        "cloud_npz": str(path),
        "points": points,
        "colors": colors,
        "aligned_depth": aligned_depth,
        "T_reconstruction_from_camera": transform,
        "camera_center_reconstruction": np.asarray(estimate.camera_center_reconstruction, dtype=np.float64),
        "camera_estimate": estimate,
        "camera": camera,
        "workspace": workspace,
        "scale_to_reconstruction_units": (
            None if scale_to_reconstruction_units is None else float(scale_to_reconstruction_units)
        ),
    }



def _generate_candidate_view(
    *,
    order: int,
    sfm_result: Any,
    b2_summary: dict[str, Any],
    output_dir: Path,
    runner: DA3ExternalWorkerRunner,
    da3_probe: Any,
    minimum_correspondences: int,
    maximum_relative_residual: float,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    estimates = {item.keyframe_id: item for item in sfm_result.cameras}
    models = {item.camera_id: item for item in sfm_result.camera_models}
    candidates = [
        item for item in sfm_result.workspace.images
        if int(item.order) == int(order) and item.keyframe_id in estimates
    ]
    if len(candidates) != 1:
        raise ReconstructionValidationError(f"E3 requires exactly one registered workspace image at order {order}.")
    workspace = candidates[0]
    estimate = estimates[workspace.keyframe_id]
    model = models.get(estimate.camera_id)
    if model is None:
        raise ReconstructionValidationError(f"E3 candidate order {order} has no SfM camera model.")
    rgb_path = Path(sfm_result.workspace.workspace_root).resolve() / workspace.staged_image_path
    bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ReconstructionValidationError(f"E3 could not decode candidate RGB keyframe: {rgb_path}")
    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)

    root = output_dir / f"candidate_order_{int(order):02d}"
    input_manifest = write_da3_input_manifest(
        root / "input_manifest.json",
        DA3InputManifest(
            request_id=f"e3-{sfm_result.session_id}-order-{int(order)}",
            image_paths=(str(rgb_path.resolve()),),
            metadata={
                "phase": "E3",
                "session_id": sfm_result.session_id,
                "workspace_order": int(order),
                "sfm_keyframe_id": estimate.keyframe_id,
                "purpose": "qualified_scene_fusion_expansion_candidate",
                "canonical_geometry_authoritative": False,
                "certified_metrology": False,
            },
        ),
    )
    print(f"E3: running isolated DA3 for candidate order={order}...", flush=True)
    manifest_path, manifest = runner.run(input_manifest, root / "worker_output")
    arrays = validate_da3_npz_result(manifest)
    if manifest.frame_count != 1 or arrays["depth"].shape[0] != 1:
        raise ReconstructionValidationError("E3 candidate DA3 result must contain exactly one frame.")
    depth_sfm, pixel_adapter = _adapt_da3_depth_to_sfm_pixel_space(
        arrays["depth"][0],
        sfm_width=model.image_size.width,
        sfm_height=model.image_size.height,
    )
    evidence = align_sfm_keyframe_dense_depth(
        sfm_result,
        keyframe_id=estimate.keyframe_id,
        dense_depth_source_units=depth_sfm,
        depth_provenance=(
            f"Depth Anything 3 model={manifest.model_id} revision={manifest.model_revision};"
            f"source_revision={manifest.source_revision};pixel_adapter={pixel_adapter['mode']};"
            f"E3 expansion candidate order {order}; arbitrary SfM gauge"
        ),
        minimum_correspondences=int(minimum_correspondences),
        maximum_relative_residual=float(maximum_relative_residual),
    )
    evidence_path = _write_json(root / "alignment.json", evidence.as_dict())
    base_summary: dict[str, Any] = {
        "workspace_order": int(order),
        "source_sequence_index": int(workspace.source_sequence_index),
        "keyframe_id": str(estimate.keyframe_id),
        "frame_id": str(estimate.frame_id),
        "camera_id": str(estimate.camera_id),
        "camera_model_name": str(model.backend_model_name),
        "candidate_correspondences": int(evidence.candidate_observations),
        "valid_correspondences": int(evidence.valid_correspondences),
        "inlier_correspondences": int(evidence.inlier_correspondences),
        "median_relative_residual": evidence.median_relative_residual,
        "p90_relative_residual": evidence.p90_relative_residual,
        "pixel_adapter": pixel_adapter,
        "alignment_evidence_json": str(evidence_path.resolve()),
        "da3_result_manifest": str(manifest_path.resolve()),
        "alignment_accepted": bool(evidence.is_accepted and evidence.alignment is not None),
        "rejection_code": None if evidence.rejection_code is None else evidence.rejection_code.value,
        "da3_environment_revision": str(da3_probe.source_revision),
    }
    if not evidence.is_accepted or evidence.alignment is None:
        return None, base_summary

    projection = project_sfm_keyframe_dense_rgb(
        estimate,
        model,
        timestamp_ns=workspace.timestamp_ns,
        rgb=rgb,
        dense_depth_source_units=depth_sfm,
        alignment_evidence=evidence,
        pixel_stride=1,
        reconstruction_frame_id="offline-sfm-reconstruction",
    )
    cloud_path = root / "canonical_cloud.npz"
    np.savez_compressed(
        cloud_path,
        points_reconstruction=projection.points_reconstruction,
        colors_rgb=projection.colors_rgb,
        aligned_depth_reconstruction_units=np.asarray(
            depth_sfm * evidence.alignment.scale_to_reconstruction_units,
            dtype=np.float64,
        ),
        T_reconstruction_from_camera=estimate.T_reconstruction_from_camera,
        camera_center_reconstruction=estimate.camera_center_reconstruction,
        keyframe_id=np.asarray(estimate.keyframe_id),
    )
    base_summary.update({
        "scale_to_reconstruction_units": float(evidence.alignment.scale_to_reconstruction_units),
        "relative_scale_uncertainty": float(evidence.alignment.relative_scale_uncertainty),
        "point_count": int(projection.points_reconstruction.shape[0]),
        "cloud_npz": str(cloud_path.resolve()),
        "bounds_min_reconstruction": projection.points_reconstruction.min(axis=0).tolist(),
        "bounds_max_reconstruction": projection.points_reconstruction.max(axis=0).tolist(),
        "camera_center_reconstruction": estimate.camera_center_reconstruction.tolist(),
    })
    record = _load_cloud_record(
        order=int(order),
        keyframe_id=str(estimate.keyframe_id),
        cloud_npz=cloud_path,
        sfm_result=sfm_result,
        scale_to_reconstruction_units=float(evidence.alignment.scale_to_reconstruction_units),
        source="e3_new_candidate",
    )
    return record, base_summary



def _diagnose_pair(
    *,
    a: dict[str, Any],
    b: dict[str, Any],
    thresholds: dict[str, Any],
    scene_scale: float,
    sparse_landmarks: np.ndarray,
    output_dir: Path,
    max_points_per_direction: int,
    nn_max_points_per_view: int,
) -> dict[str, Any]:
    order_a, order_b = int(a["workspace_order"]), int(b["workspace_order"])
    if order_a > order_b:
        a, b = b, a
        order_a, order_b = order_b, order_a
    try:
        a_to_b, diag_a_to_b = _directed_reprojection_diagnostic(
            a["points"],
            target_camera=b["camera"],
            target_T_reconstruction_from_camera=b["camera_estimate"].T_reconstruction_from_camera,
            target_aligned_depth=b["aligned_depth"],
            maximum_points=int(max_points_per_direction),
            consistency_relative_tolerance=float(thresholds["consistency_relative_tolerance"]),
            occlusion_relative_margin=float(thresholds["occlusion_relative_margin"]),
        )
        b_to_a, diag_b_to_a = _directed_reprojection_diagnostic(
            b["points"],
            target_camera=a["camera"],
            target_T_reconstruction_from_camera=a["camera_estimate"].T_reconstruction_from_camera,
            target_aligned_depth=a["aligned_depth"],
            maximum_points=int(max_points_per_direction),
            consistency_relative_tolerance=float(thresholds["consistency_relative_tolerance"]),
            occlusion_relative_margin=float(thresholds["occlusion_relative_margin"]),
        )
        status_ab, failures_ab = _assess_direction(
            a_to_b,
            minimum_comparable_samples=int(thresholds["minimum_comparable_samples"]),
            minimum_consistent_ratio=float(thresholds["minimum_consistent_ratio"]),
            maximum_median_relative_residual=float(thresholds["maximum_median_relative_residual"]),
            maximum_p90_relative_residual=float(thresholds["maximum_p90_relative_residual"]),
            maximum_front_conflict_ratio=float(thresholds["maximum_front_conflict_ratio"]),
        )
        status_ba, failures_ba = _assess_direction(
            b_to_a,
            minimum_comparable_samples=int(thresholds["minimum_comparable_samples"]),
            minimum_consistent_ratio=float(thresholds["minimum_consistent_ratio"]),
            maximum_median_relative_residual=float(thresholds["maximum_median_relative_residual"]),
            maximum_p90_relative_residual=float(thresholds["maximum_p90_relative_residual"]),
            maximum_front_conflict_ratio=float(thresholds["maximum_front_conflict_ratio"]),
        )
        status = _pair_status(status_ab, status_ba)
        nn = _spatial_nn_diagnostic(
            a["points"],
            b["points"],
            scene_scale=float(scene_scale),
            maximum_points=int(nn_max_points_per_view),
            support_scene_fraction=float(thresholds["nn_support_scene_fraction"]),
        )
        diagnostic = output_dir / f"e3_pair_{order_a:02d}_{order_b:02d}_diagnostic.npz"
        np.savez_compressed(
            diagnostic,
            workspace_orders=np.asarray([order_a, order_b], dtype=np.int32),
            keyframe_ids=np.asarray([a["keyframe_id"], b["keyframe_id"]], dtype="U160"),
            a_to_b_points_reconstruction=diag_a_to_b["points_reconstruction"],
            a_to_b_classification=diag_a_to_b["classification"],
            a_to_b_signed_relative_depth_residual=diag_a_to_b["signed_relative_depth_residual"],
            a_to_b_absolute_relative_depth_residual=diag_a_to_b["absolute_relative_depth_residual"],
            b_to_a_points_reconstruction=diag_b_to_a["points_reconstruction"],
            b_to_a_classification=diag_b_to_a["classification"],
            b_to_a_signed_relative_depth_residual=diag_b_to_a["signed_relative_depth_residual"],
            b_to_a_absolute_relative_depth_residual=diag_b_to_a["absolute_relative_depth_residual"],
            sparse_landmarks_reconstruction=np.asarray(sparse_landmarks, dtype=np.float64),
        )
        baseline = float(np.linalg.norm(
            a["camera_center_reconstruction"] - b["camera_center_reconstruction"]
        ))
        scales = [
            value for value in (
                a.get("scale_to_reconstruction_units"),
                b.get("scale_to_reconstruction_units"),
            ) if value is not None
        ]
        scale_ratio = None
        if len(scales) == 2 and min(scales) > 0.0:
            scale_ratio = float(max(scales) / min(scales))
        return {
            "workspace_orders": [order_a, order_b],
            "keyframe_ids": [str(a["keyframe_id"]), str(b["keyframe_id"])],
            "order_gap": int(order_b - order_a),
            "status": status,
            "camera_baseline_reconstruction_units": baseline,
            "camera_baseline_scene_fraction": float(baseline / scene_scale),
            "scale_ratio_max_to_min": scale_ratio,
            "a_to_b": {**a_to_b, "status": status_ab, "failure_reasons": failures_ab},
            "b_to_a": {**b_to_a, "status": status_ba, "failure_reasons": failures_ba},
            "spatial_nearest_neighbour_secondary": nn,
            "diagnostic_npz": str(diagnostic.resolve()),
        }
    except ReconstructionValidationError as exc:
        return {
            "workspace_orders": [order_a, order_b],
            "keyframe_ids": [str(a["keyframe_id"]), str(b["keyframe_id"])],
            "order_gap": int(order_b - order_a),
            "status": "inconclusive",
            "error": str(exc),
            "a_to_b": None,
            "b_to_a": None,
            "diagnostic_npz": None,
        }



def _source_scale_map(d43_summary: dict[str, Any], d4_summary: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    predecessor = d43_summary.get("predecessor_view")
    if isinstance(predecessor, dict) and predecessor.get("scale_to_reconstruction_units") is not None:
        result[str(predecessor["keyframe_id"])] = float(predecessor["scale_to_reconstruction_units"])
    for item in d4_summary.get("views", []):
        if item.get("scale_to_reconstruction_units") is not None:
            result[str(item["keyframe_id"])] = float(item["scale_to_reconstruction_units"])
    return result



def _build_dense_fusion_frames(
    records: Iterable[dict[str, Any]],
    *,
    sfm_result: Any,
) -> tuple[list[DenseFusionFrame], list[dict[str, Any]], str]:
    records = list(records)
    models = {item.camera_id: item for item in sfm_result.camera_models}
    camera_ids = {str(item["camera_id"]) for item in records}
    if len(camera_ids) != 1:
        raise ReconstructionValidationError("E3 DenseVoxelFusion path requires one shared camera model.")
    camera_id = next(iter(camera_ids))
    if camera_id not in models:
        raise ReconstructionValidationError("E3 shared SfM camera model is unavailable.")
    frames: list[DenseFusionFrame] = []
    summaries: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        workspace = record["workspace"]
        rgb_path = Path(sfm_result.workspace.workspace_root).resolve() / workspace.staged_image_path
        bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ReconstructionValidationError(f"E3 could not decode selected RGB keyframe: {rgb_path}")
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)
        aligned_depth = np.asarray(record["aligned_depth"], dtype=np.float32)
        if aligned_depth.shape != rgb.shape[:2]:
            raise ReconstructionValidationError(
                f"E3 aligned depth/RGB shape mismatch at order {record['workspace_order']}."
            )
        frames.append(DenseFusionFrame(
            frame_id=str(record["frame_id"]),
            timestamp_ns=int(workspace.timestamp_ns),
            rgb=rgb,
            depth=aligned_depth,
            T_reconstruction_from_camera=np.asarray(record["T_reconstruction_from_camera"], dtype=np.float64),
            depth_unit=DenseDepthUnit.RECONSTRUCTION_UNITS,
            depth_alignment=reconstruction_unit_depth_alignment(
                provenance=(
                    f"E3 consumes qualified aligned depth for workspace order {record['workspace_order']} "
                    "in the arbitrary offline-SfM reconstruction gauge."
                )
            ),
            pose_layer="offline_sfm_optimized",
        ))
        summaries.append({
            "fusion_index": index,
            "workspace_order": int(record["workspace_order"]),
            "keyframe_id": str(record["keyframe_id"]),
            "frame_id": str(record["frame_id"]),
            "timestamp_ns": int(workspace.timestamp_ns),
            "source": str(record["source"]),
            "cloud_npz": str(record["cloud_npz"]),
        })
    return frames, summaries, camera_id



def main() -> int:
    args = _parse_args()
    e2_dir = args.e2_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    e2_summary = _read_json(e2_dir / "e2_summary.json")
    _validate_e2_summary(e2_summary)
    seed_orders = tuple(int(v) for v in e2_summary["selected_workspace_orders"])
    seed_keyframes = tuple(str(v) for v in e2_summary["selected_keyframe_ids"])

    e1_dir = Path(str(e2_summary["source_e1_dir"])).expanduser().resolve()
    e1_summary = _read_json(e1_dir / "e1_summary.json")
    if not bool(e1_summary.get("pass")) or str(e1_summary.get("status")) != "pass":
        raise ReconstructionValidationError("E3 requires the passing E1 source behind E2.")
    if tuple(int(v) for v in e1_summary.get("selected_workspace_orders", ())) != seed_orders:
        raise ReconstructionValidationError("E3 E1/E2 seed workspace orders disagree.")

    d43_dir = Path(str(e1_summary["source_d4_3_dir"])).expanduser().resolve()
    d43_summary = _read_json(d43_dir / "d4_3_summary.json")
    thresholds = _validate_thresholds(d43_summary.get("thresholds_reused_unchanged_from_d4_1"))
    d4_dir = Path(str(d43_summary["source_d4_dir"])).expanduser().resolve()
    d4_summary = _read_json(d4_dir / "d4_summary.json")
    b2_dir = Path(str(d4_summary["source_b2_dir"])).expanduser().resolve()
    b2_summary = _read_json(b2_dir / "b2_summary.json")
    sfm_result = _sfm_result_from_payload(_read_json(b2_dir / "sfm_result.json"))
    if sfm_result.gauge.scale_status is not SfmScaleStatus.ARBITRARY:
        raise ReconstructionValidationError("E3 expects the validated SfM world to remain arbitrary-scale.")

    registered_keyframes = {camera.keyframe_id for camera in sfm_result.cameras}
    registered = sorted(
        (item for item in sfm_result.workspace.images if item.keyframe_id in registered_keyframes),
        key=lambda item: (int(item.order), item.keyframe_id),
    )
    registered_orders = tuple(int(item.order) for item in registered)
    left_candidates, right_candidates = _candidate_workspace_orders(
        registered_orders,
        seed_orders,
        left_count=int(args.left_candidate_count),
        right_count=int(args.right_candidate_count),
    )
    candidate_orders = tuple(left_candidates + right_candidates)
    if not candidate_orders:
        raise ReconstructionValidationError("E3 found no registered views outside the trusted E2 seed.")

    print("E3 qualified scene-fusion expansion:", flush=True)
    print(f"  trusted_seed={'->'.join(map(str, seed_orders))}", flush=True)
    print(f"  left_candidates={','.join(map(str, left_candidates)) or 'none'}", flush=True)
    print(f"  right_candidates={','.join(map(str, right_candidates)) or 'none'}", flush=True)
    print(f"  max_bridge_gap={int(args.max_bridge_gap)} (no threshold relaxation)", flush=True)

    scale_map = _source_scale_map(d43_summary, d4_summary)
    seed_records: dict[int, dict[str, Any]] = {}
    e1_frames_by_order = {int(item["workspace_order"]): item for item in e1_summary["frames"]}
    for order, keyframe_id in zip(seed_orders, seed_keyframes):
        source = e1_frames_by_order.get(order)
        if source is None or str(source["keyframe_id"]) != keyframe_id:
            raise ReconstructionValidationError(f"E3 cannot locate E1 seed artifact for order {order}.")
        seed_records[order] = _load_cloud_record(
            order=order,
            keyframe_id=keyframe_id,
            cloud_npz=Path(str(source["cloud_npz"])),
            sfm_result=sfm_result,
            scale_to_reconstruction_units=scale_map.get(keyframe_id),
            source="e2_trusted_seed",
        )

    da3_python = args.da3_python.expanduser().resolve()
    da3_repo = args.da3_repo.expanduser().resolve()
    da3_probe = probe_da3_environment(da3_python, da3_repo)
    if not da3_probe.available:
        raise ReconstructionValidationError(
            "E3 DA3 environment is not qualified: " + json.dumps(da3_probe.as_dict(), sort_keys=True)
        )
    b2_manifest = b2_summary.get("dense_depth", {}).get("manifest", {})
    model_id = str(args.da3_model).strip() or str(b2_manifest.get("model_id", "")).strip() or DA3_EFFICIENT_MODEL_ID
    model_revision = str(args.da3_model_revision).strip() or str(b2_manifest.get("model_revision", "")).strip()
    runner = DA3ExternalWorkerRunner(
        python_executable=da3_python,
        worker_script=_DA3_WORKER.resolve(),
        profile=DA3ProviderProfile(
            model_id=model_id,
            model_revision=model_revision,
            process_res=int(args.process_res),
            process_res_method="upper_bound_resize",
        ),
        timeout_s=float(args.da3_timeout_seconds),
    )

    candidate_records: dict[int, dict[str, Any]] = {}
    candidate_summaries: list[dict[str, Any]] = []
    for order in candidate_orders:
        record, summary = _generate_candidate_view(
            order=order,
            sfm_result=sfm_result,
            b2_summary=b2_summary,
            output_dir=output_dir,
            runner=runner,
            da3_probe=da3_probe,
            minimum_correspondences=int(args.minimum_correspondences),
            maximum_relative_residual=float(args.maximum_relative_residual),
        )
        candidate_summaries.append(summary)
        if record is not None:
            candidate_records[order] = record
            print(
                f"  candidate order={order} ALIGNMENT PASS scale={summary['scale_to_reconstruction_units']:.6f} "
                f"inliers={summary['inlier_correspondences']}/{summary['valid_correspondences']}",
                flush=True,
            )
        else:
            print(f"  candidate order={order} ALIGNMENT REJECTED reason={summary['rejection_code']}", flush=True)

    sparse_points, sparse_colors = _all_sparse_landmarks(sfm_result)
    scene_scale = float(e1_summary["scene_scale"]["robust_diagonal_reconstruction_units"])
    if not isfinite(scene_scale) or scene_scale <= 0.0:
        raise ReconstructionValidationError("E3 E1 scene scale is invalid.")

    pair_summaries: dict[tuple[int, int], dict[str, Any]] = {}
    for side, anchor, side_candidates in (
        ("left", seed_orders[0], left_candidates),
        ("right", seed_orders[-1], right_candidates),
    ):
        available_orders = [order for order in side_candidates if order in candidate_records]
        nodes = [anchor] + available_orders
        nodes = sorted(set(nodes))
        for i, order_a in enumerate(nodes):
            for order_b in nodes[i + 1:]:
                if abs(order_b - order_a) > int(args.max_bridge_gap):
                    continue
                # The anchor is the only seed node evaluated per side.  Candidate-candidate
                # links remain entirely outside the trusted seed.
                if order_a == anchor:
                    record_a = seed_records[anchor]
                else:
                    record_a = candidate_records[order_a]
                if order_b == anchor:
                    record_b = seed_records[anchor]
                else:
                    record_b = candidate_records[order_b]
                key = _pair_key(order_a, order_b)
                if key in pair_summaries:
                    continue
                pair = _diagnose_pair(
                    a=record_a,
                    b=record_b,
                    thresholds=thresholds,
                    scene_scale=scene_scale,
                    sparse_landmarks=sparse_points,
                    output_dir=output_dir,
                    max_points_per_direction=int(args.max_points_per_direction),
                    nn_max_points_per_view=int(args.nn_max_points_per_view),
                )
                pair["side"] = side
                pair_summaries[key] = pair
                print(
                    f"  edge {order_a}<->{order_b} side={side} status={pair['status']}",
                    flush=True,
                )

    left_path = _best_growth_path(
        anchor_order=seed_orders[0],
        candidate_orders=[order for order in left_candidates if order in candidate_records],
        pair_summaries=pair_summaries,
        direction="left",
    )
    right_path = _best_growth_path(
        anchor_order=seed_orders[-1],
        candidate_orders=[order for order in right_candidates if order in candidate_records],
        pair_summaries=pair_summaries,
        direction="right",
    )
    selected_left = tuple(reversed(left_path.added_orders))
    selected_right = tuple(right_path.added_orders)
    selected_orders = selected_left + seed_orders + selected_right
    if len(set(selected_orders)) != len(selected_orders) or tuple(sorted(selected_orders)) != selected_orders:
        raise ReconstructionValidationError("E3 selected expanded workspace orders are not strictly increasing.")

    selected_records = (
        [candidate_records[order] for order in selected_left]
        + [seed_records[order] for order in seed_orders]
        + [candidate_records[order] for order in selected_right]
    )
    new_view_count = len(selected_left) + len(selected_right)
    print("\nE3 selected monotonic expansion:", flush=True)
    print(f"  left_path_anchor_outward={'->'.join(map(str, left_path.orders_anchor_outward))}", flush=True)
    print(f"  right_path_anchor_outward={'->'.join(map(str, right_path.orders_anchor_outward))}", flush=True)
    print(f"  expanded_chain={'->'.join(map(str, selected_orders))}", flush=True)
    print(f"  new_views={new_view_count} total_views={len(selected_orders)}", flush=True)

    frames, frame_records, camera_id = _build_dense_fusion_frames(selected_records, sfm_result=sfm_result)
    model = {item.camera_id: item for item in sfm_result.camera_models}[camera_id]
    reconstruction_camera = reconstruction_camera_from_sfm_model(
        model,
        keyframe_id=str(selected_records[0]["keyframe_id"]),
        reconstruction_frame_id="offline-sfm-reconstruction",
    )
    camera_profile = camera_geometry_from_estimated_camera(
        profile_id=f"e3-{camera_id}-sfm-estimated",
        camera=reconstruction_camera,
        focal_length_relative_uncertainty=float(args.sfm_focal_relative_uncertainty),
        notes=(
            "E3 validation-only camera profile from the globally optimized SfM camera model; "
            "no physical metric authority is introduced."
        ),
    )
    scale_profile = arbitrary_scale_reference(
        profile_id=f"e3-{sfm_result.session_id[:12]}-arbitrary-sfm-gauge",
        notes="E3 preserves the arbitrary offline-SfM reconstruction gauge.",
    )
    voxel_size = float(e1_summary["policy"]["voxel_size_reconstruction_units"])
    policy = DenseFusionPolicy(
        scene_profile=DenseSceneProfile.SPATIAL_DIGITAL_TWIN,
        voxel_size_reconstruction_units=voxel_size,
        pixel_stride=1,
        minimum_view_observations=1,
        max_voxels=int(args.max_voxels),
    )
    result = fuse_dense_frames(
        frames,
        session_id=f"{sfm_result.session_id}:e3-expanded-qualified-fusion",
        camera_profile=camera_profile,
        scale_profile=scale_profile,
        policy=policy,
        reconstruction_frame_id="offline-sfm-reconstruction",
    )
    cloud = result.point_cloud
    counts = np.asarray(cloud.view_observation_counts, dtype=np.uint32)
    supported_mask, high_mask = _expanded_support_masks(
        counts,
        minimum_supported_views=int(args.minimum_supported_views),
        high_support_views=int(args.high_support_views),
        maximum_views=len(selected_orders),
    )
    selected_keyframes = tuple(str(item["keyframe_id"]) for item in selected_records)
    all_npz = _save_cloud_npz(
        output_dir / "e3_expanded_voxel_fused_all.npz",
        points=cloud.points_reconstruction,
        colors=cloud.colors_rgb,
        counts=counts,
        mean_confidence=cloud.mean_confidence,
        voxel_size=voxel_size,
        selected_keyframe_ids=selected_keyframes,
        selected_workspace_orders=selected_orders,
        sparse_points=sparse_points,
        sparse_colors=sparse_colors,
    )
    supported_npz = _save_cloud_npz(
        output_dir / "e3_expanded_voxel_fused_supported.npz",
        points=cloud.points_reconstruction[supported_mask],
        colors=cloud.colors_rgb[supported_mask],
        counts=counts[supported_mask],
        mean_confidence=cloud.mean_confidence[supported_mask],
        voxel_size=voxel_size,
        selected_keyframe_ids=selected_keyframes,
        selected_workspace_orders=selected_orders,
        sparse_points=sparse_points,
        sparse_colors=sparse_colors,
    )
    high_npz = _save_cloud_npz(
        output_dir / "e3_expanded_voxel_fused_high_support.npz",
        points=cloud.points_reconstruction[high_mask],
        colors=cloud.colors_rgb[high_mask],
        counts=counts[high_mask],
        mean_confidence=cloud.mean_confidence[high_mask],
        voxel_size=voxel_size,
        selected_keyframe_ids=selected_keyframes,
        selected_workspace_orders=selected_orders,
        sparse_points=sparse_points,
        sparse_colors=sparse_colors,
    )
    all_ply = _write_binary_ply(output_dir / "e3_expanded_voxel_fused_all.ply", cloud.points_reconstruction, cloud.colors_rgb)
    supported_ply = _write_binary_ply(
        output_dir / "e3_expanded_voxel_fused_supported.ply",
        cloud.points_reconstruction[supported_mask],
        cloud.colors_rgb[supported_mask],
    )

    supported_count = int(np.count_nonzero(supported_mask))
    high_count = int(np.count_nonzero(high_mask))
    baseline_supported = int(e1_summary["fusion"]["multi_view_supported_voxel_count"])
    baseline_supported_path = Path(str(e1_summary["artifacts"]["supported_npz"])).expanduser().resolve()
    with np.load(baseline_supported_path, allow_pickle=False) as payload:
        baseline_supported_points = np.asarray(payload["points_reconstruction"], dtype=np.float64)
    baseline_span = _robust_diagonal(baseline_supported_points)
    expanded_span = _robust_diagonal(cloud.points_reconstruction[supported_mask]) if supported_count else 0.0
    capacity_clean = int(result.statistics.dropped_new_voxels_capacity) == 0

    pass_conditions = {
        "minimum_new_views": new_view_count >= int(args.minimum_new_views),
        "minimum_left_views": len(selected_left) >= int(args.minimum_new_views_per_side),
        "minimum_right_views": len(selected_right) >= int(args.minimum_new_views_per_side),
        "capacity_clean": capacity_clean,
        "frames_integrated": int(result.statistics.frames_integrated) == len(selected_orders),
        "minimum_supported_voxels": supported_count >= int(args.minimum_supported_voxels),
        "supported_voxels_not_regressed": supported_count >= baseline_supported,
        "robust_supported_span_not_regressed": expanded_span >= (0.98 * baseline_span),
    }
    passed = all(pass_conditions.values())

    summary = {
        "phase": "E3",
        "pass": passed,
        "status": "pass" if passed else "degraded",
        "source_e2_dir": str(e2_dir),
        "source_e2_pass": True,
        "source_e1_dir": str(e1_dir),
        "source_sfm_session_id": str(sfm_result.session_id),
        "trusted_seed": {
            "workspace_orders": list(seed_orders),
            "keyframe_ids": list(seed_keyframes),
            "view_count": 10,
        },
        "candidate_policy": {
            "left_candidate_count": int(args.left_candidate_count),
            "right_candidate_count": int(args.right_candidate_count),
            "left_candidate_orders": list(left_candidates),
            "right_candidate_orders": list(right_candidates),
            "max_bridge_gap": int(args.max_bridge_gap),
            "interior_seed_gaps_are_not_reopened": True,
            "dense_thresholds_reused_unchanged_from_d4_3": thresholds,
            "selection_rule": (
                "Grow monotonic pass-only paths independently from each seed boundary; maximize added view count, "
                "then order span, then worst-case dense consistency. Degraded/inconclusive edges are never relaxed."
            ),
        },
        "candidate_views": candidate_summaries,
        "edge_diagnostics": [pair_summaries[key] for key in sorted(pair_summaries)],
        "growth": {
            "left": {
                "anchor_order": left_path.anchor_order,
                "orders_anchor_outward": list(left_path.orders_anchor_outward),
                "selected_orders_ascending": list(selected_left),
                "added_view_count": len(selected_left),
                "worst_consistency": left_path.worst_consistency,
                "worst_median_residual": left_path.worst_median_residual,
            },
            "right": {
                "anchor_order": right_path.anchor_order,
                "orders_anchor_outward": list(right_path.orders_anchor_outward),
                "selected_orders_ascending": list(selected_right),
                "added_view_count": len(selected_right),
                "worst_consistency": right_path.worst_consistency,
                "worst_median_residual": right_path.worst_median_residual,
            },
        },
        "selected_workspace_orders": list(selected_orders),
        "selected_keyframe_ids": list(selected_keyframes),
        "selected_view_count": len(selected_orders),
        "new_qualified_view_count": new_view_count,
        "frames": frame_records,
        "fusion": {
            "voxel_size_reconstruction_units": voxel_size,
            "voxel_size_reused_exactly_from_e1": True,
            "all_voxel_count": int(len(counts)),
            "supported_voxel_count": supported_count,
            "high_support_voxel_count": high_count,
            "support_histogram": _support_histogram(counts),
            "capacity_clean": capacity_clean,
            "capacity_drops": int(result.statistics.dropped_new_voxels_capacity),
            "frames_integrated": int(result.statistics.frames_integrated),
            "sampled_pixels": int(result.statistics.sampled_pixels),
            "baseline_e1_supported_voxel_count": baseline_supported,
            "supported_voxel_growth_ratio": float(supported_count / max(baseline_supported, 1)),
            "baseline_e1_supported_robust_diagonal": baseline_span,
            "expanded_supported_robust_diagonal": expanded_span,
            "supported_robust_diagonal_growth_ratio": float(expanded_span / baseline_span),
        },
        "pass_conditions": pass_conditions,
        "decision": {
            "authorizes_next_confidence_filtering_milestone": passed,
            "next_milestone": "F - Confidence / Consistency / Artifact Filtering" if passed else None,
            "primary_output": "e3_expanded_voxel_fused_supported.npz",
        },
        "authority": {
            "sfm_scale_status": "arbitrary",
            "sfm_world_alignment_status": "arbitrary",
            "metric_scale_claimed": False,
            "canonical_geometry_authoritative": False,
            "canonical_persistence_promoted": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
            "confidence_artifact_filtering_formalized": False,
            "note": (
                "E3 validation-only qualified scene expansion. Formal confidence/artifact filtering belongs to F; "
                "persistence belongs to G; UI integration belongs to H."
            ),
        },
        "da3_environment_probe": da3_probe.as_dict(),
        "artifacts": {
            "all_npz": str(all_npz.resolve()),
            "all_ply": str(all_ply.resolve()),
            "supported_npz": str(supported_npz.resolve()),
            "supported_ply": str(supported_ply.resolve()),
            "high_support_npz": str(high_npz.resolve()),
        },
    }
    summary_path = _write_json(output_dir / "e3_summary.json", summary)

    print(f"\nE3 qualified scene-fusion expansion: {summary['status'].upper()}")
    print(f"seed_views=10 new_qualified_views={new_view_count} total_views={len(selected_orders)}")
    print(f"selected_orders={'->'.join(map(str, selected_orders))}")
    print(
        f"left_added={len(selected_left)} right_added={len(selected_right)} "
        f"capacity_drops={summary['fusion']['capacity_drops']}"
    )
    print(
        f"supported_voxels={supported_count} baseline_e1={baseline_supported} "
        f"growth={summary['fusion']['supported_voxel_growth_ratio']:.3f}x"
    )
    print(
        f"supported_robust_span={expanded_span:.6f} baseline={baseline_span:.6f} "
        f"growth={summary['fusion']['supported_robust_diagonal_growth_ratio']:.3f}x"
    )
    print("formal_confidence_artifact_filtering=False (belongs to F)")
    print("metric_scale_claimed=False canonical_geometry_authoritative=False measurement_authoritative=False")
    print(f"supported_cloud={supported_npz}")
    print(f"summary={summary_path}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

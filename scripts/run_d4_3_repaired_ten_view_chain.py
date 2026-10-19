#!/usr/bin/env python3
"""D4.3 repaired ten-view chain qualification.

D4.3 consumes the D4/D4.1/D4.2 evidence after a single pair-specific weak
adjacent transition was isolated. It adds exactly one predecessor registered
keyframe, evaluates the two minimal repaired ten-view candidates obtained by
removing either member of the weak pair, and qualifies both candidates using
all nine adjacent bidirectional D2 reprojection/depth-consistency checks.

Only the predecessor view requires a new isolated DA3 inference. Existing D4
views are reused byte-for-byte. No voxel fusion, geometry filtering,
persistence, metric-scale qualification, measurement authority, or certified
metrology is performed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction import (  # noqa: E402
    SfmScaleStatus,
    align_sfm_keyframe_dense_depth,
    project_sfm_keyframe_dense_rgb,
    reconstruction_camera_from_sfm_model,
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
    _read_json,
    _sfm_result_from_payload,
    _write_json,
)
from scripts.run_d2_three_view_overlap_diagnostics import (  # noqa: E402
    _assess_direction,
    _directed_reprojection_diagnostic,
    _load_aligned_depth,
    _load_view_cloud,
    _pair_status,
    _robust_scene_scale,
    _spatial_nn_diagnostic,
)
from scripts.run_d4_1_ten_view_chain_diagnostics import _overall_chain_status  # noqa: E402
from scripts.run_d4_ten_view_canonical_accumulation import _write_d4_binary_ply  # noqa: E402
from scripts.run_d4_2_weak_link_bridge_diagnostics import _thresholds_from_d4_1  # noqa: E402

_DEFAULT_DA3_PYTHON = Path.home() / "anaconda3/envs/depth3d-da3/bin/python"
_DEFAULT_DA3_REPO = Path.home() / "Desktop/depth3d-da3-backends/depth-anything-3"
_DA3_WORKER = ROOT / "scripts/da3_workers/da3_worker.py"
_VIEW_COUNT = 10


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d4-dir", type=Path, default=Path("/tmp/depth3d_d4_ten_view_accumulation"))
    parser.add_argument("--d4-1-dir", type=Path, default=Path("/tmp/depth3d_d4_1_ten_view_qualification"))
    parser.add_argument("--d4-2-dir", type=Path, default=Path("/tmp/depth3d_d4_2_weak_link_bridge_diagnosis"))
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/depth3d_d4_3_repaired_ten_view_chain"))
    parser.add_argument("--da3-python", type=Path, default=_DEFAULT_DA3_PYTHON)
    parser.add_argument("--da3-repo", type=Path, default=_DEFAULT_DA3_REPO)
    parser.add_argument("--da3-model", default="")
    parser.add_argument("--da3-model-revision", default="")
    parser.add_argument("--process-res", type=int, default=392)
    parser.add_argument("--da3-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--minimum-correspondences", type=int, default=8)
    parser.add_argument("--maximum-relative-residual", type=float, default=0.35)
    parser.add_argument("--pixel-stride", type=int, default=1)
    parser.add_argument("--max-points-per-direction", type=int, default=120_000)
    parser.add_argument("--nn-max-points-per-view", type=int, default=60_000)
    args = parser.parse_args()
    if args.process_res <= 0 or args.da3_timeout_seconds <= 0 or args.pixel_stride <= 0:
        parser.error("D4.3 DA3/process arguments must be positive.")
    if args.minimum_correspondences < 3:
        parser.error("--minimum-correspondences must be at least 3.")
    if not 0.0 < args.maximum_relative_residual < 1.0:
        parser.error("--maximum-relative-residual must be in (0,1).")
    if args.max_points_per_direction <= 0 or args.nn_max_points_per_view <= 0:
        parser.error("D4.3 diagnostic point budgets must be positive.")
    return args


def _validate_source_evidence(
    d4_summary: dict[str, Any],
    d41_summary: dict[str, Any],
    d42_summary: dict[str, Any],
) -> tuple[int, int]:
    if not bool(d4_summary.get("pass")) or str(d4_summary.get("phase")) != "D4":
        raise ReconstructionValidationError("D4.3 requires a successful D4 accumulation.")
    views = d4_summary.get("views")
    if not isinstance(views, list) or len(views) != _VIEW_COUNT:
        raise ReconstructionValidationError("D4.3 requires exactly ten D4 source views.")
    if bool(d41_summary.get("pass")) or str(d41_summary.get("status")) != "degraded":
        raise ReconstructionValidationError("D4.3 requires the degraded D4.1 chain evidence.")
    if str(d42_summary.get("phase")) != "D4.2" or not bool(d42_summary.get("diagnostic_complete")):
        raise ReconstructionValidationError("D4.3 requires completed D4.2 bridge diagnosis.")
    diagnosis = d42_summary.get("diagnosis", {})
    if str(diagnosis.get("classification")) != "pair_specific_incompatibility":
        raise ReconstructionValidationError(
            "D4.3 minimal two-candidate repair is valid only for pair-specific incompatibility."
        )
    if diagnosis.get("preferred_single_view_exclusion_workspace_order") is not None:
        raise ReconstructionValidationError("D4.3 expected no individually disqualified weak-pair member.")
    weak = d42_summary.get("weak_adjacent_pair", {})
    orders = weak.get("workspace_orders")
    if not isinstance(orders, list) or len(orders) != 2:
        raise ReconstructionValidationError("D4.3 requires D4.2 weak-pair workspace orders.")
    left, right = int(orders[0]), int(orders[1])
    if right != left + 1:
        raise ReconstructionValidationError("D4.3 weak pair must remain adjacent in workspace order.")
    source_orders = [int(item["workspace_order"]) for item in views]
    if left not in source_orders or right not in source_orders:
        raise ReconstructionValidationError("D4.3 weak-pair views are missing from D4.")
    bridge_statuses = {str(item.get("role")): str(item.get("status")) for item in d42_summary.get("bridge_pairs", [])}
    if bridge_statuses != {"left": "pass", "right": "pass"}:
        raise ReconstructionValidationError("D4.3 requires both D4.2 skip-one bridges to pass.")
    return left, right


def _candidate_workspace_orders(
    source_orders: Iterable[int],
    *,
    predecessor_order: int,
    weak_left_order: int,
    weak_right_order: int,
) -> dict[str, tuple[int, ...]]:
    source = tuple(int(value) for value in source_orders)
    if len(source) != _VIEW_COUNT or len(set(source)) != _VIEW_COUNT:
        raise ReconstructionValidationError("D4.3 source chain must contain ten distinct views.")
    if predecessor_order in source:
        raise ReconstructionValidationError("D4.3 predecessor must be outside the source ten-view chain.")
    if weak_left_order not in source or weak_right_order not in source:
        raise ReconstructionValidationError("D4.3 weak-pair members must be in the source chain.")
    candidate_a = tuple(sorted((predecessor_order,) + tuple(x for x in source if x != weak_left_order)))
    candidate_b = tuple(sorted((predecessor_order,) + tuple(x for x in source if x != weak_right_order)))
    if len(candidate_a) != _VIEW_COUNT or len(candidate_b) != _VIEW_COUNT:
        raise ReconstructionValidationError("D4.3 repaired candidates must each contain ten views.")
    return {"A": candidate_a, "B": candidate_b}


def _select_repaired_candidate(candidate_summaries: dict[str, dict[str, Any]]) -> tuple[str | None, str]:
    passing = [name for name, payload in candidate_summaries.items() if bool(payload.get("pass"))]
    if not passing:
        return None, "no_candidate_qualified"
    if len(passing) == 1:
        return passing[0], "only_passing_candidate"

    def score(name: str) -> tuple[float, float, float, str]:
        payload = candidate_summaries[name]
        pairs = payload["adjacent_pairs"]
        min_consistency = min(
            min(float(item["a_to_b"]["consistent_ratio_of_comparable"]), float(item["b_to_a"]["consistent_ratio_of_comparable"]))
            for item in pairs
        )
        worst_median = max(
            max(float(item["a_to_b"]["median_absolute_relative_depth_residual"]), float(item["b_to_a"]["median_absolute_relative_depth_residual"]))
            for item in pairs
        )
        mean_consistency = float(np.mean([
            0.5 * (float(item["a_to_b"]["consistent_ratio_of_comparable"]) + float(item["b_to_a"]["consistent_ratio_of_comparable"]))
            for item in pairs
        ]))
        return (min_consistency, -worst_median, mean_consistency, name)

    selected = max(passing, key=score)
    return selected, "best_dense_consistency_among_passing_candidates"


def _load_record(item: dict[str, Any], sfm_result: Any) -> dict[str, Any]:
    keyframe_id = str(item["keyframe_id"])
    cameras = {camera.keyframe_id: camera for camera in sfm_result.cameras}
    camera_models = {camera.camera_id: camera for camera in sfm_result.camera_models}
    estimate = cameras.get(keyframe_id)
    if estimate is None:
        raise ReconstructionValidationError(f"D4.3 missing SfM camera estimate for {keyframe_id}.")
    model = camera_models.get(estimate.camera_id)
    if model is None:
        raise ReconstructionValidationError(f"D4.3 missing SfM camera model {estimate.camera_id}.")
    camera = reconstruction_camera_from_sfm_model(
        model,
        keyframe_id=keyframe_id,
        reconstruction_frame_id="offline-sfm-reconstruction",
    )
    points, colors, transform = _load_view_cloud(Path(str(item["cloud_npz"])).expanduser().resolve())
    if not np.allclose(transform, estimate.T_reconstruction_from_camera, rtol=0.0, atol=1e-9):
        raise ReconstructionValidationError("D4.3 detected pose drift between cloud and SfM result.")
    aligned_depth, pixel_adapter = _load_aligned_depth(
        item,
        sfm_width=model.image_size.width,
        sfm_height=model.image_size.height,
    )
    return {
        "summary": item,
        "keyframe_id": keyframe_id,
        "workspace_order": int(item["workspace_order"]),
        "scale_to_reconstruction_units": float(item["scale_to_reconstruction_units"]),
        "camera_center_reconstruction": np.asarray(item["camera_center_reconstruction"], dtype=np.float64),
        "camera_estimate": estimate,
        "camera": camera,
        "points": points,
        "colors": colors,
        "aligned_depth": aligned_depth,
        "pixel_adapter": pixel_adapter,
    }


def _generate_predecessor_view(
    *,
    sfm_result: Any,
    b2_summary: dict[str, Any],
    predecessor_order: int,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    cameras = {camera.keyframe_id: camera for camera in sfm_result.cameras}
    workspace_images = [
        image for image in sfm_result.workspace.images
        if image.keyframe_id in cameras and int(image.order) == int(predecessor_order)
    ]
    if len(workspace_images) != 1:
        raise ReconstructionValidationError(
            f"D4.3 requires exactly one registered predecessor at workspace order {predecessor_order}."
        )
    workspace_image = workspace_images[0]
    estimate = cameras[workspace_image.keyframe_id]
    models = {item.camera_id: item for item in sfm_result.camera_models}
    model = models.get(estimate.camera_id)
    if model is None:
        raise ReconstructionValidationError("D4.3 predecessor camera model is unavailable.")
    rgb_path = Path(sfm_result.workspace.workspace_root).resolve() / workspace_image.staged_image_path
    bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ReconstructionValidationError(f"D4.3 could not decode predecessor RGB keyframe: {rgb_path}")
    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)

    da3_python = args.da3_python.expanduser().resolve()
    da3_repo = args.da3_repo.expanduser().resolve()
    probe = probe_da3_environment(da3_python, da3_repo)
    if not probe.available:
        raise ReconstructionValidationError("D4.3 DA3 environment is not qualified: " + json.dumps(probe.as_dict(), sort_keys=True))
    b2_manifest = b2_summary.get("dense_depth", {}).get("manifest", {})
    model_id = str(args.da3_model).strip() or str(b2_manifest.get("model_id", "")).strip() or DA3_EFFICIENT_MODEL_ID
    model_revision = str(args.da3_model_revision).strip() or str(b2_manifest.get("model_revision", "")).strip()

    da3_root = output_dir / "da3_predecessor"
    input_manifest = write_da3_input_manifest(
        da3_root / "input_manifest.json",
        DA3InputManifest(
            request_id=f"d4-3-{sfm_result.session_id}-order-{predecessor_order}",
            image_paths=(str(rgb_path.resolve()),),
            metadata={
                "phase": "D4.3",
                "session_id": sfm_result.session_id,
                "workspace_order": predecessor_order,
                "sfm_keyframe_id": estimate.keyframe_id,
                "purpose": "single_predecessor_for_repaired_ten_view_candidates",
                "canonical_geometry_authoritative": False,
                "certified_metrology": False,
            },
        ),
    )
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
    print(f"D4.3: running isolated DA3 only for predecessor order={predecessor_order}...", flush=True)
    manifest_path, manifest = runner.run(input_manifest, da3_root / "worker_output")
    arrays = validate_da3_npz_result(manifest)
    if manifest.frame_count != 1 or arrays["depth"].shape[0] != 1:
        raise ReconstructionValidationError("D4.3 predecessor DA3 result must contain exactly one frame.")
    depth_sfm, adapter = _adapt_da3_depth_to_sfm_pixel_space(
        arrays["depth"][0], sfm_width=model.image_size.width, sfm_height=model.image_size.height
    )
    evidence = align_sfm_keyframe_dense_depth(
        sfm_result,
        keyframe_id=estimate.keyframe_id,
        dense_depth_source_units=depth_sfm,
        depth_provenance=(
            f"Depth Anything 3 model={manifest.model_id} revision={manifest.model_revision};"
            f"source_revision={manifest.source_revision};pixel_adapter={adapter['mode']};"
            "D4.3 predecessor validation only; no fusion; no metric scale claim"
        ),
        minimum_correspondences=int(args.minimum_correspondences),
        maximum_relative_residual=float(args.maximum_relative_residual),
    )
    evidence_path = _write_json(output_dir / "d4_3_predecessor_alignment.json", evidence.as_dict())
    if not evidence.is_accepted or evidence.alignment is None:
        raise ReconstructionValidationError(
            "D4.3 predecessor SfM-depth alignment was rejected: "
            + (evidence.rejection_code.value if evidence.rejection_code else "unknown")
        )
    projection = project_sfm_keyframe_dense_rgb(
        estimate,
        model,
        timestamp_ns=workspace_image.timestamp_ns,
        rgb=rgb,
        dense_depth_source_units=depth_sfm,
        alignment_evidence=evidence,
        pixel_stride=int(args.pixel_stride),
        reconstruction_frame_id="offline-sfm-reconstruction",
    )
    cloud_path = output_dir / "d4_3_predecessor_canonical_cloud.npz"
    np.savez_compressed(
        cloud_path,
        points_reconstruction=projection.points_reconstruction,
        colors_rgb=projection.colors_rgb,
        aligned_depth_reconstruction_units=np.asarray(
            depth_sfm * evidence.alignment.scale_to_reconstruction_units, dtype=np.float64
        ),
        T_reconstruction_from_camera=estimate.T_reconstruction_from_camera,
        camera_center_reconstruction=estimate.camera_center_reconstruction,
        keyframe_id=np.asarray(estimate.keyframe_id),
    )
    observation_counts = _observation_counts(sfm_result)
    summary = {
        "view_index": -1,
        "workspace_order": int(workspace_image.order),
        "source_sequence_index": int(workspace_image.source_sequence_index),
        "keyframe_id": estimate.keyframe_id,
        "frame_id": estimate.frame_id,
        "camera_id": estimate.camera_id,
        "camera_model_name": model.backend_model_name,
        "sparse_observations": int(observation_counts.get(estimate.keyframe_id, 0)),
        "candidate_correspondences": evidence.candidate_observations,
        "valid_correspondences": evidence.valid_correspondences,
        "inlier_correspondences": evidence.inlier_correspondences,
        "scale_to_reconstruction_units": evidence.alignment.scale_to_reconstruction_units,
        "relative_scale_uncertainty": evidence.alignment.relative_scale_uncertainty,
        "median_relative_residual": evidence.median_relative_residual,
        "p90_relative_residual": evidence.p90_relative_residual,
        "pixel_adapter": adapter,
        "point_count": int(projection.points_reconstruction.shape[0]),
        "bounds_min_reconstruction": projection.points_reconstruction.min(axis=0).tolist(),
        "bounds_max_reconstruction": projection.points_reconstruction.max(axis=0).tolist(),
        "camera_center_reconstruction": estimate.camera_center_reconstruction.tolist(),
        "alignment_evidence_json": str(evidence_path.resolve()),
        "cloud_npz": str(cloud_path.resolve()),
        "da3_result_manifest": str(manifest_path.resolve()),
    }
    print(
        f"D4.3: predecessor ACCEPTED order={predecessor_order} "
        f"scale={evidence.alignment.scale_to_reconstruction_units:.8f} "
        f"inliers={evidence.inlier_correspondences}/{evidence.valid_correspondences} "
        f"points={projection.points_reconstruction.shape[0]}",
        flush=True,
    )
    return {"summary": summary, "da3_environment_probe": probe.as_dict()}


def _diagnose_candidate(
    *,
    name: str,
    records: list[dict[str, Any]],
    thresholds: dict[str, Any],
    sparse_landmarks: np.ndarray,
    output_dir: Path,
    max_points_per_direction: int,
    nn_max_points_per_view: int,
) -> dict[str, Any]:
    if len(records) != _VIEW_COUNT:
        raise ReconstructionValidationError("D4.3 candidates must contain exactly ten loaded view records.")
    scene_scale, scene_source, low, high = _robust_scene_scale(
        sparse_landmarks, [record["points"] for record in records]
    )
    pair_summaries: list[dict[str, Any]] = []
    for a_index in range(_VIEW_COUNT - 1):
        b_index = a_index + 1
        a, b = records[a_index], records[b_index]
        a_to_b, diag_ab = _directed_reprojection_diagnostic(
            a["points"],
            target_camera=b["camera"],
            target_T_reconstruction_from_camera=b["camera_estimate"].T_reconstruction_from_camera,
            target_aligned_depth=b["aligned_depth"],
            maximum_points=int(max_points_per_direction),
            consistency_relative_tolerance=float(thresholds["consistency_relative_tolerance"]),
            occlusion_relative_margin=float(thresholds["occlusion_relative_margin"]),
        )
        b_to_a, diag_ba = _directed_reprojection_diagnostic(
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
        pair_status = _pair_status(status_ab, status_ba)
        baseline = float(np.linalg.norm(a["camera_center_reconstruction"] - b["camera_center_reconstruction"]))
        nn = _spatial_nn_diagnostic(
            a["points"], b["points"],
            scene_scale=scene_scale,
            maximum_points=int(nn_max_points_per_view),
            support_scene_fraction=float(thresholds["nn_support_scene_fraction"]),
        )
        diagnostic_path = output_dir / f"d4_3_candidate_{name.lower()}_pair_{a_index}{b_index}_diagnostic.npz"
        np.savez_compressed(
            diagnostic_path,
            candidate=np.asarray(name),
            pair=np.asarray([a_index, b_index], dtype=np.uint8),
            keyframe_ids=np.asarray([a["keyframe_id"], b["keyframe_id"]], dtype="U160"),
            workspace_orders=np.asarray([a["workspace_order"], b["workspace_order"]], dtype=np.int32),
            a_to_b_points_reconstruction=diag_ab["points_reconstruction"],
            a_to_b_classification=diag_ab["classification"],
            a_to_b_signed_relative_depth_residual=diag_ab["signed_relative_depth_residual"],
            a_to_b_absolute_relative_depth_residual=diag_ab["absolute_relative_depth_residual"],
            b_to_a_points_reconstruction=diag_ba["points_reconstruction"],
            b_to_a_classification=diag_ba["classification"],
            b_to_a_signed_relative_depth_residual=diag_ba["signed_relative_depth_residual"],
            b_to_a_absolute_relative_depth_residual=diag_ba["absolute_relative_depth_residual"],
            sparse_landmarks_reconstruction=sparse_landmarks,
        )
        pair_summaries.append({
            "pair": [a_index, b_index],
            "pair_label": f"{a_index}-{b_index}",
            "workspace_orders": [a["workspace_order"], b["workspace_order"]],
            "keyframe_ids": [a["keyframe_id"], b["keyframe_id"]],
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
        })
    status, passed, _exit_code = _overall_chain_status([str(item["status"]) for item in pair_summaries])
    return {
        "candidate": name,
        "pass": passed,
        "status": status,
        "workspace_orders": [int(record["workspace_order"]) for record in records],
        "keyframe_ids": [str(record["keyframe_id"]) for record in records],
        "scene_scale": {
            "robust_diagonal_reconstruction_units": scene_scale,
            "source": scene_source,
            "quantile_bounds_min": low,
            "quantile_bounds_max": high,
            "physical_unit_claimed": False,
        },
        "adjacent_pairs": pair_summaries,
    }


def _write_selected_accumulation(
    output_dir: Path,
    records: list[dict[str, Any]],
    sparse_points: np.ndarray,
    sparse_colors: np.ndarray,
) -> dict[str, str]:
    points_parts: list[np.ndarray] = []
    colors_parts: list[np.ndarray] = []
    source_parts: list[np.ndarray] = []
    centers: list[np.ndarray] = []
    for index, record in enumerate(records):
        points_parts.append(record["points"])
        colors_parts.append(record["colors"])
        source_parts.append(np.full(record["points"].shape[0], index, dtype=np.uint8))
        centers.append(record["camera_center_reconstruction"])
    points = np.ascontiguousarray(np.concatenate(points_parts, axis=0))
    colors = np.ascontiguousarray(np.concatenate(colors_parts, axis=0))
    source_index = np.ascontiguousarray(np.concatenate(source_parts, axis=0))
    npz_path = output_dir / "d4_3_selected_repaired_ten_view_raw_accumulation.npz"
    np.savez_compressed(
        npz_path,
        points_reconstruction=points,
        colors_rgb=colors,
        source_view_index=source_index,
        selected_keyframe_ids=np.asarray([record["keyframe_id"] for record in records], dtype="U160"),
        selected_workspace_orders=np.asarray([record["workspace_order"] for record in records], dtype=np.int32),
        camera_centers_reconstruction=np.asarray(centers, dtype=np.float64),
        sparse_landmarks_reconstruction=sparse_points,
        sparse_landmark_colors_rgb=sparse_colors,
    )
    ply_path = _write_d4_binary_ply(
        output_dir / "d4_3_selected_repaired_ten_view_raw_accumulation.ply", points, colors
    )
    return {"combined_npz": str(npz_path.resolve()), "combined_ply": str(ply_path.resolve())}


def main() -> int:
    args = _parse_args()
    d4_dir = args.d4_dir.expanduser().resolve()
    d41_dir = args.d4_1_dir.expanduser().resolve()
    d42_dir = args.d4_2_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    d4_summary = _read_json(d4_dir / "d4_summary.json")
    d41_summary = _read_json(d41_dir / "d4_1_summary.json")
    d42_summary = _read_json(d42_dir / "d4_2_summary.json")
    weak_left, weak_right = _validate_source_evidence(d4_summary, d41_summary, d42_summary)
    thresholds = _thresholds_from_d4_1(d41_summary)

    b2_dir = Path(str(d4_summary["source_b2_dir"])).expanduser().resolve()
    b2_summary = _read_json(b2_dir / "b2_summary.json")
    sfm_result = _sfm_result_from_payload(_read_json(b2_dir / "sfm_result.json"))
    if sfm_result.gauge.scale_status is not SfmScaleStatus.ARBITRARY:
        raise ReconstructionValidationError("D4.3 expects the SfM world to remain arbitrary-scale.")

    source_views = list(d4_summary["views"])
    source_orders = [int(item["workspace_order"]) for item in source_views]
    predecessor_order = min(source_orders) - 1
    candidate_orders = _candidate_workspace_orders(
        source_orders,
        predecessor_order=predecessor_order,
        weak_left_order=weak_left,
        weak_right_order=weak_right,
    )
    print(
        f"D4.3: weak pair orders={weak_left}<->{weak_right}; predecessor order={predecessor_order}",
        flush=True,
    )
    print(f"D4.3: candidate A={'->'.join(map(str, candidate_orders['A']))} (remove order {weak_left})", flush=True)
    print(f"D4.3: candidate B={'->'.join(map(str, candidate_orders['B']))} (remove order {weak_right})", flush=True)

    predecessor = _generate_predecessor_view(
        sfm_result=sfm_result,
        b2_summary=b2_summary,
        predecessor_order=predecessor_order,
        output_dir=output_dir,
        args=args,
    )
    predecessor_summary = predecessor["summary"]

    summaries_by_order = {int(item["workspace_order"]): dict(item) for item in source_views}
    summaries_by_order[predecessor_order] = predecessor_summary
    sparse_points, sparse_colors = _all_sparse_landmarks(sfm_result)

    candidate_records: dict[str, list[dict[str, Any]]] = {}
    candidate_summaries: dict[str, dict[str, Any]] = {}
    for name in ("A", "B"):
        summaries = []
        for new_index, order in enumerate(candidate_orders[name]):
            item = dict(summaries_by_order[order])
            item["view_index"] = new_index
            summaries.append(item)
        records = [_load_record(item, sfm_result) for item in summaries]
        candidate_records[name] = records
        candidate_summaries[name] = _diagnose_candidate(
            name=name,
            records=records,
            thresholds=thresholds,
            sparse_landmarks=sparse_points,
            output_dir=output_dir,
            max_points_per_direction=int(args.max_points_per_direction),
            nn_max_points_per_view=int(args.nn_max_points_per_view),
        )

    selected, reason = _select_repaired_candidate(candidate_summaries)
    selected_artifacts: dict[str, str] | None = None
    if selected is not None:
        selected_artifacts = _write_selected_accumulation(
            output_dir, candidate_records[selected], sparse_points, sparse_colors
        )

    summary = {
        "phase": "D4.3",
        "pass": selected is not None,
        "status": "pass" if selected is not None else "degraded",
        "source_d4_dir": str(d4_dir),
        "source_d4_1_dir": str(d41_dir),
        "source_d4_2_dir": str(d42_dir),
        "source_sfm_session_id": str(d4_summary["source_sfm_session_id"]),
        "weak_adjacent_pair_workspace_orders": [weak_left, weak_right],
        "predecessor_workspace_order": predecessor_order,
        "predecessor_view": predecessor_summary,
        "candidate_policy": {
            "candidate_A": f"prepend order {predecessor_order}, remove weak-left order {weak_left}",
            "candidate_B": f"prepend order {predecessor_order}, remove weak-right order {weak_right}",
            "qualification": "All nine adjacent pairs are recomputed under unchanged D4.1/D2 thresholds.",
            "selection": "If both pass, choose the stronger worst-case dense-consistency candidate; no threshold relaxation.",
        },
        "thresholds_reused_unchanged_from_d4_1": thresholds,
        "candidates": candidate_summaries,
        "selection": {
            "selected_candidate": selected,
            "reason": reason,
            "ten_view_chain_qualified": selected is not None,
            "authorizes_next_fusion_milestone": selected is not None,
        },
        "selected_artifacts": selected_artifacts,
        "authority": {
            "sfm_scale_status": "arbitrary",
            "sfm_world_alignment_status": "arbitrary",
            "metric_scale_claimed": False,
            "canonical_geometry_authoritative": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "spatial_merge_applied": False,
            "voxel_fusion_applied": False,
            "multi_view_filtering_applied": False,
            "note": "D4.3 repairs/qualifies view selection only; selected raw geometry remains unfused and non-authoritative.",
        },
        "da3_environment_probe": predecessor["da3_environment_probe"],
    }
    summary_path = _write_json(output_dir / "d4_3_summary.json", summary)

    print(f"\nD4.3 repaired ten-view chain qualification: {summary['status'].upper()}")
    for name in ("A", "B"):
        candidate = candidate_summaries[name]
        print(
            f"candidate[{name}] status={candidate['status']} "
            f"orders={'->'.join(map(str, candidate['workspace_orders']))}"
        )
        for item in candidate["adjacent_pairs"]:
            print(
                f"  pair[{item['pair_label']}] orders={item['workspace_orders'][0]}<->{item['workspace_orders'][1]} "
                f"status={item['status']} "
                f"consistent={item['a_to_b']['consistent_ratio_of_comparable']:.3f}/"
                f"{item['b_to_a']['consistent_ratio_of_comparable']:.3f} "
                f"median={item['a_to_b']['median_absolute_relative_depth_residual']:.4f}/"
                f"{item['b_to_a']['median_absolute_relative_depth_residual']:.4f}"
            )
    print(f"selected_candidate={selected} reason={reason}")
    print(f"ten_view_chain_qualified={selected is not None}")
    print(f"authorizes_next_fusion_milestone={selected is not None}")
    print("voxel_fusion_applied=False multi_view_filtering_applied=False metric_scale_claimed=False")
    if selected_artifacts is not None:
        print(f"selected_cloud={selected_artifacts['combined_npz']}")
    print(f"summary={summary_path}")
    return 0 if selected is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())

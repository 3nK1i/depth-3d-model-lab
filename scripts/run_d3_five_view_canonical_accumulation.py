#!/usr/bin/env python3
"""D3 controlled five-view canonical dense RGB accumulation validation.

D3 expands the qualified three-view result into exactly five registered views
without introducing fusion.  The selector is overlap-aware: by default it
scores contiguous five-view windows using shared sparse SfM tracks and camera
baseline, and it can be constrained to contain an already-qualified three-view
triplet (for example the D2.2 13/14/15 chain).

Each selected RGB keyframe receives an independent isolated DA3 depth estimate,
B1 SfM-depth scale alignment, and Phase C canonical XYZRGB projection.  The five
canonical clouds are concatenated only for inspection.  No spatial merge,
voxel fusion, multi-view filtering, persistence, metric-scale qualification,
measurement authority, or certified metrology is performed here.
"""

from __future__ import annotations

import argparse
from collections import Counter
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
    OfflineSfmResult,
    SfmCameraEstimate,
    SfmScaleStatus,
    SfmWorkspaceImage,
    align_sfm_keyframe_dense_depth,
    project_sfm_keyframe_dense_rgb,
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
from scripts.run_d2_three_view_overlap_diagnostics import _robust_scene_scale  # noqa: E402

_DEFAULT_DA3_PYTHON = Path.home() / "anaconda3/envs/eia-da3/bin/python"
_DEFAULT_DA3_REPO = Path.home() / "Desktop/eia-da3-backends/depth-anything-3"
_DA3_WORKER = ROOT / "scripts/da3_workers/da3_worker.py"
_D3_VIEW_COUNT = 5
_D3_QUALIFIED_COUNT = 3


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--b2-dir",
        type=Path,
        default=Path("/tmp/eia_b2_real_alignment"),
        help="Validated B2 output directory whose globally optimized SfM world is reused.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/eia_d3_five_view_accumulation"),
    )
    parser.add_argument("--da3-python", type=Path, default=_DEFAULT_DA3_PYTHON)
    parser.add_argument("--da3-repo", type=Path, default=_DEFAULT_DA3_REPO)
    parser.add_argument("--da3-model", default="")
    parser.add_argument("--da3-model-revision", default="")
    parser.add_argument("--process-res", type=int, default=392)
    parser.add_argument("--da3-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--minimum-correspondences", type=int, default=8)
    parser.add_argument("--maximum-relative-residual", type=float, default=0.35)
    parser.add_argument("--pixel-stride", type=int, default=1)
    parser.add_argument(
        "--keyframe-ids",
        nargs=5,
        default=None,
        metavar=("KF0", "KF1", "KF2", "KF3", "KF4"),
        help="Optional exact five registered keyframes. Overrides automatic selection.",
    )
    parser.add_argument(
        "--qualified-keyframe-ids",
        nargs=3,
        default=None,
        metavar=("QKF0", "QKF1", "QKF2"),
        help=(
            "Optional already-qualified consecutive three-view chain that the automatic "
            "five-view window must contain."
        ),
    )
    args = parser.parse_args()
    if args.process_res <= 0:
        parser.error("--process-res must be positive.")
    if args.da3_timeout_seconds <= 0.0:
        parser.error("--da3-timeout-seconds must be positive.")
    if args.minimum_correspondences < 3:
        parser.error("--minimum-correspondences must be at least 3.")
    if not 0.0 < args.maximum_relative_residual < 1.0:
        parser.error("--maximum-relative-residual must be in (0,1).")
    if args.pixel_stride <= 0:
        parser.error("--pixel-stride must be positive.")
    if args.keyframe_ids is not None and len(set(args.keyframe_ids)) != _D3_VIEW_COUNT:
        parser.error("--keyframe-ids must contain exactly five distinct keyframe IDs.")
    if args.qualified_keyframe_ids is not None and len(set(args.qualified_keyframe_ids)) != _D3_QUALIFIED_COUNT:
        parser.error("--qualified-keyframe-ids must contain exactly three distinct keyframe IDs.")
    if args.keyframe_ids is not None and args.qualified_keyframe_ids is not None:
        parser.error("Use either --keyframe-ids or --qualified-keyframe-ids, not both.")
    return args


def _shared_track_counts(sfm_result: OfflineSfmResult) -> Counter[tuple[str, str]]:
    """Count sparse landmarks observed by each unordered registered keyframe pair."""

    registered = {camera.keyframe_id for camera in sfm_result.cameras}
    counts: Counter[tuple[str, str]] = Counter()
    for landmark in sfm_result.landmarks:
        ids = sorted({obs.keyframe_id for obs in landmark.observations if obs.keyframe_id in registered})
        for left_index in range(len(ids)):
            for right_index in range(left_index + 1, len(ids)):
                counts[(ids[left_index], ids[right_index])] += 1
    return counts


def _adjacent_chain_evidence(
    selected: Iterable[tuple[SfmCameraEstimate, SfmWorkspaceImage, int]],
    *,
    shared_counts: Counter[tuple[str, str]],
    scene_scale: float,
) -> tuple[dict[str, Any], ...]:
    values = tuple(selected)
    evidence: list[dict[str, Any]] = []
    for pair_index in range(len(values) - 1):
        camera_a, image_a, obs_a = values[pair_index]
        camera_b, image_b, obs_b = values[pair_index + 1]
        key = tuple(sorted((camera_a.keyframe_id, camera_b.keyframe_id)))
        shared = int(shared_counts.get(key, 0))
        denom = max(1, min(int(obs_a), int(obs_b)))
        baseline = float(
            np.linalg.norm(camera_a.camera_center_reconstruction - camera_b.camera_center_reconstruction)
        )
        evidence.append(
            {
                "pair_index": pair_index,
                "workspace_orders": [int(image_a.order), int(image_b.order)],
                "keyframe_ids": [camera_a.keyframe_id, camera_b.keyframe_id],
                "shared_sparse_landmarks": shared,
                "shared_over_min_observations": float(shared / denom),
                "camera_baseline_reconstruction_units": baseline,
                "camera_baseline_scene_fraction": float(baseline / scene_scale),
            }
        )
    return tuple(evidence)


def _select_five_registered_keyframes(
    sfm_result: OfflineSfmResult,
    *,
    explicit_keyframe_ids: Iterable[str] | None = None,
    qualified_keyframe_ids: Iterable[str] | None = None,
) -> tuple[tuple[SfmCameraEstimate, SfmWorkspaceImage, int], ...]:
    """Select five registered views using sparse overlap and baseline-aware scoring.

    Automatic selection considers contiguous five-view windows in registered
    workspace order.  If a qualified three-view chain is supplied, only windows
    containing that exact consecutive chain are eligible.  Candidate windows are
    ranked first by weakest adjacent sparse overlap ratio, then weakest shared
    track count, then by the smallest worst adjacent baseline, then total sparse
    support.  This deliberately favors a chain whose neighboring views actually
    observe the same geometry.
    """

    cameras = {camera.keyframe_id: camera for camera in sfm_result.cameras}
    workspace = {
        image.keyframe_id: image
        for image in sfm_result.workspace.images
        if image.keyframe_id in cameras
    }
    ordered = sorted(workspace.values(), key=lambda item: (item.order, item.keyframe_id))
    if len(ordered) < _D3_VIEW_COUNT:
        raise ReconstructionValidationError("D3 requires at least five registered workspace keyframes.")

    observation_counts = _observation_counts(sfm_result)
    shared_counts = _shared_track_counts(sfm_result)
    sparse_points, _ = _all_sparse_landmarks(sfm_result)
    camera_centers = [camera.camera_center_reconstruction for camera in sfm_result.cameras]
    scene_scale, _source, _low, _high = _robust_scene_scale(sparse_points, camera_centers)

    if explicit_keyframe_ids is not None:
        ids = tuple(str(value).strip() for value in explicit_keyframe_ids)
        if len(ids) != _D3_VIEW_COUNT or len(set(ids)) != _D3_VIEW_COUNT or any(not value for value in ids):
            raise ReconstructionValidationError("D3 explicit selection requires exactly five distinct keyframe IDs.")
        missing = [value for value in ids if value not in workspace]
        if missing:
            raise ReconstructionValidationError(
                "D3 explicit keyframes must all be registered/materialized: " + ", ".join(missing)
            )
        chosen_images = sorted((workspace[value] for value in ids), key=lambda item: item.order)
    else:
        required_ids: tuple[str, ...] | None = None
        if qualified_keyframe_ids is not None:
            required_ids = tuple(str(value).strip() for value in qualified_keyframe_ids)
            if (
                len(required_ids) != _D3_QUALIFIED_COUNT
                or len(set(required_ids)) != _D3_QUALIFIED_COUNT
                or any(not value for value in required_ids)
            ):
                raise ReconstructionValidationError(
                    "D3 qualified chain requires exactly three distinct keyframe IDs."
                )
            missing = [value for value in required_ids if value not in workspace]
            if missing:
                raise ReconstructionValidationError(
                    "D3 qualified keyframes must all be registered/materialized: " + ", ".join(missing)
                )
            positions = [next(index for index, item in enumerate(ordered) if item.keyframe_id == value) for value in required_ids]
            if positions != sorted(positions) or positions[1] != positions[0] + 1 or positions[2] != positions[1] + 1:
                raise ReconstructionValidationError(
                    "D3 qualified keyframes must be supplied in consecutive registered workspace order."
                )

        candidates: list[tuple[tuple[float, int, float, int, int], list[SfmWorkspaceImage]]] = []
        for start in range(0, len(ordered) - _D3_VIEW_COUNT + 1):
            window = ordered[start : start + _D3_VIEW_COUNT]
            window_ids = [item.keyframe_id for item in window]
            if required_ids is not None and not all(value in window_ids for value in required_ids):
                continue
            selected = tuple(
                (cameras[item.keyframe_id], item, int(observation_counts.get(item.keyframe_id, 0)))
                for item in window
            )
            pair_evidence = _adjacent_chain_evidence(
                selected,
                shared_counts=shared_counts,
                scene_scale=scene_scale,
            )
            min_overlap_ratio = min(float(item["shared_over_min_observations"]) for item in pair_evidence)
            min_shared = min(int(item["shared_sparse_landmarks"]) for item in pair_evidence)
            max_baseline_fraction = max(float(item["camera_baseline_scene_fraction"]) for item in pair_evidence)
            total_observations = sum(int(item[2]) for item in selected)
            # max() prefers higher overlap/shared support, smaller worst baseline,
            # stronger total support, and then an earlier stable window.
            score = (
                min_overlap_ratio,
                min_shared,
                -max_baseline_fraction,
                total_observations,
                -int(window[0].order),
            )
            candidates.append((score, window))
        if not candidates:
            raise ReconstructionValidationError(
                "D3 could not find a contiguous five-view window satisfying the requested qualified chain."
            )
        chosen_images = max(candidates, key=lambda item: item[0])[1]

    return tuple(
        (cameras[item.keyframe_id], item, int(observation_counts.get(item.keyframe_id, 0)))
        for item in chosen_images
    )


def _accumulate_five_view_clouds(
    clouds: Iterable[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    point_parts: list[np.ndarray] = []
    color_parts: list[np.ndarray] = []
    view_parts: list[np.ndarray] = []
    for view_index, (points_value, colors_value) in enumerate(clouds):
        points = np.asarray(points_value, dtype=np.float64)
        colors = np.asarray(colors_value, dtype=np.uint8)
        if points.ndim != 2 or points.shape[1:] != (3,) or colors.shape != points.shape:
            raise ReconstructionValidationError("Each D3 raw view cloud must contain matching Nx3 XYZ/RGB arrays.")
        point_parts.append(points)
        color_parts.append(colors)
        view_parts.append(np.full(points.shape[0], view_index, dtype=np.uint8))
    if len(point_parts) != _D3_VIEW_COUNT:
        raise ReconstructionValidationError("D3 accumulation requires exactly five projected view clouds.")
    return (
        np.ascontiguousarray(np.concatenate(point_parts, axis=0)),
        np.ascontiguousarray(np.concatenate(color_parts, axis=0)),
        np.ascontiguousarray(np.concatenate(view_parts, axis=0)),
    )



def _write_d3_binary_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> Path:
    if points.shape != colors.shape or points.ndim != 2 or points.shape[1:] != (3,):
        raise ReconstructionValidationError("D3 PLY points/colors must be matching Nx3 arrays.")
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment EIA TECH D3 raw five-view canonical accumulation; NOT fused\n"
        f"element vertex {points.shape[0]}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    dtype = np.dtype(
        [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")]
    )
    vertices = np.empty(points.shape[0], dtype=dtype)
    vertices["x"] = points[:, 0].astype(np.float32)
    vertices["y"] = points[:, 1].astype(np.float32)
    vertices["z"] = points[:, 2].astype(np.float32)
    vertices["r"] = colors[:, 0]
    vertices["g"] = colors[:, 1]
    vertices["b"] = colors[:, 2]
    with path.open("wb") as handle:
        handle.write(header)
        handle.write(vertices.tobytes())
    return path

def main() -> int:
    args = _parse_args()
    b2_dir = args.b2_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    b2_summary = _read_json(b2_dir / "b2_summary.json")
    if not bool(b2_summary.get("pass")) or not bool(b2_summary.get("real_sfm_executed")):
        raise ReconstructionValidationError("D3 requires a passing real B2 result.")
    sfm_result = _sfm_result_from_payload(_read_json(b2_dir / "sfm_result.json"))
    if sfm_result.gauge.scale_status is not SfmScaleStatus.ARBITRARY:
        raise ReconstructionValidationError("D3 expects the validated B2 SfM world to remain arbitrary-scale.")

    selected = _select_five_registered_keyframes(
        sfm_result,
        explicit_keyframe_ids=args.keyframe_ids,
        qualified_keyframe_ids=args.qualified_keyframe_ids,
    )
    if args.keyframe_ids is not None:
        selection_mode = "explicit_five"
    elif args.qualified_keyframe_ids is not None:
        selection_mode = "overlap_aware_five_containing_qualified_triplet"
    else:
        selection_mode = "overlap_aware_best_contiguous_five"

    shared_counts = _shared_track_counts(sfm_result)
    sparse_points, sparse_colors = _all_sparse_landmarks(sfm_result)
    camera_centers_all = [camera.camera_center_reconstruction for camera in sfm_result.cameras]
    scene_scale, scene_scale_source, scene_low, scene_high = _robust_scene_scale(
        sparse_points, camera_centers_all
    )
    pair_selection_evidence = _adjacent_chain_evidence(
        selected,
        shared_counts=shared_counts,
        scene_scale=scene_scale,
    )

    print(f"D3: selected 5 registered keyframes ({selection_mode}):", flush=True)
    for index, (camera, image, count) in enumerate(selected):
        print(
            f"  view[{index}] order={image.order} keyframe={camera.keyframe_id} sparse_observations={count}",
            flush=True,
        )
    print("D3: sparse-overlap selection evidence:", flush=True)
    for item in pair_selection_evidence:
        print(
            f"  pair[{item['pair_index']}-{item['pair_index'] + 1}] "
            f"shared={item['shared_sparse_landmarks']} "
            f"overlap_ratio={item['shared_over_min_observations']:.3f} "
            f"baseline_scene_fraction={item['camera_baseline_scene_fraction']:.4f}",
            flush=True,
        )

    da3_python = args.da3_python.expanduser().resolve()
    da3_repo = args.da3_repo.expanduser().resolve()
    da3_probe = probe_da3_environment(da3_python, da3_repo)
    if not da3_probe.available:
        raise ReconstructionValidationError(
            "D3 DA3 environment is not qualified: " + json.dumps(da3_probe.as_dict(), sort_keys=True)
        )
    b2_da3_manifest = b2_summary.get("dense_depth", {}).get("manifest", {})
    model_id = str(args.da3_model).strip() or str(b2_da3_manifest.get("model_id", "")).strip() or DA3_EFFICIENT_MODEL_ID
    model_revision = str(args.da3_model_revision).strip() or str(b2_da3_manifest.get("model_revision", "")).strip()

    camera_models = {item.camera_id: item for item in sfm_result.camera_models}
    projected_views = []
    view_summaries: list[dict[str, Any]] = []
    selected_camera_centers: list[np.ndarray] = []

    for view_index, (camera, workspace_image, observation_count) in enumerate(selected):
        camera_model = camera_models.get(camera.camera_id)
        if camera_model is None:
            raise ReconstructionValidationError(f"D3 selected camera {camera.camera_id!r} has no SfM model.")
        rgb_path = Path(sfm_result.workspace.workspace_root).resolve() / workspace_image.staged_image_path
        bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ReconstructionValidationError(f"D3 could not decode selected RGB keyframe: {rgb_path}")
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)

        da3_root = output_dir / f"da3_view_{view_index:02d}"
        input_manifest_path = write_da3_input_manifest(
            da3_root / "input_manifest.json",
            DA3InputManifest(
                request_id=f"d3-{sfm_result.session_id}-{view_index}",
                image_paths=(str(rgb_path.resolve()),),
                metadata={
                    "phase": "D3",
                    "session_id": sfm_result.session_id,
                    "view_index": view_index,
                    "sfm_keyframe_id": camera.keyframe_id,
                    "purpose": "controlled_five_view_canonical_accumulation",
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
        print(f"D3: running isolated DA3 for view[{view_index}]...", flush=True)
        da3_manifest_path, da3_manifest = runner.run(input_manifest_path, da3_root / "worker_output")
        da3_arrays = validate_da3_npz_result(da3_manifest)
        if da3_manifest.frame_count != 1 or da3_arrays["depth"].shape[0] != 1:
            raise ReconstructionValidationError("D3 expects each DA3 request to return exactly one frame.")
        depth_sfm_grid, pixel_adapter = _adapt_da3_depth_to_sfm_pixel_space(
            da3_arrays["depth"][0],
            sfm_width=camera_model.image_size.width,
            sfm_height=camera_model.image_size.height,
        )
        depth_provenance = (
            f"Depth Anything 3 model={da3_manifest.model_id} revision={da3_manifest.model_revision};"
            f"source_revision={da3_manifest.source_revision};pixel_adapter={pixel_adapter['mode']};"
            f"D3 raw view {view_index}; no fusion; no metric scale claim"
        )
        evidence = align_sfm_keyframe_dense_depth(
            sfm_result,
            keyframe_id=camera.keyframe_id,
            dense_depth_source_units=depth_sfm_grid,
            depth_provenance=depth_provenance,
            minimum_correspondences=int(args.minimum_correspondences),
            maximum_relative_residual=float(args.maximum_relative_residual),
        )
        evidence_path = _write_json(output_dir / f"d3_view_{view_index:02d}_alignment.json", evidence.as_dict())
        if not evidence.is_accepted or evidence.alignment is None:
            summary = {
                "phase": "D3",
                "pass": False,
                "failure": f"view[{view_index}] alignment rejected",
                "rejection_code": evidence.rejection_code.value if evidence.rejection_code else None,
                "keyframe_id": camera.keyframe_id,
                "alignment_evidence": str(evidence_path.resolve()),
                "raw_accumulation_only": True,
                "fusion_applied": False,
            }
            _write_json(output_dir / "d3_summary.json", summary)
            print(
                f"D3 five-view canonical accumulation: REJECTED view[{view_index}] "
                f"reason={summary['rejection_code']}",
                flush=True,
            )
            return 2

        projection = project_sfm_keyframe_dense_rgb(
            camera,
            camera_model,
            timestamp_ns=workspace_image.timestamp_ns,
            rgb=rgb,
            dense_depth_source_units=depth_sfm_grid,
            alignment_evidence=evidence,
            pixel_stride=int(args.pixel_stride),
            reconstruction_frame_id="offline-sfm-reconstruction",
        )
        view_npz = output_dir / f"d3_view_{view_index:02d}_canonical_cloud.npz"
        np.savez_compressed(
            view_npz,
            points_reconstruction=projection.points_reconstruction,
            colors_rgb=projection.colors_rgb,
            aligned_depth_reconstruction_units=np.asarray(
                depth_sfm_grid * evidence.alignment.scale_to_reconstruction_units, dtype=np.float64
            ),
            T_reconstruction_from_camera=camera.T_reconstruction_from_camera,
            camera_center_reconstruction=camera.camera_center_reconstruction,
            keyframe_id=np.asarray(camera.keyframe_id),
        )
        projected_views.append(projection)
        selected_camera_centers.append(camera.camera_center_reconstruction)
        view_summaries.append(
            {
                "view_index": view_index,
                "workspace_order": workspace_image.order,
                "source_sequence_index": workspace_image.source_sequence_index,
                "keyframe_id": camera.keyframe_id,
                "frame_id": camera.frame_id,
                "camera_id": camera.camera_id,
                "camera_model_name": camera_model.backend_model_name,
                "sparse_observations": observation_count,
                "candidate_correspondences": evidence.candidate_observations,
                "valid_correspondences": evidence.valid_correspondences,
                "inlier_correspondences": evidence.inlier_correspondences,
                "scale_to_reconstruction_units": evidence.alignment.scale_to_reconstruction_units,
                "relative_scale_uncertainty": evidence.alignment.relative_scale_uncertainty,
                "median_relative_residual": evidence.median_relative_residual,
                "p90_relative_residual": evidence.p90_relative_residual,
                "pixel_adapter": pixel_adapter,
                "point_count": int(projection.points_reconstruction.shape[0]),
                "bounds_min_reconstruction": projection.points_reconstruction.min(axis=0).tolist(),
                "bounds_max_reconstruction": projection.points_reconstruction.max(axis=0).tolist(),
                "camera_center_reconstruction": camera.camera_center_reconstruction.tolist(),
                "alignment_evidence_json": str(evidence_path.resolve()),
                "cloud_npz": str(view_npz.resolve()),
                "da3_result_manifest": str(da3_manifest_path.resolve()),
            }
        )
        print(
            f"D3: view[{view_index}] ACCEPTED scale={evidence.alignment.scale_to_reconstruction_units:.8f} "
            f"inliers={evidence.inlier_correspondences}/{evidence.valid_correspondences} "
            f"points={projection.points_reconstruction.shape[0]}",
            flush=True,
        )

    points, colors, source_view_index = _accumulate_five_view_clouds(
        (item.points_reconstruction, item.colors_rgb) for item in projected_views
    )
    combined_npz = output_dir / "d3_five_view_raw_accumulation.npz"
    np.savez_compressed(
        combined_npz,
        points_reconstruction=points,
        colors_rgb=colors,
        source_view_index=source_view_index,
        selected_keyframe_ids=np.asarray([item[0].keyframe_id for item in selected], dtype="U160"),
        camera_centers_reconstruction=np.asarray(selected_camera_centers, dtype=np.float64),
        sparse_landmarks_reconstruction=sparse_points,
        sparse_landmark_colors_rgb=sparse_colors,
    )
    combined_ply = _write_d3_binary_ply(output_dir / "d3_five_view_raw_accumulation.ply", points, colors)

    scales = np.asarray([item["scale_to_reconstruction_units"] for item in view_summaries], dtype=np.float64)
    summary = {
        "phase": "D3",
        "pass": True,
        "source_b2_dir": str(b2_dir),
        "source_b2_pass": True,
        "source_sfm_session_id": sfm_result.session_id,
        "selection_mode": selection_mode,
        "qualified_keyframe_ids": (
            None if args.qualified_keyframe_ids is None else [str(value) for value in args.qualified_keyframe_ids]
        ),
        "view_count": _D3_VIEW_COUNT,
        "views": view_summaries,
        "selection_evidence": {
            "scene_scale": {
                "robust_diagonal_reconstruction_units": scene_scale,
                "source": scene_scale_source,
                "quantile_bounds_min": list(scene_low),
                "quantile_bounds_max": list(scene_high),
                "physical_unit_claimed": False,
            },
            "adjacent_pairs": list(pair_selection_evidence),
            "policy": (
                "Contiguous registered five-view windows ranked by weakest adjacent shared-track ratio, "
                "then weakest shared-track count, then smallest worst camera baseline, then total sparse support."
            ),
        },
        "raw_accumulation": {
            "point_count": int(points.shape[0]),
            "bounds_min_reconstruction": points.min(axis=0).tolist(),
            "bounds_max_reconstruction": points.max(axis=0).tolist(),
            "scale_min": float(scales.min()),
            "scale_max": float(scales.max()),
            "scale_ratio_max_to_min": float(scales.max() / scales.min()),
            "spatial_merge_applied": False,
            "voxel_fusion_applied": False,
            "multi_view_filtering_applied": False,
            "purpose": "Expose raw five-view registration before any fusion can hide disagreement.",
        },
        "authority": {
            "sfm_scale_status": sfm_result.gauge.scale_status.value,
            "sfm_world_alignment_status": sfm_result.gauge.alignment_status.value,
            "metric_scale_claimed": False,
            "canonical_geometry_authoritative": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "note": "D3 validation-only raw five-view accumulation; fusion/persistence have not run.",
        },
        "da3_environment_probe": da3_probe.as_dict(),
        "artifacts": {
            "combined_npz": str(combined_npz.resolve()),
            "combined_ply": str(combined_ply.resolve()),
        },
    }
    summary_path = _write_json(output_dir / "d3_summary.json", summary)

    print("\nD3 five-view canonical accumulation: PASS")
    for item in view_summaries:
        print(
            f"view[{item['view_index']}] order={item['workspace_order']} keyframe={item['keyframe_id']} "
            f"scale={item['scale_to_reconstruction_units']:.8f} "
            f"inliers={item['inlier_correspondences']}/{item['valid_correspondences']} "
            f"points={item['point_count']}"
        )
    print(f"raw_points={points.shape[0]} scale_ratio_max_to_min={summary['raw_accumulation']['scale_ratio_max_to_min']:.6f}")
    print("spatial_merge_applied=False voxel_fusion_applied=False multi_view_filtering_applied=False")
    print("sfm_scale_status=arbitrary metric_scale_claimed=False")
    print("canonical_geometry_authoritative=False measurement_authoritative=False certified_metrology=False")
    print(f"cloud={combined_npz}")
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

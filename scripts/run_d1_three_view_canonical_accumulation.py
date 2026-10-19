#!/usr/bin/env python3
"""D1 real three-view canonical dense RGB accumulation validation.

D1 deliberately reuses the already-validated B2 offline-SfM result instead of
rerunning PyCOLMAP.  It selects exactly three registered keyframes from that one
globally optimized SfM reconstruction, obtains a real isolated DA3 depth map for
each view, aligns every depth map independently into the common arbitrary SfM
gauge through B1, projects every RGB-D view through the Phase C bridge, and
concatenates the three canonical XYZRGB clouds without spatial merging.

This is raw multi-view accumulation for registration inspection only.  It is NOT
voxel fusion, persistent canonical geometry authority, metric scale,
measurement authority, or certified metrology.
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

from core.contracts import ImageSize  # noqa: E402
from core.reconstruction import (  # noqa: E402
    OfflineSfmResult,
    ReconstructionGauge,
    SfmCameraEstimate,
    SfmCameraModelEstimate,
    SfmLandmarkEstimate,
    SfmObservation,
    SfmScaleStatus,
    SfmStageStatistics,
    SfmWorkspaceImage,
    SfmWorkspaceManifest,
    SfmWorldAlignmentStatus,
    align_sfm_keyframe_dense_depth,
    project_sfm_keyframe_dense_rgb,
)
from core.reconstruction.contracts import (  # noqa: E402
    InputSourceKind,
    InputSourceMetadata,
    TimestampOrigin,
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

_MIB = 1024**2
_DEFAULT_DA3_PYTHON = Path.home() / "anaconda3/envs/depth3d-da3/bin/python"
_DEFAULT_DA3_REPO = Path.home() / "Desktop/depth3d-da3-backends/depth-anything-3"
_DA3_WORKER = ROOT / "scripts/da3_workers/da3_worker.py"
_D1_VIEW_COUNT = 3


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--b2-dir",
        type=Path,
        default=Path("/tmp/depth3d_b2_real_alignment"),
        help="Validated B2 output directory whose SfM world frame is reused.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/depth3d_d1_three_view_accumulation"),
    )
    parser.add_argument("--da3-python", type=Path, default=_DEFAULT_DA3_PYTHON)
    parser.add_argument("--da3-repo", type=Path, default=_DEFAULT_DA3_REPO)
    parser.add_argument(
        "--da3-model",
        default="",
        help="Optional model override. Empty reuses the model recorded by B2.",
    )
    parser.add_argument(
        "--da3-model-revision",
        default="",
        help="Optional model revision override. Empty reuses the revision recorded by B2.",
    )
    parser.add_argument("--process-res", type=int, default=392)
    parser.add_argument("--da3-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--minimum-correspondences", type=int, default=8)
    parser.add_argument("--maximum-relative-residual", type=float, default=0.35)
    parser.add_argument("--pixel-stride", type=int, default=1)
    parser.add_argument(
        "--keyframe-ids",
        nargs=3,
        default=None,
        metavar=("KF0", "KF1", "KF2"),
        help="Optional exact three registered keyframes. Default uses temporal thirds and keeps the B2 anchor.",
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
    if args.keyframe_ids is not None and len(set(args.keyframe_ids)) != _D1_VIEW_COUNT:
        parser.error("--keyframe-ids must contain exactly three distinct keyframe IDs.")
    return args


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError(f"Unable to read required D1 JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"Required D1 JSON root must be an object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def _image_size(payload: dict[str, Any] | None) -> ImageSize | None:
    if payload is None:
        return None
    return ImageSize(width=int(payload["width"]), height=int(payload["height"]))


def _sfm_result_from_payload(payload: dict[str, Any]) -> OfflineSfmResult:
    """Rehydrate the dependency-neutral B2 SfM JSON into validated core contracts."""

    gauge_payload = payload["gauge"]
    gauge = ReconstructionGauge(
        scale_status=SfmScaleStatus(str(gauge_payload["scale_status"])),
        alignment_status=SfmWorldAlignmentStatus(str(gauge_payload["alignment_status"])),
        unit_to_metre=(
            None if gauge_payload.get("unit_to_metre") is None else float(gauge_payload["unit_to_metre"])
        ),
        scale_source=str(gauge_payload.get("scale_source", "")),
        relative_scale_uncertainty=(
            None
            if gauge_payload.get("relative_scale_uncertainty") is None
            else float(gauge_payload["relative_scale_uncertainty"])
        ),
    )

    workspace_payload = payload["workspace"]
    source_payload = workspace_payload["source"]
    source = InputSourceMetadata(
        source_id=str(source_payload["source_id"]),
        kind=InputSourceKind(str(source_payload["kind"])),
        display_name=str(source_payload["display_name"]),
        source_uri=str(source_payload["source_uri"]),
        expected_image_size=_image_size(source_payload.get("expected_image_size")),
        expected_frame_rate_hz=(
            None
            if source_payload.get("expected_frame_rate_hz") is None
            else float(source_payload["expected_frame_rate_hz"])
        ),
        timestamp_origin=TimestampOrigin(str(source_payload.get("timestamp_origin", "unknown"))),
        fingerprint=str(source_payload.get("fingerprint", "")),
        metadata=source_payload.get("metadata", {}),
    )
    workspace_images = tuple(
        SfmWorkspaceImage(
            order=int(item["order"]),
            keyframe_id=str(item["keyframe_id"]),
            frame_id=str(item["frame_id"]),
            source_sequence_index=int(item["source_sequence_index"]),
            timestamp_ns=int(item["timestamp_ns"]),
            image_size=_image_size(item["image_size"]),  # type: ignore[arg-type]
            source_locator=str(item["source_locator"]),
            staged_image_path=str(item["staged_image_path"]),
            staged_sha256=str(item["staged_sha256"]),
        )
        for item in workspace_payload["images"]
    )
    paths = workspace_payload["paths"]
    workspace = SfmWorkspaceManifest(
        session_id=str(workspace_payload["session_id"]),
        source=source,
        workspace_root=str(workspace_payload["workspace_root"]),
        images_directory=str(paths["images_directory"]),
        database_path=str(paths["database_path"]),
        sparse_directory=str(paths["sparse_directory"]),
        checkpoints_directory=str(paths["checkpoints_directory"]),
        reports_directory=str(paths["reports_directory"]),
        logs_directory=str(paths["logs_directory"]),
        images=workspace_images,
        schema_version=int(workspace_payload.get("schema_version", 1)),
    )

    cameras = tuple(
        SfmCameraEstimate(
            estimate_id=str(item["estimate_id"]),
            camera_id=str(item["camera_id"]),
            keyframe_id=str(item["keyframe_id"]),
            frame_id=str(item["frame_id"]),
            backend_image_id=(
                None if item.get("backend_image_id") is None else int(item["backend_image_id"])
            ),
            T_reconstruction_from_camera=np.asarray(
                item["T_reconstruction_from_camera"], dtype=np.float64
            ),
        )
        for item in payload["cameras"]
    )
    camera_models = tuple(
        SfmCameraModelEstimate(
            camera_id=str(item["camera_id"]),
            backend_camera_id=int(item["backend_camera_id"]),
            backend_model_name=str(item["backend_model_name"]),
            image_size=_image_size(item["image_size"]),  # type: ignore[arg-type]
            parameters=tuple(float(value) for value in item["parameters"]),
        )
        for item in payload.get("camera_models", [])
    )
    landmarks: list[SfmLandmarkEstimate] = []
    for item in payload["landmarks"]:
        observations = tuple(
            SfmObservation(
                observation_id=str(observation["observation_id"]),
                landmark_id=str(observation["landmark_id"]),
                keyframe_id=str(observation["keyframe_id"]),
                feature_xy=tuple(float(value) for value in observation["feature_xy"]),
                feature_index=(
                    None
                    if observation.get("feature_index") is None
                    else int(observation["feature_index"])
                ),
                backend_point2d_index=(
                    None
                    if observation.get("backend_point2d_index") is None
                    else int(observation["backend_point2d_index"])
                ),
            )
            for observation in item["observations"]
        )
        color = item.get("color_rgb")
        landmarks.append(
            SfmLandmarkEstimate(
                landmark_id=str(item["landmark_id"]),
                track_id=str(item["track_id"]),
                position_reconstruction=np.asarray(item["position_reconstruction"], dtype=np.float64),
                observations=observations,
                color_rgb=(None if color is None else tuple(int(value) for value in color)),
            )
        )
    statistics_payload = payload["statistics"]
    statistics = SfmStageStatistics(
        input_keyframes=int(statistics_payload["input_keyframes"]),
        registered_keyframes=int(statistics_payload["registered_keyframes"]),
        landmarks=int(statistics_payload["landmarks"]),
        observations=int(statistics_payload["observations"]),
        elapsed_ns=int(statistics_payload["elapsed_ns"]),
        mean_reprojection_error_px=(
            None
            if statistics_payload.get("mean_reprojection_error_px") is None
            else float(statistics_payload["mean_reprojection_error_px"])
        ),
    )
    return OfflineSfmResult(
        session_id=str(payload["session_id"]),
        backend_id=str(payload["backend_id"]),
        backend_version=str(payload["backend_version"]),
        gauge=gauge,
        cameras=cameras,
        camera_models=camera_models,
        landmarks=tuple(landmarks),
        statistics=statistics,
        workspace=workspace,
        warnings=tuple(str(value) for value in payload.get("warnings", [])),
        metadata=payload.get("metadata", {}),
        schema_version=int(payload.get("schema_version", 1)),
    )


def _observation_counts(sfm_result: OfflineSfmResult) -> Counter[str]:
    registered = {camera.keyframe_id for camera in sfm_result.cameras}
    counts: Counter[str] = Counter()
    for landmark in sfm_result.landmarks:
        for observation in landmark.observations:
            if observation.keyframe_id in registered:
                counts[observation.keyframe_id] += 1
    return counts


def _select_three_registered_keyframes(
    sfm_result: OfflineSfmResult,
    *,
    b2_anchor_keyframe_id: str,
    explicit_keyframe_ids: Iterable[str] | None = None,
) -> tuple[tuple[SfmCameraEstimate, SfmWorkspaceImage, int], ...]:
    """Select three registered views while keeping D1 tied to the validated C/B2 anchor.

    The automatic policy splits the registered camera track into temporal thirds.
    The third containing the B2/C anchor keeps that exact keyframe; each remaining
    third contributes the view with the strongest sparse observation support.
    """

    cameras = {camera.keyframe_id: camera for camera in sfm_result.cameras}
    workspace = {
        image.keyframe_id: image
        for image in sfm_result.workspace.images
        if image.keyframe_id in cameras
    }
    counts = _observation_counts(sfm_result)
    ordered = sorted(workspace.values(), key=lambda item: (item.order, item.keyframe_id))
    if len(ordered) < _D1_VIEW_COUNT:
        raise ReconstructionValidationError("D1 requires at least three registered workspace keyframes.")

    if explicit_keyframe_ids is not None:
        values = tuple(str(value).strip() for value in explicit_keyframe_ids)
        if len(values) != _D1_VIEW_COUNT or len(set(values)) != _D1_VIEW_COUNT or any(not value for value in values):
            raise ReconstructionValidationError("D1 explicit selection requires exactly three distinct keyframe IDs.")
        missing = [value for value in values if value not in workspace]
        if missing:
            raise ReconstructionValidationError(
                "D1 explicit keyframes must all be registered/materialized: " + ", ".join(missing)
            )
        chosen = [workspace[value] for value in values]
    else:
        anchor = str(b2_anchor_keyframe_id).strip()
        if anchor not in workspace:
            raise ReconstructionValidationError("D1 cannot preserve the B2/C anchor because it is not registered.")
        bins: list[list[SfmWorkspaceImage]] = []
        n = len(ordered)
        for index in range(_D1_VIEW_COUNT):
            start = (index * n) // _D1_VIEW_COUNT
            end = ((index + 1) * n) // _D1_VIEW_COUNT
            bins.append(ordered[start:end])
        chosen = []
        for section in bins:
            if any(item.keyframe_id == anchor for item in section):
                selected = workspace[anchor]
            else:
                center = sum(float(item.order) for item in section) / float(len(section))
                selected = max(
                    section,
                    key=lambda item: (
                        int(counts.get(item.keyframe_id, 0)),
                        -abs(float(item.order) - center),
                        -int(item.order),
                    ),
                )
            chosen.append(selected)
        if len({item.keyframe_id for item in chosen}) != _D1_VIEW_COUNT:
            raise ReconstructionValidationError("D1 automatic selection did not produce three distinct views.")

    chosen.sort(key=lambda item: item.order)
    return tuple(
        (cameras[item.keyframe_id], item, int(counts.get(item.keyframe_id, 0)))
        for item in chosen
    )


def _all_sparse_landmarks(sfm_result: OfflineSfmResult) -> tuple[np.ndarray, np.ndarray]:
    if not sfm_result.landmarks:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.uint8)
    points = np.asarray(
        [item.position_reconstruction for item in sfm_result.landmarks], dtype=np.float64
    )
    colors = np.asarray(
        [item.color_rgb if item.color_rgb is not None else (255, 255, 255) for item in sfm_result.landmarks],
        dtype=np.uint8,
    )
    return points, colors


def _accumulate_view_clouds(
    clouds: Iterable[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    point_parts: list[np.ndarray] = []
    color_parts: list[np.ndarray] = []
    view_parts: list[np.ndarray] = []
    for view_index, (points_value, colors_value) in enumerate(clouds):
        points = np.asarray(points_value, dtype=np.float64)
        colors = np.asarray(colors_value, dtype=np.uint8)
        if points.ndim != 2 or points.shape[1:] != (3,) or colors.shape != points.shape:
            raise ReconstructionValidationError("Each D1 raw view cloud must contain matching Nx3 XYZ/RGB arrays.")
        point_parts.append(points)
        color_parts.append(colors)
        view_parts.append(np.full(points.shape[0], view_index, dtype=np.uint8))
    if len(point_parts) != _D1_VIEW_COUNT:
        raise ReconstructionValidationError("D1 accumulation requires exactly three projected view clouds.")
    return (
        np.ascontiguousarray(np.concatenate(point_parts, axis=0)),
        np.ascontiguousarray(np.concatenate(color_parts, axis=0)),
        np.ascontiguousarray(np.concatenate(view_parts, axis=0)),
    )


def _write_binary_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> Path:
    if points.shape != colors.shape or points.ndim != 2 or points.shape[1:] != (3,):
        raise ReconstructionValidationError("D1 PLY points/colors must be matching Nx3 arrays.")
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment Depth & 3D Model Lab D1 raw three-view canonical accumulation; NOT fused\n"
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
        raise ReconstructionValidationError("D1 requires a passing real B2 result.")
    sfm_payload = _read_json(b2_dir / "sfm_result.json")
    sfm_result = _sfm_result_from_payload(sfm_payload)
    if sfm_result.gauge.scale_status is not SfmScaleStatus.ARBITRARY:
        raise ReconstructionValidationError("D1 expects the validated B2 SfM world to remain arbitrary-scale.")

    selected = _select_three_registered_keyframes(
        sfm_result,
        b2_anchor_keyframe_id=str(b2_summary["sfm"]["selected_keyframe_id"]),
        explicit_keyframe_ids=args.keyframe_ids,
    )
    selection_mode = "explicit" if args.keyframe_ids is not None else "temporal_thirds_with_b2_anchor"
    print(f"D1: selected 3 registered keyframes ({selection_mode}):", flush=True)
    for index, (camera, image, count) in enumerate(selected):
        print(
            f"  view[{index}] order={image.order} keyframe={camera.keyframe_id} sparse_observations={count}",
            flush=True,
        )

    da3_python = args.da3_python.expanduser().resolve()
    da3_repo = args.da3_repo.expanduser().resolve()
    da3_probe = probe_da3_environment(da3_python, da3_repo)
    if not da3_probe.available:
        raise ReconstructionValidationError(
            "D1 DA3 environment is not qualified: " + json.dumps(da3_probe.as_dict(), sort_keys=True)
        )
    b2_da3_manifest = b2_summary.get("dense_depth", {}).get("manifest", {})
    model_id = str(args.da3_model).strip() or str(b2_da3_manifest.get("model_id", "")).strip() or DA3_EFFICIENT_MODEL_ID
    model_revision = str(args.da3_model_revision).strip() or str(b2_da3_manifest.get("model_revision", "")).strip()

    camera_models = {item.camera_id: item for item in sfm_result.camera_models}
    projected_views = []
    view_summaries: list[dict[str, Any]] = []
    camera_centers: list[np.ndarray] = []

    for view_index, (camera, workspace_image, observation_count) in enumerate(selected):
        camera_model = camera_models.get(camera.camera_id)
        if camera_model is None:
            raise ReconstructionValidationError(f"D1 selected camera {camera.camera_id!r} has no SfM model.")
        rgb_path = Path(sfm_result.workspace.workspace_root).resolve() / workspace_image.staged_image_path
        bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ReconstructionValidationError(f"D1 could not decode selected RGB keyframe: {rgb_path}")
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)

        da3_root = output_dir / f"da3_view_{view_index:02d}"
        input_manifest_path = write_da3_input_manifest(
            da3_root / "input_manifest.json",
            DA3InputManifest(
                request_id=f"d1-{sfm_result.session_id}-{view_index}",
                image_paths=(str(rgb_path.resolve()),),
                metadata={
                    "phase": "D1",
                    "session_id": sfm_result.session_id,
                    "view_index": view_index,
                    "sfm_keyframe_id": camera.keyframe_id,
                    "purpose": "raw_three_view_canonical_accumulation",
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
        print(f"D1: running isolated DA3 for view[{view_index}]...", flush=True)
        da3_manifest_path, da3_manifest = runner.run(input_manifest_path, da3_root / "worker_output")
        da3_arrays = validate_da3_npz_result(da3_manifest)
        if da3_manifest.frame_count != 1 or da3_arrays["depth"].shape[0] != 1:
            raise ReconstructionValidationError("D1 expects each DA3 request to return exactly one frame.")
        depth_sfm_grid, pixel_adapter = _adapt_da3_depth_to_sfm_pixel_space(
            da3_arrays["depth"][0],
            sfm_width=camera_model.image_size.width,
            sfm_height=camera_model.image_size.height,
        )
        depth_provenance = (
            f"Depth Anything 3 model={da3_manifest.model_id} revision={da3_manifest.model_revision};"
            f"source_revision={da3_manifest.source_revision};pixel_adapter={pixel_adapter['mode']};"
            f"D1 raw view {view_index}; no fusion; no metric scale claim"
        )
        evidence = align_sfm_keyframe_dense_depth(
            sfm_result,
            keyframe_id=camera.keyframe_id,
            dense_depth_source_units=depth_sfm_grid,
            depth_provenance=depth_provenance,
            minimum_correspondences=int(args.minimum_correspondences),
            maximum_relative_residual=float(args.maximum_relative_residual),
        )
        evidence_path = _write_json(output_dir / f"d1_view_{view_index:02d}_alignment.json", evidence.as_dict())
        if not evidence.is_accepted or evidence.alignment is None:
            summary = {
                "phase": "D1",
                "pass": False,
                "failure": f"view[{view_index}] alignment rejected",
                "rejection_code": evidence.rejection_code.value if evidence.rejection_code else None,
                "keyframe_id": camera.keyframe_id,
                "alignment_evidence": str(evidence_path.resolve()),
                "raw_accumulation_only": True,
                "fusion_applied": False,
            }
            _write_json(output_dir / "d1_summary.json", summary)
            print(
                f"D1 three-view canonical accumulation: REJECTED view[{view_index}] "
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
        view_npz = output_dir / f"d1_view_{view_index:02d}_canonical_cloud.npz"
        np.savez_compressed(
            view_npz,
            points_reconstruction=projection.points_reconstruction,
            colors_rgb=projection.colors_rgb,
            T_reconstruction_from_camera=camera.T_reconstruction_from_camera,
            camera_center_reconstruction=camera.camera_center_reconstruction,
            keyframe_id=np.asarray(camera.keyframe_id),
        )
        projected_views.append(projection)
        camera_centers.append(camera.camera_center_reconstruction)
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
            f"D1: view[{view_index}] ACCEPTED scale={evidence.alignment.scale_to_reconstruction_units:.8f} "
            f"inliers={evidence.inlier_correspondences}/{evidence.valid_correspondences} "
            f"points={projection.points_reconstruction.shape[0]}",
            flush=True,
        )

    points, colors, source_view_index = _accumulate_view_clouds(
        (item.points_reconstruction, item.colors_rgb) for item in projected_views
    )
    sparse_points, sparse_colors = _all_sparse_landmarks(sfm_result)
    combined_npz = output_dir / "d1_three_view_raw_accumulation.npz"
    np.savez_compressed(
        combined_npz,
        points_reconstruction=points,
        colors_rgb=colors,
        source_view_index=source_view_index,
        selected_keyframe_ids=np.asarray([item[0].keyframe_id for item in selected], dtype="U160"),
        camera_centers_reconstruction=np.asarray(camera_centers, dtype=np.float64),
        sparse_landmarks_reconstruction=sparse_points,
        sparse_landmark_colors_rgb=sparse_colors,
    )
    combined_ply = _write_binary_ply(output_dir / "d1_three_view_raw_accumulation.ply", points, colors)

    scales = np.asarray(
        [item["scale_to_reconstruction_units"] for item in view_summaries], dtype=np.float64
    )
    summary = {
        "phase": "D1",
        "pass": True,
        "source_b2_dir": str(b2_dir),
        "source_b2_pass": True,
        "source_sfm_session_id": sfm_result.session_id,
        "selection_mode": selection_mode,
        "b2_anchor_keyframe_id": str(b2_summary["sfm"]["selected_keyframe_id"]),
        "view_count": _D1_VIEW_COUNT,
        "views": view_summaries,
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
            "purpose": "Expose raw cross-view registration before any fusion can hide disagreement.",
        },
        "authority": {
            "sfm_scale_status": sfm_result.gauge.scale_status.value,
            "sfm_world_alignment_status": sfm_result.gauge.alignment_status.value,
            "metric_scale_claimed": False,
            "canonical_geometry_authoritative": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "note": "D1 validation-only raw three-view accumulation; fusion/persistence have not run.",
        },
        "da3_environment_probe": da3_probe.as_dict(),
        "artifacts": {
            "combined_npz": str(combined_npz.resolve()),
            "combined_ply": str(combined_ply.resolve()),
        },
    }
    summary_path = _write_json(output_dir / "d1_summary.json", summary)

    print("\nD1 three-view canonical accumulation: PASS")
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

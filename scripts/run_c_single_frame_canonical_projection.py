#!/usr/bin/env python3
"""Phase C real single-frame canonical XYZRGB projection from validated B2 evidence.

This script consumes the already-validated B2 output directory instead of
rerunning PyCOLMAP and DA3. It reconstructs the selected SfM camera contract,
reuses the exact B2 depth alignment, projects the selected RGB-D keyframe into
the offline-SfM reconstruction frame, and writes a validation NPZ/PLY plus a
summary.

The output is validation-only single-frame geometry. It is not multi-view
fused, persistent canonical scene authority, physical metric scale,
measurement authority, or certified metrology.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.contracts import ImageSize  # noqa: E402
from core.reconstruction import (  # noqa: E402
    DenseDepthAlignment,
    DenseDepthAlignmentMode,
    SfmCameraEstimate,
    SfmCameraModelEstimate,
    SfmDenseDepthAlignmentEvidence,
    SfmDenseDepthAlignmentRejectionCode,
    SfmDenseDepthAlignmentStatus,
    SfmScaleStatus,
    SfmWorldAlignmentStatus,
    project_sfm_keyframe_dense_rgb,
)
from core.reconstruction.da3 import read_da3_result_manifest, validate_da3_npz_result  # noqa: E402
from core.reconstruction.errors import ReconstructionValidationError  # noqa: E402
from scripts.run_b2_sfm_dense_real_alignment import _adapt_da3_depth_to_sfm_pixel_space  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--b2-dir",
        type=Path,
        default=Path("/tmp/eia_b2_real_alignment"),
        help="Validated B2 output directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/eia_c_single_frame_projection"),
    )
    parser.add_argument(
        "--pixel-stride",
        type=int,
        default=1,
        help="Regular pixel-grid stride before back-projection. Default keeps every pixel.",
    )
    args = parser.parse_args()
    if args.pixel_stride <= 0:
        parser.error("--pixel-stride must be positive.")
    return args


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError(f"Unable to read required Phase C JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"Required Phase C JSON root must be an object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return path


def _image_size(payload: dict[str, Any]) -> ImageSize:
    return ImageSize(width=int(payload["width"]), height=int(payload["height"]))


def _camera_estimate_from_payload(payload: dict[str, Any]) -> SfmCameraEstimate:
    return SfmCameraEstimate(
        estimate_id=str(payload["estimate_id"]),
        camera_id=str(payload["camera_id"]),
        keyframe_id=str(payload["keyframe_id"]),
        frame_id=str(payload["frame_id"]),
        backend_image_id=(
            None if payload.get("backend_image_id") is None else int(payload["backend_image_id"])
        ),
        T_reconstruction_from_camera=np.asarray(payload["T_reconstruction_from_camera"], dtype=np.float64),
    )


def _camera_model_from_payload(payload: dict[str, Any]) -> SfmCameraModelEstimate:
    return SfmCameraModelEstimate(
        camera_id=str(payload["camera_id"]),
        backend_camera_id=int(payload["backend_camera_id"]),
        backend_model_name=str(payload["backend_model_name"]),
        image_size=_image_size(payload["image_size"]),
        parameters=tuple(float(value) for value in payload["parameters"]),
    )


def _alignment_evidence_from_payload(payload: dict[str, Any]) -> SfmDenseDepthAlignmentEvidence:
    alignment_payload = payload.get("alignment")
    alignment = None
    if isinstance(alignment_payload, dict):
        alignment = DenseDepthAlignment(
            mode=DenseDepthAlignmentMode(str(alignment_payload["mode"])),
            scale_to_reconstruction_units=float(alignment_payload["scale_to_reconstruction_units"]),
            relative_scale_uncertainty=(
                None
                if alignment_payload.get("relative_scale_uncertainty") is None
                else float(alignment_payload["relative_scale_uncertainty"])
            ),
            scale_reference_profile_id=str(alignment_payload.get("scale_reference_profile_id", "")),
            provenance=str(alignment_payload["provenance"]),
        )
    rejection = payload.get("rejection_code")
    return SfmDenseDepthAlignmentEvidence(
        session_id=str(payload["session_id"]),
        keyframe_id=str(payload["keyframe_id"]),
        frame_id=str(payload.get("frame_id", "")),
        camera_id=str(payload.get("camera_id", "")),
        camera_model_name=str(payload.get("camera_model_name", "")),
        depth_provenance=str(payload["depth_provenance"]),
        status=SfmDenseDepthAlignmentStatus(str(payload["status"])),
        sfm_scale_status=SfmScaleStatus(str(payload["sfm_scale_status"])),
        sfm_world_alignment_status=SfmWorldAlignmentStatus(str(payload["sfm_world_alignment_status"])),
        depth_image_size=_image_size(payload["depth_image_size"]),
        minimum_correspondences=int(payload["minimum_correspondences"]),
        maximum_relative_residual=float(payload["maximum_relative_residual"]),
        candidate_observations=int(payload.get("candidate_observations", 0)),
        behind_camera_observations=int(payload.get("behind_camera_observations", 0)),
        out_of_bounds_observations=int(payload.get("out_of_bounds_observations", 0)),
        invalid_depth_observations=int(payload.get("invalid_depth_observations", 0)),
        valid_correspondences=int(payload.get("valid_correspondences", 0)),
        inlier_correspondences=int(payload.get("inlier_correspondences", 0)),
        robust_relative_residual_limit=(
            None
            if payload.get("robust_relative_residual_limit") is None
            else float(payload["robust_relative_residual_limit"])
        ),
        median_relative_residual=(
            None
            if payload.get("median_relative_residual") is None
            else float(payload["median_relative_residual"])
        ),
        p90_relative_residual=(
            None
            if payload.get("p90_relative_residual") is None
            else float(payload["p90_relative_residual"])
        ),
        alignment=alignment,
        rejection_code=(
            None if rejection is None else SfmDenseDepthAlignmentRejectionCode(str(rejection))
        ),
        message=str(payload["message"]),
        schema_version=int(payload.get("schema_version", 1)),
    )


def _visible_sparse_landmarks(
    sfm_payload: dict[str, Any], keyframe_id: str
) -> tuple[np.ndarray, np.ndarray]:
    points: list[list[float]] = []
    colors: list[list[int]] = []
    for landmark in sfm_payload.get("landmarks", []):
        observations = landmark.get("observations", []) if isinstance(landmark, dict) else []
        if not any(
            isinstance(item, dict) and str(item.get("keyframe_id", "")) == keyframe_id
            for item in observations
        ):
            continue
        position = landmark.get("position_reconstruction")
        if not isinstance(position, list) or len(position) != 3:
            continue
        points.append([float(value) for value in position])
        color = landmark.get("color_rgb")
        if isinstance(color, list) and len(color) == 3:
            colors.append([int(np.clip(int(value), 0, 255)) for value in color])
        else:
            colors.append([70, 255, 120])
    if not points:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.uint8)
    return np.asarray(points, dtype=np.float64), np.asarray(colors, dtype=np.uint8)


def _write_binary_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> Path:
    if points.shape != colors.shape or points.ndim != 2 or points.shape[1:] != (3,):
        raise ReconstructionValidationError("PLY points/colors must be matching Nx3 arrays.")
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment EIA TECH Phase C single-frame canonical validation cloud\n"
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
        vertices.tofile(handle)
    return path


def main() -> int:
    args = _parse_args()
    b2_dir = args.b2_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    b2_summary = _read_json(b2_dir / "b2_summary.json")
    if not bool(b2_summary.get("pass")) or not bool(b2_summary.get("real_sfm_executed")) or not bool(
        b2_summary.get("real_dense_depth_executed")
    ):
        raise ReconstructionValidationError("Phase C requires a fully passed real B2 result.")
    authority = b2_summary.get("authority") if isinstance(b2_summary.get("authority"), dict) else {}
    if authority.get("sfm_scale_status") != "arbitrary" or authority.get("metric_scale_claimed") is not False:
        raise ReconstructionValidationError(
            "Phase C expects the validated B2 monocular SfM result to remain explicit arbitrary scale."
        )

    sfm_payload = _read_json(b2_dir / "sfm_result.json")
    evidence_payload = _read_json(b2_dir / "b2_alignment_evidence.json")
    evidence = _alignment_evidence_from_payload(evidence_payload)
    if not evidence.is_accepted or evidence.alignment is None:
        raise ReconstructionValidationError("Phase C requires accepted B2 alignment evidence.")

    selected_keyframe_id = str(b2_summary["sfm"]["selected_keyframe_id"])
    cameras = [
        item for item in sfm_payload.get("cameras", [])
        if isinstance(item, dict) and str(item.get("keyframe_id", "")) == selected_keyframe_id
    ]
    if len(cameras) != 1:
        raise ReconstructionValidationError("Phase C could not resolve exactly one selected SfM camera pose.")
    camera_estimate = _camera_estimate_from_payload(cameras[0])

    camera_models = [
        item for item in sfm_payload.get("camera_models", [])
        if isinstance(item, dict) and str(item.get("camera_id", "")) == camera_estimate.camera_id
    ]
    if len(camera_models) != 1:
        raise ReconstructionValidationError("Phase C could not resolve exactly one selected SfM camera model.")
    camera_model = _camera_model_from_payload(camera_models[0])

    workspace_images = sfm_payload.get("workspace", {}).get("images", [])
    selected_images = [
        item for item in workspace_images
        if isinstance(item, dict) and str(item.get("keyframe_id", "")) == selected_keyframe_id
    ]
    if len(selected_images) != 1:
        raise ReconstructionValidationError("Phase C could not resolve selected SfM workspace timestamp evidence.")
    timestamp_ns = int(selected_images[0]["timestamp_ns"])

    rgb_path = Path(str(b2_summary["sfm"]["selected_workspace_image"])).expanduser().resolve()
    bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ReconstructionValidationError(f"Phase C could not decode selected RGB keyframe: {rgb_path}")
    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)

    da3_manifest_path = Path(str(b2_summary["artifacts"]["da3_result_manifest"])).expanduser().resolve()
    da3_manifest = read_da3_result_manifest(da3_manifest_path)
    da3_arrays = validate_da3_npz_result(da3_manifest)
    if da3_manifest.frame_count != 1 or da3_arrays["depth"].shape[0] != 1:
        raise ReconstructionValidationError("Phase C expects the B2 DA3 artifact to contain one frame.")
    depth_sfm_grid, pixel_adapter = _adapt_da3_depth_to_sfm_pixel_space(
        da3_arrays["depth"][0],
        sfm_width=camera_model.image_size.width,
        sfm_height=camera_model.image_size.height,
    )
    b2_adapter = b2_summary.get("dense_depth", {}).get("pixel_space_adapter", {})
    if isinstance(b2_adapter, dict) and b2_adapter.get("mode") != pixel_adapter.get("mode"):
        raise ReconstructionValidationError("Phase C pixel-space adaptation disagrees with validated B2 evidence.")

    result = project_sfm_keyframe_dense_rgb(
        camera_estimate,
        camera_model,
        timestamp_ns=timestamp_ns,
        rgb=rgb,
        dense_depth_source_units=depth_sfm_grid,
        alignment_evidence=evidence,
        pixel_stride=int(args.pixel_stride),
        reconstruction_frame_id="offline-sfm-reconstruction",
    )

    sparse_points, sparse_colors = _visible_sparse_landmarks(sfm_payload, selected_keyframe_id)
    cloud_npz = output_dir / "c_canonical_single_frame_cloud.npz"
    np.savez_compressed(
        cloud_npz,
        points_reconstruction=result.points_reconstruction,
        colors_rgb=result.colors_rgb,
        visible_sparse_landmarks_reconstruction=sparse_points,
        visible_sparse_landmark_colors_rgb=sparse_colors,
        T_reconstruction_from_camera=camera_estimate.T_reconstruction_from_camera,
    )
    cloud_ply = _write_binary_ply(
        output_dir / "c_canonical_single_frame_cloud.ply",
        result.points_reconstruction,
        result.colors_rgb,
    )

    result_json = _write_json(output_dir / "c_projection_evidence.json", result.as_dict())
    summary = {
        "phase": "C",
        "pass": True,
        "source_b2_dir": str(b2_dir),
        "source_b2_pass": True,
        "session_id": result.session_id,
        "keyframe_id": result.keyframe_id,
        "frame_id": result.frame_id,
        "camera_id": result.camera_id,
        "camera_model_name": result.camera_model_name,
        "pixel_stride": result.pixel_stride,
        "point_count": int(result.points_reconstruction.shape[0]),
        "sampled_pixels": result.projection.sampled_pixels,
        "visible_sparse_landmarks": int(sparse_points.shape[0]),
        "bounds_min_reconstruction": result.points_reconstruction.min(axis=0).tolist(),
        "bounds_max_reconstruction": result.points_reconstruction.max(axis=0).tolist(),
        "depth_alignment": evidence.alignment.as_dict(),
        "pixel_space_adapter": pixel_adapter,
        "authority": {
            "sfm_scale_status": result.sfm_scale_status,
            "sfm_world_alignment_status": result.sfm_world_alignment_status,
            "metric_scale_claimed": False,
            "canonical_geometry_authoritative": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "note": "Phase C single-frame validation geometry only; multi-view fusion/persistence have not run.",
        },
        "artifacts": {
            "cloud_npz": str(cloud_npz.resolve()),
            "cloud_ply": str(cloud_ply.resolve()),
            "projection_evidence_json": str(result_json.resolve()),
        },
    }
    summary_path = _write_json(output_dir / "c_summary.json", summary)

    print("C single-frame canonical XYZRGB projection: PASS")
    print(
        f"keyframe={result.keyframe_id} camera={result.camera_id} model={result.camera_model_name} "
        f"points={result.points_reconstruction.shape[0]} sampled_pixels={result.projection.sampled_pixels}"
    )
    print(
        f"scale_to_reconstruction_units={evidence.alignment.scale_to_reconstruction_units:.9g} "
        f"sfm_scale_status={result.sfm_scale_status} metric_scale_claimed=False"
    )
    print(
        f"bounds_min={np.array2string(result.points_reconstruction.min(axis=0), precision=4)} "
        f"bounds_max={np.array2string(result.points_reconstruction.max(axis=0), precision=4)}"
    )
    print(f"visible_sparse_landmarks={sparse_points.shape[0]}")
    print("canonical_geometry_authoritative=False measurement_authoritative=False certified_metrology=False")
    print(f"cloud_npz={cloud_npz}")
    print(f"cloud_ply={cloud_ply}")
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Isolated PyCOLMAP reader/validator for I4.1.9A A1.

This worker only loads sparse reconstructions, exports one sparse model as text,
and validates a derived sparse model.  It contains no SfM, BA, undistortion,
PatchMatch, or Stereo Fusion calls.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np


def read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return payload


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def matrix44_from_cam_from_world(image) -> np.ndarray:
    matrix = np.asarray(image.cam_from_world().matrix(), dtype=np.float64)
    if matrix.shape != (3, 4):
        raise RuntimeError(f"Unexpected PyCOLMAP rigid matrix shape: {matrix.shape}")
    result = np.eye(4, dtype=np.float64)
    result[:3, :] = matrix
    return result


def internal_pose_from_image(image) -> np.ndarray:
    return np.linalg.inv(matrix44_from_cam_from_world(image))


def camera_payload(camera_id: int, camera) -> dict:
    return {
        "camera_id": int(camera_id),
        "model_name": str(camera.model_name),
        "width": int(camera.width),
        "height": int(camera.height),
        "params": [float(v) for v in camera.params],
    }


def tracks(reconstruction) -> dict[int, set[tuple[int, int]]]:
    result: dict[int, set[tuple[int, int]]] = {}
    for point_id in reconstruction.point3D_ids():
        point = reconstruction.point3D(int(point_id))
        result[int(point_id)] = {
            (int(element.image_id), int(element.point2D_idx)) for element in point.track.elements
        }
    return result


def point_positions(reconstruction) -> dict[int, np.ndarray]:
    return {
        int(point_id): np.asarray(reconstruction.point3D(int(point_id)).xyz, dtype=np.float64)
        for point_id in reconstruction.point3D_ids()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()

    request = read_json(args.request.resolve())
    result_path = args.result.resolve()
    result: dict = {
        "schema_version": int(request.get("schema_version", 0)),
        "status": "error",
        "error": "",
        "sfm_rerun": False,
        "bundle_adjustment_rerun": False,
        "image_undistortion_started": False,
        "patch_match_started": False,
        "stereo_fusion_started": False,
        "heavy_mvs_started": False,
    }
    try:
        import pycolmap

        result["pycolmap_version"] = str(getattr(pycolmap, "__version__", ""))
        action = str(request.get("action", ""))
        native_path = Path(str(request["native_model"])).expanduser().resolve()
        if not native_path.is_dir():
            raise RuntimeError(f"Native sparse model is missing: {native_path}")
        native = pycolmap.Reconstruction(native_path)

        if action == "export_native_text":
            expected = int(request.get("expected_registered_images", -1))
            if int(native.num_reg_images()) != expected:
                raise RuntimeError(
                    f"Native model registered-image count changed: {native.num_reg_images()} != {expected}"
                )
            output = Path(str(request["output_text_model"])).expanduser().resolve()
            if output.exists():
                if any(output.iterdir()):
                    raise RuntimeError(f"Native text export directory must be empty or absent: {output}")
            else:
                output.mkdir(parents=True)
            native.write_text(output)
            # Keep the derived bridge intentionally on the classic cameras/images/points3D
            # representation. Newer rig/frame files are not required for the legacy-compatible
            # single-camera model and are not copied into the derived sparse directory.
            result.update(
                {
                    "status": "ok",
                    "action": action,
                    "registered_images": int(native.num_reg_images()),
                    "cameras": int(native.num_cameras()),
                    "points3D": int(native.num_points3D()),
                    "text_model": str(output),
                }
            )
        elif action == "validate_derived_model":
            derived_path = Path(str(request["derived_model"])).expanduser().resolve()
            manifest = read_json(Path(str(request["authority_manifest"])).expanduser().resolve())
            derived = pycolmap.Reconstruction(derived_path)
            expected_native = int(request["expected_native_registered_images"])
            expected_derived = int(request["expected_derived_registered_images"])
            checks: dict[str, bool] = {}
            checks["native_registered_images"] = int(native.num_reg_images()) == expected_native
            checks["derived_registered_images"] = int(derived.num_reg_images()) == expected_derived
            checks["camera_count_unchanged"] = int(derived.num_cameras()) == int(native.num_cameras())
            checks["point_count_unchanged"] = int(derived.num_points3D()) == int(native.num_points3D())

            native_cameras = {int(cid): camera_payload(int(cid), camera) for cid, camera in native.cameras.items()}
            derived_cameras = {int(cid): camera_payload(int(cid), camera) for cid, camera in derived.cameras.items()}
            checks["camera_intrinsics_exact"] = native_cameras == derived_cameras

            native_positions = point_positions(native)
            derived_positions = point_positions(derived)
            checks["point_ids_exact"] = set(native_positions) == set(derived_positions)
            checks["point_positions_exact"] = checks["point_ids_exact"] and all(
                np.array_equal(native_positions[pid], derived_positions[pid]) for pid in native_positions
            )

            native_tracks = tracks(native)
            derived_tracks = tracks(derived)
            checks["native_tracks_preserved"] = all(
                native_tracks[pid].issubset(derived_tracks.get(pid, set())) for pid in native_tracks
            )

            native_image_ids = {int(v) for v in native.reg_image_ids()}
            derived_image_ids = {int(v) for v in derived.reg_image_ids()}
            native_pose_ok = True
            native_name_ok = True
            for image_id in native_image_ids:
                if image_id not in derived_image_ids:
                    native_pose_ok = False
                    native_name_ok = False
                    break
                if str(native.image(image_id).name) != str(derived.image(image_id).name):
                    native_name_ok = False
                if not np.allclose(
                    matrix44_from_cam_from_world(native.image(image_id)),
                    matrix44_from_cam_from_world(derived.image(image_id)),
                    rtol=0.0,
                    atol=1e-12,
                ):
                    native_pose_ok = False
                    break
            checks["native_40_poses_preserved"] = native_pose_ok
            checks["native_40_image_names_preserved"] = native_name_ok

            views = manifest.get("views", [])
            extra_pose_ok = True
            extra_camera_ok = True
            extra_track_ok = True
            derived_names = {str(derived.image(int(v)).name): int(v) for v in derived.reg_image_ids()}
            for view in views:
                if view.get("provenance") != "localized_extra_observation":
                    continue
                name = str(view["image_name"])
                image_id = int(view["derived_image_id"])
                if derived_names.get(name) != image_id:
                    extra_pose_ok = False
                    extra_track_ok = False
                    break
                image = derived.image(image_id)
                if int(image.camera_id) != int(view["camera_id"]):
                    extra_camera_ok = False
                expected_pose = np.asarray(view["T_reconstruction_from_camera"], dtype=np.float64)
                actual_pose = internal_pose_from_image(image)
                if not np.allclose(expected_pose, actual_pose, rtol=0.0, atol=1e-9):
                    extra_pose_ok = False
                point_ids = [int(v) for v in view.get("matched_point3d_ids", [])]
                for point2d_idx, point_id in enumerate(point_ids):
                    if (image_id, point2d_idx) not in derived_tracks.get(point_id, set()):
                        extra_track_ok = False
                        break
            checks["localized_28_poses_preserved"] = extra_pose_ok
            checks["localized_28_camera_id_preserved"] = extra_camera_ok
            checks["localized_observations_attached"] = extra_track_ok
            checks["no_new_3d_points"] = checks["point_count_unchanged"] and checks["point_ids_exact"]
            checks["all_68_available"] = checks["derived_registered_images"]

            validation_pass = all(checks.values())
            result.update(
                {
                    "status": "ok",
                    "action": action,
                    "validation_pass": validation_pass,
                    "checks": checks,
                    "native_registered_images": int(native.num_reg_images()),
                    "derived_registered_images": int(derived.num_reg_images()),
                    "native_points3D": int(native.num_points3D()),
                    "derived_points3D": int(derived.num_points3D()),
                }
            )
        else:
            raise RuntimeError(f"Unsupported A1 worker action: {action!r}")
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        write_json(result_path, result)
        print(result["error"], file=sys.stderr)
        return 2

    write_json(result_path, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

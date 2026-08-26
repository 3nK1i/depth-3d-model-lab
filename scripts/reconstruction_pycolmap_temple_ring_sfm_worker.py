#!/usr/bin/env python3
"""Isolated PyCOLMAP worker for I4.2B TempleRing authoritative sparse SfM."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import sys
import time
import traceback
from typing import Any, Mapping

import pycolmap

SCHEMA_VERSION = 1
EXPECTED_VIEWS = 47
EXPECTED_NAMES = tuple(f"templeR{index:04d}.png" for index in range(1, 48))
FORBIDDEN_TOKENS = (
    "R_camera_from_world",
    "t_camera_from_world",
    "camera_center_world",
    "ground_truth_validation_manifest",
    "templeR_ang.txt",
)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Worker request must be a JSON object.")
    return payload


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def matrix34(transform: Any) -> list[list[float]]:
    matrix = transform.matrix()
    return [[float(matrix[r, c]) for c in range(4)] for r in range(3)]


def select_best_model(models: Any) -> tuple[int, Any]:
    if not models:
        raise RuntimeError("PyCOLMAP incremental mapping produced no sparse model.")
    items = [(int(model_id), model) for model_id, model in models.items()]
    items.sort(key=lambda item: (int(item[1].num_reg_images()), int(item[1].num_points3D()), -item[0]), reverse=True)
    return items[0]


def run(request: Mapping[str, Any]) -> dict[str, Any]:
    if int(request.get("schema_version", 0)) != SCHEMA_VERSION or request.get("phase") != "I4.2B":
        raise ValueError("Unsupported I4.2B worker request.")
    serialized = json.dumps(request, sort_keys=True)
    for token in FORBIDDEN_TOKENS:
        if token in serialized:
            raise ValueError(f"Held-out TempleRing ground truth leaked into SfM worker request: {token}")

    if str(pycolmap.__version__) != "4.1.1":
        raise RuntimeError(f"I4.2B requires PyCOLMAP 4.1.1, found {pycolmap.__version__}")

    workspace = Path(str(request["workspace"])).expanduser().resolve()
    images_dir = Path(str(request["images_directory"])).expanduser().resolve()
    database_path = Path(str(request["database_path"])).expanduser().resolve()
    mapping_output = Path(str(request["mapping_output"])).expanduser().resolve()
    authoritative_sparse = Path(str(request["authoritative_sparse"])).expanduser().resolve()
    if not workspace.is_dir() or images_dir.parent != workspace:
        raise RuntimeError("I4.2B worker paths must remain inside the immutable workspace.")
    for path in (database_path, mapping_output, authoritative_sparse):
        try:
            path.relative_to(workspace)
        except ValueError as exc:
            raise RuntimeError(f"I4.2B path escapes workspace: {path}") from exc

    names = tuple(str(name) for name in request.get("image_names", []))
    if names != EXPECTED_NAMES:
        raise RuntimeError("I4.2B requires the exact canonical 47-view TempleRing order.")
    if any(not (images_dir / name).is_file() for name in names):
        raise RuntimeError("I4.2B staged image set is incomplete.")

    config = request.get("config")
    intrinsics = request.get("shared_intrinsics")
    if not isinstance(config, Mapping) or not isinstance(intrinsics, Mapping):
        raise RuntimeError("I4.2B configuration/intrinsics payload is missing.")
    if config.get("matching_strategy") != "exhaustive" or config.get("camera_model") != "PINHOLE":
        raise RuntimeError("I4.2B fixed exhaustive/PINHOLE configuration changed.")
    if config.get("intrinsics_fixed") is not True or config.get("multiple_models") is not False:
        raise RuntimeError("I4.2B fixed-intrinsics/single-model contract changed.")
    use_gpu = bool(config.get("use_gpu", True))
    if use_gpu and int(pycolmap.get_num_cuda_devices()) < 1:
        raise RuntimeError("I4.2B GPU SfM requested but no CUDA device is visible.")

    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    expected_params = [fx, fy, cx, cy]
    if not all(math.isfinite(value) and value > 0.0 for value in (fx, fy)) or not all(math.isfinite(value) for value in (cx, cy)):
        raise RuntimeError("I4.2B shared intrinsics are invalid.")

    database_path.unlink(missing_ok=True)
    if mapping_output.exists():
        shutil.rmtree(mapping_output)
    if authoritative_sparse.exists():
        shutil.rmtree(authoritative_sparse)
    mapping_output.mkdir(parents=True, exist_ok=False)

    device = pycolmap.Device.cuda if use_gpu else pycolmap.Device.cpu
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.use_gpu = use_gpu
    extraction.gpu_index = "0" if use_gpu else "-1"
    extraction.max_image_size = int(config["maximum_image_size"])
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "PINHOLE"
    reader.camera_params = ",".join(f"{value:.17g}" for value in expected_params)

    stage_seconds: dict[str, float] = {}
    started = time.monotonic()
    pycolmap.extract_features(
        database_path=database_path,
        image_path=images_dir,
        image_names=list(names),
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader,
        extraction_options=extraction,
        device=device,
    )
    stage_seconds["feature_extraction"] = time.monotonic() - started

    matching = pycolmap.FeatureMatchingOptions()
    matching.use_gpu = use_gpu
    matching.gpu_index = "0" if use_gpu else "-1"
    started = time.monotonic()
    pycolmap.match_exhaustive(
        database_path=database_path,
        matching_options=matching,
        device=device,
    )
    stage_seconds["exhaustive_matching"] = time.monotonic() - started

    options = pycolmap.IncrementalPipelineOptions()
    options.multiple_models = False
    options.max_num_models = 1
    options.min_model_size = 3
    options.image_names = list(names)
    options.ba_use_gpu = False
    options.ba_refine_focal_length = False
    options.ba_refine_principal_point = False
    options.ba_refine_extra_params = False
    options.mapper.abs_pose_refine_focal_length = False
    options.mapper.abs_pose_refine_extra_params = False
    if hasattr(options, "random_seed"):
        options.random_seed = int(config.get("random_seed", 0))
    if hasattr(options.mapper, "random_seed"):
        options.mapper.random_seed = int(config.get("random_seed", 0))
    if hasattr(options.triangulation, "random_seed"):
        options.triangulation.random_seed = int(config.get("random_seed", 0))

    started = time.monotonic()
    models = pycolmap.incremental_mapping(
        database_path=database_path,
        image_path=images_dir,
        output_path=mapping_output,
        options=options,
    )
    stage_seconds["incremental_mapping_and_ba"] = time.monotonic() - started
    selected_id, reconstruction = select_best_model(models)

    authoritative_sparse.mkdir(parents=True, exist_ok=False)
    reconstruction.write(authoritative_sparse)

    camera_ids = sorted({int(reconstruction.image(int(image_id)).camera_id) for image_id in reconstruction.reg_image_ids()})
    if len(camera_ids) != 1:
        raise RuntimeError(f"I4.2B expected one shared camera, found {len(camera_ids)}")
    camera = reconstruction.camera(camera_ids[0])
    params = [float(value) for value in camera.params]
    if str(camera.model_name) != "PINHOLE" or len(params) != 4:
        raise RuntimeError(f"I4.2B camera model changed: {camera.model_name} params={params}")
    if any(abs(params[index] - expected_params[index]) > 1e-9 for index in range(4)):
        raise RuntimeError(f"I4.2B fixed intrinsics changed: expected={expected_params} actual={params}")

    registered = []
    for image_id in sorted(int(value) for value in reconstruction.reg_image_ids()):
        image = reconstruction.image(image_id)
        registered.append({
            "image_id": image_id,
            "name": str(image.name),
            "camera_id": int(image.camera_id),
            "camera_from_reconstruction": matrix34(image.cam_from_world()),
        })
    registered_names = sorted(item["name"] for item in registered)

    return {
        "schema_version": SCHEMA_VERSION,
        "phase": "I4.2B",
        "status": "ok",
        "pycolmap_version": str(pycolmap.__version__),
        "python_executable": sys.executable,
        "selected_model_id": selected_id,
        "stage_seconds": stage_seconds,
        "camera": {
            "camera_count": 1,
            "camera_id": camera_ids[0],
            "model": str(camera.model_name),
            "width": int(camera.width),
            "height": int(camera.height),
            "params": params,
            "intrinsics_fixed": True,
        },
        "registered_images": registered,
        "registered_image_names": registered_names,
        "statistics": {
            "registered_images": int(reconstruction.num_reg_images()),
            "landmarks": int(reconstruction.num_points3D()),
            "observations": int(reconstruction.compute_num_observations()),
            "mean_track_length": float(reconstruction.compute_mean_track_length()),
            "mean_reprojection_error_px": float(reconstruction.compute_mean_reprojection_error()),
        },
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


def main() -> int:
    parser = argparse.ArgumentParser(description="I4.2B TempleRing authoritative PyCOLMAP SfM worker")
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    request_path = Path(args.request).expanduser().resolve()
    result_path = Path(args.result).expanduser().resolve()
    started = time.monotonic()
    try:
        payload = run(read_json(request_path))
        payload["worker_elapsed_s"] = max(1e-9, time.monotonic() - started)
        write_json(result_path, payload)
        return 0
    except Exception as exc:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "phase": "I4.2B",
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "worker_elapsed_s": max(1e-9, time.monotonic() - started),
        }
        try:
            write_json(result_path, payload)
        except Exception:
            pass
        print(payload["traceback"], file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

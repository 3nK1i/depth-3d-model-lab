#!/usr/bin/env python3
"""Standalone PyCOLMAP 4.1.1 worker for Phase 8.3D.

This script is intentionally dependency-isolated. It runs under the external
reconstruction interpreter and imports neither the EIA application package nor
PyQt/OpenCV/Torch. Only JSON-safe values cross the process boundary.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import sys
import time
import traceback

SFM_WORKER_SCHEMA_VERSION = 1
SFM_WORKER_CHECKPOINT_SCHEMA_VERSION = 1
EVENT_PREFIX = "EIA_SFM_EVENT "
_RESUME_STAGES = {"none": 0, "features": 1, "matching": 2, "complete": 3}


def emit(
    event_type: str,
    stage: str,
    message: str,
    progress: float | None = None,
    **metadata,
) -> None:
    payload = {
        "schema_version": SFM_WORKER_SCHEMA_VERSION,
        "event_type": event_type,
        "stage": stage,
        "message": message,
        "progress_fraction": progress,
        "metadata": metadata,
    }
    print(EVENT_PREFIX + json.dumps(payload, sort_keys=True, allow_nan=False), flush=True)


class _MappingProgressTracker:
    """Keep multi-model PyCOLMAP mapping callbacks globally monotonic.

    ``incremental_mapping(multiple_models=True)`` may seed more than one sparse
    model.  PyCOLMAP invokes ``initial_image_pair_callback`` for every seed, so
    treating every seed as the same 0.68 progress event can move application
    progress backwards after an earlier model has already registered images.
    Only the first seed advances progress; later seeds are status-only.
    """

    def __init__(self, image_count: int) -> None:
        count = int(image_count)
        if count < 1:
            raise ValueError("Mapping progress requires at least one image.")
        self.image_count = count
        self.registered_callbacks = 0
        self.initial_pair_callbacks = 0
        self.last_progress = 0.64

    def initial_pair(self) -> tuple[str, float | None, str]:
        self.initial_pair_callbacks += 1
        if self.initial_pair_callbacks == 1:
            self.last_progress = max(self.last_progress, 0.68)
            return "progress", self.last_progress, "Initial image pair registered."
        return (
            "status",
            None,
            "Additional sparse-model initial image pair registered.",
        )

    def next_image(self) -> float:
        self.registered_callbacks += 1
        candidate = min(
            0.92,
            0.68 + 0.24 * self.registered_callbacks / max(1, self.image_count),
        )
        self.last_progress = max(self.last_progress, candidate)
        return self.last_progress


def load_object(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Worker request must contain a JSON object.")
    return payload


def atomic_write_object(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def safe_child(root: Path, relative: str) -> Path:
    relative_path = Path(str(relative))
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"Unsafe worker-relative path: {relative}")
    candidate = (root / relative_path).resolve()
    candidate.relative_to(root)
    return candidate


def matrix34(rigid3d) -> list[list[float]]:
    matrix = rigid3d.matrix()
    return [[float(matrix[row, col]) for col in range(4)] for row in range(3)]


def finite_or_none(value) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def select_best_model(models: dict) -> tuple[int, object, list[dict]]:
    """Select a sparse model deterministically by support, then residual quality."""

    summaries = []
    for model_id, reconstruction in models.items():
        mean_error = finite_or_none(reconstruction.compute_mean_reprojection_error())
        summaries.append(
            {
                "model_id": int(model_id),
                "registered_images": int(reconstruction.num_reg_images()),
                "points3D": int(reconstruction.num_points3D()),
                "observations": int(reconstruction.compute_num_observations()),
                "mean_reprojection_error_px": mean_error,
            }
        )
    if not summaries:
        raise RuntimeError("PyCOLMAP incremental mapping produced no sparse model.")
    summaries.sort(
        key=lambda item: (
            -item["registered_images"],
            -item["observations"],
            -item["points3D"],
            (
                float("inf")
                if item["mean_reprojection_error_px"] is None
                else item["mean_reprojection_error_px"]
            ),
            item["model_id"],
        )
    )
    selected_id = summaries[0]["model_id"]
    return selected_id, models[selected_id], summaries


def load_resume_checkpoint(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def resolve_resume_stage(
    checkpoint: dict | None,
    *,
    request_fingerprint: str,
    pycolmap_version: str,
    database_path: Path,
    image_count: int,
) -> str:
    """Return the last safely reusable completed stage or ``none``."""

    if checkpoint is None:
        return "none"
    try:
        if int(checkpoint.get("schema_version", 0)) != SFM_WORKER_CHECKPOINT_SCHEMA_VERSION:
            return "none"
        if str(checkpoint.get("request_fingerprint", "")) != request_fingerprint:
            return "none"
        if str(checkpoint.get("pycolmap_version", "")) != pycolmap_version:
            return "none"
        if int(checkpoint.get("image_count", -1)) != image_count:
            return "none"
        stage = str(checkpoint.get("completed_stage", "none"))
    except (TypeError, ValueError):
        return "none"
    if stage not in _RESUME_STAGES or stage == "none" or not database_path.is_file():
        return "none"
    # A complete prior run can safely reuse the feature/matching database while
    # mapping is rerun into a fresh sparse directory.
    return "matching" if stage == "complete" else stage


def write_stage_checkpoint(
    path: Path,
    *,
    request_fingerprint: str,
    pycolmap_version: str,
    image_count: int,
    completed_stage: str,
) -> None:
    if completed_stage not in {"features", "matching", "complete"}:
        raise ValueError(f"Unsupported SfM checkpoint stage: {completed_stage}")
    atomic_write_object(
        path,
        {
            "schema_version": SFM_WORKER_CHECKPOINT_SCHEMA_VERSION,
            "request_fingerprint": request_fingerprint,
            "pycolmap_version": pycolmap_version,
            "image_count": image_count,
            "completed_stage": completed_stage,
            "updated_unix_s": time.time(),
        },
    )


def validate_pycolmap_runtime_api(pycolmap) -> None:
    required = {
        pycolmap.Reconstruction: (
            "reg_image_ids",
            "point3D_ids",
            "image",
            "point3D",
            "camera",
            "compute_num_observations",
            "compute_mean_reprojection_error",
        ),
        pycolmap.Image: ("cam_from_world", "point2D"),
        pycolmap.Camera: ("model_name", "params", "width", "height"),
        pycolmap.Rigid3d: ("matrix",),
    }
    missing = []
    for owner, names in required.items():
        for name in names:
            if not hasattr(owner, name):
                missing.append(f"{owner.__name__}.{name}")
    if missing:
        raise RuntimeError(
            "PyCOLMAP runtime API is incompatible with Phase 8.3D: "
            + ", ".join(sorted(missing))
        )


def run(request: dict) -> dict:
    import pycolmap

    validate_pycolmap_runtime_api(pycolmap)
    if int(request.get("schema_version", 0)) != SFM_WORKER_SCHEMA_VERSION:
        raise ValueError("Unsupported worker request schema.")
    session_id = str(request["session_id"])
    root = Path(str(request["workspace_root"])).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Workspace does not exist: {root}")
    image_path = safe_child(root, request["images_directory"])
    database_path = safe_child(root, request["database_path"])
    sparse_path = safe_child(root, request["sparse_directory"])
    checkpoints_path = safe_child(root, request["checkpoints_directory"])
    checkpoint_path = checkpoints_path / "pycolmap_stage_checkpoint.json"
    image_names = [str(name) for name in request["image_names"]]
    if len(image_names) < 3 or len(image_names) != len(set(image_names)):
        raise ValueError("Worker requires at least three unique image names.")
    for name in image_names:
        image_file = image_path / name
        if image_file.parent.resolve() != image_path.resolve() or not image_file.is_file():
            raise FileNotFoundError(f"Materialized SfM image is missing: {image_file}")

    request_fingerprint = str(request.get("request_fingerprint", "")).strip().lower()
    if len(request_fingerprint) != 64 or any(
        character not in "0123456789abcdef" for character in request_fingerprint
    ):
        raise ValueError("request_fingerprint must be a SHA-256 digest.")

    configuration = request["configuration"]
    if configuration.get("feature_strategy") != "colmap_sift":
        raise ValueError("Phase 8.3D worker only supports COLMAP SIFT.")
    if configuration.get("camera_mode") != "single":
        raise ValueError("Phase 8.3D worker currently requires one shared camera model.")
    camera_model = str(configuration.get("camera_model", "SIMPLE_RADIAL"))
    if not camera_model:
        raise ValueError("camera_model must not be empty.")
    matching_strategy = str(configuration["matching_strategy"])
    if matching_strategy not in {"sequential", "exhaustive"}:
        raise ValueError(f"Unsupported matching strategy: {matching_strategy}")
    resume_policy = str(configuration.get("resume_policy", "disabled"))
    if resume_policy not in {"disabled", "reuse_completed_stages"}:
        raise ValueError(f"Unsupported resume policy: {resume_policy}")
    use_gpu = bool(configuration.get("use_gpu", True))
    if use_gpu and int(pycolmap.get_num_cuda_devices()) < 1:
        raise RuntimeError("GPU SfM requested but PyCOLMAP reports no CUDA devices.")

    checkpoints_path.mkdir(parents=True, exist_ok=True)
    resume_from_stage = "none"
    if resume_policy == "reuse_completed_stages":
        resume_from_stage = resolve_resume_stage(
            load_resume_checkpoint(checkpoint_path),
            request_fingerprint=request_fingerprint,
            pycolmap_version=str(pycolmap.__version__),
            database_path=database_path,
            image_count=len(image_names),
        )

    if resume_from_stage == "none":
        database_path.unlink(missing_ok=True)
        checkpoint_path.unlink(missing_ok=True)
    else:
        emit(
            "status",
            "resume",
            f"Reusing completed SfM {resume_from_stage} checkpoint.",
            0.08,
            resume_from_stage=resume_from_stage,
        )

    if sparse_path.exists():
        shutil.rmtree(sparse_path)
    sparse_path.mkdir(parents=True, exist_ok=True)

    device = pycolmap.Device.cuda if use_gpu else pycolmap.Device.cpu
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.use_gpu = use_gpu
    extraction.gpu_index = "0" if use_gpu else "-1"
    extraction.max_image_size = int(configuration["maximum_image_size"])
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = camera_model

    if _RESUME_STAGES[resume_from_stage] < _RESUME_STAGES["features"]:
        emit("progress", "feature_extraction", "Extracting COLMAP SIFT features.", 0.10)
        pycolmap.extract_features(
            database_path=database_path,
            image_path=image_path,
            image_names=image_names,
            camera_mode=pycolmap.CameraMode.SINGLE,
            reader_options=reader,
            extraction_options=extraction,
            device=device,
        )
        write_stage_checkpoint(
            checkpoint_path,
            request_fingerprint=request_fingerprint,
            pycolmap_version=str(pycolmap.__version__),
            image_count=len(image_names),
            completed_stage="features",
        )
        emit("progress", "feature_extraction", "COLMAP SIFT feature extraction complete.", 0.35)

    matching = pycolmap.FeatureMatchingOptions()
    matching.use_gpu = use_gpu
    matching.gpu_index = "0" if use_gpu else "-1"
    if _RESUME_STAGES[resume_from_stage] < _RESUME_STAGES["matching"]:
        emit("progress", "matching", f"Running {matching_strategy} feature matching.", 0.40)
        if matching_strategy == "sequential":
            pairing = pycolmap.SequentialPairingOptions()
            pairing.overlap = int(configuration.get("sequential_overlap", 10))
            pairing.quadratic_overlap = True
            pairing.loop_detection = False
            pycolmap.match_sequential(
                database_path=database_path,
                matching_options=matching,
                pairing_options=pairing,
                device=device,
            )
        else:
            pycolmap.match_exhaustive(
                database_path=database_path,
                matching_options=matching,
                device=device,
            )
        write_stage_checkpoint(
            checkpoint_path,
            request_fingerprint=request_fingerprint,
            pycolmap_version=str(pycolmap.__version__),
            image_count=len(image_names),
            completed_stage="matching",
        )
        emit("progress", "matching", "Feature matching and geometric verification complete.", 0.60)
    else:
        emit(
            "progress",
            "matching",
            "Reused completed feature extraction and matching database.",
            0.60,
            resume_from_stage=resume_from_stage,
        )

    mapping = pycolmap.IncrementalPipelineOptions()
    mapping.multiple_models = True
    mapping.max_num_models = 4
    mapping.min_model_size = 3
    mapping.image_names = image_names
    mapping.ba_use_gpu = bool(configuration.get("bundle_adjustment_gpu", False))
    mapping.ba_gpu_index = "0" if mapping.ba_use_gpu else "-1"

    mapping_progress = _MappingProgressTracker(len(image_names))

    def initial_callback() -> None:
        event_type, fraction, message = mapping_progress.initial_pair()
        emit(
            event_type,
            "mapping",
            message,
            fraction,
            initial_pair_callbacks=mapping_progress.initial_pair_callbacks,
            registered_callbacks=mapping_progress.registered_callbacks,
        )

    def next_image_callback() -> None:
        fraction = mapping_progress.next_image()
        emit(
            "progress",
            "mapping",
            "Registered another image in the sparse reconstruction.",
            fraction,
            registered_callbacks=mapping_progress.registered_callbacks,
            initial_pair_callbacks=mapping_progress.initial_pair_callbacks,
        )

    emit("progress", "mapping", "Starting incremental mapping and bundle adjustment.", 0.64)
    models = pycolmap.incremental_mapping(
        database_path=database_path,
        image_path=image_path,
        output_path=sparse_path,
        options=mapping,
        initial_image_pair_callback=initial_callback,
        next_image_callback=next_image_callback,
    )
    selected_id, reconstruction, model_summaries = select_best_model(models)
    emit(
        "progress",
        "conversion",
        "Converting the selected sparse model to a dependency-neutral payload.",
        0.95,
        selected_model_id=selected_id,
        model_count=len(model_summaries),
    )

    registered_camera_ids = sorted(
        {
            int(reconstruction.image(int(image_id)).camera_id)
            for image_id in reconstruction.reg_image_ids()
        }
    )
    camera_models = []
    for camera_id in registered_camera_ids:
        camera = reconstruction.camera(camera_id)
        params = [float(value) for value in camera.params]
        if not all(math.isfinite(value) for value in params):
            raise RuntimeError(f"Camera {camera_id} contains non-finite parameters.")
        camera_models.append(
            {
                "camera_id": camera_id,
                "model_name": str(camera.model_name),
                "width": int(camera.width),
                "height": int(camera.height),
                "params": params,
            }
        )

    registered_images = []
    for image_id in sorted(int(value) for value in reconstruction.reg_image_ids()):
        image = reconstruction.image(image_id)
        registered_images.append(
            {
                "image_id": image_id,
                "camera_id": int(image.camera_id),
                "name": str(image.name),
                "camera_from_reconstruction": matrix34(image.cam_from_world()),
            }
        )

    points = []
    for point_id in sorted(int(value) for value in reconstruction.point3D_ids()):
        point = reconstruction.point3D(point_id)
        track = []
        for element in point.track.elements:
            image_id = int(element.image_id)
            point2d_idx = int(element.point2D_idx)
            image = reconstruction.image(image_id)
            xy = image.point2D(point2d_idx).xy
            track.append(
                {
                    "image_id": image_id,
                    "point2D_idx": point2d_idx,
                    "xy": [float(xy[0]), float(xy[1])],
                }
            )
        xyz = point.xyz
        color = point.color
        points.append(
            {
                "point3D_id": point_id,
                "xyz": [float(xyz[0]), float(xyz[1]), float(xyz[2])],
                "color": [int(color[0]), int(color[1]), int(color[2])],
                "error": (None if not math.isfinite(float(point.error)) else float(point.error)),
                "track": track,
            }
        )

    warnings = []
    if len(model_summaries) > 1:
        warnings.append(
            f"PyCOLMAP produced {len(model_summaries)} disconnected sparse models; selected model {selected_id}."
        )
    if resume_from_stage != "none":
        warnings.append(
            f"SfM resumed from a validated {resume_from_stage} stage checkpoint."
        )

    write_stage_checkpoint(
        checkpoint_path,
        request_fingerprint=request_fingerprint,
        pycolmap_version=str(pycolmap.__version__),
        image_count=len(image_names),
        completed_stage="complete",
    )
    emit("progress", "complete", "PyCOLMAP sparse reconstruction complete.", 0.99)
    return {
        "schema_version": SFM_WORKER_SCHEMA_VERSION,
        "status": "ok",
        "session_id": session_id,
        "request_fingerprint": request_fingerprint,
        "resume_from_stage": resume_from_stage,
        "backend": {
            "backend_id": "pycolmap-offline-sfm",
            "backend_version": "1",
            "pycolmap_version": str(pycolmap.__version__),
            "python_executable": sys.executable,
            "cuda_devices": int(pycolmap.get_num_cuda_devices()),
            "use_gpu": use_gpu,
        },
        "resolved_configuration": {
            "profile": str(request.get("profile", "")),
            "matching_strategy": matching_strategy,
            "feature_strategy": str(configuration.get("feature_strategy")),
            "maximum_image_size": int(configuration["maximum_image_size"]),
            "sequential_overlap": int(configuration.get("sequential_overlap", 10)),
            "camera_mode": str(configuration.get("camera_mode")),
            "camera_model": camera_model,
            "bundle_adjustment_gpu": bool(configuration.get("bundle_adjustment_gpu", False)),
            "resume_policy": resume_policy,
        },
        "selected_model_id": selected_id,
        "models": model_summaries,
        "camera_models": camera_models,
        "registered_images": registered_images,
        "points3D": points,
        "statistics": {
            "registered_keyframes": int(reconstruction.num_reg_images()),
            "landmarks": int(reconstruction.num_points3D()),
            "observations": int(reconstruction.compute_num_observations()),
            "mean_reprojection_error_px": finite_or_none(
                reconstruction.compute_mean_reprojection_error()
            ),
        },
        "warnings": warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="EIA TECH isolated PyCOLMAP SfM worker")
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    request_path = Path(args.request).expanduser().resolve()
    result_path = Path(args.result).expanduser().resolve()
    started = time.monotonic()
    try:
        emit("status", "startup", "PyCOLMAP worker starting.", 0.0)
        payload = run(load_object(request_path))
        payload["worker_elapsed_s"] = max(0.0, time.monotonic() - started)
        atomic_write_object(result_path, payload)
        return 0
    except Exception as exc:
        error_payload = {
            "schema_version": SFM_WORKER_SCHEMA_VERSION,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "worker_elapsed_s": max(0.0, time.monotonic() - started),
        }
        try:
            atomic_write_object(result_path, error_payload)
        except Exception:
            pass
        print(error_payload["traceback"], file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Self-contained helpers for isolated Phase 8.7D2 Gaussian workers."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 1
POSE_CONVENTION = "T_reconstruction_from_camera"


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("JSON payload must be an object")
    return payload


def validate_request(request: dict[str, Any], expected_backend_kind: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if int(request.get("schema_version", 0)) != SCHEMA_VERSION:
        raise ValueError("unsupported Gaussian request schema")
    if request.get("backend_kind") != expected_backend_kind:
        raise ValueError("Gaussian worker received a request for a different backend")
    provenance = request.get("backend_provenance")
    if not isinstance(provenance, dict) or provenance.get("backend_kind") != expected_backend_kind:
        raise ValueError("backend provenance missing or inconsistent")

    dataset_path = Path(str(request["dataset_manifest_path"])).resolve()
    seed_path = Path(str(request["seed_path"])).resolve()
    if not dataset_path.is_file() or not seed_path.is_file():
        raise ValueError("dataset manifest or Gaussian seed is missing")
    dataset = load_json(dataset_path)
    if dataset.get("dataset_fingerprint_sha256") != request.get("dataset_fingerprint_sha256"):
        raise ValueError("dataset fingerprint mismatch")
    if dataset.get("pose_convention") != POSE_CONVENTION:
        raise ValueError("unsupported pose convention")
    frames = dataset.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("Gaussian dataset has no frames")
    maximum = int(request.get("maximum_input_frames", 0))
    if maximum < 1 or len(frames) > maximum:
        raise ValueError("Gaussian dataset exceeds request frame bound")

    with np.load(seed_path, allow_pickle=False) as seed_npz:
        seed = {key: np.asarray(seed_npz[key]).copy() for key in seed_npz.files}
    positions = np.asarray(seed.get("positions_reconstruction"), dtype=np.float32)
    colors = np.asarray(seed.get("colors_rgb"), dtype=np.uint8)
    log_scales = np.asarray(seed.get("log_scales_reconstruction"), dtype=np.float32)
    opacities = np.asarray(seed.get("opacities"), dtype=np.float32)
    if positions.ndim != 2 or positions.shape[1:] != (3,) or len(positions) < 1:
        raise ValueError("invalid Gaussian seed positions")
    if colors.shape != (len(positions), 3):
        raise ValueError("invalid Gaussian seed colors")
    if log_scales.shape != (len(positions), 3):
        raise ValueError("invalid Gaussian seed scales")
    if opacities.shape != (len(positions),):
        raise ValueError("invalid Gaussian seed opacities")
    if not np.isfinite(positions).all() or not np.isfinite(log_scales).all() or not np.isfinite(opacities).all():
        raise ValueError("Gaussian seed contains non-finite values")
    if not ((opacities > 0.0) & (opacities < 1.0)).all():
        raise ValueError("Gaussian seed opacities must be strictly in (0, 1)")
    return dataset, seed


def rotation_matrix_to_quaternion_wxyz(rotation: np.ndarray) -> np.ndarray:
    r = np.asarray(rotation, dtype=np.float64)
    if r.shape != (3, 3):
        raise ValueError("rotation must be 3x3")
    trace = float(np.trace(r))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = np.array([0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s])
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        q = np.array([(r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s])
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        q = np.array([(r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s])
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        q = np.array([(r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s])
    norm = float(np.linalg.norm(q))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("invalid rotation quaternion")
    q /= norm
    if q[0] < 0.0:
        q *= -1.0
    return q


def write_ascii_ply(path: Path, positions: np.ndarray, colors: np.ndarray) -> None:
    xyz = np.asarray(positions, dtype=np.float32)
    rgb = np.asarray(colors, dtype=np.uint8)
    if xyz.shape != (len(rgb), 3) or rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError("PLY positions/colors shape mismatch")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(xyz)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property float nx\nproperty float ny\nproperty float nz\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("end_header\n")
        for point, color in zip(xyz, rgb, strict=True):
            handle.write(
                f"{float(point[0]):.9g} {float(point[1]):.9g} {float(point[2]):.9g} "
                f"0 0 0 {int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def ply_vertex_count(path: Path) -> int:
    with path.open("rb") as handle:
        for raw in handle:
            line = raw.decode("ascii", errors="replace").strip()
            if line.startswith("element vertex "):
                return int(line.split()[-1])
            if line == "end_header":
                break
    raise ValueError("PLY does not contain an element vertex declaration")


def write_worker_result(
    *,
    result_path: Path,
    request: dict[str, Any],
    dataset: dict[str, Any],
    backend_id: str,
    backend_version: str,
    artifact_dir: Path,
    scene_path: Path,
    gaussian_count: int,
    training_steps: int,
    training_duration_s: float,
    device: str,
    peak_vram_mib: float | None,
) -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "request_id": request["request_id"],
        "backend_id": backend_id,
        "backend_version": backend_version,
        "backend_provenance": request["backend_provenance"],
        "optimization_state": "optimized",
        "scene_role": "appearance_only",
        "artifact_dir": str(artifact_dir.resolve()),
        "scene_path": str(scene_path.resolve()),
        "dataset_fingerprint_sha256": request["dataset_fingerprint_sha256"],
        "input_frame_count": int(dataset["frame_count"]),
        "gaussian_count": int(gaussian_count),
        "training_steps": int(training_steps),
        "training_duration_s": float(training_duration_s),
        "render_capable": True,
        "device": str(device),
        "peak_vram_mib": None if peak_vram_mib is None else float(peak_vram_mib),
        "pose_convention": dataset["pose_convention"],
        "reconstruction_frame_id": dataset["reconstruction_frame_id"],
        "metric_geometry_status": dataset["metric_geometry_status"],
        "unit_to_metre": dataset["unit_to_metre"],
        "relative_scale_uncertainty": dataset["relative_scale_uncertainty"],
        "canonical_geometry_authoritative": False,
        "certified_metrology": False,
    }
    artifact_manifest = artifact_dir / "artifact_manifest.json"
    artifact_manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def write_qualification_metrics(
    *,
    artifact_dir: Path,
    backend_kind: str,
    source_revision: str,
    training_steps: int,
    preview_path: Path,
    real_cuda_execution: bool,
    optimization_completed: bool,
    extra: dict[str, Any] | None = None,
) -> Path:
    metrics_path = artifact_dir / "qualification_metrics.json"
    payload: dict[str, Any] = {
        "backend_kind": backend_kind,
        "source_revision": str(source_revision),
        "training_steps": int(training_steps),
        "real_cuda_execution": bool(real_cuda_execution),
        "optimization_completed": bool(optimization_completed),
        "render_written": preview_path.is_file(),
        "preview_path": str(preview_path.resolve()),
    }
    if extra:
        payload.update(extra)
    metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics_path

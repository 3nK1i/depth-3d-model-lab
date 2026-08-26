#!/usr/bin/env python3
"""Isolated PyCOLMAP worker for I4.2D raw TempleRing MVS.

Allowed operations: image undistortion, PatchMatch Stereo, geometric Stereo
Fusion.  Forbidden operations: SfM, bundle adjustment, pose refinement,
held-out GT consumption, production manifold filtering, provider promotion,
metric-scale estimation, and mesh generation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

import numpy as np

EXPECTED_VIEWS = 47
EXPECTED_NAMES = [f"templeR{index:04d}.png" for index in range(1, 48)]
SPARSE_FILES = ("cameras.bin", "images.bin", "points3D.bin", "frames.bin", "rigs.bin")


def read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def process_peak_rss_gib() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0 * 1024.0)


def available_ram_gib() -> float:
    try:
        lines = Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
    except OSError:
        return 0.0
    values = {}
    for line in lines:
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        parts = raw.strip().split()
        if parts:
            try:
                values[key] = int(parts[0])
            except ValueError:
                pass
    return float(values.get("MemAvailable", values.get("MemFree", 0))) / (1024.0 * 1024.0)


def gpu_snapshot() -> list[dict]:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
    except Exception:
        return []
    if completed.returncode != 0:
        return []
    result = []
    for line in completed.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 5:
            continue
        try:
            result.append({
                "index": int(fields[0]),
                "name": fields[1],
                "total_vram_gib": float(fields[2]) / 1024.0,
                "used_vram_gib": float(fields[3]) / 1024.0,
                "free_vram_gib": float(fields[4]) / 1024.0,
            })
        except ValueError:
            continue
    return result


def resource_snapshot(workspace: Path) -> dict:
    target = workspace
    while not target.exists() and target != target.parent:
        target = target.parent
    usage = shutil.disk_usage(target)
    return {
        "time_unix": time.time(),
        "disk_free_gib": float(usage.free) / (1024.0 ** 3),
        "ram_available_gib": available_ram_gib(),
        "process_peak_rss_gib": process_peak_rss_gib(),
        "gpus": gpu_snapshot(),
    }


def matrix44_from_image(image) -> np.ndarray:
    matrix = np.asarray(image.cam_from_world().matrix(), dtype=np.float64)
    result = np.eye(4, dtype=np.float64)
    result[:3, :] = matrix
    return result


def validate_dense_sparse(pycolmap, input_sparse: Path, dense_sparse: Path, image_names: list[str]) -> dict:
    source = pycolmap.Reconstruction(input_sparse)
    dense = pycolmap.Reconstruction(dense_sparse)
    source_names = {str(source.image(int(image_id)).name): int(image_id) for image_id in source.reg_image_ids()}
    dense_names = {str(dense.image(int(image_id)).name): int(image_id) for image_id in dense.reg_image_ids()}
    names_exact = set(source_names) == set(dense_names) == set(image_names)
    poses_preserved = names_exact
    if names_exact:
        for name, source_id in source_names.items():
            dense_id = dense_names[name]
            if not np.allclose(
                matrix44_from_image(source.image(source_id)),
                matrix44_from_image(dense.image(dense_id)),
                rtol=0.0,
                atol=1e-10,
            ):
                poses_preserved = False
                break
    source_points = {int(pid): np.asarray(source.point3D(int(pid)).xyz, dtype=np.float64) for pid in source.point3D_ids()}
    dense_points = {int(pid): np.asarray(dense.point3D(int(pid)).xyz, dtype=np.float64) for pid in dense.point3D_ids()}
    point_ids_preserved = set(source_points) == set(dense_points)
    point_positions_preserved = point_ids_preserved and all(
        np.array_equal(source_points[pid], dense_points[pid]) for pid in source_points
    )
    checks = {
        "registered_47": int(dense.num_reg_images()) == EXPECTED_VIEWS,
        "image_names_exact": names_exact,
        "poses_preserved": poses_preserved,
        "sparse_point_ids_preserved": point_ids_preserved,
        "sparse_point_positions_preserved": point_positions_preserved,
    }
    return {
        "checks": checks,
        "pass": all(checks.values()),
        "source_registered_images": int(source.num_reg_images()),
        "dense_registered_images": int(dense.num_reg_images()),
        "source_points3D": int(source.num_points3D()),
        "dense_points3D": int(dense.num_points3D()),
        "dense_cameras": int(dense.num_cameras()),
    }


def parse_patch_match_cfg(path: Path, image_names: list[str], expected_neighbors: int) -> dict:
    if not path.is_file():
        raise RuntimeError(f"PatchMatch configuration is missing: {path}")
    data = [
        line.strip() for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(data) % 2 != 0:
        raise RuntimeError("PatchMatch configuration must contain reference/source line pairs.")
    refs: list[str] = []
    source_specs: list[str] = []
    source_limits: list[int] = []
    for offset in range(0, len(data), 2):
        ref = data[offset]
        spec = data[offset + 1]
        refs.append(ref)
        source_specs.append(spec)
        if spec.startswith("__auto__"):
            parts = [item.strip() for item in spec.split(",")]
            if len(parts) != 2:
                raise RuntimeError(f"Malformed automatic PatchMatch source spec: {spec}")
            source_limits.append(int(parts[1]))
        elif spec == "__all__":
            source_limits.append(len(image_names) - 1)
        else:
            source_limits.append(len([item for item in spec.split(",") if item.strip()]))
    checks = {
        "all_47_reference_views_present": len(refs) == EXPECTED_VIEWS and set(refs) == set(image_names),
        "reference_views_unique": len(set(refs)) == len(refs),
        "source_neighbor_limit_respected": all(limit <= expected_neighbors for limit in source_limits),
        "no_global_view_cap": len(refs) == EXPECTED_VIEWS,
        "no_all_sources_default": all(spec != "__all__" for spec in source_specs),
    }
    return {
        "checks": checks,
        "pass": all(checks.values()),
        "reference_views": len(refs),
        "source_limit_min": min(source_limits) if source_limits else 0,
        "source_limit_max": max(source_limits) if source_limits else 0,
        "automatic_specs": sum(spec.startswith("__auto__") for spec in source_specs),
    }


def ply_vertex_count(path: Path) -> int:
    if not path.is_file() or path.stat().st_size <= 0:
        return 0
    with path.open("rb") as handle:
        header = handle.read(1024 * 1024)
    marker = b"end_header"
    if marker not in header:
        raise RuntimeError("Fused PLY header is missing end_header.")
    text = header.split(marker, 1)[0].decode("ascii", errors="strict")
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) == 3 and parts[0] == "element" and parts[1] == "vertex":
            return int(parts[2])
    raise RuntimeError("Fused PLY header does not declare a vertex count.")


def stage_state(path: Path, result: dict, stage: str, *, started: bool = False, completed: bool = False) -> None:
    payload = {
        "schema_version": result["schema_version"],
        "phase": "I4.2D",
        "stage": stage,
        "stage_started": started,
        "stage_completed": completed,
        "heavy_mvs_started": bool(result.get("heavy_mvs_started", False)),
        "image_undistortion_started": bool(result.get("image_undistortion_started", False)),
        "patch_match_started": bool(result.get("patch_match_started", False)),
        "stereo_fusion_started": bool(result.get("stereo_fusion_started", False)),
        "resources": resource_snapshot(path.parent),
    }
    write_json(path, payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    args = parser.parse_args()

    request = read_json(args.request.resolve())
    result_path = args.result.resolve()
    state_path = args.state.resolve()
    result = {
        "schema_version": int(request.get("schema_version", 0)),
        "phase": "I4.2D",
        "status": "error",
        "error": "",
        "sfm_rerun": False,
        "bundle_adjustment_rerun": False,
        "pose_refinement": False,
        "held_out_ground_truth_consumed": False,
        "metric_scale_claimed": False,
        "measurement_authoritative": False,
        "certified_metrology": False,
        "surface_mesh_generated": False,
        "production_filter_policy_changed": False,
        "production_manifold_applied": False,
        "dense_provider_promoted": False,
        "i4_3_provider_architecture_authorized": False,
        "heavy_mvs_started": False,
        "image_undistortion_started": False,
        "image_undistortion_completed": False,
        "patch_match_started": False,
        "patch_match_completed": False,
        "stereo_fusion_started": False,
        "stereo_fusion_completed": False,
        "resource_snapshots": {},
        "stage_seconds": {},
    }
    try:
        import pycolmap

        result["pycolmap_version"] = str(getattr(pycolmap, "__version__", ""))
        if result["schema_version"] != 1:
            raise RuntimeError("Unsupported I4.2D request schema version.")
        if request.get("action") != "temple_ring_raw_mvs_dense_reconstruction":
            raise RuntimeError("Unsupported I4.2D worker action.")
        authority = request.get("authority", {})
        forbidden_true = (
            "sfm_rerun", "bundle_adjustment_rerun", "pose_refinement", "held_out_ground_truth_consumed",
            "metric_scale_claimed", "measurement_authoritative", "certified_metrology", "surface_mesh_generated",
            "production_filter_policy_changed", "production_manifold_applied", "dense_provider_promoted",
            "i4_3_provider_architecture_authorized",
        )
        if any(bool(authority.get(key)) for key in forbidden_true):
            raise RuntimeError("I4.2D request violates locked authority flags.")

        input_images = Path(str(request["input_images"])).expanduser().resolve()
        input_sparse = Path(str(request["input_sparse"])).expanduser().resolve()
        dense = Path(str(request["output_workspace"])).expanduser().resolve()
        fused_ply = Path(str(request["fused_ply"])).expanduser().resolve()
        image_names = [str(item) for item in request["image_names"]]
        image_sha256 = {str(k): str(v) for k, v in request["image_sha256"].items()}
        expected_sparse_hashes = {str(k): str(v) for k, v in request["input_sparse_sha256"].items()}
        expected_views = int(request["expected_registered_views"])
        config = request["config"]
        if expected_views != EXPECTED_VIEWS or image_names != EXPECTED_NAMES or len(set(image_names)) != EXPECTED_VIEWS:
            raise RuntimeError("I4.2D worker requires the exact canonical 47 TempleRing image names.")
        if dense.exists() and any(dense.iterdir()):
            raise RuntimeError(f"Dense workspace must be empty or absent: {dense}")
        dense.parent.mkdir(parents=True, exist_ok=True)
        for name in image_names:
            path = input_images / name
            if not path.is_file() or sha256_file(path) != image_sha256.get(name):
                raise RuntimeError(f"I4.2D staged image checksum mismatch: {name}")
        for filename in SPARSE_FILES:
            path = input_sparse / filename
            if not path.is_file() or sha256_file(path) != expected_sparse_hashes.get(filename):
                raise RuntimeError(f"I4.2D authoritative sparse checksum mismatch: {filename}")

        source_neighbors = int(config["source_neighbors"])
        max_image_size = int(config["max_image_size"])
        if source_neighbors < 1 or source_neighbors >= EXPECTED_VIEWS:
            raise RuntimeError("Invalid I4.2D source-neighbor count.")
        if config.get("geometric_consistency") is not True:
            raise RuntimeError("Geometric consistency is locked ON for I4.2D.")
        if config.get("production_manifold_applied") is not False:
            raise RuntimeError("I4.2D must produce raw MVS before production manifold qualification.")

        result["resource_snapshots"]["before_undistort"] = resource_snapshot(dense)
        result["heavy_mvs_started"] = True
        result["image_undistortion_started"] = True
        stage_state(state_path, result, "undistort", started=True)
        started = time.monotonic()
        undistort_options = pycolmap.UndistortCameraOptions()
        undistort_options.max_image_size = max_image_size
        pycolmap.undistort_images(
            output_path=dense,
            input_path=input_sparse,
            image_path=input_images,
            image_names=image_names,
            output_type="COLMAP",
            num_patch_match_src_images=source_neighbors,
            undistort_options=undistort_options,
        )
        result["stage_seconds"]["undistort"] = time.monotonic() - started
        result["image_undistortion_completed"] = True
        dense_validation = validate_dense_sparse(pycolmap, input_sparse, dense / "sparse", image_names)
        result["dense_sparse_validation"] = dense_validation
        if not dense_validation["pass"]:
            raise RuntimeError(f"Undistorted workspace changed authoritative pose/geometry: {dense_validation['checks']}")
        cfg_validation = parse_patch_match_cfg(dense / "stereo/patch-match.cfg", image_names, source_neighbors)
        result["patch_match_config_validation"] = cfg_validation
        if not cfg_validation["pass"]:
            raise RuntimeError(f"PatchMatch source/reference validation failed: {cfg_validation['checks']}")
        result["resource_snapshots"]["after_undistort"] = resource_snapshot(dense)
        stage_state(state_path, result, "undistort", completed=True)

        result["patch_match_started"] = True
        stage_state(state_path, result, "patch_match", started=True)
        started = time.monotonic()
        patch_options = pycolmap.PatchMatchOptions()
        patch_options.gpu_index = str(config["gpu_index"])
        patch_options.max_image_size = max_image_size
        patch_options.geom_consistency = True
        patch_options.filter = True
        patch_options.cache_size = float(config["patch_match_cache_gib"])
        patch_options.allow_missing_files = False
        pycolmap.patch_match_stereo(
            workspace_path=dense,
            workspace_format="COLMAP",
            options=patch_options,
        )
        result["stage_seconds"]["patch_match"] = time.monotonic() - started
        result["patch_match_completed"] = True
        depth_dir = dense / "stereo/depth_maps"
        normal_dir = dense / "stereo/normal_maps"
        geometric_depth = sorted(depth_dir.glob("*.geometric.bin"))
        geometric_normal = sorted(normal_dir.glob("*.geometric.bin"))
        result["patch_match_outputs"] = {
            "geometric_depth_maps": len(geometric_depth),
            "geometric_normal_maps": len(geometric_normal),
            "expected_reference_views": EXPECTED_VIEWS,
        }
        if len(geometric_depth) != EXPECTED_VIEWS or len(geometric_normal) != EXPECTED_VIEWS:
            raise RuntimeError(
                f"PatchMatch did not produce geometric maps for all 47 references: "
                f"depth={len(geometric_depth)} normal={len(geometric_normal)}"
            )
        result["resource_snapshots"]["after_patch_match"] = resource_snapshot(dense)
        stage_state(state_path, result, "patch_match", completed=True)

        result["stereo_fusion_started"] = True
        stage_state(state_path, result, "stereo_fusion", started=True)
        started = time.monotonic()
        fusion_options = pycolmap.StereoFusionOptions()
        fusion_options.cache_size = float(config["stereo_fusion_cache_gib"])
        fusion_options.use_cache = True
        fusion_options.max_image_size = max_image_size
        fused = pycolmap.stereo_fusion(
            output_path=fused_ply,
            workspace_path=dense,
            workspace_format="COLMAP",
            input_type="geometric",
            options=fusion_options,
            output_type="ply",
        )
        result["stage_seconds"]["stereo_fusion"] = time.monotonic() - started
        result["stereo_fusion_completed"] = True
        vertices = ply_vertex_count(fused_ply)
        returned_points = int(fused.num_points3D()) if hasattr(fused, "num_points3D") else vertices
        result["fusion"] = {
            "raw_fused_ply": str(fused_ply),
            "fused_ply_bytes": fused_ply.stat().st_size if fused_ply.is_file() else 0,
            "ply_vertex_count": vertices,
            "returned_point_count": returned_points,
            "input_type": "geometric",
            "production_manifold_applied": False,
        }
        if vertices <= 0:
            raise RuntimeError("Stereo Fusion produced an empty TempleRing fused PLY.")
        result["resource_snapshots"]["after_stereo_fusion"] = resource_snapshot(dense)
        result["process_peak_rss_gib"] = process_peak_rss_gib()
        stage_state(state_path, result, "stereo_fusion", completed=True)

        post_sparse_hashes = {filename: sha256_file(input_sparse / filename) for filename in SPARSE_FILES}
        if post_sparse_hashes != expected_sparse_hashes:
            raise RuntimeError("Authoritative sparse model changed inside I4.2D worker.")
        result["authoritative_sparse_sha256_after"] = post_sparse_hashes
        result["status"] = "pass"
        result["detail_qualification_pending"] = True
        write_json(result_path, result)
        return 0
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["process_peak_rss_gib"] = process_peak_rss_gib()
        try:
            stage_state(state_path, result, "failed", completed=False)
        except Exception:
            pass
        write_json(result_path, result)
        print(result["error"], file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

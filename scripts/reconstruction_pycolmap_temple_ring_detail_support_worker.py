#!/usr/bin/env python3
"""I4.2E.2 dependency-light native COLMAP dense-support export worker.

This worker consumes the *existing* I4.2D ``fused.ply`` and its native
``fused.ply.vis`` visibility sidecar.  It does not rerun Stereo Fusion,
PatchMatch, SfM, BA, undistortion, pose refinement, filtering, or any production
policy.  COLMAP 4.1.1 writes the PLY and visibility vectors from the same fused
point ordering, so the sidecar is the authoritative dense-support provenance for
this qualification step.

Dependency boundary
-------------------
The worker intentionally imports neither ``core.reconstruction`` nor OpenCV nor
PyCOLMAP.  NumPy plus the Python standard library are sufficient to read the
frozen PLY/visibility pair and persist compact support evidence.  It can still be
launched with the isolated PyCOLMAP interpreter to preserve the established
process boundary without broadening that environment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys

import numpy as np


_PLY_SCALAR_DTYPES: dict[str, str] = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--request", required=True, type=Path)
    p.add_argument("--result", required=True, type=Path)
    return p.parse_args()


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("I4.2E.2 worker request must be a JSON object.")
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_ply_header(handle) -> tuple[str, int, list[tuple[str, str]]]:
    if handle.readline().strip() != b"ply":
        raise RuntimeError("I4.2E fused PLY has an invalid header.")
    fmt = ""
    vertex_count: int | None = None
    vertex_properties: list[tuple[str, str]] = []
    current_element = ""
    while True:
        raw = handle.readline()
        if not raw:
            raise RuntimeError("I4.2E fused PLY header is truncated.")
        try:
            line = raw.decode("ascii").strip()
        except UnicodeDecodeError as exc:
            raise RuntimeError("I4.2E fused PLY header must be ASCII.") from exc
        if line == "end_header":
            break
        if not line or line.startswith("comment") or line.startswith("obj_info"):
            continue
        parts = line.split()
        if parts[0] == "format" and len(parts) >= 2:
            fmt = parts[1]
        elif parts[0] == "element" and len(parts) == 3:
            current_element = parts[1]
            if current_element == "vertex":
                try:
                    vertex_count = int(parts[2])
                except ValueError as exc:
                    raise RuntimeError("I4.2E fused PLY vertex count is invalid.") from exc
        elif parts[0] == "property" and current_element == "vertex":
            if len(parts) != 3 or parts[1] == "list":
                raise RuntimeError("I4.2E fused PLY vertex properties must be scalar.")
            vertex_properties.append((parts[2], parts[1]))
    if fmt not in ("binary_little_endian", "binary_big_endian", "ascii"):
        raise RuntimeError(f"I4.2E fused PLY format is unsupported: {fmt!r}")
    if vertex_count is None or vertex_count <= 0:
        raise RuntimeError("I4.2E fused PLY must contain vertices.")
    if not vertex_properties:
        raise RuntimeError("I4.2E fused PLY has no vertex properties.")
    return fmt, vertex_count, vertex_properties


def _validate_cloud(points: object, colors: object) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.asarray(points, dtype=np.float64)
    rgb = np.asarray(colors, dtype=np.uint8)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) == 0:
        raise RuntimeError("I4.2E fused point cloud must be non-empty Nx3 geometry.")
    if rgb.shape != xyz.shape:
        raise RuntimeError("I4.2E fused RGB shape must match geometry.")
    if not np.all(np.isfinite(xyz)):
        raise RuntimeError("I4.2E fused point cloud contains non-finite geometry.")
    return np.ascontiguousarray(xyz), np.ascontiguousarray(rgb)


def _read_colmap_fused_ply(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read XYZ/RGB from a COLMAP fused PLY without project-side imports."""
    source = Path(path).expanduser().resolve()
    with source.open("rb") as handle:
        fmt, vertex_count, properties = _parse_ply_header(handle)
        names = [name for name, _ in properties]
        required = {"x", "y", "z", "red", "green", "blue"}
        if not required.issubset(names):
            missing = sorted(required.difference(names))
            raise RuntimeError(f"I4.2E fused PLY is missing required properties: {missing}")
        for _, scalar_type in properties:
            if scalar_type not in _PLY_SCALAR_DTYPES:
                raise RuntimeError(f"I4.2E fused PLY uses unsupported scalar type: {scalar_type}")
        if fmt == "ascii":
            rows: list[list[str]] = []
            for _ in range(vertex_count):
                raw = handle.readline()
                if not raw:
                    raise RuntimeError("I4.2E fused PLY vertex payload is truncated.")
                try:
                    rows.append(raw.decode("ascii").split())
                except UnicodeDecodeError as exc:
                    raise RuntimeError("I4.2E fused PLY ASCII payload is invalid.") from exc
            if any(len(row) != len(properties) for row in rows):
                raise RuntimeError("I4.2E fused PLY ASCII row width is invalid.")
            try:
                matrix = np.asarray(rows, dtype=np.float64)
            except ValueError as exc:
                raise RuntimeError("I4.2E fused PLY ASCII values are invalid.") from exc
            by_name = {name: matrix[:, idx] for idx, (name, _) in enumerate(properties)}
        else:
            endian = "<" if fmt == "binary_little_endian" else ">"
            fields = [(name, endian + _PLY_SCALAR_DTYPES[scalar_type]) for name, scalar_type in properties]
            dtype = np.dtype(fields)
            expected_bytes = dtype.itemsize * vertex_count
            raw = handle.read(expected_bytes)
            if len(raw) != expected_bytes:
                raise RuntimeError("I4.2E fused PLY binary vertex payload is truncated.")
            data = np.frombuffer(raw, dtype=dtype, count=vertex_count)
            by_name = {name: data[name] for name, _ in properties}
        points = np.column_stack([by_name["x"], by_name["y"], by_name["z"]]).astype(np.float64, copy=False)
        colors = np.column_stack([by_name["red"], by_name["green"], by_name["blue"]]).astype(np.uint8, copy=False)
    return _validate_cloud(points, colors)


def _canonical_cloud_digest(points: object, colors: object) -> str:
    """Order-independent digest matching I4.2E float32 XYZ + uint8 RGB semantics."""
    xyz, rgb = _validate_cloud(points, colors)
    xyz = xyz.astype(np.float32)
    xyz[xyz == 0.0] = 0.0
    order = np.lexsort((rgb[:, 2], rgb[:, 1], rgb[:, 0], xyz[:, 2], xyz[:, 1], xyz[:, 0]))
    dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("r", "u1"), ("g", "u1"), ("b", "u1"),
    ])
    rows = np.empty(len(xyz), dtype=dtype)
    sorted_xyz = xyz[order]
    sorted_rgb = rgb[order]
    rows["x"], rows["y"], rows["z"] = sorted_xyz[:, 0], sorted_xyz[:, 1], sorted_xyz[:, 2]
    rows["r"], rows["g"], rows["b"] = sorted_rgb[:, 0], sorted_rgb[:, 1], sorted_rgb[:, 2]
    return hashlib.sha256(rows.tobytes()).hexdigest()


def _read_exact(handle, size: int, *, label: str) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise RuntimeError(f"I4.2E native visibility file is truncated while reading {label}.")
    return data


def _read_points_visibility(
    path: str | Path,
    *,
    expected_points: int,
    expected_views: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Read COLMAP ``fused.ply.vis`` in the exact 4.1.1 binary layout.

    Layout: uint64 point_count, then for every fused point uint32 support_count
    followed by ``support_count`` uint32 MVS image indices.  The indices are
    native zero-based MVS image indices, not COLMAP image IDs.
    """
    source = Path(path).expanduser().resolve()
    if int(expected_points) <= 0 or int(expected_views) <= 0:
        raise RuntimeError("I4.2E native visibility expectations must be positive.")
    offsets = np.zeros(int(expected_points) + 1, dtype=np.int64)
    flat: list[int] = []
    with source.open("rb") as handle:
        (num_points,) = struct.unpack("<Q", _read_exact(handle, 8, label="point count"))
        if int(num_points) != int(expected_points):
            raise RuntimeError(
                "I4.2E native visibility point count does not match fused.ply: "
                f"visibility={int(num_points)} ply={int(expected_points)}"
            )
        for point_index in range(int(expected_points)):
            (count,) = struct.unpack("<I", _read_exact(handle, 4, label=f"support count for point {point_index}"))
            count = int(count)
            if count <= 0:
                raise RuntimeError(f"I4.2E native visibility point {point_index} has empty support.")
            if count > int(expected_views):
                raise RuntimeError(
                    f"I4.2E native visibility point {point_index} has impossible support count {count}."
                )
            raw = _read_exact(handle, 4 * count, label=f"image indices for point {point_index}")
            ids = list(struct.unpack(f"<{count}I", raw))
            if len(set(ids)) != len(ids):
                raise RuntimeError(f"I4.2E native visibility point {point_index} contains duplicate image indices.")
            if any(int(image_idx) >= int(expected_views) for image_idx in ids):
                raise RuntimeError(
                    f"I4.2E native visibility point {point_index} contains an out-of-range image index."
                )
            ids.sort()
            flat.extend(int(value) for value in ids)
            offsets[point_index + 1] = len(flat)
        trailing = handle.read(1)
        if trailing:
            raise RuntimeError("I4.2E native visibility file contains unexpected trailing bytes.")
    image_indices = np.asarray(flat, dtype=np.int32)
    # A valid COLMAP visibility sidecar is allowed to reference only a subset
    # of the campaign views.  The file describes support of surviving fused
    # points; it is not a statement that every MVS reference image contributes
    # at least one surviving support observation.  Per-observation range and
    # per-point uniqueness checks above are the validity contract.
    return offsets, image_indices


def main() -> int:
    args = _args()
    request = _read_json(args.request.resolve())
    result = {
        "schema_version": 2,
        "phase": "I4.2E",
        "correction": "I4.2E.2",
        "status": "error",
        "error": "",
        "dependency_contract": {
            "project_reconstruction_package_required": False,
            "opencv_required": False,
            "pycolmap_import_required": False,
            "worker_local_ply_reader": True,
            "worker_local_native_visibility_reader": True,
        },
        "native_visibility_consumed": False,
        "stereo_fusion_replayed_for_support_evidence": False,
        "patch_match_rerun": False,
        "sfm_rerun": False,
        "bundle_adjustment_rerun": False,
        "pose_refinement": False,
        "held_out_ground_truth_consumed": False,
        "production_filter_policy_changed": False,
        "production_manifold_applied_to_authoritative_scene": False,
        "metric_scale_claimed": False,
        "measurement_authoritative": False,
        "certified_metrology": False,
        "surface_mesh_generated": False,
        "dense_provider_promoted": False,
        "i4_3_provider_architecture_authorized": False,
    }
    try:
        if request.get("schema_version") != 2 or request.get("action") != "temple_ring_export_native_fusion_visibility":
            raise RuntimeError("Unsupported I4.2E.2 worker request.")
        authority = request.get("authority", {})
        forbidden = (
            "stereo_fusion_replay", "patch_match_rerun", "sfm_rerun", "bundle_adjustment_rerun",
            "pose_refinement", "held_out_ground_truth_consumed", "production_filter_policy_changed",
            "production_manifold_applied_to_authoritative_scene", "metric_scale_claimed",
            "measurement_authoritative", "certified_metrology", "surface_mesh_generated",
            "dense_provider_promoted", "i4_3_provider_architecture_authorized",
        )
        if any(bool(authority.get(name)) for name in forbidden):
            raise RuntimeError("I4.2E.2 worker request violates locked authority state.")

        raw_ply = Path(str(request["raw_fused_ply"])).expanduser().resolve()
        native_vis = Path(str(request["native_visibility_path"])).expanduser().resolve()
        support_npz = Path(str(request["support_npz"])).expanduser().resolve()
        expected_raw_bytes = int(request["expected_raw_ply_bytes"])
        expected_raw_vertices = int(request["expected_raw_vertices"])
        expected_views = int(request["expected_views"])
        producer_pycolmap_version = str(request.get("producer_pycolmap_version", ""))
        if producer_pycolmap_version != "4.1.1":
            raise RuntimeError(
                "I4.2E.2 native visibility contract is frozen to the I4.2D PyCOLMAP 4.1.1 producer."
            )
        if not raw_ply.is_file() or not native_vis.is_file():
            raise RuntimeError("I4.2E.2 requires the existing I4.2D fused.ply and fused.ply.vis pair.")
        if support_npz.exists():
            raise RuntimeError("I4.2E.2 support output must be absent before execution.")
        if raw_ply.stat().st_size != expected_raw_bytes:
            raise RuntimeError("I4.2E.2 raw fused PLY size changed after I4.2D.")

        points, colors = _read_colmap_fused_ply(raw_ply)
        if len(points) != expected_raw_vertices:
            raise RuntimeError(
                f"I4.2E.2 fused PLY vertex count changed: {len(points)} != {expected_raw_vertices}."
            )
        offsets, image_indices = _read_points_visibility(
            native_vis,
            expected_points=expected_raw_vertices,
            expected_views=expected_views,
        )
        support = np.diff(offsets)
        support_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            support_npz,
            points_reconstruction=np.asarray(points, dtype=np.float64),
            colors_rgb=np.asarray(colors, dtype=np.uint8),
            visibility_offsets=np.asarray(offsets, dtype=np.int64),
            visibility_image_indices=np.asarray(image_indices, dtype=np.int32),
        )
        observed_image_indices = np.unique(image_indices).astype(np.int32)
        expected_image_indices = np.arange(expected_views, dtype=np.int32)
        result.update({
            "status": "pass",
            "native_visibility_consumed": True,
            "producer_pycolmap_version": producer_pycolmap_version,
            "raw_fused_ply_sha256": _sha(raw_ply),
            "native_visibility_sha256": _sha(native_vis),
            "cloud_digest": _canonical_cloud_digest(points, colors),
            "raw_vertices": int(len(points)),
            "visibility_point_count": int(len(points)),
            "visibility_observations": int(len(image_indices)),
            "visibility_support_min": int(np.min(support)),
            "visibility_support_max": int(np.max(support)),
            "visibility_support_median": float(np.median(support)),
            "native_visibility_expected_view_count": int(expected_views),
            "native_visibility_unique_image_indices": int(len(observed_image_indices)),
            "native_visibility_observed_image_indices": observed_image_indices.tolist(),
            "native_visibility_indices_within_expected_range": True,
            "native_visibility_exact_image_index_set": bool(
                np.array_equal(observed_image_indices, expected_image_indices)
            ),
            "native_visibility_full_view_coverage_required": False,
            "visibility_indices_are_mvs_indices_not_colmap_image_ids": True,
            "point_order_bound_to_fused_ply": True,
            "support_npz": str(support_npz),
            "support_npz_sha256": _sha(support_npz),
            "native_visibility_path": str(native_vis),
        })
        _write_json(args.result.resolve(), result)
        return 0
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        _write_json(args.result.resolve(), result)
        print(result["error"], file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())

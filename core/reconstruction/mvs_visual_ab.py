"""Controlled DA3-vs-MVS visual A/B preparation for I4.1.9A A3.

A3 is diagnostic only.  It compares the already-produced I4.1.9 DA3 dense
geometry against the already-produced A2 COLMAP MVS fused cloud in the same
authoritative arbitrary-gauge SfM frame.  It never reruns reconstruction,
changes camera geometry, establishes metric scale, or promotes measurement or
mesh authority.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from math import isfinite
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from core.reconstruction.errors import ReconstructionValidationError

MVS_VISUAL_AB_SCHEMA_VERSION = 1
MVS_VISUAL_AB_EXPECTED_VIEWS = 68
MVS_VISUAL_AB_DEFAULT_POINT_BUDGET = 250_000
MVS_VISUAL_AB_VIEWS: dict[str, tuple[float, float]] = {
    "oblique": (32.0, 24.0),
    "front": (0.0, 0.0),
    "side": (90.0, 0.0),
    "top": (0.0, 89.0),
}

_PLY_SCALAR_DTYPES: dict[str, str] = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError(f"Could not read A3 JSON evidence {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"A3 JSON evidence must contain an object: {path}")
    return payload


def write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_clean_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise ReconstructionValidationError(
            f"A3 output directory must be empty or absent; refusing to overwrite visual A/B evidence: {path}"
        )


def _finite_vec3(value: Any, *, field_name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ReconstructionValidationError(f"{field_name} must be a finite 3-vector.")
    return array


@dataclass(frozen=True, slots=True)
class SharedVisualFrame:
    lower_reconstruction: tuple[float, float, float]
    upper_reconstruction: tuple[float, float, float]
    center_reconstruction: tuple[float, float, float]
    center_viewer: tuple[float, float, float]
    robust_diagonal: float
    camera_distance: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "basis": "locked_DA3_I2_core_anchor_envelope_for_shared_framing_only",
            "lower_reconstruction": list(self.lower_reconstruction),
            "upper_reconstruction": list(self.upper_reconstruction),
            "center_reconstruction": list(self.center_reconstruction),
            "center_viewer": list(self.center_viewer),
            "robust_diagonal": self.robust_diagonal,
            "camera_distance": self.camera_distance,
            "fov_degrees": 45.0,
            "viewer_transform": "(x, y, z)_reconstruction -> (x, z, -y)_viewer",
            "metric_scale_claimed": False,
        }


@dataclass(frozen=True, slots=True)
class VisualAbSources:
    da3_i4_dir: Path
    da3_i4_summary: Path
    da3_i2_summary: Path
    da3_diagnostics_npz: Path
    mvs_a2_dir: Path
    mvs_a2_summary: Path
    mvs_fused_ply: Path
    a1_manifest: Path
    source_i4_1_7_dir: Path
    source_i4_1_7_summary: Path
    shared_frame: SharedVisualFrame
    frozen_sha256: dict[str, str]


def _shared_frame(i2: Mapping[str, Any]) -> SharedVisualFrame:
    spatial = i2.get("spatial")
    if not isinstance(spatial, Mapping):
        raise ReconstructionValidationError("A3 DA3 I2 evidence is missing spatial diagnostics.")
    envelope = spatial.get("core_anchor_envelope")
    if not isinstance(envelope, Mapping):
        raise ReconstructionValidationError("A3 DA3 I2 evidence is missing the Core anchor envelope.")
    lower = _finite_vec3(envelope.get("lower"), field_name="core_anchor_envelope.lower")
    upper = _finite_vec3(envelope.get("upper"), field_name="core_anchor_envelope.upper")
    if np.any(upper <= lower):
        raise ReconstructionValidationError("A3 Core anchor envelope must have positive extent on every axis.")
    center = 0.5 * (lower + upper)
    diagonal = float(np.linalg.norm(upper - lower))
    recorded = float(envelope.get("robust_diagonal", diagonal))
    if not isfinite(recorded) or recorded <= 0.0:
        raise ReconstructionValidationError("A3 Core anchor robust diagonal must be finite and positive.")
    if abs(recorded - diagonal) > max(1e-6, 0.02 * diagonal):
        raise ReconstructionValidationError("A3 Core anchor envelope/diagonal evidence is internally inconsistent.")
    center_viewer = np.asarray([center[0], center[2], -center[1]], dtype=np.float64)
    return SharedVisualFrame(
        lower_reconstruction=tuple(float(v) for v in lower),
        upper_reconstruction=tuple(float(v) for v in upper),
        center_reconstruction=tuple(float(v) for v in center),
        center_viewer=tuple(float(v) for v in center_viewer),
        robust_diagonal=recorded,
        camera_distance=max(recorded * 1.45, 0.75),
    )


def load_visual_ab_sources(
    da3_i4_dir: str | Path,
    mvs_a2_dir: str | Path,
) -> VisualAbSources:
    """Validate that DA3 and MVS artifacts are the intended controlled House A/B."""

    da3_root = Path(da3_i4_dir).expanduser().resolve()
    a2_root = Path(mvs_a2_dir).expanduser().resolve()
    da3_summary_path = da3_root / "i4_1_9_summary.json"
    a2_summary_path = a2_root / "i4_1_9a_a2_summary.json"
    if not da3_summary_path.is_file():
        raise ReconstructionValidationError(f"A3 DA3 I4.1.9 summary is missing: {da3_summary_path}")
    if not a2_summary_path.is_file():
        raise ReconstructionValidationError(f"A3 MVS A2 summary is missing: {a2_summary_path}")

    da3 = _read_json(da3_summary_path)
    if da3.get("phase") != "I4.1.9" or da3.get("status") != "degraded":
        raise ReconstructionValidationError("A3 requires the completed DEGRADED I4.1.9 DA3 House result.")
    if da3.get("manual_visual_review_required") is not True:
        raise ReconstructionValidationError("A3 DA3 source must explicitly require manual visual review.")
    da3_authority = da3.get("authority")
    if not isinstance(da3_authority, Mapping) or da3_authority.get("single_authoritative_sfm_frame") is not True:
        raise ReconstructionValidationError("A3 DA3 source does not preserve the single authoritative SfM frame.")
    if any(bool(da3_authority.get(key)) for key in ("metric_scale_claimed", "measurement_authoritative", "certified_metrology", "surface_mesh_generated")):
        raise ReconstructionValidationError("A3 DA3 source contains forbidden authority promotion.")

    da3_artifacts = da3.get("artifacts")
    if not isinstance(da3_artifacts, Mapping):
        raise ReconstructionValidationError("A3 DA3 artifact map is missing.")
    i2_summary_path = Path(str(da3_artifacts.get("i2_summary", ""))).expanduser().resolve()
    if not i2_summary_path.is_file():
        raise ReconstructionValidationError(f"A3 DA3 I2 summary is missing: {i2_summary_path}")
    i2 = _read_json(i2_summary_path)
    if i2.get("phase") != "I2" or i2.get("manual_visual_review_required") is not True:
        raise ReconstructionValidationError("A3 requires the exact I2 manual-review DA3 diagnostics.")
    i2_artifacts = i2.get("artifacts")
    if not isinstance(i2_artifacts, Mapping):
        raise ReconstructionValidationError("A3 DA3 I2 artifact map is missing.")
    diagnostics_npz = Path(str(i2_artifacts.get("spatial_diagnostics_npz", ""))).expanduser().resolve()
    if not diagnostics_npz.is_file():
        raise ReconstructionValidationError(f"A3 DA3 spatial diagnostics are missing: {diagnostics_npz}")

    a2 = _read_json(a2_summary_path)
    if a2.get("phase") != "I4.1.9A-A2" or a2.get("status") != "pass" or int(a2.get("return_code", -1)) != 0:
        raise ReconstructionValidationError("A3 requires a successful A2 MVS execution summary.")
    if a2.get("worker", {}).get("visual_ab_pending") is not True:
        raise ReconstructionValidationError("A3 requires A2 evidence explicitly pending visual A/B review.")
    a2_authority = a2.get("authority")
    if not isinstance(a2_authority, Mapping) or a2_authority.get("single_authoritative_sfm_frame") is not True:
        raise ReconstructionValidationError("A3 MVS source does not preserve the single authoritative SfM frame.")
    forbidden_a2 = (
        "authoritative_camera_geometry_modified",
        "bundle_adjustment_rerun",
        "house_k_txt_consumed",
        "measurement_authoritative",
        "metric_scale_claimed",
        "pose_refinement",
        "sfm_rerun",
        "surface_mesh_generated",
        "certified_metrology",
    )
    if any(bool(a2_authority.get(key)) for key in forbidden_a2):
        raise ReconstructionValidationError("A3 MVS source contains forbidden camera/authority mutation.")
    a1_authority = a2.get("a1_authority")
    if not isinstance(a1_authority, Mapping) or a1_authority.get("input_sparse_unchanged") is not True:
        raise ReconstructionValidationError("A3 requires A2 proof that the A1 sparse authority stayed unchanged.")
    config = a2.get("config")
    if not isinstance(config, Mapping) or int(config.get("all_reference_views_available", -1)) != MVS_VISUAL_AB_EXPECTED_VIEWS:
        raise ReconstructionValidationError("A3 requires the 68-reference-view MVS positive control.")
    worker = a2.get("worker")
    if not isinstance(worker, Mapping):
        raise ReconstructionValidationError("A3 MVS worker evidence is missing.")
    outputs = worker.get("patch_match_outputs")
    if not isinstance(outputs, Mapping) or int(outputs.get("geometric_depth_maps", -1)) != 68 or int(outputs.get("geometric_normal_maps", -1)) != 68:
        raise ReconstructionValidationError("A3 requires complete 68-view geometric PatchMatch output.")
    fusion = worker.get("fusion")
    if not isinstance(fusion, Mapping) or int(fusion.get("ply_vertex_count", 0)) <= 0:
        raise ReconstructionValidationError("A3 requires a non-empty A2 fused MVS cloud.")
    fused_ply = Path(str(fusion.get("fused_ply", ""))).expanduser().resolve()
    if not fused_ply.is_file():
        raise ReconstructionValidationError(f"A3 fused MVS PLY is missing: {fused_ply}")

    a1_manifest_path = a2_root / "evidence/frozen_authority/a1_camera_authority_manifest.json"
    if not a1_manifest_path.is_file():
        raise ReconstructionValidationError(f"A3 frozen A1 authority manifest is missing: {a1_manifest_path}")
    a1_manifest = _read_json(a1_manifest_path)
    source_i4 = Path(str(da3.get("source", {}).get("i4_1_7_dir", ""))).expanduser().resolve()
    a1_source_i4 = Path(str(a1_manifest.get("source_i4_1_7_dir", ""))).expanduser().resolve()
    if source_i4 != a1_source_i4:
        raise ReconstructionValidationError(
            f"A3 source mismatch: DA3={source_i4} MVS/A1={a1_source_i4}; refusing an uncontrolled A/B."
        )
    source_i4_summary = source_i4 / "i4_1_7_summary.json"
    if not source_i4_summary.is_file():
        raise ReconstructionValidationError(f"A3 shared I4.1.7 source summary is missing: {source_i4_summary}")
    frozen = a1_manifest.get("frozen_evidence_sha256")
    if not isinstance(frozen, Mapping):
        raise ReconstructionValidationError("A3 A1 frozen authority hashes are missing.")
    expected_i4_hash = str(frozen.get("i4_1_7_summary", ""))
    current_i4_hash = sha256_file(source_i4_summary)
    if expected_i4_hash != current_i4_hash:
        raise ReconstructionValidationError("A3 shared I4.1.7 source summary changed after A1 authority freeze.")

    return VisualAbSources(
        da3_i4_dir=da3_root,
        da3_i4_summary=da3_summary_path,
        da3_i2_summary=i2_summary_path,
        da3_diagnostics_npz=diagnostics_npz,
        mvs_a2_dir=a2_root,
        mvs_a2_summary=a2_summary_path,
        mvs_fused_ply=fused_ply,
        a1_manifest=a1_manifest_path,
        source_i4_1_7_dir=source_i4,
        source_i4_1_7_summary=source_i4_summary,
        shared_frame=_shared_frame(i2),
        frozen_sha256={
            "da3_i4_1_9_summary": sha256_file(da3_summary_path),
            "da3_i2_summary": sha256_file(i2_summary_path),
            "da3_spatial_diagnostics_npz": sha256_file(diagnostics_npz),
            "mvs_a2_summary": sha256_file(a2_summary_path),
            "mvs_fused_ply": sha256_file(fused_ply),
            "a1_camera_authority_manifest": sha256_file(a1_manifest_path),
            "shared_i4_1_7_summary": current_i4_hash,
        },
    )


def load_da3_cloud(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    source = Path(path).expanduser().resolve()
    try:
        with np.load(source, allow_pickle=False) as payload:
            points = np.asarray(payload["points_reconstruction"], dtype=np.float64)
            colors = np.asarray(payload["colors_rgb"], dtype=np.uint8)
    except (OSError, KeyError, ValueError) as exc:
        raise ReconstructionValidationError(f"Could not read A3 DA3 diagnostic cloud {source}: {exc}") from exc
    return _validate_cloud(points, colors, label="DA3")


def _parse_ply_header(handle) -> tuple[str, int, list[tuple[str, str]]]:
    first = handle.readline()
    if first != b"ply\n" and first.rstrip(b"\r\n") != b"ply":
        raise ReconstructionValidationError("A3 MVS fused PLY has an invalid header.")
    fmt = ""
    vertex_count: int | None = None
    vertex_properties: list[tuple[str, str]] = []
    current_element = ""
    while True:
        raw = handle.readline()
        if not raw:
            raise ReconstructionValidationError("A3 MVS fused PLY header is truncated.")
        try:
            line = raw.decode("ascii").strip()
        except UnicodeDecodeError as exc:
            raise ReconstructionValidationError("A3 MVS fused PLY header must be ASCII.") from exc
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
                vertex_count = int(parts[2])
        elif parts[0] == "property" and current_element == "vertex":
            if len(parts) != 3 or parts[1] == "list":
                raise ReconstructionValidationError("A3 MVS vertex PLY properties must be scalar.")
            vertex_properties.append((parts[2], parts[1]))
    if fmt not in ("binary_little_endian", "binary_big_endian", "ascii"):
        raise ReconstructionValidationError(f"A3 MVS fused PLY format is unsupported: {fmt!r}")
    if vertex_count is None or vertex_count <= 0:
        raise ReconstructionValidationError("A3 MVS fused PLY must contain vertices.")
    if not vertex_properties:
        raise ReconstructionValidationError("A3 MVS fused PLY has no vertex properties.")
    return fmt, vertex_count, vertex_properties


def read_colmap_fused_ply(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read XYZ/RGB from a COLMAP fused PLY without adding a PLY dependency."""

    source = Path(path).expanduser().resolve()
    with source.open("rb") as handle:
        fmt, vertex_count, properties = _parse_ply_header(handle)
        names = [name for name, _ in properties]
        required = {"x", "y", "z", "red", "green", "blue"}
        if not required.issubset(names):
            missing = sorted(required.difference(names))
            raise ReconstructionValidationError(f"A3 MVS fused PLY is missing required properties: {missing}")
        for _, scalar_type in properties:
            if scalar_type not in _PLY_SCALAR_DTYPES:
                raise ReconstructionValidationError(f"A3 MVS fused PLY uses unsupported scalar type: {scalar_type}")
        if fmt == "ascii":
            rows = []
            for _ in range(vertex_count):
                line = handle.readline()
                if not line:
                    raise ReconstructionValidationError("A3 MVS fused PLY vertex payload is truncated.")
                rows.append(line.decode("ascii").split())
            if any(len(row) != len(properties) for row in rows):
                raise ReconstructionValidationError("A3 MVS fused PLY ASCII row width is invalid.")
            matrix = np.asarray(rows, dtype=np.float64)
            by_name = {name: matrix[:, idx] for idx, (name, _) in enumerate(properties)}
        else:
            endian = "<" if fmt == "binary_little_endian" else ">"
            fields = [(name, endian + _PLY_SCALAR_DTYPES[scalar_type]) for name, scalar_type in properties]
            dtype = np.dtype(fields)
            expected_bytes = dtype.itemsize * vertex_count
            raw = handle.read(expected_bytes)
            if len(raw) != expected_bytes:
                raise ReconstructionValidationError("A3 MVS fused PLY binary vertex payload is truncated.")
            data = np.frombuffer(raw, dtype=dtype, count=vertex_count)
            by_name = {name: data[name] for name, _ in properties}
        points = np.column_stack([by_name["x"], by_name["y"], by_name["z"]]).astype(np.float64, copy=False)
        colors = np.column_stack([by_name["red"], by_name["green"], by_name["blue"]]).astype(np.uint8, copy=False)
    return _validate_cloud(points, colors, label="MVS")


def _validate_cloud(points: np.ndarray, colors: np.ndarray, *, label: str) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.asarray(points, dtype=np.float64)
    rgb = np.asarray(colors, dtype=np.uint8)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) == 0:
        raise ReconstructionValidationError(f"A3 {label} point cloud must be non-empty Nx3 geometry.")
    if rgb.shape != xyz.shape:
        raise ReconstructionValidationError(f"A3 {label} RGB shape must match geometry.")
    if not np.all(np.isfinite(xyz)):
        raise ReconstructionValidationError(f"A3 {label} point cloud contains non-finite geometry.")
    return np.ascontiguousarray(xyz), np.ascontiguousarray(rgb)


def descriptive_cloud_stats(points: np.ndarray, shared_frame: SharedVisualFrame) -> dict[str, Any]:
    """Return descriptive-only spatial statistics; these are not quality gates."""

    xyz = np.asarray(points, dtype=np.float64)
    center = np.asarray(shared_frame.center_reconstruction, dtype=np.float64)
    lower = np.asarray(shared_frame.lower_reconstruction, dtype=np.float64)
    upper = np.asarray(shared_frame.upper_reconstruction, dtype=np.float64)
    q02 = np.quantile(xyz, 0.02, axis=0)
    q98 = np.quantile(xyz, 0.98, axis=0)
    robust_diagonal = float(np.linalg.norm(q98 - q02))
    full_diagonal = float(np.linalg.norm(np.max(xyz, axis=0) - np.min(xyz, axis=0)))
    radii = np.linalg.norm(xyz - center, axis=1)
    inside = np.all((xyz >= lower) & (xyz <= upper), axis=1)
    expanded_margin = 0.25 * (upper - lower)
    expanded = np.all((xyz >= lower - expanded_margin) & (xyz <= upper + expanded_margin), axis=1)
    return {
        "point_count": int(len(xyz)),
        "quantile_02": [float(v) for v in q02],
        "quantile_98": [float(v) for v in q98],
        "robust_diagonal": robust_diagonal,
        "full_diagonal": full_diagonal,
        "full_to_robust_diagonal_ratio": float(full_diagonal / max(robust_diagonal, 1e-12)),
        "radial_p95_from_shared_center": float(np.quantile(radii, 0.95)),
        "radial_p99_from_shared_center": float(np.quantile(radii, 0.99)),
        "inside_shared_core_anchor_fraction": float(np.mean(inside)),
        "inside_shared_core_anchor_plus_25pct_fraction": float(np.mean(expanded)),
        "quality_gate": False,
        "interpretation": "descriptive_only_density_and_support_differ_between_DA3_and_MVS",
    }


def build_visual_ab_summary(
    *,
    sources: VisualAbSources,
    da3_stats: Mapping[str, Any],
    mvs_stats: Mapping[str, Any],
    rendered_views: list[str],
    artifacts: Mapping[str, Any],
    point_budget: int,
) -> dict[str, Any]:
    if point_budget <= 0:
        raise ReconstructionValidationError("A3 point budget must be positive.")
    if not rendered_views or any(view not in MVS_VISUAL_AB_VIEWS for view in rendered_views):
        raise ReconstructionValidationError("A3 rendered views are invalid.")
    return {
        "schema_version": MVS_VISUAL_AB_SCHEMA_VERSION,
        "phase": "I4.1.9A-A3",
        "status": "ready_for_manual_visual_ab_review",
        "comparison": {
            "left": "existing_DA3_I4.1.9_extended_RGB_geometry",
            "right": "A2_COLMAP_MVS_geometric_stereo_fusion",
            "same_house_input": True,
            "same_authoritative_i4_1_7_source": True,
            "same_authoritative_sfm_frame": True,
            "independent_auto_fit_forbidden": True,
            "shared_camera_parameters": True,
            "rendered_views": rendered_views,
            "point_budget_per_side": int(point_budget),
            "manual_review_dimensions": [
                "recognizability",
                "surface_thickness_and_smearing",
                "radial_or_fan_streaks",
                "halo_noise",
                "ghost_geometry",
                "detached_components",
                "roof_wall_edge_definition",
                "structural_consistency",
                "completeness_vs_noise",
            ],
        },
        "shared_frame": sources.shared_frame.as_dict(),
        "source_artifacts": {
            "da3_i4_1_9_dir": str(sources.da3_i4_dir),
            "da3_spatial_diagnostics_npz": str(sources.da3_diagnostics_npz),
            "mvs_a2_dir": str(sources.mvs_a2_dir),
            "mvs_fused_ply": str(sources.mvs_fused_ply),
            "shared_i4_1_7_dir": str(sources.source_i4_1_7_dir),
        },
        "frozen_sha256": dict(sources.frozen_sha256),
        "descriptive_stats": {"da3": dict(da3_stats), "mvs": dict(mvs_stats)},
        "artifacts": dict(artifacts),
        "authority": {
            "diagnostic_visual_comparison_only": True,
            "camera_geometry_modified": False,
            "sfm_rerun": False,
            "bundle_adjustment_rerun": False,
            "pose_refinement": False,
            "da3_rerun": False,
            "mvs_rerun": False,
            "patch_match_rerun": False,
            "stereo_fusion_rerun": False,
            "house_k_txt_consumed": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
            "dense_provider_promoted": False,
        },
        "decision": {
            "automatic_winner_selected": False,
            "manual_visual_verdict_required": True,
            "decision_matrix": {
                "mvs_clean_da3_smeared": "isolate_primary_problem_to_learned_depth_alignment_fusion_branch_and_consider_MVS_for_static_orbits",
                "mvs_similarly_smeared": "investigate_camera_intrinsics_projection_frame_transform_assumptions_before_dense_threshold_tuning",
                "mvs_better_but_imperfect": "characterize_residual_MVS_errors_and_promote_only_according_to_evidence",
            },
        },
    }

#!/usr/bin/env python3
"""I4.2E: controlled TempleRing d12 thin-structure/detail qualification."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction.errors import ReconstructionValidationError  # noqa: E402
from core.reconstruction.temple_ring_detail_qualification import (  # noqa: E402
    I42E_PHASE,
    TempleRingDetailTransferPolicy,
    build_production_scale_mvs_payload,
    json_safe_qualification,
    qualify_d12_detail_transfer,
    validate_fusion_visibility_evidence,
)

_DEFAULT_D = Path("/tmp/depth3d_i4_2_temple_ring/i4_2d_mvs_attempt01")
_DEFAULT_OUT = Path("/tmp/depth3d_i4_2_temple_ring/i4_2e_detail_qualification_attempt01")
_DEFAULT_PYCOLMAP = Path.home() / "anaconda3/envs/depth3d-pycolmap-probe/bin/python3.11"
_DEFAULT_WORKER = ROOT / "scripts/reconstruction_pycolmap_temple_ring_detail_support_worker.py"
_SPARSE_FILES = ("cameras.bin", "images.bin", "points3D.bin", "frames.bin", "rigs.bin")


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--i4-2d-dir", type=Path, default=_DEFAULT_D)
    p.add_argument("--output-dir", type=Path, default=_DEFAULT_OUT)
    p.add_argument("--python", type=Path, default=_DEFAULT_PYCOLMAP)
    p.add_argument("--worker", type=Path, default=_DEFAULT_WORKER)
    p.add_argument("--worker-timeout-s", type=float, default=900.0)
    return p.parse_args()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError(f"Could not read I4.2E {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReconstructionValidationError(f"I4.2E {label} must be a JSON object: {path}")
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return path


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ReconstructionValidationError(f"Could not hash I4.2E artifact {path}: {exc}") from exc
    return digest.hexdigest()


def _clean_output(path: Path) -> Path:
    out = path.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise ReconstructionValidationError(
            f"I4.2E output must be empty or absent; refusing to overwrite detail evidence: {out}"
        )
    return out


def _write_binary_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> Path:
    xyz = np.asarray(points, dtype=np.float64)
    rgb = np.asarray(colors, dtype=np.uint8)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or rgb.shape != xyz.shape or not np.isfinite(xyz).all():
        raise ReconstructionValidationError("I4.2E PLY requires finite aligned Nx3 XYZ/RGB arrays.")
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(xyz)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    ).encode("ascii")
    dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("r", "u1"), ("g", "u1"), ("b", "u1"),
    ])
    rows = np.empty(len(xyz), dtype=dtype)
    if len(xyz):
        rows["x"], rows["y"], rows["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        rows["r"], rows["g"], rows["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with path.open("wb") as handle:
        handle.write(header)
        handle.write(rows.tobytes())
    return path


def _validate_d(i4_2d_dir: Path) -> tuple[dict[str, Any], dict[str, Any], Path, Path, dict[str, str]]:
    root = i4_2d_dir.expanduser().resolve()
    summary_path = root / "i4_2d_summary.json"
    result_path = root / "worker/result.json"
    summary = _read_json(summary_path, label="I4.2D summary")
    result = _read_json(result_path, label="I4.2D worker result")
    if summary.get("phase") != "I4.2D" or summary.get("status") != "pass" or summary.get("pass") is not True:
        raise ReconstructionValidationError("I4.2E requires I4.2D PASS.")
    if result.get("phase") != "I4.2D" or result.get("status") != "pass":
        raise ReconstructionValidationError("I4.2E requires the passing I4.2D worker result.")
    authority = summary.get("authority")
    if not isinstance(authority, Mapping):
        raise ReconstructionValidationError("I4.2D authority evidence is missing.")
    required_true = ("mvs_started", "raw_mvs_dense_reconstruction", "single_authoritative_sfm_frame")
    if any(authority.get(name) is not True for name in required_true):
        raise ReconstructionValidationError("I4.2E requires completed raw MVS in the single authoritative frame.")
    forbidden_true = (
        "authoritative_camera_geometry_modified", "bundle_adjustment_rerun", "certified_metrology",
        "da3_started", "dense_provider_promoted", "held_out_ground_truth_consumed_by_i4_2d",
        "i4_3_provider_architecture_authorized", "measurement_authoritative", "metric_scale_claimed",
        "pose_refinement", "production_filter_policy_changed", "production_manifold_applied",
        "sfm_rerun", "surface_mesh_generated",
    )
    if any(bool(authority.get(name)) for name in forbidden_true):
        raise ReconstructionValidationError("I4.2E refuses I4.2D evidence with mutated authority/policy state.")
    if result.get("detail_qualification_pending") is not True:
        raise ReconstructionValidationError("I4.2D must leave detail qualification pending for I4.2E.")
    patch = result.get("patch_match_outputs")
    fusion = result.get("fusion")
    if not isinstance(patch, Mapping) or int(patch.get("geometric_depth_maps", -1)) != 47 or int(patch.get("geometric_normal_maps", -1)) != 47:
        raise ReconstructionValidationError("I4.2E requires complete 47-view geometric PatchMatch evidence.")
    if not isinstance(fusion, Mapping) or int(fusion.get("ply_vertex_count", 0)) <= 0:
        raise ReconstructionValidationError("I4.2E requires a non-empty raw I4.2D fused cloud.")
    raw_ply = Path(str(fusion.get("raw_fused_ply", ""))).expanduser().resolve()
    dense = Path(str(summary.get("artifacts", {}).get("dense_workspace", ""))).expanduser().resolve()
    if not raw_ply.is_file() or not dense.is_dir():
        raise ReconstructionValidationError("I4.2E raw I4.2D dense artifacts are missing.")
    if raw_ply.stat().st_size != int(fusion.get("fused_ply_bytes", -1)):
        raise ReconstructionValidationError("I4.2E raw fused PLY byte size changed after I4.2D.")

    frozen_b_path = Path(str(summary.get("artifacts", {}).get("frozen_i4_2b_summary", ""))).resolve()
    b_summary = _read_json(frozen_b_path, label="frozen I4.2B summary")
    sparse = Path(str(b_summary.get("artifacts", {}).get("authoritative_sparse", ""))).expanduser().resolve()
    expected_hashes = summary.get("input_authority", {}).get("authoritative_sparse_sha256_after")
    if not isinstance(expected_hashes, Mapping):
        raise ReconstructionValidationError("I4.2E I4.2D sparse checksum evidence is missing.")
    current: dict[str, str] = {}
    for filename in _SPARSE_FILES:
        path = sparse / filename
        if not path.is_file():
            raise ReconstructionValidationError(f"I4.2E authoritative sparse file is missing: {filename}")
        current[filename] = _sha(path)
    if current != dict(expected_hashes):
        raise ReconstructionValidationError("I4.2E authoritative sparse model changed after I4.2D.")
    return summary, result, raw_ply, dense, current


def _request(
    *,
    raw_ply: Path,
    native_visibility: Path,
    d_result: Mapping[str, Any],
    output: Path,
) -> dict[str, Any]:
    fusion = d_result["fusion"]
    return {
        "schema_version": 2,
        "phase": I42E_PHASE,
        "action": "temple_ring_export_native_fusion_visibility",
        "raw_fused_ply": str(raw_ply),
        "native_visibility_path": str(native_visibility),
        "support_npz": str(output / "support_visibility/i4_2e_fusion_support.npz"),
        "expected_raw_ply_bytes": int(fusion["fused_ply_bytes"]),
        "expected_raw_vertices": int(fusion["ply_vertex_count"]),
        "expected_views": 47,
        "producer_pycolmap_version": str(d_result.get("pycolmap_version", "")),
        "authority": {
            "stereo_fusion_replay": False,
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
        },
    }


def main() -> int:
    args = _args()
    output = _clean_output(args.output_dir)
    try:
        d_summary, d_result, raw_ply, dense, sparse_before = _validate_d(args.i4_2d_dir)
        native_visibility = Path(f"{raw_ply}.vis")
        if not native_visibility.is_file() or native_visibility.stat().st_size <= 8:
            raise ReconstructionValidationError(
                "I4.2E.2 requires the native I4.2D fused.ply.vis sidecar emitted by COLMAP Stereo Fusion."
            )
        raw_ply_sha_before = _sha(raw_ply)
        native_visibility_sha_before = _sha(native_visibility)
        python = args.python.expanduser().resolve()
        worker = args.worker.expanduser().resolve()
        if not python.is_file() or not worker.is_file():
            raise ReconstructionValidationError("I4.2E isolated Python/worker path is missing.")
        if float(args.worker_timeout_s) <= 0.0:
            raise ReconstructionValidationError("I4.2E worker timeout must be positive.")

        output.mkdir(parents=True, exist_ok=True)
        evidence_dir = output / "evidence"
        visibility_dir = output / "support_visibility"
        artifacts_dir = output / "artifacts"
        worker_dir = output / "worker"
        for path in (evidence_dir, visibility_dir, artifacts_dir, worker_dir):
            path.mkdir(parents=True, exist_ok=False)
        frozen_d = shutil.copy2(Path(args.i4_2d_dir).resolve() / "i4_2d_summary.json", evidence_dir / "i4_2d_summary.json")
        frozen_d_result = shutil.copy2(Path(args.i4_2d_dir).resolve() / "worker/result.json", evidence_dir / "i4_2d_worker_result.json")

        request = _request(
            raw_ply=raw_ply,
            native_visibility=native_visibility,
            d_result=d_result,
            output=output,
        )
        request["frozen_evidence_sha256"] = {
            "i4_2d_summary": _sha(frozen_d),
            "i4_2d_worker_result": _sha(frozen_d_result),
        }
        request_path = _write_json(worker_dir / "request.json", request)
        result_path = worker_dir / "result.json"
        completed = subprocess.run(
            [str(python), str(worker), "--request", str(request_path), "--result", str(result_path)],
            check=False,
            timeout=float(args.worker_timeout_s),
        )
        if int(completed.returncode) != 0:
            worker_result = _read_json(result_path, label="I4.2E support worker result") if result_path.is_file() else {}
            raise ReconstructionValidationError(
                f"I4.2E native visibility worker failed with code {completed.returncode}: {worker_result.get('error', '')}"
            )
        worker_result = _read_json(result_path, label="I4.2E support worker result")
        if worker_result.get("phase") != I42E_PHASE or worker_result.get("status") != "pass":
            raise ReconstructionValidationError("I4.2E native visibility extraction did not return PASS.")
        if worker_result.get("native_visibility_consumed") is not True:
            raise ReconstructionValidationError("I4.2E.2 did not consume the native I4.2D visibility sidecar.")
        if worker_result.get("stereo_fusion_replayed_for_support_evidence") is not False:
            raise ReconstructionValidationError("I4.2E.2 must not replay Stereo Fusion.")
        if worker_result.get("native_visibility_indices_within_expected_range") is not True:
            raise ReconstructionValidationError("I4.2E native visibility contains an invalid MVS image index.")
        observed_indices_raw = worker_result.get("native_visibility_observed_image_indices")
        if not isinstance(observed_indices_raw, list) or not observed_indices_raw:
            raise ReconstructionValidationError("I4.2E native visibility must report at least one observed MVS image index.")
        try:
            observed_indices = [int(value) for value in observed_indices_raw]
        except (TypeError, ValueError) as exc:
            raise ReconstructionValidationError("I4.2E native visibility reported malformed MVS image indices.") from exc
        if observed_indices != sorted(set(observed_indices)):
            raise ReconstructionValidationError("I4.2E native visibility observed image indices must be sorted and unique.")
        if observed_indices[0] < 0 or observed_indices[-1] >= 47:
            raise ReconstructionValidationError("I4.2E native visibility observed image indices lie outside 0..46.")
        if int(worker_result.get("native_visibility_unique_image_indices", -1)) != len(observed_indices):
            raise ReconstructionValidationError("I4.2E native visibility unique-image diagnostic is inconsistent.")
        if worker_result.get("native_visibility_full_view_coverage_required") is not False:
            raise ReconstructionValidationError("I4.2E.2a must not require every campaign view to survive in fused visibility.")
        if str(worker_result.get("raw_fused_ply_sha256", "")) != raw_ply_sha_before:
            raise ReconstructionValidationError("I4.2E raw fused PLY changed before/during visibility extraction.")
        if str(worker_result.get("native_visibility_sha256", "")) != native_visibility_sha_before:
            raise ReconstructionValidationError("I4.2E native visibility changed before/during extraction.")
        for name in (
            "stereo_fusion_replayed_for_support_evidence", "patch_match_rerun", "sfm_rerun", "bundle_adjustment_rerun", "pose_refinement",
            "held_out_ground_truth_consumed", "production_filter_policy_changed",
            "production_manifold_applied_to_authoritative_scene", "metric_scale_claimed",
            "measurement_authoritative", "certified_metrology", "surface_mesh_generated",
            "dense_provider_promoted", "i4_3_provider_architecture_authorized",
        ):
            if bool(worker_result.get(name)):
                raise ReconstructionValidationError(f"I4.2E support worker violated authority flag {name}=True.")

        support_npz = Path(str(worker_result["support_npz"])).resolve()
        with np.load(support_npz, allow_pickle=False) as payload_npz:
            points = np.asarray(payload_npz["points_reconstruction"], dtype=np.float64)
            colors = np.asarray(payload_npz["colors_rgb"], dtype=np.uint8)
            offsets = np.asarray(payload_npz["visibility_offsets"], dtype=np.int64)
            image_indices = np.asarray(payload_npz["visibility_image_indices"], dtype=np.int64)
        validate_fusion_visibility_evidence(points, colors, offsets, image_indices)
        transfer = build_production_scale_mvs_payload(
            points, colors, offsets, image_indices, policy=TempleRingDetailTransferPolicy()
        )
        qualification = qualify_d12_detail_transfer(transfer)
        arrays = qualification["arrays"]
        pre_points = np.asarray(arrays["pre_points"], dtype=np.float64)
        pre_colors = np.asarray(arrays["colors"], dtype=np.uint8)
        reject = np.asarray(arrays["reject"], dtype=bool)

        pre_ply = _write_binary_ply(artifacts_dir / "i4_2e_pre_d12_extended.ply", pre_points, pre_colors)
        post_ply = _write_binary_ply(artifacts_dir / "i4_2e_post_d12_extended.ply", pre_points[~reject], pre_colors[~reject])
        rejected_ply = _write_binary_ply(artifacts_dir / "i4_2e_d12_rejected_only.ply", pre_points[reject], pre_colors[reject])
        analysis_npz = artifacts_dir / "i4_2e_detail_analysis.npz"
        np.savez_compressed(
            analysis_npz,
            points_reconstruction=pre_points,
            colors_rgb=pre_colors,
            d12_reject_mask=reject,
            core_mask=np.asarray(arrays["core"], dtype=bool),
            high_support_mask=np.asarray(arrays["high_support"], dtype=bool),
            supported_line_like_proxy=np.asarray(arrays["line_like_proxy"], dtype=bool),
            supported_line_like_sparse_proxy=np.asarray(arrays["line_like_sparse_proxy"], dtype=bool),
            voxel_size_reconstruction_units=np.asarray(transfer["voxel_size_reconstruction_units"], dtype=np.float64),
        )

        sparse_after = {}
        frozen_b = _read_json(Path(str(d_summary["artifacts"]["frozen_i4_2b_summary"])), label="I4.2B summary after")
        sparse = Path(str(frozen_b["artifacts"]["authoritative_sparse"])).resolve()
        for filename in _SPARSE_FILES:
            sparse_after[filename] = _sha(sparse / filename)
        if sparse_after != sparse_before:
            raise ReconstructionValidationError("I4.2E changed the authoritative sparse model.")
        raw_ply_sha_after = _sha(raw_ply)
        native_visibility_sha_after = _sha(native_visibility)
        if raw_ply_sha_after != raw_ply_sha_before:
            raise ReconstructionValidationError("I4.2E changed the frozen I4.2D fused.ply.")
        if native_visibility_sha_after != native_visibility_sha_before:
            raise ReconstructionValidationError("I4.2E changed the frozen I4.2D fused.ply.vis.")

        persisted = json_safe_qualification(qualification)
        summary = {
            "schema_version": 1,
            "phase": I42E_PHASE,
            "status": "ready_for_manual_review",
            "pass": True,
            "execution_integrity_pass": True,
            "dataset": {"identity": "Middlebury_MVS_TempleRing", "views": 47},
            "source": {
                "i4_2d_dir": str(Path(args.i4_2d_dir).resolve()),
                "raw_fused_ply": str(raw_ply),
                "raw_fused_ply_sha256_before": raw_ply_sha_before,
                "raw_fused_ply_sha256_after": raw_ply_sha_after,
                "raw_fused_points": int(d_result["fusion"]["ply_vertex_count"]),
                "native_visibility_path": str(native_visibility),
                "native_visibility_sha256_before": native_visibility_sha_before,
                "native_visibility_sha256_after": native_visibility_sha_after,
                "native_visibility_consumed": True,
                "visibility_observations": int(worker_result["visibility_observations"]),
                "visibility_support_min": int(worker_result["visibility_support_min"]),
                "visibility_support_median": float(worker_result["visibility_support_median"]),
                "visibility_support_max": int(worker_result["visibility_support_max"]),
                "native_visibility_expected_view_count": 47,
                "native_visibility_unique_image_indices": len(observed_indices),
                "native_visibility_observed_image_indices": observed_indices,
                "native_visibility_exact_image_index_set": bool(
                    worker_result.get("native_visibility_exact_image_index_set", False)
                ),
                "native_visibility_full_view_coverage_required": False,
                "visibility_indices_are_mvs_indices_not_colmap_image_ids": True,
            },
            "transfer_evidence": dict(transfer["diagnostics"]),
            "qualification": persisted,
            "artifacts": {
                "support_npz": str(support_npz),
                "analysis_npz": str(analysis_npz),
                "pre_d12_extended_ply": str(pre_ply),
                "post_d12_extended_ply": str(post_ply),
                "d12_rejected_only_ply": str(rejected_ply),
                "native_visibility": str(native_visibility),
                "worker_request": str(request_path),
                "worker_result": str(result_path),
            },
            "artifact_sha256": {
                "support_npz": _sha(support_npz),
                "analysis_npz": _sha(analysis_npz),
                "pre_d12_extended_ply": _sha(pre_ply),
                "post_d12_extended_ply": _sha(post_ply),
                "d12_rejected_only_ply": _sha(rejected_ply),
            },
            "authority": {
                "single_authoritative_sfm_frame_preserved": True,
                "authoritative_sparse_modified": False,
                "sfm_rerun": False,
                "bundle_adjustment_rerun": False,
                "pose_refinement": False,
                "patch_match_rerun": False,
                "stereo_fusion_replayed_for_support_evidence_only": False,
                "native_i4_2d_fusion_visibility_consumed_for_support_evidence": True,
                "held_out_ground_truth_consumed_by_filtering": False,
                "production_filter_policy_changed": False,
                "production_manifold_applied_to_authoritative_scene": False,
                "d12_counterfactual_only": True,
                "dense_provider_promoted": False,
                "i4_3_provider_architecture_authorized": False,
                "metric_scale_claimed": False,
                "measurement_authoritative": False,
                "certified_metrology": False,
                "surface_mesh_generated": False,
            },
            "authoritative_sparse_sha256_before": sparse_before,
            "authoritative_sparse_sha256_after": sparse_after,
            "manual_review_contract": {
                "required": True,
                "automatic_winner_selected": False,
                "review_dimensions": [
                    "thin_vertical_member_separation",
                    "small_gap_and_opening_preservation",
                    "edge_definition",
                    "small_protrusion_preservation",
                    "surface_thickness_and_false_bridging",
                    "local_completeness_vs_noise",
                    "production_manifold_detail_loss",
                ],
                "interpretation": (
                    "Review matched pre-d12 vs post-d12 renders and the rejected-only overlay. "
                    "The line-like/sparse populations are descriptive risk proxies, not labels of legitimate detail."
                ),
            },
            "next": (
                "Render I4.2E matched detail previews and complete manual review. "
                "I4.2F may close I4.2 only after the d12 transfer verdict is explicitly recorded."
            ),
        }
        summary_path = _write_json(output / "i4_2e_summary.json", summary)
        print("I4.2E TempleRing d12 detail qualification: READY FOR MANUAL REVIEW")
        print(
            f"raw_mvs_points={summary['source']['raw_fused_points']} "
            f"pre_d12_extended={persisted['counts']['pre_d12_extended_voxels']} "
            f"rejected={persisted['counts']['d12_rejected_voxels']} "
            f"rejected_fraction={persisted['counts']['d12_rejected_fraction']:.6f}"
        )
        print(
            f"core_rejected={persisted['populations']['core_analogue']['rejected']} "
            f"supported_line_like_rejected={persisted['populations']['supported_line_like_proxy']['rejected']} "
            f"potential_detail_loss_signal={persisted['potential_detail_loss_signal']}"
        )
        print("production_policy_changed=False authoritative_sparse_modified=False i4_3_authorized=False")
        print(f"summary={summary_path}")
        return 0
    except subprocess.TimeoutExpired:
        print("I4.2E TempleRing detail qualification: FAIL — native visibility extraction timed out", file=sys.stderr)
        return 3
    except ReconstructionValidationError as exc:
        print(f"I4.2E TempleRing detail qualification: FAIL — {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())

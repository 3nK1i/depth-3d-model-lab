#!/usr/bin/env python3
"""I3.2 promote the reviewed second-scene d12 candidate and requalify it.

Consumes the completed I3.1 campaign.  The exact reviewed candidate is promoted
into formal F, a fresh portable G scene is regenerated, and baseline-relative
topology/spatial safety is requalified.  SfM, DA3, alignment and fusion are not
rerun.  Metric, measurement and mesh authority remain unpromoted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction.dense_confidence_filtering import DenseArtifactReason, artifact_reason_histogram  # noqa: E402
from core.reconstruction.dense_scene_persistence import (  # noqa: E402
    persist_filtered_dense_rgb_scene,
    read_persistent_dense_rgb_scene,
    restore_persistent_dense_geometry,
    verify_persistent_dense_scene_files,
)
from core.reconstruction.errors import ReconstructionValidationError  # noqa: E402
from core.reconstruction.offline_dense_detached_component_qualification import compare_component_lineage  # noqa: E402
from core.reconstruction.offline_dense_i3_promotion import (  # noqa: E402
    I3_2_SCHEMA_VERSION,
    build_i3_2_requalification,
    validate_exact_i3_1_candidate,
    validate_i3_1_promotion_evidence,
)
from core.reconstruction.offline_dense_i3_transfer import build_scene_relative_manifold_evidence, exact_reference_transfer_preflight  # noqa: E402
from scripts.run_e2_consistency_aware_fusion_refinement import _quantiles, _write_binary_ply  # noqa: E402

_DEFAULT_I3_1 = Path("/tmp/eia_i3_second_scene_campaign")
_DEFAULT_OUT = Path("/tmp/eia_i3_2_second_scene_promotion")

_POINT_FIELDS = (
    "points_reconstruction", "colors_rgb", "view_observation_counts", "filter_confidence",
    "median_dispersion_voxels", "p90_dispersion_voxels", "maximum_dispersion_voxels",
    "neighbor26_count", "radius2_neighbor_count", "high_support_neighbor_count",
    "component_size", "class_code", "artifact_reason_code",
)
_META_FIELDS = (
    "voxel_size_reconstruction_units", "selected_workspace_orders", "selected_keyframe_ids",
    "sparse_landmarks_reconstruction", "sparse_landmark_colors_rgb", "schema_version",
)


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--i3-1-dir", type=Path, default=_DEFAULT_I3_1)
    p.add_argument("--output-dir", type=Path, default=_DEFAULT_OUT)
    p.add_argument(
        "--confirm-manual-transfer-review-pass",
        action="store_true",
        help="Required explicit acknowledgement of the completed I3.1 baseline/candidate/decision visual review.",
    )
    return p.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError(f"Unable to read I3.2 JSON evidence: {path}") from exc
    if not isinstance(value, dict):
        raise ReconstructionValidationError(f"I3.2 JSON root must be an object: {path}")
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return path


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise ReconstructionValidationError(f"I3.2 required NPZ is missing: {path}")
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def _validate_filtered(data: Mapping[str, np.ndarray], *, label: str) -> int:
    missing = sorted(set(_POINT_FIELDS + _META_FIELDS).difference(data))
    if missing:
        raise ReconstructionValidationError(f"I3.2 {label} payload is missing fields: {missing!r}")
    points = np.asarray(data["points_reconstruction"], dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ReconstructionValidationError(f"I3.2 {label} geometry is invalid.")
    for name in _POINT_FIELDS:
        values = np.asarray(data[name])
        expected = points.shape if name in ("points_reconstruction", "colors_rgb") else (len(points),)
        if values.shape != expected:
            raise ReconstructionValidationError(f"I3.2 {label} {name} shape is invalid.")
    return int(len(points))


def _slice(data: Mapping[str, np.ndarray], mask: np.ndarray, *, class_code: int | None = None) -> dict[str, np.ndarray]:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if len(mask) != len(np.asarray(data["points_reconstruction"])):
        raise ReconstructionValidationError("I3.2 slice mask is misaligned.")
    out: dict[str, np.ndarray] = {}
    for name in _POINT_FIELDS:
        sliced = np.asarray(data[name])[mask].copy()
        if name == "artifact_reason_code":
            sliced = np.zeros(len(sliced), dtype=np.uint16)
        elif name == "class_code" and class_code is not None:
            sliced = np.full(len(sliced), int(class_code), dtype=np.uint8)
        out[name] = sliced
    for name in _META_FIELDS:
        out[name] = np.asarray(data[name]).copy()
    return out


def _append_rejected(source_rejected: Mapping[str, np.ndarray], source_extended: Mapping[str, np.ndarray], reject: np.ndarray) -> dict[str, np.ndarray]:
    reject = np.asarray(reject, dtype=bool).reshape(-1)
    out: dict[str, np.ndarray] = {}
    for name in _POINT_FIELDS:
        old = np.asarray(source_rejected[name])
        new = np.asarray(source_extended[name])[reject].copy()
        if name == "artifact_reason_code":
            new = np.full(len(new), int(DenseArtifactReason.WEAK_MANIFOLD_ATTACHMENT), dtype=np.uint16)
        elif name == "class_code":
            new = np.zeros(len(new), dtype=np.uint8)
        out[name] = np.concatenate([old, new], axis=0)
    for name in _META_FIELDS:
        out[name] = np.asarray(source_extended[name]).copy()
    return out


def _updated_coverage(source_coverage: Mapping[str, np.ndarray], source_extended: Mapping[str, np.ndarray], keep: np.ndarray) -> dict[str, np.ndarray]:
    source_count = len(np.asarray(source_extended["points_reconstruction"]))
    coverage_points = np.asarray(source_coverage["points_reconstruction"], dtype=np.float64)
    source_points = np.asarray(source_extended["points_reconstruction"], dtype=np.float64)
    if len(coverage_points) < source_count or not np.array_equal(coverage_points[:source_count], source_points):
        raise ReconstructionValidationError("I3.2 source coverage-candidate ordering is not Extended+fringe.")
    out: dict[str, np.ndarray] = {}
    for name in _POINT_FIELDS:
        values = np.asarray(source_coverage[name])
        head = values[:source_count][keep].copy()
        if name == "artifact_reason_code":
            head = np.zeros(len(head), dtype=np.uint16)
        out[name] = np.concatenate([head, values[source_count:]], axis=0)
    for name in _META_FIELDS:
        out[name] = np.asarray(source_extended[name]).copy()
    return out


def _write_npz(path: Path, data: Mapping[str, np.ndarray]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{name: np.asarray(value) for name, value in data.items()})
    return path


def _update_f_summary(
    source: Mapping[str, Any], *, source_summary_path: Path, output_dir: Path,
    promoted_extended: Mapping[str, np.ndarray], promoted_core: Mapping[str, np.ndarray],
    promoted_rejected: Mapping[str, np.ndarray], artifact_paths: Mapping[str, Path],
    i31: Mapping[str, Any], i31_summary_path: Path, rebuilt: Mapping[str, Any], exact_candidate: Mapping[str, Any],
) -> dict[str, Any]:
    summary = json.loads(json.dumps(dict(source)))
    classification = dict(summary["classification"])
    source_supported = int(classification["source_supported_voxels"])
    ext_count = len(np.asarray(promoted_extended["points_reconstruction"]))
    core_count = len(np.asarray(promoted_core["points_reconstruction"]))
    rejected_count = len(np.asarray(promoted_rejected["points_reconstruction"]))
    if ext_count + rejected_count != source_supported:
        raise ReconstructionValidationError("I3.2 promoted F supported-voxel accounting is not exact.")
    reasons = artifact_reason_histogram(np.asarray(promoted_rejected["artifact_reason_code"], dtype=np.uint16))
    reasons["single_view_detached"] = int(classification["detached_single_view_artifact_voxels"])
    classification.update({
        "core_voxels": core_count,
        "extended_voxels": ext_count,
        "rejected_supported_voxels": rejected_count,
        "core_fraction_of_supported": core_count / float(source_supported),
        "extended_fraction_of_supported": ext_count / float(source_supported),
        "artifact_reason_histogram": reasons,
    })
    summary["classification"] = classification
    summary["confidence"] = {
        **dict(summary.get("confidence", {})),
        "core_quantiles": _quantiles(np.asarray(promoted_core["filter_confidence"], dtype=np.float64)),
        "extended_quantiles": _quantiles(np.asarray(promoted_extended["filter_confidence"], dtype=np.float64)),
        "rejected_quantiles": _quantiles(np.asarray(promoted_rejected["filter_confidence"], dtype=np.float64)),
    }
    policy = dict(summary.get("policy", {}))
    policy["manifold_attachment_refinement"] = dict(rebuilt["resolved_rule"])
    policy["manifold_attachment_artifact_reason"] = "weak_manifold_attachment"
    policy["manifold_attachment_promotion_phase"] = "I3.2"
    policy["manifold_attachment_reference_scope"] = "scene-relative second-scene central Core non-driver q90"
    summary["policy"] = policy
    source_conditions = dict(summary.get("pass_conditions", {}))
    candidate = i31["candidate"]
    gates = candidate["gates"]
    promotion_conditions = {
        "i3_1_manual_transfer_review_confirmed": True,
        "i3_1_transfer_safety_gates_passed": bool(all(bool(v) for v in gates.values())),
        "exact_i3_1_candidate_equivalence": bool(exact_candidate["exact_candidate_equivalence"]),
        "promoted_core_fraction_gate": classification["core_fraction_of_supported"] >= float(policy.get("minimum_core_fraction_gate", 0.0)),
        "promoted_extended_fraction_gate": classification["extended_fraction_of_supported"] >= float(policy.get("minimum_extended_fraction_gate", 0.0)),
        "second_scene_full_to_robust_not_materially_worse": bool(gates["full_to_robust_not_materially_worse"]),
        "second_scene_topology_preserved": bool(gates["dominant_lineage_preserved"] and gates["no_newly_detached_voxels"] and gates["no_baseline_dominant_removal"]),
        "metric_measurement_mesh_authority_preserved": True,
    }
    summary["pass_conditions"] = {**source_conditions, **promotion_conditions}
    summary["pass"] = bool(all(summary["pass_conditions"].values()))
    summary["status"] = "pass" if summary["pass"] else "degraded"
    summary["promotion"] = {
        "phase": "I3.2",
        "schema_version": I3_2_SCHEMA_VERSION,
        "source_baseline_f_summary": str(source_summary_path.resolve()),
        "source_i3_1_summary": str(i31_summary_path.resolve()),
        "manual_transfer_review": "pass",
        "exact_i3_1_candidate_equivalence": True,
        "scene_relative_reference_thresholds": dict(rebuilt["reference_thresholds"]),
        "resolved_rule": dict(rebuilt["resolved_rule"]),
        "candidate_safety": dict(candidate),
        "production_filter_policy_changed": True,
        "sfm_rerun": False, "da3_rerun": False, "fusion_rerun": False,
    }
    summary["decision"] = {
        **dict(summary.get("decision", {})),
        "authorizes_next_canonical_dense_persistence_milestone": bool(summary["pass"]),
        "next_milestone": "G - Persistent Canonical Dense RGB Geometry" if summary["pass"] else None,
        "primary_output": "f_filtered_extended.npz",
        "high_confidence_output": "f_filtered_core.npz",
    }
    authority = dict(summary.get("authority", {}))
    authority.update({
        "filtered_dense_geometry_qualified": bool(summary["pass"]),
        "confidence_artifact_filtering_formalized": True,
        "canonical_geometry_authoritative": False,
        "canonical_persistence_promoted": False,
        "metric_scale_claimed": False,
        "measurement_authoritative": False,
        "certified_metrology": False,
        "surface_mesh_generated": False,
        "note": "I3.2 promotes the exact manually reviewed second-scene d12_sparse candidate into formal F. Scale remains arbitrary; metric, measurement and mesh authority remain unpromoted.",
    })
    summary["authority"] = authority
    summary["artifacts"] = {name: str(path.resolve()) for name, path in artifact_paths.items()}
    summary["i3_2_output_dir"] = str(output_dir.resolve())
    return summary


def main() -> int:
    args = _args()
    i31_dir = args.i3_1_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ReconstructionValidationError(f"I3.2 output directory must be empty or absent: {output}")
    output.mkdir(parents=True, exist_ok=True)

    i31 = _read_json(i31_dir / "i3_1_summary.json")
    promotion_preflight = validate_i3_1_promotion_evidence(
        i31, manual_transfer_review_pass=bool(args.confirm_manual_transfer_review_pass)
    )
    print("I3.2 reviewed transfer preflight: PASS")
    print("manual_transfer_review=pass transfer_safety_gates=pass exact_i2_reference_preflight=pass")

    i1_dir = Path(str(i31["source_i1_dir"])).expanduser().resolve()
    source_f_dir = i1_dir / "05_filtering"
    source_f_path = source_f_dir / "f_summary.json"
    source_f = _read_json(source_f_path)
    if str(source_f.get("phase")) != "F" or not bool(source_f.get("pass")):
        raise ReconstructionValidationError("I3.2 requires the passing second-scene baseline formal F product.")
    f_art = source_f["artifacts"]
    source_ext = _load_npz(Path(str(f_art["extended_npz"])).expanduser().resolve())
    source_core = _load_npz(Path(str(f_art["core_npz"])).expanduser().resolve())
    source_rejected = _load_npz(Path(str(f_art["rejected_supported_npz"])).expanduser().resolve())
    source_coverage = _load_npz(Path(str(f_art["coverage_candidate_npz"])).expanduser().resolve())
    _validate_filtered(source_ext, label="source Extended")
    _validate_filtered(source_core, label="source Core")
    _validate_filtered(source_rejected, label="source rejected-supported")
    _validate_filtered(source_coverage, label="source coverage-candidate")

    diag_path = Path(str(i31["artifacts"]["diagnostics_npz"])).expanduser().resolve()
    cand_ext_path = Path(str(i31["artifacts"]["candidate_extended_npz"])).expanduser().resolve()
    cand_core_path = Path(str(i31["artifacts"]["candidate_core_npz"])).expanduser().resolve()
    diag = _load_npz(diag_path)
    cand_ext = _load_npz(cand_ext_path)
    cand_core = _load_npz(cand_core_path)
    exact_candidate = validate_exact_i3_1_candidate(source_ext, diag, cand_ext, cand_core, i31)

    rebuilt = build_scene_relative_manifold_evidence(source_ext)
    refs = i31["transfer_evidence"]["reference_thresholds"]
    rebuilt_preflight = exact_reference_transfer_preflight(
        rebuilt,
        expected_reject_mask=diag["is_rejected_by_d12_transfer"],
        expected_anchor_reference_q90=float(refs["anchor_distance_q90_voxels"]),
        expected_knn_reference_q90=float(refs["knn_spacing_q90_voxels"]),
    )
    if not bool(rebuilt_preflight["pass"]):
        raise ReconstructionValidationError("I3.2 rebuilt second-scene manifold predicate does not exactly reproduce I3.1.")
    reject = np.asarray(exact_candidate["reject_mask"], dtype=bool)
    is_core = np.asarray(exact_candidate["is_core"], dtype=bool)
    keep = ~reject

    promoted_ext = _slice(source_ext, keep)
    promoted_core = _slice(source_ext, is_core & keep, class_code=3)
    promoted_rejected = _append_rejected(source_rejected, source_ext, reject)
    promoted_coverage = _updated_coverage(source_coverage, source_ext, keep)

    formal = output / "01_promoted_filtering"
    formal.mkdir(parents=True, exist_ok=True)
    ext_npz = _write_npz(formal / "f_filtered_extended.npz", promoted_ext)
    core_npz = _write_npz(formal / "f_filtered_core.npz", promoted_core)
    rej_npz = _write_npz(formal / "f_rejected_supported.npz", promoted_rejected)
    cov_npz = _write_npz(formal / "f_coverage_candidate_diagnostic.npz", promoted_coverage)
    ext_ply = _write_binary_ply(formal / "f_filtered_extended.ply", np.asarray(promoted_ext["points_reconstruction"], dtype=np.float64), np.asarray(promoted_ext["colors_rgb"], dtype=np.uint8))
    core_ply = _write_binary_ply(formal / "f_filtered_core.ply", np.asarray(promoted_core["points_reconstruction"], dtype=np.float64), np.asarray(promoted_core["colors_rgb"], dtype=np.uint8))
    fringe = formal / "f_attached_single_view_fringe_diagnostic.npz"
    detached = formal / "f_detached_single_view_artifacts.npz"
    shutil.copy2(Path(str(f_art["attached_single_view_fringe_npz"])).expanduser().resolve(), fringe)
    shutil.copy2(Path(str(f_art["detached_single_view_artifacts_npz"])).expanduser().resolve(), detached)
    artifact_paths = {
        "core_npz": core_npz, "core_ply": core_ply, "extended_npz": ext_npz, "extended_ply": ext_ply,
        "rejected_supported_npz": rej_npz, "attached_single_view_fringe_npz": fringe,
        "detached_single_view_artifacts_npz": detached, "coverage_candidate_npz": cov_npz,
    }
    promoted_f = _update_f_summary(
        source_f, source_summary_path=source_f_path, output_dir=formal,
        promoted_extended=promoted_ext, promoted_core=promoted_core, promoted_rejected=promoted_rejected,
        artifact_paths=artifact_paths, i31=i31, i31_summary_path=i31_dir / "i3_1_summary.json", rebuilt=rebuilt, exact_candidate=exact_candidate,
    )
    f_summary_path = _write_json(formal / "f_summary.json", promoted_f)
    if not bool(promoted_f["pass"]):
        raise ReconstructionValidationError("I3.2 promoted formal F product failed qualification.")

    persistent = output / "02_persistent_scene"
    persisted = persist_filtered_dense_rgb_scene(f_summary_path, persistent)
    scene = read_persistent_dense_rgb_scene(persisted.manifest_path, verify_files=True)
    refs_verified = verify_persistent_dense_scene_files(scene, persisted.manifest_path)
    restored_ext = restore_persistent_dense_geometry(persisted.manifest_path, representation="extended")
    restored_core = restore_persistent_dense_geometry(persisted.manifest_path, representation="core")

    relocation = output / "03_relocation_probe"
    shutil.copytree(persistent, relocation)
    relocated_manifest = relocation / persisted.manifest_path.name
    relocated_ext = restore_persistent_dense_geometry(relocated_manifest, representation="extended")
    relocated_core = restore_persistent_dense_geometry(relocated_manifest, representation="core")
    relocation_pass = relocated_ext.point_count == restored_ext.point_count and relocated_core.point_count == restored_core.point_count

    voxel = float(np.asarray(source_ext["voxel_size_reconstruction_units"]).item())
    lineage = compare_component_lineage(
        baseline_points=np.asarray(source_ext["points_reconstruction"], dtype=np.float64),
        promoted_points=np.asarray(promoted_ext["points_reconstruction"], dtype=np.float64),
        voxel_size=voxel,
    )
    actual_reason_count = int(promoted_f["classification"]["artifact_reason_histogram"]["weak_manifold_attachment"])
    exact_supported = (
        int(promoted_f["classification"]["extended_voxels"]) + int(promoted_f["classification"]["rejected_supported_voxels"])
        == int(promoted_f["classification"]["source_supported_voxels"])
    )
    requal = build_i3_2_requalification(
        i31,
        exact_candidate_equivalence=bool(exact_candidate["exact_candidate_equivalence"] and rebuilt_preflight["pass"]),
        promoted_extended_points=restored_ext.point_count,
        promoted_core_points=restored_core.point_count,
        weak_manifold_attachment_rejections=actual_reason_count,
        exact_supported_accounting=exact_supported,
        component_lineage=lineage,
        persistence_reference_count=len(refs_verified),
        persistence_relative=bool(persisted.portable_relative_references),
        persistence_self_contained=bool(persisted.source_independent),
        relocation_restore_pass=relocation_pass,
    )
    passed = bool(requal["pass"])
    comp = requal["component_lineage"]
    env = requal["envelope"]
    summary = {
        "phase": "I3.2", "schema_version": I3_2_SCHEMA_VERSION,
        "status": "ready_for_final_manual_review" if passed else "degraded",
        "automated_pass": passed,
        "source_i3_1_dir": str(i31_dir),
        "source_i1_dir": str(i1_dir),
        "scene_id": scene.scene_id,
        "scene_fingerprint_sha256": scene.scene_fingerprint_sha256,
        "promotion_preflight": promotion_preflight,
        "exact_candidate_equivalence": {k: v for k, v in exact_candidate.items() if k not in ("reject_mask", "is_core")},
        "rebuilt_predicate_equivalence": rebuilt_preflight,
        "geometry": {
            "baseline_extended_points": int(exact_candidate["source_extended_points"]),
            "promoted_extended_points": restored_ext.point_count,
            "promoted_core_points": restored_core.point_count,
            "weak_manifold_attachment_rejections": actual_reason_count,
            "voxel_size_reconstruction_units": voxel,
        },
        "requalification": requal,
        "formal_filtering": {
            "summary": str(f_summary_path.resolve()), "status": promoted_f["status"],
            "exact_supported_accounting": exact_supported,
            "weak_manifold_attachment_rejections": actual_reason_count,
        },
        "persistence": {
            "manifest": str(persisted.manifest_path.resolve()), "file_reference_count": len(refs_verified),
            "all_references_relative": bool(persisted.portable_relative_references),
            "bundle_self_contained": bool(persisted.source_independent),
            "relocation_restore_pass": relocation_pass,
        },
        "authority": {
            "production_f_policy_changed": True, "canonical_geometry_authoritative": scene.canonical_geometry_authoritative,
            "canonical_persistence_promoted": scene.canonical_persistence_promoted,
            "sfm_rerun": False, "da3_rerun": False, "fusion_rerun": False,
            "sfm_scale_status": "arbitrary", "metric_scale_claimed": False,
            "measurement_authoritative": False, "certified_metrology": False, "surface_mesh_generated": False,
        },
        "decision": {
            "authorizes_final_fixed_view_review": passed,
            "requires_reconstruction_ui_sanity": passed,
            "authorizes_i3_closure": False,
            "authorizes_i4_connector_orbit": False,
            "authorizes_mesh_reconstruction": False,
            "next_action": "Run I3.2 promoted fixed-view review and Reconstruction UI sanity, then finalize I3." if passed else "Investigate failed I3.2 automated requalification before any closure.",
        },
        "artifacts": {
            "promoted_f_summary": str(f_summary_path.resolve()),
            "persistent_scene_manifest": str(persisted.manifest_path.resolve()),
            "promoted_extended_npz": str(ext_npz.resolve()), "promoted_core_npz": str(core_npz.resolve()),
        },
    }
    summary_path = _write_json(output / "i3_2_summary.json", summary)

    print("\nI3.2 second-scene production promotion + requalification: " + ("READY_FOR_FINAL_MANUAL_REVIEW" if passed else "DEGRADED"))
    print(f"scene_id={scene.scene_id} fingerprint={scene.scene_fingerprint_sha256}")
    print(f"candidate_exact={rebuilt_preflight['pass']} manifold_rejections={actual_reason_count} extended={exact_candidate['source_extended_points']}->{restored_ext.point_count} core={restored_core.point_count}")
    print(
        f"components baseline={comp['baseline']['largest_component_fraction']:.6f} promoted={comp['promoted']['largest_component_fraction']:.6f} "
        f"removed_from_dominant={comp['removed_from_baseline_dominant']} newly_detached={comp['newly_detached_voxels']}"
    )
    print(
        f"full/robust={env['baseline']['full_to_robust_diagonal_ratio']:.3f}->{env['candidate']['full_to_robust_diagonal_ratio']:.3f} "
        f"improvement={env['full_to_robust_improvement_fraction']:.3f} robust_shrink={env['robust_diagonal_shrink_fraction']:.3f}"
    )
    print(f"persistence references={len(refs_verified)} relative={persisted.portable_relative_references} self_contained={persisted.source_independent} relocation_restore={relocation_pass}")
    print(f"automated_gates_pass={passed}")
    print("sfm_rerun=False da3_rerun=False fusion_rerun=False metric_scale_claimed=False measurement_authoritative=False surface_mesh_generated=False")
    print(f"final_fixed_view_review_required={passed} reconstruction_ui_sanity_required={passed} i3_closure_authorized=False mesh_authorization=False")
    print(f"summary={summary_path}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

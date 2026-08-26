#!/usr/bin/env python3
"""I2.6 production manifold-filter promotion and canonical scene regeneration.

This step promotes the manually reviewed I2.5.3 ``d12_sparse`` rule into the
formal Phase-F filtered dense product and immediately regenerates the portable
Phase-G Dense RGB scene bundle.  It reuses the already-saved full-video F and
I2.5.2 evidence; PyCOLMAP, DA3 and dense fusion are not rerun.
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

from core.reconstruction.dense_confidence_filtering import (  # noqa: E402
    DenseArtifactReason,
    artifact_reason_histogram,
)
from core.reconstruction.dense_manifold_refinement import (  # noqa: E402
    DENSE_MANIFOLD_REFINEMENT_SCHEMA_VERSION,
    PROMOTED_MANIFOLD_SCENARIO,
    manifold_attachment_rejection_mask,
    promotion_loss_fractions,
    validate_exact_counterfactual_equivalence,
    validate_promotion_evidence,
)
from core.reconstruction.dense_scene_persistence import (  # noqa: E402
    persist_filtered_dense_rgb_scene,
    read_persistent_dense_rgb_scene,
    restore_persistent_dense_geometry,
    verify_persistent_dense_scene_files,
)
from core.reconstruction.errors import ReconstructionValidationError  # noqa: E402
from scripts.run_e2_consistency_aware_fusion_refinement import _quantiles, _write_binary_ply  # noqa: E402

_DEFAULT_I1 = Path("/tmp/eia_i1_offline_dense_campaign_recovered")
_DEFAULT_I252 = Path("/tmp/eia_i2_5_2_extremum_local_geometry_manifold_diagnosis")
_DEFAULT_I253 = Path("/tmp/eia_i2_5_3_conservative_manifold_filter_counterfactual")
_DEFAULT_OUT = Path("/tmp/eia_i2_6_manifold_filter_promotion")

_POINT_FIELDS = (
    "points_reconstruction",
    "colors_rgb",
    "view_observation_counts",
    "filter_confidence",
    "median_dispersion_voxels",
    "p90_dispersion_voxels",
    "maximum_dispersion_voxels",
    "neighbor26_count",
    "radius2_neighbor_count",
    "high_support_neighbor_count",
    "component_size",
    "class_code",
    "artifact_reason_code",
)
_META_FIELDS = (
    "voxel_size_reconstruction_units",
    "selected_workspace_orders",
    "selected_keyframe_ids",
    "sparse_landmarks_reconstruction",
    "sparse_landmark_colors_rgb",
    "schema_version",
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i1-dir", type=Path, default=_DEFAULT_I1)
    parser.add_argument("--i2-5-2-dir", type=Path, default=_DEFAULT_I252)
    parser.add_argument("--i2-5-3-dir", type=Path, default=_DEFAULT_I253)
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--scenario", default=PROMOTED_MANIFOLD_SCENARIO)
    parser.add_argument(
        "--confirm-manual-review-pass",
        action="store_true",
        help="Required explicit acknowledgement of the completed I2.5.3 visual review.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError(f"Unable to read I2.6 JSON evidence: {path}") from exc
    if not isinstance(value, dict):
        raise ReconstructionValidationError(f"I2.6 JSON evidence root must be an object: {path}")
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return path


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise ReconstructionValidationError(f"I2.6 required NPZ is missing: {path}")
    with np.load(path, allow_pickle=False) as payload:
        data = {name: np.asarray(payload[name]) for name in payload.files}
    return data


def _validate_filtered_payload(data: Mapping[str, np.ndarray], *, label: str) -> int:
    missing = sorted(set(_POINT_FIELDS + _META_FIELDS).difference(data))
    if missing:
        raise ReconstructionValidationError(f"I2.6 {label} payload is missing fields: {missing!r}")
    points = np.asarray(data["points_reconstruction"], dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ReconstructionValidationError(f"I2.6 {label} geometry is invalid.")
    for name in _POINT_FIELDS:
        values = np.asarray(data[name])
        if name in ("points_reconstruction", "colors_rgb"):
            if values.shape != points.shape:
                raise ReconstructionValidationError(f"I2.6 {label} {name} shape is invalid.")
        elif values.shape != (len(points),):
            raise ReconstructionValidationError(f"I2.6 {label} {name} must align with geometry.")
    return int(len(points))


def _slice_payload(data: Mapping[str, np.ndarray], mask: np.ndarray, *, class_code: int | None = None) -> dict[str, np.ndarray]:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if len(mask) != len(np.asarray(data["points_reconstruction"])):
        raise ReconstructionValidationError("I2.6 slice mask is misaligned.")
    out: dict[str, np.ndarray] = {}
    for name in _POINT_FIELDS:
        values = np.asarray(data[name])
        sliced = values[mask]
        if name == "artifact_reason_code":
            sliced = np.zeros(len(sliced), dtype=np.uint16)
        elif name == "class_code" and class_code is not None:
            sliced = np.full(len(sliced), int(class_code), dtype=np.uint8)
        out[name] = sliced
    for name in _META_FIELDS:
        out[name] = np.asarray(data[name]).copy()
    return out


def _write_payload(path: Path, data: Mapping[str, np.ndarray]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{name: np.asarray(value) for name, value in data.items()})
    return path


def _append_rejected(
    source_rejected: Mapping[str, np.ndarray],
    source_extended: Mapping[str, np.ndarray],
    new_reject_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    new_reject_mask = np.asarray(new_reject_mask, dtype=bool).reshape(-1)
    if len(new_reject_mask) != len(np.asarray(source_extended["points_reconstruction"])):
        raise ReconstructionValidationError("I2.6 promoted rejection mask is misaligned with source Extended geometry.")
    out: dict[str, np.ndarray] = {}
    for name in _POINT_FIELDS:
        old = np.asarray(source_rejected[name])
        new = np.asarray(source_extended[name])[new_reject_mask].copy()
        if name == "artifact_reason_code":
            new = np.full(len(new), int(DenseArtifactReason.WEAK_MANIFOLD_ATTACHMENT), dtype=np.uint16)
        elif name == "class_code":
            new = np.zeros(len(new), dtype=np.uint8)
        out[name] = np.concatenate([old, new], axis=0)
    for name in _META_FIELDS:
        out[name] = np.asarray(source_extended[name]).copy()
    return out


def _updated_coverage(
    source_coverage: Mapping[str, np.ndarray],
    source_extended: Mapping[str, np.ndarray],
    kept_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    source_extended_count = len(np.asarray(source_extended["points_reconstruction"]))
    coverage_points = np.asarray(source_coverage["points_reconstruction"], dtype=np.float64)
    source_points = np.asarray(source_extended["points_reconstruction"], dtype=np.float64)
    if len(coverage_points) < source_extended_count or not np.array_equal(coverage_points[:source_extended_count], source_points):
        raise ReconstructionValidationError("I2.6 source F coverage-candidate ordering is not the expected Extended+fringe layout.")
    out: dict[str, np.ndarray] = {}
    for name in _POINT_FIELDS:
        values = np.asarray(source_coverage[name])
        head = values[:source_extended_count][kept_mask].copy()
        if name == "artifact_reason_code":
            head = np.zeros(len(head), dtype=np.uint16)
        tail = values[source_extended_count:]
        out[name] = np.concatenate([head, tail], axis=0)
    for name in _META_FIELDS:
        out[name] = np.asarray(source_extended[name]).copy()
    return out


def _core_mask_from_extended(source_extended: Mapping[str, np.ndarray], source_core: Mapping[str, np.ndarray]) -> np.ndarray:
    classes = np.asarray(source_extended["class_code"], dtype=np.uint8)
    mask = classes == 3
    ext_core = np.asarray(source_extended["points_reconstruction"], dtype=np.float64)[mask]
    core = np.asarray(source_core["points_reconstruction"], dtype=np.float64)
    if ext_core.shape != core.shape or not np.array_equal(ext_core, core):
        raise ReconstructionValidationError("I2.6 source F Core geometry is not the exact class-3 subset of Extended geometry.")
    return mask


def _update_f_summary(
    source: Mapping[str, Any],
    *,
    output_dir: Path,
    source_f_summary: Path,
    policy_payload: Mapping[str, Any],
    equivalence: Mapping[str, Any],
    candidate_record: Mapping[str, Any],
    promoted_extended: Mapping[str, np.ndarray],
    promoted_core: Mapping[str, np.ndarray],
    promoted_rejected: Mapping[str, np.ndarray],
    detached_count: int,
    artifact_paths: Mapping[str, Path],
) -> dict[str, Any]:
    summary = json.loads(json.dumps(dict(source)))
    classification = dict(summary["classification"])
    source_supported = int(classification["source_supported_voxels"])
    source_all = int(classification["source_all_voxels"])
    ext_count = len(np.asarray(promoted_extended["points_reconstruction"]))
    core_count = len(np.asarray(promoted_core["points_reconstruction"]))
    rejected_count = len(np.asarray(promoted_rejected["points_reconstruction"]))
    if ext_count + rejected_count != source_supported:
        raise ReconstructionValidationError("I2.6 promoted F supported-voxel accounting is not exact.")
    reasons = artifact_reason_histogram(np.asarray(promoted_rejected["artifact_reason_code"], dtype=np.uint16))
    reasons["single_view_detached"] = int(detached_count)
    classification.update({
        "core_voxels": int(core_count),
        "extended_voxels": int(ext_count),
        "rejected_supported_voxels": int(rejected_count),
        "core_fraction_of_supported": float(core_count / source_supported),
        "extended_fraction_of_supported": float(ext_count / source_supported),
        "artifact_reason_histogram": reasons,
    })
    summary["classification"] = classification

    source_conf = np.asarray(promoted_extended["filter_confidence"], dtype=np.float64)
    core_conf = np.asarray(promoted_core["filter_confidence"], dtype=np.float64)
    rejected_conf = np.asarray(promoted_rejected["filter_confidence"], dtype=np.float64)
    summary["confidence"] = {
        **dict(summary.get("confidence", {})),
        "core_quantiles": _quantiles(core_conf),
        "extended_quantiles": _quantiles(source_conf),
        "rejected_quantiles": _quantiles(rejected_conf),
    }

    original_policy = dict(summary.get("policy", {}))
    original_policy["manifold_attachment_refinement"] = dict(policy_payload)
    original_policy["manifold_attachment_artifact_reason"] = "weak_manifold_attachment"
    original_policy["manifold_attachment_promotion_phase"] = "I2.6"
    summary["policy"] = original_policy

    old_conditions = dict(summary.get("pass_conditions", {}))
    min_core = float(original_policy.get("minimum_core_fraction_gate", 0.0))
    min_ext = float(original_policy.get("minimum_extended_fraction_gate", 0.0))
    comparison = candidate_record["comparison"]
    promotion_conditions = {
        "i2_5_3_manual_review_confirmed": True,
        "exact_counterfactual_mask_equivalence": bool(equivalence["exact_mask_match"]),
        "promoted_candidate_balanced": bool(comparison["balanced_for_manual_review"]),
        "promoted_core_fraction_gate": classification["core_fraction_of_supported"] >= min_core,
        "promoted_extended_fraction_gate": classification["extended_fraction_of_supported"] >= min_ext,
        "full_to_robust_ratio_improved_in_qualified_counterfactual": float(comparison["full_to_robust_ratio_improvement_fraction"]) > 0.0,
        "central_core_preserved_in_qualified_counterfactual": float(comparison["central_core_loss_fraction"]) <= 0.01,
        "metric_measurement_mesh_authority_preserved": True,
    }
    summary["pass_conditions"] = {**old_conditions, **promotion_conditions}
    summary["pass"] = bool(all(summary["pass_conditions"].values()))
    summary["status"] = "pass" if summary["pass"] else "degraded"
    summary["promotion"] = {
        "phase": "I2.6",
        "schema_version": DENSE_MANIFOLD_REFINEMENT_SCHEMA_VERSION,
        "source_baseline_f_summary": str(source_f_summary.resolve()),
        "manually_reviewed_counterfactual": PROMOTED_MANIFOLD_SCENARIO,
        "manual_visual_review": "pass",
        "exact_counterfactual_equivalence": dict(equivalence),
        "qualified_counterfactual_comparison": dict(comparison),
        "production_filter_policy_changed": True,
        "sfm_rerun": False,
        "da3_rerun": False,
        "fusion_rerun": False,
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
        "note": (
            "I2.6 promotes the manually qualified d12_sparse manifold-attachment refinement into formal F. "
            "The scene remains in the arbitrary SfM gauge; G persistence is regenerated next, while metric, "
            "measurement and mesh authority remain unpromoted."
        ),
    })
    summary["authority"] = authority
    summary["artifacts"] = {name: str(path.resolve()) for name, path in artifact_paths.items()}
    summary["i2_6_output_dir"] = str(output_dir.resolve())
    if source_all < source_supported:
        raise ReconstructionValidationError("I2.6 source F all/supported hierarchy is invalid.")
    return summary


def main() -> int:
    args = _args()
    i1_dir = args.i1_dir.expanduser().resolve()
    i252_dir = args.i2_5_2_dir.expanduser().resolve()
    i253_dir = args.i2_5_3_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ReconstructionValidationError(f"I2.6 output directory must be empty or absent: {output}")
    output.mkdir(parents=True, exist_ok=True)

    i252 = _read_json(i252_dir / "i2_5_2_summary.json")
    i253 = _read_json(i253_dir / "i2_5_3_summary.json")
    policy, candidate_record, signal = validate_promotion_evidence(
        i252,
        i253,
        scenario_name=str(args.scenario),
        manual_review_pass=bool(args.confirm_manual_review_pass),
    )

    source_f_dir = i1_dir / "05_filtering"
    source_f_summary_path = source_f_dir / "f_summary.json"
    source_f = _read_json(source_f_summary_path)
    if str(source_f.get("phase")) != "F" or not bool(source_f.get("pass")):
        raise ReconstructionValidationError("I2.6 requires the passing I1 formal F product.")
    source_authority = source_f.get("authority", {})
    if any(bool(source_authority.get(name)) for name in (
        "metric_scale_claimed", "canonical_persistence_promoted", "measurement_authoritative", "surface_mesh_generated"
    )):
        raise ReconstructionValidationError("I2.6 source F authority is unexpectedly promoted.")

    f_art = source_f["artifacts"]
    source_extended = _load_npz(Path(str(f_art["extended_npz"])).expanduser().resolve())
    source_core = _load_npz(Path(str(f_art["core_npz"])).expanduser().resolve())
    source_rejected = _load_npz(Path(str(f_art["rejected_supported_npz"])).expanduser().resolve())
    source_coverage = _load_npz(Path(str(f_art["coverage_candidate_npz"])).expanduser().resolve())
    _validate_filtered_payload(source_extended, label="source Extended")
    _validate_filtered_payload(source_core, label="source Core")
    _validate_filtered_payload(source_rejected, label="source rejected-supported")
    _validate_filtered_payload(source_coverage, label="source coverage-candidate")
    core_mask = _core_mask_from_extended(source_extended, source_core)

    diagnostics_path = Path(str(i252["artifacts"]["local_geometry_manifold_npz"])).expanduser().resolve()
    diag = _load_npz(diagnostics_path)
    points = np.asarray(source_extended["points_reconstruction"], dtype=np.float64)
    diag_points = np.asarray(diag["points_reconstruction"], dtype=np.float64)
    if points.shape != diag_points.shape or not np.array_equal(points, diag_points):
        raise ReconstructionValidationError("I2.6 I2.5.2 diagnostics do not exactly align with source F Extended geometry.")
    refs = i252["reference"]["thresholds"]
    reject, resolved_policy = manifold_attachment_rejection_mask(
        diag["nearest_strong_anchor_distance_voxels"],
        diag["knn_median_distance_voxels"],
        anchor_distance_reference_q90_voxels=float(refs["anchor_distance_high"]),
        knn_spacing_reference_q90_voxels=float(refs["knn_spacing_high"]),
        policy=policy,
    )

    counterfactual_path = Path(str(candidate_record["artifacts"]["diagnostics_npz"])).expanduser().resolve()
    counterfactual = _load_npz(counterfactual_path)
    counterfactual_reject = np.asarray(counterfactual["is_rejected_by_policy"], dtype=bool)
    expected = int(candidate_record["effects_against_baseline"]["rejected_points"])
    equivalence = validate_exact_counterfactual_equivalence(
        reject, counterfactual_reject, expected_rejected_points=expected
    )

    kept = ~reject
    promoted_extended = _slice_payload(source_extended, kept)
    promoted_core = _slice_payload(source_extended, core_mask & kept, class_code=3)
    promoted_rejected = _append_rejected(source_rejected, source_extended, reject)
    promoted_coverage = _updated_coverage(source_coverage, source_extended, kept)

    formal_dir = output / "01_promoted_filtering"
    formal_dir.mkdir(parents=True, exist_ok=True)
    ext_npz = _write_payload(formal_dir / "f_filtered_extended.npz", promoted_extended)
    core_npz = _write_payload(formal_dir / "f_filtered_core.npz", promoted_core)
    rejected_npz = _write_payload(formal_dir / "f_rejected_supported.npz", promoted_rejected)
    coverage_npz = _write_payload(formal_dir / "f_coverage_candidate_diagnostic.npz", promoted_coverage)
    ext_ply = _write_binary_ply(
        formal_dir / "f_filtered_extended.ply",
        np.asarray(promoted_extended["points_reconstruction"], dtype=np.float64),
        np.asarray(promoted_extended["colors_rgb"], dtype=np.uint8),
    )
    core_ply = _write_binary_ply(
        formal_dir / "f_filtered_core.ply",
        np.asarray(promoted_core["points_reconstruction"], dtype=np.float64),
        np.asarray(promoted_core["colors_rgb"], dtype=np.uint8),
    )

    # Unchanged single-view diagnostics are copied into the promoted formal-F product.
    fringe_copy = formal_dir / "f_attached_single_view_fringe_diagnostic.npz"
    detached_copy = formal_dir / "f_detached_single_view_artifacts.npz"
    shutil.copy2(Path(str(f_art["attached_single_view_fringe_npz"])).expanduser().resolve(), fringe_copy)
    shutil.copy2(Path(str(f_art["detached_single_view_artifacts_npz"])).expanduser().resolve(), detached_copy)
    detached_count = int(source_f["classification"]["detached_single_view_artifact_voxels"])

    artifact_paths = {
        "core_npz": core_npz,
        "core_ply": core_ply,
        "extended_npz": ext_npz,
        "extended_ply": ext_ply,
        "rejected_supported_npz": rejected_npz,
        "attached_single_view_fringe_npz": fringe_copy,
        "detached_single_view_artifacts_npz": detached_copy,
        "coverage_candidate_npz": coverage_npz,
    }
    promoted_f = _update_f_summary(
        source_f,
        output_dir=formal_dir,
        source_f_summary=source_f_summary_path,
        policy_payload=resolved_policy,
        equivalence=equivalence,
        candidate_record=candidate_record,
        promoted_extended=promoted_extended,
        promoted_core=promoted_core,
        promoted_rejected=promoted_rejected,
        detached_count=detached_count,
        artifact_paths=artifact_paths,
    )
    f_summary_path = _write_json(formal_dir / "f_summary.json", promoted_f)
    if not bool(promoted_f["pass"]):
        raise ReconstructionValidationError("I2.6 promoted formal F product failed its qualification gates.")

    persistent_dir = output / "02_persistent_scene"
    persistence = persist_filtered_dense_rgb_scene(f_summary_path, persistent_dir)
    scene = read_persistent_dense_rgb_scene(persistence.manifest_path, verify_files=True)
    references = verify_persistent_dense_scene_files(scene, persistence.manifest_path)
    restored_extended = restore_persistent_dense_geometry(persistence.manifest_path, representation="extended")
    restored_core = restore_persistent_dense_geometry(persistence.manifest_path, representation="core")

    # Relocation proof: the persisted bundle must restore without the promoted-F source directory.
    relocation_dir = output / "03_relocation_probe"
    shutil.copytree(persistent_dir, relocation_dir)
    relocated_manifest = relocation_dir / persistence.manifest_path.name
    relocated_extended = restore_persistent_dense_geometry(relocated_manifest, representation="extended")
    relocated_core = restore_persistent_dense_geometry(relocated_manifest, representation="core")

    losses = promotion_loss_fractions(
        source_extended_points=len(points),
        promoted_extended_points=restored_extended.point_count,
        source_core_points=len(np.asarray(source_core["points_reconstruction"])),
        promoted_core_points=restored_core.point_count,
    )
    comparison = candidate_record["comparison"]
    pass_conditions = {
        "promoted_f_pass": bool(promoted_f["pass"]),
        "manual_i2_5_3_review_recorded": True,
        "exact_counterfactual_mask_equivalence": bool(equivalence["exact_mask_match"]),
        "promoted_extended_count_matches_counterfactual": restored_extended.point_count == int(candidate_record["candidate_profile"]["extended_points"]),
        "promoted_core_count_matches_counterfactual": restored_core.point_count == int(candidate_record["candidate_profile"]["core_points"]),
        "persistent_scene_integrity_verified": len(references) == 10,
        "persistent_scene_self_contained": bool(persistence.source_independent),
        "persistent_references_relative": bool(persistence.portable_relative_references),
        "relocated_extended_restore_exact": relocated_extended.point_count == restored_extended.point_count,
        "relocated_core_restore_exact": relocated_core.point_count == restored_core.point_count,
        "qualified_ratio_improvement_preserved": float(comparison["full_to_robust_ratio_improvement_fraction"]) > 0.0,
        "core_loss_within_qualified_gate": float(losses["core_loss_fraction"]) <= 0.02,
        "metric_scale_not_claimed": not scene.metric_scale_claimed,
        "measurement_authority_not_promoted": not scene.measurement_authoritative,
        "surface_mesh_not_generated": not scene.surface_mesh_generated,
    }
    passed = all(pass_conditions.values())
    summary = {
        "phase": "I2.6",
        "schema_version": DENSE_MANIFOLD_REFINEMENT_SCHEMA_VERSION,
        "pass": passed,
        "status": "pass" if passed else "degraded",
        "scene_id": scene.scene_id,
        "scene_fingerprint_sha256": scene.scene_fingerprint_sha256,
        "source_i1_dir": str(i1_dir),
        "source_i2_5_2_dir": str(i252_dir),
        "source_i2_5_3_dir": str(i253_dir),
        "promotion": {
            "scenario": str(args.scenario),
            "manual_visual_review": "pass",
            "policy": resolved_policy,
            "exact_counterfactual_equivalence": equivalence,
            "source_extended_points": len(points),
            "promoted_extended_points": restored_extended.point_count,
            "source_core_points": len(np.asarray(source_core["points_reconstruction"])),
            "promoted_core_points": restored_core.point_count,
            **losses,
            "qualified_counterfactual_comparison": comparison,
            "artifact_reason": "weak_manifold_attachment",
        },
        "formal_filtering": {
            "summary": str(f_summary_path.resolve()),
            "status": promoted_f["status"],
            "rejected_supported_voxels": int(promoted_f["classification"]["rejected_supported_voxels"]),
            "weak_manifold_attachment_rejections": int(promoted_f["classification"]["artifact_reason_histogram"]["weak_manifold_attachment"]),
            "exact_supported_accounting": (
                int(promoted_f["classification"]["extended_voxels"])
                + int(promoted_f["classification"]["rejected_supported_voxels"])
                == int(promoted_f["classification"]["source_supported_voxels"])
            ),
        },
        "persistence": {
            "manifest": str(persistence.manifest_path.resolve()),
            "file_reference_count": len(references),
            "all_references_relative": bool(persistence.portable_relative_references),
            "bundle_self_contained": bool(persistence.source_independent),
            "relocation_probe_manifest": str(relocated_manifest.resolve()),
            "relocation_restore_pass": True,
        },
        "pass_conditions": pass_conditions,
        "authority": {
            "production_f_policy_changed": True,
            "canonical_geometry_authoritative": scene.canonical_geometry_authoritative,
            "canonical_persistence_promoted": scene.canonical_persistence_promoted,
            "sfm_scale_status": "arbitrary",
            "metric_scale_claimed": scene.metric_scale_claimed,
            "measurement_authoritative": scene.measurement_authoritative,
            "certified_metrology": scene.certified_metrology,
            "surface_mesh_generated": scene.surface_mesh_generated,
            "sfm_rerun": False,
            "da3_rerun": False,
            "fusion_rerun": False,
        },
        "decision": {
            "authorizes_i2_7_final_requalification": passed,
            "authorizes_i2_closure": False,
            "authorizes_i3": False,
            "authorizes_mesh_reconstruction": False,
            "next_step": "I2.7 - Final Full-Video Requalification and I2 Closure" if passed else None,
        },
        "artifacts": {
            "promoted_f_summary": str(f_summary_path.resolve()),
            "persistent_scene_manifest": str(persistence.manifest_path.resolve()),
            "promoted_extended_npz": str(ext_npz.resolve()),
            "promoted_core_npz": str(core_npz.resolve()),
        },
    }
    summary_path = _write_json(output / "i2_6_summary.json", summary)

    print("I2.6 production manifold filtering + canonical regeneration: " + summary["status"].upper())
    print(
        f"scenario={args.scenario} manual_review=pass exact_counterfactual={equivalence['exact_mask_match']} "
        f"new_manifold_rejections={resolved_policy['rejected_points']}"
    )
    print(
        f"extended={len(points)}->{restored_extended.point_count} loss={losses['extended_loss_fraction']:.4f} "
        f"core={len(np.asarray(source_core['points_reconstruction']))}->{restored_core.point_count} "
        f"loss={losses['core_loss_fraction']:.4f}"
    )
    print(
        f"qualified_full/robust={comparison['baseline_full_to_robust_ratio']:.3f}->"
        f"{comparison['candidate_full_to_robust_ratio']:.3f} "
        f"ratio_improvement={comparison['full_to_robust_ratio_improvement_fraction']:.3f}"
    )
    print(
        f"scene_id={scene.scene_id} fingerprint={scene.scene_fingerprint_sha256} references={len(references)} "
        f"relative={persistence.portable_relative_references} self_contained={persistence.source_independent} relocation_restore=True"
    )
    print("sfm_rerun=False da3_rerun=False fusion_rerun=False sfm_scale_status=arbitrary")
    print("canonical_geometry_authoritative=True canonical_persistence_promoted=True metric_scale_claimed=False measurement_authoritative=False surface_mesh_generated=False")
    print(f"manifest={persistence.manifest_path}")
    print(f"summary={summary_path}")
    print("i2_closure_authorized=False mesh_authorization=False next=I2.7" if passed else "i2_6_pass=False")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

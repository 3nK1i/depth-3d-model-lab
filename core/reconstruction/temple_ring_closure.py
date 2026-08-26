"""I4.2F final closure contract for the TempleRing detail-preservation campaign.

This module is intentionally closure-only.  It validates the already-produced
I4.2A--E evidence chain and the matched I4.2E manual-review renders, then builds
an immutable I4.2F decision record.  It never reruns SfM/MVS, modifies camera
geometry, changes d12_sparse, promotes a dense provider, establishes metric
scale, or generates a mesh.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from core.reconstruction.errors import ReconstructionValidationError

I42F_SCHEMA_VERSION = 1
I42F_PHASE = "I4.2F"
I42F_REQUIRED_VIEWS = ("oblique", "front", "side", "top")
I42F_RENDER_KINDS = ("pre_d12", "post_d12", "ab", "loss_overlay")
I42F_REVIEW_DIMENSIONS = (
    "thin_vertical_member_separation",
    "small_gap_and_opening_preservation",
    "edge_definition",
    "small_protrusion_preservation",
    "surface_thickness_and_false_bridging",
    "local_completeness_vs_noise",
    "production_manifold_detail_loss",
)
I42F_ALLOWED_VERDICTS = (
    "d12_safe_on_temple_ring",
    "d12_detail_loss_observed",
    "inconclusive",
)


def _read_json(path: str | Path, *, label: str) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError(f"Could not read I4.2F {label}: {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"I4.2F {label} must contain a JSON object: {source}")
    return payload


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return output


def sha256_file(path: str | Path) -> str:
    source = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    try:
        with source.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ReconstructionValidationError(f"Could not hash I4.2F evidence {source}: {exc}") from exc
    return digest.hexdigest()


def require_empty_or_absent_output(path: str | Path) -> Path:
    output = Path(path).expanduser().resolve()
    if output.exists():
        if not output.is_dir():
            raise ReconstructionValidationError(f"I4.2F output exists and is not a directory: {output}")
        if any(output.iterdir()):
            raise ReconstructionValidationError(
                f"I4.2F closure evidence is immutable; refusing to overwrite non-empty output: {output}"
            )
    return output


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReconstructionValidationError(f"I4.2F {label} evidence is missing or malformed.")
    return value


def _require_false(mapping: Mapping[str, Any], key: str, *, label: str) -> None:
    if mapping.get(key) is not False:
        raise ReconstructionValidationError(f"I4.2F requires {label}.{key}=False.")


def _require_true(mapping: Mapping[str, Any], key: str, *, label: str) -> None:
    if mapping.get(key) is not True:
        raise ReconstructionValidationError(f"I4.2F requires {label}.{key}=True.")


def _require_hash(path: Path, expected: str, *, label: str) -> str:
    if not path.is_file():
        raise ReconstructionValidationError(f"I4.2F required {label} is missing: {path}")
    actual = sha256_file(path)
    if not expected or actual != str(expected):
        raise ReconstructionValidationError(f"I4.2F {label} hash mismatch.")
    return actual


def _validate_e_summary(e_summary: Mapping[str, Any], *, e_dir: Path) -> dict[str, Any]:
    if e_summary.get("phase") != "I4.2E" or e_summary.get("status") != "ready_for_manual_review":
        raise ReconstructionValidationError("I4.2F requires I4.2E READY_FOR_MANUAL_REVIEW evidence.")
    if e_summary.get("pass") is not True or e_summary.get("execution_integrity_pass") is not True:
        raise ReconstructionValidationError("I4.2F requires passing I4.2E execution integrity.")

    authority = _require_mapping(e_summary.get("authority"), label="I4.2E authority")
    for key in (
        "d12_counterfactual_only",
        "native_i4_2d_fusion_visibility_consumed_for_support_evidence",
        "single_authoritative_sfm_frame_preserved",
    ):
        _require_true(authority, key, label="I4.2E authority")
    for key in (
        "authoritative_sparse_modified",
        "bundle_adjustment_rerun",
        "certified_metrology",
        "dense_provider_promoted",
        "held_out_ground_truth_consumed_by_filtering",
        "i4_3_provider_architecture_authorized",
        "measurement_authoritative",
        "metric_scale_claimed",
        "patch_match_rerun",
        "pose_refinement",
        "production_filter_policy_changed",
        "production_manifold_applied_to_authoritative_scene",
        "sfm_rerun",
        "stereo_fusion_replayed_for_support_evidence_only",
        "surface_mesh_generated",
    ):
        _require_false(authority, key, label="I4.2E authority")

    manual = _require_mapping(e_summary.get("manual_review_contract"), label="I4.2E manual_review_contract")
    _require_true(manual, "required", label="I4.2E manual_review_contract")
    _require_false(manual, "automatic_winner_selected", label="I4.2E manual_review_contract")
    dimensions = tuple(str(v) for v in manual.get("review_dimensions", ()))
    if dimensions != I42F_REVIEW_DIMENSIONS:
        raise ReconstructionValidationError("I4.2F refuses a changed I4.2E manual-review dimension contract.")

    qualification = _require_mapping(e_summary.get("qualification"), label="I4.2E qualification")
    _require_true(qualification, "manual_visual_review_required", label="I4.2E qualification")
    _require_false(qualification, "automatic_detail_winner_selected", label="I4.2E qualification")
    _require_false(qualification, "production_policy_change_authorized", label="I4.2E qualification")
    _require_false(qualification, "i4_3_provider_architecture_authorized", label="I4.2E qualification")
    _require_false(qualification, "metric_scale_claimed", label="I4.2E qualification")
    _require_false(qualification, "measurement_authoritative", label="I4.2E qualification")
    _require_false(qualification, "certified_metrology", label="I4.2E qualification")
    _require_false(qualification, "surface_mesh_generated", label="I4.2E qualification")

    policy = _require_mapping(qualification.get("policy"), label="I4.2E d12 policy")
    if policy.get("scenario_name") != "d12_sparse" or policy.get("exact_production_predicate") is not True:
        raise ReconstructionValidationError("I4.2F requires the exact production d12_sparse counterfactual predicate.")
    _require_false(policy, "production_policy_changed", label="I4.2E d12 policy")

    counts = _require_mapping(qualification.get("counts"), label="I4.2E counts")
    pre = int(counts.get("pre_d12_extended_voxels", -1))
    post = int(counts.get("post_d12_extended_voxels", -1))
    rejected = int(counts.get("d12_rejected_voxels", -1))
    fraction = float(counts.get("d12_rejected_fraction", -1.0))
    if pre <= 0 or post < 0 or rejected < 0 or post + rejected != pre:
        raise ReconstructionValidationError("I4.2F I4.2E pre/post/rejected counts are inconsistent.")
    expected_fraction = rejected / pre
    if abs(fraction - expected_fraction) > 1e-12:
        raise ReconstructionValidationError("I4.2F I4.2E rejected fraction is inconsistent with counts.")

    before = _require_mapping(e_summary.get("authoritative_sparse_sha256_before"), label="I4.2E sparse hashes before")
    after = _require_mapping(e_summary.get("authoritative_sparse_sha256_after"), label="I4.2E sparse hashes after")
    if dict(before) != dict(after) or not before:
        raise ReconstructionValidationError("I4.2F requires unchanged authoritative sparse hashes through I4.2E.")

    source = _require_mapping(e_summary.get("source"), label="I4.2E source")
    _require_true(source, "native_visibility_consumed", label="I4.2E source")
    _require_false(source, "native_visibility_full_view_coverage_required", label="I4.2E source")
    if int(source.get("raw_fused_points", 0)) <= 0 or int(source.get("visibility_observations", 0)) <= 0:
        raise ReconstructionValidationError("I4.2F requires non-empty raw MVS and native visibility evidence.")
    if int(source.get("visibility_support_min", 0)) <= 0:
        raise ReconstructionValidationError("I4.2F native visibility contains invalid empty support.")

    raw_ply = Path(str(source.get("raw_fused_ply", ""))).expanduser().resolve()
    native_vis = Path(str(source.get("native_visibility_path", ""))).expanduser().resolve()
    raw_hash_before = str(source.get("raw_fused_ply_sha256_before", ""))
    raw_hash_after = str(source.get("raw_fused_ply_sha256_after", ""))
    vis_hash_before = str(source.get("native_visibility_sha256_before", ""))
    vis_hash_after = str(source.get("native_visibility_sha256_after", ""))
    if not raw_hash_before or raw_hash_before != raw_hash_after:
        raise ReconstructionValidationError("I4.2F requires unchanged I4.2D fused.ply through I4.2E.")
    if not vis_hash_before or vis_hash_before != vis_hash_after:
        raise ReconstructionValidationError("I4.2F requires unchanged I4.2D fused.ply.vis through I4.2E.")
    _require_hash(raw_ply, raw_hash_before, label="I4.2D raw fused.ply")
    _require_hash(native_vis, vis_hash_before, label="I4.2D fused.ply.vis")

    artifact_hashes = _require_mapping(e_summary.get("artifact_sha256"), label="I4.2E artifact hashes")
    artifacts = _require_mapping(e_summary.get("artifacts"), label="I4.2E artifacts")
    verified_artifacts: dict[str, str] = {}
    for key in (
        "analysis_npz",
        "d12_rejected_only_ply",
        "post_d12_extended_ply",
        "pre_d12_extended_ply",
        "support_npz",
    ):
        artifact = Path(str(artifacts.get(key, ""))).expanduser().resolve()
        verified_artifacts[key] = _require_hash(artifact, str(artifact_hashes.get(key, "")), label=f"I4.2E {key}")

    i4_2d_dir = Path(str(source.get("i4_2d_dir", ""))).expanduser().resolve()
    expected_d = (i4_2d_dir / "i4_2d_summary.json").resolve()
    if not expected_d.is_file():
        raise ReconstructionValidationError(f"I4.2F I4.2D summary is missing: {expected_d}")

    return {
        "counts": {
            "pre_d12_extended_voxels": pre,
            "post_d12_extended_voxels": post,
            "d12_rejected_voxels": rejected,
            "d12_rejected_fraction": fraction,
        },
        "potential_detail_loss_signal": bool(qualification.get("potential_detail_loss_signal", False)),
        "populations": dict(_require_mapping(qualification.get("populations"), label="I4.2E populations")),
        "envelope": dict(_require_mapping(qualification.get("envelope"), label="I4.2E envelope")),
        "raw_fused_ply": raw_ply,
        "native_visibility": native_vis,
        "i4_2d_summary": expected_d,
        "artifact_sha256": verified_artifacts,
    }


def _validate_preview(preview: Mapping[str, Any], *, preview_dir: Path, e_summary_path: Path) -> dict[str, Any]:
    if preview.get("phase") != "I4.2E-preview" or preview.get("status") != "ready_for_manual_review":
        raise ReconstructionValidationError("I4.2F requires the I4.2E preview READY_FOR_MANUAL_REVIEW state.")
    for key in ("manual_visual_review_required", "same_camera_pre_post", "same_sample_indices_pre_post"):
        _require_true(preview, key, label="I4.2E preview")
    for key in (
        "automatic_winner_selected",
        "independent_auto_fit",
        "production_filter_policy_changed",
        "i4_3_provider_architecture_authorized",
        "metric_scale_claimed",
        "measurement_authoritative",
        "surface_mesh_generated",
    ):
        _require_false(preview, key, label="I4.2E preview")

    if str(preview.get("source_i4_2e_summary_sha256", "")) != sha256_file(e_summary_path):
        raise ReconstructionValidationError("I4.2F preview no longer matches the reviewed I4.2E summary.")
    source_path = Path(str(preview.get("source_i4_2e_summary", ""))).expanduser().resolve()
    if source_path != e_summary_path:
        raise ReconstructionValidationError("I4.2F preview points at a different I4.2E summary path.")

    renders = _require_mapping(preview.get("renders"), label="I4.2E preview renders")
    if set(renders) != set(I42F_REQUIRED_VIEWS) or len(renders) != len(I42F_REQUIRED_VIEWS):
        raise ReconstructionValidationError("I4.2F requires exactly the four matched TempleRing review views.")
    render_root = (preview_dir / "renders").resolve()
    verified: dict[str, dict[str, str]] = {}
    paths: dict[str, dict[str, str]] = {}
    for view in I42F_REQUIRED_VIEWS:
        record = _require_mapping(renders.get(view), label=f"I4.2E preview {view}")
        hashes = _require_mapping(record.get("sha256"), label=f"I4.2E preview {view} hashes")
        verified[view] = {}
        paths[view] = {}
        for kind in I42F_RENDER_KINDS:
            path = Path(str(record.get(kind, ""))).expanduser().resolve()
            if path.parent != render_root:
                raise ReconstructionValidationError(f"I4.2F {view}/{kind} escaped the controlled preview render directory.")
            verified[view][kind] = _require_hash(path, str(hashes.get(kind, "")), label=f"I4.2E {view}/{kind} render")
            paths[view][kind] = str(path)
    return {
        "render_sha256": verified,
        "render_paths": paths,
        "sampled_points": int(preview.get("sampled_points", 0)),
        "sampled_d12_rejected": int(preview.get("sampled_d12_rejected", -1)),
        "shared_framing": dict(_require_mapping(preview.get("shared_framing"), label="I4.2E shared framing")),
    }


def _validate_upstream_chain(d_summary_path: Path) -> dict[str, Any]:
    d = _read_json(d_summary_path, label="I4.2D summary")
    if d.get("phase") != "I4.2D" or d.get("status") != "pass" or d.get("pass") is not True:
        raise ReconstructionValidationError("I4.2F requires I4.2D PASS.")
    d_authority = _require_mapping(d.get("authority"), label="I4.2D authority")
    for key in ("single_authoritative_sfm_frame", "raw_mvs_dense_reconstruction"):
        _require_true(d_authority, key, label="I4.2D authority")
    for key in (
        "authoritative_camera_geometry_modified",
        "bundle_adjustment_rerun",
        "certified_metrology",
        "da3_started",
        "dense_provider_promoted",
        "held_out_ground_truth_consumed_by_i4_2d",
        "i4_3_provider_architecture_authorized",
        "measurement_authoritative",
        "metric_scale_claimed",
        "pose_refinement",
        "production_filter_policy_changed",
        "production_manifold_applied",
        "sfm_rerun",
        "surface_mesh_generated",
    ):
        _require_false(d_authority, key, label="I4.2D authority")
    input_authority = _require_mapping(d.get("input_authority"), label="I4.2D input authority")
    for key in ("authoritative_sparse_unchanged", "staged_images_unchanged", "held_out_trajectory_validation_passed"):
        _require_true(input_authority, key, label="I4.2D input authority")

    artifacts = _require_mapping(d.get("artifacts"), label="I4.2D artifacts")
    b_path = Path(str(artifacts.get("frozen_i4_2b_summary", ""))).expanduser().resolve()
    c_path = Path(str(artifacts.get("frozen_i4_2c_summary", ""))).expanduser().resolve()
    b_hash = _require_hash(b_path, str(input_authority.get("i4_2b_summary_sha256", "")), label="frozen I4.2B summary")
    c_hash = _require_hash(c_path, str(input_authority.get("i4_2c_summary_sha256", "")), label="frozen I4.2C summary")

    b = _read_json(b_path, label="frozen I4.2B summary")
    c = _read_json(c_path, label="frozen I4.2C summary")
    if b.get("phase") != "I4.2B" or b.get("status") != "pass" or b.get("pass") is not True:
        raise ReconstructionValidationError("I4.2F frozen I4.2B evidence is not PASS.")
    if c.get("phase") != "I4.2C" or c.get("status") != "pass" or c.get("pass") is not True:
        raise ReconstructionValidationError("I4.2F frozen I4.2C evidence is not PASS.")
    c_quality = _require_mapping(c.get("trajectory_quality"), label="I4.2C trajectory quality")
    if c_quality.get("accepted") is not True or c_quality.get("status") != "pass":
        raise ReconstructionValidationError("I4.2F requires accepted held-out I4.2C trajectory quality.")

    b_artifacts = _require_mapping(b.get("artifacts"), label="I4.2B artifacts")
    a_path = Path(str(b_artifacts.get("frozen_i4_2a_summary", ""))).expanduser().resolve()
    if not a_path.is_file():
        raise ReconstructionValidationError(f"I4.2F frozen I4.2A summary is missing: {a_path}")
    a = _read_json(a_path, label="frozen I4.2A summary")
    if a.get("phase") != "I4.2A" or a.get("status") != "pass" or a.get("pass") is not True:
        raise ReconstructionValidationError("I4.2F frozen I4.2A evidence is not PASS.")

    return {
        "i4_2a_summary": str(a_path),
        "i4_2b_summary": str(b_path),
        "i4_2c_summary": str(c_path),
        "i4_2d_summary": str(d_summary_path),
        "summary_sha256": {
            "i4_2a_summary": sha256_file(a_path),
            "i4_2b_summary": b_hash,
            "i4_2c_summary": c_hash,
            "i4_2d_summary": sha256_file(d_summary_path),
        },
    }


def validate_i4_2f_evidence(*, i4_2e_dir: str | Path, preview_dir: str | Path) -> dict[str, Any]:
    """Fail closed unless the complete A--E chain and manual-review renders are intact."""
    e_root = Path(i4_2e_dir).expanduser().resolve()
    p_root = Path(preview_dir).expanduser().resolve()
    e_summary_path = e_root / "i4_2e_summary.json"
    preview_path = p_root / "i4_2e_preview_summary.json"
    e_summary = _read_json(e_summary_path, label="I4.2E summary")
    preview = _read_json(preview_path, label="I4.2E preview summary")

    e_valid = _validate_e_summary(e_summary, e_dir=e_root)
    p_valid = _validate_preview(preview, preview_dir=p_root, e_summary_path=e_summary_path)
    upstream = _validate_upstream_chain(Path(e_valid["i4_2d_summary"]))

    return {
        "i4_2e_summary": str(e_summary_path),
        "i4_2e_preview_summary": str(preview_path),
        "i4_2e_summary_sha256": sha256_file(e_summary_path),
        "i4_2e_preview_summary_sha256": sha256_file(preview_path),
        "upstream": upstream,
        "qualification": {
            "counts": dict(e_valid["counts"]),
            "potential_detail_loss_signal": bool(e_valid["potential_detail_loss_signal"]),
            "populations": dict(e_valid["populations"]),
            "envelope": dict(e_valid["envelope"]),
        },
        "preview": p_valid,
        "source_artifact_sha256": {
            "raw_fused_ply": sha256_file(e_valid["raw_fused_ply"]),
            "native_visibility": sha256_file(e_valid["native_visibility"]),
            **dict(e_valid["artifact_sha256"]),
        },
    }


def _observations_for_verdict(verdict: str) -> dict[str, Any]:
    if verdict == "d12_safe_on_temple_ring":
        return {
            "dimension_results": {
                "thin_vertical_member_separation": "preserved",
                "small_gap_and_opening_preservation": "preserved",
                "edge_definition": "preserved",
                "small_protrusion_preservation": "no_material_loss_observed",
                "surface_thickness_and_false_bridging": "not_introduced",
                "local_completeness_vs_noise": "no_material_degradation_observed",
                "production_manifold_detail_loss": "not_observed",
            },
            "rejected_geometry_interpretation": "detached_or_artifact_like_in_multi_view_review",
            "summary": (
                "Matched oblique/front/side/top review shows the exact d12_sparse counterfactual preserves the legitimate "
                "TempleRing thin structure, openings, edges and protrusions while the rejected points appear detached/artifact-like."
            ),
        }
    if verdict == "d12_detail_loss_observed":
        return {
            "dimension_results": {key: "material_detail_loss_observed" for key in I42F_REVIEW_DIMENSIONS},
            "rejected_geometry_interpretation": "contains_legitimate_object_detail",
            "summary": "Manual multi-view review found material legitimate-detail loss from the d12_sparse transfer.",
        }
    return {
        "dimension_results": {key: "inconclusive" for key in I42F_REVIEW_DIMENSIONS},
        "rejected_geometry_interpretation": "inconclusive",
        "summary": "Manual multi-view evidence is insufficient for a safe d12 transfer verdict.",
    }


def build_i4_2f_closure(
    *,
    evidence_validation: Mapping[str, Any],
    verdict: str,
    confirmed: bool,
) -> dict[str, Any]:
    if verdict not in I42F_ALLOWED_VERDICTS:
        raise ReconstructionValidationError(f"Unsupported I4.2F manual verdict: {verdict}")
    if not confirmed:
        raise ReconstructionValidationError("I4.2F requires explicit confirmation of the manual multi-view verdict.")

    safe = verdict == "d12_safe_on_temple_ring"
    qualification = _require_mapping(evidence_validation.get("qualification"), label="validated I4.2E qualification")
    counts = dict(_require_mapping(qualification.get("counts"), label="validated I4.2E counts"))
    observations = _observations_for_verdict(verdict)

    return {
        "schema_version": I42F_SCHEMA_VERSION,
        "phase": I42F_PHASE,
        "status": "pass" if safe else "blocked",
        "pass": bool(safe),
        "manual_review": {
            "confirmed": True,
            "verdict": verdict,
            "required_views": list(I42F_REQUIRED_VIEWS),
            "review_dimensions": list(I42F_REVIEW_DIMENSIONS),
            "observations": observations,
            "original_potential_detail_loss_signal": bool(qualification.get("potential_detail_loss_signal", False)),
            "automatic_signal_overwritten": False,
        },
        "scientific_conclusion": {
            "temple_ring_detail_preservation_qualified": bool(safe),
            "d12_sparse_safe_on_temple_ring_within_tested_transfer": bool(safe),
            "temple_ring_driven_d12_policy_correction_required": verdict == "d12_detail_loss_observed",
            "global_universal_detail_safety_claimed": False,
            "benchmark_specific_transfer_evidence": True,
            "raw_mvs_points_are_metric": False,
            "rejected_counts": counts,
            "original_detail_risk_proxy_preserved_in_provenance": True,
        },
        "decision": {
            "i4_2_closed": bool(safe),
            "i4_2f_closed": bool(safe),
            "production_d12_policy_change_authorized": False,
            "dense_provider_promoted": False,
            "i4_3_provider_architecture_authorized": bool(safe),
        },
        "authority": {
            "closure_record_only": True,
            "authoritative_sparse_modified": False,
            "sfm_rerun": False,
            "bundle_adjustment_rerun": False,
            "pose_refinement": False,
            "patch_match_rerun": False,
            "stereo_fusion_rerun": False,
            "production_filter_policy_changed": False,
            "production_manifold_applied_to_authoritative_scene": False,
            "dense_provider_promoted": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
        },
        "evidence": dict(evidence_validation),
        "scope_limits": {
            "d12_safety_scope": "Middlebury TempleRing under the I4.2 MVS-derived production-scale transfer analogue",
            "not_proven_for_every_scene_or_capture_scale": True,
            "provider_architecture_not_implemented_by_this_closure": True,
            "metric_or_metrology_authority_not_established": True,
        },
        "next": (
            "I4.3 canonical production dense-provider architecture (Auto / MVS / AI depth) is authorized. "
            "Provider selection/promotion must remain separately qualified."
            if safe else
            "I4.2 remains open. Resolve the manual detail-preservation verdict before I4.3; do not promote or retune production policy automatically."
        ),
    }

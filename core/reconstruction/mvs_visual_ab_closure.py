"""Manual verdict persistence and closure for I4.1.9A A3.1.

A3.1 closes the controlled House DA3-vs-COLMAP-MVS visual positive control
only after an explicit human verdict.  It revalidates the immutable A3 render
and source-evidence hashes before recording that verdict.  It never reruns
reconstruction, changes camera geometry, establishes metric scale, promotes a
dense provider, or generates a mesh.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from core.reconstruction.errors import ReconstructionValidationError

MVS_VISUAL_AB_CLOSURE_SCHEMA_VERSION = 1
MVS_VISUAL_AB_CLOSURE_PHASE = "I4.1.9A-A3.1"
MVS_VISUAL_AB_REQUIRED_VIEWS = ("oblique", "front", "side", "top")
MVS_VISUAL_AB_REVIEW_DIMENSIONS = (
    "recognizability",
    "surface_thickness_and_smearing",
    "radial_or_fan_streaks",
    "halo_noise",
    "ghost_geometry",
    "detached_components",
    "roof_wall_edge_definition",
    "structural_consistency",
    "completeness_vs_noise",
)
MVS_VISUAL_AB_ALLOWED_VERDICTS = (
    "mvs_clean_da3_smeared",
    "mvs_similarly_smeared",
    "mvs_better_but_imperfect",
)

_MVS_BETTER_BUT_IMPERFECT_OBSERVATIONS: dict[str, str] = {
    "recognizability": "mvs_clearly_better",
    "surface_thickness_and_smearing": "mvs_clearly_better",
    "radial_or_fan_streaks": "mvs_clearly_better",
    "halo_noise": "mvs_clearly_better",
    "ghost_geometry": "mvs_better",
    "detached_components": "mvs_better",
    "roof_wall_edge_definition": "mvs_clearly_better",
    "structural_consistency": "mvs_clearly_better",
    "completeness_vs_noise": "mvs_clearly_better",
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError(f"Could not read A3.1 JSON evidence {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"A3.1 JSON evidence must contain an object: {path}")
    return payload


def write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ReconstructionValidationError(f"Could not hash A3.1 evidence {path}: {exc}") from exc
    return digest.hexdigest()


def require_absent_output(path: str | Path) -> Path:
    output = Path(path).expanduser().resolve()
    if output.exists():
        raise ReconstructionValidationError(
            f"A3.1 closure evidence is immutable; refusing to overwrite existing output: {output}"
        )
    return output


def _require_false(authority: Mapping[str, Any], key: str) -> None:
    if bool(authority.get(key)):
        raise ReconstructionValidationError(f"A3.1 refuses forbidden authority state {key}=True.")


def _resolve_source_hash_paths(summary: Mapping[str, Any]) -> dict[str, Path]:
    source = summary.get("source_artifacts")
    if not isinstance(source, Mapping):
        raise ReconstructionValidationError("A3.1 source_artifacts evidence is missing.")

    da3_i4_dir = Path(str(source.get("da3_i4_1_9_dir", ""))).expanduser().resolve()
    da3_i4_summary = da3_i4_dir / "i4_1_9_summary.json"
    da3_i4 = _read_json(da3_i4_summary)
    da3_artifacts = da3_i4.get("artifacts")
    if not isinstance(da3_artifacts, Mapping):
        raise ReconstructionValidationError("A3.1 DA3 I4.1.9 artifact evidence is missing.")
    da3_i2_summary = Path(str(da3_artifacts.get("i2_summary", ""))).expanduser().resolve()

    mvs_a2_dir = Path(str(source.get("mvs_a2_dir", ""))).expanduser().resolve()
    shared_i4_dir = Path(str(source.get("shared_i4_1_7_dir", ""))).expanduser().resolve()
    return {
        "da3_i4_1_9_summary": da3_i4_summary,
        "da3_i2_summary": da3_i2_summary,
        "da3_spatial_diagnostics_npz": Path(str(source.get("da3_spatial_diagnostics_npz", ""))).expanduser().resolve(),
        "mvs_a2_summary": mvs_a2_dir / "i4_1_9a_a2_summary.json",
        "mvs_fused_ply": Path(str(source.get("mvs_fused_ply", ""))).expanduser().resolve(),
        "a1_camera_authority_manifest": mvs_a2_dir / "evidence/frozen_authority/a1_camera_authority_manifest.json",
        "shared_i4_1_7_summary": shared_i4_dir / "i4_1_7_summary.json",
    }


def validate_a3_manual_review_evidence(
    a3_summary: Mapping[str, Any],
    *,
    a3_dir: str | Path,
) -> dict[str, Any]:
    """Fail closed unless the exact controlled A3 evidence is still intact."""

    root = Path(a3_dir).expanduser().resolve()
    if str(a3_summary.get("phase")) != "I4.1.9A-A3":
        raise ReconstructionValidationError("A3.1 requires an I4.1.9A-A3 summary.")
    if str(a3_summary.get("status")) != "ready_for_manual_visual_ab_review":
        raise ReconstructionValidationError("A3.1 requires A3 evidence still awaiting manual visual review.")

    comparison = a3_summary.get("comparison")
    if not isinstance(comparison, Mapping):
        raise ReconstructionValidationError("A3.1 controlled-comparison evidence is missing.")
    required_true = (
        "same_house_input",
        "same_authoritative_i4_1_7_source",
        "same_authoritative_sfm_frame",
        "independent_auto_fit_forbidden",
        "shared_camera_parameters",
    )
    if any(comparison.get(key) is not True for key in required_true):
        raise ReconstructionValidationError("A3.1 refuses an uncontrolled DA3-vs-MVS comparison.")
    rendered_views = tuple(str(v) for v in comparison.get("rendered_views", ()))
    if set(rendered_views) != set(MVS_VISUAL_AB_REQUIRED_VIEWS) or len(rendered_views) != len(MVS_VISUAL_AB_REQUIRED_VIEWS):
        raise ReconstructionValidationError("A3.1 requires all four controlled A3 views exactly once.")
    dimensions = tuple(str(v) for v in comparison.get("manual_review_dimensions", ()))
    if dimensions != MVS_VISUAL_AB_REVIEW_DIMENSIONS:
        raise ReconstructionValidationError("A3.1 manual-review dimension contract changed after rendering.")

    authority = a3_summary.get("authority")
    if not isinstance(authority, Mapping) or authority.get("diagnostic_visual_comparison_only") is not True:
        raise ReconstructionValidationError("A3.1 diagnostic-only authority evidence is missing.")
    for key in (
        "camera_geometry_modified",
        "sfm_rerun",
        "bundle_adjustment_rerun",
        "pose_refinement",
        "da3_rerun",
        "mvs_rerun",
        "patch_match_rerun",
        "stereo_fusion_rerun",
        "house_k_txt_consumed",
        "metric_scale_claimed",
        "measurement_authoritative",
        "certified_metrology",
        "surface_mesh_generated",
        "dense_provider_promoted",
    ):
        _require_false(authority, key)

    decision = a3_summary.get("decision")
    if not isinstance(decision, Mapping):
        raise ReconstructionValidationError("A3.1 decision evidence is missing.")
    if decision.get("automatic_winner_selected") is not False or decision.get("manual_visual_verdict_required") is not True:
        raise ReconstructionValidationError("A3.1 requires an unmodified manual-verdict A3 decision state.")
    matrix = decision.get("decision_matrix")
    if not isinstance(matrix, Mapping) or any(verdict not in matrix for verdict in MVS_VISUAL_AB_ALLOWED_VERDICTS):
        raise ReconstructionValidationError("A3.1 decision matrix is incomplete.")

    artifacts = a3_summary.get("artifacts")
    if not isinstance(artifacts, Mapping) or not isinstance(artifacts.get("renders"), Mapping):
        raise ReconstructionValidationError("A3.1 render artifact evidence is missing.")
    renders = artifacts["renders"]
    verified_render_hashes: dict[str, dict[str, str]] = {}
    expected_render_root = (root / "renders").resolve()
    for view in MVS_VISUAL_AB_REQUIRED_VIEWS:
        record = renders.get(view)
        if not isinstance(record, Mapping):
            raise ReconstructionValidationError(f"A3.1 missing controlled render record for {view}.")
        hashes = record.get("sha256")
        if not isinstance(hashes, Mapping):
            raise ReconstructionValidationError(f"A3.1 missing render hashes for {view}.")
        verified_render_hashes[view] = {}
        for kind in ("da3", "mvs", "ab"):
            path = Path(str(record.get(kind, ""))).expanduser().resolve()
            if path.parent != expected_render_root:
                raise ReconstructionValidationError(f"A3.1 {view}/{kind} render escaped the controlled A3 render directory.")
            if not path.is_file():
                raise ReconstructionValidationError(f"A3.1 controlled render is missing: {path}")
            expected = str(hashes.get(kind, ""))
            actual = sha256_file(path)
            if not expected or actual != expected:
                raise ReconstructionValidationError(f"A3.1 controlled render hash mismatch for {view}/{kind}.")
            verified_render_hashes[view][kind] = actual

    frozen = a3_summary.get("frozen_sha256")
    if not isinstance(frozen, Mapping):
        raise ReconstructionValidationError("A3.1 frozen source hashes are missing.")
    source_paths = _resolve_source_hash_paths(a3_summary)
    verified_source_hashes: dict[str, str] = {}
    for label, path in source_paths.items():
        if not path.is_file():
            raise ReconstructionValidationError(f"A3.1 frozen source evidence is missing: {path}")
        expected = str(frozen.get(label, ""))
        actual = sha256_file(path)
        if not expected or actual != expected:
            raise ReconstructionValidationError(f"A3.1 frozen source hash mismatch for {label}.")
        verified_source_hashes[label] = actual

    return {
        "controlled_comparison_revalidated": True,
        "required_views": list(MVS_VISUAL_AB_REQUIRED_VIEWS),
        "manual_review_dimensions": list(MVS_VISUAL_AB_REVIEW_DIMENSIONS),
        "render_sha256": verified_render_hashes,
        "source_sha256": verified_source_hashes,
    }


def _manual_observations(verdict: str) -> dict[str, Any]:
    if verdict == "mvs_better_but_imperfect":
        return {
            "dimension_results": dict(_MVS_BETTER_BUT_IMPERFECT_OBSERVATIONS),
            "summary": (
                "COLMAP MVS is clearly more coherent than the current DA3 I4.1.9 reconstruction across the matched "
                "oblique/front/side/top views: substantially less fan/radial streaking and halo, cleaner silhouettes, "
                "less surface thickening/smearing, fewer detached/ghost artifacts, and stronger roof/wall structure."
            ),
            "residual_mvs_limitations": (
                "MVS remains imperfect: residual point-cloud roughness/noise, local incompleteness, and finite surface "
                "thickness remain. This is not a clean mesh or metrology-qualified surface."
            ),
        }
    if verdict == "mvs_clean_da3_smeared":
        return {
            "dimension_results": {key: "mvs_decisively_better" for key in MVS_VISUAL_AB_REVIEW_DIMENSIONS},
            "summary": "MVS is visually clean while the DA3 branch remains smeared/noisy under the controlled A/B.",
            "residual_mvs_limitations": "No material residual limitation was selected by this verdict; authority remains unchanged.",
        }
    return {
        "dimension_results": {key: "similar_material_artifact" for key in MVS_VISUAL_AB_REVIEW_DIMENSIONS},
        "summary": "MVS shows materially similar smearing/fan artifacts to the DA3 branch under the controlled A/B.",
        "residual_mvs_limitations": "Camera/intrinsics/projection/frame assumptions require investigation before dense tuning.",
    }


def build_a3_1_manual_closure(
    a3_summary: Mapping[str, Any],
    *,
    evidence_validation: Mapping[str, Any],
    verdict: str,
    confirmed: bool,
    a3_summary_sha256: str,
) -> dict[str, Any]:
    """Build the immutable A3.1 closure after explicit human confirmation."""

    if verdict not in MVS_VISUAL_AB_ALLOWED_VERDICTS:
        raise ReconstructionValidationError(f"Unsupported A3.1 manual verdict: {verdict}")
    if not confirmed:
        raise ReconstructionValidationError("A3.1 requires explicit confirmation of the manual visual verdict.")
    if evidence_validation.get("controlled_comparison_revalidated") is not True:
        raise ReconstructionValidationError("A3.1 closure requires revalidated controlled A3 evidence.")
    if str(a3_summary.get("phase")) != "I4.1.9A-A3" or str(a3_summary.get("status")) != "ready_for_manual_visual_ab_review":
        raise ReconstructionValidationError("A3.1 closure requires ready-for-review A3 evidence.")
    if len(str(a3_summary_sha256)) != 64:
        raise ReconstructionValidationError("A3.1 requires a SHA256 fingerprint of the reviewed A3 summary.")

    mvs_better = verdict in {"mvs_clean_da3_smeared", "mvs_better_but_imperfect"}
    better_but_imperfect = verdict == "mvs_better_but_imperfect"
    camera_followup = verdict == "mvs_similarly_smeared"
    return {
        "phase": MVS_VISUAL_AB_CLOSURE_PHASE,
        "schema_version": MVS_VISUAL_AB_CLOSURE_SCHEMA_VERSION,
        "status": "pass",
        "pass": True,
        "source_a3": {
            "phase": "I4.1.9A-A3",
            "summary_sha256": str(a3_summary_sha256),
            "controlled_comparison_revalidated": True,
            "same_house_input": True,
            "same_authoritative_sfm_frame": True,
            "shared_camera_parameters": True,
            "independent_auto_fit_forbidden": True,
            "reviewed_views": list(MVS_VISUAL_AB_REQUIRED_VIEWS),
        },
        "manual_review": {
            "required": True,
            "confirmed": True,
            "verdict": verdict,
            "classification": (
                "MVS clearly better than DA3, but imperfect"
                if better_but_imperfect
                else ("MVS clean, DA3 smeared" if verdict == "mvs_clean_da3_smeared" else "MVS similarly smeared")
            ),
            "observations": _manual_observations(verdict),
        },
        "evidence_validation": dict(evidence_validation),
        "scientific_conclusion": {
            "mvs_materially_better_than_current_da3": mvs_better,
            "mvs_residual_errors_require_characterization": better_but_imperfect,
            "primary_problem_supported_in_learned_depth_alignment_fusion_branch": mvs_better,
            "camera_intrinsics_projection_frame_assumptions_primary_investigation_required": camera_followup,
            "authoritative_pycolmap_camera_solution_completely_exonerated": False,
            "note": (
                "Because the controlled MVS branch uses the same authoritative SfM frame/cameras and is materially cleaner, "
                "the current House evidence supports the learned-depth/alignment/fusion branch as the primary DA3 failure "
                "location. This does not prove that camera-model error is exactly zero."
                if mvs_better
                else "Similar artifacts would require camera/intrinsics/projection investigation before further dense tuning."
            ),
        },
        "authority": {
            "single_authoritative_sfm_frame": True,
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
            "house_mvs_positive_control_closed": True,
            "i4_1_9a_closed": True,
            "mvs_static_object_orbit_provider_candidate_supported": mvs_better,
            "da3_house_branch_requires_independent_followup": mvs_better,
            "random_dense_threshold_tuning_authorized": False,
            "i4_2_detail_preservation_authorized": mvs_better,
            "i4_3_provider_architecture_authorized": False,
            "mesh_authorization": False,
        },
        "next": (
            "Proceed to I4.2 Temple Ring thin-structure/detail-preservation evaluation using the preserved House MVS baseline. "
            "Do not perform the I4.3 provider-architecture refactor yet."
            if mvs_better
            else "Investigate camera/intrinsics/projection/frame-transform assumptions before advancing dense-provider qualification."
        ),
    }


__all__ = [
    "MVS_VISUAL_AB_ALLOWED_VERDICTS",
    "MVS_VISUAL_AB_CLOSURE_PHASE",
    "MVS_VISUAL_AB_CLOSURE_SCHEMA_VERSION",
    "MVS_VISUAL_AB_REQUIRED_VIEWS",
    "MVS_VISUAL_AB_REVIEW_DIMENSIONS",
    "build_a3_1_manual_closure",
    "require_absent_output",
    "sha256_file",
    "validate_a3_manual_review_evidence",
    "write_json",
]

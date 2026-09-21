"""I5.10D evidence-only MP-SfM qualification and closure.

This closure consumes the frozen I5.10A admission, I5.10B runtime/evidence
preflight, and the single real I5.10C exhaustive-matching evaluation. It does
not execute SfM/MVS, rerun the baseline, retry the candidate, tune thresholds,
or promote exhaustive matching to a production default.

The admitted conclusion is intentionally bounded: on the exact 60-image
KITTI-style counterfactual, exhaustive matching changed the failure mode from
no sparse model / 0 registered images to a small sparse model / 3 registered
images, while the frozen production quality policy still rejected the result.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from core.reconstruction.errors import ReconstructionValidationError

I510D_SCHEMA_VERSION = 1
I510D_PHASE = "I5.10D"
I510D_RECEIPT_PHASE = "I5.10D_CLOSURE_RECEIPT"
I510D_I510A_FROZEN_SHA256 = "fca08621136404bd5ecdb7b861123f28f4c624dd3b38be0c35e51989a4f4e2db"
I510D_I510B_FROZEN_SHA256 = "0db52edcf9805d66656f874256b5219ded26afe2b25666024c7b9bd3ed02dae4"
I510D_I510C_FROZEN_SHA256 = "f40c915856be93daef782b6d793af7cc3ffa7247280df32009be67c2c80cb98c"


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReconstructionValidationError(f"I5.10D {label} must be an object.")
    return value


def _require(node: Mapping[str, Any], key: str, expected: Any, *, label: str) -> None:
    actual = node.get(key)
    if actual != expected:
        raise ReconstructionValidationError(
            f"I5.10D {label}.{key} contract drift: expected={expected!r} actual={actual!r}."
        )


def _valid_sha(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(ch in "0123456789abcdef" for ch in text)


def _binding(value: Mapping[str, Any], *, label: str, expected_sha256: str | None = None) -> dict[str, str]:
    node = _mapping(value, label=label)
    path = str(node.get("path", "")).strip()
    sha = str(node.get("sha256", "")).strip().lower()
    if not path or not _valid_sha(sha):
        raise ReconstructionValidationError(f"I5.10D {label} binding is malformed.")
    if expected_sha256 is not None and sha != expected_sha256.lower():
        raise ReconstructionValidationError(
            f"I5.10D {label} SHA256 drift: expected={expected_sha256} actual={sha}."
        )
    return {"path": path, "sha256": sha}


def validate_i5_10a_for_i5_10d(payload: Mapping[str, Any]) -> dict[str, Any]:
    root = _mapping(payload, label="I5.10A admission")
    _require(root, "schema_version", 1, label="I5.10A admission")
    _require(root, "phase", "I5.10A", label="I5.10A admission")
    _require(root, "status", "mp_sfm_inventory_and_experiment_admitted", label="I5.10A admission")
    _require(root, "pass", True, label="I5.10A admission")
    design = _mapping(root.get("experiment_design"), label="I5.10A experiment design")
    candidate = _mapping(design.get("candidate"), label="I5.10A candidate")
    _require(candidate, "matching_strategy", "exhaustive", label="I5.10A candidate")
    _require(candidate, "attempt_count", 1, label="I5.10A candidate")
    _require(candidate, "reruns_historical_sequential_baseline", False, label="I5.10A candidate")
    _require(candidate, "same_exact_source_bytes_required", True, label="I5.10A candidate")
    _require(candidate, "same_exact_source_window_required", True, label="I5.10A candidate")
    decision = _mapping(root.get("decision"), label="I5.10A decision")
    _require(decision, "i5_10a_complete", True, label="I5.10A decision")
    _require(decision, "i5_10b_authorized", True, label="I5.10A decision")
    _require(decision, "i5_10c_started", False, label="I5.10A decision")
    return deepcopy(dict(root))


def validate_i5_10b_for_i5_10d(payload: Mapping[str, Any]) -> dict[str, Any]:
    root = _mapping(payload, label="I5.10B preflight")
    _require(root, "schema_version", 1, label="I5.10B preflight")
    _require(root, "phase", "I5.10B", label="I5.10B preflight")
    _require(root, "status", "mp_sfm_runtime_and_evidence_preflight_frozen", label="I5.10B preflight")
    _require(root, "pass", True, label="I5.10B preflight")
    baseline = _mapping(root.get("historical_baseline"), label="I5.10B historical baseline")
    _require(baseline, "baseline_matching_strategy", "sequential", label="I5.10B historical baseline")
    _require(baseline, "baseline_sequential_overlap", 10, label="I5.10B historical baseline")
    _require(baseline, "input_images", 60, label="I5.10B historical baseline")
    _require(baseline, "registered_images", 0, label="I5.10B historical baseline")
    _require(baseline, "mapping_produced_sparse_model", False, label="I5.10B historical baseline")
    _require(baseline, "quality_status", "rejected", label="I5.10B historical baseline")
    source = _mapping(root.get("source_revalidation"), label="I5.10B source")
    for key in ("full_source_identity_verified", "exact_60_image_window_verified", "selected_source_sha256_verified"):
        _require(source, key, True, label="I5.10B source")
    _require(source, "window_image_count", 60, label="I5.10B source")
    candidate = _mapping(root.get("candidate_plan"), label="I5.10B candidate")
    _require(candidate, "matching_strategy", "exhaustive", label="I5.10B candidate")
    _require(candidate, "attempt_count", 1, label="I5.10B candidate")
    _require(candidate, "baseline_rerun", False, label="I5.10B candidate")
    _require(candidate, "exhaustive_pair_count", 1770, label="I5.10B candidate")
    decision = _mapping(root.get("decision"), label="I5.10B decision")
    _require(decision, "i5_10b_complete", True, label="I5.10B decision")
    _require(decision, "i5_10c_authorized", True, label="I5.10B decision")
    _require(decision, "i5_10c_started", False, label="I5.10B decision")
    return deepcopy(dict(root))


def validate_i5_10c_for_i5_10d(payload: Mapping[str, Any]) -> dict[str, Any]:
    root = _mapping(payload, label="I5.10C evaluation")
    _require(root, "schema_version", 1, label="I5.10C evaluation")
    _require(root, "phase", "I5.10C", label="I5.10C evaluation")
    _require(root, "status", "mp_sfm_real_controlled_evaluation_completed", label="I5.10C evaluation")
    _require(root, "pass", True, label="I5.10C evaluation")
    _require(root, "terminal_status", "rejected", label="I5.10C evaluation")
    _require(root, "quality_status", "rejected", label="I5.10C evaluation")

    baseline = _mapping(root.get("historical_baseline"), label="I5.10C baseline")
    _require(baseline, "baseline_matching_strategy", "sequential", label="I5.10C baseline")
    _require(baseline, "baseline_sequential_overlap", 10, label="I5.10C baseline")
    _require(baseline, "input_images", 60, label="I5.10C baseline")
    _require(baseline, "registered_images", 0, label="I5.10C baseline")
    _require(baseline, "mapping_produced_sparse_model", False, label="I5.10C baseline")
    _require(baseline, "quality_status", "rejected", label="I5.10C baseline")

    candidate = _mapping(root.get("candidate_result"), label="I5.10C candidate")
    _require(candidate, "terminal_status", "rejected", label="I5.10C candidate")
    _require(candidate, "quality_status", "rejected", label="I5.10C candidate")
    _require(candidate, "production_quality_status", "rejected", label="I5.10C candidate")
    _require(candidate, "runtime_integrity_failure", False, label="I5.10C candidate")
    _require(candidate, "input_images", 60, label="I5.10C candidate")
    _require(candidate, "mapping_produced_sparse_model", True, label="I5.10C candidate")
    registered = int(candidate.get("registered_images", -1))
    if registered != 3:
        raise ReconstructionValidationError(f"I5.10D expected observed I5.10C registered_images=3, got {registered}.")
    ratio = float(candidate.get("registration_ratio", -1.0))
    if abs(ratio - 0.05) > 1e-12:
        raise ReconstructionValidationError(f"I5.10D expected observed I5.10C registration_ratio=0.05, got {ratio}.")
    _require(candidate, "landmarks", 42, label="I5.10C candidate")
    _require(candidate, "observations", 109, label="I5.10C candidate")
    _require(candidate, "production_default_promotion_authorized", False, label="I5.10C candidate")
    _require(candidate, "retry_authorized", False, label="I5.10C candidate")

    integrity = _mapping(root.get("integrity"), label="I5.10C integrity")
    for key in ("source_integrity_before", "source_integrity_after", "staged_integrity_after", "worker_integrity_before", "worker_integrity_after"):
        _require(integrity, key, True, label="I5.10C integrity")

    source = _mapping(root.get("source_contract"), label="I5.10C source contract")
    _require(source, "same_exact_60_image_window_preserved", True, label="I5.10C source contract")
    _require(source, "same_exact_source_bytes_preserved", True, label="I5.10C source contract")
    _require(source, "baseline_rerun", False, label="I5.10C source contract")

    scope = _mapping(root.get("scope"), label="I5.10C scope")
    _require(scope, "real_candidate_run_count", 1, label="I5.10C scope")
    for key in ("baseline_rerun", "retry_until_pass", "result_driven_retuning", "threshold_relaxation", "adaptive_image_reduction", "production_default_modified", "mvs_rerun", "provider_inference", "auto_policy_implemented", "hybrid_authorized", "universal_matching_strategy_winner_declared"):
        _require(scope, key, False, label="I5.10C scope")

    authority = _mapping(root.get("authority"), label="I5.10C authority")
    _require(authority, "camera_pose_authority_owner", "pycolmap_sfm", label="I5.10C authority")
    _require(authority, "candidate_reconstruction_is_evaluation_only", True, label="I5.10C authority")
    _require(authority, "candidate_can_replace_production_default", False, label="I5.10C authority")
    for key in ("metric_scale_claimed", "measurement_authoritative", "certified_metrology", "surface_mesh_generated"):
        _require(authority, key, False, label="I5.10C authority")

    decision = _mapping(root.get("decision"), label="I5.10C decision")
    _require(decision, "i5_10c_complete", True, label="I5.10C decision")
    _require(decision, "i5_10d_authorized", True, label="I5.10C decision")
    _require(decision, "i5_10d_started", False, label="I5.10C decision")
    _require(decision, "runtime_integrity_failure", False, label="I5.10C decision")
    return deepcopy(dict(root))


def validate_i5_10d_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    root = _mapping(payload, label="I5.10D closure receipt")
    _require(root, "schema_version", 1, label="I5.10D closure receipt")
    _require(root, "phase", I510D_RECEIPT_PHASE, label="I5.10D closure receipt")
    _require(root, "status", "pass", label="I5.10D closure receipt")
    _require(root, "pass", True, label="I5.10D closure receipt")
    scope = _mapping(root.get("scope"), label="I5.10D receipt scope")
    _require(scope, "evidence_only", True, label="I5.10D receipt scope")
    for key in ("sfm_execution", "mvs_execution", "provider_inference"):
        _require(scope, key, False, label="I5.10D receipt scope")
    codes = _mapping(root.get("return_codes"), label="I5.10D receipt return codes")
    for key in ("focused", "predecessor_unit", "predecessor_integration", "predecessor_stress", "import_probe", "runner_help_probe"):
        _require(codes, key, 0, label="I5.10D receipt return codes")
    return deepcopy(dict(root))


def build_i5_10d_mp_sfm_closure(
    *,
    i5_10a_admission: Mapping[str, Any],
    i5_10a_binding: Mapping[str, Any],
    i5_10b_preflight: Mapping[str, Any],
    i5_10b_binding: Mapping[str, Any],
    i5_10c_evaluation: Mapping[str, Any],
    i5_10c_binding: Mapping[str, Any],
    closure_receipt: Mapping[str, Any],
    closure_receipt_binding: Mapping[str, Any],
) -> dict[str, Any]:
    a = validate_i5_10a_for_i5_10d(i5_10a_admission)
    b = validate_i5_10b_for_i5_10d(i5_10b_preflight)
    c = validate_i5_10c_for_i5_10d(i5_10c_evaluation)
    receipt = validate_i5_10d_receipt(closure_receipt)

    a_bind = _binding(i5_10a_binding, label="I5.10A admission", expected_sha256=I510D_I510A_FROZEN_SHA256)
    b_bind = _binding(i5_10b_binding, label="I5.10B preflight", expected_sha256=I510D_I510B_FROZEN_SHA256)
    c_bind = _binding(i5_10c_binding, label="I5.10C evaluation", expected_sha256=I510D_I510C_FROZEN_SHA256)
    receipt_bind = _binding(closure_receipt_binding, label="I5.10D closure receipt")

    # Cross-chain evidence identities must be consistent with the frozen artifacts.
    c_b = _mapping(_mapping(c.get("evidence_bindings"), label="I5.10C bindings").get("i5_10b_preflight"), label="I5.10C I5.10B binding")
    if str(c_b.get("sha256", "")).lower() != I510D_I510B_FROZEN_SHA256:
        raise ReconstructionValidationError("I5.10D I5.10C does not bind the frozen I5.10B preflight SHA256.")

    baseline = c["historical_baseline"]
    candidate = c["candidate_result"]
    diag = _mapping(candidate.get("forward_motion_diagnostic"), label="I5.10C forward-motion diagnostic")
    pairs = int(diag.get("adjacent_registered_motion_pairs", 0))

    qualification = {
        "conclusion": "matching_policy_sensitivity_with_partial_sparse_recovery_but_no_production_quality_recovery",
        "matching_policy_sensitivity_observed": True,
        "partial_sparse_recovery_observed": True,
        "production_quality_recovery_observed": False,
        "baseline_matching": "sequential_overlap10",
        "candidate_matching": "exhaustive",
        "input_images": 60,
        "baseline_registered_images": int(baseline["registered_images"]),
        "candidate_registered_images": int(candidate["registered_images"]),
        "registered_image_delta": int(candidate["registered_images"]) - int(baseline["registered_images"]),
        "baseline_sparse_model": bool(baseline["mapping_produced_sparse_model"]),
        "candidate_sparse_model": bool(candidate["mapping_produced_sparse_model"]),
        "candidate_registration_ratio": float(candidate["registration_ratio"]),
        "candidate_landmarks": int(candidate["landmarks"]),
        "candidate_observations": int(candidate["observations"]),
        "candidate_mean_reprojection_error_px": float(candidate["mean_reprojection_error_px"]),
        "candidate_mean_track_length": float(candidate["mean_track_length"]),
        "candidate_terminal_status": "rejected",
        "runtime_integrity_failure": False,
        "same_exact_source_bytes_preserved": True,
        "same_exact_60_image_window_preserved": True,
    }

    return {
        "schema_version": I510D_SCHEMA_VERSION,
        "phase": I510D_PHASE,
        "status": "mp_sfm_evidence_qualified_and_closed",
        "pass": True,
        "quality_status": "bounded_matching_policy_sensitivity_candidate_rejected",
        "qualification": qualification,
        "candidate_quality": {
            "terminal_status": "rejected",
            "production_quality_status": "rejected",
            "registration_ratio": float(candidate["registration_ratio"]),
            "quality_reasons": deepcopy(list(candidate.get("quality_reasons", []))),
            "production_default_promotion_authorized": False,
        },
        "observed_limitations": {
            "candidate_remains_production_rejected": True,
            "registered_images_only": int(candidate["registered_images"]),
            "input_images": 60,
            "forward_motion_diagnostic": {
                "adjacent_registered_motion_pairs": pairs,
                "scale_invariant": bool(diag.get("scale_invariant", False)),
                "forward_axis_dominant": bool(diag.get("forward_axis_dominant", False)),
                "sufficient_for_sequence_level_geometry_claim": False,
                "authoritative_forward_motion_geometry_proven": False,
            },
            "i5_10_alone_does_not_authorize_hybrid": True,
        },
        "claim_boundaries": {
            "exhaustive_universally_superior": False,
            "exhaustive_production_default_authorized": False,
            "universal_matching_strategy_winner_declared": False,
            "runtime_ranking_authorized": False,
            "absolute_cross_run_pose_or_translation_comparison_allowed": False,
            "metric_scale_comparison_allowed": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
        },
        "evidence_bindings": {
            "i5_10a_admission": a_bind,
            "i5_10b_preflight": b_bind,
            "i5_10c_real_evaluation": c_bind,
            "i5_10d_closure_receipt": receipt_bind,
            "post_i5_10a_source_checkpoint": deepcopy(dict(_mapping(_mapping(b.get("evidence_bindings"), label="I5.10B bindings").get("post_i5_10a_source_checkpoint"), label="post-I5.10A checkpoint"))),
            "post_i5_10b_delta_checkpoint": deepcopy(dict(_mapping(_mapping(c.get("evidence_bindings"), label="I5.10C bindings").get("post_i5_10b_delta_checkpoint"), label="post-I5.10B delta checkpoint"))),
        },
        "validation_receipt": receipt,
        "scope": {
            "evidence_only_qualification_and_closure": True,
            "sfm_execution": False,
            "mvs_execution": False,
            "provider_inference": False,
            "baseline_rerun": False,
            "candidate_rerun": False,
            "retry_until_pass": False,
            "result_driven_retuning": False,
            "threshold_relaxation": False,
            "adaptive_image_reduction": False,
            "production_default_modified": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
        },
        "authority": {
            "camera_pose_authority_owner": "pycolmap_sfm",
            "candidate_reconstruction_is_evaluation_only": True,
            "candidate_can_replace_production_default": False,
            "independent_sfm_gauges_preserved": True,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
            "permanent_global_view_cap": None,
        },
        "decision": {
            "i5_10a_complete": True,
            "i5_10b_complete": True,
            "i5_10c_complete": True,
            "i5_10d_complete": True,
            "i5_10_complete": True,
            "i5_11_authorized": True,
            "i5_11_started": False,
            "i5_12_started": False,
            "i5_13_started": False,
            "i5_14_started": False,
            "production_default_modified": False,
            "threshold_relaxation_authorized": False,
            "universal_matching_strategy_winner_declared": False,
            "hybrid_authorized": False,
            "auto_policy_implemented": False,
        },
        "next": "I5.11 OOM / disk / crash / recovery robustness.",
    }


__all__ = [
    "I510D_SCHEMA_VERSION", "I510D_PHASE", "I510D_RECEIPT_PHASE",
    "I510D_I510A_FROZEN_SHA256", "I510D_I510B_FROZEN_SHA256", "I510D_I510C_FROZEN_SHA256",
    "validate_i5_10a_for_i5_10d", "validate_i5_10b_for_i5_10d", "validate_i5_10c_for_i5_10d",
    "validate_i5_10d_receipt", "build_i5_10d_mp_sfm_closure",
]

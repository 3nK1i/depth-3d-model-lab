"""I5.10A MP-SfM inventory, experiment admission, and roadmap guard.

Repository phase history explicitly ties MP-SfM to matching-strategy evaluation.
The current production-native PyCOLMAP sparse-SfM path exposes exactly two
matching strategies: sequential and exhaustive.  I5.10A is execution-free and
admits one narrow counterfactual: reuse the frozen I5.4 KITTI-style 60-image
sequential baseline as evidence, then evaluate exhaustive matching once on the
exact same admitted source window in I5.10C after an I5.10B runtime/evidence
preflight.

This module does not execute SfM, change matching defaults, add a matching
algorithm, select a provider, create Auto/Hybrid policy, establish metric scale,
or alter PyCOLMAP camera authority.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding
from core.reconstruction.sfm_contracts import SfmFeatureStrategy, SfmMatchingStrategy
from core.reconstruction.sfm_environment import PYCOLMAP_RECOMMENDED_VERSION
from core.reconstruction.sfm_quality import SfmQualityPolicy

I510A_SCHEMA_VERSION = 1
I510A_PHASE = "I5.10A"
I510A_RECEIPT_PHASE = "I5.10A_ADMISSION_RECEIPT"
I510A_I59C_FROZEN_SHA256 = "48148d22de7e67cfa3b5891b230dfd64e69888a224e4e930e6d765b1c3383364"
I510A_PRIMARY_SCENARIO = "kitti_forward_motion"
I510A_SUPPORTED_MATCHING_STRATEGIES = tuple(item.value for item in SfmMatchingStrategy)

I510A_REPOSITORY_ANCHORS: dict[str, str] = {
    "core/reconstruction/sfm_contracts.py": "09c3044732b34e88ab1170fc6d413d586445b5f8662a814950b3542bf57b7cd0",
    "core/reconstruction/sfm_environment.py": "7195c15e1e31ad7079ab9671c10cd85d10b7a28274d3217b8e3e61ff0daa9b4e",
    "core/reconstruction/sfm_quality.py": "66e75d1bd0ae2b6ebe78705f90bc5cd58c8f63287ccfa83f9fb45fb6a5e27a1f",
    "core/reconstruction/sfm_pycolmap.py": "bdcc8a9cd2a885c51000f795ebe2edbc9a43449441fdbfae42c1d3e2f9455f59",
    "scripts/reconstruction_pycolmap_worker.py": "75277ced2640d61b7ebadb1541b1b28ae5b6dc8f5ba23183fb20e442cdd97dac",
    "scripts/run_reconstruction_pycolmap_smoke.py": "4585bc467166f8e9d4d689f97cb04b0fbefddb33c76d27ad5f4457192741180b",
    "core/reconstruction/i5_forward_motion_kitti.py": "6daae700868fe8da6cda9f8a85548ba70a46340849853cd82386776ff44f6855",
    "docs/phase_history/I5_1B_CURRENT_FROZEN_FOUNDATION_DIFFICULT_HANDHELD_REPLAY.md": "134e6a37426de622ede09c91798b50f70347b4ea7a301843217af8192af02369",
    "docs/phase_history/I5_4A_FORWARD_MOTION_KITTI_SCENARIO_ADMISSION.md": "5733a51bc8c76b1be2acb12c7e98dceeba08f7ea50b28fbf37afdd8ab90b37ca",
    "docs/phase_history/I5_4B_CURRENT_FROZEN_FOUNDATION_FORWARD_MOTION_KITTI_REPLAY.md": "e1779aaf76bc25e6e1b901f05eb5c1c9b51350a94a05c825131b8bed0a5843e1",
    "docs/phase_history/I5_4C_FORWARD_MOTION_KITTI_QUALIFICATION_CLOSURE.md": "006b63719059b145cfba1905167aef700ae5324c593dd00e4b43a6dce181edcc",
    "docs/phase_history/I5_9C_REMAINING_DEPTH_PROVIDER_ROBUSTNESS_QUALIFICATION_AND_CLOSURE.md": "1e371442f884c156c3081117f0b2b57cc1f3af7edc599f64d5b604cd58d22738",
}


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReconstructionValidationError(f"I5.10A {label} must be an object.")
    return value


def _require(node: Mapping[str, Any], key: str, expected: Any, *, label: str) -> None:
    actual = node.get(key)
    if actual != expected:
        raise ReconstructionValidationError(
            f"I5.10A {label}.{key} contract drift: expected={expected!r} actual={actual!r}."
        )


def _valid_sha(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(ch in "0123456789abcdef" for ch in text)


def _validate_binding(
    binding: I51EvidenceBinding,
    *,
    label: str,
    expected_sha256: str | None = None,
) -> None:
    if not str(binding.path).strip() or not _valid_sha(binding.sha256):
        raise ReconstructionValidationError(f"I5.10A {label} evidence binding is malformed.")
    if expected_sha256 is not None and binding.sha256.lower() != expected_sha256.lower():
        raise ReconstructionValidationError(
            f"I5.10A {label} SHA256 drifted: expected={expected_sha256} actual={binding.sha256}."
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_i5_10a_repository_anchors(repository_root: Path) -> dict[str, dict[str, Any]]:
    """Verify the exact source/history files inspected when designing I5.10A."""

    root = Path(repository_root).expanduser().resolve()
    if not root.is_dir():
        raise ReconstructionValidationError(f"I5.10A repository root missing: {root}")
    verified: dict[str, dict[str, Any]] = {}
    for relative, expected in I510A_REPOSITORY_ANCHORS.items():
        path = root / relative
        if not path.is_file():
            raise ReconstructionValidationError(f"I5.10A repository anchor missing: {relative}")
        actual = _sha256(path)
        if actual != expected:
            raise ReconstructionValidationError(
                f"I5.10A repository anchor drifted: {relative} expected={expected} actual={actual}"
            )
        verified[relative] = {
            "path": str(path),
            "sha256": actual,
            "verified": True,
        }
    return verified


def repository_mp_sfm_inventory() -> dict[str, Any]:
    """Return the narrow MP-SfM inventory supported by the current repository."""

    strategies = [item.value for item in SfmMatchingStrategy]
    if strategies != ["sequential", "exhaustive"]:
        raise ReconstructionValidationError(
            f"I5.10A inspected SfM matching strategy set drifted: {strategies!r}."
        )
    if SfmFeatureStrategy.COLMAP_SIFT.value != "colmap_sift":
        raise ReconstructionValidationError("I5.10A inspected SfM feature strategy drifted.")
    return {
        "mp_sfm_repository_meaning": "matching_strategy_evaluation",
        "sfm_backend": "pycolmap-offline-sfm",
        "pycolmap_recommended_version": PYCOLMAP_RECOMMENDED_VERSION,
        "supported_matching_strategies": strategies,
        "required_pycolmap_matching_apis": ["match_sequential", "match_exhaustive"],
        "feature_strategy": "colmap_sift",
        "camera_mode": "single",
        "camera_model": "SIMPLE_RADIAL",
        "bundle_adjustment_gpu": False,
        "sequential_pairing": {
            "overlap_configurable": True,
            "quadratic_overlap": True,
            "loop_detection": False,
        },
        "exhaustive_pairing": {"all_image_pairs": True},
        "resume_policy_available": "reuse_completed_stages",
        "vocabulary_tree_matching_admitted": False,
        "spatial_matching_admitted": False,
        "transitive_matching_admitted": False,
        "new_matching_algorithm_introduced": False,
    }


@dataclass(frozen=True, slots=True)
class I510AAdmissionPolicy:
    primary_scenario_id: str = I510A_PRIMARY_SCENARIO
    baseline_matching_strategy: str = "sequential"
    baseline_sequential_overlap: int = 10
    candidate_matching_strategy: str = "exhaustive"
    baseline_rerun_authorized: bool = False
    candidate_attempt_count: int = 1
    input_window_images: int = 60
    image_sequence_fps: float = 10.0
    profile: str = "efficient"
    max_keyframes: int = 60
    same_source_window_required: bool = True
    same_source_bytes_required: bool = True
    same_feature_strategy_required: bool = True
    same_camera_configuration_required: bool = True
    same_quality_policy_required: bool = True
    result_driven_retuning: bool = False
    retry_until_pass: bool = False
    production_default_modified: bool = False

    def __post_init__(self) -> None:
        if self.primary_scenario_id != I510A_PRIMARY_SCENARIO:
            raise ReconstructionValidationError("I5.10A primary scenario is frozen to kitti_forward_motion.")
        if self.baseline_matching_strategy != "sequential" or self.baseline_sequential_overlap != 10:
            raise ReconstructionValidationError("I5.10A historical baseline must remain sequential overlap 10.")
        if self.candidate_matching_strategy != "exhaustive":
            raise ReconstructionValidationError("I5.10A admits only the existing exhaustive counterfactual.")
        if self.baseline_rerun_authorized:
            raise ReconstructionValidationError("I5.10A does not authorize rerunning the frozen sequential baseline.")
        if self.candidate_attempt_count != 1:
            raise ReconstructionValidationError("I5.10A candidate is one immutable attempt only.")
        if self.input_window_images != 60 or self.max_keyframes != 60:
            raise ReconstructionValidationError("I5.10A preserves the frozen 60-image I5.4 window.")
        if float(self.image_sequence_fps) != 10.0 or self.profile != "efficient":
            raise ReconstructionValidationError("I5.10A preserves I5.4 sequence FPS/profile bounds.")
        for name in (
            "same_source_window_required",
            "same_source_bytes_required",
            "same_feature_strategy_required",
            "same_camera_configuration_required",
            "same_quality_policy_required",
        ):
            if getattr(self, name) is not True:
                raise ReconstructionValidationError(f"I5.10A requires {name}=True.")
        for name in ("result_driven_retuning", "retry_until_pass", "production_default_modified"):
            if getattr(self, name) is not False:
                raise ReconstructionValidationError(f"I5.10A requires {name}=False.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "primary_scenario_id": self.primary_scenario_id,
            "baseline_matching_strategy": self.baseline_matching_strategy,
            "baseline_sequential_overlap": self.baseline_sequential_overlap,
            "candidate_matching_strategy": self.candidate_matching_strategy,
            "baseline_rerun_authorized": self.baseline_rerun_authorized,
            "candidate_attempt_count": self.candidate_attempt_count,
            "input_window_images": self.input_window_images,
            "image_sequence_fps": self.image_sequence_fps,
            "profile": self.profile,
            "max_keyframes": self.max_keyframes,
            "same_source_window_required": self.same_source_window_required,
            "same_source_bytes_required": self.same_source_bytes_required,
            "same_feature_strategy_required": self.same_feature_strategy_required,
            "same_camera_configuration_required": self.same_camera_configuration_required,
            "same_quality_policy_required": self.same_quality_policy_required,
            "result_driven_retuning": self.result_driven_retuning,
            "retry_until_pass": self.retry_until_pass,
            "production_default_modified": self.production_default_modified,
            "sfm_quality_policy": SfmQualityPolicy().as_dict(),
        }


def validate_i5_9c_for_i5_10a(payload: Mapping[str, Any]) -> dict[str, Any]:
    root = _mapping(payload, label="I5.9C closure")
    _require(root, "phase", "I5.9C", label="I5.9C closure")
    _require(root, "status", "remaining_depth_provider_robustness_qualified_and_closed", label="I5.9C closure")
    _require(root, "pass", True, label="I5.9C closure")
    _require(root, "quality_status", "mixed_terminal_quality", label="I5.9C closure")

    scope = _mapping(root.get("scope"), label="I5.9C scope")
    for key in (
        "provider_inference", "sfm_rerun", "mvs_rerun", "mp_sfm_started",
        "auto_policy_implemented", "hybrid_authorized", "thresholds_modified",
        "production_default_modified", "result_driven_retuning", "retry_until_pass",
    ):
        _require(scope, key, False, label="I5.9C scope")

    qualification = _mapping(root.get("qualification"), label="I5.9C qualification")
    _require(qualification, "provider_count", 6, label="I5.9C qualification")
    _require(qualification, "scenario_count", 6, label="I5.9C qualification")
    _require(qualification, "combined_terminal_runs", 36, label="I5.9C qualification")
    _require(
        qualification,
        "combined_status_counts",
        {"pass": 35, "degraded": 1, "rejected": 0},
        label="I5.9C qualification",
    )
    _require(qualification, "all_36_current_provider_scenario_attempts_terminal", True, label="I5.9C qualification")
    for key in (
        "provider_comparison_performed", "provider_ranking_claimed",
        "universal_provider_winner_claimed", "universal_single_image_provider_robustness_claimed",
    ):
        _require(qualification, key, False, label="I5.9C qualification")

    authority = _mapping(root.get("authority"), label="I5.9C authority")
    _require(authority, "camera_pose_authority_owner", "pycolmap_sfm", label="I5.9C authority")
    for key in (
        "authoritative_sfm_modified", "metric_scale_claimed", "measurement_authoritative",
        "certified_metrology", "surface_mesh_generated", "provider_selected",
        "provider_promoted", "provider_ranking_created",
    ):
        _require(authority, key, False, label="I5.9C authority")
    _require(authority, "permanent_global_view_cap", None, label="I5.9C authority")

    decision = _mapping(root.get("decision"), label="I5.9C decision")
    for key in ("i5_9_complete", "i5_10_authorized", "mp_sfm_authorized"):
        _require(decision, key, True, label="I5.9C decision")
    for key in (
        "i5_10_started", "i5_11_started", "i5_12_started", "i5_13_started", "i5_14_started",
        "provider_selected", "provider_promoted", "provider_ranking_created",
        "universal_provider_winner_declared", "global_scalar_score_created",
        "auto_policy_implemented", "hybrid_authorized", "mp_sfm_started",
        "threshold_relaxation_authorized", "fusion_capacity_increase_authorized",
        "production_default_modified",
    ):
        _require(decision, key, False, label="I5.9C decision")
    return deepcopy(dict(root))


def validate_i5_10a_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    root = _mapping(payload, label="I5.10A receipt")
    _require(root, "phase", I510A_RECEIPT_PHASE, label="I5.10A receipt")
    _require(root, "status", "pass", label="I5.10A receipt")
    _require(root, "pass", True, label="I5.10A receipt")
    scope = _mapping(root.get("scope"), label="I5.10A receipt scope")
    for key in (
        "focused_i5_10a_tests", "sfm_contract_regression", "i5_4_forward_motion_regression",
        "i5_9c_authorization_regression", "authority_guard_regression", "roadmap_guard_regression",
        "import_probe", "runner_help_probe",
    ):
        _require(scope, key, True, label="I5.10A receipt scope")
    for key in (
        "mp_sfm_execution", "sfm_rerun", "mvs_rerun", "provider_inference",
        "thresholds_modified", "production_default_modified", "result_driven_retuning",
        "retry_until_pass", "auto_policy_implemented", "hybrid_authorized",
    ):
        _require(scope, key, False, label="I5.10A receipt scope")
    rcs = _mapping(root.get("return_codes"), label="I5.10A receipt return_codes")
    for key in (
        "focused", "predecessor_unit", "predecessor_integration", "predecessor_stress",
        "import_probe", "runner_help_probe",
    ):
        _require(rcs, key, 0, label="I5.10A receipt return_codes")
    counts = _mapping(root.get("counts"), label="I5.10A receipt counts")
    focused = int(counts.get("focused_tests", -1))
    pred = int(counts.get("predecessor_total_tests", -1))
    combined = int(counts.get("combined_exercised_tests", -1))
    if focused <= 0 or pred <= 0 or combined != focused + pred:
        raise ReconstructionValidationError("I5.10A receipt test counts are inconsistent.")
    return deepcopy(dict(root))


def build_i5_10a_mp_sfm_admission(
    *,
    i5_9c_closure: Mapping[str, Any],
    i5_9c_binding: I51EvidenceBinding,
    admission_receipt: Mapping[str, Any],
    admission_receipt_binding: I51EvidenceBinding,
    repository_anchors: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    closure = validate_i5_9c_for_i5_10a(i5_9c_closure)
    receipt = validate_i5_10a_receipt(admission_receipt)
    _validate_binding(i5_9c_binding, label="I5.9C closure", expected_sha256=I510A_I59C_FROZEN_SHA256)
    _validate_binding(admission_receipt_binding, label="I5.10A admission receipt")

    if repository_anchors is None:
        anchor_payload = {
            relative: {"sha256": sha, "verified": True}
            for relative, sha in I510A_REPOSITORY_ANCHORS.items()
        }
    else:
        anchor_payload = deepcopy(dict(repository_anchors))
        if set(anchor_payload) != set(I510A_REPOSITORY_ANCHORS):
            raise ReconstructionValidationError("I5.10A repository anchor set drifted.")
        for relative, expected in I510A_REPOSITORY_ANCHORS.items():
            node = _mapping(anchor_payload[relative], label=f"repository anchor {relative}")
            _require(node, "sha256", expected, label=f"repository anchor {relative}")
            _require(node, "verified", True, label=f"repository anchor {relative}")

    policy = I510AAdmissionPolicy().as_dict()
    inventory = repository_mp_sfm_inventory()

    return {
        "schema_version": I510A_SCHEMA_VERSION,
        "phase": I510A_PHASE,
        "status": "mp_sfm_inventory_and_experiment_admitted",
        "pass": True,
        "scope": {
            "execution_free": True,
            "repository_inventory_performed": True,
            "experiment_design_admitted": True,
            "mp_sfm_execution": False,
            "sfm_rerun": False,
            "bundle_adjustment_rerun": False,
            "mvs_rerun": False,
            "provider_inference": False,
            "source_materialized": False,
            "thresholds_modified": False,
            "production_default_modified": False,
            "result_driven_retuning": False,
            "retry_until_pass": False,
            "provider_comparison": False,
            "provider_ranking": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
        },
        "evidence_bindings": {
            "i5_9c_closure": i5_9c_binding.as_dict(),
            "i5_10a_admission_receipt": admission_receipt_binding.as_dict(),
        },
        "repository_anchors": anchor_payload,
        "repository_inventory": inventory,
        "historical_anchors": {
            "difficult_handheld_horse": {
                "historical_matching_strategy": "exhaustive",
                "global_sfm_completed": True,
                "fresh_i5_1_quality_status": "degraded",
                "primary_bottleneck": "local_dense_consistency_connectivity",
                "matching_strategy_is_not_established_primary_sparse_bottleneck": True,
                "real_i5_10_candidate": False,
                "role": "historical_context_only",
            },
            "kitti_forward_motion": {
                "historical_matching_strategy": "sequential",
                "sequential_overlap": 10,
                "input_window_images": 60,
                "profile": "efficient",
                "image_sequence_fps": 10.0,
                "historical_quality_status": "rejected",
                "historical_failure": "incremental_mapping_produced_no_sparse_model",
                "feature_extraction_reached": True,
                "matching_reached": True,
                "incremental_mapping_attempted": True,
                "sparse_model_produced": False,
                "registered_cameras": 0,
                "forward_motion_geometry_revalidated": False,
                "real_i5_10_candidate": True,
                "role": "primary_matching_policy_counterfactual_target",
            },
            "current_castle_exhaustive_success": {
                "role": "context_only_not_causal_control",
                "reason": "cross_scenario_success_cannot_establish_matching_policy_causality",
                "real_i5_10_candidate": False,
            },
        },
        "admission_policy": policy,
        "experiment_design": {
            "mp_sfm_repository_meaning": "matching_strategy_evaluation",
            "primary_scenario": {
                "scenario_id": I510A_PRIMARY_SCENARIO,
                "frozen_window_source": "I5.4A exact 60-image contiguous native-order window",
                "historical_baseline": "I5.4B current frozen-foundation replay",
                "historical_baseline_matching": "sequential",
                "historical_baseline_overlap": 10,
                "historical_baseline_quality": "rejected",
                "historical_baseline_failure": "incremental_mapping_produced_no_sparse_model",
                "forward_motion_geometry_revalidated": False,
                "baseline_reused_not_rerun": True,
            },
            "candidate": {
                "matching_strategy": "exhaustive",
                "uses_existing_production_worker": True,
                "new_matching_algorithm": False,
                "attempt_count": 1,
                "reruns_historical_sequential_baseline": False,
                "same_exact_source_window_required": True,
                "same_exact_source_bytes_required": True,
                "same_profile": "efficient",
                "same_max_keyframes": 60,
                "same_image_sequence_fps": 10.0,
                "same_feature_strategy": "colmap_sift",
                "same_camera_mode": "single",
                "same_camera_model": "SIMPLE_RADIAL",
                "same_quality_policy": deepcopy(policy["sfm_quality_policy"]),
                "sequential_overlap_not_applicable_to_exhaustive_candidate": True,
            },
            "real_candidate_run_count": 1,
            "no_broad_scenario_matrix": True,
            "why_narrow": "I5.4 is the admitted upstream sparse-SfM rejection explicitly deferred to I5.10 matching-strategy evaluation; Horse completed global SfM and failed later in dense consistency.",
        },
        "comparison_contract": {
            "like_for_like_source_identity_required": True,
            "baseline_and_candidate_matching_policy_is_primary_changed_variable": True,
            "registration_and_sparse_health_comparison_allowed": True,
            "registered_camera_count_ratio_allowed": True,
            "landmarks_observations_track_health_allowed": True,
            "mean_reprojection_error_allowed": True,
            "scale_invariant_geometry_diagnostics_allowed": True,
            "forward_axis_dominance_revalidation_allowed_if_candidate_camera_solution_exists": True,
            "absolute_pose_or_translation_cross_run_comparison_allowed": False,
            "absolute_sparse_geometry_cross_run_comparison_allowed": False,
            "metric_scale_comparison_allowed": False,
            "runtime_performance_ranking_authorized": False,
            "universal_matching_strategy_winner_authorized": False,
            "production_default_promotion_authorized": False,
        },
        "outcome_semantics": {
            "candidate_sparse_model_and_accepted_quality": "matching_policy_sensitivity_observed_on_admitted_kitti_style_case_only",
            "candidate_sparse_model_but_degraded_quality": "partial_matching_policy_sensitivity_observed_but_quality_remains_degraded",
            "candidate_no_sparse_model_or_quality_rejected": "no_exhaustive_rescue_observed_on_admitted_case",
            "runtime_or_integrity_failure": "experiment_incomplete_no_scientific_matching_policy_conclusion",
            "no_outcome_authorizes_universal_exhaustive_superiority": True,
            "no_outcome_changes_production_default_in_i5_10c": True,
        },
        "i5_10b_preflight_requirements": {
            "bind_i5_10a_admission": True,
            "bind_i5_4a_admission": True,
            "bind_i5_4b_replay": True,
            "bind_i5_4c_closure": True,
            "revalidate_i5_4_predecessor_chain": True,
            "revalidate_full_source_identity": True,
            "revalidate_exact_60_image_window": True,
            "revalidate_selected_source_sha256": True,
            "probe_pycolmap_4_1_1_and_required_apis": True,
            "require_match_sequential_api": True,
            "require_match_exhaustive_api": True,
            "require_cuda_or_explicitly_record_cpu_fallback": True,
            "freeze_candidate_command": True,
            "freeze_source_and_worker_hashes": True,
            "execute_sfm": False,
        },
        "i5_10c_execution_contract": {
            "authorized_after_i5_10b_freeze_only": True,
            "candidate_matching_strategy": "exhaustive",
            "candidate_attempt_count": 1,
            "baseline_rerun": False,
            "retry_until_pass": False,
            "result_driven_retuning": False,
            "threshold_relaxation": False,
            "production_default_modified": False,
            "output_directory_must_be_new_and_immutable": True,
            "recognized_quality_rejection_is_scientific_evidence": True,
            "runtime_integrity_failure_is_not_quality_rejection": True,
        },
        "authority": {
            "camera_pose_authority_owner": "pycolmap_sfm",
            "candidate_reconstruction_uses_pycolmap_authority": True,
            "candidate_reconstruction_is_evaluation_only": True,
            "candidate_can_replace_production_default": False,
            "independent_sfm_gauges_preserved": True,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
            "provider_selected": False,
            "provider_promoted": False,
            "provider_ranking_created": False,
            "permanent_global_view_cap": None,
        },
        "regression_summary": {
            "focused_i5_10a_tests": receipt["counts"]["focused_tests"],
            "predecessor_tests": receipt["counts"]["predecessor_total_tests"],
            "combined_exercised_tests": receipt["counts"]["combined_exercised_tests"],
            "test_counts_are_not_runtime_quality_scores": True,
        },
        "decision": {
            "i5_9_complete": True,
            "i5_10_started": True,
            "i5_10a_complete": True,
            "i5_10b_authorized": True,
            "i5_10b_started": False,
            "i5_10c_authorized": False,
            "i5_10c_started": False,
            "i5_10d_authorized": False,
            "i5_10d_started": False,
            "i5_10_complete": False,
            "i5_11_started": False,
            "i5_12_started": False,
            "i5_13_started": False,
            "i5_14_started": False,
            "mp_sfm_authorized": True,
            "mp_sfm_started": True,
            "real_mp_sfm_execution_started": False,
            "provider_selected": False,
            "provider_promoted": False,
            "provider_ranking_created": False,
            "universal_matching_strategy_winner_declared": False,
            "global_scalar_score_created": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
            "threshold_relaxation_authorized": False,
            "production_default_modified": False,
        },
        "roadmap_guard": {
            "i5_10b_preflight_next": True,
            "i5_10c_real_evaluation_deferred_until_b_freeze": True,
            "i5_10d_qualification_deferred_until_c_freeze": True,
            "i5_11_oom_disk_crash_recovery_deferred": True,
            "i5_12_hybrid_deferred_if_justified": True,
            "i5_13_first_auto_policy_deferred": True,
            "i5_14_final_i5_closure_deferred": True,
        },
        "predecessor_quality_status": closure["quality_status"],
        "next": "I5.10B MP-SfM runtime/evidence preflight.",
    }


__all__ = [
    "I510A_SCHEMA_VERSION", "I510A_PHASE", "I510A_RECEIPT_PHASE",
    "I510A_I59C_FROZEN_SHA256", "I510A_PRIMARY_SCENARIO",
    "I510A_SUPPORTED_MATCHING_STRATEGIES", "I510A_REPOSITORY_ANCHORS",
    "I510AAdmissionPolicy", "repository_mp_sfm_inventory",
    "verify_i5_10a_repository_anchors", "validate_i5_9c_for_i5_10a",
    "validate_i5_10a_receipt", "build_i5_10a_mp_sfm_admission",
]

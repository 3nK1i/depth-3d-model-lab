"""I5.9B1 metric-like single-image provider robustness replay contracts.

This module is intentionally execution-neutral.  It validates the frozen I5.9A
admission, the I5.9B1 validation receipt, runtime/source preflight evidence, and
real provider/scenario records produced by the B1 runner.  The real execution
is performed only by ``scripts/run_i5_9b1_metric_like_depth_provider_robustness_replay.py``.

B1 covers Metric3D v2, UniDepthV2, MoGe-2, and Depth Pro across the six frozen
Castle/PyCOLMAP foundations admitted by I5.9A.  Existing production semantics
remain authoritative: PASS, DEGRADED, and recognized quality REJECTED are
terminal scientific outcomes; runtime/integrity failures are not.  Fusion
capacity drops are evidence only and do not alter single-image provider status.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding
from core.reconstruction.i5_remaining_depth_provider_robustness import (
    I59A_B1_PROVIDERS,
    I59A_SCENARIO_IDS,
    I59AAdmissionPolicy,
    provider_robustness_specs,
)

I59B1_SCHEMA_VERSION = 1
I59B1_PHASE = "I5.9B1"
I59B1_RECEIPT_PHASE = "I5.9B1_REPLAY_RECEIPT"
I59B1_I59A_FROZEN_SHA256 = "50ae4a51d57390e6b20b16aa2fa83c65b110a5ba29803853af279ffa3c4a7cff"
I59B1_I56A_FROZEN_SHA256 = "506fb778d82ddc46350cbe4922ebe8a2ba43e0df4fa7818687a5768ebe0afc7a"
I59B1_PROVIDERS = I59A_B1_PROVIDERS
I59B1_SCENARIO_IDS = I59A_SCENARIO_IDS
I59B1_EXPECTED_RUN_COUNT = len(I59B1_PROVIDERS) * len(I59B1_SCENARIO_IDS)
I59B1_TERMINAL_STATUSES = ("pass", "degraded", "rejected")


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReconstructionValidationError(f"{label} must be an object.")
    return value


def _sequence(value: Any, *, label: str) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise ReconstructionValidationError(f"{label} must be an array.")
    return tuple(value)


def _require(node: Mapping[str, Any], key: str, expected: Any, *, label: str) -> None:
    if node.get(key) != expected:
        raise ReconstructionValidationError(
            f"{label}.{key} must be {expected!r}; found {node.get(key)!r}."
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
        raise ReconstructionValidationError(f"{label} evidence binding is malformed.")
    if expected_sha256 is not None and binding.sha256.lower() != expected_sha256.lower():
        raise ReconstructionValidationError(
            f"{label} SHA256 drifted: expected={expected_sha256} actual={binding.sha256}"
        )


@dataclass(frozen=True, slots=True)
class I59B1ReplayPolicy:
    providers: tuple[str, ...] = I59B1_PROVIDERS
    scenario_ids: tuple[str, ...] = I59B1_SCENARIO_IDS
    sequential_execution: bool = True
    adaptive_selection: bool = True
    engineer_override: None = None
    production_max_voxels: int = 1_500_000
    minimum_accepted_views: int = 2
    minimum_correspondences: int = 8
    maximum_relative_residual: float = 0.35
    permanent_global_view_cap: None = None
    result_driven_retuning: bool = False
    retry_until_pass: bool = False
    provider_comparison: bool = False

    def __post_init__(self) -> None:
        if tuple(self.providers) != I59B1_PROVIDERS:
            raise ReconstructionValidationError("I5.9B1 provider set is frozen to the four admitted B1 providers.")
        if tuple(self.scenario_ids) != I59B1_SCENARIO_IDS:
            raise ReconstructionValidationError("I5.9B1 scenario set is frozen to the six I5.9A Castle scenarios.")
        if not self.sequential_execution or not self.adaptive_selection or self.engineer_override is not None:
            raise ReconstructionValidationError("I5.9B1 must preserve sequential I4.3F-adaptive execution with no override.")
        if self.production_max_voxels != 1_500_000:
            raise ReconstructionValidationError("I5.9B1 cannot alter the 1,500,000-voxel production capacity.")
        if self.minimum_accepted_views != 2 or self.minimum_correspondences != 8 or self.maximum_relative_residual != 0.35:
            raise ReconstructionValidationError("I5.9B1 cannot alter existing B1 alignment/acceptance defaults.")
        if self.permanent_global_view_cap is not None:
            raise ReconstructionValidationError("I5.9B1 cannot introduce a permanent global view cap.")
        if self.result_driven_retuning or self.retry_until_pass or self.provider_comparison:
            raise ReconstructionValidationError("I5.9B1 forbids retuning, retry-until-pass, and provider comparison.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "providers": list(self.providers),
            "scenario_ids": list(self.scenario_ids),
            "expected_provider_scenario_runs": I59B1_EXPECTED_RUN_COUNT,
            "sequential_execution": self.sequential_execution,
            "adaptive_selection": self.adaptive_selection,
            "engineer_override": self.engineer_override,
            "production_max_voxels": self.production_max_voxels,
            "minimum_accepted_views": self.minimum_accepted_views,
            "minimum_correspondences": self.minimum_correspondences,
            "maximum_relative_residual": self.maximum_relative_residual,
            "permanent_global_view_cap": self.permanent_global_view_cap,
            "result_driven_retuning": self.result_driven_retuning,
            "retry_until_pass": self.retry_until_pass,
            "provider_comparison": self.provider_comparison,
            "fusion_capacity_drops_change_single_image_status": False,
        }


def validate_i5_9a_for_i5_9b1(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(payload, "schema_version", 1, label="I5.9A admission")
    _require(payload, "phase", "I5.9A", label="I5.9A admission")
    _require(
        payload,
        "status",
        "remaining_depth_provider_robustness_inventory_and_scenarios_admitted",
        label="I5.9A admission",
    )
    _require(payload, "pass", True, label="I5.9A admission")

    inventory = _mapping(payload.get("provider_inventory"), label="I5.9A provider inventory")
    if set(inventory) != {
        "metric3d_v2", "unidepth_v2", "moge_2", "depth_pro", "depth_anything_v2", "marigold"
    }:
        raise ReconstructionValidationError("I5.9B1 requires the exact six-provider I5.9A inventory.")
    specs = provider_robustness_specs()
    for provider in I59B1_PROVIDERS:
        node = _mapping(inventory.get(provider), label=f"I5.9A provider {provider}")
        _require(node, "batch", "I5.9B1", label=f"I5.9A provider {provider}")
        _require(
            node,
            "alignment_semantics",
            "per_view_sparse_correspondence_scale_to_authoritative_sfm",
            label=f"I5.9A provider {provider}",
        )
        _require(node, "fusion_capacity_drops_change_status", False, label=f"I5.9A provider {provider}")
        production = _mapping(node.get("production_config"), label=f"I5.9A provider {provider} production config")
        expected = specs[provider]["production_config"]
        for key in ("max_voxels", "minimum_accepted_views", "minimum_correspondences", "maximum_relative_residual"):
            _require(production, key, expected[key], label=f"I5.9A provider {provider} production config")

    matrix = _mapping(payload.get("scenario_matrix"), label="I5.9A scenario matrix")
    if set(matrix) != set(I59B1_SCENARIO_IDS):
        raise ReconstructionValidationError("I5.9B1 requires exactly the six I5.9A scenarios.")
    for sid in I59B1_SCENARIO_IDS:
        node = _mapping(matrix.get(sid), label=f"I5.9A scenario {sid}")
        _require(node, "scenario_id", sid, label=f"I5.9A scenario {sid}")
        rb = _mapping(node.get("report_binding"), label=f"I5.9A scenario {sid} report binding")
        if not str(rb.get("path", "")).strip() or not _valid_sha(rb.get("sha256")):
            raise ReconstructionValidationError(f"I5.9A scenario {sid} report binding is malformed.")

    plan = _mapping(payload.get("batch_plan"), label="I5.9A batch plan")
    b1 = _mapping(plan.get("I5.9B1"), label="I5.9A I5.9B1 plan")
    if tuple(b1.get("providers", ())) != I59B1_PROVIDERS:
        raise ReconstructionValidationError("I5.9A B1 provider order drifted.")
    _require(b1, "authorized", True, label="I5.9A I5.9B1 plan")
    _require(b1, "started", False, label="I5.9A I5.9B1 plan")
    _require(b1, "real_provider_scenario_runs", I59B1_EXPECTED_RUN_COUNT, label="I5.9A I5.9B1 plan")
    if tuple(plan.get("execution_order", ())) != ("I5.9B1", "I5.9B2"):
        raise ReconstructionValidationError("I5.9A execution order drifted.")

    policy = _mapping(payload.get("admission_policy"), label="I5.9A admission policy")
    frozen = I59AAdmissionPolicy().as_dict()
    for key in (
        "scenario_ids", "production_max_voxels", "minimum_accepted_views", "minimum_correspondences",
        "maximum_relative_residual", "permanent_global_view_cap", "provider_comparison",
        "result_driven_retuning", "retry_until_pass",
    ):
        _require(policy, key, frozen[key], label="I5.9A admission policy")

    authority = _mapping(payload.get("authority"), label="I5.9A authority")
    _require(authority, "camera_pose_authority_owner", "pycolmap_sfm", label="I5.9A authority")
    _require(authority, "permanent_global_view_cap", None, label="I5.9A authority")
    for key in (
        "authoritative_sfm_modified", "provider_selected", "provider_promoted", "provider_ranking_created",
        "metric_scale_claimed", "measurement_authoritative", "certified_metrology", "surface_mesh_generated",
    ):
        _require(authority, key, False, label="I5.9A authority")

    decision = _mapping(payload.get("decision"), label="I5.9A decision")
    _require(decision, "i5_9a_complete", True, label="I5.9A decision")
    _require(decision, "i5_9b1_authorized", True, label="I5.9A decision")
    _require(decision, "i5_9b1_started", False, label="I5.9A decision")
    _require(decision, "i5_9b2_started", False, label="I5.9A decision")
    _require(decision, "i5_9c_started", False, label="I5.9A decision")
    _require(decision, "i5_10_started", False, label="I5.9A decision")
    for key in (
        "provider_selected", "provider_promoted", "provider_ranking_created", "universal_provider_winner_declared",
        "global_scalar_score_created", "auto_policy_implemented", "hybrid_authorized", "mp_sfm_authorized",
        "threshold_relaxation_authorized", "fusion_capacity_increase_authorized", "production_default_modified",
    ):
        _require(decision, key, False, label="I5.9A decision")
    _require(payload, "next", "I5.9B1 metric-like single-image provider robustness replay.", label="I5.9A admission")
    return deepcopy(dict(payload))


def validate_i5_9b1_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(payload, "schema_version", I59B1_SCHEMA_VERSION, label="I5.9B1 receipt")
    _require(payload, "phase", I59B1_RECEIPT_PHASE, label="I5.9B1 receipt")
    _require(payload, "status", "pass", label="I5.9B1 receipt")
    _require(payload, "pass", True, label="I5.9B1 receipt")
    scope = _mapping(payload.get("scope"), label="I5.9B1 receipt scope")
    for key in (
        "focused_i5_9b1_tests", "b1_provider_regression", "i5_9a_admission_regression",
        "i5_6a_source_binding_regression", "authority_guard_regression", "import_probe", "runner_help_probe",
    ):
        _require(scope, key, True, label="I5.9B1 receipt scope")
    for key in (
        "real_provider_inference_executed_by_receipt", "sfm_rerun", "mvs_rerun", "thresholds_modified",
        "production_default_modified", "result_driven_retuning", "retry_until_pass", "provider_comparison_started",
        "mp_sfm_started", "auto_policy_implemented", "hybrid_authorized",
    ):
        _require(scope, key, False, label="I5.9B1 receipt scope")
    counts = _mapping(payload.get("counts"), label="I5.9B1 receipt counts")
    if int(counts.get("combined_exercised_tests", 0)) <= 0:
        raise ReconstructionValidationError("I5.9B1 receipt must record exercised tests.")
    return deepcopy(dict(payload))


def validate_i5_9b1_runtime_preflight(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(payload, "schema_version", I59B1_SCHEMA_VERSION, label="I5.9B1 preflight")
    _require(payload, "phase", I59B1_PHASE, label="I5.9B1 preflight")
    _require(payload, "status", "runtime_source_preflight_pass", label="I5.9B1 preflight")
    _require(payload, "pass", True, label="I5.9B1 preflight")
    if tuple(payload.get("providers", ())) != I59B1_PROVIDERS:
        raise ReconstructionValidationError("I5.9B1 preflight provider order drifted.")
    if tuple(payload.get("scenario_ids", ())) != I59B1_SCENARIO_IDS:
        raise ReconstructionValidationError("I5.9B1 preflight scenario order drifted.")
    _require(payload, "expected_provider_scenario_runs", I59B1_EXPECTED_RUN_COUNT, label="I5.9B1 preflight")

    runtimes = _mapping(payload.get("runtime"), label="I5.9B1 preflight runtime")
    if set(runtimes) != set(I59B1_PROVIDERS):
        raise ReconstructionValidationError("I5.9B1 preflight must bind all four B1 runtimes.")
    for provider in I59B1_PROVIDERS:
        node = _mapping(runtimes.get(provider), label=f"I5.9B1 runtime {provider}")
        _require(node, "available", True, label=f"I5.9B1 runtime {provider}")
        _require(node, "cuda_ready", True, label=f"I5.9B1 runtime {provider}")
        if not str(node.get("provider_id", "")).strip() or not str(node.get("worker_script", "")).strip():
            raise ReconstructionValidationError(f"I5.9B1 runtime {provider} provenance is incomplete.")

    plans = _mapping(payload.get("scenario_plans"), label="I5.9B1 scenario plans")
    if set(plans) != set(I59B1_SCENARIO_IDS):
        raise ReconstructionValidationError("I5.9B1 preflight scenario-plan matrix is incomplete.")
    plan_count = 0
    for sid in I59B1_SCENARIO_IDS:
        scenario = _mapping(plans.get(sid), label=f"I5.9B1 preflight scenario {sid}")
        _require(scenario, "source_integrity_verified", True, label=f"I5.9B1 preflight scenario {sid}")
        _require(scenario, "authoritative_sfm_reused_without_rerun", True, label=f"I5.9B1 preflight scenario {sid}")
        providers = _mapping(scenario.get("providers"), label=f"I5.9B1 preflight scenario {sid} providers")
        if set(providers) != set(I59B1_PROVIDERS):
            raise ReconstructionValidationError(f"I5.9B1 preflight scenario {sid} provider plans are incomplete.")
        for provider, node_raw in providers.items():
            node = _mapping(node_raw, label=f"I5.9B1 preflight {sid}/{provider}")
            if int(node.get("selected_view_count", 0)) <= 0 or int(node.get("source_view_count", 0)) <= 0:
                raise ReconstructionValidationError(f"I5.9B1 preflight {sid}/{provider} adaptive plan is empty.")
            if int(node.get("selected_view_count", 0)) > int(node.get("source_view_count", 0)):
                raise ReconstructionValidationError(f"I5.9B1 preflight {sid}/{provider} selected-view count is invalid.")
            _require(node, "provider_inference_started", False, label=f"I5.9B1 preflight {sid}/{provider}")
            plan_count += 1
    if plan_count != I59B1_EXPECTED_RUN_COUNT:
        raise ReconstructionValidationError("I5.9B1 preflight did not resolve exactly 24 provider/scenario plans.")

    scope = _mapping(payload.get("scope"), label="I5.9B1 preflight scope")
    for key in ("provider_inference_started", "sfm_rerun", "mvs_rerun", "provider_comparison_started"):
        _require(scope, key, False, label="I5.9B1 preflight scope")
    authority = _mapping(payload.get("authority"), label="I5.9B1 preflight authority")
    _require(authority, "camera_pose_authority_owner", "pycolmap_sfm", label="I5.9B1 preflight authority")
    _require(authority, "permanent_global_view_cap", None, label="I5.9B1 preflight authority")
    for key in (
        "authoritative_sfm_modified", "provider_selected", "provider_promoted", "metric_scale_claimed",
        "measurement_authoritative", "certified_metrology", "surface_mesh_generated",
    ):
        _require(authority, key, False, label="I5.9B1 preflight authority")
    return deepcopy(dict(payload))


def validate_i5_9b1_provider_scenario_record(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(payload, "schema_version", I59B1_SCHEMA_VERSION, label="I5.9B1 provider/scenario record")
    _require(payload, "phase", I59B1_PHASE, label="I5.9B1 provider/scenario record")
    provider = str(payload.get("provider_implementation", ""))
    scenario = str(payload.get("scenario_id", ""))
    if provider not in I59B1_PROVIDERS or scenario not in I59B1_SCENARIO_IDS:
        raise ReconstructionValidationError("I5.9B1 provider/scenario record identity is not admitted.")
    status = str(payload.get("runtime_status", ""))
    if status not in I59B1_TERMINAL_STATUSES:
        raise ReconstructionValidationError("I5.9B1 provider/scenario record is not a terminal scientific outcome.")
    selected = int(payload.get("selected_views", 0))
    source = int(payload.get("source_registered_views", 0))
    processed = int(payload.get("processed_views", 0))
    accepted = int(payload.get("accepted_views", 0))
    if source <= 0 or selected <= 0 or selected > source or processed != selected or accepted < 0 or accepted > selected:
        raise ReconstructionValidationError("I5.9B1 provider/scenario accounting is inconsistent.")
    _require(payload, "adaptive_selection", True, label="I5.9B1 provider/scenario record")
    _require(payload, "input_integrity_revalidated_after", True, label="I5.9B1 provider/scenario record")
    _require(payload, "fusion_capacity_drops_change_status", False, label="I5.9B1 provider/scenario record")
    if status == "rejected":
        rejection = _mapping(payload.get("rejection"), label="I5.9B1 rejection")
        _require(rejection, "recognized_existing_quality_gate", True, label="I5.9B1 rejection")
        _require(rejection, "terminal", True, label="I5.9B1 rejection")
        _require(rejection, "threshold_changed", False, label="I5.9B1 rejection")
    authority = _mapping(payload.get("authority"), label="I5.9B1 record authority")
    _require(authority, "camera_pose_authority_owner", "pycolmap_sfm", label="I5.9B1 record authority")
    _require(authority, "permanent_global_view_cap", None, label="I5.9B1 record authority")
    for key in (
        "authoritative_sfm_modified", "sfm_rerun", "bundle_adjustment_rerun", "metric_scale_claimed",
        "measurement_authoritative", "certified_metrology", "surface_mesh_generated", "production_filter_policy_changed",
        "provider_promotion_performed",
    ):
        _require(authority, key, False, label="I5.9B1 record authority")
    return deepcopy(dict(payload))


def build_i5_9b1_replay(
    *,
    i5_9a_admission: Mapping[str, Any],
    i5_9a_binding: I51EvidenceBinding,
    i5_6a_binding: I51EvidenceBinding,
    replay_receipt: Mapping[str, Any],
    replay_receipt_binding: I51EvidenceBinding,
    runtime_preflight: Mapping[str, Any],
    runtime_preflight_binding: I51EvidenceBinding,
    records: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    validate_i5_9a_for_i5_9b1(i5_9a_admission)
    validate_i5_9b1_receipt(replay_receipt)
    validate_i5_9b1_runtime_preflight(runtime_preflight)
    _validate_binding(i5_9a_binding, label="I5.9A", expected_sha256=I59B1_I59A_FROZEN_SHA256)
    _validate_binding(i5_6a_binding, label="I5.6A", expected_sha256=I59B1_I56A_FROZEN_SHA256)
    _validate_binding(replay_receipt_binding, label="I5.9B1 receipt")
    _validate_binding(runtime_preflight_binding, label="I5.9B1 runtime preflight")

    if set(records) != set(I59B1_SCENARIO_IDS):
        raise ReconstructionValidationError("I5.9B1 replay requires exactly six scenario record groups.")
    compact: dict[str, dict[str, Any]] = {}
    statuses: list[str] = []
    total = 0
    for sid in I59B1_SCENARIO_IDS:
        providers = _mapping(records.get(sid), label=f"I5.9B1 records {sid}")
        if set(providers) != set(I59B1_PROVIDERS):
            raise ReconstructionValidationError(f"I5.9B1 scenario {sid} must contain exactly four B1 providers.")
        compact[sid] = {}
        for provider in I59B1_PROVIDERS:
            record = validate_i5_9b1_provider_scenario_record(
                _mapping(providers.get(provider), label=f"I5.9B1 record {sid}/{provider}")
            )
            compact[sid][provider] = record
            statuses.append(str(record["runtime_status"]))
            total += 1
    if total != I59B1_EXPECTED_RUN_COUNT:
        raise ReconstructionValidationError("I5.9B1 replay record count must be exactly 24.")

    counts = {status: statuses.count(status) for status in I59B1_TERMINAL_STATUSES}
    quality_status = "pass" if counts["pass"] == total else "mixed_terminal_quality"
    return {
        "schema_version": I59B1_SCHEMA_VERSION,
        "phase": I59B1_PHASE,
        "status": "metric_like_single_image_robustness_replayed",
        "pass": True,
        "quality_status": quality_status,
        "providers": list(I59B1_PROVIDERS),
        "scenario_ids": list(I59B1_SCENARIO_IDS),
        "expected_provider_scenario_runs": I59B1_EXPECTED_RUN_COUNT,
        "terminal_provider_scenario_runs": total,
        "terminal_status_counts": counts,
        "execution_policy": I59B1ReplayPolicy().as_dict(),
        "evidence_bindings": {
            "i5_9a_admission": i5_9a_binding.as_dict(),
            "i5_6a_admission": i5_6a_binding.as_dict(),
            "replay_receipt": replay_receipt_binding.as_dict(),
            "runtime_preflight": runtime_preflight_binding.as_dict(),
        },
        "runtime_preflight": deepcopy(dict(runtime_preflight)),
        "records": compact,
        "scope": {
            "real_metric_like_single_image_provider_replay": True,
            "provider_scenario_runs_attempted_once": True,
            "sequential_execution": True,
            "sfm_rerun": False,
            "mvs_rerun": False,
            "provider_comparison": False,
            "result_driven_retuning": False,
            "retry_until_pass": False,
            "thresholds_modified": False,
            "fusion_capacity_increased": False,
            "production_default_modified": False,
            "mp_sfm_started": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
        },
        "authority": {
            "camera_pose_authority_owner": "pycolmap_sfm",
            "authoritative_sfm_modified": False,
            "single_authoritative_sfm_frame_preserved": True,
            "provider_selected": False,
            "provider_promoted": False,
            "provider_ranking_created": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
            "permanent_global_view_cap": None,
        },
        "decision": {
            "i5_9a_complete": True,
            "i5_9b1_complete": True,
            "i5_9b2_authorized": True,
            "i5_9b2_started": False,
            "i5_9c_started": False,
            "i5_10_started": False,
            "provider_selected": False,
            "provider_promoted": False,
            "provider_ranking_created": False,
            "universal_provider_winner_declared": False,
            "global_scalar_score_created": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
            "mp_sfm_authorized": False,
            "threshold_relaxation_authorized": False,
            "fusion_capacity_increase_authorized": False,
            "production_default_modified": False,
        },
        "next": "I5.9B2 relative-depth single-image provider robustness replay.",
    }


__all__ = [
    "I59B1_SCHEMA_VERSION", "I59B1_PHASE", "I59B1_RECEIPT_PHASE",
    "I59B1_I59A_FROZEN_SHA256", "I59B1_I56A_FROZEN_SHA256",
    "I59B1_PROVIDERS", "I59B1_SCENARIO_IDS", "I59B1_EXPECTED_RUN_COUNT",
    "I59B1_TERMINAL_STATUSES", "I59B1ReplayPolicy", "validate_i5_9a_for_i5_9b1",
    "validate_i5_9b1_receipt", "validate_i5_9b1_runtime_preflight",
    "validate_i5_9b1_provider_scenario_record", "build_i5_9b1_replay",
]

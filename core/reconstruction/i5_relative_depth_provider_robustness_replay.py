"""I5.9B2 relative-depth single-image provider robustness replay contracts.

B2 replays the two remaining relative-depth production providers, Depth Anything
V2 and Marigold, across the six frozen Castle/PyCOLMAP scenarios admitted by
I5.9A.  It requires the completed I5.9B1 replay as a predecessor and preserves
its observed 24/24 PASS outcome without converting B2 into a provider ranking.

The historical I4.3G4 Low Texture rejections for both B2 providers remain frozen
anchors.  New Castle outcomes are reported separately and never overwrite those
historical scenario-specific verdicts.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding
from core.reconstruction.i5_remaining_depth_provider_robustness import (
    I59A_B2_PROVIDERS,
    I59A_PROVIDER_IMPLEMENTATIONS,
    I59A_SCENARIO_IDS,
    provider_robustness_specs,
)
from core.reconstruction.i5_remaining_depth_provider_robustness_replay import (
    I59B1_EXPECTED_RUN_COUNT,
    I59B1_PROVIDERS,
    I59B1_SCENARIO_IDS,
)

I59B2_SCHEMA_VERSION = 1
I59B2_PHASE = "I5.9B2"
I59B2_RECEIPT_PHASE = "I5.9B2_REPLAY_RECEIPT"
I59B2_I59A_FROZEN_SHA256 = "50ae4a51d57390e6b20b16aa2fa83c65b110a5ba29803853af279ffa3c4a7cff"
I59B2_I56A_FROZEN_SHA256 = "506fb778d82ddc46350cbe4922ebe8a2ba43e0df4fa7818687a5768ebe0afc7a"
I59B2_PROVIDERS = I59A_B2_PROVIDERS
I59B2_SCENARIO_IDS = I59A_SCENARIO_IDS
I59B2_EXPECTED_RUN_COUNT = len(I59B2_PROVIDERS) * len(I59B2_SCENARIO_IDS)
I59B2_TERMINAL_STATUSES = ("pass", "degraded", "rejected")
I59B2_HISTORICAL_G4_LOW_TEXTURE = {
    "depth_anything_v2": "rejected",
    "marigold": "rejected",
}
I59B2_ALIGNMENT_SEMANTICS = {
    "depth_anything_v2": "per_view_affine_inverse_depth_to_authoritative_sfm",
    "marigold": "per_view_affine_native_domain_to_authoritative_sfm",
}


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReconstructionValidationError(f"{label} must be an object.")
    return value


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
            f"{label} SHA256 drifted from the frozen predecessor: "
            f"expected={expected_sha256} actual={binding.sha256}"
        )


@dataclass(frozen=True, slots=True)
class I59B2ReplayPolicy:
    providers: tuple[str, ...] = I59B2_PROVIDERS
    scenario_ids: tuple[str, ...] = I59B2_SCENARIO_IDS
    sequential_provider_scenario_execution: bool = True
    adaptive_view_selection_deferred_to: str = "I4.3F"
    production_max_voxels: int = 1_500_000
    minimum_accepted_views: int = 2
    minimum_correspondences: int = 8
    maximum_relative_residual: float = 0.35
    fusion_capacity_drops_change_single_image_status: bool = False
    result_driven_retuning: bool = False
    retry_until_pass: bool = False
    provider_comparison: bool = False
    threshold_relaxation: bool = False
    production_default_modified: bool = False
    permanent_global_view_cap: int | None = None

    def __post_init__(self) -> None:
        if tuple(self.providers) != I59B2_PROVIDERS:
            raise ReconstructionValidationError("I5.9B2 provider set is frozen to Depth Anything V2 and Marigold.")
        if tuple(self.scenario_ids) != I59B2_SCENARIO_IDS:
            raise ReconstructionValidationError("I5.9B2 scenario set is frozen to the six I5.9A Castle scenarios.")
        if not self.sequential_provider_scenario_execution or self.adaptive_view_selection_deferred_to != "I4.3F":
            raise ReconstructionValidationError("I5.9B2 must preserve sequential I4.3F-adaptive execution.")
        if self.production_max_voxels != 1_500_000:
            raise ReconstructionValidationError("I5.9B2 cannot alter the 1,500,000-voxel production capacity.")
        if self.minimum_accepted_views != 2 or self.minimum_correspondences != 8 or self.maximum_relative_residual != 0.35:
            raise ReconstructionValidationError("I5.9B2 cannot alter existing relative-depth alignment/acceptance defaults.")
        if self.fusion_capacity_drops_change_single_image_status:
            raise ReconstructionValidationError("I5.9B2 cannot import VDA capacity-degradation semantics.")
        if any((self.result_driven_retuning, self.retry_until_pass, self.provider_comparison, self.threshold_relaxation, self.production_default_modified)):
            raise ReconstructionValidationError("I5.9B2 forbids retuning, retry-until-pass, provider comparison, and default changes.")
        if self.permanent_global_view_cap is not None:
            raise ReconstructionValidationError("I5.9B2 cannot introduce a permanent global view cap.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "providers": list(self.providers),
            "scenario_ids": list(self.scenario_ids),
            "expected_provider_scenario_runs": I59B2_EXPECTED_RUN_COUNT,
            "sequential_provider_scenario_execution": self.sequential_provider_scenario_execution,
            "adaptive_view_selection_deferred_to": self.adaptive_view_selection_deferred_to,
            "production_max_voxels": self.production_max_voxels,
            "minimum_accepted_views": self.minimum_accepted_views,
            "minimum_correspondences": self.minimum_correspondences,
            "maximum_relative_residual": self.maximum_relative_residual,
            "fusion_capacity_drops_change_single_image_status": self.fusion_capacity_drops_change_single_image_status,
            "result_driven_retuning": self.result_driven_retuning,
            "retry_until_pass": self.retry_until_pass,
            "provider_comparison": self.provider_comparison,
            "threshold_relaxation": self.threshold_relaxation,
            "production_default_modified": self.production_default_modified,
            "permanent_global_view_cap": self.permanent_global_view_cap,
        }


def validate_i5_9a_for_i5_9b2(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(payload, "schema_version", 1, label="I5.9A admission")
    _require(payload, "phase", "I5.9A", label="I5.9A admission")
    _require(payload, "status", "remaining_depth_provider_robustness_inventory_and_scenarios_admitted", label="I5.9A admission")
    _require(payload, "pass", True, label="I5.9A admission")
    providers = _mapping(payload.get("provider_inventory"), label="I5.9A provider inventory")
    if set(providers) != set(I59A_PROVIDER_IMPLEMENTATIONS):
        raise ReconstructionValidationError("I5.9B2 requires the exact six-provider I5.9A inventory.")
    specs = provider_robustness_specs()
    for provider in I59B2_PROVIDERS:
        node = _mapping(providers.get(provider), label=f"I5.9A provider {provider}")
        _require(node, "batch", "I5.9B2", label=f"I5.9A provider {provider}")
        _require(node, "alignment_semantics", I59B2_ALIGNMENT_SEMANTICS[provider], label=f"I5.9A provider {provider}")
        _require(node, "native_depth_scale", "relative", label=f"I5.9A provider {provider}")
        _require(node, "fusion_capacity_drops_change_status", False, label=f"I5.9A provider {provider}")
        _require(node, "permanent_global_view_cap", None, label=f"I5.9A provider {provider}")
        if node.get("provider_id") != specs[provider]["provider_id"]:
            raise ReconstructionValidationError(f"I5.9A provider identity drifted for {provider}.")
    anchors = _mapping(payload.get("historical_g4_anchors"), label="I5.9A historical G4 anchors")
    low = _mapping(anchors.get("low_texture"), label="I5.9A low-texture anchor")
    for provider, expected in I59B2_HISTORICAL_G4_LOW_TEXTURE.items():
        _require(low, provider, expected, label="I5.9A low-texture anchor")
    matrix = _mapping(payload.get("scenario_matrix"), label="I5.9A scenario matrix")
    if set(matrix) != set(I59B2_SCENARIO_IDS):
        raise ReconstructionValidationError("I5.9B2 requires exactly the six I5.9A scenarios.")
    plan = _mapping(payload.get("batch_plan"), label="I5.9A batch plan")
    b2 = _mapping(plan.get("I5.9B2"), label="I5.9A I5.9B2 plan")
    if tuple(b2.get("providers", ())) != I59B2_PROVIDERS:
        raise ReconstructionValidationError("I5.9A B2 provider order drifted.")
    _require(b2, "authorized", True, label="I5.9A I5.9B2 plan")
    _require(b2, "started", False, label="I5.9A I5.9B2 plan")
    _require(b2, "execute_after_i5_9b1_freeze", True, label="I5.9A I5.9B2 plan")
    _require(b2, "real_provider_scenario_runs", I59B2_EXPECTED_RUN_COUNT, label="I5.9A I5.9B2 plan")
    decision = _mapping(payload.get("decision"), label="I5.9A decision")
    _require(decision, "i5_9a_complete", True, label="I5.9A decision")
    _require(decision, "i5_9b2_authorized", True, label="I5.9A decision")
    _require(decision, "i5_9b2_started", False, label="I5.9A decision")
    return deepcopy(dict(payload))


def validate_i5_9b1_for_i5_9b2(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(payload, "schema_version", 1, label="I5.9B1 replay")
    _require(payload, "phase", "I5.9B1", label="I5.9B1 replay")
    _require(payload, "status", "metric_like_single_image_robustness_replayed", label="I5.9B1 replay")
    _require(payload, "pass", True, label="I5.9B1 replay")
    _require(payload, "quality_status", "pass", label="I5.9B1 replay")
    if tuple(payload.get("providers", ())) != I59B1_PROVIDERS or tuple(payload.get("scenario_ids", ())) != I59B1_SCENARIO_IDS:
        raise ReconstructionValidationError("I5.9B1 predecessor provider/scenario identity drifted.")
    _require(payload, "expected_provider_scenario_runs", I59B1_EXPECTED_RUN_COUNT, label="I5.9B1 replay")
    _require(payload, "terminal_provider_scenario_runs", I59B1_EXPECTED_RUN_COUNT, label="I5.9B1 replay")
    counts = _mapping(payload.get("terminal_status_counts"), label="I5.9B1 status counts")
    if dict(counts) != {"pass": 24, "degraded": 0, "rejected": 0}:
        raise ReconstructionValidationError("I5.9B2 requires the frozen observed B1 result: 24 pass, 0 degraded, 0 rejected.")
    records = _mapping(payload.get("records"), label="I5.9B1 records")
    if set(records) != set(I59B1_SCENARIO_IDS):
        raise ReconstructionValidationError("I5.9B1 predecessor record matrix is incomplete.")
    observed = 0
    for sid in I59B1_SCENARIO_IDS:
        provider_records = _mapping(records.get(sid), label=f"I5.9B1 records {sid}")
        if set(provider_records) != set(I59B1_PROVIDERS):
            raise ReconstructionValidationError(f"I5.9B1 predecessor records are incomplete for {sid}.")
        for provider in I59B1_PROVIDERS:
            record = _mapping(provider_records.get(provider), label=f"I5.9B1 record {sid}/{provider}")
            _require(record, "runtime_status", "pass", label=f"I5.9B1 record {sid}/{provider}")
            observed += 1
    if observed != 24:
        raise ReconstructionValidationError("I5.9B1 predecessor must contain exactly 24 PASS records.")
    scope = _mapping(payload.get("scope"), label="I5.9B1 scope")
    for key in ("sfm_rerun", "mvs_rerun", "provider_comparison", "result_driven_retuning", "retry_until_pass", "thresholds_modified", "fusion_capacity_increased", "production_default_modified"):
        _require(scope, key, False, label="I5.9B1 scope")
    authority = _mapping(payload.get("authority"), label="I5.9B1 authority")
    _require(authority, "camera_pose_authority_owner", "pycolmap_sfm", label="I5.9B1 authority")
    _require(authority, "authoritative_sfm_modified", False, label="I5.9B1 authority")
    _require(authority, "metric_scale_claimed", False, label="I5.9B1 authority")
    _require(authority, "permanent_global_view_cap", None, label="I5.9B1 authority")
    decision = _mapping(payload.get("decision"), label="I5.9B1 decision")
    _require(decision, "i5_9b1_complete", True, label="I5.9B1 decision")
    _require(decision, "i5_9b2_authorized", True, label="I5.9B1 decision")
    _require(decision, "i5_9b2_started", False, label="I5.9B1 decision")
    _require(payload, "next", "I5.9B2 relative-depth single-image provider robustness replay.", label="I5.9B1 replay")
    return deepcopy(dict(payload))


def validate_i5_9b2_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(payload, "schema_version", I59B2_SCHEMA_VERSION, label="I5.9B2 receipt")
    _require(payload, "phase", I59B2_RECEIPT_PHASE, label="I5.9B2 receipt")
    _require(payload, "status", "pass", label="I5.9B2 receipt")
    _require(payload, "pass", True, label="I5.9B2 receipt")
    scope = _mapping(payload.get("scope"), label="I5.9B2 receipt scope")
    for key in ("focused_i5_9b2_tests", "b2_provider_regression", "i5_9b1_closure_regression", "i5_9a_admission_regression", "i5_6a_source_binding_regression", "historical_g4_anchor_regression", "authority_guard_regression", "import_probe", "runner_help_probe"):
        _require(scope, key, True, label="I5.9B2 receipt scope")
    for key in ("real_provider_inference_executed_by_receipt", "sfm_rerun", "mvs_rerun", "thresholds_modified", "production_default_modified", "result_driven_retuning", "retry_until_pass", "provider_comparison_started", "mp_sfm_started", "auto_policy_implemented", "hybrid_authorized"):
        _require(scope, key, False, label="I5.9B2 receipt scope")
    counts = _mapping(payload.get("counts"), label="I5.9B2 receipt counts")
    if int(counts.get("combined_exercised_tests", 0)) <= 0:
        raise ReconstructionValidationError("I5.9B2 receipt must record exercised tests.")
    return deepcopy(dict(payload))


def validate_i5_9b2_runtime_preflight(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(payload, "schema_version", I59B2_SCHEMA_VERSION, label="I5.9B2 preflight")
    _require(payload, "phase", I59B2_PHASE, label="I5.9B2 preflight")
    _require(payload, "status", "runtime_source_preflight_pass", label="I5.9B2 preflight")
    _require(payload, "pass", True, label="I5.9B2 preflight")
    if tuple(payload.get("providers", ())) != I59B2_PROVIDERS or tuple(payload.get("scenario_ids", ())) != I59B2_SCENARIO_IDS:
        raise ReconstructionValidationError("I5.9B2 preflight provider/scenario identity drifted.")
    _require(payload, "expected_provider_scenario_runs", I59B2_EXPECTED_RUN_COUNT, label="I5.9B2 preflight")
    evidence = _mapping(payload.get("evidence_bindings"), label="I5.9B2 preflight evidence bindings")
    for key in ("i5_9a_admission", "i5_6a_admission", "i5_9b1_replay", "replay_receipt"):
        node = _mapping(evidence.get(key), label=f"I5.9B2 preflight evidence {key}")
        if not str(node.get("path", "")).strip() or not _valid_sha(node.get("sha256")):
            raise ReconstructionValidationError(f"I5.9B2 preflight evidence binding is malformed: {key}.")
    historical = _mapping(payload.get("historical_g4_low_texture_anchor"), label="I5.9B2 historical anchor")
    for provider, expected in I59B2_HISTORICAL_G4_LOW_TEXTURE.items():
        _require(historical, provider, expected, label="I5.9B2 historical anchor")
    _require(historical, "preserved", True, label="I5.9B2 historical anchor")
    runtimes = _mapping(payload.get("runtime"), label="I5.9B2 preflight runtime")
    if set(runtimes) != set(I59B2_PROVIDERS):
        raise ReconstructionValidationError("I5.9B2 preflight must bind both B2 runtimes.")
    for provider in I59B2_PROVIDERS:
        node = _mapping(runtimes.get(provider), label=f"I5.9B2 runtime {provider}")
        _require(node, "available", True, label=f"I5.9B2 runtime {provider}")
        _require(node, "cuda_ready", True, label=f"I5.9B2 runtime {provider}")
        _require(node, "alignment_semantics", I59B2_ALIGNMENT_SEMANTICS[provider], label=f"I5.9B2 runtime {provider}")
        if not str(node.get("provider_id", "")).strip() or not str(node.get("worker_script", "")).strip():
            raise ReconstructionValidationError(f"I5.9B2 runtime {provider} provenance is incomplete.")
    plans = _mapping(payload.get("scenario_plans"), label="I5.9B2 scenario plans")
    if set(plans) != set(I59B2_SCENARIO_IDS):
        raise ReconstructionValidationError("I5.9B2 preflight scenario-plan matrix is incomplete.")
    count = 0
    for sid in I59B2_SCENARIO_IDS:
        scenario = _mapping(plans.get(sid), label=f"I5.9B2 scenario {sid}")
        _require(scenario, "source_integrity_verified", True, label=f"I5.9B2 scenario {sid}")
        _require(scenario, "authoritative_sfm_reused_without_rerun", True, label=f"I5.9B2 scenario {sid}")
        providers = _mapping(scenario.get("providers"), label=f"I5.9B2 scenario {sid} providers")
        if set(providers) != set(I59B2_PROVIDERS):
            raise ReconstructionValidationError(f"I5.9B2 scenario {sid} provider plans are incomplete.")
        for provider, node_raw in providers.items():
            node = _mapping(node_raw, label=f"I5.9B2 preflight {sid}/{provider}")
            source = int(node.get("source_view_count", 0))
            selected = int(node.get("selected_view_count", 0))
            if source <= 0 or selected <= 0 or selected > source:
                raise ReconstructionValidationError(f"I5.9B2 preflight {sid}/{provider} adaptive plan is invalid.")
            _require(node, "provider_inference_started", False, label=f"I5.9B2 preflight {sid}/{provider}")
            count += 1
    if count != I59B2_EXPECTED_RUN_COUNT:
        raise ReconstructionValidationError("I5.9B2 preflight did not resolve exactly 12 provider/scenario plans.")
    scope = _mapping(payload.get("scope"), label="I5.9B2 preflight scope")
    for key in ("provider_inference_started", "sfm_rerun", "mvs_rerun", "provider_comparison_started"):
        _require(scope, key, False, label="I5.9B2 preflight scope")
    authority = _mapping(payload.get("authority"), label="I5.9B2 preflight authority")
    _require(authority, "camera_pose_authority_owner", "pycolmap_sfm", label="I5.9B2 preflight authority")
    _require(authority, "permanent_global_view_cap", None, label="I5.9B2 preflight authority")
    for key in ("authoritative_sfm_modified", "provider_selected", "provider_promoted", "metric_scale_claimed", "measurement_authoritative", "certified_metrology", "surface_mesh_generated"):
        _require(authority, key, False, label="I5.9B2 preflight authority")
    return deepcopy(dict(payload))


def validate_i5_9b2_provider_scenario_record(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(payload, "schema_version", I59B2_SCHEMA_VERSION, label="I5.9B2 record")
    _require(payload, "phase", I59B2_PHASE, label="I5.9B2 record")
    provider = str(payload.get("provider_implementation", ""))
    sid = str(payload.get("scenario_id", ""))
    if provider not in I59B2_PROVIDERS or sid not in I59B2_SCENARIO_IDS:
        raise ReconstructionValidationError("I5.9B2 provider/scenario record identity is not admitted.")
    _require(payload, "alignment_semantics", I59B2_ALIGNMENT_SEMANTICS[provider], label="I5.9B2 record")
    status = str(payload.get("runtime_status", ""))
    if status not in I59B2_TERMINAL_STATUSES:
        raise ReconstructionValidationError("I5.9B2 record is not a terminal scientific outcome.")
    source = int(payload.get("source_registered_views", 0))
    selected = int(payload.get("selected_views", 0))
    processed = int(payload.get("processed_views", 0))
    accepted = int(payload.get("accepted_views", 0))
    if source <= 0 or selected <= 0 or selected > source or processed != selected or accepted < 0 or accepted > selected:
        raise ReconstructionValidationError("I5.9B2 provider/scenario accounting is inconsistent.")
    _require(payload, "adaptive_selection", True, label="I5.9B2 record")
    _require(payload, "input_integrity_revalidated_after", True, label="I5.9B2 record")
    _require(payload, "fusion_capacity_drops_change_status", False, label="I5.9B2 record")
    if status == "rejected":
        rejection = _mapping(payload.get("rejection"), label="I5.9B2 rejection")
        _require(rejection, "recognized_existing_quality_gate", True, label="I5.9B2 rejection")
        _require(rejection, "terminal", True, label="I5.9B2 rejection")
        _require(rejection, "threshold_changed", False, label="I5.9B2 rejection")
    authority = _mapping(payload.get("authority"), label="I5.9B2 record authority")
    _require(authority, "camera_pose_authority_owner", "pycolmap_sfm", label="I5.9B2 record authority")
    _require(authority, "permanent_global_view_cap", None, label="I5.9B2 record authority")
    for key in ("authoritative_sfm_modified", "sfm_rerun", "bundle_adjustment_rerun", "metric_scale_claimed", "measurement_authoritative", "certified_metrology", "surface_mesh_generated", "production_filter_policy_changed", "provider_promotion_performed"):
        _require(authority, key, False, label="I5.9B2 record authority")
    return deepcopy(dict(payload))


def build_i5_9b2_replay(
    *,
    i5_9a_admission: Mapping[str, Any],
    i5_9a_binding: I51EvidenceBinding,
    i5_6a_binding: I51EvidenceBinding,
    i5_9b1_replay: Mapping[str, Any],
    i5_9b1_binding: I51EvidenceBinding,
    replay_receipt: Mapping[str, Any],
    replay_receipt_binding: I51EvidenceBinding,
    runtime_preflight: Mapping[str, Any],
    runtime_preflight_binding: I51EvidenceBinding,
    records: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    validate_i5_9a_for_i5_9b2(i5_9a_admission)
    validate_i5_9b1_for_i5_9b2(i5_9b1_replay)
    validate_i5_9b2_receipt(replay_receipt)
    validate_i5_9b2_runtime_preflight(runtime_preflight)
    _validate_binding(i5_9a_binding, label="I5.9A", expected_sha256=I59B2_I59A_FROZEN_SHA256)
    _validate_binding(i5_6a_binding, label="I5.6A", expected_sha256=I59B2_I56A_FROZEN_SHA256)
    _validate_binding(i5_9b1_binding, label="I5.9B1 replay")
    _validate_binding(replay_receipt_binding, label="I5.9B2 receipt")
    _validate_binding(runtime_preflight_binding, label="I5.9B2 runtime preflight")

    if set(records) != set(I59B2_SCENARIO_IDS):
        raise ReconstructionValidationError("I5.9B2 replay requires exactly six scenario groups.")
    compact: dict[str, dict[str, Any]] = {}
    statuses: list[str] = []
    total = 0
    for sid in I59B2_SCENARIO_IDS:
        providers = _mapping(records.get(sid), label=f"I5.9B2 records {sid}")
        if set(providers) != set(I59B2_PROVIDERS):
            raise ReconstructionValidationError(f"I5.9B2 scenario {sid} must contain exactly both B2 providers.")
        compact[sid] = {}
        for provider in I59B2_PROVIDERS:
            record = validate_i5_9b2_provider_scenario_record(
                _mapping(providers.get(provider), label=f"I5.9B2 record {sid}/{provider}")
            )
            compact[sid][provider] = record
            statuses.append(str(record["runtime_status"]))
            total += 1
    if total != I59B2_EXPECTED_RUN_COUNT:
        raise ReconstructionValidationError("I5.9B2 replay record count must be exactly 12.")
    counts = {status: statuses.count(status) for status in I59B2_TERMINAL_STATUSES}
    quality_status = "pass" if counts["pass"] == total else "mixed_terminal_quality"
    return {
        "schema_version": I59B2_SCHEMA_VERSION,
        "phase": I59B2_PHASE,
        "status": "relative_depth_single_image_robustness_replayed",
        "pass": True,
        "quality_status": quality_status,
        "providers": list(I59B2_PROVIDERS),
        "scenario_ids": list(I59B2_SCENARIO_IDS),
        "expected_provider_scenario_runs": I59B2_EXPECTED_RUN_COUNT,
        "terminal_provider_scenario_runs": total,
        "terminal_status_counts": counts,
        "historical_g4_low_texture_anchor": {
            **I59B2_HISTORICAL_G4_LOW_TEXTURE,
            "preserved": True,
        },
        "execution_policy": I59B2ReplayPolicy().as_dict(),
        "evidence_bindings": {
            "i5_9a_admission": i5_9a_binding.as_dict(),
            "i5_6a_admission": i5_6a_binding.as_dict(),
            "i5_9b1_replay": i5_9b1_binding.as_dict(),
            "replay_receipt": replay_receipt_binding.as_dict(),
            "runtime_preflight": runtime_preflight_binding.as_dict(),
        },
        "runtime_preflight": deepcopy(dict(runtime_preflight)),
        "records": compact,
        "scope": {
            "real_relative_depth_single_image_provider_replay": True,
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
            "historical_g4_rejections_overwritten": False,
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
            "i5_9b2_complete": True,
            "i5_9c_authorized": True,
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
        "next": "I5.9C remaining depth-provider robustness qualification and closure.",
    }


__all__ = [
    "I59B2_SCHEMA_VERSION", "I59B2_PHASE", "I59B2_RECEIPT_PHASE",
    "I59B2_I59A_FROZEN_SHA256", "I59B2_I56A_FROZEN_SHA256",
    "I59B2_PROVIDERS", "I59B2_SCENARIO_IDS", "I59B2_EXPECTED_RUN_COUNT",
    "I59B2_TERMINAL_STATUSES", "I59B2_HISTORICAL_G4_LOW_TEXTURE",
    "I59B2_ALIGNMENT_SEMANTICS", "I59B2ReplayPolicy",
    "validate_i5_9a_for_i5_9b2", "validate_i5_9b1_for_i5_9b2",
    "validate_i5_9b2_receipt", "validate_i5_9b2_runtime_preflight",
    "validate_i5_9b2_provider_scenario_record", "build_i5_9b2_replay",
]

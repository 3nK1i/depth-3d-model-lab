"""I5.9C remaining depth-provider robustness qualification and closure.

I5.9C is evidence-only. It consumes the frozen I5.9A admission plus the exact
I5.9B1 and I5.9B2 production-native replay artifacts. It does not execute any
depth provider and does not rerun SfM, bundle adjustment, or MVS.

The closure preserves two distinct evidence axes:

* Current frozen Castle/PyCOLMAP replay evidence: B1 is 24 PASS; B2 is
  11 PASS and one DEGRADED Marigold dynamic-proxy result caused by one
  alignment-rejected view (18/19 accepted), with no fusion-capacity drop.
* Historical I4.3G4 evidence: Depth Anything V2 and Marigold remain REJECTED
  on the historical Low Texture scenario. Current Castle results do not
  overwrite those historical scenario-specific outcomes.

No provider comparison/ranking, universal winner, global scalar score, metric
scale, metrology, threshold relaxation, capacity increase, result-driven
retuning, retry-until-pass, Auto, or Hybrid policy is created here.
"""
from __future__ import annotations

from copy import deepcopy
from math import isclose
from typing import Any, Mapping

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding
from core.reconstruction.i5_remaining_depth_provider_robustness import (
    I59A_B1_PROVIDERS,
    I59A_B2_PROVIDERS,
    I59A_FROZEN_G4_STATUSES,
    I59A_PROVIDER_IMPLEMENTATIONS,
    I59A_SCENARIO_IDS,
)
from core.reconstruction.i5_relative_depth_provider_robustness_replay import (
    I59B2_ALIGNMENT_SEMANTICS,
    I59B2_EXPECTED_RUN_COUNT,
    I59B2_HISTORICAL_G4_LOW_TEXTURE,
    I59B2_PHASE,
    I59B2_PROVIDERS,
    I59B2_SCHEMA_VERSION,
    I59B2_SCENARIO_IDS,
    validate_i5_9a_for_i5_9b2,
    validate_i5_9b1_for_i5_9b2,
    validate_i5_9b2_provider_scenario_record,
)

I59C_SCHEMA_VERSION = 1
I59C_PHASE = "I5.9C"
I59C_RECEIPT_PHASE = "I5.9C_QUALIFICATION_RECEIPT"
I59C_I59A_FROZEN_SHA256 = "50ae4a51d57390e6b20b16aa2fa83c65b110a5ba29803853af279ffa3c4a7cff"
I59C_I59B1_FROZEN_SHA256 = "e1e66f6bebcc6b06cc6b48861da093234f3e7e9ca4c593b81569f243756585fa"
I59C_B1_STATUS_COUNTS = {"pass": 24, "degraded": 0, "rejected": 0}
I59C_B2_STATUS_COUNTS = {"pass": 11, "degraded": 1, "rejected": 0}
I59C_COMBINED_STATUS_COUNTS = {"pass": 35, "degraded": 1, "rejected": 0}
I59C_MARIGOLD_DEGRADED_SCENARIO = "castle_dynamic_proxy"
I59C_MARIGOLD_SELECTED_VIEWS = 19
I59C_MARIGOLD_ACCEPTED_VIEWS = 18
I59C_MARIGOLD_ALIGNMENT_REJECTED_VIEWS = 1


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
            f"{label} SHA256 drifted: expected={expected_sha256} actual={binding.sha256}"
        )


def _validate_mapping_binding(
    value: Any,
    *,
    label: str,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    node = deepcopy(dict(_mapping(value, label=label)))
    if not str(node.get("path", "")).strip() or not _valid_sha(node.get("sha256")):
        raise ReconstructionValidationError(f"{label} evidence binding is malformed.")
    if expected_sha256 is not None and str(node["sha256"]).lower() != expected_sha256.lower():
        raise ReconstructionValidationError(
            f"{label} SHA256 drifted: expected={expected_sha256} actual={node['sha256']}"
        )
    return node


def _status_counts(records: Mapping[str, Any], providers: tuple[str, ...]) -> dict[str, int]:
    counts = {"pass": 0, "degraded": 0, "rejected": 0}
    for sid in I59A_SCENARIO_IDS:
        scenario = _mapping(records.get(sid), label=f"records {sid}")
        for provider in providers:
            record = _mapping(scenario.get(provider), label=f"record {sid}/{provider}")
            status = str(record.get("runtime_status", ""))
            if status not in counts:
                raise ReconstructionValidationError(
                    f"record {sid}/{provider} has non-terminal status {status!r}."
                )
            counts[status] += 1
    return counts


def validate_i5_9b1_for_i5_9c(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the frozen observed B1 replay used by I5.9C."""

    replay = validate_i5_9b1_for_i5_9b2(payload)
    bindings = _mapping(replay.get("evidence_bindings"), label="I5.9B1 evidence bindings")
    _validate_mapping_binding(
        bindings.get("i5_9a_admission"),
        label="I5.9B1 I5.9A binding",
        expected_sha256=I59C_I59A_FROZEN_SHA256,
    )

    records = _mapping(replay.get("records"), label="I5.9B1 records")
    for sid in I59A_SCENARIO_IDS:
        scenario = _mapping(records.get(sid), label=f"I5.9B1 records {sid}")
        for provider in I59A_B1_PROVIDERS:
            record = _mapping(scenario.get(provider), label=f"I5.9B1 record {sid}/{provider}")
            _require(record, "runtime_status", "pass", label=f"I5.9B1 record {sid}/{provider}")
            selected = int(record.get("selected_views", 0))
            processed = int(record.get("processed_views", 0))
            accepted = int(record.get("accepted_views", 0))
            if selected <= 0 or processed != selected or accepted != selected:
                raise ReconstructionValidationError(
                    f"I5.9B1 {sid}/{provider} no longer preserves full selected-view processing and acceptance."
                )
            if int(record.get("fusion_capacity_drops", -1)) != 0:
                raise ReconstructionValidationError(
                    f"I5.9B1 {sid}/{provider} fusion-capacity observation drifted from zero."
                )
    return deepcopy(dict(replay))


def validate_i5_9b2_for_i5_9c(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact observed B2 mixed-quality production replay."""

    _require(payload, "schema_version", I59B2_SCHEMA_VERSION, label="I5.9B2 replay")
    _require(payload, "phase", I59B2_PHASE, label="I5.9B2 replay")
    _require(
        payload,
        "status",
        "relative_depth_single_image_robustness_replayed",
        label="I5.9B2 replay",
    )
    _require(payload, "pass", True, label="I5.9B2 replay")
    _require(payload, "quality_status", "mixed_terminal_quality", label="I5.9B2 replay")
    if tuple(payload.get("providers", ())) != I59B2_PROVIDERS:
        raise ReconstructionValidationError("I5.9C requires the exact B2 provider order.")
    if tuple(payload.get("scenario_ids", ())) != I59B2_SCENARIO_IDS:
        raise ReconstructionValidationError("I5.9C requires the exact six B2 Castle scenarios.")
    _require(payload, "expected_provider_scenario_runs", I59B2_EXPECTED_RUN_COUNT, label="I5.9B2 replay")
    _require(payload, "terminal_provider_scenario_runs", I59B2_EXPECTED_RUN_COUNT, label="I5.9B2 replay")
    counts = dict(_mapping(payload.get("terminal_status_counts"), label="I5.9B2 status counts"))
    if counts != I59C_B2_STATUS_COUNTS:
        raise ReconstructionValidationError(
            f"I5.9C requires the frozen B2 status counts {I59C_B2_STATUS_COUNTS}; found {counts}."
        )

    historical = _mapping(
        payload.get("historical_g4_low_texture_anchor"),
        label="I5.9B2 historical Low Texture anchor",
    )
    for provider, expected in I59B2_HISTORICAL_G4_LOW_TEXTURE.items():
        _require(historical, provider, expected, label="I5.9B2 historical Low Texture anchor")
    _require(historical, "preserved", True, label="I5.9B2 historical Low Texture anchor")

    bindings = _mapping(payload.get("evidence_bindings"), label="I5.9B2 evidence bindings")
    _validate_mapping_binding(
        bindings.get("i5_9a_admission"),
        label="I5.9B2 I5.9A binding",
        expected_sha256=I59C_I59A_FROZEN_SHA256,
    )
    _validate_mapping_binding(
        bindings.get("i5_9b1_replay"),
        label="I5.9B2 I5.9B1 binding",
        expected_sha256=I59C_I59B1_FROZEN_SHA256,
    )

    preflight = _mapping(payload.get("runtime_preflight"), label="I5.9B2 runtime preflight")
    _require(preflight, "pass", True, label="I5.9B2 runtime preflight")
    _require(preflight, "status", "runtime_source_preflight_pass", label="I5.9B2 runtime preflight")
    preflight_bindings = _mapping(preflight.get("evidence_bindings"), label="I5.9B2 preflight evidence bindings")
    _validate_mapping_binding(
        preflight_bindings.get("i5_9b1_replay"),
        label="I5.9B2 preflight I5.9B1 binding",
        expected_sha256=I59C_I59B1_FROZEN_SHA256,
    )

    records = _mapping(payload.get("records"), label="I5.9B2 records")
    if set(records) != set(I59B2_SCENARIO_IDS):
        raise ReconstructionValidationError("I5.9B2 record matrix is incomplete.")

    observed = {"pass": 0, "degraded": 0, "rejected": 0}
    for sid in I59B2_SCENARIO_IDS:
        providers = _mapping(records.get(sid), label=f"I5.9B2 records {sid}")
        if set(providers) != set(I59B2_PROVIDERS):
            raise ReconstructionValidationError(f"I5.9B2 records are incomplete for {sid}.")
        for provider in I59B2_PROVIDERS:
            record = validate_i5_9b2_provider_scenario_record(
                _mapping(providers.get(provider), label=f"I5.9B2 record {sid}/{provider}")
            )
            status = str(record["runtime_status"])
            observed[status] += 1
            selected = int(record["selected_views"])
            accepted = int(record["accepted_views"])
            if int(record.get("fusion_capacity_drops", -1)) != 0:
                raise ReconstructionValidationError(
                    f"I5.9C observed no B2 fusion-capacity drops; {sid}/{provider} drifted."
                )

            is_marigold_dynamic = (
                sid == I59C_MARIGOLD_DEGRADED_SCENARIO and provider == "marigold"
            )
            if is_marigold_dynamic:
                _require(record, "runtime_status", "degraded", label="I5.9B2 Marigold dynamic proxy")
                _require(record, "alignment_semantics", I59B2_ALIGNMENT_SEMANTICS["marigold"], label="I5.9B2 Marigold dynamic proxy")
                if selected != I59C_MARIGOLD_SELECTED_VIEWS or accepted != I59C_MARIGOLD_ACCEPTED_VIEWS:
                    raise ReconstructionValidationError(
                        "I5.9C must preserve the observed Marigold dynamic-proxy 18/19 alignment acceptance."
                    )
                metrics = _mapping(record.get("native_metrics"), label="I5.9B2 Marigold dynamic metrics")
                fraction = float(metrics.get("alignment_accepted_fraction", -1.0))
                if not isclose(fraction, 18.0 / 19.0, rel_tol=0.0, abs_tol=1e-12):
                    raise ReconstructionValidationError(
                        "I5.9C Marigold dynamic-proxy alignment accepted fraction drifted from 18/19."
                    )
                result = _mapping(record.get("execution_result"), label="I5.9B2 Marigold dynamic execution result")
                _require(result, "status", "degraded", label="I5.9B2 Marigold dynamic execution result")
                reasons = tuple(result.get("degradation_reasons", ()))
                if reasons != ("alignment_rejected_views:1",):
                    raise ReconstructionValidationError(
                        "I5.9C must preserve the observed single alignment-rejected Marigold view."
                    )
                _require(result, "failure_reason", "", label="I5.9B2 Marigold dynamic execution result")
            else:
                _require(record, "runtime_status", "pass", label=f"I5.9B2 record {sid}/{provider}")
                if accepted != selected:
                    raise ReconstructionValidationError(
                        f"I5.9C {sid}/{provider} no longer preserves full selected-view acceptance."
                    )
                execution_result = record.get("execution_result")
                if isinstance(execution_result, Mapping):
                    reasons = tuple(execution_result.get("degradation_reasons", ()))
                    if reasons:
                        raise ReconstructionValidationError(
                            f"I5.9C PASS record {sid}/{provider} unexpectedly contains degradation reasons."
                        )

    if observed != I59C_B2_STATUS_COUNTS:
        raise ReconstructionValidationError(
            f"I5.9C B2 record matrix counts drifted: expected {I59C_B2_STATUS_COUNTS}, found {observed}."
        )

    scope = _mapping(payload.get("scope"), label="I5.9B2 scope")
    for key in (
        "sfm_rerun", "mvs_rerun", "provider_comparison", "result_driven_retuning",
        "retry_until_pass", "thresholds_modified", "fusion_capacity_increased",
        "production_default_modified", "historical_g4_rejections_overwritten",
        "mp_sfm_started", "auto_policy_implemented", "hybrid_authorized",
    ):
        _require(scope, key, False, label="I5.9B2 scope")

    authority = _mapping(payload.get("authority"), label="I5.9B2 authority")
    _require(authority, "camera_pose_authority_owner", "pycolmap_sfm", label="I5.9B2 authority")
    _require(authority, "single_authoritative_sfm_frame_preserved", True, label="I5.9B2 authority")
    _require(authority, "permanent_global_view_cap", None, label="I5.9B2 authority")
    for key in (
        "authoritative_sfm_modified", "provider_selected", "provider_promoted",
        "provider_ranking_created", "metric_scale_claimed", "measurement_authoritative",
        "certified_metrology", "surface_mesh_generated",
    ):
        _require(authority, key, False, label="I5.9B2 authority")

    decision = _mapping(payload.get("decision"), label="I5.9B2 decision")
    for key in ("i5_9a_complete", "i5_9b1_complete", "i5_9b2_complete", "i5_9c_authorized"):
        _require(decision, key, True, label="I5.9B2 decision")
    for key in (
        "i5_9c_started", "i5_10_started", "provider_selected", "provider_promoted",
        "provider_ranking_created", "universal_provider_winner_declared",
        "global_scalar_score_created", "auto_policy_implemented", "hybrid_authorized",
        "mp_sfm_authorized", "threshold_relaxation_authorized",
        "fusion_capacity_increase_authorized", "production_default_modified",
    ):
        _require(decision, key, False, label="I5.9B2 decision")
    _require(
        payload,
        "next",
        "I5.9C remaining depth-provider robustness qualification and closure.",
        label="I5.9B2 replay",
    )
    return deepcopy(dict(payload))


def validate_i5_9c_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(payload, "schema_version", I59C_SCHEMA_VERSION, label="I5.9C receipt")
    _require(payload, "phase", I59C_RECEIPT_PHASE, label="I5.9C receipt")
    _require(payload, "status", "pass", label="I5.9C receipt")
    _require(payload, "pass", True, label="I5.9C receipt")
    scope = _mapping(payload.get("scope"), label="I5.9C receipt scope")
    for key in (
        "focused_i5_9c_tests", "i5_9a_admission_regression", "i5_9b1_replay_regression",
        "i5_9b2_replay_regression", "historical_g4_preservation_regression",
        "mixed_terminal_quality_regression", "authority_guard_regression",
        "roadmap_guard_regression", "import_probe", "runner_help_probe",
    ):
        _require(scope, key, True, label="I5.9C receipt scope")
    for key in (
        "provider_inference", "sfm_rerun", "mvs_rerun", "provider_comparison_started",
        "thresholds_modified", "fusion_capacity_increased", "production_default_modified",
        "result_driven_retuning", "retry_until_pass", "mp_sfm_started",
        "auto_policy_implemented", "hybrid_authorized",
    ):
        _require(scope, key, False, label="I5.9C receipt scope")
    counts = _mapping(payload.get("counts"), label="I5.9C receipt counts")
    focused = int(counts.get("focused_tests", 0))
    unit = int(counts.get("predecessor_unit_tests", 0))
    integ = int(counts.get("predecessor_integration_tests", 0))
    stress = int(counts.get("predecessor_stress_tests", 0))
    predecessor = int(counts.get("predecessor_total_tests", 0))
    combined = int(counts.get("combined_exercised_tests", 0))
    if focused <= 0 or min(unit, integ, stress) <= 0 or predecessor != unit + integ + stress or combined != focused + predecessor:
        raise ReconstructionValidationError("I5.9C receipt test counts are inconsistent.")
    rcs = _mapping(payload.get("return_codes"), label="I5.9C receipt return codes")
    for key in (
        "focused", "predecessor_unit", "predecessor_integration", "predecessor_stress",
        "import_probe", "runner_help_probe",
    ):
        if int(rcs.get(key, -1)) != 0:
            raise ReconstructionValidationError(f"I5.9C receipt return code is non-zero: {key}")
    return deepcopy(dict(payload))


def _provider_status_counts(records: Mapping[str, Any], provider: str) -> dict[str, int]:
    counts = {"pass": 0, "degraded": 0, "rejected": 0}
    for sid in I59A_SCENARIO_IDS:
        scenario = _mapping(records.get(sid), label=f"records {sid}")
        record = _mapping(scenario.get(provider), label=f"record {sid}/{provider}")
        counts[str(record["runtime_status"])] += 1
    return counts


def build_i5_9c_remaining_depth_provider_robustness_closure(
    *,
    i5_9a_admission: Mapping[str, Any],
    i5_9a_binding: I51EvidenceBinding,
    i5_9b1_replay: Mapping[str, Any],
    i5_9b1_binding: I51EvidenceBinding,
    i5_9b2_replay: Mapping[str, Any],
    i5_9b2_binding: I51EvidenceBinding,
    qualification_receipt: Mapping[str, Any],
    qualification_receipt_binding: I51EvidenceBinding,
) -> dict[str, Any]:
    admission = validate_i5_9a_for_i5_9b2(i5_9a_admission)
    b1 = validate_i5_9b1_for_i5_9c(i5_9b1_replay)
    b2 = validate_i5_9b2_for_i5_9c(i5_9b2_replay)
    receipt = validate_i5_9c_receipt(qualification_receipt)

    _validate_binding(i5_9a_binding, label="I5.9C I5.9A", expected_sha256=I59C_I59A_FROZEN_SHA256)
    _validate_binding(i5_9b1_binding, label="I5.9C I5.9B1", expected_sha256=I59C_I59B1_FROZEN_SHA256)
    _validate_binding(i5_9b2_binding, label="I5.9C I5.9B2")
    _validate_binding(qualification_receipt_binding, label="I5.9C receipt")

    b1_a = _validate_mapping_binding(
        _mapping(b1.get("evidence_bindings"), label="I5.9B1 evidence bindings").get("i5_9a_admission"),
        label="I5.9B1 I5.9A binding",
        expected_sha256=I59C_I59A_FROZEN_SHA256,
    )
    b2_a = _validate_mapping_binding(
        _mapping(b2.get("evidence_bindings"), label="I5.9B2 evidence bindings").get("i5_9a_admission"),
        label="I5.9B2 I5.9A binding",
        expected_sha256=I59C_I59A_FROZEN_SHA256,
    )
    b2_b1 = _validate_mapping_binding(
        _mapping(b2.get("evidence_bindings"), label="I5.9B2 evidence bindings").get("i5_9b1_replay"),
        label="I5.9B2 I5.9B1 binding",
        expected_sha256=I59C_I59B1_FROZEN_SHA256,
    )
    if b1_a["sha256"].lower() != i5_9a_binding.sha256.lower() or b2_a["sha256"].lower() != i5_9a_binding.sha256.lower():
        raise ReconstructionValidationError("I5.9C predecessor chain does not bind one exact I5.9A admission.")
    if b2_b1["sha256"].lower() != i5_9b1_binding.sha256.lower():
        raise ReconstructionValidationError("I5.9C B2 predecessor binding does not match the independently verified B1 artifact.")

    b1_records = _mapping(b1.get("records"), label="I5.9B1 records")
    b2_records = _mapping(b2.get("records"), label="I5.9B2 records")
    b1_counts = _status_counts(b1_records, I59A_B1_PROVIDERS)
    b2_counts = _status_counts(b2_records, I59A_B2_PROVIDERS)
    if b1_counts != I59C_B1_STATUS_COUNTS or b2_counts != I59C_B2_STATUS_COUNTS:
        raise ReconstructionValidationError("I5.9C predecessor status matrices drifted during closure construction.")

    historical = deepcopy(dict(_mapping(admission.get("historical_g4_anchors"), label="I5.9A historical G4 anchors")))
    expected_historical = {
        "complete_room": deepcopy(I59A_FROZEN_G4_STATUSES["g4_single_image_complete_room"]),
        "low_texture": deepcopy(I59A_FROZEN_G4_STATUSES["g4_single_image_low_texture"]),
        "weak_overlap": deepcopy(I59A_FROZEN_G4_STATUSES["g4_single_image_weak_overlap"]),
    }
    if historical != expected_historical:
        raise ReconstructionValidationError("I5.9C historical I4.3G4 status matrix drifted from I5.9A.")

    provider_qualification: dict[str, Any] = {}
    for provider in I59A_PROVIDER_IMPLEMENTATIONS:
        records = b1_records if provider in I59A_B1_PROVIDERS else b2_records
        counts = _provider_status_counts(records, provider)
        degraded = [
            sid for sid in I59A_SCENARIO_IDS
            if _mapping(_mapping(records.get(sid), label=f"records {sid}").get(provider), label=f"record {sid}/{provider}").get("runtime_status") == "degraded"
        ]
        rejected = [
            sid for sid in I59A_SCENARIO_IDS
            if _mapping(_mapping(records.get(sid), label=f"records {sid}").get(provider), label=f"record {sid}/{provider}").get("runtime_status") == "rejected"
        ]
        current_quality = "pass" if counts["pass"] == len(I59A_SCENARIO_IDS) else "mixed_terminal_quality"
        provider_qualification[provider] = {
            "current_castle_scenario_count": len(I59A_SCENARIO_IDS),
            "current_castle_status_counts": counts,
            "current_castle_quality_status": current_quality,
            "current_castle_degraded_scenarios": degraded,
            "current_castle_rejected_scenarios": rejected,
            "historical_g4_complete_room_status": historical["complete_room"][provider],
            "historical_g4_low_texture_status": historical["low_texture"][provider],
            "historical_g4_weak_overlap_status": historical["weak_overlap"][provider],
            "current_castle_and_historical_g4_are_distinct_evidence_axes": True,
            "universal_robustness_claimed": False,
            "provider_rank_assigned": False,
        }

    marigold_dynamic = _mapping(
        _mapping(b2_records[I59C_MARIGOLD_DEGRADED_SCENARIO], label="B2 dynamic records")["marigold"],
        label="B2 Marigold dynamic record",
    )
    marigold_result = _mapping(marigold_dynamic.get("execution_result"), label="B2 Marigold dynamic execution result")
    marigold_metrics = _mapping(marigold_dynamic.get("native_metrics"), label="B2 Marigold dynamic native metrics")

    return {
        "schema_version": I59C_SCHEMA_VERSION,
        "phase": I59C_PHASE,
        "status": "remaining_depth_provider_robustness_qualified_and_closed",
        "pass": True,
        "quality_status": "mixed_terminal_quality",
        "scope": {
            "evidence_only_qualification_and_closure": True,
            "i5_9a_admission_reused": True,
            "i5_9b1_real_replay_reused": True,
            "i5_9b2_real_replay_reused": True,
            "provider_inference": False,
            "sfm_rerun": False,
            "bundle_adjustment_rerun": False,
            "mvs_rerun": False,
            "source_rematerialized": False,
            "provider_comparison_started": False,
            "thresholds_modified": False,
            "fusion_capacity_increased": False,
            "production_default_modified": False,
            "result_driven_retuning": False,
            "retry_until_pass": False,
            "mp_sfm_started": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
        },
        "evidence_bindings": {
            "i5_9a_admission": i5_9a_binding.as_dict(),
            "i5_9b1_real_replay": i5_9b1_binding.as_dict(),
            "i5_9b2_real_replay": i5_9b2_binding.as_dict(),
            "i5_9c_qualification_receipt": qualification_receipt_binding.as_dict(),
        },
        "provenance": {
            "i5_9a_admission_preserved": True,
            "i5_9b1_24_pass_matrix_preserved": True,
            "i5_9b2_11_pass_1_degraded_matrix_preserved": True,
            "historical_g4_status_matrix_preserved": True,
            "historical_g4_rejections_overwritten": False,
            "marigold_dynamic_proxy_alignment_degradation_preserved": True,
            "fusion_capacity_degradation_substituted_for_alignment_degradation": False,
            "result_driven_retuning": False,
            "retry_until_pass": False,
        },
        "qualification": {
            "operating_envelope": "bounded_current_castle_robustness_with_one_preserved_marigold_dynamic_proxy_alignment_degradation_and_historical_g4_low_texture_rejections_preserved",
            "provider_count": len(I59A_PROVIDER_IMPLEMENTATIONS),
            "scenario_count": len(I59A_SCENARIO_IDS),
            "b1_terminal_runs": 24,
            "b1_status_counts": b1_counts,
            "b2_terminal_runs": 12,
            "b2_status_counts": b2_counts,
            "combined_terminal_runs": 36,
            "combined_status_counts": deepcopy(I59C_COMBINED_STATUS_COUNTS),
            "all_36_current_provider_scenario_attempts_terminal": True,
            "current_castle_rejected_run_count": 0,
            "current_castle_degraded_run_count": 1,
            "current_castle_degraded_provider": "marigold",
            "current_castle_degraded_scenario": I59C_MARIGOLD_DEGRADED_SCENARIO,
            "provider_comparison_performed": False,
            "provider_ranking_claimed": False,
            "universal_provider_winner_claimed": False,
            "global_scalar_score_created": False,
            "universal_single_image_provider_robustness_claimed": False,
            "metric_scale_claimed": False,
            "runtime_performance_qualification_claimed": False,
            "resource_robustness_qualification_claimed": False,
            "cross_gpu_equivalence_qualification_claimed": False,
            "ground_truth_dense_depth_available": False,
            "ground_truth_surface_accuracy_available": False,
            "ground_truth_surface_accuracy_claimed": False,
        },
        "provider_qualification": provider_qualification,
        "historical_g4_anchors": historical,
        "observed_limitations": {
            "marigold_castle_dynamic_proxy": {
                "runtime_status": "degraded",
                "alignment_semantics": marigold_dynamic["alignment_semantics"],
                "selected_views": int(marigold_dynamic["selected_views"]),
                "processed_views": int(marigold_dynamic["processed_views"]),
                "accepted_views": int(marigold_dynamic["accepted_views"]),
                "alignment_rejected_views": int(marigold_dynamic["selected_views"]) - int(marigold_dynamic["accepted_views"]),
                "alignment_accepted_fraction": float(marigold_metrics["alignment_accepted_fraction"]),
                "alignment_median_relative_residual": marigold_metrics.get("alignment_median_relative_residual"),
                "fusion_capacity_drops": int(marigold_dynamic["fusion_capacity_drops"]),
                "degradation_reasons": list(marigold_result.get("degradation_reasons", ())),
                "failure_reason": str(marigold_result.get("failure_reason", "")),
                "alignment_rejection_is_current_bounded_robustness_evidence": True,
                "exact_per_view_rejection_cause_beyond_existing_alignment_gate_claimed": False,
            },
            "historical_depth_anything_v2_low_texture_rejection_remains_active": True,
            "historical_marigold_low_texture_rejection_remains_active": True,
            "current_castle_suite_does_not_replace_historical_g4_scenarios": True,
            "runtime_performance_envelope_complete": False,
            "resource_envelope_complete": False,
            "rtx_4090_qualified": False,
            "rtx_5090_qualified": False,
            "cross_gpu_equivalence_qualified": False,
            "ground_truth_dense_depth_available": False,
            "ground_truth_surface_accuracy_available": False,
            "metric_scale_available": False,
            "certified_metrology_available": False,
        },
        "regression_summary": {
            "focused_i5_9c_tests": receipt["counts"]["focused_tests"],
            "predecessor_tests": receipt["counts"]["predecessor_total_tests"],
            "combined_exercised_tests": receipt["counts"]["combined_exercised_tests"],
            "test_counts_are_not_runtime_quality_scores": True,
        },
        "authority": {
            "camera_pose_authority_owner": "pycolmap_sfm",
            "authoritative_sfm_modified": False,
            "single_authoritative_sfm_frame_preserved": True,
            "single_image_provider_pose_authority": False,
            "model_intrinsics_are_canonical_authority": False,
            "native_metric_depth_establishes_physical_metric_scale": False,
            "relative_depth_establishes_physical_metric_scale": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
            "provider_selected": False,
            "provider_promoted": False,
            "provider_ranking_created": False,
            "permanent_global_view_cap": None,
        },
        "decision": {
            "i5_9a_complete": True,
            "i5_9b1_complete": True,
            "i5_9b2_complete": True,
            "i5_9c_complete": True,
            "i5_9_complete": True,
            "i5_10_authorized": True,
            "i5_10_started": False,
            "i5_11_started": False,
            "i5_12_started": False,
            "i5_13_started": False,
            "i5_14_started": False,
            "provider_selected": False,
            "provider_promoted": False,
            "provider_ranking_created": False,
            "universal_provider_winner_declared": False,
            "global_scalar_score_created": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
            "mp_sfm_authorized": True,
            "mp_sfm_started": False,
            "threshold_relaxation_authorized": False,
            "fusion_capacity_increase_authorized": False,
            "production_default_modified": False,
        },
        "roadmap_guard": {
            "i5_10_mp_sfm_authorized_not_started": True,
            "i5_11_oom_disk_crash_recovery_deferred": True,
            "i5_12_hybrid_deferred_if_justified": True,
            "i5_13_auto_deferred": True,
            "i5_14_final_i5_closure_deferred": True,
        },
        "next": "I5.10 MP-SfM evaluation.",
    }


__all__ = [
    "I59C_SCHEMA_VERSION", "I59C_PHASE", "I59C_RECEIPT_PHASE",
    "I59C_I59A_FROZEN_SHA256", "I59C_I59B1_FROZEN_SHA256",
    "I59C_B1_STATUS_COUNTS", "I59C_B2_STATUS_COUNTS", "I59C_COMBINED_STATUS_COUNTS",
    "I59C_MARIGOLD_DEGRADED_SCENARIO", "I59C_MARIGOLD_SELECTED_VIEWS",
    "I59C_MARIGOLD_ACCEPTED_VIEWS", "I59C_MARIGOLD_ALIGNMENT_REJECTED_VIEWS",
    "validate_i5_9b1_for_i5_9c", "validate_i5_9b2_for_i5_9c",
    "validate_i5_9c_receipt", "build_i5_9c_remaining_depth_provider_robustness_closure",
]

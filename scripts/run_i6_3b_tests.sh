#!/usr/bin/env bash
set -u
set -o pipefail

MODE="${1:-all}"
RECEIPT="${2:-}"

FOCUSED=(
  tests/unit/test_i6_3b_reconstruction_advisor_engine.py
  tests/integration/test_i6_3b_reconstruction_advisor_engine_contract.py
  tests/stress/test_i6_3b_reconstruction_advisor_engine_stress.py
)

I63A_REGRESSION=(
  tests/unit/test_i6_3a_reconstruction_advisor_contract.py
  tests/integration/test_i6_3a_reconstruction_advisor_contract.py
)

I5_EVIDENCE_REGRESSION=(
  tests/unit/test_i5_13a_auto_policy_admission.py
  tests/unit/test_i5_13b_auto_policy_engine.py
  tests/unit/test_i5_13c_auto_policy_replay.py
  tests/unit/test_i5_13d_auto_production_integration_decision.py
  tests/unit/test_i5_14a_final_qualification_readiness.py
  tests/unit/test_i5_14b_final_i5_closure.py
  tests/integration/test_i5_13a_auto_policy_admission_contract.py
  tests/integration/test_i5_13b_auto_policy_engine_contract.py
  tests/integration/test_i5_13c_auto_policy_replay_contract.py
  tests/integration/test_i5_13d_auto_production_integration_decision_contract.py
  tests/integration/test_i5_14a_final_qualification_readiness_contract.py
  tests/integration/test_i5_14b_final_i5_closure_contract.py
)

I6_PURE_PREDECESSORS=(
  tests/unit/test_i6_1a_offline_reconstruction_workspace.py
  tests/unit/test_i6_1b_controller_workspace_contract_static.py
  tests/unit/test_i6_1c_offline_workspace_presentation.py
  tests/unit/test_i6_1c_reconstruction_page_binding_static.py
  tests/integration/test_i6_1a_offline_workspace_contract.py
  tests/unit/test_i6_2a_one_start_preflight.py
  tests/integration/test_i6_2a_one_start_preflight_contract.py
  tests/unit/test_i6_2b_one_start_controller_contract_static.py
  tests/unit/test_i6_2c_controller_result_handoff_contract_static.py
  tests/unit/test_i6_2c_offline_result_handoff.py
  tests/unit/test_i6_2d_one_start_ux_presentation.py
  tests/unit/test_i6_2d_reconstruction_page_ux_binding_static.py
)

FOCUSED_RC=0; FOCUSED_COUNT=0
I63A_RC=0; I63A_COUNT=0
I5_RC=0; I5_COUNT=0
I6_RC=0; I6_COUNT=0
FULL_UNIT_RC=0; FULL_UNIT_COUNT=0
IMPORT_RC=0
COMPILE_RC=0

run_pytest_count() {
  local rc_var="$1" count_var="$2" label="$3"; shift 3
  local tmp
  tmp="$(mktemp)"
  pytest -q "$@" >"$tmp" 2>&1
  local rc=$?
  cat "$tmp"
  local count=0
  if [[ $rc -eq 0 ]]; then
    count="$(python - "$tmp" <<'PY'
import re
import sys
text = open(sys.argv[1], encoding="utf-8", errors="replace").read()
print(sum(int(value) for value in re.findall(r"(\d+) passed", text)))
PY
)"
  fi
  rm -f "$tmp"
  printf '%s_RC=%s\n' "$label" "$rc"
  printf '%s_COUNT=%s\n' "$label" "$count"
  printf -v "$rc_var" '%s' "$rc"
  printf -v "$count_var" '%s' "$count"
}

probe_import() {
  python - <<'PY'
from core.reconstruction.reconstruction_advisor_contract import (
    AdvisorContextResolution,
    AdvisorEvidenceBinding,
    AdvisorEvidenceObservation,
    AdvisorEvidenceOutcome,
    AdvisorResourceObservation,
    ReconstructionAdvisorRequest,
)
from core.reconstruction.dense_provider_architecture import DenseProviderFamily, DenseProviderImplementation
from core.reconstruction.reconstruction_advisor_engine import I63B_ENGINE_PROFILE, evaluate_reconstruction_advisor

binding = AdvisorEvidenceBinding("probe", "I5.14B", "/probe.json", "a" * 64)
observation = AdvisorEvidenceObservation(
    provider_id="colmap-mvs-external",
    provider_family=DenseProviderFamily.MVS,
    provider_implementation=DenseProviderImplementation.COLMAP_MVS,
    outcome=AdvisorEvidenceOutcome.PASS,
    evidence_context_id="probe-context",
    matched_scenario_ids=("probe-scenario",),
    reason_codes=("probe-pass",),
    limitations=("probe-bounded",),
    bindings=(binding,),
)
request = ReconstructionAdvisorRequest(
    reconstruction_mode="offline_reconstruction",
    source_workflow="image_sequence",
    source_ready=True,
    source_item_count=3,
    source_fingerprint="probe",
    workspace_state="partial_result",
    authoritative_sfm_usable=True,
    runtime_integrity_clean=True,
    evidence_context_resolution=AdvisorContextResolution.EXPLICIT_FROZEN_TOKEN,
    evidence_context_id="probe-context",
    available_provider_ids=("colmap-mvs-external",),
    resources=AdvisorResourceObservation("GPU", True, 1024, 512, 1024),
    provider_evidence=(observation,),
)
decision = evaluate_reconstruction_advisor(request)
assert I63B_ENGINE_PROFILE == "evidence_bound_advisory_v1"
assert decision.recommended_provider_id == "colmap-mvs-external"
assert decision.provider_selection_effect is False
assert decision.provider_execution is False
assert decision.start_action is False
assert decision.production_auto_authorized is False
print("I6.3B Reconstruction Advisor engine/import probe: PASS")
PY
  IMPORT_RC=$?
  echo "IMPORT_RC=$IMPORT_RC"
}

probe_compile() {
  python -m py_compile \
    core/reconstruction/reconstruction_advisor_contract.py \
    core/reconstruction/reconstruction_advisor_engine.py \
    tests/unit/test_i6_3b_reconstruction_advisor_engine.py \
    tests/integration/test_i6_3b_reconstruction_advisor_engine_contract.py \
    tests/stress/test_i6_3b_reconstruction_advisor_engine_stress.py
  COMPILE_RC=$?
  echo "COMPILE_RC=$COMPILE_RC"
}

run_focused() {
  run_pytest_count FOCUSED_RC FOCUSED_COUNT FOCUSED "${FOCUSED[@]}"
  probe_import
  probe_compile
}

run_i63a() {
  run_pytest_count I63A_RC I63A_COUNT I6_3A_REGRESSION "${I63A_REGRESSION[@]}"
}

run_i5() {
  run_pytest_count I5_RC I5_COUNT I5_EVIDENCE_REGRESSION "${I5_EVIDENCE_REGRESSION[@]}"
}

run_i6() {
  run_pytest_count I6_RC I6_COUNT I6_PURE_PREDECESSOR "${I6_PURE_PREDECESSORS[@]}"
}

run_full_unit() {
  run_pytest_count FULL_UNIT_RC FULL_UNIT_COUNT FULL_UNIT tests/unit
}

case "$MODE" in
  focused)
    run_focused
    ;;
  predecessors)
    run_i63a
    run_i5
    run_i6
    ;;
  all)
    run_focused
    run_i63a
    run_i5
    run_i6
    ;;
  full-unit)
    run_full_unit
    ;;
  closure)
    run_focused
    run_i63a
    run_i5
    run_i6
    run_full_unit
    ;;
  *)
    echo "usage: $0 [focused|predecessors|all|full-unit|closure] [receipt.json]" >&2
    exit 2
    ;;
esac

RC=0
case "$MODE" in
  focused)
    for value in "$FOCUSED_RC" "$IMPORT_RC" "$COMPILE_RC"; do [[ "$value" -eq 0 ]] || RC=1; done
    ;;
  predecessors)
    for value in "$I63A_RC" "$I5_RC" "$I6_RC"; do [[ "$value" -eq 0 ]] || RC=1; done
    ;;
  all)
    for value in "$FOCUSED_RC" "$I63A_RC" "$I5_RC" "$I6_RC" "$IMPORT_RC" "$COMPILE_RC"; do [[ "$value" -eq 0 ]] || RC=1; done
    ;;
  full-unit)
    [[ "$FULL_UNIT_RC" -eq 0 ]] || RC=1
    ;;
  closure)
    for value in "$FOCUSED_RC" "$I63A_RC" "$I5_RC" "$I6_RC" "$FULL_UNIT_RC" "$IMPORT_RC" "$COMPILE_RC"; do [[ "$value" -eq 0 ]] || RC=1; done
    ;;
esac

if [[ -n "$RECEIPT" ]]; then
  mkdir -p "$(dirname "$RECEIPT")"
  python - "$RECEIPT" "$MODE" "$RC" \
    "$FOCUSED_RC" "$FOCUSED_COUNT" \
    "$I63A_RC" "$I63A_COUNT" \
    "$I5_RC" "$I5_COUNT" \
    "$I6_RC" "$I6_COUNT" \
    "$FULL_UNIT_RC" "$FULL_UNIT_COUNT" \
    "$IMPORT_RC" "$COMPILE_RC" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1]).expanduser().resolve()
mode = sys.argv[2]
(
    rc,
    focused_rc,
    focused_count,
    i63a_rc,
    i63a_count,
    i5_rc,
    i5_count,
    i6_rc,
    i6_count,
    full_unit_rc,
    full_unit_count,
    import_rc,
    compile_rc,
) = map(int, sys.argv[3:])

payload = {
    "schema_version": 1,
    "phase": "I6.3B_EVIDENCE_BOUND_ADVISORY_REASONING_ENGINE",
    "mode": mode,
    "status": "pass" if rc == 0 else "fail",
    "pass": rc == 0,
    "scope": {
        "advisor_reasoning_engine_added": True,
        "deterministic_same_input_same_decision": True,
        "exact_unique_pass_required_for_recommendation": True,
        "explicit_abstention_preserved": True,
        "negative_evidence_preserved": True,
        "degraded_evidence_preserved": True,
        "unqualified_evidence_preserved": True,
        "evidence_sha256_provenance_preserved": True,
        "availability_can_block_but_not_tiebreak": True,
        "hardware_resource_tiebreak": False,
        "global_provider_ranking_created": False,
        "global_scalar_score_created": False,
        "fallback_retry_until_pass": False,
        "live_context_resolver_qualified": False,
        "controller_modified": False,
        "ui_modified": False,
        "worker_modified": False,
        "provider_adapter_modified": False,
        "provider_selection_effect": False,
        "provider_activation_authorized": False,
        "provider_execution_authorized": False,
        "settings_mutation_authorized": False,
        "start_action_authorized": False,
        "production_auto_activated": False,
        "production_auto_qualified": False,
        "production_auto_authorized": False,
        "pycolmap_camera_authority_preserved": True,
        "certified_metrology_claimed": False,
        "reconstruction_algorithm_modified": False,
        "production_default_modified": False,
        "scientific_threshold_modified": False,
    },
    "counts": {
        "focused_tests": focused_count,
        "i6_3a_regression_tests": i63a_count,
        "i5_evidence_regression_tests": i5_count,
        "i6_pure_predecessor_tests": i6_count,
        "full_unit_tests": full_unit_count,
    },
    "return_codes": {
        "focused": focused_rc,
        "i6_3a_regression": i63a_rc,
        "i5_evidence_regression": i5_rc,
        "i6_pure_predecessor": i6_rc,
        "full_unit": full_unit_rc,
        "import_probe": import_rc,
        "compile_probe": compile_rc,
    },
    "manual_ui_validation_required": False,
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"I6.3B receipt={path}")
PY
fi

echo "I6.3B_TEST_RC=$RC"
exit "$RC"

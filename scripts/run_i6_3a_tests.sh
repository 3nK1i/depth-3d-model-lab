#!/usr/bin/env bash
set -u
set -o pipefail

MODE="${1:-all}"
RECEIPT="${2:-}"

FOCUSED=(
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
from dataclasses import FrozenInstanceError
from core.reconstruction.reconstruction_advisor_contract import (
    ADVISOR_CAMERA_AUTHORITY_OWNER,
    ADVISOR_PRODUCTION_AUTO_SCOPE,
    I63A_ADVISOR_SCOPE,
    AdvisorContextResolution,
    AdvisorDisposition,
    AdvisorEvidenceSupport,
    ReconstructionAdvisorDecision,
)

assert ADVISOR_CAMERA_AUTHORITY_OWNER == "pycolmap_sfm"
assert ADVISOR_PRODUCTION_AUTO_SCOPE == "offline_shadow_only"
assert I63A_ADVISOR_SCOPE == "offline_reconstruction_advisory_only"
assert AdvisorContextResolution.UNRESOLVED.value == "unresolved"
assert AdvisorDisposition.ABSTAIN.value == "abstain"
assert AdvisorEvidenceSupport.EXACT_QUALIFIED.value == "exact_qualified"

decision = ReconstructionAdvisorDecision(
    disposition=AdvisorDisposition.ABSTAIN,
    evidence_support=AdvisorEvidenceSupport.INSUFFICIENT,
    reason_codes=("import_probe",),
    explanations=("No production effect.",),
)
assert decision.production_effect is False
assert decision.provider_selection_effect is False
assert decision.provider_execution is False
assert decision.production_auto_authorized is False
assert decision.global_provider_ranking_created is False
assert decision.global_scalar_score_created is False
print("I6.3A Reconstruction Advisor contract/import probe: PASS")
PY
  IMPORT_RC=$?
  echo "IMPORT_RC=$IMPORT_RC"
}

probe_compile() {
  python -m py_compile \
    core/reconstruction/reconstruction_advisor_contract.py \
    tests/unit/test_i6_3a_reconstruction_advisor_contract.py \
    tests/integration/test_i6_3a_reconstruction_advisor_contract.py
  COMPILE_RC=$?
  echo "COMPILE_RC=$COMPILE_RC"
}

run_focused() {
  run_pytest_count FOCUSED_RC FOCUSED_COUNT FOCUSED "${FOCUSED[@]}"
  probe_import
  probe_compile
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
    run_i5
    run_i6
    ;;
  all)
    run_focused
    run_i5
    run_i6
    ;;
  full-unit)
    run_full_unit
    ;;
  closure)
    run_focused
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
    for value in "$I5_RC" "$I6_RC"; do [[ "$value" -eq 0 ]] || RC=1; done
    ;;
  all)
    for value in "$FOCUSED_RC" "$I5_RC" "$I6_RC" "$IMPORT_RC" "$COMPILE_RC"; do [[ "$value" -eq 0 ]] || RC=1; done
    ;;
  full-unit)
    [[ "$FULL_UNIT_RC" -eq 0 ]] || RC=1
    ;;
  closure)
    for value in "$FOCUSED_RC" "$I5_RC" "$I6_RC" "$FULL_UNIT_RC" "$IMPORT_RC" "$COMPILE_RC"; do [[ "$value" -eq 0 ]] || RC=1; done
    ;;
esac

if [[ -n "$RECEIPT" ]]; then
  mkdir -p "$(dirname "$RECEIPT")"
  python - "$RECEIPT" "$MODE" "$RC" \
    "$FOCUSED_RC" "$FOCUSED_COUNT" \
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
    "phase": "I6.3A_RECONSTRUCTION_ADVISOR_CONTRACT_AND_EVIDENCE_MODEL",
    "mode": mode,
    "status": "pass" if rc == 0 else "fail",
    "pass": rc == 0,
    "scope": {
        "advisor_contract_added": True,
        "evidence_model_added": True,
        "advisor_engine_implemented": False,
        "controller_modified": False,
        "ui_modified": False,
        "worker_modified": False,
        "provider_adapter_modified": False,
        "runtime_action_semantics_modified": False,
        "provider_selection_effect": False,
        "provider_activation_authorized": False,
        "provider_execution_authorized": False,
        "settings_mutation_authorized": False,
        "start_action_authorized": False,
        "live_context_resolver_qualified": False,
        "unresolved_context_supported": True,
        "explicit_frozen_context_token_supported": True,
        "negative_evidence_preserved": True,
        "degraded_evidence_preserved": True,
        "sha256_evidence_provenance_required": True,
        "global_provider_ranking_created": False,
        "global_scalar_score_created": False,
        "cross_family_rank_created": False,
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
        "i5_evidence_regression_tests": i5_count,
        "i6_pure_predecessor_tests": i6_count,
        "full_unit_tests": full_unit_count,
    },
    "return_codes": {
        "focused": focused_rc,
        "i5_evidence_regression": i5_rc,
        "i6_pure_predecessor": i6_rc,
        "full_unit": full_unit_rc,
        "import_probe": import_rc,
        "compile_probe": compile_rc,
    },
    "manual_ui_validation_required": False,
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"I6.3A receipt={path}")
PY
fi

echo "I6.3A_TEST_RC=$RC"
exit "$RC"

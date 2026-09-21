#!/usr/bin/env bash
set -u
set -o pipefail

MODE="${1:-all}"
RECEIPT="${2:-}"

FOCUSED=(
  tests/unit/test_i6_4a_provider_availability_selection_contract.py
  tests/integration/test_i6_4a_provider_availability_selection_contract.py
  tests/stress/test_i6_4a_provider_availability_selection_contract_stress.py
)

PROVIDER_ARCHITECTURE_REGRESSION=(
  tests/unit/test_i4_3a_dense_provider_architecture.py
  tests/integration/test_i4_3a_dense_provider_architecture_contract.py
)

I5_AUTO_BOUNDARY_REGRESSION=(
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

I6_3_PURE_REGRESSION=(
  tests/unit/test_i6_3a_reconstruction_advisor_contract.py
  tests/integration/test_i6_3a_reconstruction_advisor_contract.py
  tests/unit/test_i6_3b_reconstruction_advisor_engine.py
  tests/integration/test_i6_3b_reconstruction_advisor_engine_contract.py
  tests/stress/test_i6_3b_reconstruction_advisor_engine_stress.py
  tests/unit/test_i6_3c_reconstruction_advisor_state.py
  tests/unit/test_i6_3c_controller_advisor_contract_static.py
  tests/unit/test_i6_3d_reconstruction_advisor_presentation.py
  tests/unit/test_i6_3d_reconstruction_page_advisor_binding_static.py
)

FOCUSED_RC=0; FOCUSED_COUNT=0
PROVIDER_ARCH_RC=0; PROVIDER_ARCH_COUNT=0
I5_RC=0; I5_COUNT=0
I6_3_RC=0; I6_3_COUNT=0
FULL_UNIT_RC=0; FULL_UNIT_COUNT=0
IMPORT_RC=0; COMPILE_RC=0

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
import re, sys
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
from core.reconstruction.reconstruction_provider_selection_contract import (
    I64A_CANONICAL_PROVIDER_COUNT,
    ProductionAutoGate,
    ProductionProviderAvailability,
    ProductionProviderQualification,
    ProductionProviderSelectionSnapshot,
    canonical_production_provider_catalog,
)

catalog = canonical_production_provider_catalog()
assert len(catalog) == I64A_CANONICAL_PROVIDER_COUNT == 10
assert all(item.availability is ProductionProviderAvailability.UNKNOWN for item in catalog)
assert all(item.qualification is ProductionProviderQualification.UNRESOLVED for item in catalog)
assert "auto" not in {item.provider_id for item in catalog}

gate = ProductionAutoGate()
assert gate.offline_shadow_qualified is True
assert gate.production_qualified is False
assert gate.production_authorized is False
assert gate.selectable is False

snapshot = ProductionProviderSelectionSnapshot(providers=catalog)
assert snapshot.provider_selection_effect is False
assert snapshot.provider_activation_authorized is False
assert snapshot.provider_execution_authorized is False
assert snapshot.settings_mutation_authorized is False
assert snapshot.start_action_authorized is False
print("I6.4A provider availability/selection contract import probe: PASS")
PY
  IMPORT_RC=$?
  echo "IMPORT_RC=$IMPORT_RC"
}

probe_compile() {
  python -m py_compile \
    core/reconstruction/reconstruction_provider_selection_contract.py \
    tests/unit/test_i6_4a_provider_availability_selection_contract.py \
    tests/integration/test_i6_4a_provider_availability_selection_contract.py \
    tests/stress/test_i6_4a_provider_availability_selection_contract_stress.py
  COMPILE_RC=$?
  echo "COMPILE_RC=$COMPILE_RC"
}

run_focused() {
  run_pytest_count FOCUSED_RC FOCUSED_COUNT FOCUSED "${FOCUSED[@]}"
  probe_import
  probe_compile
}

run_predecessors() {
  run_pytest_count PROVIDER_ARCH_RC PROVIDER_ARCH_COUNT PROVIDER_ARCHITECTURE_REGRESSION "${PROVIDER_ARCHITECTURE_REGRESSION[@]}"
  run_pytest_count I5_RC I5_COUNT I5_AUTO_BOUNDARY_REGRESSION "${I5_AUTO_BOUNDARY_REGRESSION[@]}"
  run_pytest_count I6_3_RC I6_3_COUNT I6_3_PURE_REGRESSION "${I6_3_PURE_REGRESSION[@]}"
}

run_full_unit() {
  run_pytest_count FULL_UNIT_RC FULL_UNIT_COUNT FULL_UNIT tests/unit
}

case "$MODE" in
  focused) run_focused ;;
  predecessors) run_predecessors ;;
  all) run_focused; run_predecessors ;;
  full-unit) run_full_unit ;;
  closure) run_focused; run_predecessors; run_full_unit ;;
  *) echo "usage: $0 [focused|predecessors|all|full-unit|closure] [receipt.json]" >&2; exit 2 ;;
esac

RC=0
case "$MODE" in
  focused) for value in "$FOCUSED_RC" "$IMPORT_RC" "$COMPILE_RC"; do [[ "$value" -eq 0 ]] || RC=1; done ;;
  predecessors) for value in "$PROVIDER_ARCH_RC" "$I5_RC" "$I6_3_RC"; do [[ "$value" -eq 0 ]] || RC=1; done ;;
  all) for value in "$FOCUSED_RC" "$PROVIDER_ARCH_RC" "$I5_RC" "$I6_3_RC" "$IMPORT_RC" "$COMPILE_RC"; do [[ "$value" -eq 0 ]] || RC=1; done ;;
  full-unit) [[ "$FULL_UNIT_RC" -eq 0 ]] || RC=1 ;;
  closure) for value in "$FOCUSED_RC" "$PROVIDER_ARCH_RC" "$I5_RC" "$I6_3_RC" "$FULL_UNIT_RC" "$IMPORT_RC" "$COMPILE_RC"; do [[ "$value" -eq 0 ]] || RC=1; done ;;
esac

if [[ -n "$RECEIPT" ]]; then
  mkdir -p "$(dirname "$RECEIPT")"
  python - "$RECEIPT" "$MODE" "$RC" \
    "$FOCUSED_RC" "$FOCUSED_COUNT" \
    "$PROVIDER_ARCH_RC" "$PROVIDER_ARCH_COUNT" \
    "$I5_RC" "$I5_COUNT" \
    "$I6_3_RC" "$I6_3_COUNT" \
    "$FULL_UNIT_RC" "$FULL_UNIT_COUNT" \
    "$IMPORT_RC" "$COMPILE_RC" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1]).expanduser().resolve()
mode = sys.argv[2]
(
    rc,
    focused_rc, focused_count,
    provider_arch_rc, provider_arch_count,
    i5_rc, i5_count,
    i6_3_rc, i6_3_count,
    full_unit_rc, full_unit_count,
    import_rc, compile_rc,
) = map(int, sys.argv[3:])

payload = {
    "schema_version": 1,
    "phase": "I6.4A_PRODUCTION_PROVIDER_AVAILABILITY_AND_SELECTION_CONTRACT",
    "mode": mode,
    "status": "pass" if rc == 0 else "fail",
    "pass": rc == 0,
    "scope": {
        "canonical_concrete_provider_catalog_added": True,
        "runtime_availability_contract_added": True,
        "context_qualification_contract_added": True,
        "recommendation_selection_separation_added": True,
        "manual_selection_eligibility_contract_added": True,
        "auto_gate_added": True,
        "offline_shadow_auto_qualified_preserved": True,
        "production_auto_qualified": False,
        "production_auto_authorized": False,
        "production_auto_selectable": False,
        "controller_modified": False,
        "ui_modified": False,
        "worker_modified": False,
        "provider_runtime_probed": False,
        "provider_adapter_modified": False,
        "provider_selection_effect": False,
        "provider_activation_authorized": False,
        "provider_execution_authorized": False,
        "settings_mutation_authorized": False,
        "start_action_authorized": False,
        "availability_is_qualification": False,
        "qualification_is_recommendation": False,
        "recommendation_is_selection": False,
        "selection_is_execution": False,
        "global_provider_ranking_created": False,
        "global_scalar_score_created": False,
        "production_default_modified": False,
        "scientific_threshold_modified": False,
        "pycolmap_camera_authority_preserved": True,
        "certified_metrology_claimed": False,
    },
    "counts": {
        "focused_tests": focused_count,
        "provider_architecture_regression_tests": provider_arch_count,
        "i5_auto_boundary_regression_tests": i5_count,
        "i6_3_pure_regression_tests": i6_3_count,
        "full_unit_tests": full_unit_count,
    },
    "return_codes": {
        "focused": focused_rc,
        "provider_architecture_regression": provider_arch_rc,
        "i5_auto_boundary_regression": i5_rc,
        "i6_3_pure_regression": i6_3_rc,
        "full_unit": full_unit_rc,
        "import_probe": import_rc,
        "compile_probe": compile_rc,
    },
    "manual_ui_validation_required": False,
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"I6.4A receipt={path}")
PY
fi

echo "I6.4A_TEST_RC=$RC"
exit "$RC"

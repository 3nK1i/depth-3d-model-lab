#!/usr/bin/env bash
set -u
set -o pipefail

MODE="${1:-focused}"
RECEIPT="${2:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 2

FOCUSED_PURE=(
  tests/unit/test_i6_4b_reconstruction_provider_selection_state.py
  tests/unit/test_i6_4b_controller_provider_selection_contract_static.py
  tests/stress/test_i6_4b_provider_selection_state_stress.py
)
FOCUSED_QT=(
  tests/integration/test_i6_4b_controller_provider_selection.py
)
I6_4A_REGRESSION=(
  tests/unit/test_i6_4a_provider_availability_selection_contract.py
  tests/integration/test_i6_4a_provider_availability_selection_contract.py
  tests/stress/test_i6_4a_provider_availability_selection_contract_stress.py
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
I6_CONTROLLER_PREDECESSOR=(
  tests/integration/test_i6_2b_one_start_controller_orchestration.py
  tests/integration/test_i6_2c_offline_result_handoff_controller.py
  tests/integration/test_i6_3c_controller_advisor_state.py
)

FOCUSED_PURE_RC=0; FOCUSED_PURE_COUNT=0
FOCUSED_QT_RC=0; FOCUSED_QT_COUNT=0
I6_4A_RC=0; I6_4A_COUNT=0
I6_3_RC=0; I6_3_COUNT=0
CONTROLLER_RC=0; CONTROLLER_COUNT=0
FULL_UNIT_RC=0; FULL_UNIT_COUNT=0
IMPORT_RC=0; COMPILE_RC=0

run_pytest() {
  local rc_var="$1" count_var="$2" label="$3"; shift 3
  local tmp rc count
  tmp="$(mktemp)"
  pytest -q "$@" >"$tmp" 2>&1
  rc=$?
  cat "$tmp"
  count=0
  if [[ $rc -eq 0 ]]; then
    count="$(python - "$tmp" <<'PY'
import re, sys
text=open(sys.argv[1], encoding='utf-8', errors='replace').read()
print(sum(int(v) for v in re.findall(r'(\d+) passed', text)))
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
from app.reconstruction_provider_selection_state import (
    ManualProviderSelectionStatus,
    ReconstructionProviderSelectionState,
)
from core.reconstruction.reconstruction_provider_selection_contract import (
    ProductionProviderSelectionMode,
    canonical_production_provider_catalog,
)

catalog = canonical_production_provider_catalog()
assert len(catalog) == 10
assert ManualProviderSelectionStatus.NONE.value == 'none'
assert ProductionProviderSelectionMode.MANUAL.value == 'manual'
assert ReconstructionProviderSelectionState.__name__ == 'ReconstructionProviderSelectionState'
print('I6.4B manual provider selection state/import probe: PASS')
PY
  IMPORT_RC=$?
  echo "IMPORT_RC=$IMPORT_RC"
}

probe_compile() {
  python -m py_compile \
    app/reconstruction_provider_selection_state.py \
    app/reconstruction_controller.py \
    tests/unit/test_i6_4b_reconstruction_provider_selection_state.py \
    tests/unit/test_i6_4b_controller_provider_selection_contract_static.py \
    tests/integration/test_i6_4b_controller_provider_selection.py \
    tests/stress/test_i6_4b_provider_selection_state_stress.py
  COMPILE_RC=$?
  echo "COMPILE_RC=$COMPILE_RC"
}

run_focused() {
  run_pytest FOCUSED_PURE_RC FOCUSED_PURE_COUNT FOCUSED_PURE "${FOCUSED_PURE[@]}"
  probe_import
  probe_compile
}

run_closure() {
  run_focused
  run_pytest FOCUSED_QT_RC FOCUSED_QT_COUNT FOCUSED_QT "${FOCUSED_QT[@]}"
  run_pytest I6_4A_RC I6_4A_COUNT I6_4A_REGRESSION "${I6_4A_REGRESSION[@]}"
  run_pytest I6_3_RC I6_3_COUNT I6_3_PURE_REGRESSION "${I6_3_PURE_REGRESSION[@]}"
  run_pytest CONTROLLER_RC CONTROLLER_COUNT I6_CONTROLLER_PREDECESSOR "${I6_CONTROLLER_PREDECESSOR[@]}"
  run_pytest FULL_UNIT_RC FULL_UNIT_COUNT FULL_UNIT tests/unit
}

case "$MODE" in
  focused) run_focused ;;
  closure) run_closure ;;
  *) echo "usage: $0 [focused|closure] [receipt.json]" >&2; exit 2 ;;
esac

TEST_RC=0
for rc in \
  "$FOCUSED_PURE_RC" "$IMPORT_RC" "$COMPILE_RC" \
  "$FOCUSED_QT_RC" "$I6_4A_RC" "$I6_3_RC" "$CONTROLLER_RC" "$FULL_UNIT_RC"
do
  [[ "$rc" -eq 0 ]] || TEST_RC=1
done

if [[ -n "$RECEIPT" ]]; then
  mkdir -p "$(dirname "$RECEIPT")"
  python - "$RECEIPT" "$MODE" "$TEST_RC" \
    "$FOCUSED_PURE_COUNT" "$FOCUSED_QT_COUNT" "$I6_4A_COUNT" "$I6_3_COUNT" \
    "$CONTROLLER_COUNT" "$FULL_UNIT_COUNT" \
    "$FOCUSED_PURE_RC" "$FOCUSED_QT_RC" "$I6_4A_RC" "$I6_3_RC" \
    "$CONTROLLER_RC" "$FULL_UNIT_RC" "$IMPORT_RC" "$COMPILE_RC" <<'PY'
import json, sys
(
 path, mode, test_rc,
 focused_pure_count, focused_qt_count, i64a_count, i63_count,
 controller_count, full_count,
 focused_pure_rc, focused_qt_rc, i64a_rc, i63_rc,
 controller_rc, full_rc, import_rc, compile_rc,
)=sys.argv[1:]
payload={
 "schema_version":1,
 "phase":"I6.4B_MANUAL_PROVIDER_SELECTION_STATE_AND_CONTROLLER_COMPOSITION",
 "mode":mode,
 "pass":int(test_rc)==0,
 "status":"pass" if int(test_rc)==0 else "fail",
 "counts":{
   "focused_pure_tests":int(focused_pure_count),
   "focused_qt_tests":int(focused_qt_count),
   "i6_4a_regression_tests":int(i64a_count),
   "i6_3_pure_regression_tests":int(i63_count),
   "i6_controller_predecessor_tests":int(controller_count),
   "full_unit_tests":int(full_count),
 },
 "return_codes":{
   "focused_pure":int(focused_pure_rc),
   "focused_qt":int(focused_qt_rc),
   "i6_4a":int(i64a_rc),
   "i6_3":int(i63_rc),
   "i6_controller_predecessor":int(controller_rc),
   "full_unit":int(full_rc),
   "import_probe":int(import_rc),
   "compile_probe":int(compile_rc),
 },
 "scope":{
   "manual_provider_selection_state_added":True,
   "controller_composition_added":True,
   "provider_catalog_composed_from_explicit_i6_3_evidence":True,
   "provider_runtime_probe_added":False,
   "live_context_resolver_qualified":False,
   "recommendation_is_selection":False,
   "selection_is_execution":False,
   "source_change_clears_manual_intent":True,
   "invalid_context_clears_manual_intent":True,
   "valid_passive_refresh_retains_manual_intent":True,
   "provider_activation_authorized":False,
   "provider_execution_authorized":False,
   "settings_mutation_authorized":False,
   "start_action_authorized":False,
   "worker_modified":False,
   "ui_modified":False,
   "production_auto_qualified":False,
   "production_auto_authorized":False,
   "production_auto_selectable":False,
   "global_provider_ranking_created":False,
   "global_scalar_score_created":False,
   "pycolmap_camera_authority_preserved":True,
   "certified_metrology_claimed":False,
 },
 "manual_ui_validation_required":False,
}
open(path,'w',encoding='utf-8').write(json.dumps(payload,indent=2,sort_keys=True)+'\n')
print(f'I6.4B receipt={path}')
PY
fi

echo "I6.4B_TEST_RC=$TEST_RC"
exit "$TEST_RC"

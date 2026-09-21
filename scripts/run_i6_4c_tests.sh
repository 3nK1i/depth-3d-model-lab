#!/usr/bin/env bash
set -u
set -o pipefail

MODE="${1:-focused}"
RECEIPT="${2:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 2

FOCUSED_PURE=(
  tests/unit/test_i6_4c_production_auto_qualification_boundary.py
  tests/unit/test_i6_4c_controller_auto_gate_contract_static.py
  tests/stress/test_i6_4c_production_auto_qualification_boundary_stress.py
)
FOCUSED_QT=(
  tests/integration/test_i6_4c_controller_production_auto_gate.py
)
I6_4B_REGRESSION=(
  tests/unit/test_i6_4b_reconstruction_provider_selection_state.py
  tests/unit/test_i6_4b_controller_provider_selection_contract_static.py
  tests/integration/test_i6_4b_controller_provider_selection.py
  tests/stress/test_i6_4b_provider_selection_state_stress.py
)
I6_4A_REGRESSION=(
  tests/unit/test_i6_4a_provider_availability_selection_contract.py
  tests/integration/test_i6_4a_provider_availability_selection_contract.py
  tests/stress/test_i6_4a_provider_availability_selection_contract_stress.py
)
I5_AUTO_BOUNDARY_REGRESSION=(
  tests/unit/test_i5_13d_auto_production_integration_decision.py
  tests/integration/test_i5_13d_auto_production_integration_decision_contract.py
  tests/stress/test_i5_13d_auto_production_integration_decision_stress.py
  tests/unit/test_i5_14b_final_i5_closure.py
  tests/integration/test_i5_14b_final_i5_closure_contract.py
  tests/stress/test_i5_14b_final_i5_closure_stress.py
)

FOCUSED_PURE_RC=0; FOCUSED_PURE_COUNT=0
FOCUSED_QT_RC=0; FOCUSED_QT_COUNT=0
I6_4B_RC=0; I6_4B_COUNT=0
I6_4A_RC=0; I6_4A_COUNT=0
I5_AUTO_RC=0; I5_AUTO_COUNT=0
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
from core.reconstruction.reconstruction_production_auto_gate import (
    I64C_I5_AUTHORITY_PHASE,
    I64C_POLICY_PROFILE,
    ProductionAutoBoundaryStatus,
    build_production_auto_qualification_boundary,
)
from core.reconstruction.reconstruction_provider_selection_contract import (
    ProductionProviderSelectionSnapshot,
    canonical_production_provider_catalog,
)

snapshot = ProductionProviderSelectionSnapshot(
    providers=canonical_production_provider_catalog(),
)
boundary = build_production_auto_qualification_boundary(snapshot)
assert I64C_I5_AUTHORITY_PHASE == 'I5.14B'
assert I64C_POLICY_PROFILE == 'evidence_bound_conservative_v1'
assert boundary.status is ProductionAutoBoundaryStatus.DEFERRED_NOT_QUALIFIED
assert boundary.offline_shadow_qualified is True
assert boundary.production_auto_qualified is False
assert boundary.production_auto_authorized is False
assert boundary.selectable is False
print('I6.4C production Auto qualification boundary import probe: PASS')
PY
  IMPORT_RC=$?
  echo "IMPORT_RC=$IMPORT_RC"
}

probe_compile() {
  python -m py_compile \
    core/reconstruction/reconstruction_production_auto_gate.py \
    app/reconstruction_controller.py \
    tests/unit/test_i6_4c_production_auto_qualification_boundary.py \
    tests/unit/test_i6_4c_controller_auto_gate_contract_static.py \
    tests/integration/test_i6_4c_controller_production_auto_gate.py \
    tests/stress/test_i6_4c_production_auto_qualification_boundary_stress.py
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
  run_pytest I6_4B_RC I6_4B_COUNT I6_4B_REGRESSION "${I6_4B_REGRESSION[@]}"
  run_pytest I6_4A_RC I6_4A_COUNT I6_4A_REGRESSION "${I6_4A_REGRESSION[@]}"
  run_pytest I5_AUTO_RC I5_AUTO_COUNT I5_AUTO_BOUNDARY_REGRESSION "${I5_AUTO_BOUNDARY_REGRESSION[@]}"
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
  "$FOCUSED_QT_RC" "$I6_4B_RC" "$I6_4A_RC" "$I5_AUTO_RC" "$FULL_UNIT_RC"
do
  [[ "$rc" -eq 0 ]] || TEST_RC=1
done

if [[ -n "$RECEIPT" ]]; then
  mkdir -p "$(dirname "$RECEIPT")"
  python - "$RECEIPT" "$MODE" "$TEST_RC" \
    "$FOCUSED_PURE_COUNT" "$FOCUSED_QT_COUNT" "$I6_4B_COUNT" "$I6_4A_COUNT" \
    "$I5_AUTO_COUNT" "$FULL_UNIT_COUNT" \
    "$FOCUSED_PURE_RC" "$FOCUSED_QT_RC" "$I6_4B_RC" "$I6_4A_RC" \
    "$I5_AUTO_RC" "$FULL_UNIT_RC" "$IMPORT_RC" "$COMPILE_RC" <<'PY'
import json, sys
(
 path, mode, test_rc,
 focused_pure_count, focused_qt_count, i64b_count, i64a_count,
 i5_auto_count, full_count,
 focused_pure_rc, focused_qt_rc, i64b_rc, i64a_rc,
 i5_auto_rc, full_rc, import_rc, compile_rc,
)=sys.argv[1:]
payload={
 "schema_version":1,
 "phase":"I6.4C_PRODUCTION_AUTO_GATE_AND_QUALIFICATION_BOUNDARY",
 "mode":mode,
 "pass":int(test_rc)==0,
 "status":"pass" if int(test_rc)==0 else "fail",
 "counts":{
   "focused_pure_tests":int(focused_pure_count),
   "focused_qt_tests":int(focused_qt_count),
   "i6_4b_regression_tests":int(i64b_count),
   "i6_4a_regression_tests":int(i64a_count),
   "i5_auto_boundary_regression_tests":int(i5_auto_count),
   "full_unit_tests":int(full_count),
 },
 "return_codes":{
   "focused_pure":int(focused_pure_rc),
   "focused_qt":int(focused_qt_rc),
   "i6_4b":int(i64b_rc),
   "i6_4a":int(i64a_rc),
   "i5_auto_boundary":int(i5_auto_rc),
   "full_unit":int(full_rc),
   "import_probe":int(import_rc),
   "compile_probe":int(compile_rc),
 },
 "scope":{
   "i5_offline_shadow_auto_qualified":True,
   "i5_offline_replay_qualified":True,
   "production_auto_qualified":False,
   "production_auto_authorized":False,
   "production_auto_selectable":False,
   "qualified_live_context_resolver":False,
   "production_auto_orchestration_entrypoint_added":False,
   "production_auto_engine_wired_into_app":False,
   "manual_provider_selection_preserved":True,
   "manual_selection_is_auto":False,
   "provider_activation_authorized":False,
   "provider_execution_authorized":False,
   "settings_mutation_authorized":False,
   "start_action_authorized":False,
   "production_default_modified":False,
   "worker_modified":False,
   "ui_modified":False,
   "global_provider_ranking_created":False,
   "global_scalar_score_created":False,
   "pycolmap_camera_authority_preserved":True,
   "certified_metrology_claimed":False,
 },
 "manual_ui_validation_required":False,
}
open(path,'w',encoding='utf-8').write(json.dumps(payload,indent=2,sort_keys=True)+'\n')
print(f'I6.4C receipt={path}')
PY
fi

echo "I6.4C_TEST_RC=$TEST_RC"
exit "$TEST_RC"

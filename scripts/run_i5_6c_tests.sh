#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${1:-all}"
RECEIPT="${2:-}"

FOCUSED=(
  tests/unit/test_i5_6c_mvs_robustness_closure.py
  tests/integration/test_i5_6c_mvs_robustness_contract.py
  tests/stress/test_i5_6c_mvs_robustness_stress.py
)
PREDECESSOR_UNIT=(
  tests/unit/test_i5_6b_mvs_robustness_replay.py
  tests/unit/test_i5_6a_mvs_robustness_admission.py
  tests/unit/test_i4_3b_mvs_provider.py
  tests/unit/test_i4_3f1_adaptive_view_selection.py
  tests/unit/test_i4_3f2_adaptive_production_planning.py
  tests/unit/test_i4_3f2_production_source_contract.py
  tests/unit/test_i4_3f3_adaptive_selection_closure.py
)
PREDECESSOR_INTEGRATION=(
  tests/integration/test_i5_6b_mvs_robustness_contract.py
  tests/integration/test_i5_6a_mvs_robustness_contract.py
  tests/integration/test_i4_3b_mvs_provider_contract.py
  tests/integration/test_i4_3f1_adaptive_view_selection_contract.py
  tests/integration/test_i4_3f2_production_wiring_contract.py
  tests/integration/test_i4_3f3_production_validation_contract.py
)
PREDECESSOR_STRESS=(
  tests/stress/test_i5_6b_mvs_robustness_stress.py
  tests/stress/test_i5_6a_mvs_robustness_stress.py
  tests/stress/test_i4_3b_mvs_provider_stress.py
  tests/stress/test_i4_3f1_adaptive_view_selection_stress.py
  tests/stress/test_i4_3f2_production_wiring_stress.py
  tests/stress/test_i4_3f3_production_validation_stress.py
)

run_pytest_count() {
  local label="$1"; shift
  local tmp; tmp="$(mktemp)"
  pytest -q "$@" >"$tmp" 2>&1
  local rc=$?
  cat "$tmp"
  local count=0
  if [[ $rc -eq 0 ]]; then
    count="$(python - "$tmp" <<'PY'
import re,sys
text=open(sys.argv[1],encoding='utf-8',errors='replace').read()
print(sum(int(x) for x in re.findall(r'(\d+) passed', text)))
PY
)"
  fi
  rm -f "$tmp"
  printf '%s_RC=%s\n' "$label" "$rc"
  printf '%s_COUNT=%s\n' "$label" "$count"
  eval "${label}_RC=$rc"
  eval "${label}_COUNT=$count"
}

probe_import() {
  python - <<'PY'
from core.reconstruction.i5_mvs_robustness import (
    I56C_PHASE,
    build_i5_6c_mvs_robustness_closure,
    validate_i5_6b_for_i5_6c,
    validate_i5_6c_receipt,
)
assert I56C_PHASE == "I5.6C"
assert callable(build_i5_6c_mvs_robustness_closure)
assert callable(validate_i5_6b_for_i5_6c)
assert callable(validate_i5_6c_receipt)
print("I5.6C MVS robustness closure contract/import probe: PASS")
PY
  IMPORT_RC=$?; echo "import_probe_rc=$IMPORT_RC"
}

probe_help() {
  python scripts/run_i5_6c_mvs_robustness_closure.py --help >/dev/null
  HELP_RC=$?; echo "closure_runner_help_rc=$HELP_RC"
}

write_receipt() {
  [[ -n "$RECEIPT" ]] || return 0
  mkdir -p "$(dirname "$RECEIPT")"
  python - "$RECEIPT" "$FOCUSED_COUNT" "$PUNIT_COUNT" "$PINT_COUNT" "$PSTRESS_COUNT" "$FOCUSED_RC" "$PUNIT_RC" "$PINT_RC" "$PSTRESS_RC" "$IMPORT_RC" "$HELP_RC" <<'PY'
import json,sys
path=sys.argv[1]
focused,unit,integ,stress=map(int,sys.argv[2:6])
rcs=list(map(int,sys.argv[6:12]))
predecessor=unit+integ+stress
passed=all(v==0 for v in rcs)
payload={
 "schema_version":1,"phase":"I5.6C_QUALIFICATION_RECEIPT","status":"pass" if passed else "fail","pass":passed,
 "scope":{
   "focused_i5_6c_tests":True,"predecessor_i5_6b_regression":True,"predecessor_i5_6a_regression":True,
   "evidence_only_closure_contract_regression":True,"interrupted_recovery_provenance_regression":True,
   "directional_native_support_preservation_regression":True,"authority_guard_regression":True,
   "roadmap_guard_regression":True,"import_probe":True,"runner_help_probe":True,
   "sfm_rerun":False,"mvs_rerun":False,"patch_match_rerun":False,"stereo_fusion_rerun":False,
   "provider_comparison_started":False,"da3_rerun":False,"thresholds_modified":False,
   "production_default_modified":False,"retry_until_pass":False,"mp_sfm_started":False,
   "auto_policy_implemented":False,"hybrid_authorized":False,
 },
 "counts":{
   "focused_tests":focused,"predecessor_unit_tests":unit,"predecessor_integration_tests":integ,
   "predecessor_stress_tests":stress,"predecessor_total_tests":predecessor,
   "combined_exercised_tests":focused+predecessor,
 },
 "return_codes":{
   "focused":rcs[0],"predecessor_unit":rcs[1],"predecessor_integration":rcs[2],
   "predecessor_stress":rcs[3],"import_probe":rcs[4],"runner_help_probe":rcs[5],
 },
}
with open(path,'w',encoding='utf-8') as f:
    json.dump(payload,f,indent=2,sort_keys=True,allow_nan=False); f.write('\n')
print(f"closure_receipt={path}")
PY
}

case "$MODE" in
 focused) pytest -q "${FOCUSED[@]}" ;;
 predecessor-unit) pytest -q "${PREDECESSOR_UNIT[@]}" ;;
 predecessor-integration) pytest -q "${PREDECESSOR_INTEGRATION[@]}" ;;
 predecessor-stress) pytest -q "${PREDECESSOR_STRESS[@]}" ;;
 all)
  run_pytest_count FOCUSED "${FOCUSED[@]}"; probe_import; probe_help
  run_pytest_count PUNIT "${PREDECESSOR_UNIT[@]}"
  run_pytest_count PINT "${PREDECESSOR_INTEGRATION[@]}"
  run_pytest_count PSTRESS "${PREDECESSOR_STRESS[@]}"
  write_receipt
  FINAL_RC=0
  for v in "$FOCUSED_RC" "$IMPORT_RC" "$HELP_RC" "$PUNIT_RC" "$PINT_RC" "$PSTRESS_RC"; do
    [[ "$v" -eq 0 ]] || FINAL_RC=1
  done
  echo "I5.6C_TEST_RC=$FINAL_RC"; exit "$FINAL_RC" ;;
 *) echo "usage: $0 [focused|predecessor-unit|predecessor-integration|predecessor-stress|all] [receipt.json]" >&2; exit 2 ;;
esac

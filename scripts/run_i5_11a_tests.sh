#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODE="${1:-all}"
RECEIPT="${2:-}"

FOCUSED=(
  tests/unit/test_i5_11a_failure_mode_admission.py
  tests/integration/test_i5_11a_failure_mode_admission_contract.py
  tests/stress/test_i5_11a_failure_mode_admission_stress.py
)
PREDECESSOR_UNIT=(
  tests/unit/test_i5_10d_mp_sfm_closure.py
  tests/unit/test_reconstruction_engine_hardening_g1.py
  tests/unit/test_failure_classification.py
  tests/unit/test_phase8_8g4_reconstruction_failure_recovery.py
  tests/unit/test_phase9_6e1_processing_jobs.py
)
PREDECESSOR_INTEGRATION=(
  tests/integration/test_i5_10d_mp_sfm_closure_contract.py
  tests/integration/test_reconstruction_engine_hardening_g1.py
  tests/integration/test_phase8_9e_endurance_recovery.py
  tests/integration/test_reconstruction_engine_checkpoint_g2.py
)
PREDECESSOR_STRESS=(
  tests/stress/test_i5_10d_mp_sfm_closure_stress.py
  tests/stress/test_reconstruction_engine_hardening_g1_stress.py
  tests/stress/test_phase8_9e_endurance_resource_recovery_stress.py
)

run_pytest_count(){
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
print(sum(int(x) for x in re.findall(r'(\d+) passed',text)))
PY
)"
  fi
  rm -f "$tmp"
  printf '%s_RC=%s\n' "$label" "$rc"
  printf '%s_COUNT=%s\n' "$label" "$count"
  eval "${label}_RC=$rc"
  eval "${label}_COUNT=$count"
}

probe_import(){
  python - <<'PY'
from core.reconstruction.i5_failure_robustness import (
    I511A_PHASE, I511A_FAULT_SCENARIOS, I511A_REPOSITORY_ANCHORS,
    I511AFailureAdmissionPolicy, repository_failure_robustness_inventory,
    build_i5_11a_failure_mode_admission,
)
assert I511A_PHASE == 'I5.11A'
assert I511A_FAULT_SCENARIOS == ('cuda_oom','disk_enospc','crash_orphan_descendant')
assert len(I511A_REPOSITORY_ANCHORS) >= 10
assert I511AFailureAdmissionPolicy().actual_gpu_exhaustion is False
assert repository_failure_robustness_inventory()['resource_pressure_performed'] is False
assert callable(build_i5_11a_failure_mode_admission)
print('I5.11A failure-mode inventory/admission contract/import probe: PASS')
PY
  IMPORT_RC=$?; echo "import_probe_rc=$IMPORT_RC"
}

probe_help(){
  python scripts/run_i5_11a_failure_mode_admission.py --help >/dev/null
  HELP_RC=$?; echo "admission_runner_help_rc=$HELP_RC"
}

write_receipt(){
  [[ -n "$RECEIPT" ]] || return 0
  mkdir -p "$(dirname "$RECEIPT")"
  python - "$RECEIPT" "$FOCUSED_COUNT" "$PUNIT_COUNT" "$PINT_COUNT" "$PSTRESS_COUNT" "$FOCUSED_RC" "$PUNIT_RC" "$PINT_RC" "$PSTRESS_RC" "$IMPORT_RC" "$HELP_RC" <<'PY'
import json,sys
p=sys.argv[1]
focused,unit,integ,stress=map(int,sys.argv[2:6])
rcs=list(map(int,sys.argv[6:12]))
pred=unit+integ+stress
passed=all(v==0 for v in rcs)
payload={
  "schema_version":1,
  "phase":"I5.11A_ADMISSION_RECEIPT",
  "status":"pass" if passed else "fail",
  "pass":passed,
  "scope":{
    "focused_i5_11a_tests":True,
    "engine_hardening_regression":True,
    "failure_classification_regression":True,
    "recovery_ux_regression":True,
    "processing_supervisor_regression":True,
    "import_probe":True,
    "runner_help_probe":True,
    "fault_injection":False,
    "real_gpu_oom":False,
    "real_disk_fill":False,
    "sfm_execution":False,
    "mvs_execution":False,
    "provider_inference":False,
    "production_default_modified":False,
    "auto_policy_implemented":False,
    "hybrid_authorized":False,
  },
  "counts":{
    "focused_tests":focused,
    "predecessor_unit_tests":unit,
    "predecessor_integration_tests":integ,
    "predecessor_stress_tests":stress,
    "predecessor_total_tests":pred,
    "combined_exercised_tests":focused+pred,
  },
  "return_codes":{
    "focused":rcs[0],
    "predecessor_unit":rcs[1],
    "predecessor_integration":rcs[2],
    "predecessor_stress":rcs[3],
    "import_probe":rcs[4],
    "runner_help_probe":rcs[5],
  },
}
with open(p,'w',encoding='utf-8') as f:
    json.dump(payload,f,indent=2,sort_keys=True,allow_nan=False); f.write('\n')
print(f'admission_receipt={p}')
PY
}

FOCUSED_RC=0; FOCUSED_COUNT=0; PUNIT_RC=0; PUNIT_COUNT=0; PINT_RC=0; PINT_COUNT=0; PSTRESS_RC=0; PSTRESS_COUNT=0; IMPORT_RC=0; HELP_RC=0
case "$MODE" in
  focused) run_pytest_count FOCUSED "${FOCUSED[@]}"; probe_import; probe_help ;;
  predecessor)
    run_pytest_count PUNIT "${PREDECESSOR_UNIT[@]}"
    run_pytest_count PINT "${PREDECESSOR_INTEGRATION[@]}"
    run_pytest_count PSTRESS "${PREDECESSOR_STRESS[@]}"
    probe_import; probe_help
    ;;
  all)
    run_pytest_count FOCUSED "${FOCUSED[@]}"
    probe_import; probe_help
    run_pytest_count PUNIT "${PREDECESSOR_UNIT[@]}"
    run_pytest_count PINT "${PREDECESSOR_INTEGRATION[@]}"
    run_pytest_count PSTRESS "${PREDECESSOR_STRESS[@]}"
    ;;
  *) echo "usage: $0 [focused|predecessor|all] [receipt.json]" >&2; exit 2 ;;
esac
write_receipt
RC=0
for v in "$FOCUSED_RC" "$PUNIT_RC" "$PINT_RC" "$PSTRESS_RC" "$IMPORT_RC" "$HELP_RC"; do [[ "$v" -eq 0 ]] || RC=1; done
echo "I5.11A_TEST_RC=$RC"
exit "$RC"

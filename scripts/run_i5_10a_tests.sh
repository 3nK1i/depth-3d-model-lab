#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODE="${1:-all}"
RECEIPT="${2:-}"

FOCUSED=(
  tests/unit/test_i5_10a_mp_sfm_admission.py
  tests/integration/test_i5_10a_mp_sfm_contract.py
  tests/stress/test_i5_10a_mp_sfm_stress.py
)
PREDECESSOR_UNIT=(
  tests/unit/test_i5_9c_remaining_depth_provider_robustness_closure.py
  tests/unit/test_i5_4a_forward_motion_kitti_admission.py
  tests/unit/test_i5_4b_forward_motion_kitti_replay.py
  tests/unit/test_i5_4c_forward_motion_kitti_closure.py
  tests/unit/test_reconstruction_sfm_contracts.py
  tests/unit/test_reconstruction_sfm_environment.py
  tests/unit/test_reconstruction_sfm_quality.py
)
PREDECESSOR_INTEGRATION=(
  tests/integration/test_i5_9c_remaining_depth_provider_robustness_contract.py
  tests/integration/test_i5_4b_forward_motion_kitti_contract.py
  tests/integration/test_reconstruction_pycolmap_worker_lifecycle.py
)
PREDECESSOR_STRESS=(
  tests/stress/test_i5_9c_remaining_depth_provider_robustness_stress.py
  tests/stress/test_i5_4b_forward_motion_kitti_stress.py
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
from core.reconstruction.i5_mp_sfm_evaluation import (
    I510A_PHASE, I510A_I59C_FROZEN_SHA256, I510A_PRIMARY_SCENARIO,
    I510A_SUPPORTED_MATCHING_STRATEGIES, I510A_REPOSITORY_ANCHORS,
    I510AAdmissionPolicy, repository_mp_sfm_inventory,
    build_i5_10a_mp_sfm_admission,
)
assert I510A_PHASE == 'I5.10A'
assert len(I510A_I59C_FROZEN_SHA256) == 64
assert I510A_PRIMARY_SCENARIO == 'kitti_forward_motion'
assert I510A_SUPPORTED_MATCHING_STRATEGIES == ('sequential','exhaustive')
assert len(I510A_REPOSITORY_ANCHORS) >= 10
assert I510AAdmissionPolicy().candidate_matching_strategy == 'exhaustive'
assert repository_mp_sfm_inventory()['new_matching_algorithm_introduced'] is False
assert callable(build_i5_10a_mp_sfm_admission)
print('I5.10A MP-SfM inventory/admission contract/import probe: PASS')
PY
  IMPORT_RC=$?; echo "import_probe_rc=$IMPORT_RC"
}

probe_help(){
  python scripts/run_i5_10a_mp_sfm_admission.py --help >/dev/null
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
  "phase":"I5.10A_ADMISSION_RECEIPT",
  "status":"pass" if passed else "fail",
  "pass":passed,
  "scope":{
    "focused_i5_10a_tests":True,
    "sfm_contract_regression":True,
    "i5_4_forward_motion_regression":True,
    "i5_9c_authorization_regression":True,
    "authority_guard_regression":True,
    "roadmap_guard_regression":True,
    "import_probe":True,
    "runner_help_probe":True,
    "mp_sfm_execution":False,
    "sfm_rerun":False,
    "mvs_rerun":False,
    "provider_inference":False,
    "thresholds_modified":False,
    "production_default_modified":False,
    "result_driven_retuning":False,
    "retry_until_pass":False,
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
    probe_import
    probe_help
    run_pytest_count PUNIT "${PREDECESSOR_UNIT[@]}"
    run_pytest_count PINT "${PREDECESSOR_INTEGRATION[@]}"
    run_pytest_count PSTRESS "${PREDECESSOR_STRESS[@]}"
    ;;
  *) echo "usage: $0 [focused|predecessor|all] [receipt.json]" >&2; exit 2 ;;
esac
write_receipt
RC=0
for v in "$FOCUSED_RC" "$PUNIT_RC" "$PINT_RC" "$PSTRESS_RC" "$IMPORT_RC" "$HELP_RC"; do [[ "$v" -eq 0 ]] || RC=1; done
echo "I5.10A_TEST_RC=$RC"
exit "$RC"

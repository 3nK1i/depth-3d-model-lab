#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODE="${1:-all}"
RECEIPT="${2:-}"

FOCUSED=(
  tests/unit/test_i5_10b_mp_sfm_preflight.py
  tests/integration/test_i5_10b_mp_sfm_preflight_contract.py
  tests/stress/test_i5_10b_mp_sfm_preflight_stress.py
)
PREDECESSOR_UNIT=(
  tests/unit/test_i5_10a_mp_sfm_admission.py
  tests/unit/test_i5_4a_forward_motion_kitti_admission.py
  tests/unit/test_i5_4b_forward_motion_kitti_replay.py
  tests/unit/test_i5_4c_forward_motion_kitti_closure.py
  tests/unit/test_reconstruction_sfm_contracts.py
  tests/unit/test_reconstruction_sfm_pycolmap.py
)
PREDECESSOR_INTEGRATION=(
  tests/integration/test_i5_10a_mp_sfm_contract.py
  tests/integration/test_i5_4a_forward_motion_kitti_contract.py
  tests/integration/test_i5_4b_forward_motion_kitti_contract.py
  tests/integration/test_i5_4c_forward_motion_kitti_contract.py
)
PREDECESSOR_STRESS=(
  tests/stress/test_i5_10a_mp_sfm_stress.py
  tests/stress/test_i5_4a_forward_motion_kitti_stress.py
  tests/stress/test_i5_4b_forward_motion_kitti_stress.py
  tests/stress/test_i5_4c_forward_motion_kitti_stress.py
)

run_pytest_count() {
  local prefix="$1"; shift
  local out rc count
  out="$(mktemp)"
  set +e
  pytest -q "$@" | tee "$out"
  rc=${PIPESTATUS[0]}
  set -e
  count="$(python - "$out" <<'PY'
import re,sys
text=open(sys.argv[1],encoding='utf-8',errors='replace').read()
matches=re.findall(r'(\d+) passed',text)
print(matches[-1] if matches else 0)
PY
)"
  rm -f "$out"
  printf -v "${prefix}_RC" '%s' "$rc"
  printf -v "${prefix}_COUNT" '%s' "$count"
  echo "${prefix}_RC=$rc"
  echo "${prefix}_COUNT=$count"
}

probe_import() {
  python - <<'PY'
from core.reconstruction.i5_mp_sfm_preflight import (
    build_i5_10b_mp_sfm_preflight,
    revalidate_i5_10b_source,
    validate_i5_10b_runtime_probe,
)
assert callable(build_i5_10b_mp_sfm_preflight)
assert callable(revalidate_i5_10b_source)
assert callable(validate_i5_10b_runtime_probe)
print('I5.10B MP-SfM preflight contract/import probe: PASS')
PY
  IMPORT_RC=$?; echo "import_probe_rc=$IMPORT_RC"
}

probe_help() {
  python scripts/run_i5_10b_mp_sfm_preflight.py --help >/dev/null
  HELP_RC=$?; echo "preflight_runner_help_rc=$HELP_RC"
}

write_receipt() {
  [[ -n "$RECEIPT" ]] || return 0
  mkdir -p "$(dirname "$RECEIPT")"
  python - "$RECEIPT" "$FOCUSED_COUNT" "$PUNIT_COUNT" "$PINT_COUNT" "$PSTRESS_COUNT" "$FOCUSED_RC" "$PUNIT_RC" "$PINT_RC" "$PSTRESS_RC" "$IMPORT_RC" "$HELP_RC" <<'PY'
import json,sys
path=sys.argv[1]
focused,unit,integ,stress=map(int,sys.argv[2:6])
rcs=list(map(int,sys.argv[6:12])); pred=unit+integ+stress
payload={
 "schema_version":1,"phase":"I5.10B_PREFLIGHT_RECEIPT","status":"pass" if all(x==0 for x in rcs) else "fail","pass":all(x==0 for x in rcs),
 "scope":{
  "focused_i5_10b_tests":True,"i5_10a_regression":True,"i5_4_chain_regression":True,"source_identity_regression":True,
  "pycolmap_environment_regression":True,"candidate_command_regression":True,"authority_guard_regression":True,"roadmap_guard_regression":True,
  "import_probe":True,"runner_help_probe":True,"real_mp_sfm_execution":False,"sfm_rerun":False,"mvs_rerun":False,
  "thresholds_modified":False,"production_default_modified":False,"retry_until_pass":False,"auto_policy_implemented":False,"hybrid_authorized":False,
 },
 "counts":{"focused_tests":focused,"predecessor_unit_tests":unit,"predecessor_integration_tests":integ,"predecessor_stress_tests":stress,"predecessor_total_tests":pred,"combined_exercised_tests":focused+pred},
 "return_codes":{"focused":rcs[0],"predecessor_unit":rcs[1],"predecessor_integration":rcs[2],"predecessor_stress":rcs[3],"import_probe":rcs[4],"runner_help_probe":rcs[5]},
}
with open(path,'w',encoding='utf-8') as f: json.dump(payload,f,indent=2,sort_keys=True,allow_nan=False); f.write('\n')
print(f'preflight_receipt={path}')
PY
}

case "$MODE" in
 focused) pytest -q "${FOCUSED[@]}" ;;
 predecessor-unit) pytest -q "${PREDECESSOR_UNIT[@]}" ;;
 predecessor-integration) pytest -q "${PREDECESSOR_INTEGRATION[@]}" ;;
 predecessor-stress) pytest -q "${PREDECESSOR_STRESS[@]}" ;;
 all)
  set -e
  run_pytest_count FOCUSED "${FOCUSED[@]}"; probe_import; probe_help
  run_pytest_count PUNIT "${PREDECESSOR_UNIT[@]}"
  run_pytest_count PINT "${PREDECESSOR_INTEGRATION[@]}"
  run_pytest_count PSTRESS "${PREDECESSOR_STRESS[@]}"
  write_receipt
  FINAL_RC=0
  for v in "$FOCUSED_RC" "$IMPORT_RC" "$HELP_RC" "$PUNIT_RC" "$PINT_RC" "$PSTRESS_RC"; do [[ "$v" -eq 0 ]] || FINAL_RC=1; done
  echo "I5.10B_TEST_RC=$FINAL_RC"
  exit "$FINAL_RC" ;;
 *) echo "usage: $0 [focused|predecessor-unit|predecessor-integration|predecessor-stress|all] [receipt.json]" >&2; exit 2 ;;
esac

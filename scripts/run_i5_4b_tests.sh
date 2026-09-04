#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODE="${1:-all}"
RECEIPT="${2:-}"

FOCUSED=(
  tests/unit/test_i5_4b_forward_motion_kitti_replay.py
  tests/integration/test_i5_4b_forward_motion_kitti_contract.py
  tests/stress/test_i5_4b_forward_motion_kitti_stress.py
)
PREDECESSOR_UNIT=(
  tests/unit/test_i5_4a_forward_motion_kitti_admission.py
  tests/unit/test_i5_3c_blur_exposure_clutter_symmetry_closure.py
  tests/unit/test_reconstruction_ingestion_quality_sampling.py
  tests/unit/test_i2_4_support_diversity_parallax_diagnosis.py
)
PREDECESSOR_INTEGRATION=(
  tests/integration/test_i5_4a_forward_motion_kitti_contract.py
  tests/integration/test_i5_3c_blur_exposure_clutter_symmetry_contract.py
  tests/integration/test_i5_2c_low_overlap_weak_parallax_contract.py
)
PREDECESSOR_STRESS=(
  tests/stress/test_i5_4a_forward_motion_kitti_stress.py
  tests/stress/test_i5_3c_blur_exposure_clutter_symmetry_stress.py
)

run_pytest_count() {
  local label="$1"; shift
  local tmp; tmp="$(mktemp)"
  pytest -q "$@" | tee "$tmp"
  local rc=${PIPESTATUS[0]}; local count=0
  if [[ $rc -eq 0 ]]; then
    count="$(python - "$tmp" <<'PY'
import re,sys
text=open(sys.argv[1],encoding='utf-8',errors='replace').read()
print(sum(int(x) for x in re.findall(r'(\d+) passed', text)))
PY
)"
  fi
  rm -f "$tmp"
  printf '%s_RC=%s\n' "$label" "$rc"; printf '%s_COUNT=%s\n' "$label" "$count"
  eval "${label}_RC=$rc"; eval "${label}_COUNT=$count"
}

probe_import() {
  python - <<'PY'
from core.reconstruction.i5_forward_motion_kitti import (
    I54B_PHASE, analyze_i5_4b_mapping_no_sparse_model_rejection, analyze_i5_4b_sfm_report, build_i5_4b_forward_motion_kitti_replay,
)
assert I54B_PHASE == "I5.4B"
assert callable(analyze_i5_4b_mapping_no_sparse_model_rejection)
assert callable(analyze_i5_4b_sfm_report)
assert callable(build_i5_4b_forward_motion_kitti_replay)
print("I5.4B forward-motion/KITTI replay contract/import probe: PASS")
PY
  IMPORT_RC=$?; echo "import_probe_rc=$IMPORT_RC"
}
probe_help() {
  python scripts/run_i5_4b_forward_motion_kitti_replay.py --help >/dev/null
  HELP_RC=$?; echo "replay_runner_help_rc=$HELP_RC"
}
write_receipt() {
  [[ -n "$RECEIPT" ]] || return 0
  mkdir -p "$(dirname "$RECEIPT")"
  python - "$RECEIPT" "$FOCUSED_COUNT" "$PUNIT_COUNT" "$PINT_COUNT" "$PSTRESS_COUNT" "$FOCUSED_RC" "$PUNIT_RC" "$PINT_RC" "$PSTRESS_RC" "$IMPORT_RC" "$HELP_RC" <<'PY'
import json,sys
path=sys.argv[1]; focused,unit,integ,stress=map(int,sys.argv[2:6]); rcs=list(map(int,sys.argv[6:12])); predecessor=unit+integ+stress
payload={
 "schema_version":1,"phase":"I5.4B_REPLAY_RECEIPT","status":"pass" if all(v==0 for v in rcs) else "fail","pass":all(v==0 for v in rcs),
 "scope":{"focused_i5_4b_tests":True,"predecessor_i5_4a_regression":True,"production_native_sfm_contract_regression":True,"forward_motion_geometry_regression":True,"i2_4_scale_invariant_geometry_regression":True,"import_probe":True,"runner_help_probe":True,"real_sfm_executed_by_test_receipt":False,"dense_provider_rerun":False,"thresholds_modified":False,"production_default_modified":False,"mp_sfm_started":False,"auto_policy_implemented":False,"hybrid_authorized":False},
 "counts":{"focused_tests":focused,"predecessor_unit_tests":unit,"predecessor_integration_tests":integ,"predecessor_stress_tests":stress,"predecessor_total_tests":predecessor,"combined_exercised_tests":focused+predecessor},
 "return_codes":{"focused":rcs[0],"predecessor_unit":rcs[1],"predecessor_integration":rcs[2],"predecessor_stress":rcs[3],"import_probe":rcs[4],"runner_help_probe":rcs[5]},
}
with open(path,'w',encoding='utf-8') as f: json.dump(payload,f,indent=2,sort_keys=True,allow_nan=False); f.write('\n')
print(f"replay_receipt={path}")
PY
}
case "$MODE" in
 focused) pytest -q "${FOCUSED[@]}" ;;
 predecessor-unit) pytest -q "${PREDECESSOR_UNIT[@]}" ;;
 predecessor-integration) pytest -q "${PREDECESSOR_INTEGRATION[@]}" ;;
 predecessor-stress) pytest -q "${PREDECESSOR_STRESS[@]}" ;;
 all)
  run_pytest_count FOCUSED "${FOCUSED[@]}"; probe_import; probe_help
  run_pytest_count PUNIT "${PREDECESSOR_UNIT[@]}"; run_pytest_count PINT "${PREDECESSOR_INTEGRATION[@]}"; run_pytest_count PSTRESS "${PREDECESSOR_STRESS[@]}"
  write_receipt
  FINAL_RC=0; for v in "$FOCUSED_RC" "$IMPORT_RC" "$HELP_RC" "$PUNIT_RC" "$PINT_RC" "$PSTRESS_RC"; do [[ "$v" -eq 0 ]] || FINAL_RC=1; done
  echo "I5.4B_TEST_RC=$FINAL_RC"; exit "$FINAL_RC" ;;
 *) echo "usage: $0 [focused|predecessor-unit|predecessor-integration|predecessor-stress|all] [receipt.json]" >&2; exit 2 ;;
esac

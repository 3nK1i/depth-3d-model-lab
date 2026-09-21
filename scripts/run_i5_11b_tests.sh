#!/usr/bin/env bash
set -u
set -o pipefail

MODE="${1:-all}"
RECEIPT="${2:-}"

FOCUSED=(
  tests/unit/test_i5_11b_failure_preflight.py
  tests/integration/test_i5_11b_failure_preflight_contract.py
  tests/stress/test_i5_11b_failure_preflight_stress.py
)
PREDECESSOR_UNIT=(
  tests/unit/test_i5_11a_failure_mode_admission.py
  tests/unit/test_failure_classification.py
  tests/unit/test_reconstruction_engine_hardening_g1.py
)
PREDECESSOR_INTEGRATION=(
  tests/integration/test_i5_11a_failure_mode_admission_contract.py
  tests/integration/test_reconstruction_engine_hardening_g1.py
)
PREDECESSOR_STRESS=(
  tests/stress/test_i5_11a_failure_mode_admission_stress.py
)

FOCUSED_RC=99; PUNIT_RC=99; PINT_RC=99; PSTRESS_RC=99; IMPORT_RC=99; HELP_RC=99
FOCUSED_COUNT=0; PUNIT_COUNT=0; PINT_COUNT=0; PSTRESS_COUNT=0

run_pytest_count() {
  local prefix="$1"; shift
  local tmp rc count
  tmp="$(mktemp)"
  set +e
  pytest -q "$@" >"$tmp" 2>&1
  rc=$?
  set -e
  cat "$tmp"
  count="$(python - "$tmp" <<'PYCOUNT'
import re,sys
t=open(sys.argv[1],encoding="utf-8",errors="replace").read()
m=re.findall(r"(\d+) passed",t)
print(m[-1] if m else 0)
PYCOUNT
)"
  rm -f "$tmp"
  printf -v "${prefix}_RC" '%s' "$rc"
  printf -v "${prefix}_COUNT" '%s' "$count"
  echo "${prefix}_RC=$rc"
  echo "${prefix}_COUNT=$count"
}

probe_imports() {
  set +e
  python - <<'PY'
import core.reconstruction.i5_failure_robustness_preflight as m
assert m.I511B_PHASE == "I5.11B"
assert m.I511B_FAULT_SCENARIOS == ("cuda_oom", "disk_enospc", "crash_orphan_descendant")
print("I5.11B failure preflight contract/import probe: PASS")
PY
  IMPORT_RC=$?
  python scripts/run_i5_11b_failure_preflight.py --help >/dev/null
  HELP_RC=$?
  set -e
  echo "import_probe_rc=$IMPORT_RC"
  echo "preflight_runner_help_rc=$HELP_RC"
}

write_receipt() {
  [[ -n "$RECEIPT" ]] || return 0
  mkdir -p "$(dirname "$RECEIPT")"
  python - "$RECEIPT" "$FOCUSED_COUNT" "$PUNIT_COUNT" "$PINT_COUNT" "$PSTRESS_COUNT" \
    "$FOCUSED_RC" "$PUNIT_RC" "$PINT_RC" "$PSTRESS_RC" "$IMPORT_RC" "$HELP_RC" <<'PY'
import json,sys
from pathlib import Path
path=Path(sys.argv[1])
fc,pu,pi,ps=map(int,sys.argv[2:6])
fr,ur,ir,sr,im,hr=map(int,sys.argv[6:12])
payload={
 "schema_version":1,"phase":"I5.11B_PREFLIGHT_RECEIPT","status":"pass" if all(v==0 for v in (fr,ur,ir,sr,im,hr)) else "fail",
 "pass":all(v==0 for v in (fr,ur,ir,sr,im,hr)),
 "scope":{
  "focused_i5_11b_tests":True,"engine_hardening_regression":True,"failure_classification_regression":True,
  "import_probe":True,"runner_help_probe":True,"synthetic_exception_classification_probe":True,
  "fault_injection":False,"supervised_fault_worker_execution":False,"real_gpu_oom":False,"real_disk_fill":False,
  "sfm_execution":False,"mvs_execution":False,"production_default_modified":False,
  "production_failure_classifier_modified":False,"production_recovery_engine_modified":False,
  "auto_policy_implemented":False,"hybrid_authorized":False,
 },
 "counts":{"focused_tests":fc,"predecessor_unit_tests":pu,"predecessor_integration_tests":pi,"predecessor_stress_tests":ps,
           "predecessor_total_tests":pu+pi+ps,"combined_exercised_tests":fc+pu+pi+ps},
 "return_codes":{"focused":fr,"predecessor_unit":ur,"predecessor_integration":ir,"predecessor_stress":sr,
                 "import_probe":im,"runner_help_probe":hr},
}
path.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
print(f"preflight_receipt={path}")
PY
}

case "$MODE" in
 focused) run_pytest_count FOCUSED "${FOCUSED[@]}" ;;
 predecessor-unit) run_pytest_count PUNIT "${PREDECESSOR_UNIT[@]}" ;;
 predecessor-integration) run_pytest_count PINT "${PREDECESSOR_INTEGRATION[@]}" ;;
 predecessor-stress) run_pytest_count PSTRESS "${PREDECESSOR_STRESS[@]}" ;;
 all)
  set -e
  run_pytest_count FOCUSED "${FOCUSED[@]}"
  probe_imports
  run_pytest_count PUNIT "${PREDECESSOR_UNIT[@]}"
  run_pytest_count PINT "${PREDECESSOR_INTEGRATION[@]}"
  run_pytest_count PSTRESS "${PREDECESSOR_STRESS[@]}"
  write_receipt
  FINAL_RC=0
  for v in "$FOCUSED_RC" "$IMPORT_RC" "$HELP_RC" "$PUNIT_RC" "$PINT_RC" "$PSTRESS_RC"; do [[ "$v" -eq 0 ]] || FINAL_RC=1; done
  echo "I5.11B_TEST_RC=$FINAL_RC"
  exit "$FINAL_RC" ;;
 *) echo "usage: $0 [focused|predecessor-unit|predecessor-integration|predecessor-stress|all] [receipt.json]" >&2; exit 2 ;;
esac

#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${1:-all}"
RECEIPT="${2:-}"

FOCUSED=(
  tests/unit/test_i5_1a_difficult_handheld_baseline.py
  tests/integration/test_i5_1a_difficult_handheld_contract.py
  tests/stress/test_i5_1a_difficult_handheld_stress.py
)
AFFECTED_UNIT=(
  tests/unit/test_i4_1_4_decoupled_sampling_density_diagnosis.py
  tests/unit/test_i4_1_6b_decoupled_dense_ab.py
  tests/unit/test_i4_4d_final_i4_closure.py
)
AFFECTED_INTEGRATION=(
  tests/integration/test_i4_1_4_sampling_density_contract.py
  tests/integration/test_i4_1_6b_dense_ab_contract.py
  tests/integration/test_i4_4d_final_i4_closure_contract.py
)
AFFECTED_STRESS=(
  tests/stress/test_i4_1_4_sampling_density_stress.py
  tests/stress/test_i4_1_6b_dense_ab_stress.py
  tests/stress/test_i4_4d_final_i4_closure_stress.py
)

run_pytest_count() {
  local label="$1"; shift
  local tmp
  tmp="$(mktemp)"
  pytest -q "$@" | tee "$tmp"
  local rc=${PIPESTATUS[0]}
  local count=0
  if [[ $rc -eq 0 ]]; then
    count="$(python - "$tmp" <<'PY'
import re,sys
text=open(sys.argv[1], encoding='utf-8', errors='replace').read()
counts=[int(x) for x in re.findall(r'(\d+) passed', text)]
print(sum(counts))
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
from core.reconstruction.i5_difficult_handheld_robustness import I51A_PHASE
assert I51A_PHASE == "I5.1A"
print("I5.1A difficult-handheld baseline contract/import probe: PASS")
PY
  IMPORT_RC=$?
  echo "import_probe_rc=$IMPORT_RC"
}

probe_help() {
  python scripts/run_i5_1a_difficult_handheld_baseline.py --help >/dev/null
  HELP_RC=$?
  echo "baseline_runner_help_rc=$HELP_RC"
}

write_receipt() {
  [[ -n "$RECEIPT" ]] || return 0
  mkdir -p "$(dirname "$RECEIPT")"
  python - "$RECEIPT" \
    "$FOCUSED_COUNT" "$AUNIT_COUNT" "$AINT_COUNT" "$ASTRESS_COUNT" \
    "$FOCUSED_RC" "$AUNIT_RC" "$AINT_RC" "$ASTRESS_RC" "$IMPORT_RC" "$HELP_RC" <<'PY'
import json,sys
path=sys.argv[1]
focused,unit,integ,stress=map(int,sys.argv[2:6])
rcs=list(map(int,sys.argv[6:12]))
affected=unit+integ+stress
payload={
  "schema_version":1,
  "phase":"I5.1A_BASELINE_RECEIPT",
  "status":"pass" if all(v==0 for v in rcs) else "fail",
  "pass":all(v==0 for v in rcs),
  "scope":{
    "focused_i5_1a_tests":True,
    "affected_i4_horse_regression":True,
    "i4_4d_handoff_regression":True,
    "import_probe":True,
    "runner_help_probe":True,
    "provider_execution_started":False,
    "sfm_rerun":False,
    "dense_pair_rerun":False,
    "current_difficult_handheld_replay_started":False,
    "auto_policy_implemented":False,
    "hybrid_authorized":False,
  },
  "counts":{
    "focused_tests":focused,
    "affected_unit_tests":unit,
    "affected_integration_tests":integ,
    "affected_stress_tests":stress,
    "affected_total_tests":affected,
    "combined_exercised_tests":focused+affected,
  },
  "return_codes":{
    "focused":rcs[0],
    "affected_unit":rcs[1],
    "affected_integration":rcs[2],
    "affected_stress":rcs[3],
    "import_probe":rcs[4],
    "runner_help_probe":rcs[5],
  },
}
with open(path,'w',encoding='utf-8') as f:
    json.dump(payload,f,indent=2,sort_keys=True,allow_nan=False); f.write('\n')
print(f"baseline_receipt={path}")
PY
}

case "$MODE" in
  focused)
    pytest -q "${FOCUSED[@]}"
    ;;
  affected-unit)
    pytest -q "${AFFECTED_UNIT[@]}"
    ;;
  affected-integration)
    pytest -q "${AFFECTED_INTEGRATION[@]}"
    ;;
  affected-stress)
    pytest -q "${AFFECTED_STRESS[@]}"
    ;;
  all)
    run_pytest_count FOCUSED "${FOCUSED[@]}"
    probe_import
    probe_help
    run_pytest_count AUNIT "${AFFECTED_UNIT[@]}"
    run_pytest_count AINT "${AFFECTED_INTEGRATION[@]}"
    run_pytest_count ASTRESS "${AFFECTED_STRESS[@]}"
    write_receipt
    FINAL_RC=0
    for v in "$FOCUSED_RC" "$IMPORT_RC" "$HELP_RC" "$AUNIT_RC" "$AINT_RC" "$ASTRESS_RC"; do
      [[ "$v" -eq 0 ]] || FINAL_RC=1
    done
    echo "I5.1A_TEST_RC=$FINAL_RC"
    exit "$FINAL_RC"
    ;;
  *)
    echo "usage: $0 [focused|affected-unit|affected-integration|affected-stress|all] [receipt.json]" >&2
    exit 2
    ;;
esac

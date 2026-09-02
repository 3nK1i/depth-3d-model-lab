#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${1:-all}"
RECEIPT="${2:-}"

FOCUSED=(
  tests/unit/test_i5_1c_difficult_handheld_closure.py
  tests/integration/test_i5_1c_difficult_handheld_contract.py
  tests/stress/test_i5_1c_difficult_handheld_stress.py
)
PREDECESSOR_UNIT=(
  tests/unit/test_i5_1a_difficult_handheld_baseline.py
  tests/unit/test_i5_1b_difficult_handheld_replay.py
  tests/unit/test_i4_1_6b_decoupled_dense_ab.py
)
PREDECESSOR_INTEGRATION=(
  tests/integration/test_i5_1a_difficult_handheld_contract.py
  tests/integration/test_i5_1b_difficult_handheld_contract.py
  tests/integration/test_i4_1_6b_dense_ab_contract.py
)
PREDECESSOR_STRESS=(
  tests/stress/test_i5_1a_difficult_handheld_stress.py
  tests/stress/test_i5_1b_difficult_handheld_stress.py
  tests/stress/test_i4_1_6b_dense_ab_stress.py
)

run_pytest_count() {
  local label="$1"; shift
  local out rc count
  out="$(mktemp)"
  python -m pytest -q "$@" | tee "$out"
  rc=${PIPESTATUS[0]}
  count=0
  if [[ $rc -eq 0 ]]; then
    count="$(grep -Eo '[0-9]+ passed' "$out" | tail -1 | awk '{print $1}')"
  fi
  rm -f "$out"
  echo "${label}_RC=$rc"
  echo "${label}_COUNT=${count:-0}"
  eval "${label}_RC=$rc"
  eval "${label}_COUNT=${count:-0}"
  return "$rc"
}

probe_import() {
  python - <<'PY'
from core.reconstruction.i5_difficult_handheld_robustness import I51C_PHASE, build_i5_1c_difficult_handheld_closure
assert I51C_PHASE == "I5.1C"
assert callable(build_i5_1c_difficult_handheld_closure)
print("I5.1C difficult-handheld closure contract/import probe: PASS")
PY
  IMPORT_RC=$?
  echo "import_probe_rc=$IMPORT_RC"
}

probe_help() {
  python scripts/run_i5_1c_difficult_handheld_closure.py --help >/dev/null
  HELP_RC=$?
  echo "closure_runner_help_rc=$HELP_RC"
}

write_receipt() {
  [[ -n "$RECEIPT" ]] || return 0
  mkdir -p "$(dirname "$RECEIPT")"
  python - "$RECEIPT" \
    "$FOCUSED_COUNT" "$PUNIT_COUNT" "$PINT_COUNT" "$PSTRESS_COUNT" \
    "$FOCUSED_RC" "$PUNIT_RC" "$PINT_RC" "$PSTRESS_RC" "$IMPORT_RC" "$HELP_RC" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]).expanduser().resolve()
focused,unit,integ,stress=map(int,sys.argv[2:6])
rcs=list(map(int,sys.argv[6:12]))
predecessor=unit+integ+stress
payload={
  "schema_version":1,
  "phase":"I5.1C_QUALIFICATION_RECEIPT",
  "status":"pass" if all(v==0 for v in rcs) else "fail",
  "pass":all(v==0 for v in rcs),
  "scope":{
    "focused_i5_1c_tests":True,
    "predecessor_i5_1_regression":True,
    "oom_correction_regression":True,
    "import_probe":True,
    "runner_help_probe":True,
    "provider_execution_started":False,
    "sfm_rerun":False,
    "dense_pair_rerun":False,
    "quality_replay_rerun":False,
    "thresholds_modified":False,
    "auto_policy_implemented":False,
    "hybrid_authorized":False,
  },
  "counts":{
    "focused_tests":focused,
    "predecessor_unit_tests":unit,
    "predecessor_integration_tests":integ,
    "predecessor_stress_tests":stress,
    "predecessor_total_tests":predecessor,
    "combined_exercised_tests":focused+predecessor,
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
p.parent.mkdir(parents=True,exist_ok=True)
p.write_text(json.dumps(payload,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
print(f"qualification_receipt={p}")
PY
}

case "$MODE" in
  focused)
    python -m pytest -q "${FOCUSED[@]}"
    ;;
  predecessor-unit)
    python -m pytest -q "${PREDECESSOR_UNIT[@]}"
    ;;
  predecessor-integration)
    python -m pytest -q "${PREDECESSOR_INTEGRATION[@]}"
    ;;
  predecessor-stress)
    python -m pytest -q "${PREDECESSOR_STRESS[@]}"
    ;;
  all)
    rc=0
    run_pytest_count FOCUSED "${FOCUSED[@]}" || rc=$?
    probe_import; (( IMPORT_RC == 0 )) || rc=1
    probe_help; (( HELP_RC == 0 )) || rc=1
    run_pytest_count PUNIT "${PREDECESSOR_UNIT[@]}" || rc=$?
    run_pytest_count PINT "${PREDECESSOR_INTEGRATION[@]}" || rc=$?
    run_pytest_count PSTRESS "${PREDECESSOR_STRESS[@]}" || rc=$?
    if [[ $rc -eq 0 ]]; then
      write_receipt || rc=$?
    fi
    echo "I5.1C_TEST_RC=$rc"
    exit "$rc"
    ;;
  *)
    echo "usage: $0 [focused|predecessor-unit|predecessor-integration|predecessor-stress|all] [receipt.json]" >&2
    exit 2
    ;;
esac

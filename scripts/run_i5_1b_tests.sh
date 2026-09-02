#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${1:-all}"
RECEIPT="${2:-}"

run_pytest() {
  local label="$1"; shift
  python -m pytest -q "$@"
  local rc=$?
  echo "${label}_RC=$rc"
  return "$rc"
}

FOCUSED=(
  tests/unit/test_i5_1b_difficult_handheld_replay.py
  tests/integration/test_i5_1b_difficult_handheld_contract.py
  tests/stress/test_i5_1b_difficult_handheld_stress.py
)
AFFECTED_UNIT=(
  tests/unit/test_i1_offline_dense_campaign.py
  tests/unit/test_i4_1_6_decoupled_sampling_integration.py
  tests/unit/test_i4_1_6b_decoupled_dense_ab.py
  tests/unit/test_i5_1a_difficult_handheld_baseline.py
)
AFFECTED_INTEGRATION=(
  tests/integration/test_i4_1_6_decoupled_sampling_contract.py
  tests/integration/test_i4_1_6b_dense_ab_contract.py
  tests/integration/test_i5_1a_difficult_handheld_contract.py
)
AFFECTED_STRESS=(
  tests/stress/test_i4_1_6_decoupled_sampling_stress.py
  tests/stress/test_i4_1_6b_dense_ab_stress.py
  tests/stress/test_i5_1a_difficult_handheld_stress.py
)

rc=0
focused_count=0
unit_count=0
integration_count=0
stress_count=0

case "$MODE" in
  focused)
    run_pytest FOCUSED "${FOCUSED[@]}" || rc=$?
    ;;
  affected-unit)
    run_pytest AUNIT "${AFFECTED_UNIT[@]}" || rc=$?
    ;;
  affected-integration)
    run_pytest AINT "${AFFECTED_INTEGRATION[@]}" || rc=$?
    ;;
  affected-stress)
    run_pytest ASTRESS "${AFFECTED_STRESS[@]}" || rc=$?
    ;;
  all)
    out="$(mktemp)"
    python -m pytest -q "${FOCUSED[@]}" | tee "$out"; frc=${PIPESTATUS[0]}; focused_count="$(grep -Eo '[0-9]+ passed' "$out" | tail -1 | awk '{print $1}')"; rm -f "$out"
    echo "FOCUSED_RC=$frc"; echo "FOCUSED_COUNT=${focused_count:-0}"; (( frc == 0 )) || rc=$frc

    out="$(mktemp)"
    python -m pytest -q "${AFFECTED_UNIT[@]}" | tee "$out"; urc=${PIPESTATUS[0]}; unit_count="$(grep -Eo '[0-9]+ passed' "$out" | tail -1 | awk '{print $1}')"; rm -f "$out"
    echo "AUNIT_RC=$urc"; echo "AUNIT_COUNT=${unit_count:-0}"; (( urc == 0 )) || rc=$urc

    out="$(mktemp)"
    python -m pytest -q "${AFFECTED_INTEGRATION[@]}" | tee "$out"; irc=${PIPESTATUS[0]}; integration_count="$(grep -Eo '[0-9]+ passed' "$out" | tail -1 | awk '{print $1}')"; rm -f "$out"
    echo "AINT_RC=$irc"; echo "AINT_COUNT=${integration_count:-0}"; (( irc == 0 )) || rc=$irc

    out="$(mktemp)"
    python -m pytest -q "${AFFECTED_STRESS[@]}" | tee "$out"; src=${PIPESTATUS[0]}; stress_count="$(grep -Eo '[0-9]+ passed' "$out" | tail -1 | awk '{print $1}')"; rm -f "$out"
    echo "ASTRESS_RC=$src"; echo "ASTRESS_COUNT=${stress_count:-0}"; (( src == 0 )) || rc=$src

    python -c 'from core.reconstruction.i5_difficult_handheld_robustness import I51B_PHASE, build_i5_1b_difficult_handheld_replay; assert I51B_PHASE == "I5.1B" and callable(build_i5_1b_difficult_handheld_replay)'
    import_rc=$?
    python scripts/run_i5_1b_difficult_handheld_replay.py --help >/dev/null
    help_rc=$?
    echo "import_probe_rc=$import_rc"
    echo "replay_runner_help_rc=$help_rc"
    (( import_rc == 0 && help_rc == 0 )) || rc=1

    if [[ -n "$RECEIPT" ]]; then
      mkdir -p "$(dirname "$RECEIPT")"
      python - "$RECEIPT" "${focused_count:-0}" "${unit_count:-0}" "${integration_count:-0}" "${stress_count:-0}" "$frc" "$urc" "$irc" "$src" "$import_rc" "$help_rc" <<'PY'
import json, sys
from pathlib import Path
p=Path(sys.argv[1])
focused,unit,integ,stress=map(int,sys.argv[2:6])
rcs=list(map(int,sys.argv[6:12]))
payload={
 "phase":"I5.1B_REPLAY_RECEIPT","status":"pass" if all(v==0 for v in rcs) else "failed","pass":all(v==0 for v in rcs),
 "counts":{"focused_tests":focused,"affected_unit_tests":unit,"affected_integration_tests":integ,"affected_stress_tests":stress,"affected_total_tests":unit+integ+stress,"combined_exercised_tests":focused+unit+integ+stress},
 "return_codes":{"focused":rcs[0],"affected_unit":rcs[1],"affected_integration":rcs[2],"affected_stress":rcs[3],"import_probe":rcs[4],"runner_help_probe":rcs[5]},
 "scope":{"real_replay_executed_by_test_suite":False,"provider_inference_executed_by_test_suite":False,"sfm_rerun_by_test_suite":False,"thresholds_modified":False,"auto_policy_implemented":False,"hybrid_authorized":False}
}
p.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
print(f"replay_receipt={p}")
PY
    fi
    ;;
  *)
    echo "usage: $0 [focused|affected-unit|affected-integration|affected-stress|all] [receipt.json]" >&2
    exit 2
    ;;
esac

echo "I5.1B_TEST_RC=$rc"
exit "$rc"

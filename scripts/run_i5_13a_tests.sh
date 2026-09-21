#!/usr/bin/env bash
set -u

MODE="${1:-all}"
RECEIPT="${2:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

focused=(
  tests/unit/test_i5_13a_auto_policy_admission.py
  tests/integration/test_i5_13a_auto_policy_admission_contract.py
  tests/stress/test_i5_13a_auto_policy_admission_stress.py
)
pre_unit=(
  tests/unit/test_i4_3a_dense_provider_architecture.py
  tests/unit/test_i4_3g5b_cross_provider_qualification_closure.py
  tests/unit/test_i5_6c_mvs_robustness_closure.py
  tests/unit/test_i5_7c_rolling_depth_robustness_closure.py
  tests/unit/test_i5_8c_video_depth_anything_robustness_closure.py
  tests/unit/test_i5_9c_remaining_depth_provider_robustness_closure.py
  tests/unit/test_i5_10d_mp_sfm_closure.py
  tests/unit/test_i5_11d_failure_robustness_closure.py
  tests/unit/test_i5_12a_hybrid_admission.py
)
pre_int=(
  tests/integration/test_i4_3a_dense_provider_architecture_contract.py
  tests/integration/test_i4_3g5b_final_g_closure_contract.py
  tests/integration/test_i5_6c_mvs_robustness_contract.py
  tests/integration/test_i5_7c_rolling_depth_robustness_contract.py
  tests/integration/test_i5_8c_video_depth_anything_robustness_contract.py
  tests/integration/test_i5_9c_remaining_depth_provider_robustness_contract.py
  tests/integration/test_i5_10d_mp_sfm_closure_contract.py
  tests/integration/test_i5_11d_failure_robustness_closure_contract.py
  tests/integration/test_i5_12a_hybrid_admission_contract.py
)
pre_stress=(
  tests/stress/test_i4_3a_dense_provider_architecture_stress.py
  tests/stress/test_i4_3g5b_final_g_closure_stress.py
  tests/stress/test_i5_6c_mvs_robustness_stress.py
  tests/stress/test_i5_7c_rolling_depth_robustness_stress.py
  tests/stress/test_i5_8c_video_depth_anything_robustness_stress.py
  tests/stress/test_i5_9c_remaining_depth_provider_robustness_stress.py
  tests/stress/test_i5_10d_mp_sfm_closure_stress.py
  tests/stress/test_i5_11d_failure_robustness_closure_stress.py
  tests/stress/test_i5_12a_hybrid_admission_stress.py
)

run_pytest() {
  local label="$1"; shift
  pytest -q "$@"
  local rc=$?
  echo "${label}_RC=$rc"
  return $rc
}

case "$MODE" in
  focused)
    run_pytest FOCUSED "${focused[@]}"; exit $?
    ;;
  all)
    TMPDIR_I513A="$(mktemp -d)"
    trap 'rm -rf "$TMPDIR_I513A"' EXIT
    set +e
    pytest -q "${focused[@]}" | tee "$TMPDIR_I513A/focused.log"; FOCUSED_RC=${PIPESTATUS[0]}
    pytest -q "${pre_unit[@]}" | tee "$TMPDIR_I513A/punit.log"; PUNIT_RC=${PIPESTATUS[0]}
    pytest -q "${pre_int[@]}" | tee "$TMPDIR_I513A/pint.log"; PINT_RC=${PIPESTATUS[0]}
    pytest -q "${pre_stress[@]}" | tee "$TMPDIR_I513A/pstress.log"; PSTRESS_RC=${PIPESTATUS[0]}
    python - <<'PY2'
from core.reconstruction.i5_auto_policy_admission import build_i5_13a_auto_policy_admission
print("I5.13A Auto-policy admission contract/import probe: PASS")
PY2
    IMPORT_RC=$?
    python scripts/run_i5_13a_auto_policy_admission.py --help >/dev/null; HELP_RC=$?
    set -e

    echo "FOCUSED_RC=$FOCUSED_RC"
    echo "PUNIT_RC=$PUNIT_RC"
    echo "PINT_RC=$PINT_RC"
    echo "PSTRESS_RC=$PSTRESS_RC"
    echo "import_probe_rc=$IMPORT_RC"
    echo "admission_runner_help_rc=$HELP_RC"

    extract_count() {
      local log="$1"
      local value
      value="$(grep -Eo '[0-9]+ passed' "$log" | tail -n 1 | awk '{print $1}')"
      [[ "$value" =~ ^[0-9]+$ ]] || value=0
      printf '%s' "$value"
    }
    FOCUSED_COUNT="$(extract_count "$TMPDIR_I513A/focused.log")"
    PUNIT_COUNT="$(extract_count "$TMPDIR_I513A/punit.log")"
    PINT_COUNT="$(extract_count "$TMPDIR_I513A/pint.log")"
    PSTRESS_COUNT="$(extract_count "$TMPDIR_I513A/pstress.log")"
    PRE=$((PUNIT_COUNT + PINT_COUNT + PSTRESS_COUNT))
    COMBINED=$((FOCUSED_COUNT + PRE))
    RC=0
    for value in "$FOCUSED_RC" "$PUNIT_RC" "$PINT_RC" "$PSTRESS_RC" "$IMPORT_RC" "$HELP_RC"; do
      [[ "$value" -eq 0 ]] || RC=1
    done

    if [[ -n "$RECEIPT" ]]; then
      mkdir -p "$(dirname "$RECEIPT")"
      python - "$RECEIPT" "$FOCUSED_COUNT" "$PUNIT_COUNT" "$PINT_COUNT" "$PSTRESS_COUNT" "$PRE" "$COMBINED" "$FOCUSED_RC" "$PUNIT_RC" "$PINT_RC" "$PSTRESS_RC" "$IMPORT_RC" "$HELP_RC" <<'PY2'
import json, sys
(path, focused, pu, pi, ps, pred, combined, frc, urc, irc, src, imprc, helprc) = sys.argv[1:]
all_zero = all(int(x) == 0 for x in (frc, urc, irc, src, imprc, helprc))
payload = {
    "schema_version": 1,
    "phase": "I5.13A_ADMISSION_RECEIPT",
    "status": "pass" if all_zero else "failed",
    "pass": all_zero,
    "scope": {
        "focused_i5_13a_tests": True,
        "dense_architecture_regression": True,
        "cross_provider_closure_regression": True,
        "i5_10_to_i5_12_regression": True,
        "import_probe": True,
        "runner_help_probe": True,
        "provider_execution": False,
        "sfm_execution": False,
        "mvs_execution": False,
        "auto_policy_engine_implemented": False,
        "auto_provider_selection_performed": False,
        "production_auto_integration": False,
        "production_default_modified": False,
        "hybrid_execution": False,
    },
    "counts": {
        "focused_tests": int(focused),
        "predecessor_unit_tests": int(pu),
        "predecessor_integration_tests": int(pi),
        "predecessor_stress_tests": int(ps),
        "predecessor_total_tests": int(pred),
        "combined_exercised_tests": int(combined),
    },
    "return_codes": {
        "focused": int(frc),
        "predecessor_unit": int(urc),
        "predecessor_integration": int(irc),
        "predecessor_stress": int(src),
        "import_probe": int(imprc),
        "runner_help_probe": int(helprc),
    },
}
with open(path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, sort_keys=True)
    f.write("\n")
print(f"admission_receipt={path}")
PY2
    fi
    echo "FOCUSED_COUNT=$FOCUSED_COUNT"
    echo "PUNIT_COUNT=$PUNIT_COUNT"
    echo "PINT_COUNT=$PINT_COUNT"
    echo "PSTRESS_COUNT=$PSTRESS_COUNT"
    echo "I5.13A_TEST_RC=$RC"
    exit "$RC"
    ;;
  *)
    echo "usage: $0 [focused|all] [receipt.json]" >&2
    exit 2
    ;;
esac

#!/usr/bin/env bash
set -u

MODE="${1:-all}"
RECEIPT="${2:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

focused=(
  tests/unit/test_i5_13d_auto_production_integration_decision.py
  tests/integration/test_i5_13d_auto_production_integration_decision_contract.py
  tests/stress/test_i5_13d_auto_production_integration_decision_stress.py
)
pre_unit=(
  tests/unit/test_i5_13c_auto_policy_replay.py
  tests/unit/test_i5_13b_auto_policy_engine.py
  tests/unit/test_i5_13a_auto_policy_admission.py
  tests/unit/test_i4_3a_dense_provider_architecture.py
)
pre_int=(
  tests/integration/test_i5_13c_auto_policy_replay_contract.py
  tests/integration/test_i5_13b_auto_policy_engine_contract.py
  tests/integration/test_i5_13a_auto_policy_admission_contract.py
  tests/integration/test_i4_3a_dense_provider_architecture_contract.py
)
pre_stress=(
  tests/stress/test_i5_13c_auto_policy_replay_stress.py
  tests/stress/test_i5_13b_auto_policy_engine_stress.py
  tests/stress/test_i5_13a_auto_policy_admission_stress.py
  tests/stress/test_i4_3a_dense_provider_architecture_stress.py
)

case "$MODE" in
  focused)
    pytest -q "${focused[@]}"; exit $?
    ;;
  all)
    TMPDIR_I513D="$(mktemp -d)"
    trap 'rm -rf "$TMPDIR_I513D"' EXIT
    set +e
    pytest -q "${focused[@]}" | tee "$TMPDIR_I513D/focused.log"; FOCUSED_RC=${PIPESTATUS[0]}
    pytest -q "${pre_unit[@]}" | tee "$TMPDIR_I513D/punit.log"; PUNIT_RC=${PIPESTATUS[0]}
    pytest -q "${pre_int[@]}" | tee "$TMPDIR_I513D/pint.log"; PINT_RC=${PIPESTATUS[0]}
    pytest -q "${pre_stress[@]}" | tee "$TMPDIR_I513D/pstress.log"; PSTRESS_RC=${PIPESTATUS[0]}
    python - <<'PY2'
from pathlib import Path
from core.reconstruction.i5_auto_policy_production_qualification import inspect_i5_13d_production_integration_surface
root = Path.cwd()
inv = inspect_i5_13d_production_integration_surface(root)
assert inv["adaptive_production_planning_mode"] == "explicit_provider_only"
assert inv["production_ui_provider_selection_mode"] == "inspection_only"
assert inv["production_auto_orchestration_entrypoint_implemented"] is False
print("I5.13D production-integration decision contract/import probe: PASS")
PY2
    IMPORT_RC=$?
    python scripts/run_i5_13d_auto_production_integration_decision.py --help >/dev/null; HELP_RC=$?
    set -e

    echo "FOCUSED_RC=$FOCUSED_RC"
    echo "PUNIT_RC=$PUNIT_RC"
    echo "PINT_RC=$PINT_RC"
    echo "PSTRESS_RC=$PSTRESS_RC"
    echo "import_probe_rc=$IMPORT_RC"
    echo "decision_runner_help_rc=$HELP_RC"

    extract_count() {
      local log="$1"
      local value
      value="$(grep -Eo '[0-9]+ passed' "$log" | tail -n 1 | awk '{print $1}')"
      [[ "$value" =~ ^[0-9]+$ ]] || value=0
      printf '%s' "$value"
    }
    FOCUSED_COUNT="$(extract_count "$TMPDIR_I513D/focused.log")"
    PUNIT_COUNT="$(extract_count "$TMPDIR_I513D/punit.log")"
    PINT_COUNT="$(extract_count "$TMPDIR_I513D/pint.log")"
    PSTRESS_COUNT="$(extract_count "$TMPDIR_I513D/pstress.log")"
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
    "phase": "I5.13D_INTEGRATION_DECISION_RECEIPT",
    "status": "pass" if all_zero else "failed",
    "pass": all_zero,
    "scope": {
        "focused_i5_13d_tests": True,
        "i5_13c_regression": True,
        "production_integration_surface_regression": True,
        "evidence_only_decision": True,
        "import_probe": True,
        "runner_help_probe": True,
        "provider_execution": False,
        "sfm_execution": False,
        "mvs_execution": False,
        "scene_classifier_execution": False,
        "production_auto_integration": False,
        "production_default_modified": False,
        "provider_promoted": False,
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
print(f"decision_receipt={path}")
PY2
    fi
    echo "FOCUSED_COUNT=$FOCUSED_COUNT"
    echo "PUNIT_COUNT=$PUNIT_COUNT"
    echo "PINT_COUNT=$PINT_COUNT"
    echo "PSTRESS_COUNT=$PSTRESS_COUNT"
    echo "I5.13D_TEST_RC=$RC"
    exit "$RC"
    ;;
  *)
    echo "usage: $0 [focused|all] [receipt.json]" >&2
    exit 2
    ;;
esac

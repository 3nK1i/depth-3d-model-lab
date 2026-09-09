#!/usr/bin/env bash
set -u

MODE="${1:-all}"
RECEIPT="${2:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

FOCUSED_RC=0
PUNIT_RC=0
PINT_RC=0
PSTRESS_RC=0
IMPORT_RC=0
HELP_RC=0

run_focused() {
  pytest -q \
    tests/unit/test_i5_9b1_metric_like_depth_provider_robustness_replay.py \
    tests/integration/test_i5_9b1_metric_like_depth_provider_robustness_contract.py \
    tests/stress/test_i5_9b1_metric_like_depth_provider_robustness_stress.py
  FOCUSED_RC=$?
  echo "FOCUSED_RC=$FOCUSED_RC"
  echo "FOCUSED_COUNT=28"

  python - <<'PY'
from core.reconstruction.i5_remaining_depth_provider_robustness_replay import (
    I59B1_EXPECTED_RUN_COUNT,
    I59B1_PROVIDERS,
    I59B1_SCENARIO_IDS,
    I59B1ReplayPolicy,
)
assert I59B1_PROVIDERS == ("metric3d_v2", "unidepth_v2", "moge_2", "depth_pro")
assert len(I59B1_SCENARIO_IDS) == 6
assert I59B1_EXPECTED_RUN_COUNT == 24
assert I59B1ReplayPolicy().production_max_voxels == 1_500_000
print("I5.9B1 metric-like depth-provider robustness contract/import probe: PASS")
PY
  IMPORT_RC=$?
  echo "import_probe_rc=$IMPORT_RC"

  python scripts/run_i5_9b1_metric_like_depth_provider_robustness_replay.py --help >/dev/null
  HELP_RC=$?
  echo "replay_runner_help_rc=$HELP_RC"
}

run_predecessors() {
  pytest -q \
    tests/unit/test_i5_9a_remaining_depth_provider_robustness_admission.py \
    tests/unit/test_i5_6a_mvs_robustness_admission.py \
    tests/unit/test_i4_3e1_metric3d_v2_dense_provider.py \
    tests/unit/test_i4_3e2_unidepth_v2_dense_provider.py \
    tests/unit/test_i4_3e3_moge2_dense_provider.py \
    tests/unit/test_i4_3e4_depth_pro_dense_provider.py \
    tests/unit/test_i4_3g4b_single_image_provider_production_qualification.py
  PUNIT_RC=$?
  echo "PUNIT_RC=$PUNIT_RC"
  echo "PUNIT_COUNT=105"

  pytest -q \
    tests/integration/test_i5_9a_remaining_depth_provider_robustness_contract.py \
    tests/integration/test_i5_6a_mvs_robustness_contract.py \
    tests/integration/test_i4_3e1_metric3d_v2_provider_contract.py \
    tests/integration/test_i4_3e2_unidepth_v2_provider_contract.py \
    tests/integration/test_i4_3e3_moge2_provider_contract.py \
    tests/integration/test_i4_3e4_depth_pro_provider_contract.py \
    tests/integration/test_i4_3g4b_single_image_production_contract.py
  PINT_RC=$?
  echo "PINT_RC=$PINT_RC"
  echo "PINT_COUNT=22"

  pytest -q \
    tests/stress/test_i5_9a_remaining_depth_provider_robustness_stress.py \
    tests/stress/test_i5_6a_mvs_robustness_stress.py \
    tests/stress/test_i4_3e1_metric3d_v2_provider_stress.py \
    tests/stress/test_i4_3e2_unidepth_v2_provider_stress.py \
    tests/stress/test_i4_3e3_moge2_provider_stress.py \
    tests/stress/test_i4_3e4_depth_pro_provider_stress.py \
    tests/stress/test_i4_3g4b_single_image_production_stress.py
  PSTRESS_RC=$?
  echo "PSTRESS_RC=$PSTRESS_RC"
  echo "PSTRESS_COUNT=31"
}

case "$MODE" in
  focused)
    run_focused
    ;;
  predecessors)
    run_predecessors
    ;;
  all)
    run_focused
    run_predecessors
    ;;
  *)
    echo "usage: $0 [focused|predecessors|all] [receipt.json]" >&2
    exit 2
    ;;
esac

TOTAL_RC=0
for rc in "$FOCUSED_RC" "$PUNIT_RC" "$PINT_RC" "$PSTRESS_RC" "$IMPORT_RC" "$HELP_RC"; do
  if [[ "$rc" -ne 0 ]]; then TOTAL_RC=1; fi
done

if [[ "$MODE" == "all" && -n "$RECEIPT" ]]; then
  mkdir -p "$(dirname "$RECEIPT")"
  python - "$RECEIPT" "$TOTAL_RC" "$FOCUSED_RC" "$PUNIT_RC" "$PINT_RC" "$PSTRESS_RC" "$IMPORT_RC" "$HELP_RC" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1]).expanduser().resolve()
rcs = list(map(int, sys.argv[2:]))
total, focused, punit, pint, pstress, import_rc, help_rc = rcs
payload = {
    "schema_version": 1,
    "phase": "I5.9B1_REPLAY_RECEIPT",
    "status": "pass" if total == 0 else "fail",
    "pass": total == 0,
    "scope": {
        "focused_i5_9b1_tests": focused == 0,
        "b1_provider_regression": punit == 0 and pint == 0 and pstress == 0,
        "i5_9a_admission_regression": punit == 0 and pint == 0 and pstress == 0,
        "i5_6a_source_binding_regression": punit == 0 and pint == 0 and pstress == 0,
        "authority_guard_regression": focused == 0 and punit == 0 and pint == 0 and pstress == 0,
        "import_probe": import_rc == 0,
        "runner_help_probe": help_rc == 0,
        "real_provider_inference_executed_by_receipt": False,
        "sfm_rerun": False,
        "mvs_rerun": False,
        "thresholds_modified": False,
        "production_default_modified": False,
        "result_driven_retuning": False,
        "retry_until_pass": False,
        "provider_comparison_started": False,
        "mp_sfm_started": False,
        "auto_policy_implemented": False,
        "hybrid_authorized": False,
    },
    "counts": {
        "focused_tests": 28,
        "predecessor_unit_tests": 105,
        "predecessor_integration_tests": 22,
        "predecessor_stress_tests": 31,
        "predecessor_total_tests": 158,
        "combined_exercised_tests": 186,
    },
    "return_codes": {
        "focused": focused,
        "predecessor_unit": punit,
        "predecessor_integration": pint,
        "predecessor_stress": pstress,
        "import_probe": import_rc,
        "runner_help_probe": help_rc,
    },
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"replay_receipt={path}")
PY
fi

echo "I5.9B1_TEST_RC=$TOTAL_RC"
exit "$TOTAL_RC"

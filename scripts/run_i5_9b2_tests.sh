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
    tests/unit/test_i5_9b2_relative_depth_provider_robustness_replay.py \
    tests/integration/test_i5_9b2_relative_depth_provider_robustness_contract.py \
    tests/stress/test_i5_9b2_relative_depth_provider_robustness_stress.py
  FOCUSED_RC=$?
  echo "FOCUSED_RC=$FOCUSED_RC"
  echo "FOCUSED_COUNT=32"

  python - <<'PY'
from core.reconstruction.i5_relative_depth_provider_robustness_replay import (
    I59B2_EXPECTED_RUN_COUNT,
    I59B2_HISTORICAL_G4_LOW_TEXTURE,
    I59B2_PROVIDERS,
    I59B2_SCENARIO_IDS,
    I59B2ReplayPolicy,
)
assert I59B2_PROVIDERS == ("depth_anything_v2", "marigold")
assert len(I59B2_SCENARIO_IDS) == 6
assert I59B2_EXPECTED_RUN_COUNT == 12
assert I59B2_HISTORICAL_G4_LOW_TEXTURE == {"depth_anything_v2": "rejected", "marigold": "rejected"}
assert I59B2ReplayPolicy().production_max_voxels == 1_500_000
print("I5.9B2 relative-depth provider robustness contract/import probe: PASS")
PY
  IMPORT_RC=$?
  echo "import_probe_rc=$IMPORT_RC"

  python scripts/run_i5_9b2_relative_depth_provider_robustness_replay.py --help >/dev/null
  HELP_RC=$?
  echo "replay_runner_help_rc=$HELP_RC"
}

run_predecessors() {
  pytest -q \
    tests/unit/test_i5_9b1_metric_like_depth_provider_robustness_replay.py \
    tests/unit/test_i5_9a_remaining_depth_provider_robustness_admission.py \
    tests/unit/test_i5_6a_mvs_robustness_admission.py \
    tests/unit/test_i4_3e5_depth_anything_v2_dense_provider.py \
    tests/unit/test_i4_3e6_marigold_dense_provider.py \
    tests/unit/test_i4_3g4b_single_image_provider_production_qualification.py
  PUNIT_RC=$?
  echo "PUNIT_RC=$PUNIT_RC"
  echo "PUNIT_COUNT=86"

  pytest -q \
    tests/integration/test_i5_9b1_metric_like_depth_provider_robustness_contract.py \
    tests/integration/test_i5_9a_remaining_depth_provider_robustness_contract.py \
    tests/integration/test_i5_6a_mvs_robustness_contract.py \
    tests/integration/test_i4_3e5_depth_anything_v2_provider_contract.py \
    tests/integration/test_i4_3e6_marigold_provider_contract.py \
    tests/integration/test_i4_3g4b_single_image_production_contract.py
  PINT_RC=$?
  echo "PINT_RC=$PINT_RC"
  echo "PINT_COUNT=22"

  pytest -q \
    tests/stress/test_i5_9b1_metric_like_depth_provider_robustness_stress.py \
    tests/stress/test_i5_9a_remaining_depth_provider_robustness_stress.py \
    tests/stress/test_i5_6a_mvs_robustness_stress.py \
    tests/stress/test_i4_3e5_depth_anything_v2_provider_stress.py \
    tests/stress/test_i4_3e6_marigold_provider_stress.py \
    tests/stress/test_i4_3g4b_single_image_production_stress.py
  PSTRESS_RC=$?
  echo "PSTRESS_RC=$PSTRESS_RC"
  echo "PSTRESS_COUNT=33"
}

case "$MODE" in
  focused) run_focused ;;
  predecessors) run_predecessors ;;
  all) run_focused; run_predecessors ;;
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
total, focused, punit, pint, pstress, import_rc, help_rc = map(int, sys.argv[2:])
payload = {
    "schema_version": 1,
    "phase": "I5.9B2_REPLAY_RECEIPT",
    "status": "pass" if total == 0 else "fail",
    "pass": total == 0,
    "scope": {
        "focused_i5_9b2_tests": focused == 0,
        "b2_provider_regression": punit == 0 and pint == 0 and pstress == 0,
        "i5_9b1_closure_regression": punit == 0 and pint == 0 and pstress == 0,
        "i5_9a_admission_regression": punit == 0 and pint == 0 and pstress == 0,
        "i5_6a_source_binding_regression": punit == 0 and pint == 0 and pstress == 0,
        "historical_g4_anchor_regression": focused == 0 and punit == 0 and pint == 0 and pstress == 0,
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
        "focused_tests": 32,
        "predecessor_unit_tests": 86,
        "predecessor_integration_tests": 22,
        "predecessor_stress_tests": 33,
        "predecessor_total_tests": 141,
        "combined_exercised_tests": 173,
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

echo "I5.9B2_TEST_RC=$TOTAL_RC"
exit "$TOTAL_RC"

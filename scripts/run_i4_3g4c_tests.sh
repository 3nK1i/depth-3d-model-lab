#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${1:-all}"
RC=0

run_pytest() {
  python -m pytest -q "$@"
  local rc=$?
  echo "pytest_rc=$rc"
  if [ "$rc" -ne 0 ]; then RC=$rc; fi
}

if [ "$MODE" = "focused" ] || [ "$MODE" = "all" ]; then
  run_pytest \
    tests/unit/test_i4_3g4c_single_image_matched_view_review.py \
    tests/integration/test_i4_3g4c_single_image_review_contract.py \
    tests/stress/test_i4_3g4c_single_image_review_stress.py

  python - <<'PY'
from core.reconstruction.single_image_provider_matched_view_review import (
    I43G4C_PROVIDER_IMPLEMENTATIONS,
    I43G4C_REVIEW_DIMENSIONS,
    I43G4C_SCENARIOS,
)
assert I43G4C_SCENARIOS == (
    "g4_single_image_complete_room",
    "g4_single_image_low_texture",
    "g4_single_image_weak_overlap",
)
assert I43G4C_PROVIDER_IMPLEMENTATIONS == (
    "metric3d_v2",
    "unidepth_v2",
    "moge_2",
    "depth_pro",
    "depth_anything_v2",
    "marigold",
)
assert len(I43G4C_REVIEW_DIMENSIONS) == 7
print("I4.3G4C single-image matched-view contract/import probe: PASS")
PY
  probe_rc=$?
  echo "import_probe_rc=$probe_rc"
  if [ "$probe_rc" -ne 0 ]; then RC=$probe_rc; fi

  python scripts/run_i4_3g4c_single_image_review.py --help >/dev/null
  help_rc=$?
  echo "review_runner_help_rc=$help_rc"
  if [ "$help_rc" -ne 0 ]; then RC=$help_rc; fi
fi

if [ "$MODE" = "regression" ] || [ "$MODE" = "all" ]; then
  run_pytest \
    tests/unit/test_i4_3g1_cross_provider_qualification.py \
    tests/integration/test_i4_3g1_qualification_framework_contract.py \
    tests/stress/test_i4_3g1_qualification_framework_stress.py \
    tests/unit/test_i4_3g2c_matched_view_review.py \
    tests/unit/test_i4_3g2c_remaining_scenario_review.py \
    tests/integration/test_i4_3g2c_entry_contract.py \
    tests/integration/test_i4_3g2c_remaining_scenario_contract.py \
    tests/stress/test_i4_3g2c_review_stress.py \
    tests/stress/test_i4_3g2c_remaining_scenario_stress.py \
    tests/unit/test_i4_3g3c_temporal_matched_view_review.py \
    tests/integration/test_i4_3g3c_temporal_review_contract.py \
    tests/stress/test_i4_3g3c_temporal_review_stress.py \
    tests/unit/test_i4_3g4a_single_image_provider_qualification.py \
    tests/integration/test_i4_3g4a_single_image_preflight_contract.py \
    tests/stress/test_i4_3g4a_single_image_preflight_stress.py \
    tests/unit/test_i4_3g4b_single_image_provider_production_qualification.py \
    tests/integration/test_i4_3g4b_single_image_production_contract.py \
    tests/stress/test_i4_3g4b_single_image_production_stress.py \
    tests/unit/test_i4_3e1_metric3d_v2_dense_provider.py
fi

echo "I4.3G4C_TEST_RC=$RC"
exit "$RC"

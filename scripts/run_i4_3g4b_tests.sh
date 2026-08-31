#!/usr/bin/env bash
set +e
set -u
set -o pipefail

MODE="${1:-all}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 2
export PYTHONDONTWRITEBYTECODE=1

run_pytest() {
  python -m pytest -q -p no:cacheprovider "$@"
  local rc=$?
  printf 'pytest_rc=%s\n' "$rc"
  return "$rc"
}

run_focused() {
  run_pytest \
    tests/unit/test_i4_3g4b_single_image_provider_production_qualification.py \
    tests/integration/test_i4_3g4b_single_image_production_contract.py \
    tests/stress/test_i4_3g4b_single_image_production_stress.py
  local rc=$?
  if (( rc != 0 )); then
    return "$rc"
  fi

  python - <<'PY'
from core.reconstruction.single_image_provider_production_qualification import (
    I43G4B_PROVIDER_IMPLEMENTATIONS,
    I43G4B_SCENARIOS,
)
assert I43G4B_SCENARIOS == (
    "g4_single_image_complete_room",
    "g4_single_image_low_texture",
    "g4_single_image_weak_overlap",
)
assert I43G4B_PROVIDER_IMPLEMENTATIONS == (
    "metric3d_v2",
    "unidepth_v2",
    "moge_2",
    "depth_pro",
    "depth_anything_v2",
    "marigold",
)
print("I4.3G4B single-image production-native contract/import probe: PASS")
PY
  rc=$?
  printf 'import_probe_rc=%s\n' "$rc"
  if (( rc != 0 )); then
    return "$rc"
  fi

  python scripts/run_i4_3g4b_single_image_scenario.py --help >/dev/null
  rc=$?
  printf 'scenario_runner_help_rc=%s\n' "$rc"
  return "$rc"
}

run_backward() {
  run_pytest \
    tests/unit/test_i4_3e1_metric3d_v2_dense_provider.py \
    tests/integration/test_i4_3e1_metric3d_v2_provider_contract.py \
    tests/unit/test_i4_3e2_unidepth_v2_dense_provider.py \
    tests/integration/test_i4_3e2_unidepth_v2_provider_contract.py \
    tests/unit/test_i4_3e3_moge2_dense_provider.py \
    tests/integration/test_i4_3e3_moge2_provider_contract.py \
    tests/unit/test_i4_3e4_depth_pro_dense_provider.py \
    tests/integration/test_i4_3e4_depth_pro_provider_contract.py \
    tests/unit/test_i4_3e5_depth_anything_v2_dense_provider.py \
    tests/integration/test_i4_3e5_depth_anything_v2_provider_contract.py \
    tests/unit/test_i4_3e6_marigold_dense_provider.py \
    tests/integration/test_i4_3e6_marigold_provider_contract.py \
    tests/unit/test_i4_3f2_adaptive_production_planning.py \
    tests/integration/test_i4_3f2_production_wiring_contract.py \
    tests/unit/test_i4_3f3_adaptive_selection_closure.py \
    tests/integration/test_i4_3f3_production_validation_contract.py \
    tests/unit/test_i4_3g1_cross_provider_qualification.py \
    tests/integration/test_i4_3g1_qualification_framework_contract.py \
    tests/stress/test_i4_3g1_qualification_framework_stress.py \
    tests/unit/test_i4_3g4a_single_image_provider_qualification.py \
    tests/integration/test_i4_3g4a_single_image_preflight_contract.py \
    tests/stress/test_i4_3g4a_single_image_preflight_stress.py
}

case "$MODE" in
  focused)
    run_focused
    rc=$?
    ;;
  backward)
    run_backward
    rc=$?
    ;;
  all)
    run_focused
    rc=$?
    if (( rc == 0 )); then
      run_backward
      rc=$?
    fi
    ;;
  *)
    echo "usage: $0 [focused|backward|all]" >&2
    rc=2
    ;;
esac

printf 'I4.3G4B_TEST_RC=%s\n' "$rc"
exit "$rc"

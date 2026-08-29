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
    tests/unit/test_i4_3g3c_temporal_matched_view_review.py \
    tests/integration/test_i4_3g3c_temporal_review_contract.py \
    tests/stress/test_i4_3g3c_temporal_review_stress.py
  local rc=$?
  if (( rc != 0 )); then
    return "$rc"
  fi

  python - <<'PY'
from core.reconstruction.temporal_provider_matched_view_review import (
    I43G3C_REVIEW_DIMENSIONS,
    I43G3C_SCENARIOS,
)
assert I43G3C_SCENARIOS == (
    "g3_temporal_static_sweep",
    "g3_temporal_motion",
    "g3_temporal_low_texture",
)
assert I43G3C_REVIEW_DIMENSIONS == (
    "temporal_flicker",
    "temporal_drift",
    "static_geometry_consistency",
    "motion_boundary_stability",
    "transient_artifacts",
    "textureless_region_stability",
)
print("I4.3G3C temporal matched-view/review contract import probe: PASS")
PY
  rc=$?
  printf 'import_probe_rc=%s\n' "$rc"
  if (( rc != 0 )); then
    return "$rc"
  fi

  python scripts/run_i4_3g3c_temporal_review.py --help >/dev/null
  rc=$?
  printf 'review_runner_help_rc=%s\n' "$rc"
  return "$rc"
}

run_backward() {
  run_pytest \
    tests/unit/test_i4_3g3b_temporal_production_qualification.py \
    tests/integration/test_i4_3g3b_temporal_production_contract.py \
    tests/stress/test_i4_3g3b_temporal_production_stress.py \
    tests/unit/test_i4_3g3a_temporal_provider_qualification.py \
    tests/integration/test_i4_3g3a_temporal_preflight_contract.py \
    tests/stress/test_i4_3g3a_temporal_preflight_stress.py \
    tests/unit/test_i4_3g2c_matched_view_review.py \
    tests/integration/test_i4_3g2c_entry_contract.py \
    tests/stress/test_i4_3g2c_review_stress.py \
    tests/unit/test_i4_3g1_cross_provider_qualification.py \
    tests/integration/test_i4_3g1_qualification_framework_contract.py \
    tests/stress/test_i4_3g1_qualification_framework_stress.py
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

printf 'I4.3G3C_TEST_RC=%s\n' "$rc"
exit "$rc"

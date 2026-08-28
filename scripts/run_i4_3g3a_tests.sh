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
    tests/unit/test_i4_3g3a_temporal_provider_qualification.py \
    tests/integration/test_i4_3g3a_temporal_preflight_contract.py \
    tests/stress/test_i4_3g3a_temporal_preflight_stress.py
  local rc=$?
  if (( rc != 0 )); then
    return "$rc"
  fi

  python - <<'PY'
from core.reconstruction.temporal_provider_qualification import I43G3A_SCENARIOS
assert I43G3A_SCENARIOS == (
    "g3_temporal_static_sweep",
    "g3_temporal_motion",
    "g3_temporal_low_texture",
)
print("I4.3G3A temporal preflight contract/import probe: PASS")
PY
  rc=$?
  printf 'import_probe_rc=%s\n' "$rc"
  if (( rc != 0 )); then
    return "$rc"
  fi

  python scripts/run_i4_3g3a_temporal_preflight.py --help >/dev/null
  rc=$?
  printf 'preflight_help_rc=%s\n' "$rc"
  return "$rc"
}

run_backward() {
  run_pytest \
    tests/unit/test_i4_3d1_video_depth_anything_dense_provider.py \
    tests/integration/test_i4_3d1_video_depth_anything_provider_contract.py \
    tests/unit/test_i4_3d2_rolling_depth_dense_provider.py \
    tests/integration/test_i4_3d2_rolling_depth_provider_contract.py \
    tests/unit/test_i4_3d3_temporal_provider_closure.py \
    tests/integration/test_i4_3d3_temporal_provider_closure_contract.py \
    tests/stress/test_i4_3d1_video_depth_anything_provider_stress.py \
    tests/stress/test_i4_3d2_rolling_depth_provider_stress.py \
    tests/stress/test_i4_3d3_temporal_provider_closure_stress.py \
    tests/unit/test_i4_3f1_adaptive_view_selection.py \
    tests/integration/test_i4_3f1_adaptive_view_selection_contract.py \
    tests/unit/test_i4_3f2_adaptive_production_planning.py \
    tests/integration/test_i4_3f2_production_wiring_contract.py \
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

printf 'I4.3G3A_TEST_RC=%s\n' "$rc"
exit "$rc"

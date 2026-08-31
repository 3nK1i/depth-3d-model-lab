#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${1:-all}"

run_pytest() {
  python -m pytest -q "$@"
  local rc=$?
  echo "pytest_rc=$rc"
  return "$rc"
}

focused() {
  run_pytest \
    tests/unit/test_i4_3g5b_cross_provider_qualification_closure.py \
    tests/integration/test_i4_3g5b_final_g_closure_contract.py \
    tests/stress/test_i4_3g5b_final_g_closure_stress.py
}

probe() {
  python - <<'PY'
from core.reconstruction.cross_provider_qualification_closure import (
    I43G5B_PHASE,
    build_i4_3g5b_final_closure,
)
assert I43G5B_PHASE == "I4.3G5B"
assert callable(build_i4_3g5b_final_closure)
print("I4.3G5B formal G-closure contract/import probe: PASS")
PY
  local rc=$?
  echo "import_probe_rc=$rc"
  [ "$rc" -eq 0 ] || return "$rc"
  python scripts/run_i4_3g5b_final_g_closure.py --help >/dev/null
  rc=$?
  echo "closure_runner_help_rc=$rc"
  return "$rc"
}

regression() {
  run_pytest \
    tests/unit/test_i4_3g5a_cross_provider_evidence_synthesis.py \
    tests/integration/test_i4_3g5a_cross_provider_synthesis_contract.py \
    tests/stress/test_i4_3g5a_cross_provider_synthesis_stress.py \
    tests/unit/test_i4_3g1_cross_provider_qualification.py \
    tests/integration/test_i4_3g1_qualification_framework_contract.py \
    tests/stress/test_i4_3g1_qualification_framework_stress.py \
    tests/unit/test_i4_3g3c_temporal_matched_view_review.py \
    tests/integration/test_i4_3g3c_temporal_review_contract.py \
    tests/stress/test_i4_3g3c_temporal_review_stress.py \
    tests/unit/test_i4_3g4c_single_image_matched_view_review.py \
    tests/integration/test_i4_3g4c_single_image_review_contract.py \
    tests/stress/test_i4_3g4c_single_image_review_stress.py
}

case "$MODE" in
  focused)
    focused
    ;;
  regression)
    regression
    ;;
  all)
    focused || exit $?
    probe || exit $?
    regression || exit $?
    ;;
  *)
    echo "Usage: $0 [focused|regression|all]" >&2
    exit 64
    ;;
esac

rc=$?
echo "I4.3G5B_TEST_RC=$rc"
exit "$rc"

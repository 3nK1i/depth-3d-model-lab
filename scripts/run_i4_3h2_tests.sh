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
    tests/unit/test_i4_3h2_final_closure.py \
    tests/integration/test_i4_3h2_final_closure_contract.py \
    tests/stress/test_i4_3h2_final_closure_stress.py
}

probe() {
  python - <<'PY'
from core.reconstruction.i4_3_final_closure import (
    I43H2_PHASE,
    build_i4_3h2_final_closure,
    validate_i4_3h1_qualification_audit,
)
assert I43H2_PHASE == "I4.3H2"
assert callable(build_i4_3h2_final_closure)
assert callable(validate_i4_3h1_qualification_audit)
print("I4.3H2 formal I4.3-closure contract/import probe: PASS")
PY
  local rc=$?
  echo "import_probe_rc=$rc"
  [ "$rc" -eq 0 ] || return "$rc"
  python scripts/run_i4_3h2_final_closure.py --help >/dev/null
  rc=$?
  echo "closure_runner_help_rc=$rc"
  return "$rc"
}

affected_regression() {
  run_pytest \
    tests/unit/test_i4_3h1_final_qualification.py \
    tests/integration/test_i4_3h1_final_qualification_contract.py \
    tests/stress/test_i4_3h1_final_qualification_stress.py \
    tests/unit/test_i4_3g5b_cross_provider_qualification_closure.py \
    tests/integration/test_i4_3g5b_final_g_closure_contract.py \
    tests/stress/test_i4_3g5b_final_g_closure_stress.py
}

case "$MODE" in
  focused) focused ;;
  probe) probe ;;
  regression) affected_regression ;;
  all)
    focused || exit $?
    probe || exit $?
    affected_regression || exit $?
    ;;
  *)
    echo "usage: $0 [focused|probe|regression|all]" >&2
    exit 2
    ;;
esac

rc=$?
echo "I4.3H2_TEST_RC=$rc"
exit "$rc"

#!/usr/bin/env bash
set -u
set -o pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODE="${1:-all}"

run_core() {
  pytest -q \
    tests/unit/test_i4_3g1_cross_provider_qualification.py \
    tests/integration/test_i4_3g1_qualification_framework_contract.py \
    tests/stress/test_i4_3g1_qualification_framework_stress.py || return $?
  python - <<'PY'
from core.reconstruction.cross_provider_qualification import (
    qualification_metric_registry,
    qualification_scenario_matrix,
)
assert len(qualification_scenario_matrix()) == 11
assert len(qualification_metric_registry()) >= 20
print("I4.3G1 controlled-qualification framework/import probe: PASS")
PY
}

run_affected() {
  pytest -q \
    tests/unit/test_i4_3f3_adaptive_selection_closure.py \
    tests/integration/test_i4_3f3_production_validation_contract.py \
    tests/stress/test_i4_3f3_production_validation_stress.py \
    tests/unit/test_i4_3a_dense_provider_architecture.py \
    tests/integration/test_i4_3a_dense_provider_architecture_contract.py
}

case "$MODE" in
  core) run_core ;;
  affected) run_affected ;;
  all)
    run_core || exit $?
    run_affected || exit $?
    ;;
  *) echo "usage: $0 [core|affected|all]" >&2; exit 2 ;;
esac

#!/usr/bin/env bash
set -u
set -o pipefail
MODE="${1:-all}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

run_focused() {
  python -m pytest -q \
    tests/unit/test_i4_3g2a_mvs_da3_qualification.py \
    tests/integration/test_i4_3g2a_source_binding_contract.py \
    tests/stress/test_i4_3g2a_source_binding_stress.py || return $?
  python - <<'PY'
from core.reconstruction.mvs_da3_qualification import validate_i4_3g1_for_g2
from tests.i4_3g2a_helpers import passing_i4_3g1_snapshot
validate_i4_3g1_for_g2(passing_i4_3g1_snapshot())
print("I4.3G2A source-binding contract/import probe: PASS")
PY
}

case "$MODE" in
  focused) run_focused ;;
  backward) bash scripts/run_i4_3g1_tests.sh all ;;
  all)
    run_focused || exit $?
    bash scripts/run_i4_3g1_tests.sh all || exit $?
    ;;
  *) echo "usage: $0 [focused|backward|all]" >&2; exit 2 ;;
esac

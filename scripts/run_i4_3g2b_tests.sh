#!/usr/bin/env bash
set -u
set -o pipefail
MODE="${1:-all}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

run_focused() {
  python -m pytest -q \
    tests/unit/test_i4_3g2b_mvs_da3_production_comparison.py \
    tests/integration/test_i4_3g2b_production_comparison_contract.py \
    tests/stress/test_i4_3g2b_production_comparison_stress.py || return $?
  python - <<'PY'
from core.reconstruction.mvs_da3_production_comparison import (
    I43G2B_FRESH_SCENARIOS,
    I43G2B_HISTORICAL_SCENARIO,
)
assert len(I43G2B_FRESH_SCENARIOS) == 4
assert I43G2B_HISTORICAL_SCENARIO == "g2_house_mvs_da3_historical"
print("I4.3G2B production comparison contract/import probe: PASS")
PY
  python scripts/run_i4_3g2b_mvs_da3_scenario.py --help >/dev/null || return $?
}

case "$MODE" in
  focused) run_focused ;;
  backward) bash scripts/run_i4_3g2a3_tests.sh all ;;
  all)
    run_focused || exit $?
    bash scripts/run_i4_3g2a3_tests.sh all || exit $?
    ;;
  *) echo "usage: $0 [focused|backward|all]" >&2; exit 2 ;;
esac

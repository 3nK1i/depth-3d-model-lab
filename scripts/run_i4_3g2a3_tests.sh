#!/usr/bin/env bash
set -u
set -o pipefail
MODE="${1:-all}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

run_focused() {
  python -m pytest -q \
    tests/unit/test_i4_3g2a3_fresh_scenario_binding.py \
    tests/integration/test_i4_3g2a3_fresh_binding_contract.py \
    tests/stress/test_i4_3g2a3_fresh_binding_stress.py || return $?
  python - <<'PY'
from core.reconstruction.mvs_da3_fresh_scenario_binding import (
    FreshBindingPolicy,
    LOW_TEXTURE_AMENDMENT_ID,
    LOW_TEXTURE_METHOD,
)
p = FreshBindingPolicy()
assert p.weak_overlap_minimum_stride == 2
assert p.low_texture_max_median_ratio_to_reference == 0.75
assert p.low_texture_max_component_ratio_to_reference == 0.75
assert LOW_TEXTURE_AMENDMENT_ID == "I4.3G2A.3-A1"
assert LOW_TEXTURE_METHOD == "raw_component_ratio_gate_v2"
print("I4.3G2A.3-A1 fresh-scenario binding contract/import probe: PASS")
PY
}

case "$MODE" in
  focused) run_focused ;;
  backward) bash scripts/run_i4_3g2a_tests.sh all ;;
  all)
    run_focused || exit $?
    bash scripts/run_i4_3g2a_tests.sh all || exit $?
    ;;
  *) echo "usage: $0 [focused|backward|all]" >&2; exit 2 ;;
esac

#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODE="${1:-all}"

focused() {
  python -m pytest -q \
    tests/unit/test_i4_3f1_adaptive_view_selection.py \
    tests/integration/test_i4_3f1_adaptive_view_selection_contract.py \
    tests/stress/test_i4_3f1_adaptive_view_selection_stress.py
}

backward() {
  python -m pytest -q \
    tests/integration/test_i4_3a_dense_provider_architecture_contract.py \
    tests/unit/test_i4_3e6_marigold_dense_provider.py
}

probe() {
  python - <<'PY'
from core.reconstruction.adaptive_view_selection import (
    i4_3f1_architecture_snapshot,
    provider_view_selection_requirements,
)
from core.reconstruction.dense_provider_architecture import DenseProviderImplementation
snapshot = i4_3f1_architecture_snapshot()
assert snapshot["pass"] is True
assert snapshot["selection"]["permanent_global_view_cap"] is None
assert len(snapshot["provider_requirements"]) == len(DenseProviderImplementation)
for implementation in DenseProviderImplementation:
    provider_view_selection_requirements(implementation)
print("I4.3F1 adaptive-selection contract/import probe: PASS")
PY
}

case "$MODE" in
  focused) focused ;;
  backward) backward ;;
  probe) probe ;;
  all)
    focused || exit $?
    backward || exit $?
    probe
    ;;
  *)
    echo "Usage: $0 [focused|backward|probe|all]" >&2
    exit 2
    ;;
esac

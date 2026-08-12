#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODE="${1:-quick}"

case "$MODE" in
  quick)
    ./scripts/run_phase8_8h5_tests.sh quick
    python -m compileall -q \
      scripts/run_phase8_8h6_closure.py \
      tests/unit/test_phase8_8h6_closure_contract.py
    bash -n scripts/run_phase8_8h6_tests.sh
    python -m pytest -q tests/unit/test_phase8_8h6_closure_contract.py
    ;;
  closure)
    "$0" quick
    ./scripts/run_phase8_8h5_tests.sh supporting
    python scripts/run_phase8_8h6_closure.py
    ;;
  *)
    echo "usage: $0 {quick|closure}" >&2
    exit 2
    ;;
esac

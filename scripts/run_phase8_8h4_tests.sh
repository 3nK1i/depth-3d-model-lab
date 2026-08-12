#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODE="${1:-quick}"

case "$MODE" in
  quick)
    python -m compileall -q \
      scripts/run_phase8_8h4_full_regression.py \
      tests/unit/test_phase8_8h4_full_regression_contract.py
    bash -n scripts/run_phase8_8h4_tests.sh
    python -m pytest -q tests/unit/test_phase8_8h4_full_regression_contract.py
    ;;
  full)
    "$0" quick
    rm -rf /tmp/phase8_8h4_full_regression
    python scripts/run_phase8_8h4_full_regression.py
    ;;
  *)
    echo "usage: $0 {quick|full}" >&2
    exit 2
    ;;
esac

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODE="${1:-quick}"

case "$MODE" in
  quick)
    python -m compileall -q \
      scripts/run_phase8_9g2_full_regression.py \
      tests/unit/test_phase8_9g2_full_regression.py \
      tests/integration/test_phase8_9g2_full_regression_contract.py
    bash -n scripts/run_phase8_9g2_tests.sh
    python -m pytest -q \
      tests/unit/test_phase8_9g2_full_regression.py \
      tests/integration/test_phase8_9g2_full_regression_contract.py
    ;;
  affected)
    "$0" quick
    python -m pytest -q \
      tests/unit/test_phase8_9g1_regression_inventory.py \
      tests/integration/test_phase8_9g1_regression_contract.py \
      tests/unit/test_phase8_8h4_full_regression_contract.py \
      tests/unit/test_release_metadata.py \
      tests/unit/test_phase6_release_metadata.py \
      tests/unit/test_phase7_release_metadata.py \
      tests/unit/test_phase8_9f6_actual_application_acceptance.py \
      tests/integration/test_phase8_9f6_actual_application_contract.py
    ;;
  *)
    echo "Usage: $0 {quick|affected}" >&2
    exit 2
    ;;
esac

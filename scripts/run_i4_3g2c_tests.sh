#!/usr/bin/env bash
set -u
MODE="${1:-focused}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

case "$MODE" in
  unit)
    python -m pytest -q \
      tests/unit/test_i4_3g2c_matched_view_review.py \
      tests/unit/test_i4_3g2c_remaining_scenario_review.py
    ;;
  integration)
    python -m pytest -q \
      tests/integration/test_i4_3g2c_entry_contract.py \
      tests/integration/test_i4_3g2c_remaining_scenario_contract.py
    ;;
  stress)
    python -m pytest -q \
      tests/stress/test_i4_3g2c_review_stress.py \
      tests/stress/test_i4_3g2c_remaining_scenario_stress.py
    ;;
  focused)
    python -m pytest -q \
      tests/unit/test_i4_3g2c_matched_view_review.py \
      tests/unit/test_i4_3g2c_remaining_scenario_review.py \
      tests/integration/test_i4_3g2c_entry_contract.py \
      tests/integration/test_i4_3g2c_remaining_scenario_contract.py \
      tests/stress/test_i4_3g2c_review_stress.py \
      tests/stress/test_i4_3g2c_remaining_scenario_stress.py \
      tests/unit/test_i4_3g2b_matched_view.py \
      tests/integration/test_i4_3g2b_matched_view_contract.py \
      tests/stress/test_i4_3g2b_matched_view_stress.py
    ;;
  *)
    echo "usage: $0 [unit|integration|stress|focused]" >&2
    exit 2
    ;;
esac

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${1:-all}"
UNIT=(tests/unit/test_i4_3g4a_single_image_provider_qualification.py)
INTEGRATION=(tests/integration/test_i4_3g4a_single_image_preflight_contract.py)
STRESS=(tests/stress/test_i4_3g4a_single_image_preflight_stress.py)
AFFECTED=(
  tests/unit/test_i4_3g1_cross_provider_qualification.py
  tests/unit/test_i4_3g3b_temporal_production_qualification.py
  tests/unit/test_i4_3g3c_temporal_matched_view_review.py
)

case "$MODE" in
  unit) python -m pytest -q "${UNIT[@]}" ;;
  integration) python -m pytest -q "${INTEGRATION[@]}" ;;
  stress) python -m pytest -q "${STRESS[@]}" ;;
  regression) python -m pytest -q "${AFFECTED[@]}" ;;
  all)
    python -m pytest -q "${UNIT[@]}" "${INTEGRATION[@]}" "${STRESS[@]}" "${AFFECTED[@]}"
    ;;
  *)
    echo "usage: $0 [unit|integration|stress|regression|all]" >&2
    exit 2
    ;;
esac

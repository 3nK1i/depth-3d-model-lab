#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

focused=(
  tests/unit/test_i4_2e_2a_native_visibility_coverage_contract.py
  tests/integration/test_i4_2e_2a_native_visibility_coverage_contract.py
)
stress=(
  tests/stress/test_i4_2e_2a_native_visibility_coverage_stress.py
)

case "${1:-all}" in
  focused) python -m pytest -q "${focused[@]}" ;;
  stress) python -m pytest -q "${stress[@]}" ;;
  all) python -m pytest -q "${focused[@]}" "${stress[@]}" ;;
  *) echo "usage: $0 [focused|stress|all]" >&2; exit 2 ;;
esac

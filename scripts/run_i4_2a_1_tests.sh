#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mode="${1:-all}"

focused_tests=(
  tests/unit/test_i4_2a_temple_ring_preflight.py
  tests/integration/test_i4_2a_temple_ring_preflight_contract.py
  tests/unit/test_i4_2a_1_middlebury_header.py
  tests/integration/test_i4_2a_1_middlebury_header_contract.py
)
stress_tests=(
  tests/stress/test_i4_2a_temple_ring_preflight_stress.py
  tests/stress/test_i4_2a_1_middlebury_header_stress.py
)

case "$mode" in
  focused)
    python -m pytest -q "${focused_tests[@]}"
    ;;
  stress)
    python -m pytest -q "${stress_tests[@]}"
    ;;
  all)
    python -m pytest -q "${focused_tests[@]}" "${stress_tests[@]}"
    ;;
  *)
    echo "Usage: $0 [focused|stress|all]" >&2
    exit 2
    ;;
esac

python -m py_compile \
  core/reconstruction/temple_ring_preflight.py \
  scripts/run_i4_2a_temple_ring_preflight.py

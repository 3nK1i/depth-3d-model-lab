#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-quick}"

case "$MODE" in
  quick)
    python -m pytest -q \
      tests/unit/test_phase8_8a2_reconstruction_controller.py \
      tests/unit/test_phase8_8a1_reconstruction_controller.py \
      tests/ui/test_phase8_8a1_reconstruction_controller.py \
      tests/unit/test_phase8_4c_backward_compatibility.py \
      tests/unit/test_phase8_4d_backward_compatibility.py
    ;;
  integration)
    python -m pytest -q \
      tests/integration/test_phase8_8a2_live_controller_pipeline.py
    ;;
  backward)
    ./scripts/run_phase8_4d_tests.sh quick
    ./scripts/run_phase8_8a1_tests.sh quick
    ;;
  full)
    for group in tests/unit tests/ui tests/integration tests/stress; do
      echo "==> pytest group: $group"
      python -m pytest -q "$group"
    done
    ;;
  *)
    echo "Usage: $0 {quick|integration|backward|full}" >&2
    exit 2
    ;;
esac

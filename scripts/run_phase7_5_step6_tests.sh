#!/usr/bin/env bash
set -euo pipefail
mode="${1:-}"
case "$mode" in
  quick)
    python -m pytest -q \
      tests/unit/test_measurement_geometry.py \
      tests/unit/test_measurement_uncertainty_pipeline.py \
      tests/unit/test_phase75_step6_backward_compatibility.py
    ;;
  integration)
    python -m pytest -q tests/integration/test_fitting_measurement_computation.py
    ;;
  full)
    for group in tests/unit tests/ui tests/integration tests/stress; do
      echo "==> pytest group: $group"
      python -m pytest -q "$group"
    done
    ;;
  *)
    echo "Usage: $0 {quick|integration|full}"
    exit 2
    ;;
esac

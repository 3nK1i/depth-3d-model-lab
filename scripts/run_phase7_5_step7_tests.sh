#!/usr/bin/env bash
set -euo pipefail
mode="${1:-}"
case "$mode" in
  quick)
    python -m pytest -q \
      tests/unit/test_tracking_contracts.py \
      tests/unit/test_tracking_association.py \
      tests/unit/test_tracking_stabilization.py \
      tests/unit/test_tracking_pipeline.py \
      tests/unit/test_phase75_step7_backward_compatibility.py
    ;;
  integration)
    python -m pytest -q tests/integration/test_temporal_tracking_measurement_stabilization.py
    ;;
  full)
    for group in tests/unit tests/ui tests/integration tests/stress; do
      echo "==> pytest group: $group"
      python -m pytest -q "$group"
    done
    ;;
  *) echo "Usage: $0 {quick|integration|full}"; exit 2;;
esac

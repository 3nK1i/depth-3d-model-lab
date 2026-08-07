#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-quick}"

case "$MODE" in
  quick)
    python -m pytest -q \
      tests/unit/test_phase8_4c_backward_compatibility.py \
      tests/unit/test_reconstruction_live_capture_runtime.py \
      tests/unit/test_reconstruction_live_source_metadata.py \
      tests/unit/test_reconstruction_live_calibration.py \
      tests/unit/test_capture_live_source_identity.py
    ;;
  integration)
    python -m pytest -q \
      tests/integration/test_reconstruction_live_capture_pipeline.py
    ;;
  stress)
    python -m pytest -q \
      tests/stress/test_reconstruction_live_capture_backpressure_stress.py
    ;;
  full)
    for group in tests/unit tests/ui tests/integration tests/stress; do
      echo "==> pytest group: $group"
      python -m pytest -q "$group"
    done
    ;;
  *)
    echo "Usage: $0 {quick|integration|stress|full}" >&2
    exit 2
    ;;
esac

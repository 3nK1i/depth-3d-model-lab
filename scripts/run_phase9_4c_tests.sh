#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

mode="${1:-quick}"
case "$mode" in
  quick)
    pytest -q tests/unit/test_phase9_4c_gaussian_optimization_execution.py
    ;;
  integration)
    pytest -q tests/integration/test_phase9_4c_gaussian_optimization_execution.py
    python scripts/run_phase9_4c_gaussian_optimization_smoke.py
    ;;
  backward)
    pytest -q \
      tests/unit/test_phase9_4a_gaussian_runtime_isolation.py \
      tests/unit/test_phase9_4b_gaussian_input_bridge.py \
      tests/unit/test_reconstruction_gaussian_splatting.py \
      tests/integration/test_phase9_4a_gaussian_runtime_isolation.py \
      tests/integration/test_phase9_4b_gaussian_input_bridge.py \
      tests/integration/test_reconstruction_gaussian_splatting_pipeline.py
    ;;
  *)
    echo "usage: $0 {quick|integration|backward}" >&2
    exit 2
    ;;
esac

#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODE="${1:-quick}"
case "$MODE" in
  quick)
    pytest -q tests/unit/test_phase9_5c_nerf_optimization_execution.py
    ;;
  integration)
    pytest -q tests/integration/test_phase9_5c_nerf_optimization_execution.py
    python scripts/run_phase9_5c_nerf_optimization_smoke.py
    ;;
  backward)
    pytest -q \
      tests/unit/test_phase9_5a_nerf_runtime_isolation.py \
      tests/integration/test_phase9_5a_nerf_runtime_isolation.py \
      tests/unit/test_phase9_5b_nerf_input_bridge.py \
      tests/integration/test_phase9_5b_nerf_input_bridge.py \
      tests/unit/test_reconstruction_nerf.py \
      tests/unit/test_reconstruction_nerf_e2.py \
      tests/integration/test_reconstruction_nerf_e2_qualification.py
    ;;
  *)
    echo "Usage: $0 {quick|integration|backward}" >&2
    exit 2
    ;;
esac

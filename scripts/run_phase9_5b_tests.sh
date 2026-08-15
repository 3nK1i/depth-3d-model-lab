#!/usr/bin/env bash
set -euo pipefail
mode="${1:-quick}"
case "$mode" in
  quick)
    pytest -q tests/unit/test_phase9_5b_nerf_input_bridge.py
    ;;
  integration)
    pytest -q tests/integration/test_phase9_5b_nerf_input_bridge.py
    python scripts/run_phase9_5b_nerf_input_smoke.py
    ;;
  backward)
    pytest -q \
      tests/unit/test_phase9_5a_nerf_runtime_isolation.py \
      tests/integration/test_phase9_5a_nerf_runtime_isolation.py \
      tests/unit/test_reconstruction_nerf.py \
      tests/unit/test_reconstruction_nerf_e2.py \
      tests/integration/test_reconstruction_nerf_external_pipeline.py
    ;;
  *) echo "usage: $0 {quick|integration|backward}" >&2; exit 2;;
esac

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODE="${1:-quick}"

case "$MODE" in
  quick)
    pytest -q tests/unit/test_phase9_4b_gaussian_input_bridge.py
    ;;
  integration)
    pytest -q tests/integration/test_phase9_4b_gaussian_input_bridge.py
    python scripts/run_phase9_4b_gaussian_input_smoke.py
    ;;
  backward)
    pytest -q \
      tests/unit/test_phase9_4a_gaussian_runtime_isolation.py \
      tests/integration/test_phase9_4a_gaussian_runtime_isolation.py \
      tests/unit/test_reconstruction_gaussian_splatting.py \
      tests/integration/test_reconstruction_gaussian_splatting_pipeline.py \
      tests/unit/test_phase9_3c_canonical_camera_presentation.py
    ;;
  *)
    echo "Usage: $0 {quick|integration|backward}" >&2
    exit 2
    ;;
esac

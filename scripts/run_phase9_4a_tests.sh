#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-quick}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

case "$MODE" in
  quick)
    pytest -q \
      tests/unit/test_phase9_4a_gaussian_runtime_isolation.py \
      tests/integration/test_phase9_4a_gaussian_runtime_isolation.py
    ;;
  integration)
    pytest -q tests/integration/test_phase9_4a_gaussian_runtime_isolation.py
    python scripts/run_phase9_4a_gaussian_runtime_smoke.py
    ;;
  backward)
    pytest -q \
      tests/unit/test_reconstruction_gaussian_splatting.py \
      tests/unit/test_reconstruction_gaussian_backends.py \
      tests/integration/test_reconstruction_gaussian_splatting_pipeline.py \
      tests/integration/test_reconstruction_gaussian_backend_qualification.py \
      tests/unit/test_phase8_8f2_reconstruction_gaussian_visualization.py
    ;;
  *)
    echo "Usage: $0 {quick|integration|backward}" >&2
    exit 2
    ;;
esac

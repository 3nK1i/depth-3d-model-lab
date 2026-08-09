#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mode="${1:-quick}"
case "$mode" in
  quick)
    pytest -q \
      tests/unit/test_reconstruction_pipeline_geometry_f3.py \
      tests/unit/test_phase8_7f3_backward_compatibility.py
    ;;
  integration)
    pytest -q tests/integration/test_reconstruction_pipeline_geometry_f3.py
    ;;
  stress)
    pytest -q tests/stress/test_reconstruction_pipeline_geometry_f3_stress.py
    ;;
  *)
    echo "usage: $0 {quick|integration|stress}" >&2
    exit 2
    ;;
esac

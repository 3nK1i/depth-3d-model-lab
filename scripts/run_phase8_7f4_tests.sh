#!/usr/bin/env bash
set -euo pipefail
mode="${1:-quick}"
cd "$(dirname "$0")/.."
case "$mode" in
  quick)
    pytest -q tests/unit/test_reconstruction_pipeline_appearance_f4.py tests/unit/test_phase8_7f4_backward_compatibility.py
    ;;
  integration)
    pytest -q tests/integration/test_reconstruction_pipeline_appearance_f4.py
    ;;
  stress)
    pytest -q tests/stress/test_reconstruction_pipeline_appearance_f4_stress.py
    ;;
  *) echo "usage: $0 {quick|integration|stress}" >&2; exit 2 ;;
esac

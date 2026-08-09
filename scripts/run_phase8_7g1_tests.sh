#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mode="${1:-quick}"
case "$mode" in
  quick)
    python -m pytest -q \
      tests/unit/test_reconstruction_engine_hardening_g1.py \
      tests/unit/test_phase8_7g1_backward_compatibility.py
    ;;
  integration)
    python -m pytest -q tests/integration/test_reconstruction_engine_hardening_g1.py
    ;;
  stress)
    python -m pytest -q tests/stress/test_reconstruction_engine_hardening_g1_stress.py
    ;;
  *)
    echo "usage: $0 {quick|integration|stress}" >&2
    exit 2
    ;;
esac

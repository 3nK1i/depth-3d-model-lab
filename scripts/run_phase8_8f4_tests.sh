#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mode="${1:-quick}"
case "$mode" in
  quick)
    ./scripts/run_phase8_8f3_tests.sh quick
    python -m pytest -q \
      tests/unit/test_phase8_8f4_contract.py \
      tests/unit/test_phase8_8f4_reconstruction_representation_switching.py \
      tests/ui/test_phase8_8f4_representation_switching.py
    ;;
  protocol)
    ./scripts/run_phase8_8f3_tests.sh protocol
    python -m pytest -q tests/unit/test_phase8_8f4_contract.py
    ;;
  backward)
    ./scripts/run_phase8_8f3_tests.sh backward
    python -m pytest -q \
      tests/ui/test_phase8_8f2_gaussian_splatting_visualization.py \
      tests/ui/test_phase8_8f3_nerf_visualization.py
    ;;
  *)
    echo "usage: $0 {quick|protocol|backward}" >&2
    exit 2
    ;;
esac

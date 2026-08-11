#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mode="${1:-quick}"
case "$mode" in
  quick)
    ./scripts/run_phase8_8f2_tests.sh quick
    python -m pytest -q \
      tests/unit/test_phase8_8f3_contract.py \
      tests/unit/test_phase8_8f3_reconstruction_nerf_visualization.py \
      tests/ui/test_phase8_8f3_nerf_visualization.py
    ;;
  protocol)
    ./scripts/run_phase8_8f2_tests.sh protocol
    python -m pytest -q tests/unit/test_phase8_8f3_contract.py
    ;;
  backward)
    ./scripts/run_phase8_8f2_tests.sh backward
    python -m pytest -q \
      tests/unit/test_reconstruction_nerf.py \
      tests/unit/test_reconstruction_nerf_e2.py \
      tests/unit/test_reconstruction_nerf_e3.py \
      tests/unit/test_reconstruction_nerf_e4.py \
      tests/unit/test_reconstruction_persistent_scene_f5.py
    ;;
  *)
    echo "usage: $0 {quick|protocol|backward}" >&2
    exit 2
    ;;
esac

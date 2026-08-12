#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${1:-quick}"
case "$MODE" in
  quick)
    pytest -q \
      tests/unit/test_phase8_9f1_external_backend_preflight.py \
      tests/integration/test_phase8_9f1_external_backend_contract.py
    ;;
  affected)
    pytest -q \
      tests/unit/test_phase8_9f1_external_backend_preflight.py \
      tests/integration/test_phase8_9f1_external_backend_contract.py \
      tests/unit/test_reconstruction_da3.py \
      tests/unit/test_reconstruction_da3_provider.py \
      tests/unit/test_reconstruction_gaussian_backends.py \
      tests/unit/test_reconstruction_nerf.py \
      tests/unit/test_reconstruction_persistent_scene_f5.py \
      tests/unit/test_phase8_8f2_reconstruction_gaussian_visualization.py \
      tests/unit/test_phase8_8f3_reconstruction_nerf_visualization.py
    ;;
  *)
    echo "usage: $0 [quick|affected]" >&2
    exit 2
    ;;
esac

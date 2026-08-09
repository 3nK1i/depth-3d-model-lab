#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODE="${1:-quick}"
case "$MODE" in
  quick)
    pytest -q \
      tests/unit/test_reconstruction_nerf.py \
      tests/unit/test_phase8_7e1_backward_compatibility.py \
      tests/unit/test_reconstruction_nerf_e2.py \
      tests/unit/test_phase8_7e2_backward_compatibility.py \
      tests/unit/test_reconstruction_nerf_e3.py \
      tests/unit/test_phase8_7e3_backward_compatibility.py
    ;;
  integration)
    pytest -q \
      tests/integration/test_reconstruction_nerf_external_pipeline.py \
      tests/integration/test_reconstruction_nerf_e2_qualification.py \
      tests/integration/test_reconstruction_nerf_e3_real_scene_validation.py
    ;;
  stress)
    pytest -q \
      tests/stress/test_reconstruction_nerf_contracts_stress.py \
      tests/stress/test_reconstruction_nerf_e2_stress.py \
      tests/stress/test_reconstruction_nerf_e3_stress.py
    ;;
  full)
    for group in tests/unit tests/ui tests/integration tests/stress; do
      echo "==> pytest group: $group"
      pytest -q "$group"
    done
    ;;
  *)
    echo "Usage: $0 {quick|integration|stress|full}" >&2
    exit 2
    ;;
esac

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mode="${1:-quick}"
case "$mode" in
  quick)
    pytest -q tests/unit/test_phase9_4d_gaussian_persistence.py
    ;;
  integration)
    pytest -q tests/integration/test_phase9_4d_persistent_gaussian_representation.py
    python scripts/run_phase9_4d_gaussian_persistence_smoke.py
    ;;
  backward)
    pytest -q \
      tests/unit/test_reconstruction_persistent_scene_f5.py \
      tests/unit/test_phase8_9f5_persistent_multi_artifact_scene.py \
      tests/unit/test_phase8_8f2_reconstruction_gaussian_visualization.py \
      tests/unit/test_phase9_4a_gaussian_runtime_isolation.py \
      tests/unit/test_phase9_4b_gaussian_input_bridge.py \
      tests/unit/test_phase9_4c_gaussian_optimization_execution.py \
      tests/integration/test_phase8_9f5_persistent_real_scene_contract.py
    ;;
  *)
    echo "usage: $0 {quick|integration|backward}" >&2
    exit 2
    ;;
esac

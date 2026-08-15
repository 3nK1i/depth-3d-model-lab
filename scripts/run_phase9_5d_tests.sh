#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mode="${1:-quick}"
case "$mode" in
  quick)
    pytest -q tests/unit/test_phase9_5d_nerf_persistence.py
    ;;
  integration)
    pytest -q tests/integration/test_phase9_5d_persistent_nerf_representation.py
    python scripts/run_phase9_5d_nerf_persistence_smoke.py
    ;;
  backward)
    pytest -q \
      tests/unit/test_reconstruction_persistent_scene_f5.py \
      tests/unit/test_phase8_9f5_persistent_multi_artifact_scene.py \
      tests/unit/test_phase8_8f3_reconstruction_nerf_visualization.py \
      tests/unit/test_phase9_5a_nerf_runtime_isolation.py \
      tests/unit/test_phase9_5b_nerf_input_bridge.py \
      tests/unit/test_phase9_5c_nerf_optimization_execution.py \
      tests/integration/test_phase8_9f5_persistent_real_scene_contract.py
    ;;
  *)
    echo "usage: $0 {quick|integration|backward}" >&2
    exit 2
    ;;
esac

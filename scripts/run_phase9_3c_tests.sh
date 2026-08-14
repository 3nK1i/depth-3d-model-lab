#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODE="${1:-quick}"

case "$MODE" in
  quick)
    pytest -q \
      tests/unit/test_phase9_3c_canonical_camera_presentation.py \
      tests/ui/test_phase9_3c_world_camera_viewer.py
    ;;
  integration)
    pytest -q tests/integration/test_phase9_3c_persistent_camera_scene.py
    python scripts/run_phase9_3c_world_camera_smoke.py
    ;;
  backward)
    pytest -q \
      tests/unit/test_phase9_3b_canonical_geometry_loader.py \
      tests/integration/test_phase9_3b_persistent_canonical_scene_viewer.py \
      tests/ui/test_phase9_3b_canonical_geometry_viewer.py \
      tests/unit/test_phase9_3a_canonical_scene_presentation.py \
      tests/unit/test_phase8_8c1_reconstruction_visualization.py \
      tests/unit/test_phase8_8c2_visualizer_contract.py \
      tests/unit/test_phase8_8d2_contract.py
    ;;
  *)
    echo "Usage: $0 {quick|integration|backward}" >&2
    exit 2
    ;;
esac

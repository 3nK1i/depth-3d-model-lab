#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mode="${1:-quick}"
case "$mode" in
  quick)
    pytest -q \
      tests/unit/test_phase9_3f_scene_layers.py \
      tests/ui/test_phase9_3f_scene_layers.py
    ;;
  integration)
    pytest -q tests/integration/test_phase9_3f_scene_layers.py
    python scripts/run_phase9_3f_scene_layers_smoke.py
    ;;
  backward)
    pytest -q \
      tests/ui/test_phase9_3b_canonical_geometry_viewer.py \
      tests/ui/test_phase9_3c_world_camera_viewer.py \
      tests/ui/test_phase9_3d_semantic_canonical_viewer.py \
      tests/ui/test_phase9_3e_canonical_measurement_overlay.py
    ;;
  full)
    "$0" quick
    "$0" integration
    "$0" backward
    ;;
  *)
    echo "Usage: $0 {quick|integration|backward|full}" >&2
    exit 2
    ;;
esac

#!/usr/bin/env bash
set -euo pipefail

mode="${1:-quick}"
case "$mode" in
  quick)
    pytest -q \
      tests/unit/test_phase9_3i_scene_runtime_health.py \
      tests/ui/test_phase9_3i_scene_runtime_hardening.py
    ;;
  integration)
    pytest -q tests/integration/test_phase9_3i_scene_runtime_health.py
    python scripts/run_phase9_3i_scene_health_smoke.py
    ;;
  backward)
    pytest -q \
      tests/unit/test_phase9_3b_canonical_geometry_loader.py \
      tests/ui/test_phase9_3c_world_camera_viewer.py \
      tests/ui/test_phase9_3f_scene_layers.py \
      tests/ui/test_phase9_3h_da3_canonical_integration.py
    ;;
  *)
    echo "usage: $0 {quick|integration|backward}" >&2
    exit 2
    ;;
esac

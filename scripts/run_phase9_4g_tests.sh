#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mode="${1:-quick}"
case "$mode" in
  quick)
    pytest -q \
      tests/unit/test_phase9_4g_gaussian_resource_lifecycle.py \
      tests/ui/test_phase9_4g_gaussian_resource_lifecycle_ui.py
    ;;
  integration)
    pytest -q tests/integration/test_phase9_4g_gaussian_resource_lifecycle.py
    python scripts/run_phase9_4g_gaussian_resource_smoke.py
    ;;
  stress)
    pytest -q tests/stress/test_phase9_4g_gaussian_resource_lifecycle_stress.py
    ;;
  backward)
    pytest -q \
      tests/unit/test_phase9_4f_gaussian_display_controls.py \
      tests/ui/test_phase9_4f_gaussian_controls_layers_ui.py \
      tests/unit/test_phase9_4e_interactive_gaussian_scene.py \
      tests/ui/test_phase9_4e_interactive_gaussian_viewer_ui.py \
      tests/unit/test_pre9_4e_interactive_distance.py \
      tests/ui/test_pre9_4e_interactive_measurement_ui.py \
      tests/unit/test_phase9_3i_scene_runtime_health.py \
      tests/integration/test_phase9_3i_scene_runtime_health.py
    ;;
  *)
    echo "usage: $0 {quick|integration|stress|backward}" >&2
    exit 2
    ;;
esac

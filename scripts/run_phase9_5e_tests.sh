#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mode="${1:-quick}"
case "$mode" in
  quick)
    pytest -q \
      tests/unit/test_phase9_5e_nerf_interactive.py \
      tests/ui/test_phase9_5e_interactive_nerf_viewer_ui.py
    ;;
  integration)
    pytest -q tests/integration/test_phase9_5e_interactive_nerf_viewer.py
    python scripts/run_phase9_5e_nerf_interactive_smoke.py
    ;;
  backward)
    pytest -q \
      tests/unit/test_phase8_8f3_contract.py \
      tests/unit/test_phase8_8f3_reconstruction_nerf_visualization.py \
      tests/ui/test_phase8_8f3_nerf_visualization.py \
      tests/unit/test_phase9_4e_interactive_gaussian_scene.py \
      tests/integration/test_phase9_4e_interactive_gaussian_viewer.py \
      tests/ui/test_phase9_4e_interactive_gaussian_viewer_ui.py \
      tests/unit/test_phase9_4f_gaussian_display_controls.py \
      tests/integration/test_phase9_4f_gaussian_controls_layers.py \
      tests/ui/test_phase9_4f_gaussian_controls_layers_ui.py \
      tests/unit/test_phase9_4g_gaussian_resource_lifecycle.py \
      tests/integration/test_phase9_4g_gaussian_resource_lifecycle.py \
      tests/ui/test_phase9_4g_gaussian_resource_lifecycle_ui.py \
      tests/integration/test_pre9_4e_interactive_measurement_bundle.py \
      tests/ui/test_pre9_4e_interactive_measurement_ui.py
    ;;
  *)
    echo "usage: $0 {quick|integration|backward}" >&2
    exit 2
    ;;
esac

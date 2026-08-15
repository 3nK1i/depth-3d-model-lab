#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mode="${1:-quick}"
case "$mode" in
  quick)
    pytest -q \
      tests/unit/test_phase9_5g_nerf_resource_lifecycle.py \
      tests/ui/test_phase9_5g_nerf_resource_lifecycle_ui.py
    ;;
  integration)
    pytest -q tests/integration/test_phase9_5g_nerf_resource_lifecycle.py
    python scripts/run_phase9_5g_nerf_resource_smoke.py
    ;;
  stress)
    pytest -q tests/stress/test_phase9_5g_nerf_resource_lifecycle_stress.py
    ;;
  backward)
    pytest -q \
      tests/unit/test_phase9_5f_nerf_display_controls.py \
      tests/integration/test_phase9_5f_nerf_controls_layers.py \
      tests/ui/test_phase9_5f_nerf_controls_layers_ui.py \
      tests/unit/test_phase9_5e_nerf_interactive.py \
      tests/integration/test_phase9_5e_interactive_nerf_viewer.py \
      tests/ui/test_phase9_5e_interactive_nerf_viewer_ui.py \
      tests/unit/test_phase8_8f3_contract.py \
      tests/unit/test_pre9_4e_interactive_distance.py \
      tests/ui/test_pre9_4e_interactive_measurement_ui.py
    ;;
  *)
    echo "usage: $0 {quick|integration|stress|backward}" >&2
    exit 2
    ;;
esac

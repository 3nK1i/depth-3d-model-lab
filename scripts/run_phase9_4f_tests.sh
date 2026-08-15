#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mode="${1:-quick}"
case "$mode" in
  quick)
    pytest -q \
      tests/unit/test_phase9_4f_gaussian_display_controls.py \
      tests/ui/test_phase9_4f_gaussian_controls_layers_ui.py
    ;;
  integration)
    pytest -q tests/integration/test_phase9_4f_gaussian_controls_layers.py
    python scripts/run_phase9_4f_gaussian_controls_smoke.py
    ;;
  backward)
    pytest -q \
      tests/unit/test_phase9_4e_interactive_gaussian_scene.py \
      tests/ui/test_phase9_4e_interactive_gaussian_viewer_ui.py \
      tests/unit/test_phase9_3f_scene_layers.py \
      tests/ui/test_phase9_3f_scene_layers.py \
      tests/unit/test_pre9_4e_interactive_distance.py \
      tests/ui/test_pre9_4e_interactive_measurement_ui.py
    ;;
  *)
    echo "usage: $0 {quick|integration|backward}" >&2
    exit 2
    ;;
esac

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mode="${1:-quick}"
case "$mode" in
  quick)
    pytest -q \
      tests/unit/test_phase9_4e_interactive_gaussian_scene.py \
      tests/ui/test_phase9_4e_interactive_gaussian_viewer_ui.py
    ;;
  integration)
    pytest -q tests/integration/test_phase9_4e_interactive_gaussian_viewer.py
    python scripts/run_phase9_4e_interactive_gaussian_smoke.py
    ;;
  backward)
    pytest -q \
      tests/ui/test_phase8_8f2_gaussian_splatting_visualization.py \
      tests/ui/test_phase8_8f4_representation_switching.py \
      tests/unit/test_phase9_4d_gaussian_persistence.py \
      tests/integration/test_phase9_4d_persistent_gaussian_representation.py \
      tests/unit/test_pre9_4e_interactive_distance.py \
      tests/ui/test_pre9_4e_interactive_measurement_ui.py
    ;;
  *)
    echo "usage: $0 {quick|integration|backward}" >&2
    exit 2
    ;;
esac

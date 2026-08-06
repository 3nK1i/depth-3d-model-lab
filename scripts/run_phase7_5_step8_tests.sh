#!/usr/bin/env bash
set -euo pipefail
mode="${1:-}"
case "$mode" in
  quick)
    python -m pytest -q \
      tests/unit/test_inspection_contracts.py \
      tests/unit/test_inspection_bundle.py \
      tests/unit/test_measurement_box_override.py \
      tests/unit/test_moge2_adapter.py \
      tests/unit/test_orientation_overlay.py \
      tests/unit/test_phase75_step8_backward_compatibility.py
    ;;
  integration)
    python -m pytest -q tests/integration/test_measurement_inspection_job.py
    ;;
  ui)
    python -m pytest -q \
      tests/ui/test_inspection_page.py \
      tests/ui/test_inspection_canvas_painting.py \
      tests/ui/test_measurement_inspection_controller.py \
      tests/ui/test_main_window_shell.py \
      tests/ui/test_phase7_mode_switching.py
    ;;
  full)
    for group in tests/unit tests/ui tests/integration tests/stress; do
      echo "==> pytest group: $group"
      python -m pytest -q "$group"
    done
    ;;
  *)
    echo "Usage: $0 {quick|integration|ui|full}"
    exit 2
    ;;
esac

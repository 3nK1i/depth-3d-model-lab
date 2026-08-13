#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mode="${1:-quick}"

case "$mode" in
  quick)
    python -m pytest -q \
      tests/ui/test_phase9_1f_camera_calibration_dialog.py \
      tests/ui/test_phase9_1f_reconstruction_calibration_ui.py \
      tests/ui/test_phase8_8b3_reconstruction_runtime_controls.py
    ;;
  integration)
    python -m pytest -q \
      tests/ui/test_phase9_1f_camera_calibration_dialog.py \
      tests/ui/test_phase9_1f_reconstruction_calibration_ui.py \
      tests/ui/test_phase8_8g6_full_reconstruction_ux_review.py \
      tests/unit/test_phase9_1a_calibration_profile.py \
      tests/unit/test_phase9_1b_calibration_application.py \
      tests/unit/test_phase9_1c_calibration_management.py \
      tests/unit/test_phase9_1d_automatic_calibration.py \
      tests/unit/test_phase9_1e_metric_qualification.py
    ;;
  *)
    echo "Usage: $0 {quick|integration}" >&2
    exit 2
    ;;
esac

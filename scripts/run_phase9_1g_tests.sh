#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${1:-quick}"

case "$MODE" in
  quick)
    python -m pytest -q \
      tests/unit/test_phase9_1g_measurement_calibration.py \
      tests/unit/test_phase9_1g_inspection_calibration_provenance.py \
      tests/unit/test_alignment_intrinsics.py \
      tests/unit/test_inspection_contracts.py \
      tests/ui/test_phase9_1g_measurement_calibration_ui.py \
      tests/ui/test_measurement_inspection_controller.py \
      tests/ui/test_inspection_page.py \
      tests/ui/test_phase9_1f_reconstruction_calibration_ui.py
    ;;
  integration)
    python -m pytest -q \
      tests/integration/test_measurement_inspection_job.py \
      tests/integration/test_reconstruction_camera_geometry_pipeline.py \
      tests/integration/test_phase9_1g_measurement_calibration_pipeline.py
    ;;
  full)
    "$0" quick
    "$0" integration
    python -m pytest -q \
      tests/unit/test_phase9_1a_calibration_profile.py \
      tests/unit/test_phase9_1b_calibration_application.py \
      tests/unit/test_phase9_1c_calibration_management.py \
      tests/unit/test_phase9_1d_automatic_calibration.py \
      tests/unit/test_phase9_1e_metric_qualification.py \
      tests/unit/test_reconstruction_persistent_scene_f5.py
    ;;
  *)
    echo "Usage: $0 {quick|integration|full}" >&2
    exit 2
    ;;
esac

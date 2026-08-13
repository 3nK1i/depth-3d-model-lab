#!/usr/bin/env bash
set -euo pipefail

mode="${1:-quick}"

case "$mode" in
  quick)
    python -m pytest -q \
      tests/unit/test_phase9_1a_calibration_profile.py \
      tests/unit/test_phase9_1b_calibration_application.py \
      tests/unit/test_phase9_1c_calibration_management.py \
      tests/unit/test_reconstruction_live_calibration.py
    ;;
  integration)
    python -m pytest -q \
      tests/integration/test_phase9_1a_calibration_compatibility.py \
      tests/integration/test_phase9_1b_calibration_resolution_compatibility.py \
      tests/integration/test_phase9_1c_calibration_profile_store.py
    ;;
  *)
    echo "usage: $0 {quick|integration}" >&2
    exit 2
    ;;
esac

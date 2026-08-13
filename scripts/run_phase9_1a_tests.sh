#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${1:-quick}"
case "$MODE" in
  quick)
    pytest -q \
      tests/unit/test_phase9_1a_calibration_profile.py \
      tests/unit/test_reconstruction_live_calibration.py \
      tests/unit/test_reconstruction_camera_geometry.py \
      tests/unit/test_alignment_intrinsics.py \
      tests/unit/test_reconstruction_scale_reference.py \
      tests/unit/test_known_size_validation.py
    ;;
  integration)
    pytest -q \
      tests/integration/test_phase9_1a_calibration_compatibility.py \
      tests/integration/test_reconstruction_camera_geometry_pipeline.py \
      tests/integration/test_reconstruction_scale_reference_pipeline.py \
      tests/integration/test_known_size_validation_export.py
    ;;
  full)
    for group in tests/unit tests/ui tests/integration tests/stress; do
      echo "==> pytest group: $group"
      pytest -q "$group"
    done
    ;;
  *)
    echo "Usage: $0 {quick|integration|full}" >&2
    exit 2
    ;;
esac

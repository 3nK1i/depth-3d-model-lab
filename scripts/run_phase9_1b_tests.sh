#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${1:-quick}"
case "$MODE" in
  quick)
    pytest -q \
      tests/unit/test_phase9_1b_calibration_application.py \
      tests/unit/test_phase9_1a_calibration_profile.py \
      tests/unit/test_alignment_intrinsics.py \
      tests/unit/test_alignment_transforms.py \
      tests/unit/test_reconstruction_live_frames.py \
      tests/unit/test_reconstruction_live_contracts.py
    ;;
  integration)
    pytest -q \
      tests/integration/test_phase9_1b_calibration_resolution_compatibility.py \
      tests/integration/test_phase9_1a_calibration_compatibility.py \
      tests/integration/test_reconstruction_camera_geometry_pipeline.py
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

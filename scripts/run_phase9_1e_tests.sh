#!/usr/bin/env bash
set -euo pipefail

mode="${1:-quick}"

case "$mode" in
  quick)
    python -m pytest -q \
      tests/unit/test_phase9_1e_metric_qualification.py \
      tests/unit/test_reconstruction_scale_reference.py \
      tests/unit/test_known_size_validation.py \
      tests/unit/test_reconstruction_camera_geometry.py \
      tests/unit/test_measurement_uncertainty_pipeline.py
    ;;
  integration)
    python -m pytest -q \
      tests/integration/test_phase9_1e_metric_qualification_pipeline.py \
      tests/integration/test_known_size_validation_export.py \
      tests/integration/test_reconstruction_scale_reference_pipeline.py \
      tests/integration/test_reconstruction_camera_geometry_pipeline.py
    ;;
  *)
    echo "usage: $0 {quick|integration}" >&2
    exit 2
    ;;
esac

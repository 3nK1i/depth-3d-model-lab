#!/usr/bin/env bash
set -euo pipefail

mode="${1:-quick}"
case "$mode" in
  quick)
    python -m pytest -q \
      tests/unit/test_phase9_6f_digital_twin_persistence.py \
      tests/ui/test_phase9_6f_digital_twin_persistence_ui.py
    ;;
  integration)
    python -m pytest -q tests/integration/test_phase9_6f_digital_twin_persistence.py
    ;;
  backward)
    python -m pytest -q \
      tests/unit/test_phase9_6e0_estimated_metric_measurement.py \
      tests/integration/test_phase9_6e0_estimated_metric_measurement.py \
      tests/ui/test_phase9_6e0_estimated_metric_ruler_ui.py \
      tests/unit/test_phase9_6e1_processing_jobs.py \
      tests/ui/test_phase9_6e1_processing_job_ui.py \
      tests/unit/test_phase9_5d_nerf_persistence.py \
      tests/integration/test_phase9_5d_persistent_nerf_representation.py \
      tests/unit/test_phase9_4d_gaussian_persistence.py \
      tests/integration/test_phase9_4d_persistent_gaussian_representation.py \
      tests/unit/test_phase8_9c_persistence_reproducibility.py \
      tests/integration/test_phase8_9c_persistence_reproducibility.py \
      tests/ui/test_phase8_9c_persistence_reload_qualification.py \
      tests/unit/test_phase8_9a2_reconstruction_export_ui_contract.py \
      tests/integration/test_phase8_9a1_reconstruction_export.py
    ;;
  full)
    "$0" quick
    "$0" integration
    "$0" backward
    ;;
  *)
    echo "Usage: $0 {quick|integration|backward|full}" >&2
    exit 2
    ;;
esac

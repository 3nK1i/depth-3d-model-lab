#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mode="${1:-quick}"
case "$mode" in
  quick)
    python -m pytest -q \
      tests/unit/test_phase9_6e2_da3_full_scene.py \
      tests/integration/test_phase9_6e2_da3_full_scene_contract.py \
      tests/ui/test_phase9_6e2_da3_full_scene_ui.py
    ;;
  integration)
    python -m pytest -q \
      tests/integration/test_phase9_6e2_da3_full_scene_contract.py \
      tests/integration/test_phase9_6e2_da3_full_scene_worker.py
    ;;
  backward)
    python -m pytest -q \
      tests/unit/test_phase9_6e1_processing_jobs.py \
      tests/integration/test_phase9_6e1_processing_job_contract.py \
      tests/integration/test_phase9_6e1_processing_job_supervisor.py \
      tests/ui/test_phase9_6e1_processing_job_ui.py \
      tests/unit/test_phase9_3h_da3_application_contract.py \
      tests/unit/test_phase9_3h_da3_canonical_scene.py \
      tests/ui/test_phase9_3h_da3_canonical_integration.py \
      tests/integration/test_reconstruction_recorded_video_sfm_hardening.py \
      tests/unit/test_phase9_6d_digital_twin_interaction.py \
      tests/unit/test_phase9_6e0_estimated_metric_measurement.py
    ;;
  *)
    echo "usage: $0 {quick|integration|backward}" >&2
    exit 2
    ;;
esac

#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-quick}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
export QT_OPENGL="${QT_OPENGL:-software}"
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
export DEPTH3D_TEST_FAKE_VISPY="${DEPTH3D_TEST_FAKE_VISPY:-1}"

case "$MODE" in
  quick)
    python -m pytest -q \
      tests/unit/test_phase9_6e1_processing_jobs.py \
      tests/integration/test_phase9_6e1_processing_job_contract.py \
      tests/ui/test_phase9_6e1_processing_job_ui.py
    ;;
  integration)
    python -m pytest -q \
      tests/integration/test_phase9_6e1_processing_job_contract.py \
      tests/integration/test_phase9_6e1_processing_job_supervisor.py
    ;;
  backward)
    python -m pytest -q \
      tests/ui/test_phase8_8g5_progressive_disclosure_ux.py \
      tests/ui/test_phase8_8h2_application_resource_coordination.py \
      tests/integration/test_phase8_8a5_controller_hardening.py \
      tests/ui/test_phase9_3i_scene_runtime_hardening.py \
      tests/ui/test_phase9_4g_gaussian_resource_lifecycle_ui.py \
      tests/ui/test_phase9_5g_nerf_resource_lifecycle_ui.py \
      tests/ui/test_phase9_6c_navigation_synchronization_ui.py \
      tests/unit/test_phase9_6d_digital_twin_interaction.py \
      tests/ui/test_phase9_6d_unified_interaction_ui.py \
      tests/unit/test_phase9_6e0_estimated_metric_measurement.py \
      tests/ui/test_phase9_6e0_estimated_metric_ruler_ui.py
    ;;
  *)
    echo "usage: $0 {quick|integration|backward}" >&2
    exit 2
    ;;
esac

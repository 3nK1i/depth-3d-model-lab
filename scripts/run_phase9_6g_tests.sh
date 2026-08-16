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
      tests/unit/test_phase9_6g_digital_twin_resources.py \
      tests/ui/test_phase9_6g_resource_coordination_ui.py
    ;;
  integration)
    python -m pytest -q \
      tests/integration/test_phase9_6g_resource_coordination_contract.py
    ;;
  backward)
    python -m pytest -q \
      tests/unit/test_phase8_8h2_contract.py \
      tests/integration/test_phase8_8h2_resource_coordination.py \
      tests/ui/test_phase8_8h2_application_resource_coordination.py \
      tests/unit/test_phase9_6e1_processing_jobs.py \
      tests/integration/test_phase9_6e1_processing_job_contract.py \
      tests/ui/test_phase9_6e1_processing_job_ui.py \
      tests/integration/test_phase9_4g_gaussian_resource_lifecycle.py \
      tests/ui/test_phase9_4g_gaussian_resource_lifecycle_ui.py \
      tests/integration/test_phase9_5g_nerf_resource_lifecycle.py \
      tests/ui/test_phase9_5g_nerf_resource_lifecycle_ui.py \
      tests/unit/test_phase9_6e2_da3_full_scene.py \
      tests/ui/test_phase9_6e2_da3_full_scene_ui.py \
      tests/unit/test_phase9_6e3_nerf_build.py \
      tests/ui/test_phase9_6e3_nerf_build_ui.py \
      tests/ui/test_phase9_6f_digital_twin_persistence_ui.py
    ;;
  full)
    "$0" quick
    "$0" integration
    "$0" backward
    ;;
  *)
    echo "usage: $0 {quick|integration|backward|full}" >&2
    exit 2
    ;;
esac

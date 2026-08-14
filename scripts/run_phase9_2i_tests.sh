#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODE="${1:-quick}"

export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
export QT_OPENGL="${QT_OPENGL:-software}"
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
export DEPTH3D_TEST_FAKE_VISPY="${DEPTH3D_TEST_FAKE_VISPY:-1}"

compile_closure() {
  python -m compileall -q \
    scripts/run_phase9_2i_closure_smoke.py \
    tests/integration/test_phase9_2i_phase_closure.py
  bash -n scripts/run_phase9_2i_tests.sh
}

run_backward_matrix() {
  bash scripts/run_phase5_tests.sh smoke
  bash scripts/run_phase6_step7_tests.sh quick
  bash scripts/run_phase7_step4_tests.sh quick
  bash scripts/run_phase7_5_step10_tests.sh quick
  bash scripts/run_phase8_8h3_tests.sh backward
  python -m pytest -q tests/integration/test_phase8_9g3_backward_compatibility_contract.py
  bash scripts/run_phase9_1i_tests.sh quick
}

case "$MODE" in
  quick)
    compile_closure
    python -m pytest -q \
      tests/unit/test_phase9_2*.py \
      tests/ui/test_phase9_2*.py \
      tests/integration/test_phase9_2i_phase_closure.py
    ;;
  integration)
    compile_closure
    python -m pytest -q \
      tests/integration/test_phase9_2*.py \
      tests/integration/test_phase9_1g_measurement_calibration_pipeline.py \
      tests/integration/test_depth_perception_alignment.py \
      tests/integration/test_reconstruction_semantic_geometry_pipeline.py
    ;;
  backward)
    compile_closure
    run_backward_matrix
    ;;
  full)
    compile_closure
    python -m compileall -q app core tests scripts main.py
    python -m pytest -q tests/unit
    python -m pytest -q tests/ui
    python -m pytest -q tests/integration
    python -m pytest -q tests/stress
    run_backward_matrix
    ;;
  *)
    echo "Usage: $0 {quick|integration|backward|full}" >&2
    exit 2
    ;;
esac

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
    scripts/run_phase9_1i_closure_smoke.py \
    tests/integration/test_phase9_1i_phase_closure.py
  bash -n scripts/run_phase9_1i_tests.sh
}

run_backward_matrix() {
  bash scripts/run_phase5_tests.sh smoke
  bash scripts/run_phase6_step7_tests.sh quick
  bash scripts/run_phase7_step4_tests.sh quick
  bash scripts/run_phase7_5_step10_tests.sh quick
  bash scripts/run_phase8_8h3_tests.sh backward
  python -m pytest -q tests/integration/test_phase8_9g3_backward_compatibility_contract.py
}

case "$MODE" in
  quick)
    compile_closure
    python -m pytest -q \
      tests/unit/test_phase9_1*.py \
      tests/ui/test_phase9_1*.py \
      tests/integration/test_phase9_1i_phase_closure.py
    ;;
  integration)
    compile_closure
    python -m pytest -q \
      tests/integration/test_phase9_1*.py \
      tests/integration/test_known_size_validation_export.py \
      tests/integration/test_reconstruction_camera_geometry_pipeline.py \
      tests/integration/test_reconstruction_scale_reference_pipeline.py \
      tests/integration/test_measurement_inspection_job.py \
      tests/integration/test_phase8_9c_persistence_reproducibility.py
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

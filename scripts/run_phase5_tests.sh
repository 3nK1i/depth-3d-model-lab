#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-full}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
export QT_OPENGL="${QT_OPENGL:-software}"
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"

# Offscreen Qt does not provide a safe QOpenGLWidget framebuffer on every
# driver/Mesa combination. UI tests use a non-OpenGL PointCloudCanvas test
# double; real VisPy rendering remains part of desktop manual validation.
export DEPTH3D_TEST_FAKE_VISPY="${DEPTH3D_TEST_FAKE_VISPY:-1}"

python -m compileall -q app core tests scripts main.py

run_suite() {
  local label="$1"
  shift
  echo
  echo "=================================================="
  echo "$label"
  echo "=================================================="
  pytest -q "$@"
}

case "$MODE" in
  smoke)
    run_suite "Phase 5 smoke tests" \
      tests/unit/test_session_tracker.py \
      tests/unit/test_geometry_projection.py \
      tests/unit/test_geometry_filtering.py \
      tests/unit/test_geometry_sampling.py \
      tests/unit/test_mock_adapter.py \
      tests/unit/test_model_lifecycle.py \
      tests/unit/test_failure_classification.py \
      tests/unit/test_hardware_tuning_matrix.py \
      tests/unit/test_release_metadata.py \
      tests/unit/test_logging_setup.py
    ;;
  unit)
    run_suite "Unit tests" tests/unit
    ;;
  ui)
    run_suite "UI tests" tests/ui
    ;;
  integration)
    run_suite "Integration tests" tests/integration
    ;;
  stress)
    run_suite "Stress tests" tests/stress
    ;;
  all)
    # Keep native Qt/pytest-qt suites isolated in separate Python processes.
    # This prevents deferred paint events and Qt teardown state from leaking
    # across unit, UI, integration, and stress suite boundaries.
    run_suite "Unit tests" tests/unit
    run_suite "UI tests" tests/ui
    run_suite "Integration tests" tests/integration
    run_suite "Stress tests" tests/stress
    ;;
  coverage)
    pytest --cov=app --cov=core --cov-report=term-missing --cov-report=html \
      tests/unit tests/ui tests/integration
    echo "Coverage report: htmlcov/index.html"
    ;;
  full)
    # Release regression gate without stress tests. Each suite runs in its own
    # process for deterministic Qt resource creation and teardown.
    run_suite "Unit tests" tests/unit
    run_suite "UI tests" tests/ui
    run_suite "Integration tests" tests/integration
    ;;
  *)
    echo "Usage: $0 {smoke|unit|ui|integration|stress|full|all|coverage}" >&2
    exit 2
    ;;
esac

#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-full}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
export QT_OPENGL="${QT_OPENGL:-software}"
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"

# Offscreen Qt does not provide a safe QOpenGLWidget framebuffer on every
# driver/Mesa combination.  UI tests use a non-OpenGL PointCloudCanvas test
# double; real VisPy rendering remains part of desktop manual validation.
export DEPTH3D_TEST_FAKE_VISPY="${DEPTH3D_TEST_FAKE_VISPY:-1}"

python -m compileall -q app core tests scripts main.py

case "$MODE" in
  smoke)
    pytest -q \
      tests/unit/test_session_tracker.py \
      tests/unit/test_geometry_projection.py \
      tests/unit/test_geometry_filtering.py \
      tests/unit/test_geometry_sampling.py \
      tests/unit/test_mock_adapter.py \
      tests/unit/test_model_lifecycle.py
    ;;
  unit)
    pytest -q tests/unit
    ;;
  ui)
    pytest -q tests/ui
    ;;
  integration)
    pytest -q tests/integration
    ;;
  coverage)
    pytest --cov=app --cov=core --cov-report=term-missing --cov-report=html tests
    echo "Coverage report: htmlcov/index.html"
    ;;
  full)
    pytest -q tests/unit tests/ui tests/integration
    ;;
  *)
    echo "Usage: $0 {smoke|unit|ui|integration|coverage|full}" >&2
    exit 2
    ;;
esac

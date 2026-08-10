#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
MODE="${1:-quick}"

export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
export QT_OPENGL="${QT_OPENGL:-software}"
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
export DEPTH3D_TEST_FAKE_VISPY="${DEPTH3D_TEST_FAKE_VISPY:-1}"

case "$MODE" in
  quick)
    python -m compileall -q \
      app/reconstruction_controller.py \
      app/offline_reconstruction_controller.py \
      app/main_window.py \
      tests/unit/test_phase8_8a5_controller_hardening.py \
      tests/ui/test_phase8_8a5_controller_readiness.py
    python -m pytest -q \
      tests/unit/test_phase8_8a5_controller_hardening.py \
      tests/ui/test_phase8_8a5_controller_readiness.py
    ;;
  integration)
    python -m pytest -q tests/integration/test_phase8_8a5_controller_hardening.py
    ;;
  backward)
    ./scripts/run_phase8_8a4_tests.sh quick
    ./scripts/run_phase8_8a4_tests.sh integration
    ./scripts/run_phase8_8a4_tests.sh backward
    ;;
  *)
    echo "Usage: $0 {quick|integration|backward}" >&2
    exit 2
    ;;
esac

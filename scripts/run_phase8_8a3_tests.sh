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
      scripts/run_reconstruction_offline_job.py \
      scripts/run_phase8_8a3_offline_controller_pycolmap_smoke.py \
      tests/unit/test_phase8_8a3_reconstruction_controller.py
    python -m pytest -q \
      tests/unit/test_phase8_8a3_reconstruction_controller.py \
      tests/unit/test_phase8_8a2_reconstruction_controller.py \
      tests/unit/test_phase8_8a1_reconstruction_controller.py \
      tests/ui/test_phase8_8a1_reconstruction_controller.py
    ;;
  integration)
    python -m pytest -q \
      tests/integration/test_phase8_8a3_offline_controller_process.py
    ;;
  backward)
    ./scripts/run_phase8_8a2_tests.sh quick
    ./scripts/run_phase8_8a2_tests.sh integration
    ;;
  full)
    for group in tests/unit tests/ui tests/integration tests/stress; do
      echo "==> pytest group: $group"
      python -m pytest -q "$group"
    done
    ;;
  *)
    echo "Usage: $0 {quick|integration|backward|full}" >&2
    exit 2
    ;;
esac

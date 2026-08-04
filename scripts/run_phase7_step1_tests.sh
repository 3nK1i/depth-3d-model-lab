#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-full}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
export QT_OPENGL="${QT_OPENGL:-software}"
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
export DEPTH3D_TEST_FAKE_VISPY="${DEPTH3D_TEST_FAKE_VISPY:-1}"

python -m compileall -q app core tests scripts main.py

case "$MODE" in
  quick)
    pytest -q \
      tests/unit/test_benchmark_events.py \
      tests/unit/test_benchmark_cli.py \
      tests/unit/test_benchmark_runner.py \
      tests/ui/test_benchmark_page.py \
      tests/ui/test_benchmark_controller.py \
      tests/ui/test_main_window_shell.py
    pytest -q tests/stress/test_interactive_benchmark_stress.py
    ;;
  unit)
    pytest -q tests/unit
    ;;
  full)
    ./scripts/run_phase5_tests.sh all
    ;;
  *)
    echo "Usage: $0 {quick|unit|full}" >&2
    exit 2
    ;;
esac

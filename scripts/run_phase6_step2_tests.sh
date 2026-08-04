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
      tests/unit/test_depth_pro_adapter.py \
      tests/unit/test_depth_pro_checkpoint_script.py \
      tests/unit/test_model_registry.py \
      tests/unit/test_model_output_contract.py \
      tests/unit/test_worker_helpers.py
    pytest -q tests/stress/test_depth_pro_adapter_stress.py
    ;;
  unit)
    pytest -q tests/unit
    ;;
  stress)
    pytest -q tests/stress/test_depth_pro_adapter_stress.py
    ;;
  full)
    ./scripts/run_phase5_tests.sh all
    ;;
  *)
    echo "Usage: $0 {quick|unit|stress|full}" >&2
    exit 2
    ;;
esac

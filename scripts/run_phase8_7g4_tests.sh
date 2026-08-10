#!/usr/bin/env bash
# Phase 8.7G4 final regression / UI-readiness gate.
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
      scripts/run_phase8_7g4_full_regression.py \
      scripts/run_reconstruction_g4_readiness_smoke.py \
      scripts/run_reconstruction_g4_real_e2e.py \
      tests/unit/test_phase8_7g4_readiness.py \
      tests/ui/test_phase8_7g4_ui_readiness.py \
      tests/integration/test_phase8_7g4_reconstruction_readiness.py \
      tests/stress/test_phase8_7g4_readiness_stress.py
    python -m pytest -q tests/unit/test_phase8_7g4_readiness.py
    python -m pytest -q tests/ui/test_phase8_7g4_ui_readiness.py
    python -m pytest -q tests/integration/test_phase8_7g4_reconstruction_readiness.py
    python -m pytest -q tests/stress/test_phase8_7g4_readiness_stress.py
    ;;
  full)
    python scripts/run_phase8_7g4_full_regression.py \
      --output-dir /tmp/phase8_7g4_full_regression
    ;;
  *)
    echo "Usage: $0 {quick|full}" >&2
    exit 2
    ;;
esac

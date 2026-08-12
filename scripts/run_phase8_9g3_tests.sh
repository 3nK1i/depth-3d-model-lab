#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODE="${1:-quick}"

export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
export QT_OPENGL="${QT_OPENGL:-software}"
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
export DEPTH3D_TEST_FAKE_VISPY="${DEPTH3D_TEST_FAKE_VISPY:-1}"

python -m compileall -q \
  scripts/run_phase8_9g3_backward_compatibility.py \
  tests/unit/test_phase8_9g3_backward_compatibility.py \
  tests/integration/test_phase8_9g3_backward_compatibility_contract.py

case "$MODE" in
  quick)
    python -m pytest -q tests/unit/test_phase8_9g3_backward_compatibility.py
    ;;
  affected)
    python -m pytest -q tests/unit/test_phase8_9g3_backward_compatibility.py
    python -m pytest -q tests/integration/test_phase8_9g3_backward_compatibility_contract.py
    ;;
  *)
    echo "Usage: $0 {quick|affected}" >&2
    exit 2
    ;;
esac

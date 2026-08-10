#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODE="${1:-quick}"

export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
export QT_OPENGL="${QT_OPENGL:-software}"
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
export DEPTH3D_TEST_FAKE_VISPY="${DEPTH3D_TEST_FAKE_VISPY:-1}"

run_quick() {
  python -m compileall -q \
    app/reconstruction_controller.py \
    app/widgets/reconstruction_page.py \
    app/theme.py \
    tests/unit/test_phase8_8b2_reconstruction_source_selection.py \
    tests/ui/test_phase8_8b2_reconstruction_source_controls.py
  python -m pytest -q \
    tests/unit/test_phase8_8b2_reconstruction_source_selection.py \
    tests/ui/test_phase8_8b2_reconstruction_source_controls.py \
    tests/unit/test_phase8_8b1_reconstruction_page_contract.py
}

run_backward() {
  ./scripts/run_phase8_8b1_tests.sh quick
  ./scripts/run_phase8_8b1_tests.sh backward
}

case "$MODE" in
  quick|ui)
    run_quick
    ;;
  backward)
    run_backward
    ;;
  full)
    run_quick
    run_backward
    ;;
  *)
    echo "Usage: $0 {quick|ui|backward|full}" >&2
    exit 2
    ;;
esac

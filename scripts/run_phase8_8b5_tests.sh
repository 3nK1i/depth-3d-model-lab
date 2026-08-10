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
    app/main_window.py \
    app/widgets/mode_tabs.py \
    tests/unit/test_phase8_8b5_navigation_contract.py \
    tests/ui/test_phase8_8b5_reconstruction_navigation.py
  python -m pytest -q \
    tests/unit/test_phase8_8b5_navigation_contract.py \
    tests/ui/test_phase8_8b5_reconstruction_navigation.py \
    tests/ui/test_phase8_7g4_ui_readiness.py \
    tests/ui/test_phase7_release_modes.py \
    tests/ui/test_phase7_mode_switching.py
}

run_history() {
  mapfile -t files < <(grep -RIl --include='*.py' '_ENABLED_MODES' tests/unit | sort)
  python -m pytest -q "${files[@]}"
  python -m pytest -q \
    tests/ui/test_main_window_shell.py \
    tests/ui/test_phase8_8a1_reconstruction_controller.py \
    tests/ui/test_phase8_8a5_controller_readiness.py
}

run_backward() {
  ./scripts/run_phase8_8b4_tests.sh quick
  ./scripts/run_phase8_8b4_tests.sh backward
}

case "$MODE" in
  quick|ui)
    run_quick
    ;;
  history)
    run_history
    ;;
  backward)
    run_backward
    ;;
  full)
    run_quick
    run_history
    run_backward
    ;;
  *)
    echo "Usage: $0 {quick|ui|history|backward|full}" >&2
    exit 2
    ;;
esac

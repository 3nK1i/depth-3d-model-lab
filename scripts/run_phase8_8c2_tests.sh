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
    app/reconstruction_visualization.py \
    app/widgets/reconstruction_orientation_gizmo.py \
    app/widgets/reconstruction_page.py \
    app/widgets/viewport/vispy_canvas.py \
    app/icons.py \
    tests/conftest.py \
    tests/unit/test_phase8_8c1_reconstruction_visualization.py \
    tests/unit/test_phase8_8c1_visualizer_contract.py \
    tests/ui/test_phase8_8c1_live_visualizer_foundation.py \
    tests/unit/test_phase8_8c2_live_trajectory_visualization.py \
    tests/unit/test_phase8_8c2_visualizer_contract.py \
    tests/ui/test_phase8_8c2_live_trajectory_visualization.py
  python -m pytest -q \
    tests/unit/test_phase8_8c1_reconstruction_visualization.py \
    tests/unit/test_phase8_8c1_visualizer_contract.py \
    tests/ui/test_phase8_8c1_live_visualizer_foundation.py \
    tests/unit/test_phase8_8c2_live_trajectory_visualization.py \
    tests/unit/test_phase8_8c2_visualizer_contract.py \
    tests/ui/test_phase8_8c2_live_trajectory_visualization.py
}

run_backward() {
  ./scripts/run_phase8_8b5_tests.sh quick
  ./scripts/run_phase8_8b4_tests.sh quick
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

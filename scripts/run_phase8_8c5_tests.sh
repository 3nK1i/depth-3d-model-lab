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
    app/reconstruction_diagnostics.py \
    app/reconstruction_visualization.py \
    app/widgets/reconstruction_diagnostics_hud.py \
    app/widgets/reconstruction_page.py \
    app/widgets/reconstruction_orientation_gizmo.py \
    app/widgets/viewport/vispy_canvas.py \
    app/icons.py \
    tests/unit/test_phase8_8c5_tracking_diagnostics.py \
    tests/unit/test_phase8_8c5_contract.py \
    tests/ui/test_phase8_8c5_tracking_diagnostics_hud.py
  bash -n scripts/run_phase8_8c5_tests.sh
  python -m pytest -q \
    tests/unit/test_phase8_8c1_reconstruction_visualization.py \
    tests/unit/test_phase8_8c1_visualizer_contract.py \
    tests/ui/test_phase8_8c1_live_visualizer_foundation.py \
    tests/unit/test_phase8_8c2_live_trajectory_visualization.py \
    tests/unit/test_phase8_8c2_visualizer_contract.py \
    tests/ui/test_phase8_8c2_live_trajectory_visualization.py \
    tests/unit/test_phase8_8c3_sparse_map_visualization.py \
    tests/unit/test_phase8_8c3_sparse_map_protocol.py \
    tests/unit/test_phase8_8c3_contract.py \
    tests/ui/test_phase8_8c3_live_sparse_map_visualization.py \
    tests/unit/test_phase8_8c4_keyframe_map_structure.py \
    tests/unit/test_phase8_8c4_keyframe_protocol.py \
    tests/unit/test_phase8_8c4_keyframe_visualization.py \
    tests/unit/test_phase8_8c4_contract.py \
    tests/ui/test_phase8_8c4_live_keyframe_map_structure.py \
    tests/unit/test_phase8_8c5_tracking_diagnostics.py \
    tests/unit/test_phase8_8c5_contract.py \
    tests/ui/test_phase8_8c5_tracking_diagnostics_hud.py
}

run_protocol() {
  # C5 does not alter protocol/native code; keep the qualified C4 boundary green.
  ./scripts/run_phase8_8c4_tests.sh protocol
  python -m pytest -q tests/unit/test_phase8_8c5_contract.py
}

run_backward() {
  ./scripts/run_phase8_8c4_tests.sh quick
  ./scripts/run_phase8_8c3_tests.sh quick
  ./scripts/run_phase8_8c2_tests.sh quick
  ./scripts/run_phase8_8b5_tests.sh quick
  ./scripts/run_phase8_8b4_tests.sh quick
}

case "$MODE" in
  quick|ui)
    run_quick
    ;;
  protocol)
    run_protocol
    ;;
  backward)
    run_backward
    ;;
  full)
    run_quick
    run_protocol
    run_backward
    ;;
  *)
    echo "Usage: $0 {quick|ui|protocol|backward|full}" >&2
    exit 2
    ;;
esac

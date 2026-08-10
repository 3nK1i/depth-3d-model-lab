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
    app/widgets/reconstruction_page.py \
    app/widgets/viewport/vispy_canvas.py \
    app/widgets/reconstruction_orientation_gizmo.py \
    app/icons.py \
    core/reconstruction/live_contracts.py \
    core/reconstruction/live_protocol.py \
    tests/unit/test_phase8_8c3_sparse_map_visualization.py \
    tests/unit/test_phase8_8c3_sparse_map_protocol.py \
    tests/unit/test_phase8_8c3_contract.py \
    tests/ui/test_phase8_8c3_live_sparse_map_visualization.py
  bash -n scripts/build_stella_live_worker.sh
  bash -n scripts/run_phase8_8c3_tests.sh
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
    tests/ui/test_phase8_8c3_live_sparse_map_visualization.py
}

run_protocol() {
  python -m pytest -q \
    tests/unit/test_reconstruction_live_contracts.py \
    tests/unit/test_reconstruction_live_stella_protocol.py \
    tests/unit/test_reconstruction_live_stella_worker_source.py \
    tests/unit/test_phase8_8c3_sparse_map_protocol.py \
    tests/unit/test_phase8_8c3_contract.py
}

run_backward() {
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

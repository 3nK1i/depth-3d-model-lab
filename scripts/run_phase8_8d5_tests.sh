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
    app/offline_reconstruction_semantics.py \
    app/offline_reconstruction_spatial.py \
    app/offline_reconstruction_quality.py \
    app/offline_reconstruction_inspection.py \
    app/offline_reconstruction_visualization.py \
    app/widgets/reconstruction_page.py \
    app/widgets/viewport/vispy_canvas.py \
    scripts/run_phase8_8d5_semantic_object_preview.py \
    scripts/run_phase8_8d5_semantic_presentation_smoke.py \
    tests/unit/test_phase8_8d5_offline_semantic_presentation.py \
    tests/unit/test_phase8_8d5_contract.py \
    tests/ui/test_phase8_8d5_semantic_object_presentation.py
  bash -n scripts/run_phase8_8d5_tests.sh
  python -m pytest -q \
    tests/unit/test_phase8_8d1_offline_visualization.py \
    tests/unit/test_phase8_8d1_worker_visualization_artifact.py \
    tests/unit/test_phase8_8d1_contract.py \
    tests/ui/test_phase8_8d1_offline_reconstruction_visualization.py \
    tests/unit/test_phase8_8d2_offline_camera_inspection.py \
    tests/unit/test_phase8_8d2_contract.py \
    tests/ui/test_phase8_8d2_offline_camera_inspection.py \
    tests/unit/test_phase8_8d3_offline_sfm_health.py \
    tests/unit/test_phase8_8d3_contract.py \
    tests/ui/test_phase8_8d3_offline_sfm_health.py \
    tests/unit/test_phase8_8d4_offline_spatial_coverage.py \
    tests/unit/test_phase8_8d4_contract.py \
    tests/ui/test_phase8_8d4_offline_spatial_coverage.py \
    tests/unit/test_phase8_8d5_offline_semantic_presentation.py \
    tests/unit/test_phase8_8d5_contract.py \
    tests/ui/test_phase8_8d5_semantic_object_presentation.py
}

run_protocol() {
  # D5 consumes existing Phase 8.7 semantic results. It does not change
  # PyCOLMAP, Stella, reconstruction protocols, semantic algorithms, or authority.
  ./scripts/run_phase8_8d4_tests.sh protocol
  python -m pytest -q tests/unit/test_phase8_8d5_contract.py
}

run_backward() {
  ./scripts/run_phase8_8d4_tests.sh quick
  ./scripts/run_phase8_8d4_tests.sh backward
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

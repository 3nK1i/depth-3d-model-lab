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
    app/reconstruction_inspection_qc_presentation.py \
    app/reconstruction_measurement_validity_presentation.py \
    app/reconstruction_measurement_presentation.py \
    app/reconstruction_inspection_selection.py \
    app/offline_reconstruction_semantics.py \
    app/widgets/reconstruction_page.py \
    scripts/run_phase8_8e4_inspection_qc_preview.py \
    scripts/run_phase8_8e4_inspection_qc_smoke.py \
    tests/unit/test_phase8_8e4_reconstruction_inspection_qc_presentation.py \
    tests/unit/test_phase8_8e4_contract.py \
    tests/ui/test_phase8_8e4_inspection_qc_presentation.py
  bash -n scripts/run_phase8_8e4_tests.sh
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
    tests/ui/test_phase8_8d5_semantic_object_presentation.py \
    tests/unit/test_phase8_8e1_reconstruction_inspection_selection.py \
    tests/unit/test_phase8_8e1_contract.py \
    tests/ui/test_phase8_8e1_scene_object_selection.py \
    tests/unit/test_phase8_8e2_reconstruction_measurement_presentation.py \
    tests/unit/test_phase8_8e2_contract.py \
    tests/ui/test_phase8_8e2_reconstruction_measurement_ui.py \
    tests/unit/test_phase8_8e3_reconstruction_measurement_validity_presentation.py \
    tests/unit/test_phase8_8e3_contract.py \
    tests/ui/test_phase8_8e3_measurement_validity_confidence.py \
    tests/unit/test_phase8_8e4_reconstruction_inspection_qc_presentation.py \
    tests/unit/test_phase8_8e4_contract.py \
    tests/ui/test_phase8_8e4_inspection_qc_presentation.py
}

run_protocol() {
  # E4 consumes authoritative Phase 8.7C report decisions. It does not rerun
  # registration/QC, derive a decision from E2/E3, or implement E5 reference UI.
  ./scripts/run_phase8_8e3_tests.sh protocol
  python -m pytest -q tests/unit/test_phase8_8e4_contract.py
}

run_backward() {
  ./scripts/run_phase8_8e3_tests.sh backward
  # Focused compatibility gate for the exact 8.7C report contract that E4 reads.
  python -m pytest -q \
    tests/unit/test_reconstruction_inspection_qc.py \
    tests/unit/test_phase8_7c_backward_compatibility.py \
    tests/integration/test_reconstruction_inspection_qc_pipeline.py
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
